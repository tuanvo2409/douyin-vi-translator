"""Production-owned, strict Gemini boundary for CP2 text localization."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from llm_transport import ProviderError, StructuredGeminiTransport, StructuredResponseError, gemini_json_transport
from vn_localizer import validate_translation_document
from vn_quality import QA_RESPONSE_SCHEMA, validate_qa_document


MAX_STRUCTURED_DIAGNOSTIC_BYTES = 4 * 1024
_SCENE_FIELDS = {
    "summary": "STRING",
    "tone": "STRING",
    "register": "STRING",
    "speakers": "OBJECT",
    "relationships": "OBJECT",
    "pronoun_policy": "OBJECT",
    "roleplay_terms": "OBJECT",
    "running_jokes": "ARRAY",
    "glossary": "OBJECT",
}
_INSTRUCTIONS = {
    "scene": "Infer only source-supported scene context. Return exactly keys summary, tone, register, speakers, relationships, pronoun_policy, roleplay_terms, running_jokes, glossary; use empty objects/lists if absent. Do not invent facts.",
    "semantic": "Translate each supplied resolved Chinese dialogue faithfully to Vietnamese. Return exactly the requested JSON; do not translate unresolved or visual-only rows.",
    "native": "Edit supplied semantic Vietnamese into natural spoken vi-VN, following editorial_contract and scene_memory. Preserve facts, uncertainty, relationship evidence, and roleplay. Return exactly requested JSON.",
    "qa": "Assess accepted Vietnamese only using the supplied criteria. Separate SEMANTIC_DRIFT from STIFF_VI. Do not invent issues. Return exactly requested JSON.",
    "repair": "Repair only listed rows, preserve utteranceId and source facts, and address the QA flags. Return exactly requested JSON.",
}


def _translation_schema(value_key: str) -> dict[str, object]:
    return {
        "type": "OBJECT",
        "properties": {
            "translations": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "additionalProperties": False,
                    "properties": {
                        "utteranceId": {"type": "STRING"},
                        value_key: {"type": "STRING"},
                    },
                    "required": ["utteranceId", value_key],
                },
            },
        },
        "required": ["translations"],
        "additionalProperties": False,
    }


SCENE_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        **{key: {"type": value} for key, value in _SCENE_FIELDS.items() if value != "ARRAY"},
        "running_jokes": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": list(_SCENE_FIELDS),
    "additionalProperties": False,
}
OPERATION_SCHEMAS = {
    "scene": SCENE_RESPONSE_SCHEMA,
    "semantic": _translation_schema("semanticVi"),
    "native": _translation_schema("nativeVi"),
    "qa": deepcopy(QA_RESPONSE_SCHEMA),
    "repair": _translation_schema("nativeVi"),
}


class CP2StructuredResponseError(StructuredResponseError):
    """A safe structural failure that never retains a provider response body."""

    def __init__(self, operation: str, classification: str) -> None:
        super().__init__(classification)
        self.operation = operation
        self.classification = classification


def _type_name(value: object) -> str:
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        return "string"
    if value is None:
        return "null"
    return type(value).__name__


def _parity_node(expected: Mapping[str, object], actual: object, path: str, violations: list[str], required_paths: list[str]) -> None:
    if not isinstance(actual, Mapping):
        violations.append(path + ":NODE_MISSING")
        return
    if actual.get("type") != expected.get("type"):
        violations.append(path + ":TYPE")
    if bool(actual.get("nullable", False)) != bool(expected.get("nullable", False)):
        violations.append(path + ":NULLABLE")
    if "additionalProperties" in expected and actual.get("additionalProperties") != expected["additionalProperties"]:
        violations.append(path + ":ADDITIONAL_PROPERTIES")
    if "enum" in expected and actual.get("enum") != expected.get("enum"):
        violations.append(path + ":ENUM")
    expected_required = expected.get("required", [])
    actual_required = actual.get("required", [])
    if not isinstance(expected_required, list) or not isinstance(actual_required, list):
        violations.append(path + ":REQUIRED")
    else:
        for key in expected_required:
            required_paths.append(path + "." + str(key))
            if key not in actual_required:
                violations.append(path + "." + str(key) + ":REQUIRED")
    expected_properties = expected.get("properties")
    if isinstance(expected_properties, Mapping):
        actual_properties = actual.get("properties")
        if not isinstance(actual_properties, Mapping):
            violations.append(path + ":PROPERTIES")
        else:
            for key, child in expected_properties.items():
                _parity_node(child, actual_properties.get(key), path + "." + str(key), violations, required_paths)
    if isinstance(expected.get("items"), Mapping):
        _parity_node(expected["items"], actual.get("items"), path + "[]", violations, required_paths)


def validate_schema_parity(operation: str, provider_schema: object) -> dict[str, object]:
    """Ensure a provider schema cannot weaken its matching CP2 validator."""
    expected = OPERATION_SCHEMAS.get(operation)
    if expected is None:
        raise ValueError("unsupported CP2 operation")
    violations: list[str] = []
    required_paths: list[str] = []
    _parity_node(expected, provider_schema, operation, violations, required_paths)
    return {
        "operation": operation,
        "status": "PASS" if not violations else "FAIL",
        "canonicalRequiredPaths": required_paths,
        "canonicalRequiredFieldCount": len(required_paths),
        "violations": violations,
    }


def validate_scene_document(document: object) -> dict[str, object]:
    if not isinstance(document, Mapping) or set(document) != set(_SCENE_FIELDS):
        raise ValueError("scene response schema is invalid")
    for key, expected_type in _SCENE_FIELDS.items():
        value = document.get(key)
        if expected_type == "STRING" and not isinstance(value, str):
            raise ValueError("scene response field is invalid")
        if expected_type == "OBJECT" and not isinstance(value, Mapping):
            raise ValueError("scene response field is invalid")
        if expected_type == "ARRAY" and (not isinstance(value, list) or not all(isinstance(item, str) for item in value)):
            raise ValueError("scene response field is invalid")
    return dict(document)


def _normalize_document(raw: object) -> tuple[object, bool]:
    """Accept only lossless JSON serialization wrappers; never invent content."""
    normalized = False
    value = raw
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("```") and text.endswith("```"):
            text = text.split("\n", 1)[1].rsplit("\n", 1)[0].strip() if "\n" in text else ""
            normalized = True
        try:
            value = json.loads(text)
        except (TypeError, ValueError) as error:
            raise ValueError("provider response is not JSON") from error
        normalized = True
    if isinstance(value, Mapping) and len(value) == 1:
        for outer_key in ("data", "result", "response", "output"):
            if outer_key in value and isinstance(value[outer_key], Mapping):
                value = value[outer_key]
                normalized = True
                break
    return value, normalized


class CP2GeminiAdapter:
    """One callable CP2 provider boundary; transport retries remain transport-owned."""

    def __init__(
        self,
        *,
        transport: StructuredGeminiTransport,
        model: str,
        diagnostic_path: Path | None = None,
    ) -> None:
        self.transport = transport
        self.model = model
        self.diagnostic_path = diagnostic_path
        self.logical_call_count = 0
        self.last_diagnostic: dict[str, object] | None = None

    def _persist_diagnostic(self, operation: str, classification: str, document: object, *, parity_status: str = "PASS") -> None:
        safe = {
            "operation": operation,
            "logicalCall": self.logical_call_count,
            "failureClass": classification,
            "validatorPath": "root",
            "expected": "canonical " + operation + " response",
            "actualType": _type_name(document),
            "schemaParity": parity_status,
            "transportStatus": None,
        }
        encoded = json.dumps(safe, separators=(",", ":")).encode("utf-8")
        if len(encoded) > MAX_STRUCTURED_DIAGNOSTIC_BYTES:
            raise AssertionError("structured diagnostic unexpectedly exceeded its bound")
        self.last_diagnostic = safe
        if self.diagnostic_path is not None:
            self.diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.diagnostic_path.with_name("." + self.diagnostic_path.name + ".tmp")
            temporary.write_bytes(encoded)
            os.replace(temporary, self.diagnostic_path)

    def _payload(self, operation: str, document: Mapping[str, object]) -> dict[str, object]:
        schema = OPERATION_SCHEMAS[operation]
        parity = validate_schema_parity(operation, schema)
        if parity["status"] != "PASS":
            self._persist_diagnostic(operation, "PROVIDER_SCHEMA_PARITY_INVALID", document, parity_status="FAIL")
            raise CP2StructuredResponseError(operation, "PROVIDER_SCHEMA_PARITY_INVALID")
        return {
            "contents": [{"role": "user", "parts": [{"text": _INSTRUCTIONS[operation] + "\nINPUT_JSON:\n" + json.dumps(document, ensure_ascii=False, separators=(",", ":"))}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.2,
                "responseSchema": deepcopy(schema),
            },
        }

    def _validate(self, operation: str, document: object, records: Sequence[Mapping[str, object]]) -> dict[str, object]:
        if operation == "scene":
            return validate_scene_document(document)
        if operation in {"semantic", "native", "repair"}:
            key = "semanticVi" if operation == "semantic" else "nativeVi"
            by_id = validate_translation_document(document, records, key)
            return {"translations": [{"utteranceId": str(record["utteranceId"]), key: by_id[str(record["utteranceId"])]} for record in records]}
        if operation == "qa":
            validate_qa_document(document, records)
            return dict(document) if isinstance(document, Mapping) else {}
        raise ValueError("unsupported CP2 operation")

    def __call__(self, payload: Mapping[str, object]) -> dict[str, object]:
        operation = str(payload.get("mode", ""))
        if operation not in OPERATION_SCHEMAS:
            raise ValueError("unsupported CP2 operation")
        records = payload.get("utterances")
        identifiers = [record.get("utteranceId") for record in records] if isinstance(records, list) else []
        if (
            not isinstance(records, list)
            or not all(isinstance(record, Mapping) for record in records)
            or not all(isinstance(identifier, str) and identifier == identifier.strip() and identifier for identifier in identifiers)
            or len(identifiers) != len(set(identifiers))
        ):
            raise ValueError("CP2 utterance identity is invalid")
        self.logical_call_count += 1
        try:
            raw = self.transport.call_raw(self._payload(operation, payload))
        except ProviderError:
            raise
        except StructuredResponseError as error:
            self._persist_diagnostic(operation, "PROVIDER_RESPONSE_ENVELOPE_INVALID", None)
            raise CP2StructuredResponseError(operation, "PROVIDER_RESPONSE_ENVELOPE_INVALID") from error
        try:
            document, _normalized = _normalize_document(raw)
            return self._validate(operation, document, records)
        except (TypeError, ValueError, KeyError) as error:
            self._persist_diagnostic(operation, "CP2_RESPONSE_SCHEMA_INVALID", raw)
            raise CP2StructuredResponseError(operation, "CP2_RESPONSE_SCHEMA_INVALID") from error


def live_cp2_gemini_adapter(*, model: str, diagnostic_path: Path | None = None) -> CP2GeminiAdapter:
    """Create the production CP2 boundary with the existing secret-safe transport."""
    return CP2GeminiAdapter(
        transport=gemini_json_transport(model=model),
        model=model,
        diagnostic_path=diagnostic_path,
    )
