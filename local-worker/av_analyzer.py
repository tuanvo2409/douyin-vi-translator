"""Bounded multimodal analysis, deliberately separate from the legacy worker."""
from __future__ import annotations

import base64
from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Callable, Mapping, Sequence

from llm_transport import ProviderError, StructuredGeminiTransport, StructuredResponseError, gemini_json_transport

MAX_WINDOW_MS = 30_000
WINDOW_OVERLAP_MS = 5_000
MAX_DIAGNOSTIC_BYTES = 64 * 1024
_AV_KEYS = {"windowStartMs", "windowEndMs", "scene", "utterances", "visualOnly"}
_SCENE_KEYS = {"summary", "tone", "register"}
_UTTERANCE_KEYS = {
    "startMs", "endMs", "spokenZh", "subtitleZh", "speakerRef", "intent", "emotion", "confidence",
}
_VISUAL_KEYS = {"startMs", "endMs", "textZh", "confidence"}
_CONFIDENCES = {"HIGH", "MEDIUM", "LOW"}
_RECOVERY_WRAPPER_KEYS = {"data", "result", "response", "output"}
_BOUNDARY_WINDOW_TOLERANCE_MS = WINDOW_OVERLAP_MS
_RECOVERY_KEYS = (
    "windowStartMs",
    "windowEndMs",
    "attemptCount",
    "logicalAttempt",
    "transportAttempts",
    "totalTransportAttempts",
    "httpStatus",
    "finalTransportStatus",
    "transientRetryUsed",
    "schemaValidationStatus",
    "repairGenerationUsed",
    "failureClass",
    "normalizationApplied",
    "coordinateNormalizationApplied",
    "boundaryNormalizationApplied",
    "repairRetryUsed",
    "finalStatus",
)

AV_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "windowStartMs": {"type": "INTEGER"},
        "windowEndMs": {"type": "INTEGER"},
        "scene": {
            "type": "OBJECT",
            "properties": {
                "summary": {"type": "STRING"},
                "tone": {"type": "STRING"},
                "register": {"type": "STRING"},
            },
            "required": ["summary", "tone", "register"],
        },
        "utterances": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "startMs": {"type": "INTEGER"},
                    "endMs": {"type": "INTEGER"},
                    "spokenZh": {"type": "STRING", "nullable": True},
                    "subtitleZh": {"type": "STRING", "nullable": True},
                    "speakerRef": {"type": "STRING", "nullable": True},
                    "intent": {"type": "STRING", "nullable": True},
                    "emotion": {"type": "STRING", "nullable": True},
                    "confidence": {"type": "STRING", "enum": ["HIGH", "MEDIUM", "LOW"]},
                },
                "required": [
                    "startMs", "endMs", "spokenZh", "subtitleZh", "speakerRef",
                    "intent", "emotion", "confidence",
                ],
            },
        },
        "visualOnly": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "startMs": {"type": "INTEGER"},
                    "endMs": {"type": "INTEGER"},
                    "textZh": {"type": "STRING"},
                    "confidence": {"type": "STRING", "enum": ["HIGH", "MEDIUM", "LOW"]},
                },
                "required": ["startMs", "endMs", "textZh", "confidence"],
            },
        },
    },
    "required": ["windowStartMs", "windowEndMs", "scene", "utterances", "visualOnly"],
}


_CANONICAL_AV_NODES = {
    "": {"type": "OBJECT"},
    "windowStartMs": {"type": "INTEGER"},
    "windowEndMs": {"type": "INTEGER"},
    "scene": {"type": "OBJECT"},
    "scene.summary": {"type": "STRING"},
    "scene.tone": {"type": "STRING"},
    "scene.register": {"type": "STRING"},
    "utterances": {"type": "ARRAY"},
    "utterances[]": {"type": "OBJECT"},
    "utterances[].startMs": {"type": "INTEGER"},
    "utterances[].endMs": {"type": "INTEGER"},
    "utterances[].spokenZh": {"type": "STRING", "nullable": True},
    "utterances[].subtitleZh": {"type": "STRING", "nullable": True},
    "utterances[].speakerRef": {"type": "STRING", "nullable": True},
    "utterances[].intent": {"type": "STRING", "nullable": True},
    "utterances[].emotion": {"type": "STRING", "nullable": True},
    "utterances[].confidence": {"type": "STRING", "enum": sorted(_CONFIDENCES)},
    "visualOnly": {"type": "ARRAY"},
    "visualOnly[]": {"type": "OBJECT"},
    "visualOnly[].startMs": {"type": "INTEGER"},
    "visualOnly[].endMs": {"type": "INTEGER"},
    "visualOnly[].textZh": {"type": "STRING"},
    "visualOnly[].confidence": {"type": "STRING", "enum": sorted(_CONFIDENCES)},
}
_CANONICAL_AV_REQUIRED_PATHS = (
    "windowStartMs",
    "windowEndMs",
    "scene",
    "utterances",
    "visualOnly",
    "scene.summary",
    "scene.tone",
    "scene.register",
    "utterances[].startMs",
    "utterances[].endMs",
    "utterances[].spokenZh",
    "utterances[].subtitleZh",
    "utterances[].speakerRef",
    "utterances[].intent",
    "utterances[].emotion",
    "utterances[].confidence",
    "visualOnly[].startMs",
    "visualOnly[].endMs",
    "visualOnly[].textZh",
    "visualOnly[].confidence",
)


def _schema_node_at(schema: object, path: str) -> Mapping[str, object] | None:
    node = schema
    if path:
        for segment in path.split("."):
            if not isinstance(node, Mapping):
                return None
            is_item = segment.endswith("[]")
            name = segment[:-2] if is_item else segment
            properties = node.get("properties")
            if not isinstance(properties, Mapping) or name not in properties:
                return None
            node = properties[name]
            if is_item:
                if not isinstance(node, Mapping):
                    return None
                node = node.get("items")
    return node if isinstance(node, Mapping) else None


def _schema_parent_path(path: str) -> tuple[str, str]:
    parent, separator, name = path.rpartition(".")
    return (parent if separator else "", name if separator else path)


def _schema_required_count(node: object) -> int:
    if not isinstance(node, Mapping):
        return 0
    required = node.get("required")
    count = sum(isinstance(value, str) for value in required) if isinstance(required, list) else 0
    properties = node.get("properties")
    if isinstance(properties, Mapping):
        count += sum(_schema_required_count(value) for value in properties.values())
    items = node.get("items")
    if items is not None:
        count += _schema_required_count(items)
    return count


def _parity_violation(code: str, path: str, detail: str) -> dict[str, str]:
    return {"code": code, "path": path or "$", "detail": detail}


def validate_av_schema_parity(provider_schema: Mapping[str, object] | None = None) -> dict[str, object]:
    """Compare a provider response schema with the canonical AV contract."""
    provider = AV_RESPONSE_SCHEMA if provider_schema is None else provider_schema
    violations: list[dict[str, str]] = []
    if not isinstance(provider, Mapping):
        violations.append(_parity_violation("CONTAINER_SHAPE_MISMATCH", "", "provider schema must be an object"))
        provider = {}

    for path, expected in _CANONICAL_AV_NODES.items():
        actual = _schema_node_at(provider, path)
        if actual is None:
            violations.append(_parity_violation("CONTAINER_SHAPE_MISMATCH", path, "provider schema node is missing"))
            continue
        actual_type = actual.get("type")
        if actual_type != expected["type"]:
            violations.append(_parity_violation("TYPE_MISMATCH", path, f"expected {expected['type']}, got {actual_type}"))
            continue
        if expected["type"] == "OBJECT" and not isinstance(actual.get("properties"), Mapping):
            violations.append(_parity_violation("CONTAINER_SHAPE_MISMATCH", path, "object properties are missing"))
        if expected["type"] == "ARRAY" and not isinstance(actual.get("items"), Mapping):
            violations.append(_parity_violation("CONTAINER_SHAPE_MISMATCH", path, "array items are missing"))

        expected_nullable = bool(expected.get("nullable", False))
        actual_nullable = actual.get("nullable", False)
        if actual_nullable is not expected_nullable:
            violations.append(_parity_violation(
                "UNEXPECTED_NULLABILITY",
                path,
                f"expected nullable={expected_nullable}, got nullable={actual_nullable}",
            ))

        expected_enum = expected.get("enum")
        actual_enum = actual.get("enum")
        if expected_enum is not None and (
            not isinstance(actual_enum, list) or any(value not in actual_enum for value in expected_enum)
        ):
            violations.append(_parity_violation("ENUM_MISMATCH", path, "provider enum does not cover canonical enum"))
        elif expected_enum is None and actual_enum is not None:
            violations.append(_parity_violation("ENUM_MISMATCH", path, "provider narrows an unconstrained canonical string"))

    for path in _CANONICAL_AV_REQUIRED_PATHS:
        parent_path, name = _schema_parent_path(path)
        parent = _schema_node_at(provider, parent_path)
        properties = parent.get("properties") if isinstance(parent, Mapping) else None
        required = parent.get("required") if isinstance(parent, Mapping) else None
        if not isinstance(properties, Mapping) or name not in properties:
            code = "CANONICAL_REQUIRED_MISSING_FROM_PROVIDER_SCHEMA" if not parent_path else "NESTED_REQUIRED_MISSING"
            violations.append(_parity_violation(code, path, "canonical required property is missing"))
        elif not isinstance(required, list) or name not in required:
            code = "CANONICAL_REQUIRED_MISSING_FROM_PROVIDER_SCHEMA" if not parent_path else "NESTED_REQUIRED_MISSING"
            violations.append(_parity_violation(code, path, "canonical required property is not provider-required"))

    missing_paths = sorted({item["path"] for item in violations if item["code"] in {
        "CANONICAL_REQUIRED_MISSING_FROM_PROVIDER_SCHEMA", "NESTED_REQUIRED_MISSING",
    }})
    violation_codes = list(dict.fromkeys(item["code"] for item in violations))
    return {
        "status": "PASS" if not violations else "FAIL",
        "canonicalRequiredFieldCount": len(_CANONICAL_AV_REQUIRED_PATHS),
        "providerRequiredFieldCount": _schema_required_count(provider),
        "canonicalRequiredPaths": list(_CANONICAL_AV_REQUIRED_PATHS),
        "missingPaths": missing_paths,
        "violationCodes": violation_codes,
        "violations": violations,
    }


audit_av_schema_parity = validate_av_schema_parity


def persist_schema_parity_report(path: Path, report: Mapping[str, object]) -> None:
    """Persist bounded schema audit evidence without provider payloads."""
    safe = {
        key: report.get(key)
        for key in (
            "status",
            "canonicalRequiredFieldCount",
            "providerRequiredFieldCount",
            "missingPaths",
            "violationCodes",
            "violations",
        )
    }
    payload = json.dumps(safe, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(payload) > MAX_DIAGNOSTIC_BYTES:
        raise ValueError("AV schema parity diagnostics exceed bounded size")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _normalize_window_coordinates(value: object, start_ms: int, end_ms: int) -> tuple[object, bool]:
    """Convert an explicitly relative child response to source coordinates once."""
    if start_ms <= 0 or not isinstance(value, dict):
        return value, False
    if value.get("windowStartMs") != 0 or value.get("windowEndMs") != end_ms - start_ms:
        return value, False
    normalized = deepcopy(value)
    normalized["windowStartMs"] += start_ms
    normalized["windowEndMs"] += start_ms
    for collection_name in ("utterances", "visualOnly"):
        collection = normalized.get(collection_name)
        if not isinstance(collection, list):
            continue
        for item in collection:
            if not isinstance(item, dict):
                continue
            for key in ("startMs", "endMs"):
                if _is_int(item.get(key)):
                    item[key] += start_ms
    return normalized, True


def _normalize_boundary_utterance_window(value: object, start_ms: int, end_ms: int) -> tuple[object, bool]:
    """Preserve edge-crossing utterances without changing their source timing."""
    if not isinstance(value, dict) or set(value) != _AV_KEYS:
        return value, False
    if value.get("windowStartMs") != start_ms or value.get("windowEndMs") != end_ms:
        return value, False
    utterances = value.get("utterances")
    if not isinstance(utterances, list):
        return value, False

    starts: list[int] = []
    ends: list[int] = []
    for item in utterances:
        if not isinstance(item, dict) or not _is_int(item.get("startMs")) or not _is_int(item.get("endMs")):
            return value, False
        item_start, item_end = int(item["startMs"]), int(item["endMs"])
        if item_end <= start_ms or item_start >= end_ms:
            return value, False
        if item_start < start_ms and start_ms - item_start > _BOUNDARY_WINDOW_TOLERANCE_MS:
            return value, False
        if item_end > end_ms and item_end - end_ms > _BOUNDARY_WINDOW_TOLERANCE_MS:
            return value, False
        starts.append(item_start)
        ends.append(item_end)
    if not starts:
        return value, False
    lower, upper = min(starts), max(ends)
    if lower >= start_ms and upper <= end_ms:
        return value, False
    # ponytail: edge-only expansion is bounded by the configured overlap; a
    # larger or non-overlapping item remains a hard schema failure.
    normalized = deepcopy(value)
    normalized["windowStartMs"] = min(start_ms, lower)
    normalized["windowEndMs"] = max(end_ms, upper)
    return normalized, True


def _valid_interval(value: Mapping[str, object], start_key: str, end_key: str) -> None:
    start, end = value[start_key], value[end_key]
    if not _is_int(start) or not _is_int(end) or start < 0 or end <= start:
        raise ValueError("AV analysis interval is invalid")


def _optional_text(value: object) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError("AV analysis text field is invalid")


def validate_av_analysis(document: object) -> dict[str, object]:
    if not isinstance(document, dict) or set(document) != _AV_KEYS:
        raise ValueError("AV analysis schema is invalid")
    _valid_interval(document, "windowStartMs", "windowEndMs")
    window_start, window_end = document["windowStartMs"], document["windowEndMs"]
    scene = document["scene"]
    if not isinstance(scene, dict) or set(scene) != _SCENE_KEYS or any(
        not isinstance(scene[key], str) or not scene[key].strip() for key in _SCENE_KEYS
    ):
        raise ValueError("AV analysis scene is invalid")
    if not isinstance(document["utterances"], list) or not isinstance(document["visualOnly"], list):
        raise ValueError("AV analysis lists are invalid")
    for value in document["utterances"]:
        if not isinstance(value, dict) or set(value) != _UTTERANCE_KEYS:
            raise ValueError("AV analysis utterance is invalid")
        _valid_interval(value, "startMs", "endMs")
        if value["startMs"] < window_start or value["endMs"] > window_end:
            raise ValueError("AV analysis utterance is outside its window")
        if value["spokenZh"] is None and value["subtitleZh"] is None:
            raise ValueError("AV analysis utterance has no text evidence")
        for key in ("spokenZh", "subtitleZh", "speakerRef", "intent", "emotion"):
            _optional_text(value[key])
        if value["confidence"] not in _CONFIDENCES:
            raise ValueError("AV analysis confidence is invalid")
    for value in document["visualOnly"]:
        if not isinstance(value, dict) or set(value) != _VISUAL_KEYS:
            raise ValueError("AV analysis visual-only item is invalid")
        _valid_interval(value, "startMs", "endMs")
        if value["startMs"] < window_start or value["endMs"] > window_end:
            raise ValueError("AV analysis visual-only item is outside its window")
        if not isinstance(value["textZh"], str) or not value["textZh"].strip() or value["confidence"] not in _CONFIDENCES:
            raise ValueError("AV analysis visual-only values are invalid")
    return document


def persist_diagnostics(path: Path, document: Mapping[str, object]) -> None:
    safe = {key: value for key, value in document.items() if key != "debug"}
    validate_av_analysis(safe)
    payload = json.dumps(safe, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(payload) > MAX_DIAGNOSTIC_BYTES:
        raise ValueError("AV diagnostics exceed bounded size")
    path.write_bytes(payload)


def persist_recovery_diagnostics(path: Path, document: Mapping[str, object]) -> None:
    """Persist only bounded, non-provider recovery metadata."""
    safe = {key: document.get(key) for key in _RECOVERY_KEYS}
    payload = json.dumps(safe, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(payload) > MAX_DIAGNOSTIC_BYTES:
        raise ValueError("AV recovery diagnostics exceed bounded size")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


class AVResponseRecoveryError(ValueError):
    """The provider response remained outside the canonical AV contract."""

    def __init__(self, failure_class: str):
        super().__init__("AV_SCHEMA_INVALID_FINAL: " + failure_class)
        self.failure_class = failure_class


def _normalize_structural_response(value: object) -> tuple[object, bool]:
    """Normalize only serialization/envelope wrappers; never semantic fields."""
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("```") and text.endswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3 and lines[0].strip().startswith("```") and lines[-1].strip() == "```":
                text = "\n".join(lines[1:-1]).strip()
        try:
            return _normalize_structural_response(json.loads(text))[0], True
        except (TypeError, ValueError, json.JSONDecodeError):
            return value, False
    if isinstance(value, dict) and set(value) == _AV_KEYS:
        return value, False
    if isinstance(value, dict) and len(value) == 1:
        key, nested = next(iter(value.items()))
        if key in _RECOVERY_WRAPPER_KEYS:
            normalized, _ = _normalize_structural_response(nested)
            return normalized, True
    return value, False


def _failure_class(value: object, error: Exception) -> str:
    if isinstance(value, str):
        return "INVALID_JSON"
    if not isinstance(value, dict):
        return "WRONG_ROOT_TYPE"
    if set(value) != _AV_KEYS:
        return "ROOT_FIELDS_MISMATCH"
    message = str(error)
    if "scene" in message:
        return "NESTED_SCENE_SCHEMA_MISMATCH"
    if "utterance" in message:
        return "NESTED_UTTERANCE_SCHEMA_MISMATCH"
    if "visual-only" in message:
        return "NESTED_VISUAL_SCHEMA_MISMATCH"
    return "CANONICAL_SCHEMA_MISMATCH"


def analysis_windows(duration_ms: int) -> list[tuple[int, int]]:
    if duration_ms <= 0:
        raise ValueError("AV duration must be positive")
    windows = []
    start = 0
    while start < duration_ms:
        end = min(duration_ms, start + MAX_WINDOW_MS)
        windows.append((start, end))
        if end == duration_ms:
            break
        start = end - WINDOW_OVERLAP_MS
    return windows


SUBDIVISION_CHILD_WINDOW_MS = 15_000
SUBDIVISION_OVERLAP_MS = 3_000
MAX_SUBDIVISION_DEPTH = 1
MIN_SUBDIVISION_CHILD_WINDOW_MS = 5_000
_SUBDIVISION_FAILURE_CLASSES = {
    "INVALID_JSON",
    "WRONG_ROOT_TYPE",
    "ROOT_FIELDS_MISMATCH",
    "CANONICAL_SCHEMA_MISMATCH",
    "NESTED_SCENE_SCHEMA_MISMATCH",
    "NESTED_UTTERANCE_SCHEMA_MISMATCH",
    "NESTED_VISUAL_SCHEMA_MISMATCH",
    "AV_SCHEMA_INVALID_FINAL",
}


def subdivision_windows(
    start_ms: int,
    end_ms: int,
    *,
    child_window_ms: int = SUBDIVISION_CHILD_WINDOW_MS,
    overlap_ms: int = SUBDIVISION_OVERLAP_MS,
    minimum_child_window_ms: int = MIN_SUBDIVISION_CHILD_WINDOW_MS,
    depth: int = 0,
) -> list[tuple[int, int]]:
    """Return one deterministic, bounded subdivision level for a parent window."""
    if not _is_int(start_ms) or not _is_int(end_ms) or end_ms <= start_ms:
        raise ValueError("AV subdivision parent window is invalid")
    if not _is_int(child_window_ms) or not _is_int(overlap_ms) or not _is_int(minimum_child_window_ms):
        raise ValueError("AV subdivision policy is invalid")
    if child_window_ms < minimum_child_window_ms or overlap_ms < 0 or overlap_ms >= child_window_ms:
        raise ValueError("AV subdivision policy is invalid")
    if depth >= MAX_SUBDIVISION_DEPTH or end_ms - start_ms <= child_window_ms:
        return []

    step_ms = child_window_ms - overlap_ms
    windows = [(start_ms, min(start_ms + child_window_ms, end_ms))]
    cursor = start_ms
    while True:
        next_start = cursor + step_ms
        if next_start >= end_ms:
            break
        if end_ms - next_start < minimum_child_window_ms:
            next_start = end_ms - minimum_child_window_ms
            if next_start <= cursor:
                break
            windows.append((next_start, end_ms))
            break
        windows.append((next_start, min(next_start + child_window_ms, end_ms)))
        cursor = next_start
    if windows[-1][1] != end_ms:
        windows.append((max(windows[-1][0], end_ms - minimum_child_window_ms), end_ms))
    return windows


def _attempt_count(recovery: Mapping[str, object]) -> int:
    value = recovery.get("attemptCount", 0)
    return value if _is_int(value) else 0


def _transport_count(recovery: Mapping[str, object]) -> int:
    value = recovery.get("totalTransportAttempts", recovery.get("transportAttempts", 0))
    return value if _is_int(value) else 0


def _valid_window_status(prefix: str, recovery: Mapping[str, object]) -> str:
    final_status = recovery.get("finalStatus")
    if final_status == "AV_SCHEMA_VALID_FIRST_PASS":
        return prefix + "_SCHEMA_VALID_FIRST_PASS"
    if final_status == "AV_SCHEMA_VALID_AFTER_NORMALIZATION":
        return prefix + "_SCHEMA_VALID_AFTER_NORMALIZATION"
    return prefix + "_SCHEMA_VALID_AFTER_RETRY"


def _merge_analysis_documents(
    documents: Sequence[Mapping[str, object]],
    start_ms: int,
    end_ms: int,
) -> dict[str, object]:
    if not documents:
        raise AVResponseRecoveryError("AV_SCHEMA_INVALID_FINAL")
    utterances: list[dict[str, object]] = []
    visual_only: list[dict[str, object]] = []
    for document in documents:
        validate_av_analysis(document)
        utterances.extend(deepcopy(document["utterances"]))
        visual_only.extend(deepcopy(document["visualOnly"]))
    utterances.sort(key=lambda value: (
        int(value["startMs"]),
        int(value["endMs"]),
        str(value.get("spokenZh") or value.get("subtitleZh") or ""),
    ))
    visual_only.sort(key=lambda value: (
        int(value["startMs"]),
        int(value["endMs"]),
        str(value["textZh"]),
    ))
    return validate_av_analysis({
        "windowStartMs": start_ms,
        "windowEndMs": end_ms,
        "scene": deepcopy(documents[0]["scene"]),
        "utterances": utterances,
        "visualOnly": visual_only,
    })


def _default_reconciliation(
    cues: Sequence[Mapping[str, object]],
    av_analysis: Mapping[str, object],
    unresolved: Mapping[str, object] | None,
) -> object:
    from source_reconciler import reconcile_source

    return reconcile_source(cues, av_analysis, unresolved)


def persist_subdivision_diagnostics(path: Path, document: Mapping[str, object]) -> None:
    """Persist only bounded parent/child window evidence, never provider payloads."""
    safe_windows = []
    for item in document.get("windows", []):
        if not isinstance(item, Mapping):
            continue
        safe_children = []
        for child in item.get("children", []):
            if not isinstance(child, Mapping):
                continue
            safe_children.append({
                key: child.get(key)
                for key in (
                    "startMs", "endMs", "attemptCount", "logicalAttemptCount",
                    "transportAttempts", "totalTransportAttempts", "httpStatus", "finalTransportStatus",
                    "transientRetryUsed", "schemaValidationStatus",
                    "repairGenerationUsed", "status", "finalStatus", "failureClass",
                    "coordinateNormalizationApplied", "boundaryNormalizationApplied",
                )
            })
        safe_windows.append({
            key: item.get(key)
            for key in (
                "parentWindowStartMs", "parentWindowEndMs", "subdivisionDepth",
                "attemptCount", "logicalAttemptCount", "transportAttempts", "totalTransportAttempts", "httpStatus",
                "finalTransportStatus", "transientRetryUsed", "schemaValidationStatus",
                "repairGenerationUsed", "status", "finalStatus", "failureClass",
                "boundaryNormalizationApplied",
            )
        } | {"children": safe_children})
    safe = {
        "providerCalls": document.get("providerCalls", 0),
        "logicalAvGenerations": document.get("logicalAvGenerations", 0),
        "httpRequests": document.get("httpRequests", document.get("providerCalls", 0)),
        "schemaRepairGenerations": document.get("schemaRepairGenerations", 0),
        "subdivisionChildLogicalGenerations": document.get("subdivisionChildLogicalGenerations", 0),
        "finalReconciliationStatus": document.get("finalReconciliationStatus"),
        "windows": safe_windows,
    }
    payload = json.dumps(safe, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(payload) > MAX_DIAGNOSTIC_BYTES:
        raise ValueError("AV subdivision diagnostics exceed bounded size")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


class GeminiAVAnalyzer:
    def __init__(self, transport: StructuredGeminiTransport | None = None, *, model: str | None = None) -> None:
        self.model = model or os.environ.get("DUBVI_AV_MODEL", "gemini-2.5-flash")
        self.transport = transport or gemini_json_transport(model=self.model)
        self.last_recovery: dict[str, object] = {}

    def _payload(self, media: bytes, start_ms: int, end_ms: int, *, repair: bool) -> dict[str, object]:
        window_instruction = (
            f"Analyze only source window {start_ms}-{end_ms} ms and return absolute timestamps within that window. "
        )
        instruction = (
            "The previous attempt violated the required canonical AV structure. "
            "Return a fresh complete canonical response for the same window. "
            "Do not infer or repair missing semantic values from the malformed response. "
            if repair else ""
        )
        payload: dict[str, object] = {
            "contents": [{"role": "user", "parts": [
                {"text": instruction + window_instruction + "Return JSON only using exactly windowStartMs, windowEndMs, scene, utterances, and visualOnly. Each utterance must separately provide spokenZh and subtitleZh plus speakerRef, intent, emotion, and confidence HIGH/MEDIUM/LOW. Visual-only text belongs only in visualOnly and is never dialogue."},
                {"inline_data": {"mime_type": "video/mp4", "data": base64.b64encode(media).decode("ascii")}},
            ]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0},
        }
        if repair:
            parity = validate_av_schema_parity(AV_RESPONSE_SCHEMA)
            if parity["status"] != "PASS":
                raise AVResponseRecoveryError("PROVIDER_SCHEMA_PARITY_INVALID")
            payload["generationConfig"]["responseSchema"] = deepcopy(AV_RESPONSE_SCHEMA)
        return payload

    def _raw_call(self, payload: Mapping[str, object]) -> object:
        call_raw = getattr(self.transport, "call_raw", None)
        if call_raw is not None:
            return call_raw(payload)
        return self.transport.call(payload, required_keys=_AV_KEYS)

    def _save_recovery(self, path: Path | None) -> None:
        if path is not None:
            persist_recovery_diagnostics(path, self.last_recovery)

    def analyze_media(
        self,
        media_path: Path,
        *,
        start_ms: int = 0,
        end_ms: int | None = None,
        recovery_metadata_path: Path | None = None,
    ) -> dict[str, object]:
        if end_ms is None or end_ms - start_ms > MAX_WINDOW_MS:
            raise ValueError("AV analysis window must be at most 30 seconds")
        media = media_path.read_bytes()
        self.last_recovery = {
            "windowStartMs": start_ms,
            "windowEndMs": end_ms,
            "attemptCount": 0,
            "logicalAttempt": 0,
            "transportAttempts": 0,
            "totalTransportAttempts": 0,
            "httpStatus": None,
            "finalTransportStatus": None,
            "transientRetryUsed": False,
            "schemaValidationStatus": "NOT_RUN",
            "repairGenerationUsed": False,
            "failureClass": None,
            "normalizationApplied": False,
            "coordinateNormalizationApplied": False,
            "boundaryNormalizationApplied": False,
            "repairRetryUsed": False,
            "finalStatus": "RUNNING",
        }
        try:
            structural_failure = False
            for attempt in range(2):
                self.last_recovery["attemptCount"] = attempt + 1
                self.last_recovery["logicalAttempt"] = attempt + 1
                self.last_recovery["repairRetryUsed"] = structural_failure
                self.last_recovery["repairGenerationUsed"] = structural_failure
                try:
                    raw = self._raw_call(self._payload(media, start_ms, end_ms, repair=structural_failure))
                except ProviderError as error:
                    self.last_recovery["failureClass"] = error.classification
                    transport_attempts = getattr(error, "transport_attempts", 1) or 1
                    self.last_recovery["transportAttempts"] = transport_attempts
                    self.last_recovery["totalTransportAttempts"] = (
                        int(self.last_recovery.get("totalTransportAttempts", 0)) + transport_attempts
                    )
                    self.last_recovery["httpStatus"] = getattr(error, "http_status", None) or error.classification
                    self.last_recovery["finalTransportStatus"] = error.classification
                    self.last_recovery["transientRetryUsed"] = bool(getattr(error, "transient_retry_used", False))
                    self.last_recovery["schemaValidationStatus"] = "NOT_RUN"
                    self.last_recovery["finalStatus"] = (
                        "AV_PROVIDER_TEMPORARILY_UNAVAILABLE"
                        if error.transient
                        else "AV_PROVIDER_ERROR"
                    )
                    raise
                except StructuredResponseError as error:
                    self.last_recovery["failureClass"] = "PROVIDER_RESPONSE_ENVELOPE_INVALID"
                    self.last_recovery["transportAttempts"] = 1
                    self.last_recovery["totalTransportAttempts"] = (
                        int(self.last_recovery.get("totalTransportAttempts", 0)) + 1
                    )
                    self.last_recovery["schemaValidationStatus"] = "INVALID"
                    structural_failure = True
                    if attempt == 0:
                        continue
                    self.last_recovery["finalStatus"] = "AV_SCHEMA_INVALID_FINAL"
                    raise AVResponseRecoveryError("PROVIDER_RESPONSE_ENVELOPE_INVALID") from error

                transport = getattr(self.transport, "last_call", {})
                transport_attempts = transport.get("transportAttempts", 1)
                self.last_recovery["transportAttempts"] = transport_attempts
                self.last_recovery["totalTransportAttempts"] = (
                    int(self.last_recovery.get("totalTransportAttempts", 0)) + transport_attempts
                )
                self.last_recovery["httpStatus"] = transport.get("httpStatus", 200)
                self.last_recovery["finalTransportStatus"] = transport.get("finalTransportStatus", "HTTP_200")
                self.last_recovery["transientRetryUsed"] = bool(transport.get("transientRetryUsed", False))
                candidate, normalized = _normalize_structural_response(raw)
                candidate, coordinate_normalized = _normalize_window_coordinates(candidate, start_ms, end_ms)
                candidate, boundary_normalized = _normalize_boundary_utterance_window(candidate, start_ms, end_ms)
                self.last_recovery["normalizationApplied"] = bool(self.last_recovery["normalizationApplied"] or normalized)
                self.last_recovery["coordinateNormalizationApplied"] = bool(
                    self.last_recovery["coordinateNormalizationApplied"] or coordinate_normalized
                )
                self.last_recovery["boundaryNormalizationApplied"] = bool(
                    self.last_recovery["boundaryNormalizationApplied"] or boundary_normalized
                )
                try:
                    result = validate_av_analysis(candidate)
                except ValueError as error:
                    self.last_recovery["failureClass"] = _failure_class(candidate, error)
                    self.last_recovery["schemaValidationStatus"] = "INVALID"
                    structural_failure = True
                    if attempt == 0:
                        continue
                    self.last_recovery["finalStatus"] = "AV_SCHEMA_INVALID_FINAL"
                    raise AVResponseRecoveryError(str(self.last_recovery["failureClass"])) from error

                self.last_recovery["schemaValidationStatus"] = "VALID"
                self.last_recovery["finalStatus"] = (
                    "AV_SCHEMA_VALID_FIRST_PASS"
                    if attempt == 0 and not (normalized or boundary_normalized)
                    else "AV_SCHEMA_VALID_AFTER_NORMALIZATION"
                    if attempt == 0 and (normalized or boundary_normalized)
                    else "AV_SCHEMA_VALID_AFTER_RETRY"
                )
                return result
            raise AVResponseRecoveryError(str(self.last_recovery["failureClass"] or "CANONICAL_SCHEMA_MISMATCH"))
        finally:
            self._save_recovery(recovery_metadata_path)

    def analyze_cached(
        self,
        media_path: Path,
        cache_path: Path,
        *,
        start_ms: int,
        end_ms: int,
        recovery_metadata_path: Path | None = None,
    ) -> dict[str, object]:
        if cache_path.is_file():
            self.last_recovery = {
                "windowStartMs": start_ms,
                "windowEndMs": end_ms,
                "attemptCount": 0,
                "logicalAttempt": 0,
                "transportAttempts": 0,
                "totalTransportAttempts": 0,
                "httpStatus": None,
                "finalTransportStatus": None,
                "transientRetryUsed": False,
                "schemaValidationStatus": "NOT_RUN",
                "repairGenerationUsed": False,
                "failureClass": None,
                "normalizationApplied": False,
                "coordinateNormalizationApplied": False,
                "boundaryNormalizationApplied": False,
                "repairRetryUsed": False,
                "finalStatus": "AV_CACHE_HIT",
            }
            self._save_recovery(recovery_metadata_path)
            return validate_av_analysis(json.loads(cache_path.read_text(encoding="utf-8")))
        result = self.analyze_media(
            media_path,
            start_ms=start_ms,
            end_ms=end_ms,
            recovery_metadata_path=recovery_metadata_path,
        )
        persist_diagnostics(cache_path, result)
        return result


def analyze_media_windows_with_subdivision(
    analyzer: GeminiAVAnalyzer,
    media_path: Path,
    duration_ms: int,
    *,
    cues: Sequence[Mapping[str, object]] = (),
    unresolved: Mapping[str, object] | None = None,
    reconciliation: Callable[[Sequence[Mapping[str, object]], Mapping[str, object], Mapping[str, object] | None], object] | None = None,
    diagnostics_path: Path | None = None,
) -> dict[str, object]:
    """Analyze primary windows and use one bounded child fallback for schema failures."""
    primary_windows = analysis_windows(duration_ms)
    documents: list[Mapping[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    provider_calls = 0
    logical_av_generations = 0
    schema_repair_generations = 0
    subdivision_child_logical_generations = 0

    def record_accounting(recovery: Mapping[str, object], *, child: bool = False) -> tuple[int, int]:
        nonlocal provider_calls, logical_av_generations
        nonlocal schema_repair_generations, subdivision_child_logical_generations
        logical_count = _attempt_count(recovery)
        transport_count = _transport_count(recovery)
        provider_calls += transport_count
        logical_av_generations += logical_count
        if recovery.get("repairGenerationUsed"):
            schema_repair_generations += 1
        if child:
            subdivision_child_logical_generations += logical_count
        return logical_count, transport_count

    def persist(status: str | None = None) -> None:
        if diagnostics_path is not None:
            persist_subdivision_diagnostics(
                diagnostics_path,
                {
                    "providerCalls": provider_calls,
                    "logicalAvGenerations": logical_av_generations,
                    "httpRequests": provider_calls,
                    "schemaRepairGenerations": schema_repair_generations,
                    "subdivisionChildLogicalGenerations": subdivision_child_logical_generations,
                    "finalReconciliationStatus": status,
                    "windows": diagnostics,
                },
            )

    for parent_start, parent_end in primary_windows:
        entry: dict[str, object] = {
            "parentWindowStartMs": parent_start,
            "parentWindowEndMs": parent_end,
            "subdivisionDepth": 0,
            "attemptCount": 0,
            "logicalAttemptCount": 0,
            "transportAttempts": 0,
            "httpStatus": None,
            "finalTransportStatus": None,
            "transientRetryUsed": False,
            "schemaValidationStatus": "NOT_RUN",
            "repairGenerationUsed": False,
            "boundaryNormalizationApplied": False,
            "finalStatus": None,
            "status": "AV_PARENT_WINDOW_STARTED",
            "failureClass": None,
            "children": [],
        }
        try:
            document = analyzer.analyze_media(media_path, start_ms=parent_start, end_ms=parent_end)
        except ProviderError as error:
            recovery = dict(analyzer.last_recovery)
            logical_count, transport_count = record_accounting(recovery)
            entry.update({
                "attemptCount": logical_count,
                "logicalAttemptCount": logical_count,
                "transportAttempts": transport_count,
                "httpStatus": recovery.get("httpStatus", error.classification),
                "finalTransportStatus": recovery.get("finalTransportStatus", error.classification),
                "transientRetryUsed": bool(recovery.get("transientRetryUsed", False)),
                "schemaValidationStatus": recovery.get("schemaValidationStatus", "NOT_RUN"),
                "repairGenerationUsed": bool(recovery.get("repairGenerationUsed", False)),
                "boundaryNormalizationApplied": bool(recovery.get("boundaryNormalizationApplied", False)),
                "finalStatus": recovery.get("finalStatus"),
                "failureClass": error.classification,
                "status": "AV_PARENT_PROVIDER_ERROR",
            })
            diagnostics.append(entry)
            persist("AV_PARENT_PROVIDER_ERROR")
            raise
        except AVResponseRecoveryError as error:
            recovery = dict(analyzer.last_recovery)
            parent_attempts, parent_transport_attempts = record_accounting(recovery)
            entry.update({
                "attemptCount": parent_attempts,
                "logicalAttemptCount": parent_attempts,
                "transportAttempts": parent_transport_attempts,
                "httpStatus": recovery.get("httpStatus"),
                "finalTransportStatus": recovery.get("finalTransportStatus"),
                "transientRetryUsed": bool(recovery.get("transientRetryUsed", False)),
                "schemaValidationStatus": recovery.get("schemaValidationStatus", "INVALID"),
                "repairGenerationUsed": bool(recovery.get("repairGenerationUsed", False)),
                "boundaryNormalizationApplied": bool(recovery.get("boundaryNormalizationApplied", False)),
                "finalStatus": recovery.get("finalStatus"),
                "failureClass": error.failure_class,
                "status": "AV_PARENT_WINDOW_FAILED",
            })
            if error.failure_class not in _SUBDIVISION_FAILURE_CLASSES:
                diagnostics.append(entry)
                persist("AV_PARENT_WINDOW_FAILED")
                raise
            child_windows = subdivision_windows(parent_start, parent_end, depth=0)
            if not child_windows:
                diagnostics.append(entry)
                persist("AV_PARENT_WINDOW_FAILED")
                raise
            entry["status"] = "AV_WINDOW_SUBDIVISION_STARTED"
            children = entry["children"]
            for child_start, child_end in child_windows:
                child: dict[str, object] = {
                    "startMs": child_start,
                    "endMs": child_end,
                    "attemptCount": 0,
                    "logicalAttemptCount": 0,
                    "transportAttempts": 0,
                    "httpStatus": None,
                    "finalTransportStatus": None,
                    "transientRetryUsed": False,
                    "schemaValidationStatus": "NOT_RUN",
                    "repairGenerationUsed": False,
                    "status": "AV_CHILD_WINDOW_STARTED",
                    "finalStatus": None,
                    "failureClass": None,
                    "coordinateNormalizationApplied": False,
                    "boundaryNormalizationApplied": False,
                }
                try:
                    child_document = analyzer.analyze_media(
                        media_path,
                        start_ms=child_start,
                        end_ms=child_end,
                    )
                except ProviderError as child_error:
                    recovery = dict(analyzer.last_recovery)
                    attempts, transport_attempts = record_accounting(recovery, child=True)
                    child.update({
                        "attemptCount": attempts,
                        "logicalAttemptCount": attempts,
                        "transportAttempts": transport_attempts,
                        "httpStatus": recovery.get("httpStatus", child_error.classification),
                        "finalTransportStatus": recovery.get("finalTransportStatus", child_error.classification),
                        "transientRetryUsed": bool(recovery.get("transientRetryUsed", False)),
                        "schemaValidationStatus": recovery.get("schemaValidationStatus", "NOT_RUN"),
                        "repairGenerationUsed": bool(recovery.get("repairGenerationUsed", False)),
                        "finalStatus": recovery.get("finalStatus"),
                        "failureClass": child_error.classification,
                        "status": "AV_CHILD_PROVIDER_ERROR",
                    })
                    children.append(child)
                    entry["status"] = "AV_WINDOW_SUBDIVISION_FAILED"
                    diagnostics.append(entry)
                    persist("AV_WINDOW_SUBDIVISION_FAILED")
                    raise
                except AVResponseRecoveryError as child_error:
                    recovery = dict(analyzer.last_recovery)
                    attempts, transport_attempts = record_accounting(recovery, child=True)
                    child.update({
                        "attemptCount": attempts,
                        "logicalAttemptCount": attempts,
                        "transportAttempts": transport_attempts,
                        "httpStatus": recovery.get("httpStatus"),
                        "finalTransportStatus": recovery.get("finalTransportStatus"),
                        "transientRetryUsed": bool(recovery.get("transientRetryUsed", False)),
                        "schemaValidationStatus": recovery.get("schemaValidationStatus", "INVALID"),
                        "repairGenerationUsed": bool(recovery.get("repairGenerationUsed", False)),
                        "finalStatus": recovery.get("finalStatus"),
                        "failureClass": child_error.failure_class,
                        "status": "AV_CHILD_SCHEMA_INVALID_FINAL",
                    })
                    children.append(child)
                    entry["status"] = "AV_WINDOW_SUBDIVISION_FAILED"
                    diagnostics.append(entry)
                    persist("AV_WINDOW_SUBDIVISION_FAILED")
                    raise
                else:
                    recovery = dict(analyzer.last_recovery)
                    attempts, transport_attempts = record_accounting(recovery, child=True)
                    child.update({
                        "attemptCount": attempts,
                        "logicalAttemptCount": attempts,
                        "transportAttempts": transport_attempts,
                        "httpStatus": recovery.get("httpStatus", 200),
                        "finalTransportStatus": recovery.get("finalTransportStatus", "HTTP_200"),
                        "transientRetryUsed": bool(recovery.get("transientRetryUsed", False)),
                        "schemaValidationStatus": recovery.get("schemaValidationStatus", "VALID"),
                        "repairGenerationUsed": bool(recovery.get("repairGenerationUsed", False)),
                        "finalStatus": recovery.get("finalStatus"),
                        "coordinateNormalizationApplied": bool(recovery.get("coordinateNormalizationApplied")),
                        "boundaryNormalizationApplied": bool(recovery.get("boundaryNormalizationApplied")),
                        "status": _valid_window_status("AV_CHILD", recovery),
                    })
                    children.append(child)
                    documents.append(child_document)
            entry["status"] = "AV_WINDOW_SUBDIVISION_SUCCEEDED"
        else:
            recovery = dict(analyzer.last_recovery)
            attempts, transport_attempts = record_accounting(recovery)
            entry.update({
                "attemptCount": attempts,
                "logicalAttemptCount": attempts,
                "transportAttempts": transport_attempts,
                "httpStatus": recovery.get("httpStatus", 200),
                "finalTransportStatus": recovery.get("finalTransportStatus", "HTTP_200"),
                "transientRetryUsed": bool(recovery.get("transientRetryUsed", False)),
                "schemaValidationStatus": recovery.get("schemaValidationStatus", "VALID"),
                "repairGenerationUsed": bool(recovery.get("repairGenerationUsed", False)),
                "finalStatus": recovery.get("finalStatus"),
                "coordinateNormalizationApplied": bool(recovery.get("coordinateNormalizationApplied")),
                "boundaryNormalizationApplied": bool(recovery.get("boundaryNormalizationApplied")),
                "status": _valid_window_status("AV_PRIMARY", recovery),
            })
            documents.append(document)
        diagnostics.append(entry)

    merged = _merge_analysis_documents(documents, 0, duration_ms)
    reconciler = reconciliation or _default_reconciliation
    reconstructed = reconciler(cues, merged, unresolved)
    persist("AV_WINDOW_SUBDIVISION_RECONCILED")
    return {
        "av_analysis": merged,
        "reconstructed": reconstructed,
        "diagnostics": diagnostics,
        "providerCalls": provider_calls,
        "logicalAvGenerations": logical_av_generations,
        "httpRequests": provider_calls,
        "schemaRepairGenerations": schema_repair_generations,
        "subdivisionChildLogicalGenerations": subdivision_child_logical_generations,
    }
