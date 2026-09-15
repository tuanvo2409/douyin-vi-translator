from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import llm_transport
import multimodal_vi_harness

try:
    import cp2_gemini_adapter
except ModuleNotFoundError:
    cp2_gemini_adapter = None


def _scene() -> dict[str, object]:
    return {
        "summary": "Lớp học nhập vai.",
        "tone": "light",
        "register": "spoken",
        "speakers": {},
        "relationships": {},
        "pronoun_policy": {},
        "roleplay_terms": {},
        "running_jokes": [],
        "glossary": {},
    }


def _records() -> list[dict[str, object]]:
    return [
        {"utteranceId": "u1", "resolvedZh": "甲", "kind": "dialogue"},
        {"utteranceId": "u2", "resolvedZh": "乙", "kind": "dialogue"},
    ]


class CP2GeminiAdapterTests(unittest.TestCase):
    def _adapter(self, responses, diagnostic_path: Path | None = None):
        calls = []
        remaining = list(responses)

        def request(payload):
            calls.append(payload)
            value = remaining.pop(0)
            if isinstance(value, BaseException):
                raise value
            return value

        transport = llm_transport.StructuredGeminiTransport(
            request=request,
            sleeper=lambda _delay: None,
        )
        return cp2_gemini_adapter.CP2GeminiAdapter(
            transport=transport,
            model="test-model",
            diagnostic_path=diagnostic_path,
        ), calls

    def test_module_exists(self) -> None:
        self.assertIsNotNone(cp2_gemini_adapter)

    @unittest.skipIf(cp2_gemini_adapter is None, "CP2 Gemini adapter is not implemented")
    def test_scene_context_validates_and_declares_its_schema(self) -> None:
        bridge, calls = self._adapter([_scene()])
        self.assertEqual(_scene(), bridge({"mode": "scene", "utterances": _records()}))
        request = calls[0]
        self.assertEqual(
            cp2_gemini_adapter.OPERATION_SCHEMAS["scene"],
            request["generationConfig"]["responseSchema"],
        )
        self.assertEqual("PASS", cp2_gemini_adapter.validate_schema_parity("scene", request["generationConfig"]["responseSchema"])["status"])

    @unittest.skipIf(cp2_gemini_adapter is None, "CP2 Gemini adapter is not implemented")
    def test_malformed_scene_fails_closed(self) -> None:
        bridge, _ = self._adapter([{**_scene(), "tone": ["wrong"]}])
        with self.assertRaises(cp2_gemini_adapter.CP2StructuredResponseError):
            bridge({"mode": "scene", "utterances": _records()})

    @unittest.skipIf(cp2_gemini_adapter is None, "CP2 Gemini adapter is not implemented")
    def test_semantic_valid_and_reordered_rows_are_canonicalized_by_identity(self) -> None:
        bridge, _ = self._adapter([{"translations": [
            {"utteranceId": "u2", "semanticVi": "Hai"},
            {"utteranceId": "u1", "semanticVi": "Một"},
        ]}])
        result = bridge({"mode": "semantic", "utterances": _records()})
        self.assertEqual(["u1", "u2"], [item["utteranceId"] for item in result["translations"]])
        self.assertEqual(["Một", "Hai"], [item["semanticVi"] for item in result["translations"]])

    @unittest.skipIf(cp2_gemini_adapter is None, "CP2 Gemini adapter is not implemented")
    def test_semantic_rejects_missing_root_translation_field_unknown_duplicate_and_missing_ids(self) -> None:
        bad_documents = [
            {},
            {"translations": [{"utteranceId": "u1"}, {"utteranceId": "u2", "semanticVi": "Hai"}]},
            {"translations": [{"utteranceId": "unknown", "semanticVi": "Một"}, {"utteranceId": "u2", "semanticVi": "Hai"}]},
            {"translations": [{"utteranceId": "u1", "semanticVi": "Một"}, {"utteranceId": "u1", "semanticVi": "Một nữa"}]},
            {"translations": [{"utteranceId": "u1", "semanticVi": "Một"}]},
        ]
        for document in bad_documents:
            bridge, _ = self._adapter([document])
            with self.subTest(document=document), self.assertRaises(cp2_gemini_adapter.CP2StructuredResponseError):
                bridge({"mode": "semantic", "utterances": _records()})

    @unittest.skipIf(cp2_gemini_adapter is None, "CP2 Gemini adapter is not implemented")
    def test_native_and_repair_preserve_the_expected_identity_set(self) -> None:
        responses = [
            {"translations": [{"utteranceId": "u1", "nativeVi": "Một"}, {"utteranceId": "u2", "nativeVi": "Hai"}]},
            {"translations": [{"utteranceId": "u1", "nativeVi": "Một đã sửa"}]},
        ]
        bridge, _ = self._adapter(responses)
        native = bridge({"mode": "native", "utterances": _records()})
        repair = bridge({"mode": "repair", "utterances": [_records()[0]]})
        self.assertEqual({"u1", "u2"}, {item["utteranceId"] for item in native["translations"]})
        self.assertEqual(["u1"], [item["utteranceId"] for item in repair["translations"]])

    @unittest.skipIf(cp2_gemini_adapter is None, "CP2 Gemini adapter is not implemented")
    def test_malformed_repair_fails_closed_without_inventing_a_translation(self) -> None:
        bridge, _ = self._adapter([{"translations": [{"utteranceId": "u1", "nativeVi": ""}]}])
        with self.assertRaises(cp2_gemini_adapter.CP2StructuredResponseError):
            bridge({"mode": "repair", "utterances": [_records()[0]]})

    @unittest.skipIf(cp2_gemini_adapter is None, "CP2 Gemini adapter is not implemented")
    def test_qa_validates_canonical_enum_and_reason_shape(self) -> None:
        valid = {"issues": [{"utteranceId": "u1", "code": "STIFF_VI", "severity": "LOW", "reason": "brief", "evidence": "line"}]}
        bridge, _ = self._adapter([valid])
        self.assertEqual(valid, bridge({"mode": "qa", "utterances": _records()}))
        invalid = {"issues": [{"utteranceId": "u1", "code": "STIFF_VI", "severity": "BAD", "reason": 3, "evidence": "line"}]}
        bridge, _ = self._adapter([invalid])
        with self.assertRaises(cp2_gemini_adapter.CP2StructuredResponseError):
            bridge({"mode": "qa", "utterances": _records()})

    @unittest.skipIf(cp2_gemini_adapter is None, "CP2 Gemini adapter is not implemented")
    def test_schema_parity_requires_every_canonical_field_nested_requirement_and_nonnullable_semantics(self) -> None:
        for operation, schema in cp2_gemini_adapter.OPERATION_SCHEMAS.items():
            with self.subTest(operation=operation):
                self.assertEqual("PASS", cp2_gemini_adapter.validate_schema_parity(operation, schema)["status"])
        weakened = deepcopy(cp2_gemini_adapter.OPERATION_SCHEMAS["semantic"])
        weakened["properties"]["translations"]["items"]["required"].remove("semanticVi")
        self.assertEqual("FAIL", cp2_gemini_adapter.validate_schema_parity("semantic", weakened)["status"])
        nullable = deepcopy(cp2_gemini_adapter.OPERATION_SCHEMAS["semantic"])
        nullable["properties"]["translations"]["items"]["properties"]["semanticVi"]["nullable"] = True
        self.assertEqual("FAIL", cp2_gemini_adapter.validate_schema_parity("semantic", nullable)["status"])

    @unittest.skipIf(cp2_gemini_adapter is None, "CP2 Gemini adapter is not implemented")
    def test_final_transport_payload_keeps_nested_schema_requirements(self) -> None:
        bridge, calls = self._adapter([{"translations": [{"utteranceId": "u1", "semanticVi": "Một"}, {"utteranceId": "u2", "semanticVi": "Hai"}]}])
        bridge({"mode": "semantic", "utterances": _records()})
        schema = calls[0]["generationConfig"]["responseSchema"]
        self.assertEqual(["utteranceId", "semanticVi"], schema["properties"]["translations"]["items"]["required"])
        self.assertEqual("application/json", calls[0]["generationConfig"]["responseMimeType"])

    @unittest.skipIf(cp2_gemini_adapter is None, "CP2 Gemini adapter is not implemented")
    def test_transport_errors_are_not_structural_repairs_or_duplicate_http_retries(self) -> None:
        attempts = []

        def request(_payload):
            attempts.append(True)
            if len(attempts) == 1:
                raise llm_transport.ProviderError("HTTP_503", transient=True, http_status=503)
            return {"translations": [{"utteranceId": "u1", "semanticVi": "Một"}, {"utteranceId": "u2", "semanticVi": "Hai"}]}

        transport = llm_transport.StructuredGeminiTransport(request=request, max_transport_attempts=2, sleeper=lambda _delay: None)
        bridge = cp2_gemini_adapter.CP2GeminiAdapter(transport=transport, model="test-model")
        bridge({"mode": "semantic", "utterances": _records()})
        self.assertEqual(2, len(attempts))

        bridge, calls = self._adapter([
            llm_transport.ProviderError("HTTP_503", transient=True, http_status=503),
            llm_transport.ProviderError("HTTP_503", transient=True, http_status=503),
            llm_transport.ProviderError("HTTP_503", transient=True, http_status=503),
        ])
        with self.assertRaises(llm_transport.ProviderError):
            bridge({"mode": "semantic", "utterances": _records()})
        self.assertEqual(3, len(calls))

    @unittest.skipIf(cp2_gemini_adapter is None, "CP2 Gemini adapter is not implemented")
    def test_structural_diagnostic_is_bounded_and_never_contains_provider_or_fixture_material(self) -> None:
        fake_secret = "TEST_PROVIDER_SECRET_SUFFIX"
        with tempfile.TemporaryDirectory() as directory:
            diagnostic = Path(directory) / "diagnostic.json"
            bridge, _ = self._adapter([{"translations": [{"utteranceId": "u1", "semanticVi": fake_secret}]}], diagnostic)
            with self.assertRaises(cp2_gemini_adapter.CP2StructuredResponseError):
                bridge({"mode": "semantic", "utterances": _records()})
            saved = diagnostic.read_text(encoding="utf-8")
        self.assertNotIn(fake_secret, saved)
        self.assertIn("semantic", saved)
        self.assertNotIn("BLIND_FIXTURE_02", Path(cp2_gemini_adapter.__file__).read_text(encoding="utf-8"))

    @unittest.skipIf(cp2_gemini_adapter is None, "CP2 Gemini adapter is not implemented")
    def test_cached_scene_with_wrong_nested_types_is_not_resume_compatible(self) -> None:
        records = _records()
        contract = multimodal_vi_harness._input_contract(records, "test-model", "vn_native_light_comedy", "cp2-v2")
        state = {
            "version": 2,
            "input": contract,
            "stages": {
                "scene": {"status": "completed", "payload": {**_scene(), "speakers": ["not an object"]}},
                "semantic": {"status": "pending", "payload": None},
                "native": {"status": "pending", "payload": None},
                "qa": {"status": "pending", "payload": None},
            },
        }
        self.assertFalse(multimodal_vi_harness._state_is_compatible(state, contract, records))


if __name__ == "__main__":
    unittest.main()
