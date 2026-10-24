from __future__ import annotations

import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def _optional_module(name: str):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        return None


scene_memory = _optional_module("scene_memory")
vn_localizer = _optional_module("vn_localizer")
vn_quality = _optional_module("vn_quality")
vi_harness = _optional_module("multimodal_vi_harness")
llm_transport = _optional_module("llm_transport")


class SceneAwareLocalizationTests(unittest.TestCase):
    def test_cp2_modules_exist(self) -> None:
        self.assertIsNotNone(scene_memory)
        self.assertIsNotNone(vn_localizer)
        self.assertIsNotNone(vn_quality)
        self.assertIsNotNone(vi_harness)

    @unittest.skipIf(scene_memory is None, "CP2 scene memory is not implemented")
    def test_scene_memory_is_bounded_serializable_and_preserves_uncertainty(self) -> None:
        memory = scene_memory.build_scene_memory({
            "summary": "Lớp học nhập vai cung đình",
            "register": "roleplay",
            "relationships": {"S1:S2": {"value": "unknown", "confidence": "low"}},
            "roleplay_terms": {"本公主": "bổn công chúa", "老奴": "lão nô"},
            "running_jokes": ["school roleplay"],
        })
        encoded = json.dumps(memory, ensure_ascii=False)
        self.assertLessEqual(len(encoded.encode("utf-8")), scene_memory.MAX_SCENE_MEMORY_BYTES)
        self.assertEqual("low", memory["relationships"]["S1:S2"]["confidence"])
        self.assertEqual("unknown", memory["relationships"]["S1:S2"]["value"])
        self.assertEqual("bổn công chúa", memory["roleplayTerms"]["本公主"])

    @unittest.skipIf(vn_localizer is None, "CP2 localizer is not implemented")
    def test_injectable_two_pass_localizer_excludes_visual_and_marks_unresolved_for_review(self) -> None:
        calls = []

        def request(payload):
            calls.append(payload)
            mode = payload["mode"]
            records = payload["utterances"]
            key = "semanticVi" if mode == "semantic" else "nativeVi"
            return {"translations": [{"utteranceId": item["utteranceId"], key: "Câu tiếng Việt tự nhiên"} for item in records]}

        records = [
            {"utteranceId": "u1", "resolvedZh": "本公主说话", "source_confidence": "high", "kind": "dialogue"},
            {"utteranceId": "v1", "resolvedZh": "垃圾桶", "source_confidence": "high", "kind": "visual_only"},
            {"utteranceId": "u2", "resolvedZh": "甲乙", "source_confidence": "unresolved", "kind": "dialogue"},
        ]
        localized = vn_localizer.localize(records, scene_memory={}, request=request, profile="vn_native_light_comedy")
        self.assertEqual(["u1"], [item["utteranceId"] for item in localized["accepted"]])
        self.assertEqual(["u2"], [item["utteranceId"] for item in localized["review_required"]])
        self.assertEqual(2, len(calls))
        self.assertTrue(all(len(call["utterances"]) == 1 for call in calls))
        self.assertNotIn("\u672c\u516c\u4e3b", localized["accepted"][0]["semanticVi"])

    @unittest.skipIf(vn_localizer is None, "CP2 localizer is not implemented")
    def test_native_pass_receives_spoken_vietnamese_editorial_contract(self) -> None:
        seen = []

        def request(payload):
            seen.append(payload)
            key = "semanticVi" if payload["mode"] == "semantic" else "nativeVi"
            return {"translations": [{"utteranceId": "u1", key: "Câu tự nhiên"}]}

        vn_localizer.localize(
            [{"utteranceId": "u1", "resolvedZh": "甲", "source_confidence": "high", "kind": "dialogue"}],
            scene_memory={}, request=request, profile="vn_native_light_comedy")
        contract = seen[1]["editorial_contract"]
        self.assertIn("spoken Vietnamese", contract["prefer"])
        self.assertIn("discourse particles", contract["prefer"])
        self.assertIn("contextual wit without invented jokes", contract["prefer"])
        self.assertIn("mechanically preserving the source subject", contract["avoid"])

    @unittest.skipIf(vn_quality is None, "CP2 quality checks are not implemented")
    def test_quality_detects_identity_loss_cjk_unresolved_and_forced_style(self) -> None:
        issues = vn_quality.deterministic_issues([
            {"utteranceId": "u1", "resolvedZh": "甲", "semanticVi": "中文", "nativeVi": "", "source_confidence": "high"},
            {"utteranceId": "u1", "resolvedZh": "乙", "semanticVi": "Câu", "nativeVi": "Câu", "source_confidence": "high"},
            {"utteranceId": "u2", "resolvedZh": "丙", "semanticVi": "Câu", "nativeVi": "Câu này quá cháy luôn bro", "source_confidence": "unresolved"},
        ])
        codes = {issue["code"] for issue in issues}
        self.assertTrue({"MISSING_TRANSLATION", "DUPLICATE_UTTERANCE", "CJK_IN_VI", "UNRESOLVED_SOURCE", "FORCED_SLANG"} <= codes)

    @unittest.skipIf(vn_quality is None, "CP2 quality checks are not implemented")
    def test_one_item_repair_must_preserve_utterance_identity(self) -> None:
        original = {"utteranceId": "u1", "semanticVi": "Nghĩa", "nativeVi": "Bản cũ"}
        repaired = vn_quality.apply_one_repair(original, {"utteranceId": "u1", "nativeVi": "Bản mới"})
        self.assertEqual("u1", repaired["utteranceId"])
        with self.assertRaises(ValueError):
            vn_quality.apply_one_repair(original, {"utteranceId": "other", "nativeVi": "Bản mới"})

    @unittest.skipIf(vn_quality is None, "CP2 quality checks are not implemented")
    def test_llm_quality_boundary_receives_source_semantic_and_native_text(self) -> None:
        self.assertTrue(hasattr(vn_quality, "llm_issues"))
        seen = []
        records = [{"utteranceId": "u1", "resolvedZh": "新学期", "semanticVi": "Năm học mới", "nativeVi": "Năm học mới rồi", "source_confidence": "high"}]
        issues = vn_quality.llm_issues(records, lambda payload: seen.append(payload) or {"issues": []})
        self.assertEqual([], issues)
        self.assertEqual("新学期", seen[0]["utterances"][0]["resolvedZh"])
        self.assertEqual("Năm học mới", seen[0]["utterances"][0]["semanticVi"])
        self.assertEqual("Năm học mới rồi", seen[0]["utterances"][0]["nativeVi"])
        self.assertIn("STIFF_VI", seen[0]["criteria"])
        self.assertIn("SEMANTIC_DRIFT", seen[0]["criteria"])

    @unittest.skipIf(vi_harness is None, "CP2 harness is not implemented")
    def test_text_only_harness_writes_ab_table_without_localizing_visual_or_unresolved_source(self) -> None:
        calls = []
        def request(payload):
            calls.append(payload["mode"])
            if payload["mode"] == "qa":
                return {"issues": []}
            key = "semanticVi" if payload["mode"] == "semantic" else "nativeVi"
            return {"translations": [{"utteranceId": item["utteranceId"], key: "Câu tiếng Việt tự nhiên"} for item in payload["utterances"]]}
        records = [
            {"utteranceId": "u1", "start_ms": 0, "end_ms": 1000, "resolvedZh": "新学期", "source_confidence": "high", "kind": "dialogue"},
            {"utteranceId": "v1", "start_ms": 0, "end_ms": 1000, "resolvedZh": "标题", "source_confidence": "high", "kind": "visual_only"},
            {"utteranceId": "u2", "start_ms": 1000, "end_ms": 2000, "resolvedZh": "甲乙", "source_confidence": "unresolved", "kind": "dialogue"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            created = vi_harness.run_text_ab(records, Path(directory), scene_evidence={}, request=request)
            self.assertEqual({"multimodal_vi_ab_25s.json", "multimodal_vi_ab_25s.md"}, {path.name for path in created})
            artifact = json.loads((Path(directory) / "multimodal_vi_ab_25s.json").read_text(encoding="utf-8"))
            self.assertEqual(["u1"], [item["utteranceId"] for item in artifact["accepted"]])
            self.assertEqual(["u2"], [item["utteranceId"] for item in artifact["review_required"]])
            self.assertEqual(["semantic", "native", "qa"], calls)

    @unittest.skipIf(vi_harness is None, "CP2 harness is not implemented")
    def test_full_reconstruction_fixture_has_lossless_typed_ab_coverage(self) -> None:
        fixture = Path(__file__).parent / "fixtures" / "multimodal_25s" / "full_source_reconstruction.json"
        source = json.loads(fixture.read_text(encoding="utf-8"))
        records = vi_harness.records_from_source_reconstruction(source)
        calls = []

        def request(payload):
            calls.append(payload["mode"])
            if payload["mode"] == "qa":
                return {"issues": []}
            key = "semanticVi" if payload["mode"] == "semantic" else "nativeVi"
            return {"translations": [{"utteranceId": item["utteranceId"], key: "Câu tiếng Việt tự nhiên"} for item in payload["utterances"]]}

        with tempfile.TemporaryDirectory() as directory:
            vi_harness.run_text_ab(records, Path(directory), scene_evidence={}, request=request)
            root = Path(directory)
            artifact = json.loads((root / "multimodal_vi_ab_25s.json").read_text(encoding="utf-8"))
            markdown = (root / "multimodal_vi_ab_25s.md").read_text(encoding="utf-8")

        self.assertEqual(23, len(records))
        self.assertEqual({"sourceRecords": 23, "dialogueResolved": 19, "dialogueUnresolved": 1, "visualOnly": 3, "semanticTranslated": 19, "nativeTranslated": 19, "qaReviewed": 19}, artifact["coverage"])
        self.assertEqual([str(item["utteranceId"]) for item in records], [str(item["utteranceId"]) for item in artifact["records"]])
        self.assertEqual({"DIALOGUE": 19, "UNRESOLVED_DIALOGUE": 1, "VISUAL_ONLY": 3}, {kind: sum(item["type"] == kind for item in artifact["records"]) for kind in ("DIALOGUE", "UNRESOLVED_DIALOGUE", "VISUAL_ONLY")})
        self.assertEqual({"RESOLVED_DIALOGUE": 19, "UNRESOLVED_DIALOGUE": 1, "VISUAL_ONLY": 3}, {kind: sum(item["classification"] == kind for item in artifact["records"]) for kind in ("RESOLVED_DIALOGUE", "UNRESOLVED_DIALOGUE", "VISUAL_ONLY")})
        accepted = {str(item["utteranceId"]): item for item in artifact["accepted"]}
        self.assertEqual(19, len(accepted))
        self.assertTrue(all(item["semanticVi"] and item["nativeVi"] for item in accepted.values()))
        self.assertEqual(["22"], [str(item["utteranceId"]) for item in artifact["review_required"]])
        self.assertNotIn("22", accepted)
        self.assertTrue(all(str(item["utteranceId"]) not in accepted for item in artifact["records"] if item["classification"] == "VISUAL_ONLY"))
        self.assertIn("| # | Time | Positions | Type |", markdown)
        self.assertIn("sourceRecords: 23", markdown)
        self.assertEqual(["semantic", "native", "qa"], calls)

    @unittest.skipIf(vi_harness is None, "CP2 harness is not implemented")
    def test_observed_harness_persists_call_states_and_atomic_success_artifacts(self) -> None:
        self.assertTrue(hasattr(vi_harness, "run_observed_text_ab"))
        def scene(_payload):
            return {"summary": "", "tone": "", "register": "", "speakers": {}, "relationships": {}, "pronoun_policy": {}, "roleplay_terms": {}, "running_jokes": [], "glossary": {}}
        def request(payload):
            if payload["mode"] == "qa":
                return {"issues": []}
            key = "semanticVi" if payload["mode"] == "semantic" else "nativeVi"
            return {"translations": [{"utteranceId": item["utteranceId"], key: "Câu hợp lệ"} for item in payload["utterances"]]}
        records = [
            {"utteranceId": "u1", "start_ms": 0, "end_ms": 1, "resolvedZh": "新学期", "source_confidence": "high", "kind": "dialogue"},
            {"utteranceId": "v1", "start_ms": 0, "end_ms": 1, "resolvedZh": "标题", "source_confidence": "high", "kind": "visual_only"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            vi_harness.run_observed_text_ab(records, Path(directory), scene_request=scene, request=request, model="fake")
            state = json.loads((Path(directory) / "cp2_run_state.json").read_text(encoding="utf-8"))
            self.assertEqual("complete", state["status"])
            self.assertEqual(4, state["providerCallsAttempted"])
            self.assertEqual(4, state["providerCallsCompleted"])
            self.assertEqual(["scene", "semantic", "native", "qa"], [call["operation"] for call in state["calls"]])
            self.assertTrue((Path(directory) / "multimodal_vi_ab_25s.json").is_file())
            self.assertTrue((Path(directory) / "multimodal_vi_ab_25s.md").is_file())

    @unittest.skipIf(vi_harness is None, "CP2 harness is not implemented")
    def test_observed_harness_writes_safe_failure_artifact_without_swallowing_provider_error(self) -> None:
        self.assertTrue(hasattr(vi_harness, "run_observed_text_ab"))
        def scene(_payload):
            raise RuntimeError("provider unavailable")
        records = [
            {"utteranceId": "u1", "start_ms": 0, "end_ms": 1, "resolvedZh": "新学期", "source_confidence": "high", "kind": "dialogue"},
            {"utteranceId": "v1", "start_ms": 0, "end_ms": 1, "resolvedZh": "标题", "source_confidence": "high", "kind": "visual_only"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                vi_harness.run_observed_text_ab(records, Path(directory), scene_request=scene, request=lambda _payload: {}, model="fake")
            failure = json.loads((Path(directory) / "cp2_failure.json").read_text(encoding="utf-8"))
            self.assertEqual("scene", failure["phase"])
            self.assertEqual(1, failure["providerCallsAttempted"])
            self.assertEqual(0, failure["providerCallsCompleted"])
            self.assertEqual("RuntimeError", failure["errorType"])

    @unittest.skipIf(vi_harness is None, "CP2 harness is not implemented")
    def test_one_observed_repair_is_accounted_and_preserves_utterance_identity(self) -> None:
        self.assertTrue(hasattr(vi_harness, "repair_completed_run"))
        def scene(_payload):
            return {"summary": "", "tone": "", "register": "", "speakers": {}, "relationships": {}, "pronoun_policy": {}, "roleplay_terms": {}, "running_jokes": [], "glossary": {}}
        def request(payload):
            if payload["mode"] == "qa":
                return {"issues": [{"utteranceId": "u1", "code": "UNNATURAL_VI", "severity": "MEDIUM", "reason": "synthetic", "evidence": "synthetic"}]}
            if payload["mode"] == "repair":
                return {"translations": [{"utteranceId": "u1", "nativeVi": "Bản đã sửa"}]}
            key = "semanticVi" if payload["mode"] == "semantic" else "nativeVi"
            return {"translations": [{"utteranceId": item["utteranceId"], key: "Bản cũ"} for item in payload["utterances"]]}
        records = [
            {"utteranceId": "u1", "start_ms": 0, "end_ms": 1, "resolvedZh": "新学期", "source_confidence": "high", "kind": "dialogue"},
            {"utteranceId": "v1", "start_ms": 0, "end_ms": 1, "resolvedZh": "标题", "source_confidence": "high", "kind": "visual_only"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vi_harness.run_observed_text_ab(records, root, scene_request=scene, request=request, model="fake")
            vi_harness.repair_completed_run(root, request=request)
            artifact = json.loads((root / "multimodal_vi_ab_25s.json").read_text(encoding="utf-8"))
            state = json.loads((root / "cp2_run_state.json").read_text(encoding="utf-8"))
            self.assertEqual("u1", artifact["accepted"][0]["utteranceId"])
            self.assertEqual("Bản đã sửa", artifact["accepted"][0]["nativeVi"])
            self.assertEqual(["scene", "semantic", "native", "qa", "repair"], [call["operation"] for call in state["calls"]])
            self.assertEqual("complete", state["status"])
            self.assertIn("标题", (root / "multimodal_vi_ab_25s.md").read_text(encoding="utf-8"))

    @unittest.skipIf(vn_localizer is None, "CP2 localizer is not implemented")
    def test_localization_passes_scene_plan_neighbors_and_separate_profile(self) -> None:
        seen = {}
        records = [
            {"utteranceId": "u1", "resolvedZh": "甲", "source_confidence": "high", "kind": "dialogue", "channel_profile": "channel-a"},
            {"utteranceId": "u2", "resolvedZh": "乙", "source_confidence": "high", "kind": "dialogue", "channel_profile": "channel-a"},
            {"utteranceId": "u3", "resolvedZh": "丙", "source_confidence": "high", "kind": "dialogue", "channel_profile": "channel-a"},
        ]
        scene = {"register": "mock-palace", "roleplayTerms": {"老奴": "lão nô"}}

        def request(payload):
            seen[payload["mode"]] = payload
            key = "semanticVi" if payload["mode"] == "semantic" else "nativeVi"
            return {"translations": [{"utteranceId": item["utteranceId"], key: "Câu tự nhiên"} for item in payload["utterances"]]}

        vn_localizer.localize(records, scene_memory=scene, request=request, profile="vn_native_light_comedy")
        self.assertEqual(scene, seen["native"]["scene_memory"])
        self.assertEqual("vn_native_light_comedy", seen["native"]["localization_profile"])
        native_by_id = {item["utteranceId"]: item for item in seen["native"]["utterances"]}
        self.assertEqual(["u1", "u3"], [item["utteranceId"] for item in native_by_id["u2"]["neighbors"]])
        semantic_by_id = {item["utteranceId"]: item for item in seen["semantic"]["utterances"]}
        self.assertEqual(["u1", "u3"], [item["utteranceId"] for item in semantic_by_id["u2"]["neighbors"]])

    @unittest.skipIf(vi_harness is None, "CP2 harness is not implemented")
    def test_stage_models_are_explicit_and_configurable_without_downgrade(self) -> None:
        with patch.dict(os.environ, {"DUBVI_LOCALIZATION_MODEL": "loc-model", "DUBVI_QA_MODEL": "qa-model"}, clear=False):
            self.assertEqual("loc-model", vi_harness.model_for_operation("native"))
            self.assertEqual("loc-model", vi_harness.model_for_operation("repair"))
            self.assertEqual("qa-model", vi_harness.model_for_operation("qa"))
        self.assertEqual("gemini-2.5-flash", vi_harness.DEFAULT_LOCALIZATION_MODEL)
        self.assertEqual("gemini-2.5-flash", vi_harness.DEFAULT_QA_MODEL)

    @unittest.skipIf(vn_quality is None, "CP2 quality checks are not implemented")
    def test_quality_contract_preserves_canonical_and_unknown_codes_with_metadata(self) -> None:
        records = [{"utteranceId": "u1", "resolvedZh": "老钱", "semanticVi": "Nghĩa", "nativeVi": "Lời thoại", "source_confidence": "high"}]

        def request(_payload):
            return {"issues": [
                {"utteranceId": "u1", "code": "TERM_MISTRANSLATION", "severity": "HIGH", "reason": "Sai sắc thái", "evidence": "老钱"},
                {"utteranceId": "u1", "code": "UNTRUSTED_CODE", "severity": "LOW", "reason": "Không chuẩn", "evidence": "x"},
            ]}

        issues = vn_quality.llm_issues(records, request, scene_memory={})
        self.assertEqual(["TERM_MISTRANSLATION", "UNKNOWN_QA_CODE"], [item["code"] for item in issues])
        self.assertEqual("UNTRUSTED_CODE", issues[1]["originalCode"])
        self.assertTrue({"severity", "reason", "evidence"} <= set(issues[0]))
        self.assertTrue({
            "SEMANTIC_DRIFT", "TERM_MISTRANSLATION", "TRANSLATIONESE", "UNNATURAL_VI",
            "PRONOUN_INCONSISTENCY", "REGISTER_MISMATCH", "ROLEPLAY_INCONSISTENCY",
            "FORCED_SLANG", "FORCED_HUMOR", "CONTEXT_LOSS", "SOURCE_UNCERTAIN",
        } <= set(vn_quality.CANONICAL_QA_CODES))

    @unittest.skipIf(vn_quality is None, "CP2 quality checks are not implemented")
    def test_quality_receives_scene_memory_and_neighbor_context(self) -> None:
        seen = []
        records = [{
            "utteranceId": "u1", "resolvedZh": "甲", "semanticVi": "Nghĩa", "nativeVi": "Lời thoại",
            "source_confidence": "high", "neighbors": [{"utteranceId": "u2", "resolvedZh": "乙"}],
            "source_flags": ["PARTIAL_CUE"],
        }]
        vn_quality.llm_issues(records, lambda payload: seen.append(payload) or {"issues": []}, scene_memory={"register": "roleplay"})
        self.assertEqual({"register": "roleplay"}, seen[0]["scene_memory"])
        self.assertEqual("u2", seen[0]["utterances"][0]["neighbors"][0]["utteranceId"])
        self.assertEqual(["PARTIAL_CUE"], seen[0]["utterances"][0]["source_flags"])

    @unittest.skipIf(vn_quality is None, "CP2 quality checks are not implemented")
    def test_suspicious_resolved_source_is_not_reported_clean(self) -> None:
        issues = vn_quality.deterministic_issues([{
            "utteranceId": "u20", "resolvedZh": "你呀老奴一声就行", "semanticVi": "Nghĩa", "nativeVi": "Lời thoại",
            "source_confidence": "high", "source_flags": ["PARTIAL_CUE"],
        }])
        self.assertIn("SOURCE_UNCERTAIN", {item["code"] for item in issues})

    @unittest.skipIf(vi_harness is None, "CP2 harness is not implemented")
    def test_v2_artifact_stem_keeps_previous_artifact_name_available(self) -> None:
        records = [{"utteranceId": "u1", "resolvedZh": "甲", "source_confidence": "high", "kind": "dialogue"}]

        def request(payload):
            if payload["mode"] == "qa":
                return {"issues": []}
            key = "semanticVi" if payload["mode"] == "semantic" else "nativeVi"
            return {"translations": [{"utteranceId": "u1", key: "Câu hợp lệ"}]}

        with tempfile.TemporaryDirectory() as directory:
            created = vi_harness.run_text_ab(records, Path(directory), scene_evidence={}, request=request, artifact_stem="multimodal_vi_ab_25s_v2")
            self.assertEqual({"multimodal_vi_ab_25s_v2.json", "multimodal_vi_ab_25s_v2.md"}, {path.name for path in created})

    @unittest.skipIf(vn_quality is None, "CP2 quality checks are not implemented")
    def test_qa_request_declares_complete_provider_schema(self) -> None:
        seen = []
        records = [{"utteranceId": "u1", "resolvedZh": "甲", "semanticVi": "Nghĩa", "nativeVi": "Lời thoại", "source_confidence": "high"}]
        vn_quality.llm_issues(records, lambda payload: seen.append(payload) or {"issues": []}, scene_memory={})
        self.assertEqual("application/json", seen[0]["generationConfig"]["responseMimeType"])
        issue_schema = seen[0]["generationConfig"]["responseSchema"]["properties"]["issues"]["items"]
        self.assertEqual(["utteranceId", "code", "severity", "reason", "evidence"], issue_schema["required"])
        self.assertEqual(["LOW", "MEDIUM", "HIGH"], issue_schema["properties"]["severity"]["enum"])

    @unittest.skipIf(vi_harness is None, "CP2 harness is not implemented")
    def test_observed_state_persists_input_contract_and_validated_stage_payloads(self) -> None:
        def scene(_payload):
            return {"summary": "scene", "tone": "comic", "register": "spoken", "speakers": {}, "relationships": {}, "pronoun_policy": {}, "roleplay_terms": {}, "running_jokes": [], "glossary": {}}

        def request(payload):
            if payload["mode"] == "qa":
                return {"issues": []}
            key = "semanticVi" if payload["mode"] == "semantic" else "nativeVi"
            return {"translations": [{"utteranceId": item["utteranceId"], key: "Câu hợp lệ"} for item in payload["utterances"]]}

        records = [{"utteranceId": "u1", "start_ms": 0, "end_ms": 1, "resolvedZh": "甲", "source_confidence": "high", "kind": "dialogue"}]
        with tempfile.TemporaryDirectory() as directory:
            vi_harness.run_observed_text_ab(records, Path(directory), scene_request=scene, request=request, model="fake")
            state = json.loads((Path(directory) / "cp2_run_state.json").read_text(encoding="utf-8"))
        self.assertEqual(2, state["version"])
        self.assertTrue(state["input"]["sourceFingerprint"])
        self.assertEqual("fake", state["input"]["localizationModel"])
        self.assertEqual("fake", state["input"]["qaModel"])
        for stage in ("scene", "semantic", "native", "qa"):
            self.assertEqual("completed", state["stages"][stage]["status"])
            self.assertIsNotNone(state["stages"][stage]["payload"])

    @unittest.skipIf(vi_harness is None, "CP2 harness is not implemented")
    def test_observed_resume_reuses_completed_stages_after_qa_failure(self) -> None:
        def scene(_payload):
            return {"summary": "scene", "tone": "comic", "register": "spoken", "speakers": {}, "relationships": {}, "pronoun_policy": {}, "roleplay_terms": {}, "running_jokes": [], "glossary": {}}

        first_calls = []
        def first_request(payload):
            first_calls.append(payload["mode"])
            if payload["mode"] == "qa":
                raise RuntimeError("synthetic qa failure")
            key = "semanticVi" if payload["mode"] == "semantic" else "nativeVi"
            return {"translations": [{"utteranceId": item["utteranceId"], key: "Câu hợp lệ"} for item in payload["utterances"]]}

        second_calls = []
        def second_request(payload):
            second_calls.append(payload["mode"])
            self.assertEqual("qa", payload["mode"])
            return {"issues": []}

        records = [{"utteranceId": "u1", "start_ms": 0, "end_ms": 1, "resolvedZh": "甲", "source_confidence": "high", "kind": "dialogue"}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(RuntimeError):
                vi_harness.run_observed_text_ab(records, root, scene_request=scene, request=first_request, model="fake")
            failed = json.loads((root / "cp2_run_state.json").read_text(encoding="utf-8"))
            self.assertEqual(["semantic", "native", "qa"], first_calls)
            self.assertEqual("completed", failed["stages"]["scene"]["status"])
            self.assertEqual("completed", failed["stages"]["semantic"]["status"])
            self.assertEqual("completed", failed["stages"]["native"]["status"])
            vi_harness.run_observed_text_ab(records, root, scene_request=lambda _payload: self.fail("scene must be reused"), request=second_request, model="fake")
            resumed = json.loads((root / "cp2_run_state.json").read_text(encoding="utf-8"))
        self.assertEqual(["qa"], second_calls)
        self.assertEqual("complete", resumed["status"])
        self.assertEqual(["scene", "semantic", "native", "qa", "qa"], [call["operation"] for call in resumed["calls"]])

    @unittest.skipIf(vi_harness is None, "CP2 harness is not implemented")
    def test_metadata_only_failed_state_is_superseded_before_fresh_run(self) -> None:
        def request(payload):
            if payload["mode"] == "qa":
                return {"issues": []}
            key = "semanticVi" if payload["mode"] == "semantic" else "nativeVi"
            return {"translations": [{"utteranceId": item["utteranceId"], key: "Câu hợp lệ"} for item in payload["utterances"]]}

        records = [{"utteranceId": "u1", "start_ms": 0, "end_ms": 1, "resolvedZh": "甲", "source_confidence": "high", "kind": "dialogue"}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.mkdir(exist_ok=True)
            (root / "cp2_run_state.json").write_text(json.dumps({"status": "failed", "phase": "qa", "calls": []}), encoding="utf-8")
            vi_harness.run_observed_text_ab(records, root, scene_request=lambda _payload: {"summary": "", "tone": "", "register": "", "speakers": {}, "relationships": {}, "pronoun_policy": {}, "roleplay_terms": {}, "running_jokes": [], "glossary": {}}, request=request, model="fake")
            self.assertTrue((root / "cp2_run_state_superseded.json").is_file())
            self.assertEqual("failed", json.loads((root / "cp2_run_state_superseded.json").read_text(encoding="utf-8"))["status"])

    @unittest.skipIf(vi_harness is None or llm_transport is None, "CP2 harness is not implemented")
    def test_qa_structured_failure_gets_one_qa_only_retry(self) -> None:
        qa_calls = []
        def scene(_payload):
            return {"summary": "", "tone": "", "register": "", "speakers": {}, "relationships": {}, "pronoun_policy": {}, "roleplay_terms": {}, "running_jokes": [], "glossary": {}}

        def request(payload):
            qa_calls.append(payload["mode"])
            if payload["mode"] == "qa" and qa_calls.count("qa") == 1:
                raise llm_transport.StructuredResponseError("synthetic malformed JSON")
            if payload["mode"] == "qa":
                return {"issues": []}
            key = "semanticVi" if payload["mode"] == "semantic" else "nativeVi"
            return {"translations": [{"utteranceId": item["utteranceId"], key: "Câu hợp lệ"} for item in payload["utterances"]]}

        records = [{"utteranceId": "u1", "start_ms": 0, "end_ms": 1, "resolvedZh": "甲", "source_confidence": "high", "kind": "dialogue"}]
        with tempfile.TemporaryDirectory() as directory:
            vi_harness.run_observed_text_ab(records, Path(directory), scene_request=scene, request=request, model="fake")
            state = json.loads((Path(directory) / "cp2_run_state.json").read_text(encoding="utf-8"))
        self.assertEqual(["semantic", "native", "qa", "qa"], qa_calls)
        self.assertEqual("complete", state["status"])
        self.assertEqual(["scene", "semantic", "native", "qa", "qa"], [call["operation"] for call in state["calls"]])

    @unittest.skipIf(vi_harness is None, "CP2 harness is not implemented")
    def test_corrupt_completed_stage_payload_is_invalidated_before_resume(self) -> None:
        def scene(_payload):
            return {"summary": "", "tone": "", "register": "", "speakers": {}, "relationships": {}, "pronoun_policy": {}, "roleplay_terms": {}, "running_jokes": [], "glossary": {}}

        def request(payload):
            if payload["mode"] == "qa":
                return {"issues": []}
            key = "semanticVi" if payload["mode"] == "semantic" else "nativeVi"
            return {"translations": [{"utteranceId": item["utteranceId"], key: "Câu hợp lệ"} for item in payload["utterances"]]}

        records = [{"utteranceId": "u1", "start_ms": 0, "end_ms": 1, "resolvedZh": "甲", "source_confidence": "high", "kind": "dialogue"}]
        second_calls = []
        scene_calls = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vi_harness.run_observed_text_ab(records, root, scene_request=scene, request=request, model="fake")
            state = json.loads((root / "cp2_run_state.json").read_text(encoding="utf-8"))
            state["stages"]["semantic"]["payload"] = [{"utteranceId": "wrong", "semanticVi": "Câu hợp lệ"}]
            (root / "cp2_run_state.json").write_text(json.dumps(state), encoding="utf-8")
            def second_request(payload):
                second_calls.append(payload["mode"])
                return request(payload)
            vi_harness.run_observed_text_ab(records, root, scene_request=lambda payload: scene_calls.append(payload["mode"]) or scene(payload), request=second_request, model="fake")
        self.assertEqual(["scene"], scene_calls)
        self.assertEqual(["semantic", "native", "qa"], second_calls)

    @unittest.skipIf(vi_harness is None, "CP2 harness is not implemented")
    def test_state_write_keeps_previous_document_when_atomic_replace_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text("{\"old\": true}\n", encoding="utf-8")
            with patch.object(vi_harness.os, "replace", side_effect=OSError("synthetic crash")):
                with self.assertRaises(OSError):
                    vi_harness._write(path, {"new": True})
            self.assertEqual("{\"old\": true}\n", path.read_text(encoding="utf-8"))

    @unittest.skipIf(vi_harness is None, "CP2 harness is not implemented")
    def test_v2_markdown_contains_positions_and_versioned_comparison_columns(self) -> None:
        records = [{"utteranceId": "u1", "start_ms": 0, "end_ms": 1, "positions": [4, 5], "resolvedZh": "甲", "oldVi": "Cũ", "source_confidence": "high", "kind": "dialogue"}]
        def request(payload):
            if payload["mode"] == "qa":
                return {"issues": []}
            key = "semanticVi" if payload["mode"] == "semantic" else "nativeVi"
            return {"translations": [{"utteranceId": "u1", key: "Câu hợp lệ"}]}
        with tempfile.TemporaryDirectory() as directory:
            vi_harness.run_text_ab(records, Path(directory), scene_evidence={}, request=request, artifact_stem="multimodal_vi_ab_25s_v2")
            markdown = (Path(directory) / "multimodal_vi_ab_25s_v2.md").read_text(encoding="utf-8")
        self.assertIn("| # | Time | Positions | Type | Resolved Chinese | Old DUBVI VI | CP2-v1 Native VI | CP2-v2 Semantic VI | CP2-v2 Native VI | QA |", markdown)
        self.assertIn("| 1 | 0–1 | [4,5] |", markdown)

    @unittest.skipIf(scene_memory is None, "CP2 scene memory is not implemented")
    def test_source_grounded_terminology_spans_visual_and_dialogue(self) -> None:
        self.assertTrue(hasattr(scene_memory, "build_terminology_evidence"))
        evidence = scene_memory.build_terminology_evidence([
            {"utteranceId": "v1", "resolvedZh": "当你穿进老钱班", "kind": "visual_only"},
            {"utteranceId": "u1", "resolvedZh": "最夸张的老钱来了", "kind": "dialogue"},
        ])
        item = next(value for value in evidence if value["sourceTerm"] == "老钱")
        self.assertEqual(
            [("VISUAL_ONLY", "当你穿进老钱班"), ("DIALOGUE", "最夸张的老钱来了")],
            [(occurrence["type"], occurrence["text"]) for occurrence in item["occurrences"]],
        )

    @unittest.skipIf(vn_localizer is None, "CP2 localizer is not implemented")
    def test_visual_only_terminology_context_never_becomes_dialogue(self) -> None:
        seen = []

        def request(payload):
            seen.append(payload)
            key = "semanticVi" if payload["mode"] == "semantic" else "nativeVi"
            return {"translations": [{"utteranceId": "u1", key: "Câu tự nhiên"}]}

        vn_localizer.localize([
            {"utteranceId": "v1", "resolvedZh": "当你穿进老钱班", "source_confidence": "high", "kind": "visual_only"},
            {"utteranceId": "u1", "resolvedZh": "最夸张的老钱来了", "source_confidence": "high", "kind": "dialogue"},
        ], scene_memory={"terminologyEvidence": [{"sourceTerm": "老钱", "occurrences": []}]}, request=request, profile="vn_native")
        self.assertEqual(["u1"], [item["utteranceId"] for item in seen[0]["utterances"]])
        self.assertNotIn("v1", json.dumps(seen, ensure_ascii=False))

    @unittest.skipIf(vi_harness is None, "CP2 harness is not implemented")
    def test_terminology_evidence_reaches_semantic_native_and_qa(self) -> None:
        seen = []

        def request(payload):
            seen.append(payload)
            if payload["mode"] == "qa":
                return {"issues": []}
            key = "semanticVi" if payload["mode"] == "semantic" else "nativeVi"
            return {"translations": [{"utteranceId": "u1", key: "Câu tự nhiên"}]}

        records = [
            {"utteranceId": "v1", "start_ms": 0, "end_ms": 1, "resolvedZh": "当你穿进老钱班", "source_confidence": "high", "kind": "visual_only"},
            {"utteranceId": "u1", "start_ms": 1, "end_ms": 2, "resolvedZh": "最夸张的老钱来了", "source_confidence": "high", "kind": "dialogue"},
        ]
        vi_harness.run_text_ab(records, Path(tempfile.mkdtemp()), scene_evidence={}, request=request)
        self.assertEqual(["semantic", "native", "qa"], [payload["mode"] for payload in seen])
        for payload in seen:
            self.assertTrue(any(item["sourceTerm"] == "老钱" for item in payload.get("terminology_evidence", [])))
            self.assertNotIn("v1", {item["utteranceId"] for item in payload.get("utterances", [])})

    @unittest.skipIf(vn_quality is None, "CP2 quality checks are not implemented")
    def test_qa_can_flag_fluent_line_as_term_mistranslation_with_context(self) -> None:
        seen = []
        records = [{"utteranceId": "u1", "resolvedZh": "最夸张的老钱来了", "semanticVi": "Khái niệm tài sản lâu đời đến rồi", "nativeVi": "Lão Tiền khoa trương nhất đến rồi.", "source_confidence": "high"}]
        memory = {"terminologyEvidence": [{"sourceTerm": "老钱", "occurrences": [{"type": "VISUAL_ONLY", "text": "当你穿进老钱班"}, {"type": "DIALOGUE", "text": "最夸张的老钱来了"}]}]}

        def request(payload):
            seen.append(payload)
            return {"issues": [{"utteranceId": "u1", "code": "TERM_MISTRANSLATION", "severity": "HIGH", "reason": "contextual term is rendered as a name", "evidence": "老钱"}]}

        issues = vn_quality.llm_issues(records, request, scene_memory=memory)
        self.assertEqual(["TERM_MISTRANSLATION"], [issue["code"] for issue in issues])
        self.assertTrue(seen[0]["terminology_evidence"])
        self.assertTrue(seen[0]["term_check_guidance"])

    @unittest.skipIf(vn_quality is None, "CP2 quality checks are not implemented")
    def test_deterministic_term_guard_flags_unsupported_name_rendering(self) -> None:
        evidence = [{"sourceTerm": "老钱", "occurrences": [{"type": "VISUAL_ONLY", "text": "当你穿进老钱班"}, {"type": "DIALOGUE", "text": "最夸张的老钱来了"}]}]
        issues = vn_quality.deterministic_issues([{
            "utteranceId": "u1", "resolvedZh": "最夸张的老钱来了", "semanticVi": "Lão Tiền khoa trương nhất đến rồi", "nativeVi": "Lão Tiền khoa trương nhất đến rồi.", "source_confidence": "high",
        }], scene_memory={"terminologyEvidence": evidence})
        self.assertIn("TERM_MISTRANSLATION", {issue["code"] for issue in issues})
        clean = vn_quality.deterministic_issues([{
            "utteranceId": "u1", "resolvedZh": "最夸张的老钱来了", "semanticVi": "Giới old-money ấn tượng nhất đến rồi", "nativeVi": "Giới old-money ấn tượng nhất đến rồi.", "source_confidence": "high",
        }], scene_memory={"terminologyEvidence": evidence})
        self.assertNotIn("TERM_MISTRANSLATION", {issue["code"] for issue in clean})

    @unittest.skipIf(vi_harness is None, "CP2 harness is not implemented")
    def test_targeted_rejudge_repair_receives_context_and_writes_v3(self) -> None:
        self.assertTrue(hasattr(vi_harness, "rejudge_completed_run"))
        def scene(_payload):
            return {"summary": "roleplay", "tone": "comic", "register": "spoken", "speakers": {}, "relationships": {}, "pronoun_policy": {}, "roleplay_terms": {}, "running_jokes": [], "glossary": {}}

        records = [
            {"utteranceId": "v1", "start_ms": 0, "end_ms": 1, "resolvedZh": "当你穿进老钱班", "source_confidence": "high", "kind": "visual_only", "positions": []},
            {"utteranceId": "u1", "start_ms": 1, "end_ms": 2, "resolvedZh": "最夸张的老钱来了", "source_confidence": "high", "kind": "dialogue", "positions": [1, 2]},
            {"utteranceId": "u2", "start_ms": 2, "end_ms": 3, "resolvedZh": "你呀老奴一声就行", "source_confidence": "high", "source_flags": ["PARTIAL_CUE"], "kind": "dialogue", "positions": [3]},
        ]
        repair_payload = {}

        def initial_request(payload):
            if payload["mode"] == "qa":
                return {"issues": []}
            key = "semanticVi" if payload["mode"] == "semantic" else "nativeVi"
            return {"translations": [{"utteranceId": item["utteranceId"], key: "Bản hiện tại"} for item in payload["utterances"]]}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vi_harness.run_observed_text_ab(records, root, scene_request=scene, request=initial_request, model="fake", artifact_stem="multimodal_vi_ab_25s_v2")

            def rejudge_request(payload):
                if payload["mode"] == "qa":
                    self.assertTrue(payload["terminology_evidence"])
                    return {"issues": [{"utteranceId": "u1", "code": "TERM_MISTRANSLATION", "severity": "HIGH", "reason": "wrong term sense", "evidence": "老钱"}, {"utteranceId": "u2", "code": "SOURCE_UNCERTAIN", "severity": "HIGH", "reason": "source uncertain", "evidence": "你呀老奴一声就行"}]}
                repair_payload.update(payload)
                return {"translations": [{"utteranceId": "u1", "nativeVi": "Lớp old-money đặc biệt nhất tới rồi."}]}

            vi_harness.rejudge_completed_run(root, request=rejudge_request, artifact_stem="multimodal_vi_ab_25s_v2", output_artifact_stem="multimodal_vi_ab_25s_v3")
            artifact = json.loads((root / "multimodal_vi_ab_25s_v3.json").read_text(encoding="utf-8"))
            self.assertEqual("u1", repair_payload["utterances"][0]["utteranceId"])
            self.assertTrue(repair_payload["scene_memory"]["terminologyEvidence"])
            self.assertTrue(repair_payload["utterances"][0]["neighbors"])
            repaired = next(item for item in artifact["accepted"] if item["utteranceId"] == "u1")
            self.assertEqual({"utteranceId": "u1", "resolvedZh": "最夸张的老钱来了", "start_ms": 1, "end_ms": 2, "positions": [1, 2]}, {key: repaired[key] for key in ("utteranceId", "resolvedZh", "start_ms", "end_ms", "positions")})
            self.assertEqual("Bản hiện tại", next(item for item in artifact["accepted"] if item["utteranceId"] == "u2")["nativeVi"])
            self.assertIn("| # | Time | Type | Resolved Chinese | V1 Native VI | V2 Native VI | V3 Native VI | QA |", (root / "multimodal_vi_ab_25s_v3.md").read_text(encoding="utf-8"))

    @unittest.skipIf(vi_harness is None, "CP2 harness is not implemented")
    def test_source_uncertain_row_is_not_selected_for_targeted_repair(self) -> None:
        self.assertTrue(hasattr(vi_harness, "rejudge_completed_run"))
        issues = [
            {"utteranceId": "u1", "code": "TERM_MISTRANSLATION", "severity": "HIGH"},
            {"utteranceId": "u2", "code": "SOURCE_UNCERTAIN", "severity": "HIGH"},
        ]
        self.assertEqual(["TERM_MISTRANSLATION"], [issue["code"] for issue in vi_harness._serious_repairable_issues(issues)])

    @unittest.skipIf(vi_harness is None, "CP2 harness is not implemented")
    def test_repair_recomputes_contextual_qa_before_selecting_targets(self) -> None:
        records = [
            {"utteranceId": "v1", "start_ms": 0, "end_ms": 1, "resolvedZh": "当你穿进老钱班", "source_confidence": "high", "kind": "visual_only"},
            {"utteranceId": "u1", "start_ms": 1, "end_ms": 2, "resolvedZh": "最夸张的老钱来了", "source_confidence": "high", "kind": "dialogue"},
        ]
        repaired = []

        def request(payload):
            if payload["mode"] == "qa":
                return {"issues": []}
            if payload["mode"] == "repair":
                repaired.append(payload)
                return {"translations": [{"utteranceId": "u1", "nativeVi": "Giới old-money ấn tượng nhất tới rồi."}]}
            key = "semanticVi" if payload["mode"] == "semantic" else "nativeVi"
            return {"translations": [{"utteranceId": "u1", key: "Lão Tiền khoa trương nhất đến rồi."}]}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vi_harness.run_observed_text_ab(records, root, scene_request=lambda _payload: {"summary": "", "tone": "", "register": "", "speakers": {}, "relationships": {}, "pronoun_policy": {}, "roleplay_terms": {}, "running_jokes": [], "glossary": {}}, request=request, model="fake", artifact_stem="multimodal_vi_ab_25s_v3")
            artifact_path = root / "multimodal_vi_ab_25s_v3.json"
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            artifact["qa"] = []
            artifact_path.write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")
            vi_harness.repair_completed_run(root, request=request, artifact_stem="multimodal_vi_ab_25s_v3")
            repaired_artifact = json.loads((root / "multimodal_vi_ab_25s_v3.json").read_text(encoding="utf-8"))
        self.assertEqual(["u1"], [item["utteranceId"] for item in repaired[0]["utterances"]])
        self.assertIn("TERM_MISTRANSLATION", {issue["code"] for issue in repaired_artifact["qa_history"]})

    @unittest.skipIf(vi_harness is None, "CP2 harness is not implemented")
    def test_repairing_v3_artifact_preserves_v3_review_table(self) -> None:
        records = [
            {"utteranceId": "v1", "start_ms": 0, "end_ms": 1, "resolvedZh": "当你穿进老钱班", "source_confidence": "high", "kind": "visual_only"},
            {"utteranceId": "u1", "start_ms": 1, "end_ms": 2, "resolvedZh": "最夸张的老钱来了", "source_confidence": "high", "kind": "dialogue"},
        ]

        def request(payload):
            if payload["mode"] == "qa":
                return {"issues": [{"utteranceId": "u1", "code": "TERM_MISTRANSLATION", "severity": "HIGH", "reason": "wrong sense", "evidence": "老钱"}]}
            if payload["mode"] == "repair":
                return {"translations": [{"utteranceId": "u1", "nativeVi": "Giới old-money ấn tượng nhất đến rồi."}]}
            key = "semanticVi" if payload["mode"] == "semantic" else "nativeVi"
            return {"translations": [{"utteranceId": "u1", key: "Lão Tiền khoa trương nhất đến rồi."}]}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vi_harness.run_observed_text_ab(records, root, scene_request=lambda _payload: {"summary": "", "tone": "", "register": "", "speakers": {}, "relationships": {}, "pronoun_policy": {}, "roleplay_terms": {}, "running_jokes": [], "glossary": {}}, request=request, model="fake", artifact_stem="multimodal_vi_ab_25s_v3")
            vi_harness.repair_completed_run(root, request=request, artifact_stem="multimodal_vi_ab_25s_v3")
            markdown = (root / "multimodal_vi_ab_25s_v3.md").read_text(encoding="utf-8")
        self.assertIn("| # | Time | Type | Resolved Chinese | V1 Native VI | V2 Native VI | V3 Native VI | QA |", markdown)

    @unittest.skipIf(vi_harness is None, "CP2 harness is not implemented")
    def test_rejudge_persists_deterministic_term_issue_in_history(self) -> None:
        records = [
            {"utteranceId": "v1", "start_ms": 0, "end_ms": 1, "resolvedZh": "当你穿进老钱班", "source_confidence": "high", "kind": "visual_only"},
            {"utteranceId": "u1", "start_ms": 1, "end_ms": 2, "resolvedZh": "最夸张的老钱来了", "source_confidence": "high", "kind": "dialogue"},
        ]

        def initial_request(payload):
            if payload["mode"] == "qa":
                return {"issues": []}
            key = "semanticVi" if payload["mode"] == "semantic" else "nativeVi"
            return {"translations": [{"utteranceId": "u1", key: "Lão Tiền khoa trương nhất đến rồi."}]}

        def rejudge_request(payload):
            if payload["mode"] == "qa":
                return {"issues": []}
            return {"translations": [{"utteranceId": "u1", "nativeVi": "Giới old-money ấn tượng nhất đến rồi."}]}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vi_harness.run_observed_text_ab(records, root, scene_request=lambda _payload: {"summary": "", "tone": "", "register": "", "speakers": {}, "relationships": {}, "pronoun_policy": {}, "roleplay_terms": {}, "running_jokes": [], "glossary": {}}, request=initial_request, model="fake", artifact_stem="multimodal_vi_ab_25s_v2")
            artifact_path = root / "multimodal_vi_ab_25s_v2.json"
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            artifact["qa"] = []
            artifact_path.write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")
            vi_harness.rejudge_completed_run(root, request=rejudge_request, artifact_stem="multimodal_vi_ab_25s_v2", output_artifact_stem="multimodal_vi_ab_25s_v3")
            artifact = json.loads((root / "multimodal_vi_ab_25s_v3.json").read_text(encoding="utf-8"))
        self.assertIn("TERM_MISTRANSLATION", {issue["code"] for issue in artifact["qa_history"]})


if __name__ == "__main__":
    unittest.main()
