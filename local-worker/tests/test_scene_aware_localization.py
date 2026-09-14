from __future__ import annotations

import importlib
import json
import tempfile
import unittest
from pathlib import Path


def _optional_module(name: str):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        return None


scene_memory = _optional_module("scene_memory")
vn_localizer = _optional_module("vn_localizer")
vn_quality = _optional_module("vn_quality")
vi_harness = _optional_module("multimodal_vi_harness")


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
                return {"issues": [{"utteranceId": "u1", "code": "UNNATURAL_VI"}]}
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


if __name__ == "__main__":
    unittest.main()
