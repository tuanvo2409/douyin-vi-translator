from __future__ import annotations

import importlib
import copy
import json
import tempfile
import unittest
from pathlib import Path


def _optional_module(name: str):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        return None


llm_transport = _optional_module("llm_transport")
av_analyzer = _optional_module("av_analyzer")
source_reconciler = _optional_module("source_reconciler")
utterance_builder = _optional_module("utterance_builder")
source_harness = _optional_module("multimodal_source_harness")
vi_harness = _optional_module("multimodal_vi_harness")

FIXTURE = Path(__file__).parent / "fixtures" / "multimodal_25s" / "source_fixture.json"
PERSISTED_FIXTURE = Path(__file__).parent / "fixtures" / "multimodal_25s" / "persisted_runtime.json"


def _canonical_av(*, utterances=None, visual_only=None):
    return {
        "windowStartMs": 0,
        "windowEndMs": 25_300,
        "scene": {"summary": "classroom comedy", "tone": "comic", "register": "colloquial"},
        "utterances": list(utterances or []),
        "visualOnly": list(visual_only or []),
    }


def _av_utterance(start, end, spoken, subtitle=None, *, confidence="HIGH"):
    return {
        "startMs": start,
        "endMs": end,
        "spokenZh": spoken,
        "subtitleZh": spoken if subtitle is None else subtitle,
        "speakerRef": "speaker-1",
        "intent": "statement",
        "emotion": "neutral",
        "confidence": confidence,
    }


def _av_visual(start, end, text, *, confidence="HIGH"):
    return {"startMs": start, "endMs": end, "textZh": text, "confidence": confidence}


class MultimodalSourceContractTests(unittest.TestCase):
    def test_cp1_modules_exist(self) -> None:
        self.assertIsNotNone(llm_transport)
        self.assertIsNotNone(av_analyzer)
        self.assertIsNotNone(source_reconciler)
        self.assertIsNotNone(utterance_builder)
        self.assertIsNotNone(source_harness)

    @unittest.skipIf(llm_transport is None, "CP1 transport is not implemented")
    def test_structured_transport_retries_only_transient_errors_and_fails_closed(self) -> None:
        calls = []

        def request(_payload):
            calls.append(_payload)
            if len(calls) == 1:
                raise llm_transport.ProviderError("HTTP_429", transient=True)
            return _canonical_av(utterances=[_av_utterance(0, 1, "新学期")])

        transport = llm_transport.StructuredGeminiTransport(request=request, max_attempts=2)
        result = transport.call({"contents": []}, required_keys={"windowStartMs", "windowEndMs", "scene", "utterances", "visualOnly"})
        self.assertEqual("新学期", result["utterances"][0]["spokenZh"])
        self.assertEqual(2, len(calls))
        with self.assertRaises(llm_transport.StructuredResponseError):
            llm_transport.validate_structured_document({"utterances": []}, {"utterances", "visual_only"})

    @unittest.skipIf(source_reconciler is None or utterance_builder is None, "CP1 reconciliation is not implemented")
    def test_fixture_repairs_av_supported_dialogue_without_promoting_visual_or_unresolved_text(self) -> None:
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        result = source_reconciler.reconcile_source(fixture["cues"], fixture["av_analysis"], fixture["unresolved"])
        by_text = {item["resolvedZh"]: item for item in result["dialogue"] if item["resolvedZh"]}
        self.assertIn("新学期", by_text)
        self.assertIn("我攒了重金买了新书包，这书包可贵", by_text)
        self.assertIn("你们管这叫垃圾？", by_text)
        self.assertIn("本公主说话", by_text)
        self.assertEqual(["当你穿进老钱班", "垃圾桶"], [item["textZh"] for item in result["visual_only"]])
        self.assertEqual("UNRESOLVED", result["unresolved"][0]["flags"][0])
        utterances = utterance_builder.build_utterances(result["dialogue"], fixture["cues"])
        self.assertEqual(["cue-1", "cue-2"], utterances[1]["cue_ids"])
        self.assertEqual(["cue-22", "cue-23"], utterances[-1]["cue_ids"])
        self.assertTrue(all(item.get("utteranceId") for item in result["dialogue"]))
        self.assertEqual("新学期", result["dialogue"][0]["spokenZh"])
        self.assertEqual("新学期", result["dialogue"][0]["subtitleZh"])
        self.assertEqual("speaker-1", result["dialogue"][0]["speakerRef"])

    @unittest.skipIf(av_analyzer is None, "CP1 analyzer is not implemented")
    def test_av_analysis_rejects_invalid_structured_schema_and_bounds_diagnostics(self) -> None:
        with self.assertRaises(ValueError):
            av_analyzer.validate_av_analysis({"utterances": "not-a-list", "visual_only": []})
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "av_analysis.json"
            av_analyzer.persist_diagnostics(target, {**_canonical_av(), "debug": "x" * 50_000})
            self.assertLessEqual(target.stat().st_size, av_analyzer.MAX_DIAGNOSTIC_BYTES)

    @unittest.skipIf(av_analyzer is None or llm_transport is None, "CP1 analyzer is not implemented")
    def test_av_windowing_is_bounded_and_cached_analysis_avoids_a_second_provider_call(self) -> None:
        self.assertTrue(hasattr(av_analyzer, "analysis_windows"))
        self.assertEqual([(0, 25_300)], av_analyzer.analysis_windows(25_300))
        self.assertEqual([(0, 30_000), (25_000, 55_000), (50_000, 60_000)], av_analyzer.analysis_windows(60_000))
        calls = []

        def request(_payload):
            calls.append(None)
            return _canonical_av(utterances=[_av_utterance(0, 1, "新学期")])

        with tempfile.TemporaryDirectory() as directory:
            media, cache = Path(directory) / "source.mp4", Path(directory) / "av_analysis.json"
            media.write_bytes(b"fake media")
            analyzer = av_analyzer.GeminiAVAnalyzer(llm_transport.StructuredGeminiTransport(request=request))
            first = analyzer.analyze_cached(media, cache, start_ms=0, end_ms=1)
            second = analyzer.analyze_cached(media, cache, start_ms=0, end_ms=1)
            self.assertEqual(first, second)
            self.assertEqual(1, len(calls))

    @unittest.skipIf(source_harness is None, "CP1 harness is not implemented")
    def test_text_only_harness_writes_bounded_traceable_source_artifacts(self) -> None:
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            created = source_harness.run_fixture(fixture, Path(directory))
            self.assertEqual(
                {"av_analysis.json", "source_reconstruction.json", "multimodal_source_ab.json", "multimodal_source_ab.md"},
                {path.name for path in created},
            )
            artifact = json.loads((Path(directory) / "multimodal_source_ab.json").read_text(encoding="utf-8"))
            self.assertEqual("新学期", next(item["resolvedZh"] for item in artifact["after"] if item.get("resolvedZh")))
            self.assertEqual("UNRESOLVED", artifact["unresolved"][0]["flags"][0])

    @unittest.skipIf(av_analyzer is None, "CP1 analyzer is not implemented")
    def test_canonical_av_contract_is_strict_and_preserves_spoken_subtitle_fields(self) -> None:
        document = _canonical_av(
            utterances=[_av_utterance(500, 1500, "新学期", "新學期")],
            visual_only=[_av_visual(0, 500, "当你穿进老钱班")],
        )
        self.assertEqual(document, av_analyzer.validate_av_analysis(document))
        with self.assertRaises(ValueError):
            av_analyzer.validate_av_analysis({"utterances": [], "visual_only": []})

    @unittest.skipIf(source_reconciler is None, "CP1 reconciliation is not implemented")
    def test_reconciler_consumes_canonical_analyzer_output_directly(self) -> None:
        cues = [{
            "cue_id": "cue-7",
            "position": 7,
            "start_ms": 500,
            "end_ms": 1500,
            "asr_text_zh": "新字期",
            "ocr_text_zh": "新字期",
        }]
        result = source_reconciler.reconcile_source(
            cues,
            _canonical_av(utterances=[_av_utterance(500, 1500, "新学期", "新學期")]),
        )
        self.assertEqual("新学期", result["dialogue"][0]["resolvedZh"])
        self.assertEqual([7], result["dialogue"][0]["positions"])
        self.assertEqual({"asr", "ocr", "avSpoken", "avSubtitle"}, set(result["dialogue"][0]["evidence"]))
        self.assertEqual("新学期", result["dialogue"][0]["evidence"]["avSpoken"])

    @unittest.skipIf(source_reconciler is None, "CP1 reconciliation is not implemented")
    def test_visual_only_is_never_emitted_as_dialogue(self) -> None:
        result = source_reconciler.reconcile_source(
            [],
            _canonical_av(visual_only=[_av_visual(0, 500, "垃圾桶")]),
        )
        self.assertEqual([], result["dialogue"])
        self.assertEqual("VISUAL_ONLY", result["visual_only"][0]["type"])
        self.assertEqual("垃圾桶", result["visual_only"][0]["textZh"])

    @unittest.skipIf(source_reconciler is None, "CP1 reconciliation is not implemented")
    def test_overlapping_duplicate_av_cues_collapse_without_losing_numeric_identity(self) -> None:
        result = source_reconciler.reconcile_source(
            [{"cue_id": "cue-22", "position": 22, "start_ms": 0, "end_ms": 1000, "asr_text_zh": "", "ocr_text_zh": ""}],
            _canonical_av(utterances=[
                _av_utterance(100, 700, "本公主说话"),
                _av_utterance(650, 900, "本公主说话"),
            ]),
        )
        self.assertEqual(1, len(result["dialogue"]))
        self.assertEqual([22], result["dialogue"][0]["positions"])
        self.assertNotIn("AV_CONFLICT", result["dialogue"][0]["flags"])

    @unittest.skipIf(source_reconciler is None, "CP1 reconciliation is not implemented")
    def test_overlapping_conflicting_av_cues_are_bounded_and_unresolved(self) -> None:
        result = source_reconciler.reconcile_source(
            [{"cue_id": "cue-22", "position": 22, "start_ms": 0, "end_ms": 1000, "asr_text_zh": "甲", "ocr_text_zh": "乙"}],
            _canonical_av(utterances=[
                _av_utterance(100, 700, "甲"),
                _av_utterance(650, 900, "乙"),
            ]),
        )
        self.assertEqual(1, len(result["dialogue"]))
        self.assertIsNone(result["dialogue"][0]["resolvedZh"])
        self.assertIn("AV_CONFLICT", result["dialogue"][0]["flags"])
        self.assertLessEqual(len(result["dialogue"][0]["alternatives"]), 2)

    @unittest.skipIf(source_harness is None, "CP1 harness is not implemented")
    def test_persisted_runtime_loader_reconstructs_full_23_row_fixture_without_expected_input(self) -> None:
        self.assertTrue(hasattr(source_harness, "run_persisted_runtime"))
        with tempfile.TemporaryDirectory() as directory:
            result = source_harness.run_persisted_runtime(PERSISTED_FIXTURE, Path(directory))
            self.assertEqual(23, result["metrics"]["total"])
            self.assertEqual(19, result["metrics"]["resolved_dialogue"])
            self.assertEqual(1, result["metrics"]["unresolved_dialogue"])
            self.assertEqual(3, result["metrics"]["visual_only"])
            resolved = {item["resolvedZh"] for item in result["records"] if item.get("resolvedZh")}
            self.assertIn("我攒了重金买了新书包，这书包可贵", resolved)
            self.assertIn("你们管这叫垃圾？", resolved)
            self.assertIn("本公主说话", resolved)
            self.assertIsNone(result["records"][-1]["resolvedZh"])
            markdown = (Path(directory) / "multimodal_source_ab.md").read_text(encoding="utf-8")
            rows = [line for line in markdown.splitlines() if line.startswith("|") and line.split("|")[1].strip().isdigit()]
            self.assertEqual(23, len(rows))
            self.assertEqual({12}, {len(row.split("|")) - 1 for row in rows})
            self.assertEqual(
                {"av_analysis.json", "source_reconstruction.json", "multimodal_source_ab.json", "multimodal_source_ab.md"},
                {path.name for path in Path(directory).iterdir()},
            )

    @unittest.skipIf(source_reconciler is None, "CP1 reconciliation is not implemented")
    def test_canonical_identity_is_owned_after_reconciliation_and_ignores_text(self) -> None:
        records = [
            {"startMs": 1_000, "endMs": 2_000, "resolvedZh": "好", "spokenZh": "好", "subtitleZh": "好", "positions": []},
            {"startMs": 3_000, "endMs": 4_000, "resolvedZh": "原文", "spokenZh": "原文", "subtitleZh": "原文", "positions": []},
        ]
        assigned = source_reconciler.assign_canonical_utterance_ids(records)
        changed = copy.deepcopy(records)
        changed[0]["resolvedZh"] = "改写后的源文"
        changed[0]["spokenZh"] = "改写后的源文"
        changed[0]["subtitleZh"] = "改写后的源文"
        reassigned = source_reconciler.assign_canonical_utterance_ids(changed)
        self.assertEqual(
            ["utt_v1_d_00001000_00002000", "utt_v1_d_00003000_00004000"],
            [item["utteranceId"] for item in assigned],
        )
        self.assertEqual(
            [item["utteranceId"] for item in assigned],
            [item["utteranceId"] for item in reassigned],
        )

    @unittest.skipIf(source_reconciler is None, "CP1 reconciliation is not implemented")
    def test_identity_is_stable_across_key_order_and_input_reordering(self) -> None:
        first = [
            {"startMs": 2_000, "endMs": 3_000, "resolvedZh": "好", "positions": []},
            {"startMs": 1_000, "endMs": 1_500, "resolvedZh": "好", "positions": []},
        ]
        second = [
            {"positions": [], "resolvedZh": "好", "endMs": 1_500, "startMs": 1_000},
            {"positions": [], "resolvedZh": "好", "endMs": 3_000, "startMs": 2_000},
        ]
        first_by_span = {(item["startMs"], item["endMs"]): item["utteranceId"] for item in source_reconciler.assign_canonical_utterance_ids(first)}
        second_by_span = {(item["startMs"], item["endMs"]): item["utteranceId"] for item in source_reconciler.assign_canonical_utterance_ids(second)}
        self.assertEqual(first_by_span, second_by_span)

    @unittest.skipIf(source_reconciler is None, "CP1 reconciliation is not implemented")
    def test_same_text_at_different_times_has_distinct_identity(self) -> None:
        assigned = source_reconciler.assign_canonical_utterance_ids([
            {"startMs": 1_000, "endMs": 2_000, "resolvedZh": "好", "positions": []},
            {"startMs": 4_000, "endMs": 5_000, "resolvedZh": "好", "positions": []},
        ])
        self.assertEqual(2, len({item["utteranceId"] for item in assigned}))

    @unittest.skipIf(source_reconciler is None, "CP1 reconciliation is not implemented")
    def test_same_time_records_use_stable_cue_or_position_anchors(self) -> None:
        cue_records = [
            {"startMs": 1_000, "endMs": 2_000, "resolvedZh": "好", "cue_ids": ["cue-b"]},
            {"startMs": 1_000, "endMs": 2_000, "resolvedZh": "不同内容", "cue_ids": ["cue-a"]},
        ]
        reversed_ids = {
            tuple(item["cue_ids"]): item["utteranceId"]
            for item in source_reconciler.assign_canonical_utterance_ids(list(reversed(cue_records)))
        }
        original_ids = {
            tuple(item["cue_ids"]): item["utteranceId"]
            for item in source_reconciler.assign_canonical_utterance_ids(cue_records)
        }
        self.assertEqual(original_ids, reversed_ids)
        self.assertEqual(2, len(set(original_ids.values())))

        position_records = [
            {"startMs": 1_000, "endMs": 2_000, "resolvedZh": "甲", "positions": [11]},
            {"startMs": 1_000, "endMs": 2_000, "resolvedZh": "乙", "positions": [12]},
        ]
        position_ids = source_reconciler.assign_canonical_utterance_ids(position_records)
        self.assertEqual(2, len({item["utteranceId"] for item in position_ids}))

    @unittest.skipIf(source_reconciler is None, "CP1 reconciliation is not implemented")
    def test_same_time_without_stable_anchor_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "IDENTITY_AMBIGUOUS"):
            source_reconciler.assign_canonical_utterance_ids([
                {"startMs": 1_000, "endMs": 2_000, "spokenZh": "好"},
                {"startMs": 1_000, "endMs": 2_000, "spokenZh": "不同内容"},
            ])

    @unittest.skipIf(source_harness is None or vi_harness is None, "CP1/CP2 bridge is not implemented")
    def test_persisted_canonical_artifact_feeds_cp2_without_identity_synthesis(self) -> None:
        fixture = {
            "fixture_version": 2,
            "cues": [],
            "unresolved": None,
            "av_analysis": _canonical_av(
                utterances=[_av_utterance(500, 1500, "好", "好")],
                visual_only=[_av_visual(0, 500, "字幕")],
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            source_harness.run_fixture(fixture, Path(directory))
            document = json.loads((Path(directory) / "source_reconstruction.json").read_text(encoding="utf-8"))
            records = vi_harness.records_from_source_reconstruction(document)
        self.assertEqual(2, document["schemaVersion"])
        self.assertEqual(1, document["identityVersion"])
        self.assertEqual(1, len(document["utterances"]))
        self.assertEqual("utt_v1_d_00000500_00001500", document["utterances"][0]["utteranceId"])
        self.assertEqual(500, records[0]["start_ms"])
        self.assertEqual(1500, records[0]["end_ms"])
        self.assertEqual("好", document["utterances"][0]["spokenZh"])
        self.assertEqual("好", document["utterances"][0]["subtitleZh"])
        self.assertEqual([], document["utterances"][0]["positions"])
        self.assertNotIn("utteranceId", document["visualOnly"][0])
        self.assertEqual(document["utterances"][0]["utteranceId"], records[0]["utteranceId"])

    @unittest.skipIf(vi_harness is None, "CP2 bridge is not implemented")
    def test_cp2_preserves_identity_and_rejects_missing_identity_or_timing(self) -> None:
        canonical = {
            "schemaVersion": 2,
            "identityVersion": 1,
            "scene": {"summary": "scene", "tone": "comic", "register": "spoken"},
            "utterances": [{
                "utteranceId": "stable-source-id",
                "startMs": 100,
                "endMs": 200,
                "resolvedZh": "改过的文字",
                "spokenZh": "改过的文字",
                "subtitleZh": "改过的文字",
                "positions": [],
                "flags": ["AV_CONFLICT", "UNRESOLVED"],
                "confidence": "LOW",
            }],
            "visualOnly": [],
        }
        record = vi_harness.records_from_source_reconstruction(canonical)[0]
        self.assertEqual("stable-source-id", record["utteranceId"])
        self.assertEqual(["AV_CONFLICT", "UNRESOLVED"], record["source_flags"])
        self.assertEqual("low", record["source_confidence"])
        with self.assertRaisesRegex(ValueError, "identity"):
            vi_harness.records_from_source_reconstruction({**canonical, "utterances": [{"startMs": 1, "endMs": 2}]})
        with self.assertRaisesRegex(ValueError, "timing"):
            vi_harness.records_from_source_reconstruction({**canonical, "utterances": [{"utteranceId": "id"}]})

    @unittest.skipIf(vi_harness is None, "CP2 bridge is not implemented")
    def test_legacy_records_without_canonical_utterances_fail_without_synthesis(self) -> None:
        with self.assertRaisesRegex(ValueError, "utterance"):
            vi_harness.records_from_source_reconstruction({
                "records": [{"startMs": 1, "endMs": 2, "resolvedZh": "好"}],
            })

    @unittest.skipIf(source_harness is None, "CP1 persistence is not implemented")
    def test_persisted_scene_and_flags_are_canonical_and_resume_is_deterministic(self) -> None:
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            source_harness.run_fixture(fixture, Path(directory))
            path = Path(directory) / "source_reconstruction.json"
            first = json.loads(path.read_text(encoding="utf-8"))
            second = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(first["scene"], fixture["av_analysis"]["scene"])
        self.assertEqual(
            [item["utteranceId"] for item in first["utterances"]],
            [item["utteranceId"] for item in second["utterances"]],
        )
        self.assertEqual(
            [item["flags"] for item in first["utterances"]],
            [item["flags"] for item in second["utterances"]],
        )


if __name__ == "__main__":
    unittest.main()
