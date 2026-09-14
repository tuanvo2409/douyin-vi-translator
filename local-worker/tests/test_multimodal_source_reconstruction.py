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


llm_transport = _optional_module("llm_transport")
av_analyzer = _optional_module("av_analyzer")
source_reconciler = _optional_module("source_reconciler")
utterance_builder = _optional_module("utterance_builder")
source_harness = _optional_module("multimodal_source_harness")

FIXTURE = Path(__file__).parent / "fixtures" / "multimodal_25s" / "source_fixture.json"


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
            return {"utterances": [{"start_ms": 0, "end_ms": 1, "text_zh": "新学期"}], "visual_only": []}

        transport = llm_transport.StructuredGeminiTransport(request=request, max_attempts=2)
        result = transport.call({"contents": []}, required_keys={"utterances", "visual_only"})
        self.assertEqual("新学期", result["utterances"][0]["text_zh"])
        self.assertEqual(2, len(calls))
        with self.assertRaises(llm_transport.StructuredResponseError):
            llm_transport.validate_structured_document({"utterances": []}, {"utterances", "visual_only"})

    @unittest.skipIf(source_reconciler is None or utterance_builder is None, "CP1 reconciliation is not implemented")
    def test_fixture_repairs_av_supported_dialogue_without_promoting_visual_or_unresolved_text(self) -> None:
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        result = source_reconciler.reconcile_source(fixture["cues"], fixture["av_windows"], fixture["unresolved"])
        by_text = {item["text_zh"]: item for item in result["dialogue"]}
        self.assertIn("新学期", by_text)
        self.assertIn("我攒了重金买了新书包，这书包可贵", by_text)
        self.assertIn("你们管这叫垃圾？", by_text)
        self.assertIn("本公主说话", by_text)
        self.assertEqual(["当你穿进老钱班", "垃圾桶"], [item["text_zh"] for item in result["visual_only"]])
        self.assertEqual("unresolved", result["unresolved"][0]["status"])
        utterances = utterance_builder.build_utterances(result["dialogue"], fixture["cues"])
        self.assertEqual(["cue-1", "cue-2"], utterances[1]["cue_ids"])
        self.assertEqual(["cue-22", "cue-23"], utterances[-1]["cue_ids"])

    @unittest.skipIf(av_analyzer is None, "CP1 analyzer is not implemented")
    def test_av_analysis_rejects_invalid_structured_schema_and_bounds_diagnostics(self) -> None:
        with self.assertRaises(ValueError):
            av_analyzer.validate_av_analysis({"utterances": "not-a-list", "visual_only": []})
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "av_analysis.json"
            av_analyzer.persist_diagnostics(target, {"utterances": [], "visual_only": [], "debug": "x" * 50_000})
            self.assertLessEqual(target.stat().st_size, av_analyzer.MAX_DIAGNOSTIC_BYTES)

    @unittest.skipIf(av_analyzer is None or llm_transport is None, "CP1 analyzer is not implemented")
    def test_av_windowing_is_bounded_and_cached_analysis_avoids_a_second_provider_call(self) -> None:
        self.assertTrue(hasattr(av_analyzer, "analysis_windows"))
        self.assertEqual([(0, 25_300)], av_analyzer.analysis_windows(25_300))
        self.assertEqual([(0, 30_000), (25_000, 55_000), (50_000, 60_000)], av_analyzer.analysis_windows(60_000))
        calls = []

        def request(_payload):
            calls.append(None)
            return {"utterances": [{"start_ms": 0, "end_ms": 1, "text_zh": "新学期"}], "visual_only": []}

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
            self.assertEqual("新学期", artifact["after"][0]["text_zh"])
            self.assertEqual("unresolved", artifact["unresolved"][0]["status"])


if __name__ == "__main__":
    unittest.main()
