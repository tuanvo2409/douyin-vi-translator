import json
import hashlib
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from modern_p1c_adapter import ModernP1CAdapterError, SourceReviewRequiredError, build_modern_stage_bundle


class ModernSourceOCRRouteTests(unittest.TestCase):
    @staticmethod
    def _cache_runtime(**overrides):
        runtime = {
            "localizationProfile": "page_goc_tro_bat_on",
            "sourceEvidenceMode": "local_asr_text_only",
            "sourceOcrVerify": True,
            "page1MaskMode": "none",
            "whisperModel": "small",
            "ocr": {"discoveryFrames": 8},
            "roi": {"xPercent": 2, "yPercent": 66, "widthPercent": 96, "heightPercent": 9.8},
        }
        runtime.update(overrides)
        return runtime

    @staticmethod
    def _cache_media_mocks(asr, scan):
        detections = [{
            "timestampMs": 500,
            "bbox": [150, 880, 400, 915],
            "text": "第一句",
            "category": "DIALOGUE_SUBTITLE",
            "confidence": 0.98,
        }]
        return (
            patch("dubvi_worker.ffprobe_duration", return_value=1.0),
            patch("dubvi_worker.ffprobe_dimensions", return_value=(576, 1024)),
            patch("dubvi_worker.extract_audio"),
            patch("dubvi_worker.ensure_whisper_model"),
            patch("dubvi_worker.transcribe", return_value=asr),
            patch("auto_roi.sparse_whole_frame_discovery", return_value=detections),
            patch("source_ocr_verifier.infer_stable_subtitle_roi", return_value=(100, 800, 500, 950)),
            patch("source_ocr_verifier.scan_cue_ocr", return_value=scan),
        )

    def _run_cache_source(self, source, output, runtime=None, *, transcribe=None, scan=None):
        stages = build_modern_stage_bundle(runtime_config=runtime or self._cache_runtime())
        asr = transcribe or [{
            "position": 0, "startMs": 0, "endMs": 1000,
            "asrTextZh": "第一句", "confidence": 95,
        }]
        readings = scan or {0: [{"text": "第一句", "confidence": 0.98}] * 2}
        mocks = self._cache_media_mocks(asr, readings)
        with ExitStack() as stack:
            entered = [stack.enter_context(mocked) for mocked in mocks]
            result = stages["source"](source, output)
        return result, entered

    def test_complete_provenance_bound_evidence_skips_asr_and_cue_ocr_on_second_call(self):
        asr = [{"position": 0, "startMs": 0, "endMs": 1000, "asrTextZh": "第一句", "confidence": 95}]
        readings = {0: [{"text": "第一句", "confidence": 0.98}] * 2}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            source.write_bytes(b"raw-page-one")
            output = Path(directory) / "attempt"
            output.mkdir()
            self._run_cache_source(source, output, transcribe=asr, scan=readings)
            evidence = json.loads((output / "source_evidence.json").read_text(encoding="utf-8"))
            self.assertEqual("complete", evidence["status"])
            self.assertEqual("sha256:" + hashlib.sha256(source.read_bytes()).hexdigest(), evidence["provenance"]["rawSha256"])
            self.assertEqual(1000, evidence["provenance"]["durationMs"])
            self.assertEqual("small", evidence["provenance"]["model"])
            self.assertEqual({"device": "cpu", "type": "int8"}, evidence["provenance"]["compute"])
            self.assertEqual(self._cache_runtime()["roi"], evidence["provenance"]["roi"])

            stages = build_modern_stage_bundle(runtime_config=self._cache_runtime())
            with patch("dubvi_worker.ffprobe_duration", return_value=1.0), \
                 patch("dubvi_worker.transcribe", side_effect=AssertionError("cache miss retranscribed")) as transcribe, \
                 patch("source_ocr_verifier.scan_cue_ocr", side_effect=AssertionError("cache miss rescanned")) as scan:
                cached = stages["source"](source, output)
            self.assertEqual("第一句", cached[0]["resolvedZh"])
            transcribe.assert_not_called()
            scan.assert_not_called()

    def test_changed_raw_bytes_are_a_cache_miss(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            source.write_bytes(b"raw-before")
            output = Path(directory) / "attempt"
            output.mkdir()
            self._run_cache_source(source, output)
            source.write_bytes(b"raw-after")
            _, mocks = self._run_cache_source(source, output)
            self.assertEqual(1, mocks[3].call_count)

    def test_changed_model_or_roi_is_a_cache_miss(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            source.write_bytes(b"raw-stable")
            output = Path(directory) / "attempt"
            output.mkdir()
            original = self._cache_runtime()
            self._run_cache_source(source, output, original)
            changed = self._cache_runtime(whisperModel="medium")
            _, mocks = self._run_cache_source(source, output, changed)
            self.assertEqual(1, mocks[3].call_count)

            changed_roi = self._cache_runtime(roi={"xPercent": 3, "yPercent": 66, "widthPercent": 96, "heightPercent": 9.8})
            _, mocks = self._run_cache_source(source, output, changed_roi)
            self.assertEqual(1, mocks[3].call_count)

    def test_changed_whisper_compute_type_is_a_cache_miss(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            source.write_bytes(b"raw-compute")
            output = Path(directory) / "attempt"
            output.mkdir()
            self._run_cache_source(source, output)
            with patch.dict("os.environ", {"DUBVI_WHISPER_COMPUTE_TYPE": "float32"}):
                _, mocks = self._run_cache_source(source, output)
            self.assertEqual(1, mocks[3].call_count)

    def test_page1_mask_route_reuses_verified_source_without_repeating_asr_or_cue_ocr(self):
        runtime = self._cache_runtime(page1MaskMode="fixed")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            source.write_bytes(b"raw-with-fixed-mask")
            output = Path(directory) / "attempt"
            output.mkdir()
            with patch("modern_p1c_adapter._page1_mask_geometry", return_value=None):
                self._run_cache_source(source, output, runtime)
            stages = build_modern_stage_bundle(runtime_config=runtime)
            with patch("dubvi_worker.ffprobe_duration", return_value=1.0), \
                 patch("dubvi_worker.ffprobe_dimensions", return_value=(576, 1024)), \
                 patch("auto_roi.sparse_whole_frame_discovery", return_value=[]), \
                 patch("modern_p1c_adapter._page1_mask_geometry", return_value=None), \
                 patch("dubvi_worker.transcribe", side_effect=AssertionError("ASR repeated")) as transcribe, \
                 patch("source_ocr_verifier.scan_cue_ocr", side_effect=AssertionError("cue OCR repeated")) as scan:
                cached = stages["source"](source, output)
            self.assertEqual("第一句", cached[0]["resolvedZh"])
            transcribe.assert_not_called()
            scan.assert_not_called()

    def test_incomplete_malformed_or_out_of_bounds_evidence_is_a_cache_miss(self):
        malformed_documents = [
            {"schemaVersion": 1, "status": "complete"},
            {"schemaVersion": 1, "status": "complete", "provenance": {"rawSha256": "bad"}, "cues": []},
            {"schemaVersion": 1, "status": "complete", "provenance": {"rawSha256": "sha256:" + "0" * 64, "durationMs": 1000}, "cues": [{"position": 0, "start_ms": 0, "end_ms": 2000}]},
        ]
        for document in malformed_documents:
            with self.subTest(document=document), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "source.mp4"
                source.write_bytes(b"raw-malformed")
                output = Path(directory) / "attempt"
                output.mkdir()
                (output / "source_evidence.json").write_text(json.dumps(document), encoding="utf-8")
                _, mocks = self._run_cache_source(source, output)
                self.assertEqual(1, mocks[3].call_count)

    def test_symlinked_evidence_is_not_followed_as_attempt_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            source.write_bytes(b"raw-symlink")
            output = root / "attempt"
            output.mkdir()
            external = root / "external-source-evidence.json"
            external.write_text(json.dumps({"status": "complete"}), encoding="utf-8")
            try:
                (output / "source_evidence.json").symlink_to(external)
            except OSError as exc:
                self.skipTest(f"symlink unavailable: {exc}")
            _, mocks = self._run_cache_source(source, output)
            self.assertEqual(1, mocks[3].call_count)

    def test_cache_hit_still_applies_source_truth_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            source.write_bytes(b"raw-gate")
            output = Path(directory) / "attempt"
            output.mkdir()
            self._run_cache_source(source, output)
            evidence_path = output / "source_evidence.json"
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            evidence["records"][0]["decision"] = "ASR_OCR_CONFLICT"
            evidence["records"][0]["classification"] = "UNRESOLVED_DIALOGUE"
            evidence["cues"][0]["needs_review"] = True
            evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
            stages = build_modern_stage_bundle(runtime_config=self._cache_runtime())
            conflict = {0: [{"text": "第二句", "confidence": 0.98}] * 2}
            with ExitStack() as stack:
                mocks = [stack.enter_context(mocked) for mocked in self._cache_media_mocks(
                    [{"position": 0, "startMs": 0, "endMs": 1000, "asrTextZh": "第一句", "confidence": 95}], conflict,
                )]
                with self.assertRaises(SourceReviewRequiredError):
                    stages["source"](source, output)
                mocks[4].assert_called_once()

    def test_tampered_resolved_text_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            source.write_bytes(b"raw-source-truth")
            output = Path(directory) / "attempt"
            output.mkdir()
            self._run_cache_source(source, output)
            evidence_path = output / "source_evidence.json"
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            evidence["records"][0]["resolvedZh"] = "伪造的话"
            evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
            restored, mocks = self._run_cache_source(source, output)
            self.assertEqual("第一句", restored[0]["resolvedZh"])
            mocks[4].assert_called_once()

    def test_page1_missing_stable_roi_records_exact_unverified_cue(self):
        stages = build_modern_stage_bundle(runtime_config={
            "localizationProfile": "page_goc_tro_bat_on",
            "sourceEvidenceMode": "local_asr_text_only",
            "sourceOcrVerify": True,
            "page1MaskMode": "none",
        })
        asr = [{"position": 7, "startMs": 0, "endMs": 1000, "asrTextZh": "空间太小", "confidence": 99}]
        with tempfile.TemporaryDirectory() as directory, \
             patch("dubvi_worker.ffprobe_duration", return_value=1.0), \
             patch("dubvi_worker.ffprobe_dimensions", return_value=(576, 1024)), \
             patch("dubvi_worker.extract_audio"), \
             patch("dubvi_worker.ensure_whisper_model"), \
             patch("dubvi_worker.transcribe", return_value=asr), \
             patch("auto_roi.sparse_whole_frame_discovery", return_value=[]):
            output = Path(directory)
            with self.assertRaisesRegex(SourceReviewRequiredError, "SOURCE_EVIDENCE_UNRESOLVED"):
                stages["source"](Path("source.mp4"), output)
            evidence = json.loads((output / "source_evidence.json").read_text(encoding="utf-8"))
        self.assertEqual("review_required", evidence["status"])
        self.assertEqual([7], evidence["unresolvedPositions"])

    def test_ocr_discovery_exception_writes_explicit_failure_evidence(self):
        stages = build_modern_stage_bundle(runtime_config={
            "localizationProfile": "page_goc_tro_bat_on",
            "sourceEvidenceMode": "local_asr_text_only",
            "page1MaskMode": "none",
        })
        asr = [{
            "position": 0, "startMs": 0, "endMs": 1000,
            "asrTextZh": "对白", "confidence": 90,
        }]
        with tempfile.TemporaryDirectory() as directory, \
             patch("dubvi_worker.ffprobe_duration", return_value=1.0), \
             patch("dubvi_worker.extract_audio"), \
             patch("dubvi_worker.ensure_whisper_model"), \
             patch("dubvi_worker.transcribe", return_value=asr), \
             patch("auto_roi.sparse_whole_frame_discovery", side_effect=RuntimeError("OCR backend unavailable")):
            output = Path(directory)
            with self.assertRaisesRegex(ModernP1CAdapterError, "OCR discovery failed"):
                stages["source"](Path("source.mp4"), output)
            evidence_path = output / "source_evidence.json"
            self.assertTrue(evidence_path.is_file())
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))

        self.assertIn(evidence.get("status"), {"failed", "incomplete"})
        self.assertNotEqual("success", evidence.get("status"))
        self.assertIn("OCR", json.dumps(evidence, ensure_ascii=False))

    def test_page2_matching_changing_ocr_is_promoted_to_typed_dialogue_mask_input(self):
        stages = build_modern_stage_bundle(runtime_config={
            "localizationProfile": "page_giai_cuu_chuong_lon",
            "sourceEvidenceMode": "multimodal_av",
            "page1MaskMode": "none",
        })
        analysis = {
            "windowStartMs": 0,
            "windowEndMs": 1000,
            "scene": {"summary": "scene", "tone": "neutral", "register": "colloquial"},
            "utterances": [
                {"startMs": 100, "endMs": 400, "spokenZh": "第一句", "subtitleZh": "第一句", "speakerRef": "s1", "intent": "statement", "emotion": "neutral", "confidence": "HIGH"},
                {"startMs": 500, "endMs": 800, "spokenZh": "第二句", "subtitleZh": "第二句", "speakerRef": "s1", "intent": "statement", "emotion": "neutral", "confidence": "HIGH"},
            ],
            "visualOnly": [],
        }
        detections = [
            {"timestampMs": 250, "bbox": [100, 1500, 400, 1560], "text": "第一句", "category": "UNKNOWN", "confidence": 0.95},
            {"timestampMs": 650, "bbox": [110, 1502, 430, 1562], "text": "第二句", "category": "UNKNOWN", "confidence": 0.94},
        ]
        seen = []

        def capture_mask_plan(items, _frame_size):
            seen.extend(items)
            return []

        with tempfile.TemporaryDirectory() as directory, \
             patch("dubvi_worker.ffprobe_duration", return_value=1.0), \
             patch("dubvi_worker.ffprobe_dimensions", return_value=(1080, 1920)), \
             patch("av_analyzer.GeminiAVAnalyzer", return_value=object()), \
             patch("av_analyzer.analyze_media_windows_with_subdivision", return_value={"av_analysis": analysis}), \
             patch("auto_roi.sparse_whole_frame_discovery", return_value=detections), \
             patch("localized_masking.build_mask_plan", side_effect=capture_mask_plan):
            stages["source"](Path("source.mp4"), Path(directory))

        self.assertEqual(2, len(seen))
        self.assertEqual(["DIALOGUE_SUBTITLE", "DIALOGUE_SUBTITLE"], [item["category"] for item in seen])
        self.assertEqual([(100, 1500, 400, 1560), (110, 1502, 430, 1562)], [tuple(item["bbox"]) for item in seen])

    def test_page2_unmatched_lower_third_requires_review_without_dialogue_mask(self):
        stages = build_modern_stage_bundle(runtime_config={
            "localizationProfile": "page_giai_cuu_chuong_lon",
            "sourceEvidenceMode": "multimodal_av",
            "page1MaskMode": "none",
        })
        analysis = {
            "windowStartMs": 0,
            "windowEndMs": 1000,
            "scene": {"summary": "scene", "tone": "neutral", "register": "colloquial"},
            "utterances": [{"startMs": 100, "endMs": 400, "spokenZh": "对白", "subtitleZh": "对白", "speakerRef": "s1", "intent": "statement", "emotion": "neutral", "confidence": "HIGH"}],
            "visualOnly": [],
        }
        detections = [{
            "timestampMs": 700,
            "bbox": [100, 1500, 500, 1560],
            "text": "未匹配下三分之一字幕",
            "category": "UNKNOWN",
            "confidence": 0.96,
        }]
        seen = []

        def capture_mask_plan(items, _frame_size):
            seen.extend(items)
            return []

        with tempfile.TemporaryDirectory() as directory, \
             patch("dubvi_worker.ffprobe_duration", return_value=1.0), \
             patch("dubvi_worker.ffprobe_dimensions", return_value=(1080, 1920)), \
             patch("av_analyzer.GeminiAVAnalyzer", return_value=object()), \
             patch("av_analyzer.analyze_media_windows_with_subdivision", return_value={"av_analysis": analysis}), \
             patch("auto_roi.sparse_whole_frame_discovery", return_value=detections), \
             patch("localized_masking.build_mask_plan", side_effect=capture_mask_plan):
            with self.assertRaisesRegex(SourceReviewRequiredError, "SOURCE_REVIEW_REQUIRED"):
                stages["source"](Path("source.mp4"), Path(directory))

        self.assertEqual([], seen)

    def test_page1_uses_repeated_ocr_to_correct_asr_before_localization(self):
        stages = build_modern_stage_bundle(runtime_config={
            "localizationProfile": "page_goc_tro_bat_on", "sourceEvidenceMode": "local_asr_text_only",
            "page1MaskMode": "none", "sourceOcrVerify": True,
            "sensevoiceExe": "sensevoice.exe", "sensevoiceModel": "sensevoice.gguf",
        })
        asr = [
            {"position": 0, "startMs": 0, "endMs": 1000, "asrTextZh": "老婆生了龙胖胎", "confidence": 90},
            {"position": 1, "startMs": 1000, "endMs": 2000, "asrTextZh": "空间太小", "confidence": 90},
            {"position": 2, "startMs": 2000, "endMs": 3000, "asrTextZh": "再去二舅家", "confidence": 90},
        ]
        detections = [
            {"timestampMs": n * 1000 + 500, "bbox": [150, 880, 400, 915], "text": text, "category": "UNKNOWN"}
            for n, text in enumerate(("老婆生了龙凤胎", "空间太小", "再去二舅家"))
        ]
        readings = {
            0: [{"text": "老婆生了龙凤胎", "confidence": 0.98}] * 2,
            1: [{"text": "空间太小", "confidence": 0.97}] * 2,
            2: [{"text": "再去二舅家", "confidence": 0.98}] * 2,
        }
        with tempfile.TemporaryDirectory() as directory, \
             patch("dubvi_worker.ffprobe_duration", return_value=3.0), \
             patch("dubvi_worker.ffprobe_dimensions", return_value=(576, 1024)), \
             patch("dubvi_worker.extract_audio"), \
             patch("dubvi_worker.ensure_whisper_model"), \
             patch("dubvi_worker.transcribe", return_value=asr), \
             patch("auto_roi.sparse_whole_frame_discovery", return_value=detections), \
             patch("source_ocr_verifier.scan_cue_ocr", return_value=readings) as scan, \
             patch("source_ocr_verifier.scan_second_asr", return_value={0: "老婆生了龙凤胎"}) as second:
            output = Path(directory)
            records = stages["source"](Path("source.mp4"), output)
            evidence = json.loads((output / "source_evidence.json").read_text(encoding="utf-8"))
        self.assertEqual("老婆生了龙凤胎", records[0]["resolvedZh"])
        self.assertEqual("OCR_SECOND_ASR_AGREE", records[0]["decision"])
        self.assertEqual("空间太小", records[1]["resolvedZh"])
        self.assertEqual("RESOLVED_DIALOGUE", records[2]["classification"])
        self.assertEqual("ASR_OCR_AGREE", records[2]["decision"])
        self.assertEqual("老婆生了龙凤胎", evidence["cues"][0]["ocr_text_zh"])
        scan.assert_called_once()
        second.assert_called_once()

    def test_page1_conflicting_ocr_records_position_and_stops_before_localization(self):
        stages = build_modern_stage_bundle(runtime_config={
            "localizationProfile": "page_goc_tro_bat_on", "sourceEvidenceMode": "local_asr_text_only",
            "page1MaskMode": "none", "sourceOcrVerify": True,
        })
        asr = [
            {"position": n, "startMs": n * 1000, "endMs": (n + 1) * 1000,
             "asrTextZh": text, "confidence": 99}
            for n, text in enumerate(("第一句", "空间太小", "再去二舅家"))
        ]
        detections = [
            {"timestampMs": n * 1000 + 500, "bbox": [150, 880, 400, 915],
             "text": text, "category": "DIALOGUE_SUBTITLE"}
            for n, text in enumerate(("第一句", "空间太小", "再去二舅家"))
        ]
        readings = {
            0: [{"text": "第一句", "confidence": 0.98}] * 2,
            1: [{"text": "空间太小", "confidence": 0.98}] * 2,
            2: [{"text": "再去二舅家搬床垫", "confidence": 0.98}] * 2,
        }
        with tempfile.TemporaryDirectory() as directory, \
             patch("dubvi_worker.ffprobe_duration", return_value=3.0), \
             patch("dubvi_worker.ffprobe_dimensions", return_value=(576, 1024)), \
             patch("dubvi_worker.extract_audio"), \
             patch("dubvi_worker.ensure_whisper_model"), \
             patch("dubvi_worker.transcribe", return_value=asr), \
             patch("auto_roi.sparse_whole_frame_discovery", return_value=detections), \
             patch("source_ocr_verifier.scan_cue_ocr", return_value=readings):
            output = Path(directory)
            with self.assertRaisesRegex(SourceReviewRequiredError, "SOURCE_EVIDENCE_UNRESOLVED"):
                stages["source"](Path("source.mp4"), output)
            evidence = json.loads((output / "source_evidence.json").read_text(encoding="utf-8"))
        self.assertEqual("review_required", evidence["status"])
        self.assertEqual([2], evidence["unresolvedPositions"])
        self.assertEqual("UNRESOLVED_DIALOGUE", evidence["records"][2]["classification"])

    def test_ocr_repeat_accepts_and_caches_verified_source_record(self):
        stages = build_modern_stage_bundle(runtime_config=self._cache_runtime())
        asr = [{"position": 0, "startMs": 0, "endMs": 1000, "asrTextZh": "放冬天的免费", "confidence": 95}]
        readings = {
            0: [
                {"text": "放冬天的棉被", "confidence": 0.96, "timestampMs": 350, "bbox": [150, 880, 400, 915], "category": "DIALOGUE_SUBTITLE"},
                {"text": "放冬天的棉被", "confidence": 0.95, "timestampMs": 650, "bbox": [150, 880, 400, 915], "category": "DIALOGUE_SUBTITLE"},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            source.write_bytes(b"raw-ocr-repeat")
            output = Path(directory) / "attempt"
            output.mkdir()
            records, _ = self._run_cache_source(source, output, transcribe=asr, scan=readings)
            self.assertEqual("放冬天的棉被", records[0]["resolvedZh"])
            self.assertEqual("OCR_REPEAT", records[0]["decision"])
            self.assertEqual("RESOLVED_DIALOGUE", records[0]["classification"])
            evidence = json.loads((output / "source_evidence.json").read_text(encoding="utf-8"))
            self.assertEqual("complete", evidence["status"])
            self.assertEqual("OCR_REPEAT", evidence["records"][0]["decision"])
            self.assertEqual(2, len(evidence["cues"][0].get("ocr_readings", [])))

            # Warm hit must skip ASR and OCR
            with patch("dubvi_worker.ffprobe_duration", return_value=1.0), \
                 patch("dubvi_worker.transcribe", side_effect=AssertionError("ASR re-run")), \
                 patch("source_ocr_verifier.scan_cue_ocr", side_effect=AssertionError("OCR re-run")):
                cached = stages["source"](source, output)
            self.assertEqual("放冬天的棉被", cached[0]["resolvedZh"])
            self.assertEqual("OCR_REPEAT", cached[0]["decision"])

    def test_old_stage_version_or_decision_version_is_cache_miss(self):
        asr = [{"position": 0, "startMs": 0, "endMs": 1000, "asrTextZh": "第一句", "confidence": 95}]
        readings = {0: [{"text": "第一句", "confidence": 0.98}] * 2}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            source.write_bytes(b"raw-old-version")
            output = Path(directory) / "attempt"
            output.mkdir()
            self._run_cache_source(source, output, transcribe=asr, scan=readings)
            evidence_path = output / "source_evidence.json"
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            evidence["provenance"]["stageVersion"] = "local-asr-ocr-v1"
            evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

            _, mocks = self._run_cache_source(source, output, transcribe=asr, scan=readings)
            self.assertEqual(1, mocks[4].call_count)

    def test_ocr_repeat_missing_readings_in_cache_is_cache_miss(self):
        asr = [{"position": 0, "startMs": 0, "endMs": 1000, "asrTextZh": "第一句", "confidence": 95}]
        readings = {0: [{"text": "第一句", "confidence": 0.98}] * 2}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            source.write_bytes(b"raw-missing-readings")
            output = Path(directory) / "attempt"
            output.mkdir()
            self._run_cache_source(source, output, transcribe=asr, scan=readings)
            evidence_path = output / "source_evidence.json"
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            evidence["records"][0]["decision"] = "OCR_REPEAT"
            evidence["cues"][0]["ocr_readings"] = []
            evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

            _, mocks = self._run_cache_source(source, output, transcribe=asr, scan=readings)
            self.assertEqual(1, mocks[4].call_count)

    def test_old_manual_or_asr_only_lock_is_cache_miss(self):
        asr = [{"position": 0, "startMs": 0, "endMs": 1000, "asrTextZh": "第一句", "confidence": 95}]
        readings = {0: [{"text": "第一句", "confidence": 0.98}] * 2}
        for bad_decision in ("MANUAL_EVIDENCE_RECONCILIATION", "ASR_ONLY"):
            with tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "source.mp4"
                source.write_bytes(b"raw-bad-decision")
                output = Path(directory) / "attempt"
                output.mkdir()
                self._run_cache_source(source, output, transcribe=asr, scan=readings)
                evidence_path = output / "source_evidence.json"
                evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
                evidence["records"][0]["decision"] = bad_decision
                evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

                _, mocks = self._run_cache_source(source, output, transcribe=asr, scan=readings)
                self.assertEqual(1, mocks[4].call_count)


if __name__ == "__main__":
    unittest.main()

