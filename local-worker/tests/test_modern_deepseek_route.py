import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from modern_p1c_adapter import build_modern_stage_bundle


class ModernDeepSeekRouteTests(unittest.TestCase):
    def test_page1_deepseek_setting_uses_cp2_and_records_provider(self):
        stages = build_modern_stage_bundle(runtime_config={
            "localizationProfile": "page_goc_tro_bat_on",
            "sourceEvidenceMode": "local_asr_text_only", "page1MaskMode": "none",
            "localizationProvider": "deepseek",
        })

        def answer(payload):
            key = "semanticVi" if payload["mode"] == "semantic" else "nativeVi"
            return {"translations": [{"utteranceId": row["utteranceId"], key: "Hay là sửa thế này"}
                                     for row in payload["utterances"]]}

        with tempfile.TemporaryDirectory() as directory, \
             patch("dubvi_worker.ffprobe_duration", return_value=1.0), \
             patch("dubvi_worker.extract_audio"), \
             patch("dubvi_worker.ensure_whisper_model"), \
             patch("dubvi_worker.transcribe", return_value=[{"position": 0, "startMs": 0, "endMs": 800, "asrTextZh": "不如这样改", "confidence": 90}]), \
             patch("auto_roi.sparse_whole_frame_discovery", return_value=[]), \
             patch("cp2_deepseek_adapter.CP2DeepSeekAdapter", return_value=Mock(side_effect=answer, last_provider="deepseek", gemini_failure=None)) as deepseek, \
             patch("cp2_gemini_adapter.live_cp2_gemini_adapter") as gemini:
            output = Path(directory)
            records = stages["source"](Path("source.mp4"), output)
            localized = stages["localization"](records, output_dir=output)
            provider = json.loads((output / "translation_provider.json").read_text(encoding="utf-8"))
        self.assertEqual("Hay là sửa thế này", localized[0]["nativeVi"])
        self.assertEqual("deepseek", provider["providerPath"])
        self.assertEqual(1, deepseek.call_count)
        gemini.assert_not_called()

    def test_canonical_page1_delivery_flow_with_ocr_repeat(self):
        runtime_config = {
            "localizationProfile": "page_goc_tro_bat_on",
            "sourceEvidenceMode": "local_asr_text_only",
            "sourceOcrVerify": True,
            "page1MaskMode": "fixed",
            "localizationProvider": "deepseek",
            "ttsProvider": "capcut",
            "voice": {"name": "BV075_streaming"},
            "ocr": {"discoveryFrames": 8},
            "roi": {"xPercent": 2, "yPercent": 66, "widthPercent": 96, "heightPercent": 9.8},
        }
        stages = build_modern_stage_bundle(runtime_config=runtime_config)

        asr = [
            {"position": 0, "startMs": 0, "endMs": 1000, "asrTextZh": "放冬天的免费", "confidence": 95},
            {"position": 1, "startMs": 1000, "endMs": 2000, "asrTextZh": "第二句对话", "confidence": 95},
        ]
        detections = [
            {"timestampMs": 0, "bbox": [50, 100, 500, 200], "text": "故事标题", "category": "STORY_TEXT"},
            {"timestampMs": 350, "bbox": [150, 880, 400, 915], "text": "放冬天的棉被", "category": "DIALOGUE_SUBTITLE"},
            {"timestampMs": 650, "bbox": [150, 880, 400, 915], "text": "放冬天的棉被", "category": "DIALOGUE_SUBTITLE"},
            {"timestampMs": 1350, "bbox": [150, 880, 400, 915], "text": "第二句对话", "category": "DIALOGUE_SUBTITLE"},
            {"timestampMs": 1650, "bbox": [150, 880, 400, 915], "text": "第二句对话", "category": "DIALOGUE_SUBTITLE"},
        ]
        readings = {
            0: [
                {"text": "放冬天的棉被", "confidence": 0.96, "timestampMs": 350, "bbox": [150, 880, 400, 915], "category": "DIALOGUE_SUBTITLE"},
                {"text": "放冬天的棉被", "confidence": 0.95, "timestampMs": 650, "bbox": [150, 880, 400, 915], "category": "DIALOGUE_SUBTITLE"},
            ],
            1: [
                {"text": "第二句对话", "confidence": 0.96, "timestampMs": 1350, "bbox": [150, 880, 400, 915], "category": "DIALOGUE_SUBTITLE"},
                {"text": "第二句对话", "confidence": 0.95, "timestampMs": 1650, "bbox": [150, 880, 400, 915], "category": "DIALOGUE_SUBTITLE"},
            ],
        }

        deepseek_inputs = []
        def deepseek_fake(payload):
            deepseek_inputs.append(payload)
            key = "semanticVi" if payload["mode"] == "semantic" else "nativeVi"
            mapping = {
                "放冬天的棉被": "Cất cái chăn bông mùa đông",
                "第二句对话": "Lời đối thoại thứ hai",
            }
            return {
                "translations": [
                    {"utteranceId": row["utteranceId"], key: mapping.get(row.get("resolvedZh"), "Đoạn dịch")}
                    for row in payload["utterances"]
                ]
            }

        synthesizer_calls = []
        def fake_synthesize(text, voice, target, settings):
            synthesizer_calls.append({"text": text, "voice": voice, "target": target})
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"RIFFdummywavdata")

        def fake_fit_voice(raw, fitted, slot_ms, max_tempo):
            fitted.parent.mkdir(parents=True, exist_ok=True)
            fitted.write_bytes(b"dummy-fitted")
            return (800, False)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            source = output / "source.mp4"
            source.write_bytes(b"dummy-source")
            with patch("dubvi_worker.ffprobe_duration", return_value=2.0), \
                 patch("dubvi_worker.ffprobe_dimensions", return_value=(576, 1024)), \
                 patch("dubvi_worker.extract_audio"), \
                 patch("dubvi_worker.ensure_whisper_model"), \
                 patch("dubvi_worker.transcribe", return_value=asr), \
                 patch("auto_roi.sparse_whole_frame_discovery", return_value=detections), \
                 patch("source_ocr_verifier.infer_stable_subtitle_roi", return_value=(100, 800, 500, 950)), \
                 patch("source_ocr_verifier.scan_cue_ocr", return_value=readings), \
                 patch("cp2_deepseek_adapter.CP2DeepSeekAdapter", return_value=Mock(side_effect=deepseek_fake, last_provider="deepseek", gemini_failure=None)), \
                 patch("dubvi_worker.validate_capcut_voice"), \
                 patch("dubvi_worker.synthesize", side_effect=fake_synthesize), \
                 patch("dubvi_worker.fit_voice", side_effect=fake_fit_voice), \
                 patch("cp2_gemini_adapter.live_cp2_gemini_adapter") as gemini:

                records = stages["source"](source, output)
                self.assertEqual("放冬天的棉被", records[0]["resolvedZh"])
                self.assertEqual("OCR_REPEAT", records[0]["decision"])

                localized = stages["localization"](records, output_dir=output)
                self.assertEqual("Cất cái chăn bông mùa đông", localized[0]["nativeVi"])

                # Verify DeepSeek received OCR_REPEAT resolved Chinese
                first_payload = deepseek_inputs[0]
                self.assertEqual("放冬天的棉被", first_payload["utterances"][0]["resolvedZh"])

                tts_result = stages["tts"](localized, output_dir=output)
                # Verify CapCut BV075_streaming received localized text
                self.assertTrue(any(c["voice"] == "BV075_streaming" and c["text"] == "Cất cái chăn bông mùa đông" for c in synthesizer_calls))

                plans = stages["delivery"](localized, voice_events=tts_result["voiceEvents"])
                timeline = stages["timeline"](tts_result["voiceEvents"], localized, plans)
                self.assertIsNotNone(timeline)

                gemini.assert_not_called()

    def test_unresolved_cue_stops_before_deepseek_and_capcut(self):
        from modern_p1c_adapter import SourceReviewRequiredError

        runtime_config = {
            "localizationProfile": "page_goc_tro_bat_on",
            "sourceEvidenceMode": "local_asr_text_only",
            "sourceOcrVerify": True,
            "page1MaskMode": "none",
            "localizationProvider": "deepseek",
            "ttsProvider": "capcut",
            "voice": {"name": "BV075_streaming"},
        }
        stages = build_modern_stage_bundle(runtime_config=runtime_config)
        asr = [{"position": 0, "startMs": 0, "endMs": 1000, "asrTextZh": "放冬天的免费", "confidence": 50}]
        readings = {0: []}

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            source = output / "source.mp4"
            source.write_bytes(b"dummy-source")
            with patch("dubvi_worker.ffprobe_duration", return_value=1.0), \
                 patch("dubvi_worker.ffprobe_dimensions", return_value=(576, 1024)), \
                 patch("dubvi_worker.extract_audio"), \
                 patch("dubvi_worker.ensure_whisper_model"), \
                 patch("dubvi_worker.transcribe", return_value=asr), \
                 patch("auto_roi.sparse_whole_frame_discovery", return_value=[]), \
                 patch("source_ocr_verifier.infer_stable_subtitle_roi", return_value=(100, 800, 500, 950)), \
                 patch("source_ocr_verifier.scan_cue_ocr", return_value=readings), \
                 patch("cp2_deepseek_adapter.CP2DeepSeekAdapter") as deepseek, \
                 patch("dubvi_worker.synthesize") as synthesize:

                with self.assertRaises(SourceReviewRequiredError):
                    stages["source"](source, output)

                deepseek.assert_not_called()
                synthesize.assert_not_called()

    def test_localization_stage_caches_and_skips_provider_on_warm_run(self):
        stages = build_modern_stage_bundle(runtime_config={
            "localizationProfile": "page_goc_tro_bat_on",
            "sourceEvidenceMode": "local_asr_text_only",
            "page1MaskMode": "none",
            "localizationProvider": "deepseek",
        })
        records = [{
            "utteranceId": "u1", "positions": [0], "startMs": 0, "endMs": 1000,
            "spokenZh": "第一句", "resolvedZh": "第一句", "confidence": "HIGH", "source_confidence": "high",
            "classification": "RESOLVED_DIALOGUE", "type": "DIALOGUE", "flags": [],
        }]

        def answer(payload):
            key = "semanticVi" if payload["mode"] == "semantic" else "nativeVi"
            return {"translations": [{"utteranceId": "u1", key: "Hay là sửa thế này"}]}

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            source = output / "source.mp4"
            source.write_bytes(b"dummy")
            stages["source"] = lambda _s, _o: records

            # First run (cold)
            with patch("cp2_deepseek_adapter.CP2DeepSeekAdapter", return_value=Mock(side_effect=answer, last_provider="deepseek", gemini_failure=None)) as deepseek:
                loc1 = stages["localization"](records, output_dir=output)
                self.assertEqual(1, deepseek.call_count)
                self.assertEqual("Hay là sửa thế này", loc1[0]["nativeVi"])

            # Second run (warm): DeepSeek adapter must NOT be called
            with patch("cp2_deepseek_adapter.CP2DeepSeekAdapter", side_effect=AssertionError("DeepSeek called on warm hit")) as deepseek_warm:
                loc2 = stages["localization"](records, output_dir=output)
                self.assertEqual("Hay là sửa thế này", loc2[0]["nativeVi"])

    def test_changed_chinese_invalidates_localization_cache(self):
        stages = build_modern_stage_bundle(runtime_config={
            "localizationProfile": "page_goc_tro_bat_on",
            "sourceEvidenceMode": "local_asr_text_only",
            "page1MaskMode": "none",
            "localizationProvider": "deepseek",
        })
        records1 = [{
            "utteranceId": "u1", "positions": [0], "startMs": 0, "endMs": 1000,
            "spokenZh": "第一句", "resolvedZh": "第一句", "confidence": "HIGH", "source_confidence": "high",
            "classification": "RESOLVED_DIALOGUE", "type": "DIALOGUE", "flags": [],
        }]
        records2 = [{
            "utteranceId": "u1", "positions": [0], "startMs": 0, "endMs": 1000,
            "spokenZh": "第一句", "resolvedZh": "改变后的句子", "confidence": "HIGH", "source_confidence": "high",
            "classification": "RESOLVED_DIALOGUE", "type": "DIALOGUE", "flags": [],
        }]

        call_count = 0
        def answer(payload):
            nonlocal call_count
            call_count += 1
            key = "semanticVi" if payload["mode"] == "semantic" else "nativeVi"
            return {"translations": [{"utteranceId": "u1", key: f"Bản dịch {call_count}"}]}

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            source = output / "source.mp4"
            source.write_bytes(b"dummy")
            stages["source"] = lambda _s, _o: records1

            # First run (cold with records1)
            with patch("cp2_deepseek_adapter.CP2DeepSeekAdapter", return_value=Mock(side_effect=answer, last_provider="deepseek", gemini_failure=None)):
                loc1 = stages["localization"](records1, output_dir=output)
                self.assertEqual(2, call_count)
                self.assertEqual("Bản dịch 2", loc1[0]["nativeVi"])

            # Second run with records2 (changed Chinese): must invalidate and re-run DeepSeek
            with patch("cp2_deepseek_adapter.CP2DeepSeekAdapter", return_value=Mock(side_effect=answer, last_provider="deepseek", gemini_failure=None)):
                loc2 = stages["localization"](records2, output_dir=output)
                self.assertEqual(4, call_count)
                self.assertEqual("Bản dịch 4", loc2[0]["nativeVi"])

    def test_tts_cache_miss_on_corrupt_or_missing_audio(self):
        from tts_timing import run_tts_timing

        alignment = {
            "dialogue": [{
                "utteranceId": "u1", "sourcePositions": [0], "startMs": 0, "endMs": 1000,
                "nativeVi": "Câu thoại", "approvedNativeVi": "Câu thoại", "deliveryNativeVi": "Câu thoại",
                "semanticStartMs": 0, "semanticEndMs": 1000,
                "ttsEligible": True,
                "reviewRequired": False,
                "alignedCues": [{"position": 0, "startMs": 0, "endMs": 1000, "translatedTextVi": "Câu thoại"}],
            }],
            "excluded": [],
        }

        call_count = 0
        def fake_synthesize(text, voice, target):
            nonlocal call_count
            call_count += 1
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"audio-sample-data")

        def fake_fit_audio(raw, fitted, slot_ms, max_tempo):
            fitted.parent.mkdir(parents=True, exist_ok=True)
            fitted.write_bytes(b"fitted-sample-data")
            return (800, False)

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "audio"
            # Cold run
            res1 = run_tts_timing(
                alignment, output_dir, voice="BV075_streaming", tts_provider="capcut",
                synthesizer=fake_synthesize, fit_audio=fake_fit_audio,
                measure_duration=lambda _p: 800,
            )
            self.assertEqual(1, call_count)
            self.assertEqual(0, res1["cacheHits"])

            # Corrupt audio file (truncate to 0 bytes)
            audio_path = output_dir / res1["voiceEvents"][0]["audioPath"]
            audio_path.write_bytes(b"")

            # Second run: corrupt audio must cause cache miss and re-synthesize
            res2 = run_tts_timing(
                alignment, output_dir, voice="BV075_streaming", tts_provider="capcut",
                synthesizer=fake_synthesize, fit_audio=fake_fit_audio,
                measure_duration=lambda _p: 800,
            )
            self.assertEqual(2, call_count)
            self.assertEqual(0, res2["cacheHits"])


if __name__ == "__main__":
    unittest.main()


