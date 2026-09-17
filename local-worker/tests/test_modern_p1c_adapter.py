from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from canonical_profile import resolve_canonical_profile
from llm_transport import ProviderError
from modern_p1c_adapter import (
    ModernP1CAdapterError,
    _canonical_av_analysis,
    _localization_stage_profile,
    _presentation_render_config,
    build_modern_stage_bundle,
    run_modern_p1c,
)


class ModernP1CAdapterTests(unittest.TestCase):
    def test_delivery_uses_fitted_tts_window_without_fake_tempo(self) -> None:
        stages = build_modern_stage_bundle(
            runtime_config={"localizationProfile": "page_goc_tro_bat_on"}
        )
        records = [{
            "utteranceId": "u1",
            "startMs": 0,
            "endMs": 600,
            "semanticStartMs": 0,
            "semanticEndMs": 600,
            "nativeVi": "Thuê nhà",
            "deliveryNativeVi": "Thuê nhà",
            "sourcePositions": [0],
            "ttsEligible": True,
        }]
        voice_events = [{
            "utteranceId": "u1",
            "baseStartMs": 0,
            "baseEndMs": 600,
            "effectiveStartMs": 0,
            "effectiveEndMs": 959,
            "fittedDurationMs": 888,
            "audioPath": "audio/u1.mp3",
            "nativeViApproved": "Thuê nhà",
            "deliveryNativeVi": "Thuê nhà",
        }]

        plans = stages["delivery"](records, voice_events=voice_events)

        self.assertEqual(1, len(plans))
        self.assertEqual("BORROW_GAP", plans[0]["action"])
        self.assertEqual(288, plans[0]["borrowedGapMs"])
        self.assertEqual(888, plans[0]["effectiveWindowMs"])
        self.assertEqual(1.0, plans[0]["tempo"])

    def test_page_a_text_only_source_uses_local_asr_and_preserves_source_contract(self) -> None:
        stages = build_modern_stage_bundle(
            runtime_config={
                "localizationProfile": "page_goc_tro_bat_on",
                "sourceEvidenceMode": "local_asr_text_only",
                "ocr": {"discoveryFrames": 0},
            }
        )
        asr_rows = [
            {"position": 0, "startMs": 120, "endMs": 860, "asrTextZh": "第一句", "confidence": 87},
            {"position": 1, "startMs": 920, "endMs": 1_700, "asrTextZh": "第二句", "confidence": 87},
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with (
                patch("dubvi_worker.ffprobe_duration", return_value=2.0),
                patch("dubvi_worker.ffprobe_dimensions", return_value=(720, 1280)),
                patch("dubvi_worker.extract_audio") as extract_audio,
                patch("dubvi_worker.transcribe", return_value=asr_rows),
                patch("auto_roi.sparse_whole_frame_discovery", return_value=[]),
                patch("localized_masking.build_mask_plan", return_value=[]),
                patch("av_analyzer.GeminiAVAnalyzer") as av_provider,
                patch("av_analyzer.analyze_media_windows_with_subdivision") as av_call,
                patch("groq_vision.GroqVisionAnalyzer") as groq_provider,
            ):
                records = list(stages["source"](Path("source.mp4"), output))

        extract_audio.assert_called_once()
        av_provider.assert_not_called()
        av_call.assert_not_called()
        groq_provider.assert_not_called()
        self.assertEqual([120, 920], [row["startMs"] for row in records])
        self.assertEqual([860, 1_700], [row["endMs"] for row in records])
        self.assertEqual([[0], [1]], [row["positions"] for row in records])
        self.assertEqual(["第一句", "第二句"], [row["resolvedZh"] for row in records])
        self.assertEqual(2, len({row["utteranceId"] for row in records}))
        self.assertTrue(all(row["classification"] == "RESOLVED_DIALOGUE" for row in records))

    def test_text_translation_boundary_receives_ids_and_source_text_only(self) -> None:
        stages = build_modern_stage_bundle(
            runtime_config={"localizationProfile": "page_goc_tro_bat_on", "sourceEvidenceMode": "local_asr_text_only"}
        )
        records = [
            {
                "utteranceId": "u1",
                "startMs": 100,
                "endMs": 900,
                "resolvedZh": "租房",
                "source_confidence": "high",
                "kind": "dialogue",
                "positions": [4],
            }
        ]
        payloads = []
        source_id = ""

        def request(payload):
            payloads.append(payload)
            key = "semanticVi" if payload["mode"] == "semantic" else "nativeVi"
            return {"translations": [{"utteranceId": source_id, key: "Thuê nhà"}]}

        with tempfile.TemporaryDirectory() as directory, patch("cp2_gemini_adapter.live_cp2_gemini_adapter", return_value=request):
            with (
                patch("dubvi_worker.ffprobe_duration", return_value=2.0),
                patch("dubvi_worker.extract_audio"),
                patch("dubvi_worker.transcribe", return_value=[{"position": 4, "startMs": 100, "endMs": 900, "asrTextZh": "租房", "confidence": 87}]),
                patch("auto_roi.sparse_whole_frame_discovery", return_value=[]),
            ):
                source_records = list(stages["source"](Path("source.mp4"), Path(directory)))
            source_id = str(source_records[0]["utteranceId"])
            stages["localization"](source_records, output_dir=Path(directory))

        self.assertEqual(2, len(payloads))
        for payload in payloads:
            self.assertTrue({"utteranceId", "resolvedZh", "source_confidence", "kind"} <= set(payload["utterances"][0]))
            self.assertNotIn("startMs", payload["utterances"][0])
            self.assertNotIn("endMs", payload["utterances"][0])
            self.assertNotIn("positions", payload["utterances"][0])
            self.assertEqual(source_id, payload["utterances"][0]["utteranceId"])
            self.assertEqual("租房", payload["utterances"][0]["resolvedZh"])

    def test_page_a_localization_uses_groq_text_after_gemini_quota_failure(self) -> None:
        stages = build_modern_stage_bundle(
            runtime_config={"localizationProfile": "page_goc_tro_bat_on", "sourceEvidenceMode": "local_asr_text_only"}
        )
        gemini = Mock(side_effect=ProviderError("HTTP_429", transient=True, http_status=429))
        groq = Mock(side_effect=lambda payload: {
            "translations": [
                {"utteranceId": item["utteranceId"], ("semanticVi" if payload["mode"] == "semantic" else "nativeVi"): "Thuê nhà"}
                for item in payload["utterances"]
            ]
        })
        with tempfile.TemporaryDirectory() as directory, patch("cp2_gemini_adapter.live_cp2_gemini_adapter", return_value=gemini), patch("text_translation_fallback.GroqTextTranslationAdapter", return_value=groq):
            with (
                patch("dubvi_worker.ffprobe_duration", return_value=2.0),
                patch("dubvi_worker.extract_audio"),
                patch("dubvi_worker.transcribe", return_value=[{"position": 4, "startMs": 100, "endMs": 900, "asrTextZh": "租房", "confidence": 87}]),
                patch("auto_roi.sparse_whole_frame_discovery", return_value=[]),
            ):
                source_records = list(stages["source"](Path("source.mp4"), Path(directory)))
            try:
                localized = stages["localization"](source_records, output_dir=Path(directory))
            except ProviderError:
                localized = None

        self.assertIsNotNone(localized)
        self.assertEqual(2, groq.call_count)
        self.assertEqual("Thuê nhà", localized[0]["nativeVi"])

    def test_source_stage_passes_inner_canonical_av_document_to_reconciliation(self) -> None:
        inner = {
            "windowStartMs": 0,
            "windowEndMs": 1_000,
            "scene": {"summary": "scene", "tone": "neutral", "register": "spoken"},
            "utterances": [],
            "visualOnly": [],
        }
        wrapper = {
            "av_analysis": inner,
            "reconstructed": {},
            "diagnostics": {},
            "providerCalls": [],
        }
        stages = build_modern_stage_bundle(runtime_config={"avModel": "gemini-2.5-flash", "ocrDiscoveryFrames": 8})
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("dubvi_worker.ffprobe_duration", return_value=1.0),
                patch("dubvi_worker.ffprobe_dimensions", return_value={"width": 720, "height": 1280}),
                patch("av_analyzer.GeminiAVAnalyzer"),
                patch("av_analyzer.analyze_media_windows_with_subdivision", return_value=wrapper),
                patch("auto_roi.sparse_whole_frame_discovery", return_value=[]),
                patch("localized_masking.build_mask_plan", return_value=None),
                patch("utterance_builder.build_utterances", return_value=[]),
            ):
                records = stages["source"](Path("source.mp4"), Path(directory))
        self.assertEqual([], records)

    def test_canonical_av_accessor_preserves_inner_identity_and_data(self) -> None:
        inner = {"windowStartMs": 0, "windowEndMs": 1_000, "utterances": [], "visualOnly": []}
        self.assertIs(inner, _canonical_av_analysis({"av_analysis": inner}))

    def test_canonical_av_accessor_rejects_missing_or_malformed_inner_document(self) -> None:
        with self.assertRaisesRegex(ModernP1CAdapterError, "av_analysis"):
            _canonical_av_analysis({"diagnostics": {}})
        with self.assertRaisesRegex(ModernP1CAdapterError, "av_analysis"):
            _canonical_av_analysis({"av_analysis": []})

    def test_canonical_page_profiles_choose_explicit_localization_stage_profiles(self) -> None:
        self.assertEqual("vn_native_light_comedy", _localization_stage_profile({"localizationProfile": "page_goc_tro_bat_on"}))
        self.assertEqual("vn_native_light_comedy", _localization_stage_profile({"localizationProfile": "page_giai_cuu_chuong_lon"}))
        self.assertEqual("neutral_faithful", _localization_stage_profile({"localizationProfile": "loc-v1"}))

    def test_page_profiles_reach_modern_presentation_render_configuration(self) -> None:
        page_a = resolve_canonical_profile({"localization_profile": "page_goc_tro_bat_on", "target_platform": "tiktok"})
        page_b = resolve_canonical_profile({"localization_profile": "page_giai_cuu_chuong_lon", "target_platform": "tiktok"})
        self.assertEqual("#00FFFF", _presentation_render_config(page_a)["presentation_color"])
        self.assertEqual("#FFE600", _presentation_render_config(page_b)["presentation_color"])

    def test_p1c_fallback_keeps_legacy_graph_reachable_without_modern_stages(self) -> None:
        import p1c_processing

        with patch.object(p1c_processing, "_legacy_process_job") as legacy:
            p1c_processing._modern_process_job(object(), object(), {"id": "job"})
        legacy.assert_called_once()

    def test_injected_stage_bundle_selects_modern_adapter(self) -> None:
        import p1c_processing

        profile = resolve_canonical_profile({"localization_profile": "page_giai_cuu_chuong_lon", "target_platform": "tiktok"})
        stages = {name: (lambda *args, **kwargs: None) for name in ("source", "localization", "tts", "delivery", "timeline", "subtitle")}
        with patch("modern_p1c_adapter.run_modern_p1c", return_value={}) as modern, patch.object(p1c_processing, "_legacy_process_job") as legacy:
            p1c_processing._modern_process_job(
                object(), object(), {
                    "id": "job", "handoffId": "handoff", "sourcePath": "source.mp4",
                    "outputDir": "out", "modernProfile": profile, "modernStages": stages,
                }
            )
        modern.assert_called_once()
        legacy.assert_not_called()

    def test_adapter_composes_identity_safe_modern_stages_without_rendering(self) -> None:
        calls: list[str] = []
        profile = resolve_canonical_profile({"localization_profile": "page_goc_tro_bat_on", "target_platform": "tiktok"})
        identity = {"translator_job_id": "job-1", "handoff_id": "handoff-1"}

        def source_stage(source: Path, output: Path, **kwargs):
            calls.append("source")
            return [{"utteranceId": "u1", "resolvedZh": "你好", "startMs": 0, "endMs": 1000, "visualOnly": False}]

        def localization_stage(records, **kwargs):
            calls.append("localization")
            return [{**records[0], "nativeVi": "Xin chào", "ttsEligible": True, "semanticStartMs": 0, "semanticEndMs": 1000, "sourcePositions": [0]}]

        def tts_stage(records, output, **kwargs):
            calls.append("tts")
            return {"voiceEvents": [{"utteranceId": "u1", "audioPath": "audio/u1.mp3", "baseStartMs": 0, "baseEndMs": 1000, "effectiveStartMs": 0, "effectiveEndMs": 1000, "fittedDurationMs": 500, "nativeViApproved": "Xin chào", "deliveryNativeVi": "Xin chào"}]}

        def delivery_stage(records, **kwargs):
            calls.append("delivery")
            return [{"utteranceId": "u1", "sourceStartMs": 0, "sourceEndMs": 1000, "action": "KEEP", "tempo": 1.0, "borrowedGapMs": 0}]

        def timeline_stage(voices, cues, plans, **kwargs):
            calls.append("timeline")
            return {"voiceEvents": voices, "subtitleCues": cues, "plans": plans}

        def subtitle_stage(cues, **kwargs):
            calls.append("subtitle")
            return {"cues": cues, "maxVisibleLines": 2}

        result = run_modern_p1c(
            Path("source.mp4"), Path("out"), identity, profile,
            source_stage=source_stage, localization_stage=localization_stage,
            tts_stage=tts_stage, delivery_stage=delivery_stage,
            timeline_stage=timeline_stage, subtitle_stage=subtitle_stage,
        )
        self.assertEqual(["source", "localization", "tts", "delivery", "timeline", "subtitle"], calls)
        self.assertEqual(identity, result["identity"])
        self.assertEqual("u1", result["timeline"]["voiceEvents"][0]["utteranceId"])
        self.assertEqual(2, result["subtitles"]["maxVisibleLines"])

    def test_failed_tts_event_is_rejected_before_delivery_can_convert_none(self) -> None:
        profile = resolve_canonical_profile({"localization_profile": "page_goc_tro_bat_on", "target_platform": "tiktok"})
        identity = {"translator_job_id": "job-1", "handoff_id": "handoff-1"}
        failed_event = {
            "utteranceId": "u1",
            "reviewRequired": True,
            "reviewReason": "TTS_FAILED",
            "audioPath": None,
            "fittedDurationMs": None,
        }
        delivery_called = False

        def delivery_stage(records, **kwargs):
            nonlocal delivery_called
            delivery_called = True
            measured = kwargs["voice_events"][0].get("fittedDurationMs", 0)
            return [{"utteranceId": records[0]["utteranceId"], "ttsMeasuredMs": int(measured)}]

        with self.assertRaisesRegex(ModernP1CAdapterError, "TTS_FAILED.*u1"):
            run_modern_p1c(
                Path("source.mp4"), Path("out"), identity, profile,
                source_stage=lambda *args, **kwargs: [{"utteranceId": "u1"}],
                localization_stage=lambda records, **kwargs: [{**records[0], "nativeVi": "Thuê nhà", "ttsEligible": True}],
                tts_stage=lambda *args, **kwargs: {"voiceEvents": [failed_event]},
                delivery_stage=delivery_stage,
                timeline_stage=lambda *args, **kwargs: {},
                subtitle_stage=lambda *args, **kwargs: {},
            )
        self.assertFalse(delivery_called)

    def test_success_event_requires_audio_artifact_metadata(self) -> None:
        profile = resolve_canonical_profile({"localization_profile": "page_goc_tro_bat_on", "target_platform": "tiktok"})
        identity = {"translator_job_id": "job-1", "handoff_id": "handoff-1"}
        malformed_success = {
            "utteranceId": "u1",
            "reviewRequired": False,
            "reviewReason": None,
            "audioPath": "audio/u1.mp3",
            "fittedDurationMs": None,
        }
        with self.assertRaisesRegex(ModernP1CAdapterError, "renderable.*u1"):
            run_modern_p1c(
                Path("source.mp4"), Path("out"), identity, profile,
                source_stage=lambda *args, **kwargs: [{"utteranceId": "u1"}],
                localization_stage=lambda records, **kwargs: [{**records[0], "nativeVi": "Xin chào", "ttsEligible": True}],
                tts_stage=lambda *args, **kwargs: {"voiceEvents": [malformed_success]},
                delivery_stage=lambda *args, **kwargs: [],
                timeline_stage=lambda *args, **kwargs: {},
                subtitle_stage=lambda *args, **kwargs: {},
            )

    def test_valid_tts_events_reach_timeline_once_per_spoken_utterance(self) -> None:
        profile = resolve_canonical_profile({"localization_profile": "page_goc_tro_bat_on", "target_platform": "tiktok"})
        identity = {"translator_job_id": "job-1", "handoff_id": "handoff-1"}
        records = [
            {"utteranceId": "u1", "nativeVi": "Một", "ttsEligible": True},
            {"utteranceId": "u2", "nativeVi": "Hai", "ttsEligible": True},
        ]
        events = [
            {"utteranceId": "u1", "audioPath": "audio/u1.mp3", "fittedDurationMs": 100},
            {"utteranceId": "u2", "audioPath": "audio/u2.mp3", "fittedDurationMs": 200},
        ]
        timeline_inputs: list[list[str]] = []

        def timeline_stage(voices, cues, plans, **kwargs):
            timeline_inputs.append([str(event["utteranceId"]) for event in voices])
            return {"voiceEvents": voices, "subtitleCues": cues}

        result = run_modern_p1c(
            Path("source.mp4"), Path("out"), identity, profile,
            source_stage=lambda *args, **kwargs: records,
            localization_stage=lambda records, **kwargs: records,
            tts_stage=lambda *args, **kwargs: {"voiceEvents": events},
            delivery_stage=lambda records, **kwargs: [{"utteranceId": item["utteranceId"]} for item in records],
            timeline_stage=timeline_stage,
            subtitle_stage=lambda cues, **kwargs: {"outputPath": "out.mp4"},
        )
        self.assertEqual([["u1", "u2"]], timeline_inputs)
        self.assertEqual(["u1", "u2"], [event["utteranceId"] for event in result["timeline"]["voiceEvents"]])


if __name__ == "__main__":
    unittest.main()
