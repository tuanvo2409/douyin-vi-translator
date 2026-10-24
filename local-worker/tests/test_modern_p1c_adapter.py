from __future__ import annotations

import json
import os
import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from canonical_profile import resolve_canonical_profile
from llm_transport import ProviderError
from modern_p1c_adapter import (
    ModernP1CAdapterError,
    SourceReviewRequiredError,
    _page2_placement_evidence,
    _page1_mask_geometry,
    _canonical_av_analysis,
    _localization_stage_profile,
    _presentation_render_config,
    _subtitle_cues,
    build_modern_stage_bundle,
    run_modern_p1c,
)
from timeline_mapper import TimelineMap


class ModernP1CAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self._ensure_whisper_model = patch("dubvi_worker.ensure_whisper_model", return_value=object())
        self._ensure_whisper_model.start()
        self.addCleanup(self._ensure_whisper_model.stop)

    def test_modern_p1c_rejects_an_incomplete_identity_chain_before_stages(self) -> None:
        profile = resolve_canonical_profile({"localization_profile": "loc-v1", "target_platform": "tiktok"})
        identity = {
            "translator_job_id": "job-1", "dispatch_id": "dispatch-1",
            "reup_job_id": "reup-1", "handoff_id": "handoff-1",
            "schedule_id": "schedule-1",
        }
        def source_stage(source, output, **kwargs):
            return [{"utteranceId": "u1", "startMs": 0, "endMs": 1000}]

        def localization_stage(records, **kwargs):
            return [{**records[0], "nativeVi": "Xin chào", "ttsEligible": True}]

        def tts_stage(records, output, **kwargs):
            return {"voiceEvents": [{
                "utteranceId": "u1", "audioPath": "audio/u1.mp3",
                "fittedDurationMs": 500,
            }]}

        for supplied in (identity, {"translator_job_id": "job-1", "handoff_id": "handoff-1"}):
            with self.subTest(supplied=supplied), self.assertRaisesRegex(
                ModernP1CAdapterError, "identity is incomplete"
            ):
                run_modern_p1c(
                    Path("source.mp4"), Path("out"), supplied, profile,
                    source_stage=source_stage, localization_stage=localization_stage,
                    tts_stage=tts_stage,
                    delivery_stage=lambda records, **kwargs: [{"utteranceId": "u1"}],
                    timeline_stage=lambda *args, **kwargs: {},
                    subtitle_stage=lambda *args, **kwargs: {},
                )

    def _run_subtitle_with_audio_path(self, audio_path: str, output_dir: Path):
        stages = build_modern_stage_bundle(runtime_config={
            "localizationProfile": "loc-v1",
            "sourceEvidenceMode": "local_asr_text_only",
            "page1MaskMode": "none",
        })
        with patch("subtitle_layout.write_ass_subtitles", return_value={}):
            with patch("dubvi_worker.ffprobe_duration", return_value=1.0), \
                 patch("dubvi_worker.ffprobe_dimensions", return_value=(576, 1152)), \
                 patch("dubvi_worker.extract_audio"), \
                 patch("dubvi_worker.transcribe", return_value=[{
                     "position": 0, "startMs": 0, "endMs": 800,
                     "asrTextZh": "源句", "confidence": 90,
                 }]), \
                 patch("auto_roi.sparse_whole_frame_discovery", return_value=[]):
                stages["source"](Path("source.mp4"), output_dir)
            timeline = {
                "schemaVersion": 1, "artifact": "adaptive_retime_timeline_v1",
                "sourceDurationMs": 1000, "outputDurationMs": 1000,
                "segments": [{
                    "sourceStartMs": 0, "sourceEndMs": 1000,
                    "outputStartMs": 0, "outputEndMs": 1000,
                    "speed": 1.0, "reason": "KEEP", "utteranceIds": ["u1"],
                }],
                "voiceEvents": [{"utteranceId": "u1", "audioPath": audio_path, "scheduledStartMs": 0}],
                "subtitleCues": [],
            }
            with patch("adaptive_renderer.build_adaptive_timeline", return_value=timeline):
                stages["timeline"]([], [], [])
            stages["subtitle"]([], profile=resolve_canonical_profile({"localization_profile": "loc-v1", "target_platform": "tiktok"}))
    def test_audio_paths_outside_attempt_never_reach_ffmpeg(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            enclosing = Path(directory)
            root = enclosing / "attempt"
            root.mkdir()
            outside = enclosing / "outside.mp3"
            outside.write_bytes(b"audio")
            absolute = str(outside)
            for candidate in ("../" + outside.name, absolute):
                with self.subTest(candidate=candidate):
                    with patch("adaptive_renderer.render_adaptive") as render:
                        with patch("adaptive_renderer.build_adaptive_ffmpeg_command", return_value=["ffmpeg"]) as build:
                            with self.assertRaises(ModernP1CAdapterError):
                                self._run_subtitle_with_audio_path(candidate, root)
                            build.assert_not_called()
                        render.assert_not_called()

    def test_owned_audio_path_reaches_ffmpeg_builder_as_contained_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "attempt"
            (root / "audio").mkdir(parents=True)
            owned = root / "audio" / "u1.mp3"
            owned.write_bytes(b"audio")
            with patch("adaptive_renderer.render_adaptive") as render, \
                 patch("adaptive_renderer.build_adaptive_ffmpeg_command", return_value=["ffmpeg"]) as build:
                self._run_subtitle_with_audio_path("audio/u1.mp3", root)
            voice_inputs = build.call_args.args[3]
            self.assertEqual(owned, voice_inputs[0][0])
            render.assert_called_once_with(["ffmpeg"])

    def test_symlink_audio_escape_never_reaches_ffmpeg(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            enclosing = Path(directory)
            root = enclosing / "attempt"
            root.mkdir()
            audio_dir = root / "audio"
            audio_dir.mkdir()
            outside = enclosing / "outside.mp3"
            outside.write_bytes(b"audio")
            link = audio_dir / "u1.mp3"
            try:
                os.symlink(outside, link)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")
            with patch("adaptive_renderer.render_adaptive") as render:
                with patch("adaptive_renderer.build_adaptive_ffmpeg_command", return_value=["ffmpeg"]) as build:
                    with self.assertRaises(ModernP1CAdapterError):
                        self._run_subtitle_with_audio_path("audio/u1.mp3", root)
                    build.assert_not_called()
                render.assert_not_called()
    def test_av_conflict_is_review_required_before_semantic_request(self) -> None:
        stages = build_modern_stage_bundle(
            runtime_config={
                "localizationProfile": "page_goc_tro_bat_on",
                "sourceEvidenceMode": "local_asr_text_only",
                "page1MaskMode": "none",
            }
        )
        records = [{
            "utteranceId": "u-av-conflict",
            "startMs": 100,
            "endMs": 900,
            "resolvedZh": "甲",
            "source_confidence": "high",
            "classification": "RESOLVED_DIALOGUE",
            "kind": "dialogue",
            "flags": ["AV_CONFLICT"],
            "decision": "AV_CONFLICT",
            "evidence": {"avSpoken": "甲", "avSubtitle": "乙"},
        }]
        def provider_response(payload):
            key = "semanticVi" if payload["mode"] == "semantic" else "nativeVi"
            return {
                "translations": [
                    {"utteranceId": item["utteranceId"], key: "Thuê nhà"}
                    for item in payload["utterances"]
                ]
            }

        provider = Mock(side_effect=provider_response)

        with tempfile.TemporaryDirectory() as directory, patch(
            "cp2_gemini_adapter.live_cp2_gemini_adapter", return_value=provider
        ):
            self._prepare_page1_output_dir(stages, Path(directory))
            with self.assertRaisesRegex(SourceReviewRequiredError, "SOURCE_REVIEW_REQUIRED"):
                stages["localization"](records, output_dir=Path(directory))

        self.assertEqual(0, provider.call_count)

    def test_resolved_dialogue_keeps_source_evidence_and_translates(self) -> None:
        stages = build_modern_stage_bundle(
            runtime_config={
                "localizationProfile": "page_goc_tro_bat_on",
                "sourceEvidenceMode": "local_asr_text_only",
                "page1MaskMode": "none",
            }
        )
        records = [{
            "utteranceId": "u-resolved",
            "startMs": 100,
            "endMs": 900,
            "resolvedZh": "租房",
            "source_confidence": "high",
            "classification": "RESOLVED_DIALOGUE",
            "kind": "dialogue",
            "positions": [4],
            "flags": [],
            "decision": "AV_SPOKEN_SUBTITLE_AGREE",
            "evidence": {"asr": "租房", "ocr": "租房", "avSpoken": "租房", "avSubtitle": "租房"},
        }]

        def request(payload):
            key = "semanticVi" if payload["mode"] == "semantic" else "nativeVi"
            return {"translations": [{"utteranceId": "u-resolved", key: "Thuê nhà"}]}

        with tempfile.TemporaryDirectory() as directory, patch(
            "cp2_gemini_adapter.live_cp2_gemini_adapter", return_value=request
        ):
            self._prepare_page1_output_dir(stages, Path(directory))
            localized = stages["localization"](records, output_dir=Path(directory))
            alignment = json.loads((Path(directory) / "cue_alignment_v3.json").read_text(encoding="utf-8"))

        self.assertEqual("Thuê nhà", localized[0]["nativeVi"])
        row = alignment["dialogue"][0]
        self.assertEqual("u-resolved", row["utteranceId"])
        self.assertEqual([4], row["sourcePositions"])
        self.assertEqual("AV_SPOKEN_SUBTITLE_AGREE", row.get("decision"))
        self.assertEqual(
            {"asr": "租房", "ocr": "租房", "avSpoken": "租房", "avSubtitle": "租房"},
            row.get("evidence"),
        )

    def test_page2_placement_evidence_matches_typed_source_and_scales_bbox(self) -> None:
        detections = [
            {
                "timestampMs": 300, "bbox": (150, 300, 750, 400), "text": "　你好，",
                "category": "DIALOGUE_SUBTITLE", "confidence": 0.95, "trackId": "ocr-1",
            },
            {
                "timestampMs": 300, "bbox": (10, 300, 90, 400), "text": "你好",
                "category": "UNKNOWN", "confidence": 0.99,
            },
            {
                "timestampMs": 300, "bbox": (10, 300, 90, 400), "text": "别的内容",
                "category": "DIALOGUE_SUBTITLE", "confidence": 0.99,
            },
            {
                "timestampMs": 300, "bbox": (10, 300, 90, 400), "text": "你好",
                "category": "DIALOGUE_SUBTITLE", "confidence": 0.5,
            },
        ]
        dialogue = [{"startMs": 250, "endMs": 350, "subtitleZh": "你好"}]

        evidence = _page2_placement_evidence(detections, dialogue, (1080, 1920))

        self.assertEqual(1, len(evidence))
        self.assertEqual("DIALOGUE_SUBTITLE", evidence[0]["category"])
        self.assertEqual("ocr-1", evidence[0]["trackId"])
        self.assertEqual((250, 350), (evidence[0]["startMs"], evidence[0]["endMs"]))
        self.assertEqual([100, 200, 500, 267], evidence[0]["bbox"])

    def test_page2_placement_evidence_promotes_matching_sparse_unknown_discovery(self) -> None:
        from auto_roi import discover_cjk_regions

        detections = discover_cjk_regions([{
            "timestampMs": 300,
            "items": [
                {"bbox": (100, 300, 300, 350), "text": "你好", "confidence": 0.91, "category": "UNKNOWN"},
                {"bbox": (800, 300, 900, 350), "text": "你好", "confidence": 0.55, "category": "UNKNOWN"},
            ],
        }], (1080, 1920))
        dialogue = [{"startMs": 250, "endMs": 350, "subtitleZh": "你好"}]

        evidence = _page2_placement_evidence(detections, dialogue, (1080, 1920))

        self.assertEqual(1, len(evidence))
        self.assertEqual("DIALOGUE_SUBTITLE", evidence[0]["category"])
        self.assertEqual("你好", evidence[0]["subtitleZh"])
        self.assertEqual("HIGH", evidence[0]["confidence"])

    def test_all_low_confidence_page1_source_requires_review_before_translation(self) -> None:
        stages = build_modern_stage_bundle(
            runtime_config={"localizationProfile": "page_goc_tro_bat_on", "sourceEvidenceMode": "local_asr_text_only", "page1MaskMode": "none"}
        )
        records = [{
            "utteranceId": "u1", "startMs": 100, "endMs": 900,
            "resolvedZh": "租房", "source_confidence": "low",
            "classification": "UNRESOLVED_DIALOGUE", "kind": "dialogue",
        }]
        def provider_response(payload):
            key = "semanticVi" if payload["mode"] == "semantic" else "nativeVi"
            return {
                "translations": [
                    {"utteranceId": item["utteranceId"], key: "Thuê nhà"}
                    for item in payload["utterances"]
                ]
            }

        provider = Mock(side_effect=provider_response)
        with tempfile.TemporaryDirectory() as directory, patch(
            "cp2_gemini_adapter.live_cp2_gemini_adapter", return_value=provider
        ):
            self._prepare_page1_output_dir(stages, Path(directory))
            with self.assertRaisesRegex(SourceReviewRequiredError, "SOURCE_REVIEW_REQUIRED"):
                stages["localization"](records, output_dir=Path(directory))
            self.assertEqual(0, provider.call_count)

    def _prepare_page1_output_dir(self, stages, output_dir: Path) -> None:
        with patch("dubvi_worker.ffprobe_duration", return_value=1.0), \
             patch("dubvi_worker.extract_audio"), \
             patch("dubvi_worker.transcribe", return_value=[{
                 "position": 0, "startMs": 0, "endMs": 800, "asrTextZh": "源句", "confidence": 90,
             }]), \
             patch("auto_roi.sparse_whole_frame_discovery", return_value=[]):
            stages["source"](Path("source.mp4"), output_dir)

    def test_page2_subtitle_stage_uses_matched_evidence_and_accepted_timeline(self) -> None:
        profile = resolve_canonical_profile({"localization_profile": "page_goc_tro_bat_on", "target_platform": "tiktok"})
        stages = build_modern_stage_bundle(runtime_config={"localizationProfile": "page_goc_tro_bat_on"})
        detections = [{
            "timestampMs": 300,
            "bbox": [150, 300, 750, 400],
            "text": "你好",
            "category": "DIALOGUE_SUBTITLE",
            "confidence": 0.95,
            "trackId": "ocr-u1",
        }]
        dialogue = [{"utteranceId": "u1", "startMs": 250, "endMs": 350, "subtitleZh": "你好"}]
        source_analysis = {"av_analysis": {"windowStartMs": 0, "windowEndMs": 1_000, "scene": {}, "utterances": [], "visualOnly": []}}
        captured: dict[str, object] = {}

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            audio_path = output_dir / "audio" / "u1.mp3"
            audio_path.parent.mkdir(parents=True)
            audio_path.write_bytes(b"owned test audio")
            with patch("dubvi_worker.ffprobe_duration", return_value=1.0), \
                 patch("dubvi_worker.ffprobe_dimensions", return_value=(1080, 1920)), \
                 patch("av_analyzer.GeminiAVAnalyzer", return_value=object()), \
                 patch("av_analyzer.analyze_media_windows_with_subdivision", return_value=source_analysis), \
                 patch("auto_roi.sparse_whole_frame_discovery", return_value=detections), \
                 patch("localized_masking.build_mask_plan", return_value=[]), \
                 patch("source_reconciler.reconcile_source", return_value={"dialogue": [], "visualOnly": []}), \
                 patch("utterance_builder.build_utterances", return_value=dialogue):
                stages["source"](Path("source.mp4"), output_dir)

            timeline = {
                "schemaVersion": 1,
                "artifact": "adaptive_retime_timeline_v1",
                "sourceDurationMs": 1_000,
                "outputDurationMs": 2_000,
                "segments": [{
                    "sourceStartMs": 0, "sourceEndMs": 1_000,
                    "outputStartMs": 0, "outputEndMs": 2_000,
                    "speed": 0.5, "reason": "SLOW_VISUAL", "utteranceIds": ["u1"],
                }],
                "voiceEvents": [{"utteranceId": "u1", "audioPath": "audio/u1.mp3", "scheduledStartMs": 200}],
                "subtitleCues": [],
            }
            with patch("adaptive_renderer.build_adaptive_timeline", return_value=timeline):
                stages["timeline"]([], [], [])

            def capture_layout(_cues, _path, **kwargs):
                captured.update(kwargs)
                return {"displayCues": []}

            with patch("subtitle_layout.write_ass_subtitles", side_effect=capture_layout), \
                 patch("adaptive_renderer.build_adaptive_ffmpeg_command", return_value=["ffmpeg"]), \
                 patch("adaptive_renderer.render_adaptive"):
                stages["subtitle"]([], profile=profile)

        self.assertEqual(1, len(captured["placement_evidence"]))
        self.assertEqual([100, 200, 500, 267], captured["placement_evidence"][0]["bbox"])
        accepted_timeline = captured["source_to_output"]
        self.assertIsInstance(accepted_timeline, TimelineMap)
        self.assertEqual((500, 700), accepted_timeline.map_interval(250, 350))

    def test_timing_review_keeps_subtitle_for_voice_that_is_rendered(self) -> None:
        events = [{
            "utteranceId": "u1", "baseStartMs": 0, "baseEndMs": 1000,
            "deliveryNativeVi": "Câu này vẫn có giọng.",
            "reviewRequired": True, "reviewReason": "TTS_TIMING_OVERFLOW",
        }]
        cues = _subtitle_cues(events)
        self.assertEqual(1, len(cues))
        self.assertEqual("Câu này vẫn có giọng.", cues[0]["translatedTextVi"])

    def test_page1_static_mask_covers_widest_subtitle_in_stable_lower_band(self) -> None:
        detections = [
            {"timestampMs": 0, "bbox": (33, 88, 538, 315), "text": "故事", "category": "STORY_TEXT"},
            {"timestampMs": 22_800, "bbox": (189, 880, 387, 917), "text": "第一句", "category": "DIALOGUE_SUBTITLE"},
            {"timestampMs": 45_633, "bbox": (256, 878, 319, 920), "text": "第二句", "category": "DIALOGUE_SUBTITLE"},
            {"timestampMs": 68_466, "bbox": (130, 884, 448, 914), "text": "最长的一句字幕", "category": "DIALOGUE_SUBTITLE"},
        ]

        bbox, evidence_ms = _page1_mask_geometry(detections, (576, 1152))

        self.assertLessEqual(bbox[0], 130)
        self.assertGreaterEqual(bbox[2], 448)
        self.assertLessEqual(bbox[1], 878)
        self.assertGreaterEqual(bbox[3], 920)
        self.assertEqual(22_800, evidence_ms)

    def test_page1_mask_requires_explicit_dialogue_classification(self) -> None:
        detections = [
            {"timestampMs": 22_800, "bbox": (189, 880, 387, 917), "text": "静态标题", "category": "STORY_TEXT"},
            {"timestampMs": 45_633, "bbox": (256, 878, 319, 920), "text": "固定水印", "category": "WATERMARK"},
        ]

        with self.assertRaisesRegex(ModernP1CAdapterError, "MASK_REVIEW_REQUIRED"):
            _page1_mask_geometry(detections, (576, 1152))

    def test_page1_mask_accepts_repeated_dialogue_samples_with_bounded_stationary_box(self) -> None:
        detections = [
            {"timestampMs": 700, "bbox": (180, 880, 388, 917), "text": "第一句", "category": "DIALOGUE_SUBTITLE", "confidence": 0.91},
            {"timestampMs": 1_900, "bbox": (205, 878, 430, 920), "text": "第二句", "category": "DIALOGUE_SUBTITLE", "confidence": 0.93},
        ]

        bbox, evidence_ms = _page1_mask_geometry(detections, (576, 1152))

        self.assertEqual(700, evidence_ms)
        self.assertLessEqual(bbox[0], 180)
        self.assertGreaterEqual(bbox[2], 430)
        self.assertGreaterEqual(bbox[1], 840)
        self.assertLessEqual(bbox[3], 960)
        self.assertLessEqual((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 576 * 1152 // 4)

    def test_page1_unmatched_dialogue_ocr_requires_review_before_masking(self) -> None:
        stages = build_modern_stage_bundle(runtime_config={
            "localizationProfile": "page_goc_tro_bat_on",
            "sourceEvidenceMode": "local_asr_text_only",
        })
        asr_rows = [{"position": 0, "startMs": 0, "endMs": 700, "asrTextZh": "只有这一句口语", "confidence": 90}]
        ocr_detections = [{
            "timestampMs": 2_000, "bbox": (180, 880, 388, 917), "text": "画面中未对应的字幕",
            "category": "DIALOGUE_SUBTITLE", "confidence": 0.91,
        }]

        with tempfile.TemporaryDirectory() as directory:
            with patch("dubvi_worker.ffprobe_duration", return_value=3.0), \
                 patch("dubvi_worker.ffprobe_dimensions", return_value=(576, 1152)), \
                 patch("dubvi_worker.extract_audio"), \
                 patch("dubvi_worker.transcribe", return_value=asr_rows), \
                 patch("auto_roi.sparse_whole_frame_discovery", return_value=ocr_detections):
                with self.assertRaisesRegex(ModernP1CAdapterError, "MASK_REVIEW_REQUIRED"):
                    stages["source"](Path("source.mp4"), Path(directory))

    def test_unmatched_dialogue_ocr_persists_visual_gap_with_nested_mask_diagnostic(self) -> None:
        stages = build_modern_stage_bundle(runtime_config={
            "localizationProfile": "page_goc_tro_bat_on",
            "sourceEvidenceMode": "local_asr_text_only",
        })
        asr_rows = [{"position": 0, "startMs": 0, "endMs": 700, "asrTextZh": "只有这一句口语", "confidence": 90}]
        ocr_detections = [{
            "timestampMs": 2_000, "bbox": (180, 880, 388, 917), "text": "画面中未对应的字幕",
            "category": "DIALOGUE_SUBTITLE", "confidence": 0.91,
        }]

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch("dubvi_worker.ffprobe_duration", return_value=3.0), \
                 patch("dubvi_worker.ffprobe_dimensions", return_value=(576, 1152)), \
                 patch("dubvi_worker.extract_audio"), \
                 patch("dubvi_worker.transcribe", return_value=asr_rows), \
                 patch("auto_roi.sparse_whole_frame_discovery", return_value=ocr_detections):
                with self.assertRaisesRegex(ModernP1CAdapterError, "(?=.*SOURCE_EVIDENCE_UNRESOLVED)(?=.*MASK_REVIEW_REQUIRED)"):
                    stages["source"](Path("source.mp4"), output)
            evidence = json.loads((output / "source_evidence.json").read_text(encoding="utf-8"))

        self.assertEqual("local_asr_text_only", evidence["mode"])
        self.assertEqual(1, len(evidence["unresolvedVisualDialogue"]))
        self.assertEqual(2_000, evidence["unresolvedVisualDialogue"][0]["timestampMs"])
        self.assertEqual("DIALOGUE_SUBTITLE", evidence["unresolvedVisualDialogue"][0]["classification"])

    def test_page1_story_title_and_watermark_evidence_never_auto_masks(self) -> None:
        stages = build_modern_stage_bundle(runtime_config={
            "localizationProfile": "page_goc_tro_bat_on",
            "sourceEvidenceMode": "local_asr_text_only",
        })
        asr_rows = [{"position": 0, "startMs": 0, "endMs": 3_000, "asrTextZh": "背景口语", "confidence": 90}]
        ocr_detections = [
            {"timestampMs": 700, "bbox": (150, 875, 430, 920), "text": "剧情标题", "category": "STORY_TEXT", "confidence": 0.95},
            {"timestampMs": 1_900, "bbox": (160, 878, 420, 918), "text": "频道水印", "category": "WATERMARK", "confidence": 0.95},
        ]

        with tempfile.TemporaryDirectory() as directory:
            with patch("dubvi_worker.ffprobe_duration", return_value=3.0), \
                 patch("dubvi_worker.ffprobe_dimensions", return_value=(576, 1152)), \
                 patch("dubvi_worker.extract_audio"), \
                 patch("dubvi_worker.transcribe", return_value=asr_rows), \
                 patch("auto_roi.sparse_whole_frame_discovery", return_value=ocr_detections):
                with self.assertRaisesRegex(ModernP1CAdapterError, "MASK_REVIEW_REQUIRED"):
                    stages["source"](Path("source.mp4"), Path(directory))

    def test_page1_repeated_dialogue_ocr_overlapping_distinct_asr_cues_promotes_to_mask(self) -> None:
        stages = build_modern_stage_bundle(runtime_config={
            "localizationProfile": "page_goc_tro_bat_on",
            "sourceEvidenceMode": "local_asr_text_only",
        })
        asr_rows = [
            {"position": 0, "startMs": 600, "endMs": 1_000, "asrTextZh": "第一句口语", "confidence": 90},
            {"position": 1, "startMs": 1_700, "endMs": 2_100, "asrTextZh": "第二句口语", "confidence": 90},
        ]
        ocr_detections = [
            {"timestampMs": 700, "bbox": (180, 880, 388, 917), "text": "第一句字幕", "category": "UNKNOWN", "confidence": 0.91},
            {"timestampMs": 1_900, "bbox": (205, 878, 430, 920), "text": "第二句字幕", "category": "UNKNOWN", "confidence": 0.93},
        ]

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            with patch("dubvi_worker.ffprobe_duration", return_value=3.0), \
                 patch("dubvi_worker.ffprobe_dimensions", return_value=(576, 1152)), \
                 patch("dubvi_worker.extract_audio"), \
                 patch("dubvi_worker.transcribe", return_value=asr_rows), \
                 patch("auto_roi.sparse_whole_frame_discovery", return_value=ocr_detections):
                records = list(stages["source"](Path("source.mp4"), output_dir))
            mask_overlay_written = (output_dir / "page1_mask_overlay.png").is_file()

        self.assertEqual(2, len(records))
        self.assertTrue(mask_overlay_written)

    def test_page1_same_text_static_unknown_lower_title_is_not_promoted(self) -> None:
        stages = build_modern_stage_bundle(runtime_config={
            "localizationProfile": "page_goc_tro_bat_on",
            "sourceEvidenceMode": "local_asr_text_only",
        })
        asr_rows = [
            {"position": 0, "startMs": 600, "endMs": 1_000, "asrTextZh": "第一句口语", "confidence": 90},
            {"position": 1, "startMs": 1_700, "endMs": 2_100, "asrTextZh": "第二句口语", "confidence": 90},
        ]
        ocr_detections = [
            {"timestampMs": 700, "bbox": (180, 880, 388, 917), "text": "固定标题", "category": "UNKNOWN", "confidence": 0.91},
            {"timestampMs": 1_900, "bbox": (205, 878, 430, 920), "text": "固定标题", "category": "UNKNOWN", "confidence": 0.93},
        ]
        with tempfile.TemporaryDirectory() as directory:
            with patch("dubvi_worker.ffprobe_duration", return_value=3.0), patch("dubvi_worker.ffprobe_dimensions", return_value=(576, 1152)), patch("dubvi_worker.extract_audio"), patch("dubvi_worker.transcribe", return_value=asr_rows), patch("auto_roi.sparse_whole_frame_discovery", return_value=ocr_detections):
                with self.assertRaisesRegex(ModernP1CAdapterError, "MASK_REVIEW_REQUIRED"):
                    stages["source"](Path("source.mp4"), Path(directory))

    def test_page1_upper_story_text_is_ignored_without_becoming_a_mask(self) -> None:
        stages = build_modern_stage_bundle(runtime_config={
            "localizationProfile": "page_goc_tro_bat_on",
            "sourceEvidenceMode": "local_asr_text_only",
        })
        asr_rows = [{"position": 0, "startMs": 0, "endMs": 800, "asrTextZh": "背景口语", "confidence": 90}]
        ocr_detections = [{"timestampMs": 400, "bbox": (50, 80, 300, 140), "text": "上方剧情标题", "category": "STORY_TEXT", "confidence": 0.95}]
        with tempfile.TemporaryDirectory() as directory:
            with patch("dubvi_worker.ffprobe_duration", return_value=1.0), patch("dubvi_worker.ffprobe_dimensions", return_value=(576, 1152)), patch("dubvi_worker.extract_audio"), patch("dubvi_worker.transcribe", return_value=asr_rows), patch("auto_roi.sparse_whole_frame_discovery", return_value=ocr_detections):
                records = list(stages["source"](Path("source.mp4"), Path(directory)))
            self.assertFalse((Path(directory) / "page1_mask_overlay.png").exists())
        self.assertEqual(1, len(records))

    def test_page1_malformed_raw_candidate_cannot_hide_among_valid_candidates(self) -> None:
        stages = build_modern_stage_bundle(runtime_config={
            "localizationProfile": "page_goc_tro_bat_on",
            "sourceEvidenceMode": "local_asr_text_only",
        })
        asr_rows = [
            {"position": 0, "startMs": 600, "endMs": 1_000, "asrTextZh": "第一句口语", "confidence": 90},
            {"position": 1, "startMs": 1_700, "endMs": 2_100, "asrTextZh": "第二句口语", "confidence": 90},
        ]
        ocr_detections = [
            {"timestampMs": 700, "bbox": (180, 880, 388, 917), "text": "第一句字幕", "category": "UNKNOWN", "confidence": 0.91},
            {"timestampMs": 1_900, "text": "损坏候选", "category": "UNKNOWN", "confidence": 0.93},
        ]
        with tempfile.TemporaryDirectory() as directory:
            with patch("dubvi_worker.ffprobe_duration", return_value=3.0), patch("dubvi_worker.ffprobe_dimensions", return_value=(576, 1152)), patch("dubvi_worker.extract_audio"), patch("dubvi_worker.transcribe", return_value=asr_rows), patch("auto_roi.sparse_whole_frame_discovery", return_value=ocr_detections):
                with self.assertRaisesRegex(ModernP1CAdapterError, "MASK_REVIEW_REQUIRED"):
                    stages["source"](Path("source.mp4"), Path(directory))

    def test_modern_tts_stage_uses_runtime_provider_instead_of_hard_coding_capcut(self) -> None:
        stages = build_modern_stage_bundle(
            runtime_config={
                "localizationProfile": "loc-v1",
                "ttsProvider": "piper",
                "voice": {"name": "BV421_vivn_streaming"},
            }
        )
        captured: dict[str, object] = {}

        def capture(_alignment, _output_dir, **kwargs):
            captured.update(kwargs)
            return {"voiceEvents": []}

        with patch("tts_timing.run_tts_timing", side_effect=capture):
            stages["tts"]([], Path("output"))

        self.assertEqual("piper", captured["tts_provider"])
        self.assertNotEqual("capcut_synthesize", getattr(captured["synthesizer"], "__name__", ""))

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
                "page1MaskMode": "none",
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
            evidence = json.loads((output / "source_evidence.json").read_text(encoding="utf-8"))
            self.assertEqual("local_asr_text_only", evidence["mode"])
            self.assertEqual([], evidence["unresolvedVisualDialogue"])

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

    def test_page1_asr_confidence_is_not_gated_by_ocr_threshold(self) -> None:
        stages = build_modern_stage_bundle(runtime_config={
            "localizationProfile": "page_goc_tro_bat_on",
            "sourceEvidenceMode": "local_asr_text_only",
            "ocr": {"minConfidence": 90},
            "page1MaskMode": "none",
        })
        rows = [
            {"position": 0, "startMs": 0, "endMs": 700, "asrTextZh": "清楚的一句", "confidence": 58},
            {"position": 1, "startMs": 700, "endMs": 1500, "asrTextZh": "不确定的一句", "confidence": 49},
        ]
        with tempfile.TemporaryDirectory() as directory:
            with patch("dubvi_worker.ffprobe_duration", return_value=2.0), \
                 patch("dubvi_worker.extract_audio"), \
                 patch("dubvi_worker.transcribe", return_value=rows), \
                 patch("auto_roi.sparse_whole_frame_discovery", return_value=[]):
                records = list(stages["source"](Path("source.mp4"), Path(directory)))
        self.assertEqual("RESOLVED_DIALOGUE", records[0]["classification"])
        self.assertEqual("MEDIUM", records[0]["confidence"])
        self.assertEqual("medium", records[0]["source_confidence"])
        self.assertEqual("UNRESOLVED_DIALOGUE", records[1]["classification"])

    def test_page1_ocr_failure_is_typed_and_does_not_handoff_original_source(self) -> None:
        stages = build_modern_stage_bundle(runtime_config={
            "localizationProfile": "page_goc_tro_bat_on",
            "sourceEvidenceMode": "local_asr_text_only",
        })
        with tempfile.TemporaryDirectory() as directory:
            with patch("dubvi_worker.ffprobe_duration", return_value=1.0), patch("dubvi_worker.extract_audio"), patch("dubvi_worker.transcribe", return_value=[{
                "position": 0, "startMs": 0, "endMs": 800, "asrTextZh": "第一句", "confidence": 90,
            }]), patch("auto_roi.sparse_whole_frame_discovery", side_effect=RuntimeError("ocr unavailable")):
                with self.assertRaises(ModernP1CAdapterError):
                    stages["source"](Path("source.mp4"), Path(directory))

    def test_page1_mask_mode_none_is_the_explicit_no_mask_route(self) -> None:
        stages = build_modern_stage_bundle(runtime_config={
            "localizationProfile": "page_goc_tro_bat_on",
            "sourceEvidenceMode": "local_asr_text_only",
            "page1MaskMode": "none",
        })
        with tempfile.TemporaryDirectory() as directory:
            with patch("dubvi_worker.ffprobe_duration", return_value=1.0), patch("dubvi_worker.extract_audio"), patch("dubvi_worker.transcribe", return_value=[{
                "position": 0, "startMs": 0, "endMs": 800, "asrTextZh": "第一句", "confidence": 90,
            }]), patch("auto_roi.sparse_whole_frame_discovery", return_value=[]):
                records = stages["source"](Path("source.mp4"), Path(directory))
        self.assertEqual(1, len(records))

    def test_text_translation_boundary_receives_ids_and_source_text_only(self) -> None:
        stages = build_modern_stage_bundle(
            runtime_config={"localizationProfile": "page_goc_tro_bat_on", "sourceEvidenceMode": "local_asr_text_only", "page1MaskMode": "none"}
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
            runtime_config={"localizationProfile": "page_goc_tro_bat_on", "sourceEvidenceMode": "local_asr_text_only", "page1MaskMode": "none"}
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

    def test_localization_persists_gemini_provider_path_without_payload_or_key(self) -> None:
        stages = build_modern_stage_bundle(
            runtime_config={"localizationProfile": "page_goc_tro_bat_on", "sourceEvidenceMode": "local_asr_text_only", "page1MaskMode": "none"}
        )
        dummy_key = "GEMINI_API_KEY_TEST_VALUE"
        source_text = "租房"
        translated_text = "Thuê nhà"

        def gemini_response(payload):
            key = "semanticVi" if payload["mode"] == "semantic" else "nativeVi"
            return {"translations": [{"utteranceId": payload["utterances"][0]["utteranceId"], key: translated_text}]}

        gemini = Mock(side_effect=gemini_response)

        with tempfile.TemporaryDirectory() as directory, patch("cp2_gemini_adapter.live_cp2_gemini_adapter", return_value=gemini), patch.dict("os.environ", {"GEMINI_API_KEY": dummy_key}, clear=False):
            self._prepare_page1_output_dir(stages, Path(directory))
            records = [{
                "utteranceId": "u1", "startMs": 100, "endMs": 900,
                "resolvedZh": source_text, "source_confidence": "high", "kind": "dialogue",
            }]
            stages["localization"](records, output_dir=Path(directory))
            artifact = Path(directory) / "translation_provider.json"
            serialized = artifact.read_bytes()
            document = json.loads(serialized)

        self.assertEqual(1, document["schemaVersion"])
        self.assertEqual("gemini", document["providerPath"])
        self.assertIsNone(document["fallbackClassification"])
        self.assertEqual(2, gemini.call_count)
        self.assertNotIn(dummy_key.encode("utf-8"), serialized)
        self.assertNotIn(source_text.encode("utf-8"), serialized)
        self.assertNotIn(translated_text.encode("utf-8"), serialized)

    def test_localization_provider_artifact_does_not_claim_gemini_when_no_translation_is_needed(self) -> None:
        stages = build_modern_stage_bundle(
            runtime_config={"localizationProfile": "page_goc_tro_bat_on", "sourceEvidenceMode": "local_asr_text_only", "page1MaskMode": "none"}
        )
        records = [{
            "utteranceId": "u1", "startMs": 100, "endMs": 900,
            "resolvedZh": "租房", "source_confidence": "low",
            "classification": "UNRESOLVED_DIALOGUE", "kind": "dialogue",
        }]
        gemini = Mock()

        with tempfile.TemporaryDirectory() as directory, patch("cp2_gemini_adapter.live_cp2_gemini_adapter", return_value=gemini):
            self._prepare_page1_output_dir(stages, Path(directory))
            with self.assertRaisesRegex(ModernP1CAdapterError, "SOURCE_REVIEW_REQUIRED"):
                stages["localization"](records, output_dir=Path(directory))
            document = json.loads((Path(directory) / "translation_provider.json").read_text(encoding="utf-8"))

        self.assertIsNone(document["providerPath"])
        gemini.assert_not_called()

    def test_localization_persists_groq_provider_path_and_gemini_failure_classification(self) -> None:
        stages = build_modern_stage_bundle(
            runtime_config={"localizationProfile": "page_goc_tro_bat_on", "sourceEvidenceMode": "local_asr_text_only", "page1MaskMode": "none"}
        )
        gemini = Mock(side_effect=ProviderError("HTTP_429", transient=True, http_status=429))
        groq = Mock(side_effect=lambda payload: {
            "translations": [
                {"utteranceId": item["utteranceId"], ("semanticVi" if payload["mode"] == "semantic" else "nativeVi"): "Thuê nhà"}
                for item in payload["utterances"]
            ]
        })
        records = [{
            "utteranceId": "u1", "startMs": 100, "endMs": 900,
            "resolvedZh": "租房", "source_confidence": "high", "kind": "dialogue",
        }]

        with tempfile.TemporaryDirectory() as directory, patch("cp2_gemini_adapter.live_cp2_gemini_adapter", return_value=gemini), patch("text_translation_fallback.GroqTextTranslationAdapter", return_value=groq):
            self._prepare_page1_output_dir(stages, Path(directory))
            stages["localization"](records, output_dir=Path(directory))
            document = json.loads((Path(directory) / "translation_provider.json").read_text(encoding="utf-8"))

        self.assertEqual(1, document["schemaVersion"])
        self.assertEqual("groq", document["providerPath"])
        self.assertEqual("HTTP_429", document["fallbackClassification"])

    def test_localization_provider_failure_leaves_no_success_artifact(self) -> None:
        stages = build_modern_stage_bundle(
            runtime_config={"localizationProfile": "page_goc_tro_bat_on", "sourceEvidenceMode": "local_asr_text_only", "page1MaskMode": "none"}
        )
        failure = ProviderError("PROVIDER_REJECTED", http_status=400)
        records = [{
            "utteranceId": "u1", "startMs": 100, "endMs": 900,
            "resolvedZh": "租房", "source_confidence": "high", "kind": "dialogue",
        }]

        with tempfile.TemporaryDirectory() as directory, patch("cp2_gemini_adapter.live_cp2_gemini_adapter", side_effect=failure):
            with self.assertRaisesRegex(ProviderError, "PROVIDER_REJECTED"):
                stages["localization"](records, output_dir=Path(directory))
            self.assertFalse((Path(directory) / "translation_provider.json").exists())

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
                patch("dubvi_worker.ffprobe_dimensions", return_value=(720, 1280)),
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
        identity = {
            "translator_job_id": "job-1", "dispatch_id": "dispatch-1",
            "reup_job_id": "reup-1", "handoff_id": "handoff-1",
            "candidate_id": "candidate-1", "schedule_id": "schedule-1",
        }

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
        identity = {
            "translator_job_id": "job-1", "dispatch_id": "dispatch-1",
            "reup_job_id": "reup-1", "handoff_id": "handoff-1",
            "candidate_id": "candidate-1", "schedule_id": "schedule-1",
        }
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
        identity = {
            "translator_job_id": "job-1", "dispatch_id": "dispatch-1",
            "reup_job_id": "reup-1", "handoff_id": "handoff-1",
            "candidate_id": "candidate-1", "schedule_id": "schedule-1",
        }
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
        identity = {
            "translator_job_id": "job-1", "dispatch_id": "dispatch-1",
            "reup_job_id": "reup-1", "handoff_id": "handoff-1",
            "candidate_id": "candidate-1", "schedule_id": "schedule-1",
        }
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
