from __future__ import annotations

import unittest

import auto_roi
import dubvi_worker
import llm_translator
import localized_masking


class TimelineCorrectionTests(unittest.TestCase):
    def test_canonical_voice_inputs_preserve_source_starts_and_long_gaps(self) -> None:
        clips = [
            {"startMs": 1200, "endMs": 1800, "fitted_voice": "a"},
            {"startMs": 12_000, "endMs": 12_600, "fitted_voice": "b"},
        ]
        self.assertEqual([(1200, "a"), (12_000, "b")], dubvi_worker.preserve_source_timeline(clips))

    def test_render_command_keeps_source_duration_when_voice_ends_early(self) -> None:
        command = dubvi_worker.build_render_command("source.mp4", "out.mp4", "subs.ass", [], "duck", 200.0)
        self.assertNotIn("-shortest", command)
        self.assertIn("-t", command)
        self.assertEqual("200.000", command[command.index("-t") + 1])


class MaskCorrectionTests(unittest.TestCase):
    def test_plan_keeps_regions_after_the_old_global_limit(self) -> None:
        detections = [
            {"timestampMs": index * 1000, "text": "字幕", "bbox": (10, 10, 40, 30)}
            for index in range(70)
        ]
        plan = localized_masking.build_mask_plan(detections, (100, 100))
        self.assertEqual(70, len(plan))
        self.assertEqual(69_000, plan[-1]["timestampMs"])

    def test_sparse_whole_frame_discovery_keeps_non_bottom_regions_separate(self) -> None:
        regions = auto_roi.discover_cjk_regions([
            {"timestampMs": 0, "items": [{"text": "上", "bbox": (100, 120, 160, 150)}]},
            {"timestampMs": 1000, "items": [{"text": "中", "bbox": (120, 800, 190, 830)}]},
            {"timestampMs": 2000, "items": [{"text": "下", "bbox": (120, 1600, 190, 1630)}]},
        ], (1080, 1920))
        self.assertEqual([120, 800, 1600], [region["bbox"][1] for region in regions])

    def test_tracker_updates_motion_and_resets_scene(self) -> None:
        tracker = auto_roi.TemporalCJKTracker(hold_ms=250, frame_size=(1000, 1000))
        tracker.observe(0, [(100, 100, 200, 130)], scene_id=1)
        tracker.observe(100, [(130, 100, 230, 130)], scene_id=1)
        self.assertEqual([(130, 100, 230, 130)], tracker.active_regions(200, scene_id=1))
        self.assertEqual([], tracker.active_regions(200, scene_id=2))

    def test_targeted_repair_expands_only_residual_tracks_once(self) -> None:
        initial = [{"timestampMs": 1000, "bbox": (100, 100, 140, 120)}, {"timestampMs": 5000, "bbox": (400, 400, 440, 420)}]
        residual = [{"timestampMs": 1000, "text": "残", "bbox": (102, 102, 138, 118)}]
        retry = localized_masking.expand_affected_plan(initial, residual, (1000, 1000))
        self.assertEqual(1, len(retry))
        self.assertEqual(1000, retry[0]["timestampMs"])
        self.assertGreater(retry[0]["bbox"][2] - retry[0]["bbox"][0], 40)

    def test_qc_sampling_covers_beginning_middle_and_end(self) -> None:
        samples = localized_masking.stratified_sample_indices(100, 9)
        self.assertEqual(0, samples[0])
        self.assertEqual(99, samples[-1])
        self.assertTrue(any(40 <= item <= 60 for item in samples))

    def test_lossless_intermediate_uses_ffv1_not_mp4v(self) -> None:
        command = localized_masking.lossless_transport_command("input.mp4", "temp.mkv", 64, 48, 30.0)
        self.assertIn("ffv1", command)
        self.assertNotIn("mp4v", " ".join(command).lower())


class TranslationCorrectionTests(unittest.TestCase):
    def test_unknown_profiles_are_neutral_but_explicit_profiles_keep_persona(self) -> None:
        self.assertIn("TRUNG TÍNH", llm_translator.build_system_prompt("p1c-demo"))
        self.assertIn("TRUNG TÍNH", llm_translator.build_system_prompt(None))
        self.assertIn("GÓC TRỌ BẤT ỔN", llm_translator.build_system_prompt("goc_tro"))
        self.assertIn("GIẢI CỨU CHUỒNG LỢN", llm_translator.build_system_prompt("giai_cuu"))

    def test_cjk_translation_is_rejected_not_silently_stripped(self) -> None:
        self.assertFalse(llm_translator.is_valid_translation_text("Xin chào 中文"))
        self.assertEqual("Xin chào 中文", llm_translator.clean_vietnamese_text("Xin chào 中文"))

    def test_pass_b_prompt_has_bounded_dialogue_context_and_glossary(self) -> None:
        prompt = llm_translator.build_timing_repair_prompt(
            {"position": 2, "startMs": 0, "endMs": 1000, "sourceTextZh": "甲", "translatedTextVi": "A"},
            measured_ms=1500, previous=[{"position": 1, "sourceTextZh": "前", "translatedTextVi": "Trước"}],
            following=[{"position": 3, "sourceTextZh": "后"}], glossary=({"source": "甲", "target": "Giáp", "category": "name", "confidence": 1.0},), context_card={"names": {"甲": "Giáp"}},
        )
        self.assertIn("PREVIOUS", prompt)
        self.assertIn("FOLLOWING_SOURCE_ONLY", prompt)
        self.assertIn("GLOSSARY", prompt)
        self.assertIn("CONTEXT_CARD", prompt)

    def test_rolling_context_only_records_validated_glossary_choices(self) -> None:
        card = llm_translator.advance_context_card({}, [{"sourceTextZh": "小明", "translatedTextVi": "Tiểu Minh"}], ({"source": "小明", "target": "Tiểu Minh", "category": "name", "confidence": 1.0},))
        self.assertEqual({"glossaryChoices": {"小明": "Tiểu Minh"}}, card)


if __name__ == "__main__":
    unittest.main()
