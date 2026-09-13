from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import auto_roi
import llm_translator


class CharacterizationApiTests(unittest.TestCase):
    def test_quality_helpers_are_available_without_new_dependencies(self) -> None:
        self.assertTrue(hasattr(auto_roi, "map_ocr_quad_to_source"))
        self.assertTrue(hasattr(auto_roi, "contains_cjk_text"))
        self.assertTrue(hasattr(auto_roi, "group_cjk_regions"))
        self.assertTrue(hasattr(auto_roi, "TemporalCJKTracker"))
        self.assertTrue(hasattr(llm_translator, "validate_translation_response"))
        self.assertTrue(hasattr(llm_translator, "build_contextual_batches"))


class MaskGeometryTests(unittest.TestCase):
    def _helper(self, name: str):
        self.assertTrue(hasattr(auto_roi, name), name)
        return getattr(auto_roi, name)

    def test_enlarged_roi_quad_maps_back_to_source_coordinates(self) -> None:
        mapped = self._helper("map_ocr_quad_to_source")(
            [[25, 50], [125, 50], [125, 100], [25, 100]],
            crop_origin=(100, 200), enlargement=2.5, frame_size=(1080, 1920),
        )
        self.assertEqual(((110, 220), (150, 220), (150, 240), (110, 240)), mapped)

    def test_cjk_filter_excludes_latin_and_icons(self) -> None:
        contains = self._helper("contains_cjk_text")
        self.assertTrue(contains("中文字幕"))
        self.assertFalse(contains("Xin chao"))
        self.assertFalse(contains("🔥 123"))

    def test_nearby_cjk_boxes_group_but_distant_text_does_not(self) -> None:
        group = self._helper("group_cjk_regions")
        regions = group([
            {"text": "中文", "bbox": (100, 400, 200, 440), "confidence": 0.9},
            {"text": "字幕", "bbox": (410, 400, 520, 440), "confidence": 0.9},
            {"text": "水印", "bbox": (800, 100, 900, 140), "confidence": 0.9},
        ], frame_size=(1080, 1920))
        self.assertEqual(2, len(regions))
        self.assertIn((100, 400, 520, 440), [region["bbox"] for region in regions])

    def test_two_near_lines_become_one_subtitle_region(self) -> None:
        group = self._helper("group_cjk_regions")
        regions = group([
            {"text": "第一行", "bbox": (120, 700, 500, 740), "confidence": 0.9},
            {"text": "第二行", "bbox": (150, 750, 530, 790), "confidence": 0.9},
        ], frame_size=(1080, 1920))
        self.assertEqual([(120, 700, 530, 790)], [region["bbox"] for region in regions])

    def test_adaptive_dilation_covers_edges_and_clamps_to_frame(self) -> None:
        dilate = self._helper("dilate_region")
        self.assertEqual((0, 0, 1080, 1920), dilate((0, 0, 1080, 1920), frame_size=(1080, 1920)))
        self.assertEqual((94, 190, 156, 250), dilate((100, 200, 150, 240), frame_size=(1080, 1920)))

    def test_track_holds_short_miss_but_resets_at_scene_boundary(self) -> None:
        tracker = self._helper("TemporalCJKTracker")(hold_ms=200, frame_size=(1080, 1920))
        tracker.observe(0, [(100, 400, 500, 440)], scene_id=1)
        self.assertEqual(1, len(tracker.active_regions(150, scene_id=1)))
        self.assertEqual([], tracker.active_regions(250, scene_id=1))
        self.assertEqual([], tracker.active_regions(150, scene_id=2))


class LocalizationStructureTests(unittest.TestCase):
    def _helper(self, name: str):
        self.assertTrue(hasattr(llm_translator, name), name)
        return getattr(llm_translator, name)

    def setUp(self) -> None:
        self.cues = [
            {"position": 2, "startMs": 0, "endMs": 1000, "sourceTextZh": "甲"},
            {"position": 7, "startMs": 1000, "endMs": 2000, "sourceTextZh": "乙"},
        ]

    def test_translation_response_rejects_missing_duplicate_unknown_blank_and_malformed(self) -> None:
        validate = self._helper("validate_translation_response")
        valid = {"translations": [{"position": 7, "translatedTextVi": "B"}, {"position": 2, "translatedTextVi": "A"}]}
        self.assertEqual({2: "A", 7: "B"}, validate(self.cues, valid))
        for document in (
            {"translations": [{"position": 2, "translatedTextVi": "A"}]},
            {"translations": [{"position": 2, "translatedTextVi": "A"}, {"position": 2, "translatedTextVi": "B"}]},
            {"translations": [{"position": 2, "translatedTextVi": "A"}, {"position": 99, "translatedTextVi": "B"}]},
            {"translations": [{"position": 2, "translatedTextVi": ""}, {"position": 7, "translatedTextVi": "B"}]},
            "not json",
        ):
            with self.assertRaises(ValueError):
                validate(self.cues, document)

    def test_context_batches_preserve_cues_and_expose_only_read_only_neighbors(self) -> None:
        batches = self._helper("build_contextual_batches")([
            {"position": n, "startMs": n * 1000, "endMs": n * 1000 + 900, "sourceTextZh": str(n)}
            for n in range(18)
        ], max_current=12, max_duration_ms=75_000)
        self.assertEqual([tuple(range(12)), tuple(range(12, 18))], [tuple(c["position"] for c in batch["current"]) for batch in batches])
        self.assertEqual((10, 11), tuple(c["position"] for c in batches[1]["previous"]))
        self.assertEqual((), tuple(c["position"] for c in batches[1]["following"]))

    def test_glossary_and_context_card_are_bounded_and_deterministic(self) -> None:
        glossary = self._helper("normalize_glossary")([
            {"source": "小明", "target": "Tiểu Minh", "category": "name", "confidence": 1.0},
            {"source": "小明", "target": "Khác", "category": "name", "confidence": 1.0},
        ])
        self.assertEqual(({"source": "小明", "target": "Tiểu Minh", "category": "name", "confidence": 1.0},), glossary)
        card = self._helper("normalize_context_card")({"pronouns": {"小明": "mình"}, "unknown": "discard"})
        self.assertEqual({"pronouns": {"小明": "mình"}}, card)

    def test_contextual_prompt_prohibits_invented_events_and_keeps_identity_local(self) -> None:
        prompt = self._helper("build_contextual_prompt")(self.cues, [], [], {}, ())
        self.assertIn("Không được bịa", prompt)
        self.assertIn('"position": 2', prompt)
        self.assertNotIn('"startMs"', prompt)

    def test_tts_overflow_uses_one_structured_pass_b_without_local_truncation(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": '{"translations":[{"position":2,"translatedTextVi":"Bản ngắn"}]}' }]}}]
        }
        cue = {**self.cues[0], "translatedTextVi": "Bản hiện tại dài hơn"}
        with patch.object(llm_translator.gemini_pool, "get_key", return_value=None), patch.object(llm_translator.requests, "post", return_value=response) as post:
            self.assertEqual("Bản ngắn", llm_translator.refine_for_tts_overflow(cue, api_key="test-key", measured_ms=2400))
        post.assert_called_once()


if __name__ == "__main__":
    unittest.main()
