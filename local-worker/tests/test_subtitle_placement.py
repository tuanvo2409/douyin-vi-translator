from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from subtitle_layout import SubtitleLayoutPolicy, layout_subtitles, render_ass_document
from timeline_mapper import TimelineMap, TimelineSegment


def _cue(
    text: str = "Xin chào mọi người.",
    *,
    identifier: str = "utt-1",
    start: int = 8_000,
    end: int = 12_000,
) -> dict[str, object]:
    return {
        "utteranceId": identifier,
        "voiceEventId": identifier,
        "translatedTextVi": text,
        "scheduledStartMs": start,
        "scheduledEndMs": end,
    }


def _timeline() -> TimelineMap:
    return TimelineMap(
        10_000,
        [TimelineSegment(0, 10_000, 0, 20_000, 0.5, "TEST")],
    )


def _evidence(
    category: str = "DIALOGUE_SUBTITLE",
    *,
    start: int = 4_000,
    end: int = 5_000,
    bbox: list[object] | None = None,
    track_id: str = "dialogue-1",
) -> dict[str, object]:
    return {
        "category": category,
        "trackId": track_id,
        "startMs": start,
        "endMs": end,
        "bbox": bbox if bbox is not None else [120, 730, 600, 770],
        "text": "源字幕",
        "subtitleZh": "源字幕",
    }


class SubtitlePlacementTests(unittest.TestCase):
    def test_dialogue_subtitle_is_preferred_anchor(self) -> None:
        cue = layout_subtitles([_cue()], placement_evidence=[_evidence()], source_to_output=_timeline())["displayCues"][0]
        self.assertEqual("DIALOGUE_SUBTITLE", cue["anchorSource"])
        self.assertEqual("dialogue-1", cue["sourceDialogueTrackId"])
        self.assertEqual(750, cue["sourceAnchorYPx"])
        self.assertEqual(750, cue["finalAnchorYPx"])
        self.assertFalse(cue["fallbackUsed"])

    def test_watermark_does_not_anchor(self) -> None:
        cue = layout_subtitles([_cue()], placement_evidence=[_evidence("WATERMARK")], source_to_output=_timeline())["displayCues"][0]
        self.assertEqual("V5_FALLBACK_BOTTOM", cue["anchorSource"])
        self.assertTrue(cue["fallbackUsed"])

    def test_story_text_does_not_anchor(self) -> None:
        cue = layout_subtitles([_cue()], placement_evidence=[_evidence("STORY_TEXT")], source_to_output=_timeline())["displayCues"][0]
        self.assertTrue(cue["fallbackUsed"])
        self.assertIsNone(cue["sourceAnchorYPx"])

    def test_unknown_text_does_not_anchor(self) -> None:
        cue = layout_subtitles([_cue()], placement_evidence=[_evidence("UNKNOWN")], source_to_output=_timeline())["displayCues"][0]
        self.assertTrue(cue["fallbackUsed"])
        self.assertIsNone(cue["sourceDialogueTrackId"])

    def test_untrusted_page2_evidence_uses_explicit_bottom_fallback(self) -> None:
        cases = [
            _evidence("WATERMARK"),
            _evidence("STORY_TEXT"),
            _evidence("UNKNOWN"),
            {**_evidence(), "text": "不同的字幕", "subtitleZh": "原字幕"},
            {key: value for key, value in _evidence().items() if key != "bbox"},
        ]
        for evidence in cases:
            with self.subTest(evidence=evidence):
                cue = layout_subtitles(
                    [_cue()], placement_evidence=[evidence], source_to_output=_timeline()
                )["displayCues"][0]
                self.assertEqual("V5_FALLBACK_BOTTOM", cue["anchorSource"])
                self.assertTrue(cue["fallbackUsed"])
                self.assertIsNone(cue["sourceDialogueTrackId"])
                self.assertIsNone(cue["sourceAnchorYPx"])

    def test_missing_source_timeline_cannot_claim_a_source_anchor(self) -> None:
        cue = layout_subtitles([_cue()], placement_evidence=[_evidence()])["displayCues"][0]
        self.assertEqual("V5_FALLBACK_BOTTOM", cue["anchorSource"])
        self.assertEqual("SOURCE_TIMELINE_MISSING", cue["fallbackReason"])
        self.assertTrue(cue["fallbackUsed"])
        self.assertIsNone(cue["sourceAnchorYPx"])

    def test_low_confidence_dialogue_does_not_anchor(self) -> None:
        evidence = _evidence()
        evidence["confidence"] = "LOW"
        cue = layout_subtitles([_cue()], placement_evidence=[evidence], source_to_output=_timeline())["displayCues"][0]
        self.assertTrue(cue["fallbackUsed"])

    def test_source_time_is_mapped_before_overlap_is_checked(self) -> None:
        cue = layout_subtitles(
            [_cue(start=8_000, end=10_000)],
            placement_evidence=[_evidence(start=4_000, end=5_000)],
            source_to_output=_timeline(),
        )["displayCues"][0]
        self.assertEqual("DIALOGUE_SUBTITLE", cue["anchorSource"])

    def test_noisy_dialogue_y_uses_one_stable_median_anchor(self) -> None:
        evidence = [
            _evidence(bbox=[100, 730, 620, 750]),
            _evidence(bbox=[100, 735, 620, 755]),
            _evidence(bbox=[100, 738, 620, 758]),
        ]
        cue = layout_subtitles([_cue()], placement_evidence=evidence, source_to_output=_timeline())["displayCues"][0]
        self.assertEqual(745, cue["sourceAnchorYPx"])
        self.assertEqual(745, cue["finalAnchorYPx"])

    def test_vietnamese_box_may_be_wider_than_source_bbox(self) -> None:
        cue = layout_subtitles(
            [_cue("Một câu tiếng Việt dài hơn vùng chữ nguồn.")],
            placement_evidence=[_evidence(bbox=[350, 730, 370, 770])],
            source_to_output=_timeline(),
        )["displayCues"][0]
        self.assertGreater(cue["boxWidthPx"], 20)

    def test_source_bbox_controls_approximate_horizontal_center(self) -> None:
        cue = layout_subtitles(
            [_cue("Neo vùng thoại")],
            placement_evidence=[_evidence(bbox=[250, 730, 350, 770])],
            source_to_output=_timeline(),
        )["displayCues"][0]
        self.assertEqual(300, cue["sourceAnchorXPx"])
        self.assertEqual(300, cue["finalAnchorXPx"])
        self.assertGreaterEqual(cue["boxLeftPx"], 0)
        self.assertLessEqual(cue["boxRightPx"], 720)

    def test_anchored_box_overlaps_cleaned_dialogue_zone(self) -> None:
        cue = layout_subtitles(
            [_cue()], placement_evidence=[_evidence(bbox=[120, 730, 600, 770])], source_to_output=_timeline()
        )["displayCues"][0]
        self.assertLess(cue["boxTopPx"], 770)
        self.assertGreater(cue["boxBottomPx"], 730)

    def test_two_line_anchored_box_stays_inside_vertical_bounds(self) -> None:
        policy = SubtitleLayoutPolicy(frame_width=480, frame_height=640, font_size=38, min_font_size=28)
        cue = layout_subtitles(
            [_cue("Một câu dài để kiểm tra hộp hai dòng ở gần đáy.", start=16_000, end=20_000)],
            policy=policy,
            placement_evidence=[_evidence(bbox=[80, 500, 400, 540])],
            source_to_output=_timeline(),
        )["displayCues"][0]
        self.assertEqual(2, cue["lineCount"])
        self.assertGreaterEqual(cue["boxTopPx"], 0)
        self.assertLessEqual(cue["boxBottomPx"], policy.frame_height)

    def test_split_children_share_the_same_anchor(self) -> None:
        policy = SubtitleLayoutPolicy(frame_width=360, frame_height=640, font_size=32, min_font_size=24)
        result = layout_subtitles(
            [_cue("Đây là một câu rất dài cần chia thành nhiều đoạn hiển thị tuần tự.", start=8_000, end=20_000)],
            policy=policy,
            placement_evidence=[_evidence(bbox=[80, 500, 280, 540])],
            source_to_output=_timeline(),
        )
        cues = result["displayCues"]
        self.assertGreater(len(cues), 1)
        self.assertEqual({520}, {cue["finalAnchorYPx"] for cue in cues})

    def test_missing_dialogue_evidence_uses_v5_bottom_fallback(self) -> None:
        result = layout_subtitles([_cue()])
        cue = result["displayCues"][0]
        self.assertEqual("V5_FALLBACK_BOTTOM", cue["anchorSource"])
        self.assertEqual(874, cue["finalAnchorYPx"])

    def test_malformed_or_unsafe_evidence_falls_back(self) -> None:
        evidence = [_evidence(bbox=[1, 2, 3]), _evidence(bbox=[100, 80, 600, 120])]
        cue = layout_subtitles([_cue()], placement_evidence=evidence, source_to_output=_timeline())["displayCues"][0]
        self.assertTrue(cue["fallbackUsed"])
        self.assertIsNone(cue["sourceAnchorYPx"])

    def test_source_aware_layout_still_has_at_most_two_lines(self) -> None:
        policy = SubtitleLayoutPolicy(frame_width=360, frame_height=640, font_size=32, min_font_size=24)
        result = layout_subtitles(
            [_cue("Một câu cực kỳ dài có nhiều từ phải được chia tuần tự để đọc rõ ràng.")],
            policy=policy,
            placement_evidence=[_evidence()],
            source_to_output=_timeline(),
        )
        self.assertTrue(all(cue["lineCount"] <= 2 for cue in result["displayCues"]))

    def test_text_outline_and_shadow_remain_inside_anchored_box(self) -> None:
        policy = SubtitleLayoutPolicy()
        cue = layout_subtitles(
            [_cue("Một câu neo theo vùng phụ đề nguồn.")],
            policy=policy,
            placement_evidence=[_evidence()],
            source_to_output=_timeline(),
        )["displayCues"][0]
        self.assertGreaterEqual(cue["boxWidthPx"], max(cue["lineWidthsPx"]) + 2 * policy.outline_px + 2 * policy.shadow_px)
        self.assertGreaterEqual(cue["boxHeightPx"], policy.shadow_px + 2 * policy.outline_px)

    def test_tts_timeline_masking_inputs_are_unchanged(self) -> None:
        cues = [_cue()]
        evidence = [_evidence()]
        voice_timeline = {"voiceEvents": [{"utteranceId": "utt-1"}]}
        masking = {"artifact": "mask_plan_v4", "regions": []}
        before = copy.deepcopy((cues, evidence, voice_timeline, masking))
        layout_subtitles(cues, placement_evidence=evidence, source_to_output=_timeline())
        self.assertEqual(before, (cues, evidence, voice_timeline, masking))

    def test_render_ass_uses_source_aware_anchor(self) -> None:
        policy = SubtitleLayoutPolicy()
        result = layout_subtitles([_cue()], placement_evidence=[_evidence()], source_to_output=_timeline())
        ass = render_ass_document(result["displayCues"], policy)
        self.assertIn(r"\pos(360,750)", ass)

    def test_page1_fixed_mask_box_remains_the_placement_authority(self) -> None:
        from subtitle_layout import write_ass_subtitles

        with tempfile.TemporaryDirectory() as directory:
            result = write_ass_subtitles(
                [_cue()],
                Path(directory) / "page1-fixed-box.ass",
                width=720,
                height=960,
                page1_mask_bbox=(100, 700, 620, 850),
                placement_evidence=[_evidence()],
                source_to_output=_timeline(),
            )
        cue = result["displayCues"][0]
        self.assertEqual("V5_FALLBACK_BOTTOM", cue["anchorSource"])
        self.assertGreaterEqual(cue["boxLeftPx"], 100)
        self.assertLessEqual(cue["boxRightPx"], 620)
        self.assertGreaterEqual(cue["boxTopPx"], 700)
        self.assertLessEqual(cue["boxBottomPx"], 850)

    def test_production_placement_logic_is_not_fixture_specific(self) -> None:
        source = Path(__file__).resolve().parents[1] / "subtitle_layout.py"
        text = source.read_text(encoding="utf-8").lower()
        self.assertNotIn("fixture2", text)
        self.assertNotIn("blind_fixture_02", text)
        self.assertNotIn("utt_v1_d_", text)


if __name__ == "__main__":
    unittest.main()
