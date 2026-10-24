from __future__ import annotations

import copy
import subprocess
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from subtitle_layout import SubtitleLayoutError, SubtitleLayoutPolicy, layout_subtitles, render_ass_document, write_ass_subtitles


def _cue(
    text: str,
    *,
    identifier: str = "utt-1",
    start: int = 1_000,
    end: int = 8_000,
) -> dict[str, object]:
    return {
        "utteranceId": identifier,
        "voiceEventId": identifier,
        "translatedTextVi": text,
        "scheduledStartMs": start,
        "scheduledEndMs": end,
    }


class SubtitleLayoutTests(unittest.TestCase):
    def test_short_subtitle_remains_one_line(self) -> None:
        result = layout_subtitles([_cue("Xin chào.")])
        self.assertEqual(1, len(result["displayCues"]))
        self.assertEqual(1, result["displayCues"][0]["lineCount"])

    def test_longer_subtitle_wraps_to_exactly_two_lines(self) -> None:
        result = layout_subtitles([_cue("Đây là một câu tiếng Việt đủ dài để xuống dòng")])
        cue = result["displayCues"][0]
        self.assertEqual(2, cue["lineCount"])
        self.assertEqual(2, len(cue["lines"]))

    def test_no_display_cue_renders_more_than_two_lines(self) -> None:
        result = layout_subtitles([_cue("Một câu rất dài có nhiều từ tiếng Việt để kiểm tra giới hạn hiển thị")])
        self.assertTrue(all(cue["lineCount"] <= 2 for cue in result["displayCues"]))

    def test_long_line_uses_balanced_wrapping(self) -> None:
        result = layout_subtitles([_cue("Một hai ba bốn năm sáu bảy tám chín mười")])
        widths = result["displayCues"][0]["lineWidthsPx"]
        self.assertEqual(2, len(widths))
        self.assertLess(abs(widths[0] - widths[1]), max(widths) * 0.35)

    def test_wrapping_never_splits_a_vietnamese_word(self) -> None:
        text = "Câu tiếng Việt dài có dấu không được cắt"
        result = layout_subtitles([_cue(text)])
        rendered_words = " ".join(result["displayCues"][0]["lines"]).split()
        self.assertEqual(text.split(), rendered_words)

    def test_measured_glyph_width_stays_within_safe_box_width(self) -> None:
        policy = SubtitleLayoutPolicy()
        result = layout_subtitles([_cue("Một câu tiếng Việt dài cần đo bằng glyph thực tế")], policy=policy)
        max_box = policy.frame_width * policy.max_box_width_ratio
        self.assertLessEqual(max(width for cue in result["displayCues"] for width in cue["lineWidthsPx"]), max_box)
        self.assertLessEqual(result["maximumBoxWidthRatio"], policy.max_box_width_ratio)

    def test_outline_remains_inside_background_box(self) -> None:
        policy = SubtitleLayoutPolicy()
        cue = layout_subtitles([_cue("Đường viền phải nằm trong hộp")], policy=policy)["displayCues"][0]
        self.assertGreaterEqual(cue["boxWidthPx"], max(cue["lineWidthsPx"]) + 2 * policy.outline_px)
        self.assertLessEqual(cue["boxLeftPx"], policy.frame_width / 2)
        self.assertGreaterEqual(cue["boxRightPx"], policy.frame_width / 2)

    def test_shadow_remains_inside_background_box(self) -> None:
        policy = SubtitleLayoutPolicy()
        cue = layout_subtitles([_cue("Bóng chữ không được tràn khỏi nền")], policy=policy)["displayCues"][0]
        required_width = max(cue["lineWidthsPx"]) + 2 * policy.outline_px + 2 * policy.shadow_px
        self.assertGreaterEqual(cue["boxWidthPx"], required_width)
        self.assertGreaterEqual(cue["boxHeightPx"], policy.shadow_px + 2 * policy.outline_px)

    def test_box_stays_inside_frame_left_and_right_bounds(self) -> None:
        result = layout_subtitles([_cue("Hộp phụ đề phải nằm trong khung hình")])
        for cue in result["displayCues"]:
            self.assertGreaterEqual(cue["boxLeftPx"], 0)
            self.assertLessEqual(cue["boxRightPx"], 720)

    def test_two_line_box_stays_inside_frame_bottom_bound(self) -> None:
        result = layout_subtitles([_cue("Một câu dài để kiểm tra vị trí đáy của hộp phụ đề")])
        cue = result["displayCues"][0]
        self.assertEqual(2, cue["lineCount"])
        self.assertLessEqual(cue["boxBottomPx"], 960)

    def test_font_size_decreases_only_when_wrapping_alone_cannot_fit(self) -> None:
        policy = SubtitleLayoutPolicy(frame_width=480, frame_height=640, font_size=42, min_font_size=24)
        result = layout_subtitles([_cue("Một câu rất dài cần giảm cỡ chữ để vừa hai dòng", start=0, end=8_000)], policy=policy)
        cue = result["displayCues"][0]
        self.assertEqual(2, cue["lineCount"])
        self.assertLess(cue["fontSize"], policy.font_size)

    def test_font_size_never_decreases_below_configured_minimum(self) -> None:
        policy = SubtitleLayoutPolicy(frame_width=360, frame_height=640, font_size=42, min_font_size=24)
        result = layout_subtitles([_cue("Một câu dài cần giữ cỡ chữ tối thiểu", start=0, end=8_000)], policy=policy)
        self.assertTrue(all(cue["fontSize"] >= policy.min_font_size for cue in result["displayCues"]))
        self.assertGreaterEqual(result["minimumFontSizeUsed"], policy.min_font_size)

    def test_uncapable_two_line_text_becomes_multiple_display_cues(self) -> None:
        policy = SubtitleLayoutPolicy(frame_width=360, frame_height=640, font_size=32, min_font_size=24)
        result = layout_subtitles([_cue(
            "Đây là một câu cực kỳ dài cần được chia thành nhiều phụ đề tuần tự để dễ đọc",
            start=0,
            end=20_000,
        )], policy=policy)
        self.assertGreater(len(result["displayCues"]), 1)
        self.assertEqual(1, result["splitUtteranceCount"])

    def test_display_split_preserves_all_vietnamese_words_in_order(self) -> None:
        text = "Đây là câu dài cần chia hiển thị nhưng vẫn giữ nguyên toàn bộ từ"
        policy = SubtitleLayoutPolicy(frame_width=360, frame_height=640, font_size=32, min_font_size=24)
        result = layout_subtitles([_cue(text, start=0, end=20_000)], policy=policy)
        rendered_words = " ".join(line for cue in result["displayCues"] for line in cue["lines"]).split()
        self.assertEqual(text.split(), rendered_words)

    def test_split_display_cues_stay_inside_original_output_interval(self) -> None:
        result = layout_subtitles([_cue(
            "Một câu cực kỳ dài cần chia thành các đoạn hiển thị tuần tự không mất chữ",
            start=1_250,
            end=20_000,
        )], policy=SubtitleLayoutPolicy(frame_width=360, frame_height=640, font_size=32, min_font_size=24))
        for cue in result["displayCues"]:
            self.assertGreaterEqual(cue["outputStartMs"], 1_250)
            self.assertLessEqual(cue["outputEndMs"], 20_000)

    def test_split_cue_intervals_have_no_gap_or_overlap(self) -> None:
        result = layout_subtitles([_cue(
            "Một câu cực kỳ dài cần chia thành các đoạn hiển thị tuần tự không mất chữ",
            start=0,
            end=20_000,
        )], policy=SubtitleLayoutPolicy(frame_width=360, frame_height=640, font_size=32, min_font_size=24))
        cues = result["displayCues"]
        self.assertEqual(cues[0]["outputStartMs"], 0)
        self.assertEqual(cues[-1]["outputEndMs"], 20_000)
        for left, right in zip(cues, cues[1:]):
            self.assertEqual(left["outputEndMs"], right["outputStartMs"])

    def test_tts_voice_and_adaptive_timeline_inputs_are_unchanged(self) -> None:
        cues = [_cue("Một câu dài để hiển thị")]
        voice_timeline = {"voiceEvents": [{"utteranceId": "utt-1", "scheduledStartMs": 1_000, "scheduledEndMs": 8_000}]}
        adaptive_timeline = {"segments": [{"sourceStartMs": 0, "sourceEndMs": 8_000, "outputStartMs": 0, "outputEndMs": 8_000}]}
        before = (copy.deepcopy(cues), copy.deepcopy(voice_timeline), copy.deepcopy(adaptive_timeline))
        layout_subtitles(cues)
        self.assertEqual(before, (cues, voice_timeline, adaptive_timeline))

    def test_accepted_masking_plan_input_is_unchanged(self) -> None:
        cue = _cue("Một câu dài để hiển thị")
        cue["acceptedMaskingPlan"] = {"artifact": "mask_plan_v4", "regions": []}
        before = copy.deepcopy(cue)
        layout_subtitles([cue])
        self.assertEqual(before, cue)

    def test_ass_output_contains_the_layout_lines_and_box_events(self) -> None:
        result = layout_subtitles([_cue("Một câu tiếng Việt dài để xuống dòng")])
        ass = render_ass_document(result["displayCues"], SubtitleLayoutPolicy())
        self.assertIn(r"\p1", ass)
        self.assertIn(r"\N", ass)
        self.assertIn("Dialogue:", ass)

    def test_ass_text_style_uses_the_configured_font_size(self) -> None:
        policy = SubtitleLayoutPolicy(font_size=34, min_font_size=24)
        result = layout_subtitles([_cue("Cỡ chữ cấu hình phải được tôn trọng")], policy=policy)
        ass = render_ass_document(result["displayCues"], policy)
        self.assertIn("Style: SubtitleText,Arial,34,", ass)

    def test_ass_text_style_uses_explicit_presentation_color(self) -> None:
        result = layout_subtitles([_cue("Màu thương hiệu")])
        cyan = render_ass_document(result["displayCues"], presentation_color="#00FFFF")
        yellow = render_ass_document(result["displayCues"], presentation_color="#FFE600")
        self.assertIn("Style: SubtitleText,Arial,38,&H00FFFF00&,&H00FFFF00&", cyan)
        self.assertIn("Style: SubtitleText,Arial,38,&H0000E6FF&,&H0000E6FF&", yellow)

    def test_page1_mask_mode_uses_black_text_inside_mask_without_duplicate_ass_box(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = write_ass_subtitles(
                [_cue("Xin chào.")],
                Path(temporary) / "page1.ass",
                width=720,
                height=1280,
                page1_mask_bbox=(120, 900, 600, 1010),
            )
            ass = (Path(temporary) / "page1.ass").read_text(encoding="utf-8")
        cue = result["displayCues"][0]
        self.assertGreaterEqual(cue["boxLeftPx"], 120)
        self.assertLessEqual(cue["boxRightPx"], 600)
        self.assertGreaterEqual(cue["boxTopPx"], 900)
        self.assertLessEqual(cue["boxBottomPx"], 1010)
        self.assertIn("Style: SubtitleText,Arial,38,&H00000000&,&H00000000&", ass)
        self.assertNotIn("SubtitleBox", ass)

    def test_page1_mask_mode_rejects_text_that_cannot_fit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(SubtitleLayoutError):
                write_ass_subtitles(
                    [_cue("Một câu rất dài không thể vừa trong hộp hẹp")],
                    Path(temporary) / "page1.ass",
                    width=720,
                    height=1280,
                    page1_mask_bbox=(300, 900, 420, 940),
                )

    def test_rendered_box_is_centered_around_the_text_position(self) -> None:
        policy = SubtitleLayoutPolicy()
        result = layout_subtitles([_cue("Xin chào.", start=0, end=1_000)], policy=policy)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ass = root / "centered.ass"
            frame = root / "frame.png"
            ass.write_text(render_ass_document(result["displayCues"], policy), encoding="utf-8")
            subtitle_path = str(ass).replace("\\", "/").replace(":", r"\:")
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", "color=c=white:s=720x960:r=1:d=1",
                    "-vf", f"subtitles='{subtitle_path}'", "-frames:v", "1", "-y", str(frame),
                ],
                check=True,
            )
            image = Image.open(frame).convert("RGB")
            cue = result["displayCues"][0]
            center = policy.frame_width // 2
            sample_y = int(cue["boxTopPx"]) + 5
            left_sample = (int(cue["boxLeftPx"]) + center) // 2
            right_sample = (center + int(cue["boxRightPx"])) // 2
            self.assertLess(image.getpixel((left_sample, sample_y))[0], 240)
            self.assertLess(image.getpixel((right_sample, sample_y))[0], 240)

    def test_production_layout_logic_is_not_fixture_specific(self) -> None:
        source = Path(__file__).resolve().parents[1] / "subtitle_layout.py"
        text = source.read_text(encoding="utf-8").lower()
        self.assertNotIn("fixture2", text)
        self.assertNotIn("blind_fixture_02", text)
        self.assertNotIn("utt_v1_d_", text)

    def test_invalid_interval_fails_closed(self) -> None:
        with self.assertRaises(SubtitleLayoutError):
            layout_subtitles([_cue("Không hợp lệ", start=500, end=500)])


if __name__ == "__main__":
    unittest.main()
