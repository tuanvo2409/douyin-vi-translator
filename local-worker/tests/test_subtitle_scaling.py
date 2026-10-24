from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dubvi_subtitle import (
    draw_ass,
    generate_clean_ass_subtitles,
    render_clean_ass,
)


class SubtitleScalingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.ass_path = Path(self.temp_dir.name) / "test_subtitles.ass"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_720p_source_ycenterpx_scaled_to_1080x1920_canvas(self) -> None:
        """Test 720x1280 source with yCenterPx=960 (75%) scales to 1440 on 1080x1920 canvas."""
        segments = [
            {
                "startMs": 1000,
                "endMs": 3000,
                "translatedTextVi": "Đây là câu thoại kiểm tra độ phân giải 720p",
                "yCenterPx": 960,
            }
        ]
        roi = {"yCenterPx": 960}
        draw_ass(segments, 720, 1280, roi, self.ass_path, source_h=1280)

        content = self.ass_path.read_text(encoding="utf-8")
        self.assertIn("PlayResX: 1080", content)
        self.assertIn("PlayResY: 1920", content)

        # Center Y for text should be 1440 (75% of 1920), not 960
        self.assertIn(r"\pos(540,1440)", content)
        self.assertNotIn(r"\pos(540,960)", content)

        # Box top should be common_cy - 80 = 1440 - 80 = 1360, not 960 - 80 = 880
        self.assertIn(r"\pos(80,1360)", content)
        self.assertNotIn(r"\pos(80,880)", content)

    def test_720p_source_auto_detects_source_h_from_roi_or_video_height(self) -> None:
        """Test scaling occurs when source_h is given in roi or inferred from video_height."""
        segments = [
            {
                "startMs": 500,
                "endMs": 2500,
                "translatedTextVi": "Kiểm tra tự động nhận diện sourceHeight",
                "yCenterPx": 960,
            }
        ]
        roi = {"yCenterPx": 960, "sourceHeight": 1280}
        draw_ass(segments, 720, 1280, roi, self.ass_path)

        content = self.ass_path.read_text(encoding="utf-8")
        self.assertIn("PlayResY: 1920", content)
        self.assertIn(r"\pos(540,1440)", content)
        self.assertNotIn(r"\pos(540,960)", content)

    def test_1080p_source_ycenterpx_remains_1440(self) -> None:
        """Test 1080x1920 source with yCenterPx=1440 remains 1440 (no scaling needed)."""
        segments = [
            {
                "startMs": 1000,
                "endMs": 4000,
                "translatedTextVi": "Video gốc chuẩn 1080p không cần co giãn tọa độ",
                "yCenterPx": 1440,
            }
        ]
        roi = {"yCenterPx": 1440}
        draw_ass(segments, 1080, 1920, roi, self.ass_path, source_h=1920)

        content = self.ass_path.read_text(encoding="utf-8")
        self.assertIn("PlayResX: 1080", content)
        self.assertIn("PlayResY: 1920", content)
        self.assertIn(r"\pos(540,1440)", content)
        self.assertIn(r"\pos(80,1360)", content)

    def test_ypercent_only_produces_1440_on_1920p_canvas(self) -> None:
        """Test that when only yPercent = 75.0 is provided, Y coordinate is 1440 on 1920p canvas."""
        segments = [
            {
                "startMs": 200,
                "endMs": 1800,
                "translatedTextVi": "Tọa độ theo tỷ lệ phần trăm",
                "yPercent": 75.0,
            }
        ]
        roi = {"yPercent": 75.0}
        draw_ass(segments, 720, 1280, roi, self.ass_path, source_h=1280)

        content = self.ass_path.read_text(encoding="utf-8")
        self.assertIn("PlayResY: 1920", content)
        self.assertIn(r"\pos(540,1440)", content)
        self.assertIn(r"\pos(80,1360)", content)

    def test_roi_fallback_when_segments_have_no_coordinates(self) -> None:
        """Test fallback to scaled roi center_y when segments lack coordinate keys."""
        segments = [
            {
                "startMs": 1000,
                "endMs": 2000,
                "translatedTextVi": "Segment không có tọa độ Y",
            }
        ]
        roi = {"yCenterPx": 960}
        draw_ass(segments, 720, 1280, roi, self.ass_path, source_h=1280)

        content = self.ass_path.read_text(encoding="utf-8")
        self.assertIn(r"\pos(540,1440)", content)
        self.assertNotIn(r"\pos(540,960)", content)

    def test_aliases_generate_clean_ass_subtitles_and_render_clean_ass(self) -> None:
        """Test generate_clean_ass_subtitles and render_clean_ass work identically to draw_ass."""
        segments = [
            {
                "startMs": 0,
                "endMs": 2000,
                "translatedTextVi": "Kiểm tra alias hàm phụ đề",
                "yCenterPx": 960,
            }
        ]
        roi = {"yCenterPx": 960}

        path_alias1 = Path(self.temp_dir.name) / "alias1.ass"
        generate_clean_ass_subtitles(segments, 720, 1280, roi, path_alias1, source_h=1280)
        self.assertIn(r"\pos(540,1440)", path_alias1.read_text(encoding="utf-8"))

        path_alias2 = Path(self.temp_dir.name) / "alias2.ass"
        render_clean_ass(segments, 720, 1280, roi, path_alias2, source_h=1280)
        self.assertIn(r"\pos(540,1440)", path_alias2.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
