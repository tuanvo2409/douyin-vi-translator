from __future__ import annotations

import unittest
import hashlib
import tempfile
from pathlib import Path

import cv2
import numpy as np

import localized_masking


class LocalizedMaskPlanTests(unittest.TestCase):
    def test_plan_retains_cjk_geometry_but_omits_ocr_text_from_diagnostics(self) -> None:
        plan = localized_masking.build_mask_plan([
            {"timestampMs": 500, "text": "中文字幕", "bbox": (100, 200, 300, 240), "confidence": 0.9, "category": "DIALOGUE_SUBTITLE"},
            {"timestampMs": 500, "text": "latin", "bbox": (400, 200, 500, 240), "confidence": 0.9},
        ], frame_size=(1080, 1920))
        self.assertEqual(1, len(plan))
        self.assertEqual((94, 190, 306, 250), plan[0]["bbox"])
        self.assertNotIn("text", plan[0])

    def test_mask_plan_excludes_story_unknown_and_unclassified_cjk(self) -> None:
        plan = localized_masking.build_mask_plan([
            {"timestampMs": 100, "text": "剧情标题", "bbox": (100, 700, 300, 740), "category": "STORY_TEXT"},
            {"timestampMs": 200, "text": "无法判断", "bbox": (100, 700, 300, 740), "category": "UNKNOWN"},
            {"timestampMs": 300, "text": "未分类字幕", "bbox": (100, 700, 300, 740)},
        ], frame_size=(1080, 1920))
        self.assertEqual([], plan)

    def test_watermark_like_text_never_becomes_dialogue_mask(self) -> None:
        plan = localized_masking.build_mask_plan([
            {"timestampMs": 100, "text": "频道标记", "bbox": (100, 700, 300, 740), "category": "PERSISTENT_WATERMARK"},
        ], frame_size=(1080, 1920))
        self.assertTrue(plan)
        self.assertNotIn("DIALOGUE_SUBTITLE", {item.get("category") for item in plan})
        self.assertEqual({"WATERMARK"}, {item.get("category") for item in plan})

    def test_clean_plate_is_used_only_for_a_similar_localized_region(self) -> None:
        frame = np.full((80, 100, 3), 120, dtype=np.uint8)
        clean = frame.copy()
        repaired, method = localized_masking.repair_region(frame, (20, 20, 60, 40), clean)
        self.assertEqual("clean_plate", method)
        self.assertTrue(np.array_equal(clean[20:40, 20:60], repaired[20:40, 20:60]))
        frame[25:30, 35:45] = 255
        unrelated = np.zeros_like(frame)
        _, fallback = localized_masking.repair_region(frame, (20, 20, 60, 40), unrelated)
        self.assertIn(fallback, {"inpaint", "plate"})

    def test_repair_masks_text_pixels_without_repainting_the_whole_bbox(self) -> None:
        frame = np.full((80, 120, 3), 96, dtype=np.uint8)
        frame[35:39, 45:75] = 255
        original_background = frame[12:25, 12:25].copy()
        repaired, method = localized_masking.repair_region(frame, (10, 10, 100, 50), None)
        self.assertIn(method, {"inpaint", "plate"})
        self.assertTrue(np.array_equal(original_background, repaired[12:25, 12:25]))
        self.assertLess(float(np.mean(repaired[35:39, 45:75])), 240.0)

    def test_repair_masks_dark_outline_around_bright_subtitle_glyphs(self) -> None:
        frame = np.full((80, 120, 3), 220, dtype=np.uint8)
        cv2.rectangle(frame, (42, 30), (78, 50), (0, 0, 0), -1)
        cv2.rectangle(frame, (46, 34), (74, 46), (0, 220, 255), -1)

        repaired, method = localized_masking.repair_region(frame, (38, 26, 82, 54), None)

        self.assertIn(method, {"inpaint", "plate"})
        self.assertGreater(float(np.mean(repaired[30:50, 42:78])), 210.0)

    def test_watermark_repair_removes_dim_glyphs_within_its_category(self) -> None:
        frame = np.full((100, 180, 3), 110, dtype=np.uint8)
        frame[28:42, 42:142] = 160
        repaired, method = localized_masking.repair_region(frame, (20, 20, 165, 50), None, category="WATERMARK")
        self.assertIn(method, {"inpaint", "plate"})
        self.assertLess(float(np.mean(repaired[28:42, 42:142])), 150.0)

    def test_mask_geometry_padding_is_bounded_and_clipped(self) -> None:
        bounded = localized_masking.dilate_region((100, 100, 120, 120), (200, 200))
        self.assertLessEqual(bounded[0] - 100, 0)
        self.assertLessEqual(120 - bounded[2], 0)
        self.assertLessEqual(bounded[2] - bounded[0], 30)
        self.assertLessEqual(bounded[3] - bounded[1], 32)
        self.assertEqual((0, 0, 200, 200), localized_masking.dilate_region((-40, -20, 220, 240), (200, 200)))

    def test_story_and_unknown_categories_are_preserved(self) -> None:
        points, intervals = localized_masking._partition_mask_plan([
            {"timestampMs": 100, "bbox": (10, 10, 20, 20), "category": "STORY_TEXT"},
            {"startMs": 100, "endMs": 200, "bbox": (10, 10, 20, 20), "category": "UNKNOWN"},
        ])
        self.assertEqual([], points)
        self.assertEqual([], intervals)

    def test_same_category_tracks_merge_but_different_categories_do_not(self) -> None:
        stabilized = localized_masking.stabilize_mask_plan([
            {"timestampMs": 100, "bbox": (10, 10, 30, 20), "category": "WATERMARK"},
            {"timestampMs": 200, "bbox": (12, 10, 32, 20), "category": "WATERMARK"},
            {"timestampMs": 250, "bbox": (12, 10, 32, 20), "category": "DIALOGUE_SUBTITLE"},
        ], (100, 100), lead_ms=0, tail_ms=0, max_gap_ms=500)
        self.assertEqual(2, len(stabilized))
        self.assertEqual("WATERMARK", stabilized[0]["category"])
        self.assertEqual("DIALOGUE_SUBTITLE", stabilized[1]["category"])

    def test_watermark_track_is_stable_but_temporally_bounded(self) -> None:
        stabilized = localized_masking.stabilize_mask_plan([
            {"timestampMs": 1000, "bbox": (10, 10, 30, 20), "category": "WATERMARK"},
            {"timestampMs": 1400, "bbox": (11, 10, 31, 20), "category": "WATERMARK"},
        ], (100, 100), lead_ms=100, tail_ms=200, max_gap_ms=500)
        self.assertEqual([{
            "startMs": 900, "endMs": 1600, "bbox": (10, 10, 31, 20), "category": "WATERMARK",
        }], stabilized)

    def test_production_masking_has_no_fixture_specific_rules(self) -> None:
        source = Path(localized_masking.__file__).read_text(encoding="utf-8")
        for marker in ("BLIND_FIXTURE_02", "DUBVI_FIXTURE2", "豆包AI生成"):
            self.assertNotIn(marker, source)

    def test_qc_document_is_bounded_and_secret_safe(self) -> None:
        document = localized_masking.build_mask_qc(plan_count=100, applied_count=99, residual_count=1)
        self.assertEqual({"schemaVersion", "plannedRegions", "appliedRegions", "residualCjkRegions", "needsReview"}, set(document))
        self.assertTrue(document["needsReview"])
        self.assertEqual(100, document["plannedRegions"])

    def test_short_spatially_matching_detection_gap_becomes_one_bounded_interval(self) -> None:
        stabilized = localized_masking.stabilize_mask_plan([
            {"timestampMs": 300, "bbox": (10, 10, 30, 20)},
            {"timestampMs": 700, "bbox": (12, 10, 32, 20)},
        ], (100, 100), lead_ms=200, tail_ms=300, max_gap_ms=500)
        self.assertEqual([{"startMs": 100, "endMs": 1000, "bbox": (10, 10, 32, 20)}], stabilized)

    def test_temporal_stabilization_never_merges_distant_or_separate_regions(self) -> None:
        stabilized = localized_masking.stabilize_mask_plan([
            {"timestampMs": 100, "bbox": (10, 10, 30, 20)},
            {"timestampMs": 200, "bbox": (60, 10, 80, 20)},
            {"timestampMs": 2000, "bbox": (10, 10, 30, 20)},
        ], (100, 100), lead_ms=0, tail_ms=0, max_gap_ms=500)
        self.assertEqual(3, len(stabilized))
        self.assertEqual((10, 10, 30, 20), stabilized[0]["bbox"])

    def test_interval_mask_applies_only_inside_its_bounded_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "source.mp4", root / "masked.mkv"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 2, (64, 48))
            for _ in range(4):
                writer.write(np.full((48, 64, 3), 120, dtype=np.uint8))
            writer.release()
            qc = localized_masking.apply_localized_masks(source, target, [{"startMs": 500, "endMs": 900, "bbox": (10, 10, 30, 25)}])
            self.assertEqual(1, qc["appliedRegions"])

    def test_active_interval_never_uses_an_unmasked_prior_frame_as_a_clean_plate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "source.mp4", root / "masked.mkv"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 2, (64, 48))
            frame = np.full((48, 64, 3), 120, dtype=np.uint8)
            frame[14:19, 16:25] = 255
            for _ in range(4):
                writer.write(frame)
            writer.release()
            localized_masking.apply_localized_masks(source, target, [{"startMs": 500, "endMs": 900, "bbox": (10, 10, 30, 25)}])
            capture = cv2.VideoCapture(str(target))
            capture.read()  # The unmasked frame at 0 ms establishes the old false clean plate.
            ok, repaired = capture.read()
            capture.release()
            self.assertTrue(ok)
            self.assertLess(float(np.mean(repaired[10:25, 10:30])), 200.0)

    def test_masked_video_is_local_and_source_preserving(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "source.mp4", root / "masked.mkv"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 2, (64, 48))
            for _ in range(4):
                writer.write(np.full((48, 64, 3), 120, dtype=np.uint8))
            writer.release()
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            qc = localized_masking.apply_localized_masks(source, target, [{"timestampMs": 500, "bbox": (10, 10, 30, 25)}])
            self.assertTrue(target.is_file() and target.stat().st_size > 0)
            self.assertEqual(before, hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertGreater(qc["appliedRegions"], 0)


if __name__ == "__main__":
    unittest.main()
