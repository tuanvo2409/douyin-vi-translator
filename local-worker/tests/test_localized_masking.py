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
            {"timestampMs": 500, "text": "中文字幕", "bbox": (100, 200, 300, 240), "confidence": 0.9},
            {"timestampMs": 500, "text": "latin", "bbox": (400, 200, 500, 240), "confidence": 0.9},
        ], frame_size=(1080, 1920))
        self.assertEqual(1, len(plan))
        self.assertEqual((94, 190, 306, 250), plan[0]["bbox"])
        self.assertNotIn("text", plan[0])

    def test_clean_plate_is_used_only_for_a_similar_localized_region(self) -> None:
        frame = np.full((80, 100, 3), 120, dtype=np.uint8)
        clean = frame.copy()
        repaired, method = localized_masking.repair_region(frame, (20, 20, 60, 40), clean)
        self.assertEqual("clean_plate", method)
        self.assertTrue(np.array_equal(clean[20:40, 20:60], repaired[20:40, 20:60]))
        unrelated = np.zeros_like(frame)
        _, fallback = localized_masking.repair_region(frame, (20, 20, 60, 40), unrelated)
        self.assertIn(fallback, {"inpaint", "plate"})

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
            frame[10:25, 10:30] = 255
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
