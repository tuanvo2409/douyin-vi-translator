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
        self.assertLessEqual(document["plannedRegions"], 64)

    def test_masked_video_is_local_and_source_preserving(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "source.mp4", root / "masked.mp4"
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
