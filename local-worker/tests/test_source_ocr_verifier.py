import unittest

from source_ocr_verifier import decide_source, infer_stable_subtitle_roi


class SourceOCRVerifierTests(unittest.TestCase):
    def test_missing_ocr_cannot_validate_high_confidence_asr(self):
        result = decide_source("空间太小", [])
        self.assertEqual("ASR_ONLY", result["decision"])
        self.assertTrue(result["needsReview"])

    def test_repeated_lower_caption_defines_scan_roi_without_top_story_text(self):
        detections = [
            {"timestampMs": 0, "bbox": [32, 88, 539, 315], "text": "两个孩子一间房"},
            {"timestampMs": 1000, "bbox": [150, 880, 400, 915], "text": "老婆生了龙凤胎"},
            {"timestampMs": 5000, "bbox": [170, 884, 405, 920], "text": "空间太小"},
            {"timestampMs": 9000, "bbox": [160, 882, 410, 916], "text": "不如这样改"},
        ]
        roi = infer_stable_subtitle_roi(detections, (576, 1024))
        self.assertIsNotNone(roi)
        self.assertGreater(roi[1], 750)
        self.assertLess(roi[1], 880)
        self.assertGreater(roi[3], 920)

    def test_no_stable_region_does_not_guess_a_crop(self):
        self.assertIsNone(infer_stable_subtitle_roi([
            {"timestampMs": 0, "bbox": [10, 800, 100, 830], "text": "你好"},
        ], (576, 1024)))

    def test_repeated_static_lower_watermark_is_not_dialogue_roi(self):
        detections = [
            {"timestampMs": n * 1000, "bbox": [150, 880, 400, 915], "text": "关注我的账号", "category": "UNKNOWN"}
            for n in range(4)
        ]
        self.assertIsNone(infer_stable_subtitle_roi(detections, (576, 1024)))

    def test_repeated_ocr_alone_cannot_overrule_independent_asr(self):
        result = decide_source("老婆一不小心生了龙胖胎", [
            {"text": "老婆一不小心生了龙凤胎", "confidence": 0.98},
            {"text": "老婆一不小心生了龙凤胎", "confidence": 0.97},
        ])
        self.assertEqual("老婆一不小心生了龙胖胎", result["resolvedZh"])
        self.assertEqual("SOURCE_REVIEW_REQUIRED", result["decision"])
        self.assertTrue(result["needsReview"])

    def test_second_asr_agreement_can_promote_ocr(self):
        result = decide_source("内面墙麻省网格固定", [
            {"text": "内面墙麻绳网格固定", "confidence": 0.98},
            {"text": "内面墙麻绳网格固定", "confidence": 0.97},
        ], second_asr="内面墙麻绳网格固定")
        self.assertEqual("内面墙麻绳网格固定", result["resolvedZh"])
        self.assertEqual("OCR_SECOND_ASR_AGREE", result["decision"])

    def test_digit_corruption_cannot_override_source_even_with_second_asr(self):
        result = decide_source("老婆一不小心生了龙胖胎", [
            {"text": "老婆一不小心生7龙凤胎", "confidence": 0.99},
            {"text": "老婆一不小心生7龙凤胎", "confidence": 0.99},
        ], second_asr="老婆一不小心生了龙凤胎")
        self.assertTrue(result["needsReview"])
        self.assertEqual("老婆一不小心生了龙胖胎", result["resolvedZh"])

    def test_caption_spanning_next_cue_is_review_not_replacement(self):
        result = decide_source("再去二舅家", [
            {"text": "再去二舅家搬个松松软软的床垫", "confidence": 0.99},
            {"text": "再去二舅家搬个松松软软的床垫", "confidence": 0.98},
        ])
        self.assertEqual("再去二舅家", result["resolvedZh"])
        self.assertTrue(result["needsReview"])

    def test_unstable_ocr_does_not_silently_override_asr(self):
        result = decide_source("找二舅妈借一堆横丈螺丝", [
            {"text": "找二舅妈借一堆", "confidence": 0.97},
            {"text": "找二舅妈借一堆膨胀", "confidence": 0.96},
        ])
        self.assertEqual("找二舅妈借一堆横丈螺丝", result["resolvedZh"])
        self.assertTrue(result["needsReview"])

    def test_verified_repeated_dialogue_ocr_accepts_ocr_repeat(self):
        readings = [
            {"text": "放冬天的棉被", "confidence": 0.96, "timestampMs": 1200, "bbox": [100, 800, 450, 840], "category": "DIALOGUE_SUBTITLE"},
            {"text": "放冬天的棉被", "confidence": 0.95, "timestampMs": 1600, "bbox": [100, 800, 450, 840], "category": "DIALOGUE_SUBTITLE"},
        ]
        result = decide_source("放冬天的免费", readings, cue_timing=(1000, 2000), roi=(50, 750, 500, 900))
        self.assertEqual("放冬天的棉被", result["resolvedZh"])
        self.assertEqual("放冬天的棉被", result["ocrZh"])
        self.assertEqual("OCR_REPEAT", result["decision"])
        self.assertFalse(result["needsReview"])

    def test_duplicate_frame_ocr_returns_review(self):
        readings = [
            {"text": "放冬天的棉被", "confidence": 0.96, "timestampMs": 1200, "bbox": [100, 800, 450, 840], "category": "DIALOGUE_SUBTITLE"},
            {"text": "放冬天的棉被", "confidence": 0.95, "timestampMs": 1200, "bbox": [100, 800, 450, 840], "category": "DIALOGUE_SUBTITLE"},
        ]
        result = decide_source("放冬天的免费", readings, cue_timing=(1000, 2000), roi=(50, 750, 500, 900))
        self.assertEqual("放冬天的免费", result["resolvedZh"])
        self.assertEqual("SOURCE_REVIEW_REQUIRED", result["decision"])
        self.assertTrue(result["needsReview"])

    def test_missing_bbox_or_geometry_returns_review(self):
        readings = [
            {"text": "放冬天的棉被", "confidence": 0.96, "timestampMs": 1200, "category": "DIALOGUE_SUBTITLE"},
            {"text": "放冬天的棉被", "confidence": 0.95, "timestampMs": 1600, "category": "DIALOGUE_SUBTITLE"},
        ]
        result = decide_source("放冬天的免费", readings, cue_timing=(1000, 2000), roi=(50, 750, 500, 900))
        self.assertEqual("放冬天的免费", result["resolvedZh"])
        self.assertEqual("SOURCE_REVIEW_REQUIRED", result["decision"])
        self.assertTrue(result["needsReview"])

    def test_unresolved_unknown_category_returns_review(self):
        readings = [
            {"text": "放冬天的棉被", "confidence": 0.96, "timestampMs": 1200, "bbox": [100, 800, 450, 840], "category": "UNKNOWN"},
            {"text": "放冬天的棉被", "confidence": 0.95, "timestampMs": 1600, "bbox": [100, 800, 450, 840], "category": "UNKNOWN"},
        ]
        result = decide_source("放冬天的免费", readings, cue_timing=(1000, 2000), roi=(50, 750, 500, 900))
        self.assertEqual("放冬天的免费", result["resolvedZh"])
        self.assertEqual("SOURCE_REVIEW_REQUIRED", result["decision"])
        self.assertTrue(result["needsReview"])

    def test_story_text_category_returns_review(self):
        readings = [
            {"text": "放冬天的棉被", "confidence": 0.96, "timestampMs": 1200, "bbox": [100, 800, 450, 840], "category": "STORY_TEXT"},
            {"text": "放冬天的棉被", "confidence": 0.95, "timestampMs": 1600, "bbox": [100, 800, 450, 840], "category": "STORY_TEXT"},
        ]
        result = decide_source("放冬天的免费", readings, cue_timing=(1000, 2000), roi=(50, 750, 500, 900))
        self.assertEqual("放冬天的免费", result["resolvedZh"])
        self.assertEqual("SOURCE_REVIEW_REQUIRED", result["decision"])
        self.assertTrue(result["needsReview"])

    def test_watermark_category_returns_review(self):
        readings = [
            {"text": "放冬天的棉被", "confidence": 0.96, "timestampMs": 1200, "bbox": [100, 800, 450, 840], "category": "WATERMARK"},
            {"text": "放冬天的棉被", "confidence": 0.95, "timestampMs": 1600, "bbox": [100, 800, 450, 840], "category": "WATERMARK"},
        ]
        result = decide_source("放冬天的免费", readings, cue_timing=(1000, 2000), roi=(50, 750, 500, 900))
        self.assertEqual("放冬天的免费", result["resolvedZh"])
        self.assertEqual("SOURCE_REVIEW_REQUIRED", result["decision"])
        self.assertTrue(result["needsReview"])

    def test_low_confidence_ocr_returns_review(self):
        readings = [
            {"text": "放冬天的棉被", "confidence": 0.89, "timestampMs": 1200, "bbox": [100, 800, 450, 840], "category": "DIALOGUE_SUBTITLE"},
            {"text": "放冬天的棉被", "confidence": 0.95, "timestampMs": 1600, "bbox": [100, 800, 450, 840], "category": "DIALOGUE_SUBTITLE"},
        ]
        result = decide_source("放冬天的免费", readings, cue_timing=(1000, 2000), roi=(50, 750, 500, 900))
        self.assertEqual("放冬天的免费", result["resolvedZh"])
        self.assertEqual("SOURCE_REVIEW_REQUIRED", result["decision"])
        self.assertTrue(result["needsReview"])

    def test_conflicting_ocr_texts_returns_review(self):
        readings = [
            {"text": "放冬天的棉被", "confidence": 0.96, "timestampMs": 1200, "bbox": [100, 800, 450, 840], "category": "DIALOGUE_SUBTITLE"},
            {"text": "放冬天的毛毯", "confidence": 0.95, "timestampMs": 1600, "bbox": [100, 800, 450, 840], "category": "DIALOGUE_SUBTITLE"},
        ]
        result = decide_source("放冬天的免费", readings, cue_timing=(1000, 2000), roi=(50, 750, 500, 900))
        self.assertEqual("放冬天的免费", result["resolvedZh"])
        self.assertEqual("SOURCE_REVIEW_REQUIRED", result["decision"])
        self.assertTrue(result["needsReview"])

    def test_digit_conflict_in_repeated_ocr_returns_review(self):
        readings = [
            {"text": "买3套房子", "confidence": 0.96, "timestampMs": 1200, "bbox": [100, 800, 450, 840], "category": "DIALOGUE_SUBTITLE"},
            {"text": "买3套房子", "confidence": 0.95, "timestampMs": 1600, "bbox": [100, 800, 450, 840], "category": "DIALOGUE_SUBTITLE"},
        ]
        result = decide_source("买2套房子", readings, cue_timing=(1000, 2000), roi=(50, 750, 500, 900))
        self.assertEqual("买2套房子", result["resolvedZh"])
        self.assertEqual("SOURCE_REVIEW_REQUIRED", result["decision"])
        self.assertTrue(result["needsReview"])

    def test_clipped_ocr_line_returns_review(self):
        readings = [
            {"text": "借一堆", "confidence": 0.96, "timestampMs": 1200, "bbox": [100, 800, 450, 840], "category": "DIALOGUE_SUBTITLE"},
            {"text": "借一堆", "confidence": 0.95, "timestampMs": 1600, "bbox": [100, 800, 450, 840], "category": "DIALOGUE_SUBTITLE"},
        ]
        result = decide_source("找二舅妈借一堆横丈螺丝", readings, cue_timing=(1000, 2000), roi=(50, 750, 500, 900))
        self.assertEqual("找二舅妈借一堆横丈螺丝", result["resolvedZh"])
        self.assertEqual("SOURCE_REVIEW_REQUIRED", result["decision"])
        self.assertTrue(result["needsReview"])

    def test_cross_cue_timestamp_sample_returns_review(self):
        readings = [
            {"text": "放冬天的棉被", "confidence": 0.96, "timestampMs": 1200, "bbox": [100, 800, 450, 840], "category": "DIALOGUE_SUBTITLE"},
            {"text": "放冬天的棉被", "confidence": 0.95, "timestampMs": 2500, "bbox": [100, 800, 450, 840], "category": "DIALOGUE_SUBTITLE"},
        ]
        result = decide_source("放冬天的免费", readings, cue_timing=(1000, 2000), roi=(50, 750, 500, 900))
        self.assertEqual("放冬天的免费", result["resolvedZh"])
        self.assertEqual("SOURCE_REVIEW_REQUIRED", result["decision"])
        self.assertTrue(result["needsReview"])

    def test_ocr_bbox_outside_subtitle_roi_returns_review(self):
        readings = [
            {"text": "放冬天的棉被", "confidence": 0.96, "timestampMs": 1200, "bbox": [100, 100, 450, 150], "category": "DIALOGUE_SUBTITLE"},
            {"text": "放冬天的棉被", "confidence": 0.95, "timestampMs": 1600, "bbox": [100, 100, 450, 150], "category": "DIALOGUE_SUBTITLE"},
        ]
        result = decide_source("放冬天的免费", readings, cue_timing=(1000, 2000), roi=(50, 750, 500, 900))
        self.assertEqual("放冬天的免费", result["resolvedZh"])
        self.assertEqual("SOURCE_REVIEW_REQUIRED", result["decision"])
        self.assertTrue(result["needsReview"])


if __name__ == "__main__":
    unittest.main()

