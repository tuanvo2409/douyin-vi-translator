from __future__ import annotations

import unittest

from delivery_planner import DeliveryPolicy, VisualThreshold, plan_utterance


def _utterance(
    identifier: str = "u1",
    *,
    semantic_start: int = 0,
    semantic_end: int = 1000,
    tts_ms: int = 800,
    speech_start: int | None = None,
    speech_end: int | None = None,
    confidence: str = "UNKNOWN",
    next_speech: int | None = None,
    visual_class: str = "TALKING_HEAD",
    flags: list[str] | None = None,
) -> dict[str, object]:
    return {
        "utteranceId": identifier,
        "semanticStartMs": semantic_start,
        "semanticEndMs": semantic_end,
        "speechStartMs": speech_start,
        "speechEndMs": speech_end,
        "speechTimingConfidence": confidence,
        "sourcePositions": [identifier],
        "speakerRef": None,
        "approvedNativeVi": "Approved line",
        "deliveryNativeVi": "Delivery line",
        "ttsMeasuredMs": tts_ms,
        "sourceFlags": flags or [],
        "previousUtteranceId": None,
        "nextUtteranceId": None,
        "nextSpeechStartMs": next_speech,
        "visualClass": visual_class,
        "visualConfidence": "HIGH",
    }


class DeliveryPlannerTests(unittest.TestCase):
    def test_fitting_utterance_keeps_base_window(self) -> None:
        plan = plan_utterance(_utterance(tts_ms=900))
        self.assertEqual("KEEP", plan["action"])
        self.assertEqual(1.0, plan["tempo"])
        self.assertEqual(1.0, plan["visualSpeed"])
        self.assertIn("FITS_BASE_WINDOW", plan["reasonCodes"])

    def test_slight_overflow_uses_tempo(self) -> None:
        plan = plan_utterance(_utterance(tts_ms=1100))
        self.assertEqual("TEMPO", plan["action"])
        self.assertEqual(1.1, plan["tempo"])
        self.assertLessEqual(plan["tempo"], 1.12)

    def test_safe_speech_gap_is_borrowed(self) -> None:
        plan = plan_utterance(_utterance(tts_ms=1400, next_speech=1300))
        self.assertEqual("BORROW_GAP", plan["action"])
        self.assertEqual(300, plan["availableGapMs"])
        self.assertEqual(250, plan["borrowedGapMs"])

    def test_gap_borrowing_uses_only_required_gap(self) -> None:
        plan = plan_utterance(_utterance(tts_ms=1200, next_speech=1600))
        self.assertEqual("BORROW_GAP", plan["action"])
        self.assertEqual(600, plan["availableGapMs"])
        self.assertEqual(72, plan["borrowedGapMs"])

    def test_gap_cannot_cross_next_spoken_utterance(self) -> None:
        plan = plan_utterance(_utterance(tts_ms=2500, next_speech=1100))
        self.assertEqual(100, plan["availableGapMs"])
        self.assertLessEqual(plan["effectiveWindowMs"], 1100)

    def test_talking_head_mild_slowdown_is_visual_action(self) -> None:
        plan = plan_utterance(_utterance(tts_ms=1250, visual_class="TALKING_HEAD"))
        self.assertEqual("SLOW_VISUAL", plan["action"])
        self.assertGreaterEqual(plan["visualSpeed"], 0.85)

    def test_talking_head_strong_slowdown_requests_delivery_rewrite(self) -> None:
        plan = plan_utterance(_utterance(semantic_end=3000, tts_ms=5000, visual_class="TALKING_HEAD"))
        self.assertEqual("DELIVERY_REWRITE", plan["action"])
        self.assertTrue(plan["needsDeliveryRewrite"])
        self.assertIn("targetDurationMs", plan["deliveryRewriteRequest"])

    def test_broll_allows_stronger_slowdown(self) -> None:
        plan = plan_utterance(_utterance(semantic_end=2000, tts_ms=3000, visual_class="BROLL"))
        self.assertEqual("SLOW_VISUAL", plan["action"])
        self.assertGreaterEqual(plan["visualSpeed"], 0.65)

    def test_action_class_is_conservative(self) -> None:
        plan = plan_utterance(_utterance(semantic_end=2000, tts_ms=3500, visual_class="ACTION"))
        self.assertIn(plan["action"], {"DELIVERY_REWRITE", "REVIEW_REQUIRED"})

    def test_source_uncertain_requires_review(self) -> None:
        plan = plan_utterance(_utterance(flags=["SOURCE_UNCERTAIN"], tts_ms=500))
        self.assertEqual("REVIEW_REQUIRED", plan["action"])
        self.assertTrue(plan["reviewRequired"])
        self.assertIn("SOURCE_UNCERTAIN", plan["reasonCodes"])

    def test_suspicious_short_semantic_window_refines_speech_window(self) -> None:
        plan = plan_utterance(
            _utterance(semantic_end=500, tts_ms=1800, confidence="LOW")
        )
        self.assertEqual("REFINE_SPEECH_WINDOW", plan["action"])
        self.assertTrue(plan["needsSpeechWindowRefinement"])
        self.assertIn("TIMING_EVIDENCE_SUSPICIOUS", plan["reasonCodes"])

    def test_measured_audio_uses_visual_delivery_when_semantic_window_is_short(self) -> None:
        item = _utterance(semantic_end=480, tts_ms=1344)
        item["audioPath"] = "audio/u1.mp3"

        plan = plan_utterance(item)

        self.assertEqual("DELIVERY_REWRITE", plan["action"])
        self.assertFalse(plan["needsSpeechWindowRefinement"])
        self.assertLess(plan["visualSpeed"], 0.5)

    def test_reliable_wider_speech_window_overrides_semantic_window(self) -> None:
        plan = plan_utterance(
            _utterance(
                semantic_end=500,
                tts_ms=1200,
                speech_start=400,
                speech_end=2200,
                confidence="HIGH",
            )
        )
        self.assertEqual("speech", plan["timingBasis"])
        self.assertEqual(1800, plan["baseWindowMs"])
        self.assertEqual("KEEP", plan["action"])

    def test_planner_preserves_identity_and_approved_text(self) -> None:
        item = _utterance(identifier="trace-7")
        plan = plan_utterance(item)
        self.assertEqual("trace-7", plan["utteranceId"])
        self.assertEqual("Approved line", plan["approvedNativeVi"])
        self.assertEqual("Delivery line", plan["deliveryNativeVi"])

    def test_delivery_rewrite_is_a_request_not_a_provider_call(self) -> None:
        plan = plan_utterance(_utterance(semantic_end=3000, tts_ms=5000))
        self.assertEqual("DELIVERY_REWRITE", plan["action"])
        self.assertNotIn("provider", plan["deliveryRewriteRequest"])
        self.assertNotIn("model", plan["deliveryRewriteRequest"])

    def test_policy_is_injectable(self) -> None:
        policy = DeliveryPolicy(max_tempo=1.05)
        plan = plan_utterance(_utterance(tts_ms=1100), policy=policy)
        self.assertEqual("SLOW_VISUAL", plan["action"])
        self.assertEqual(1.05, plan["tempo"])

    def test_custom_visual_class_policy_is_used(self) -> None:
        policy = DeliveryPolicy(
            visual_classes={
                "UNKNOWN": VisualThreshold(0.80, 0.65),
                "CINEMATIC": VisualThreshold(0.90, 0.80),
            }
        )
        plan = plan_utterance(
            _utterance(semantic_end=1000, tts_ms=1500, visual_class="CINEMATIC"),
            policy=policy,
        )
        self.assertEqual("DELIVERY_REWRITE", plan["action"])


if __name__ == "__main__":
    unittest.main()
