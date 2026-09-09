from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import requests


WORKER_DIR = Path(__file__).resolve().parents[1]
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from content_policy import (
    ContentPolicyConfig,
    ContentPolicyInput,
    PolicyDecision,
    PolicyStageResult,
    combine_stage_results,
    evaluate_content_policy,
    evaluate_prefilter,
    judge_script,
)
from dubvi_worker import Settings


class FakeResponse:
    def __init__(self, payload: dict | None = None, status_code: int = 200) -> None:
        self.payload = payload or {}
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            error = requests.exceptions.HTTPError(f"HTTP {self.status_code}")
            error.response = self
            raise error

    def json(self) -> dict:
        return self.payload


def judge_payload(scores: dict[str, float], critique: str = "Clear narrative.") -> dict:
    return {
        "candidates": [{"content": {"parts": [{"text": json.dumps({
            "hook_score": scores["hook"],
            "conflict_score": scores["conflict"],
            "pacing_score": scores["pacing"],
            "relatability_score": scores["relatability"],
            "critique": critique,
        })}]}}],
    }


class ContentPolicyTests(unittest.TestCase):
    def acceptable_input(self, **overrides: object) -> ContentPolicyInput:
        values: dict[str, object] = {
            "duration_seconds": 60.0,
            "segments": (
                {"startMs": 0, "endMs": 25000},
                {"startMs": 28000, "endMs": 58000},
            ),
            "transcript": "室友又把房间弄得一团糟",
            "title": "室友日常",
        }
        values.update(overrides)
        return ContentPolicyInput(**values)

    def test_acceptable_content_passes_local_and_trust_stages(self) -> None:
        result = evaluate_content_policy(self.acceptable_input())

        self.assertEqual(PolicyDecision.PASS, result.overall_decision)
        self.assertEqual(["pass", "pass"], [stage.decision.value for stage in result.stage_results])

    def test_hard_commercial_content_rejects(self) -> None:
        result = evaluate_content_policy(self.acceptable_input(title="小黄车限时优惠", transcript="点击左下角下单"))

        self.assertEqual(PolicyDecision.REJECT, result.overall_decision)
        self.assertIn("HARD_COMMERCE_SIGNAL", result.reason_codes)

    def test_photo_metadata_rejects(self) -> None:
        result = evaluate_content_policy(self.acceptable_input(metadata={"is_photo": True}))

        self.assertEqual(PolicyDecision.REJECT, result.overall_decision)
        self.assertIn("PHOTO_MEDIA_METADATA", result.reason_codes)

    def test_photo_slideshow_text_requires_review(self) -> None:
        result = evaluate_content_policy(self.acceptable_input(title="图文日常"))

        self.assertEqual(PolicyDecision.REVIEW, result.overall_decision)
        self.assertIn("PHOTO_FORMAT_TEXT_SIGNAL", result.reason_codes)

    def test_low_speech_density_requires_review(self) -> None:
        result = evaluate_content_policy(self.acceptable_input(
            duration_seconds=100.0,
            segments=({"startMs": 0, "endMs": 10000},),
        ))

        self.assertEqual(PolicyDecision.REVIEW, result.overall_decision)
        self.assertIn("LOW_SPEECH_DENSITY", result.reason_codes)

    def test_empty_transcript_never_accidentally_passes(self) -> None:
        result = evaluate_content_policy(self.acceptable_input(transcript="", title="", description=""))

        self.assertEqual(PolicyDecision.REVIEW, result.overall_decision)
        self.assertIn("TRUST_EVIDENCE_MISSING", result.reason_codes)

    def test_invalid_prefilter_evidence_returns_error(self) -> None:
        result = evaluate_prefilter(self.acceptable_input(duration_seconds="not-a-duration"))

        self.assertEqual(PolicyDecision.ERROR, result.decision)
        self.assertIn("INVALID_DURATION", result.reason_codes)

    def test_invalid_segment_evidence_returns_error_not_pass(self) -> None:
        result = evaluate_prefilter(self.acceptable_input(segments=({"startMs": 0, "endMs": "broken"},)))

        self.assertEqual(PolicyDecision.ERROR, result.decision)
        self.assertIn("SPEECH_METRICS_INVALID", result.reason_codes)

    def test_multilingual_trust_and_commercial_signals(self) -> None:
        trust_result = evaluate_content_policy(self.acceptable_input(transcript="室友和房东的合租日常"))
        commercial_result = evaluate_content_policy(self.acceptable_input(title="Mua ngay ở giỏ hàng"))

        self.assertEqual(PolicyDecision.PASS, trust_result.overall_decision)
        self.assertEqual(PolicyDecision.REJECT, commercial_result.overall_decision)
        self.assertIn("HARD_COMMERCE_SIGNAL", commercial_result.reason_codes)

    def test_ai_judge_valid_structured_response_passes(self) -> None:
        result = judge_script(
            title="Test",
            transcript="A complete story with a sharp opening.",
            api_key="test-key",
            http_post=lambda *args, **kwargs: FakeResponse(judge_payload({
                "hook": 9.0, "conflict": 8.5, "pacing": 8.0, "relatability": 8.5,
            })),
        )

        self.assertEqual(PolicyDecision.PASS, result.decision)
        self.assertEqual(8.55, result.metrics["overall_score"])

    def test_ai_judge_malformed_json_requires_review(self) -> None:
        result = judge_script(
            title="Test",
            transcript="Script text.",
            api_key="test-key",
            http_post=lambda *args, **kwargs: FakeResponse({
                "candidates": [{"content": {"parts": [{"text": "not-json"}]}}],
            }),
        )

        self.assertEqual(PolicyDecision.REVIEW, result.decision)
        self.assertIn("JUDGE_MALFORMED_RESPONSE", result.reason_codes)

    def test_ai_judge_timeout_is_unavailable_not_pass(self) -> None:
        def timeout(*args: object, **kwargs: object) -> FakeResponse:
            raise requests.exceptions.Timeout()

        result = judge_script(title="Test", transcript="Script text.", api_key="test-key", http_post=timeout)

        self.assertEqual(PolicyDecision.UNAVAILABLE, result.decision)
        self.assertIn("JUDGE_PROVIDER_TIMEOUT", result.reason_codes)

    def test_ai_judge_provider_error_is_unavailable_not_pass(self) -> None:
        def provider_error(*args: object, **kwargs: object) -> FakeResponse:
            raise requests.exceptions.ConnectionError()

        result = judge_script(title="Test", transcript="Script text.", api_key="test-key", http_post=provider_error)

        self.assertEqual(PolicyDecision.UNAVAILABLE, result.decision)
        self.assertIn("JUDGE_PROVIDER_UNAVAILABLE", result.reason_codes)

    def test_ai_judge_rate_limit_is_unavailable_not_pass(self) -> None:
        result = judge_script(
            title="Test",
            transcript="Script text.",
            api_key="test-key",
            http_post=lambda *args, **kwargs: FakeResponse(status_code=429),
        )

        self.assertEqual(PolicyDecision.UNAVAILABLE, result.decision)
        self.assertIn("JUDGE_RATE_LIMITED", result.reason_codes)

    def test_ai_judge_disabled_or_missing_key_is_unavailable(self) -> None:
        disabled = judge_script(title="Test", transcript="Script text.", enabled=False)

        class EmptyPool:
            def get_key(self) -> None:
                return None

        missing_key = judge_script(title="Test", transcript="Script text.", key_pool=EmptyPool())
        self.assertEqual(PolicyDecision.UNAVAILABLE, disabled.decision)
        self.assertIn("JUDGE_DISABLED", disabled.reason_codes)
        self.assertEqual(PolicyDecision.UNAVAILABLE, missing_key.decision)
        self.assertIn("JUDGE_NO_API_KEY", missing_key.reason_codes)

    def test_combined_precedence_never_upgrades_risk_to_pass(self) -> None:
        passed = PolicyStageResult(stage="prefilter", decision=PolicyDecision.PASS)
        rejected = PolicyStageResult(stage="trust", decision=PolicyDecision.REJECT, reason_codes=("R",))
        errored = PolicyStageResult(stage="judge", decision=PolicyDecision.ERROR, reason_codes=("E",))
        reviewed = PolicyStageResult(stage="judge", decision=PolicyDecision.REVIEW, reason_codes=("V",))

        self.assertEqual(PolicyDecision.REJECT, combine_stage_results((passed, rejected)).overall_decision)
        self.assertEqual(PolicyDecision.ERROR, combine_stage_results((passed, errored)).overall_decision)
        self.assertEqual(PolicyDecision.REVIEW, combine_stage_results((passed, reviewed)).overall_decision)

    def test_enabled_ai_judge_unavailability_remains_visible_in_overall_result(self) -> None:
        result = evaluate_content_policy(
            self.acceptable_input(),
            ContentPolicyConfig(enable_ai_judge=True),
            judge=lambda **kwargs: PolicyStageResult(
                stage="ai_script_judge",
                decision=PolicyDecision.UNAVAILABLE,
                reason_codes=("JUDGE_NO_API_KEY",),
            ),
        )

        self.assertEqual(PolicyDecision.REVIEW, result.overall_decision)
        self.assertTrue(result.needs_review)

    def test_policy_settings_are_explicit_and_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with patch.dict(os.environ, {
                "DUBVI_MEDIA_DIR": str(root / "media"),
                "DUBVI_OUTPUT_DIR": str(root / "output"),
                "DUBVI_MODEL_DIR": str(root / "models"),
                "DUBVI_LOG_DIR": str(root / "logs"),
                "DUBVI_POLICY_MODE": "review",
                "DUBVI_POLICY_ENABLE_AI_JUDGE": "true",
                "DUBVI_POLICY_MIN_DURATION_SECONDS": "40",
                "DUBVI_POLICY_MIN_SPEECH_DENSITY": "0.70",
                "DUBVI_POLICY_JUDGE_THRESHOLD": "8.5",
            }, clear=False):
                settings = Settings.from_env()

        self.assertEqual("review", settings.policy_mode)
        self.assertTrue(settings.policy_enable_ai_judge)
        self.assertEqual(40.0, settings.policy_min_duration_seconds)
        self.assertEqual(0.70, settings.policy_min_speech_density)
        self.assertEqual(8.5, settings.policy_judge_threshold)


if __name__ == "__main__":
    unittest.main()
