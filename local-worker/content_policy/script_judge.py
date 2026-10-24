"""Optional Gemini-backed script-quality judge using the existing key pool."""
from __future__ import annotations

import json
from typing import Any, Callable, Iterable, Mapping

import requests

from .models import PolicyDecision, PolicyStageResult


JUDGE_PROMPT = """You evaluate a short-video script for production readiness.
Return JSON only with numeric values from 0 to 10:
{
  "hook_score": 0,
  "conflict_score": 0,
  "pacing_score": 0,
  "relatability_score": 0,
  "critique": "brief explanation"
}
Judge the opening hook, conflict or narrative tension, pacing, and relatability
to the intended Vietnamese short-video audience. Do not include markdown."""


def _response_text(payload: Mapping[str, Any]) -> str:
    return str(payload["candidates"][0]["content"]["parts"][0]["text"]).strip()


def _parse_judge_payload(raw_text: str, threshold: float) -> PolicyStageResult:
    cleaned = raw_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    try:
        payload = json.loads(cleaned.strip())
        scores = {
            "hook": float(payload["hook_score"]),
            "conflict": float(payload["conflict_score"]),
            "pacing": float(payload["pacing_score"]),
            "relatability": float(payload["relatability_score"]),
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return PolicyStageResult(
            stage="ai_script_judge",
            decision=PolicyDecision.REVIEW,
            reason_codes=("JUDGE_MALFORMED_RESPONSE",),
            human_summary="AI judge returned an invalid score payload.",
        )
    if any(score < 0 or score > 10 for score in scores.values()):
        return PolicyStageResult(
            stage="ai_script_judge",
            decision=PolicyDecision.REVIEW,
            reason_codes=("JUDGE_SCORE_OUT_OF_RANGE",),
            human_summary="AI judge returned scores outside the allowed range.",
            metrics={"scores": scores},
        )
    overall_score = round(
        scores["hook"] * 0.30
        + scores["conflict"] * 0.30
        + scores["pacing"] * 0.20
        + scores["relatability"] * 0.20,
        2,
    )
    passed = overall_score >= threshold
    return PolicyStageResult(
        stage="ai_script_judge",
        decision=PolicyDecision.PASS if passed else PolicyDecision.REJECT,
        reason_codes=("JUDGE_SCORE_AT_OR_ABOVE_THRESHOLD",) if passed else ("JUDGE_SCORE_BELOW_THRESHOLD",),
        human_summary=str(payload.get("critique") or "AI script judge completed."),
        confidence=0.8,
        metrics={"scores": scores, "overall_score": overall_score, "threshold": threshold},
    )


def _resolve_key(api_key: str | None, key_pool: Any | None) -> tuple[str | None, Any | None]:
    if api_key:
        return api_key, key_pool
    if key_pool is None:
        try:
            from gemini_pool import gemini_pool as configured_pool
            key_pool = configured_pool
        except ImportError:
            return None, None
    return key_pool.get_key(), key_pool


def judge_script(
    *,
    title: str,
    transcript: str,
    author: str = "",
    channel_profile: str = "",
    api_key: str | None = None,
    key_pool: Any | None = None,
    threshold: float = 8.0,
    timeout_seconds: float = 20.0,
    enabled: bool = True,
    model_names: Iterable[str] = ("gemini-2.5-flash", "gemini-flash-lite-latest"),
    http_post: Callable[..., Any] | None = None,
) -> PolicyStageResult:
    """Return a structured score or a visible unavailable/review result.

    No provider request is attempted when disabled, no transcript exists, or no
    configured key is available. The existing Gemini pool is used lazily only
    when an explicit key was not supplied.
    """
    if not enabled:
        return PolicyStageResult(
            stage="ai_script_judge",
            decision=PolicyDecision.UNAVAILABLE,
            reason_codes=("JUDGE_DISABLED",),
            human_summary="AI script judging is disabled.",
        )
    if not transcript.strip():
        return PolicyStageResult(
            stage="ai_script_judge",
            decision=PolicyDecision.REVIEW,
            reason_codes=("JUDGE_TRANSCRIPT_MISSING",),
            human_summary="AI script judge needs a transcript.",
        )

    key, key_pool = _resolve_key(api_key, key_pool)
    if not key:
        return PolicyStageResult(
            stage="ai_script_judge",
            decision=PolicyDecision.UNAVAILABLE,
            reason_codes=("JUDGE_NO_API_KEY",),
            human_summary="AI script judge has no configured provider key.",
        )

    request = http_post or requests.post
    prompt = (
        f"Channel profile: {channel_profile or 'unspecified'}\n"
        f"Author: {author or 'unspecified'}\n"
        f"Title: {title or 'untitled'}\n"
        f"Transcript:\n{transcript}\n"
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": f"{JUDGE_PROMPT}\n\n{prompt}"}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2},
    }
    last_error_code = "JUDGE_PROVIDER_UNAVAILABLE"
    for model_name in dict.fromkeys(model_names):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"
        try:
            response = request(url, json=payload, timeout=timeout_seconds)
            response.raise_for_status()
            return _parse_judge_payload(_response_text(response.json()), threshold)
        except requests.exceptions.HTTPError as exc:
            status_code = getattr(exc.response, "status_code", None)
            if status_code == 429:
                last_error_code = "JUDGE_RATE_LIMITED"
            else:
                last_error_code = "JUDGE_PROVIDER_HTTP_ERROR"
            if key_pool is not None and status_code is not None:
                key_pool.report_error(key, status_code)
        except requests.exceptions.Timeout:
            last_error_code = "JUDGE_PROVIDER_TIMEOUT"
        except (KeyError, TypeError, IndexError):
            return PolicyStageResult(
                stage="ai_script_judge",
                decision=PolicyDecision.REVIEW,
                reason_codes=("JUDGE_MALFORMED_RESPONSE",),
                human_summary="AI judge response did not contain the expected payload.",
            )
        except requests.RequestException:
            last_error_code = "JUDGE_PROVIDER_UNAVAILABLE"
        except Exception:
            last_error_code = "JUDGE_PROVIDER_ERROR"
    return PolicyStageResult(
        stage="ai_script_judge",
        decision=PolicyDecision.UNAVAILABLE,
        reason_codes=(last_error_code,),
        human_summary="AI script judge could not be reached; no score was accepted.",
    )
