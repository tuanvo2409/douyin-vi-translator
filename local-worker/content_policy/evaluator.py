"""Pure local prefilter and trust-classification policy stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from .keywords import (
    HARD_COMMERCE_TERMS,
    PHOTO_SLIDESHOW_TERMS,
    SOFT_COMMERCE_TERMS,
    TRUST_PATTERNS,
    matched_terms,
    normalize_text,
)
from .models import PolicyDecision, PolicyResult, PolicyStageResult, combine_stage_results


@dataclass(frozen=True)
class PrefilterConfig:
    min_duration_seconds: float = 35.0
    max_duration_seconds: float = 240.0
    min_speech_density: float = 0.60


@dataclass(frozen=True)
class ContentPolicyInput:
    duration_seconds: float | None
    segments: Sequence[Mapping[str, Any]] | None = None
    transcript: str = ""
    description: str = ""
    title: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    author: str = ""
    channel_profile: str = ""


@dataclass(frozen=True)
class ContentPolicyConfig:
    prefilter: PrefilterConfig = field(default_factory=PrefilterConfig)
    enable_ai_judge: bool = False
    judge_threshold: float = 8.0
    judge_timeout_seconds: float = 20.0
    judge_api_key: str | None = None


def calculate_speech_metrics(
    segments: Sequence[Mapping[str, Any]],
    total_duration_seconds: float,
) -> tuple[float, float]:
    """Return de-overlapped speech seconds and density from ASR-like segments."""
    if total_duration_seconds <= 0:
        raise ValueError("total_duration_seconds must be positive")

    intervals: list[tuple[float, float]] = []
    for segment in segments:
        if not isinstance(segment, Mapping):
            raise ValueError("segment must be a mapping")
        if "startMs" in segment or "endMs" in segment:
            if "startMs" not in segment or "endMs" not in segment:
                raise ValueError("segment must contain both startMs and endMs")
            start = float(segment["startMs"]) / 1000.0
            end = float(segment["endMs"]) / 1000.0
        else:
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", 0.0))
        start = max(0.0, start)
        end = min(total_duration_seconds, end)
        if end > start:
            intervals.append((start, end))

    if not intervals:
        return 0.0, 0.0
    intervals.sort()
    merged: list[tuple[float, float]] = [intervals[0]]
    for start, end in intervals[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    speech_seconds = sum(end - start for start, end in merged)
    return speech_seconds, speech_seconds / total_duration_seconds


def evaluate_prefilter(
    content: ContentPolicyInput,
    config: PrefilterConfig | None = None,
) -> PolicyStageResult:
    """Evaluate local duration, media-format, and ASR-density evidence only."""
    config = config or PrefilterConfig()
    metadata = dict(content.metadata or {})
    duration = content.duration_seconds
    if duration is None:
        return PolicyStageResult(
            stage="prefilter",
            decision=PolicyDecision.REVIEW,
            reason_codes=("DURATION_EVIDENCE_MISSING",),
            human_summary="Video duration is unavailable for prefilter evaluation.",
        )
    if not isinstance(duration, (int, float)) or isinstance(duration, bool) or duration <= 0:
        return PolicyStageResult(
            stage="prefilter",
            decision=PolicyDecision.ERROR,
            reason_codes=("INVALID_DURATION",),
            human_summary="Video duration is invalid.",
            details={"duration_seconds": duration},
        )

    photo_flags = ("is_photo", "is_slideshow", "is_image_post")
    if any(bool(metadata.get(flag)) for flag in photo_flags):
        return PolicyStageResult(
            stage="prefilter",
            decision=PolicyDecision.REJECT,
            reason_codes=("PHOTO_MEDIA_METADATA",),
            human_summary="Metadata identifies this item as photo or slideshow media.",
            confidence=0.99,
            metrics={"duration_seconds": duration},
        )
    if duration < config.min_duration_seconds:
        return PolicyStageResult(
            stage="prefilter",
            decision=PolicyDecision.REJECT,
            reason_codes=("DURATION_TOO_SHORT",),
            human_summary="Video is shorter than the configured minimum duration.",
            confidence=0.99,
            metrics={"duration_seconds": duration, "min_duration_seconds": config.min_duration_seconds},
        )
    if duration > config.max_duration_seconds:
        return PolicyStageResult(
            stage="prefilter",
            decision=PolicyDecision.REVIEW,
            reason_codes=("DURATION_ABOVE_SHORT_FORM_TARGET",),
            human_summary="Video exceeds the configured short-form target duration.",
            confidence=0.75,
            metrics={"duration_seconds": duration, "max_duration_seconds": config.max_duration_seconds},
        )

    text = normalize_text((content.title, content.description, content.transcript))
    photo_matches = matched_terms(text, PHOTO_SLIDESHOW_TERMS)
    if photo_matches:
        return PolicyStageResult(
            stage="prefilter",
            decision=PolicyDecision.REVIEW,
            reason_codes=("PHOTO_FORMAT_TEXT_SIGNAL",),
            human_summary="Text suggests photo or slideshow content and needs confirmation.",
            confidence=0.65,
            metrics={"duration_seconds": duration},
            details={"matched_terms": list(photo_matches)},
        )

    if content.segments is None:
        return PolicyStageResult(
            stage="prefilter",
            decision=PolicyDecision.REVIEW,
            reason_codes=("SPEECH_EVIDENCE_MISSING",),
            human_summary="Speech-density evidence is not available.",
            metrics={"duration_seconds": duration},
        )
    try:
        speech_seconds, speech_density = calculate_speech_metrics(content.segments, float(duration))
    except (TypeError, ValueError) as exc:
        return PolicyStageResult(
            stage="prefilter",
            decision=PolicyDecision.ERROR,
            reason_codes=("SPEECH_METRICS_INVALID",),
            human_summary="Speech-density evidence could not be evaluated.",
            details={"error_type": type(exc).__name__},
        )

    metrics = {
        "duration_seconds": round(float(duration), 3),
        "speech_seconds": round(speech_seconds, 3),
        "speech_density": round(speech_density, 4),
        "segment_count": len(content.segments),
        "min_speech_density": config.min_speech_density,
    }
    if not content.segments or speech_seconds <= 0:
        return PolicyStageResult(
            stage="prefilter",
            decision=PolicyDecision.REVIEW,
            reason_codes=("NO_SPEECH_DETECTED",),
            human_summary="No usable speech was detected; verify ASR or media content.",
            confidence=0.8,
            metrics=metrics,
        )
    if speech_density < config.min_speech_density:
        return PolicyStageResult(
            stage="prefilter",
            decision=PolicyDecision.REVIEW,
            reason_codes=("LOW_SPEECH_DENSITY",),
            human_summary="Speech density is below the configured threshold.",
            confidence=0.8,
            metrics=metrics,
        )
    return PolicyStageResult(
        stage="prefilter",
        decision=PolicyDecision.PASS,
        reason_codes=("LOCAL_MEDIA_CHECKS_PASSED",),
        human_summary="Duration and speech-density checks passed.",
        confidence=0.9,
        metrics=metrics,
    )


def classify_trust(
    *,
    transcript: str = "",
    description: str = "",
    title: str = "",
    metadata: Mapping[str, Any] | None = None,
    prefilter_result: PolicyStageResult | None = None,
) -> PolicyStageResult:
    """Classify commercial risk and organic/trust signals from explicit inputs."""
    metadata = dict(metadata or {})
    if any(bool(metadata.get(flag)) for flag in ("is_ads", "commercial_video", "commerce_info", "anchor_info")):
        return PolicyStageResult(
            stage="trust",
            decision=PolicyDecision.REJECT,
            reason_codes=("COMMERCE_METADATA",),
            human_summary="Metadata identifies commercial or product-anchor content.",
            confidence=0.99,
        )
    if prefilter_result and "PHOTO_MEDIA_METADATA" in prefilter_result.reason_codes:
        return PolicyStageResult(
            stage="trust",
            decision=PolicyDecision.REJECT,
            reason_codes=("PHOTO_MEDIA_METADATA",),
            human_summary="Prefilter identified photo or slideshow media.",
            confidence=0.99,
        )

    corpus = normalize_text((title, description, transcript))
    if not corpus:
        return PolicyStageResult(
            stage="trust",
            decision=PolicyDecision.REVIEW,
            reason_codes=("TRUST_EVIDENCE_MISSING",),
            human_summary="No title, description, or transcript is available for trust classification.",
        )

    hard_matches = matched_terms(corpus, HARD_COMMERCE_TERMS)
    soft_matches = matched_terms(corpus, SOFT_COMMERCE_TERMS)
    trust_matches = matched_terms(corpus, TRUST_PATTERNS)
    if hard_matches:
        return PolicyStageResult(
            stage="trust",
            decision=PolicyDecision.REJECT,
            reason_codes=("HARD_COMMERCE_SIGNAL",),
            human_summary="Strong commercial call-to-action signals were found.",
            confidence=0.95,
            details={"hard_commerce_matches": list(hard_matches)},
        )
    if len(soft_matches) >= 2:
        return PolicyStageResult(
            stage="trust",
            decision=PolicyDecision.REJECT,
            reason_codes=("MULTIPLE_COMMERCE_SIGNALS",),
            human_summary="Multiple commercial or product-promotion signals were found.",
            confidence=0.88,
            details={"soft_commerce_matches": list(soft_matches)},
        )
    if soft_matches:
        return PolicyStageResult(
            stage="trust",
            decision=PolicyDecision.REVIEW,
            reason_codes=("SINGLE_COMMERCE_SIGNAL",),
            human_summary="One commercial or product-promotion signal needs human review.",
            confidence=0.65,
            details={"soft_commerce_matches": list(soft_matches)},
        )
    if trust_matches:
        return PolicyStageResult(
            stage="trust",
            decision=PolicyDecision.PASS,
            reason_codes=("ORGANIC_TRUST_SIGNAL",),
            human_summary="Organic lifestyle or relationship-content signals were found.",
            confidence=0.85,
            details={"trust_matches": list(trust_matches)},
        )
    return PolicyStageResult(
        stage="trust",
        decision=PolicyDecision.PASS,
        reason_codes=("NO_COMMERCE_SIGNALS",),
        human_summary="No commercial signals were found in the supplied text.",
        confidence=0.65,
    )


def evaluate_content_policy(
    content: ContentPolicyInput,
    config: ContentPolicyConfig | None = None,
    judge: Callable[..., PolicyStageResult] | None = None,
    judge_http_post: Callable[..., Any] | None = None,
) -> PolicyResult:
    """Evaluate the configured local policy stages and optional AI script judge."""
    config = config or ContentPolicyConfig()
    prefilter = evaluate_prefilter(content, config.prefilter)
    trust = classify_trust(
        transcript=content.transcript,
        description=content.description,
        title=content.title,
        metadata=content.metadata,
        prefilter_result=prefilter,
    )
    stages = [prefilter, trust]
    if config.enable_ai_judge:
        if judge is None:
            from .script_judge import judge_script
            judge = judge_script
        stages.append(judge(
            title=content.title,
            transcript=content.transcript,
            author=content.author,
            channel_profile=content.channel_profile,
            api_key=config.judge_api_key,
            threshold=config.judge_threshold,
            timeout_seconds=config.judge_timeout_seconds,
            http_post=judge_http_post,
        ))
    return combine_stage_results(stages)
