"""Deterministic delivery planning from semantic, speech, and TTS facts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence


class DeliveryPlanningError(ValueError):
    """The delivery input cannot be planned safely."""


@dataclass(frozen=True)
class VisualThreshold:
    good_speed: float
    rewrite_speed: float


@dataclass(frozen=True)
class DeliveryPolicy:
    max_tempo: float = 1.12
    max_gap_borrow_ms: int = 2_000
    minimum_meaningful_source_window_ms: int = 600
    suspicion_tts_ratio: float = 2.5
    suspicion_speed: float = 0.65
    visual_classes: Mapping[str, VisualThreshold] = field(default_factory=lambda: {
        "TALKING_HEAD": VisualThreshold(0.85, 0.75),
        "BROLL": VisualThreshold(0.65, 0.50),
        "STATIC": VisualThreshold(0.50, 0.35),
        "ACTION": VisualThreshold(0.85, 0.70),
        "UNKNOWN": VisualThreshold(0.80, 0.65),
    })


def plan_utterance(
    utterance: Mapping[str, object], *, policy: DeliveryPolicy | None = None
) -> dict[str, object]:
    """Plan one utterance without changing text or calling a provider."""

    policy = policy or DeliveryPolicy()
    _validate_policy(policy)
    identifier = _text(utterance, "utteranceId")
    semantic_start = _integer(utterance, "semanticStartMs")
    semantic_end = _integer(utterance, "semanticEndMs")
    if semantic_end <= semantic_start:
        raise DeliveryPlanningError("semantic timing window is invalid")
    speech_start = _optional_integer(utterance, "speechStartMs")
    speech_end = _optional_integer(utterance, "speechEndMs")
    confidence = str(utterance.get("speechTimingConfidence") or "UNKNOWN").upper()
    reliable_speech = (
        confidence in {"HIGH", "MEDIUM"}
        and speech_start is not None
        and speech_end is not None
        and speech_end > speech_start
    )
    if reliable_speech:
        start, end, timing_basis = speech_start, speech_end, "speech"
    else:
        start, end, timing_basis = semantic_start, semantic_end, "semantic"
    base_window = end - start
    measured = _integer(utterance, "ttsMeasuredMs")
    if measured <= 0:
        raise DeliveryPlanningError("TTS duration must be positive")
    available_gap = _available_gap(utterance, end)
    source_flags = _flags(utterance.get("sourceFlags"))
    result: dict[str, object] = {
        "utteranceId": identifier,
        "timingBasis": timing_basis,
        "timingConfidence": confidence,
        "sourceStartMs": start,
        "sourceEndMs": end,
        "baseWindowMs": base_window,
        "effectiveWindowMs": base_window,
        "ttsMeasuredMs": measured,
        "availableGapMs": available_gap,
        "tempo": 1.0,
        "borrowedGapMs": 0,
        "visualSpeed": 1.0,
        "needsDeliveryRewrite": False,
        "needsSpeechWindowRefinement": False,
        "reviewRequired": False,
        "approvedNativeVi": utterance.get("approvedNativeVi"),
        "deliveryNativeVi": utterance.get("deliveryNativeVi"),
        "sourcePositions": list(utterance.get("sourcePositions") or []),
        "visualClass": _visual_class(utterance, policy.visual_classes),
        "reasonCodes": [],
    }

    if "SOURCE_UNCERTAIN" in source_flags:
        result["action"] = "REVIEW_REQUIRED"
        result["reviewRequired"] = True
        result["reasonCodes"] = ["SOURCE_UNCERTAIN", "REVIEW_REQUIRED"]
        return result

    if measured <= base_window:
        result["action"] = "KEEP"
        result["reasonCodes"] = ["FITS_BASE_WINDOW"]
        return result

    required_at_max_tempo = measured / policy.max_tempo
    if required_at_max_tempo <= base_window:
        result["action"] = "TEMPO"
        result["tempo"] = round(measured / base_window, 6)
        result["reasonCodes"] = ["FITS_WITH_TEMPO"]
        return result

    needed_gap = max(0, math.ceil(required_at_max_tempo - base_window))
    borrowed = min(needed_gap, available_gap, policy.max_gap_borrow_ms)
    effective_window = base_window + borrowed
    result["effectiveWindowMs"] = effective_window
    result["borrowedGapMs"] = borrowed
    if borrowed:
        result["reasonCodes"].append("SAFE_GAP_AVAILABLE")
    if measured <= effective_window:
        result["action"] = "BORROW_GAP"
        result["tempo"] = round(measured / effective_window, 6)
        return result
    if required_at_max_tempo <= effective_window:
        result["action"] = "BORROW_GAP"
        result["tempo"] = round(measured / effective_window, 6)
        return result

    if (
        timing_basis != "speech"
        and base_window <= policy.minimum_meaningful_source_window_ms
        and measured / base_window >= policy.suspicion_tts_ratio
        and base_window / required_at_max_tempo < policy.suspicion_speed
    ):
        result["action"] = "REFINE_SPEECH_WINDOW"
        result["needsSpeechWindowRefinement"] = True
        result["reviewRequired"] = True
        result["reasonCodes"].extend(["TIMING_EVIDENCE_SUSPICIOUS", "REVIEW_REQUIRED"])
        result["speechWindowRefinementRequest"] = {
            "timingBasis": timing_basis,
            "windowStartMs": start,
            "windowEndMs": end,
            "timingConfidence": confidence,
            "reason": "semantic window is implausibly short for measured speech",
        }
        return result

    visual_speed = base_window / required_at_max_tempo
    thresholds = policy.visual_classes[_visual_class(utterance, policy.visual_classes)]
    result["tempo"] = policy.max_tempo
    result["visualSpeed"] = round(visual_speed, 6)
    result["requiredVisualDurationMs"] = math.ceil(required_at_max_tempo)
    result["reasonCodes"].append("VISUAL_SLOWDOWN_REQUIRED")
    rating = _speed_rating(visual_speed, thresholds.good_speed)
    result["visualSpeedRating"] = rating
    if visual_speed >= thresholds.rewrite_speed:
        result["action"] = "SLOW_VISUAL"
        if rating != "GOOD":
            result["reviewRequired"] = True
            result["reasonCodes"].append("VISUAL_SLOWDOWN_REVIEW")
        return result

    result["action"] = "DELIVERY_REWRITE"
    result["needsDeliveryRewrite"] = True
    result["reviewRequired"] = True
    result["reasonCodes"].extend(["VISUAL_SLOWDOWN_TOO_STRONG", "REVIEW_REQUIRED"])
    result["deliveryRewriteRequest"] = {
        "targetDurationMs": effective_window,
        "maxDurationMs": effective_window,
        "currentMeasuredMs": measured,
        "requiredReductionRatio": round(max(0.0, 1 - (effective_window / measured)), 6),
    }
    return result


def plan_delivery(
    utterances: Sequence[Mapping[str, object]], *, policy: DeliveryPolicy | None = None
) -> dict[str, object]:
    """Plan an ordered batch while keeping each input identity traceable."""

    plans = [plan_utterance(item, policy=policy) for item in utterances]
    identifiers = [str(item["utteranceId"]) for item in plans]
    if len(set(identifiers)) != len(identifiers):
        raise DeliveryPlanningError("utterance identities are duplicated")
    return {"schemaVersion": 1, "artifact": "delivery_plan_v1", "plans": plans}


def _validate_policy(policy: DeliveryPolicy) -> None:
    if policy.max_tempo <= 0 or policy.max_tempo > 1.12:
        raise DeliveryPlanningError("maximum tempo is invalid")
    if policy.max_gap_borrow_ms < 0 or policy.minimum_meaningful_source_window_ms <= 0:
        raise DeliveryPlanningError("delivery policy bounds are invalid")
    if not policy.visual_classes or "UNKNOWN" not in policy.visual_classes:
        raise DeliveryPlanningError("visual policy must define UNKNOWN")
    for threshold in policy.visual_classes.values():
        if not 0 < threshold.rewrite_speed <= threshold.good_speed <= 1:
            raise DeliveryPlanningError("visual slowdown thresholds are invalid")


def _available_gap(utterance: Mapping[str, object], base_end: int) -> int:
    value = utterance.get("nextSpeechStartMs")
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise DeliveryPlanningError("next speech timing must be an integer")
    return max(0, value - base_end)


def _visual_class(utterance: Mapping[str, object], visual_classes: Mapping[str, VisualThreshold]) -> str:
    value = str(utterance.get("visualClass") or "UNKNOWN").upper()
    return value if value in visual_classes else "UNKNOWN"


def _speed_rating(speed: float, good_speed: float) -> str:
    if speed >= 0.85:
        return "GOOD"
    if speed >= 0.70:
        return "NOTICEABLE"
    if speed >= 0.50:
        return "HEAVY"
    return "EXTREME"


def _flags(value: object) -> set[str]:
    if value is None:
        return set()
    if not isinstance(value, (list, tuple, set)):
        raise DeliveryPlanningError("source flags must be a sequence")
    return {str(item).upper() for item in value}


def _text(item: Mapping[str, object], key: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DeliveryPlanningError(f"{key} must be nonblank text")
    return value.strip()


def _integer(item: Mapping[str, object], key: str) -> int:
    value = item.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise DeliveryPlanningError(f"{key} must be an integer")
    return value


def _optional_integer(item: Mapping[str, object], key: str) -> int | None:
    value = item.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise DeliveryPlanningError(f"{key} must be an integer or null")
    return value
