"""Source-locked delivery timing for the bounded V2 preview.

The CP3A source windows and approved Vietnamese remain authoritative.  This
module only creates a delivery projection for cached or newly rewritten audio;
it never moves a later utterance to accommodate an earlier one.
"""

from __future__ import annotations

import re
from typing import Mapping, Sequence


class SourceLockedTimingError(RuntimeError):
    """A source-locked preview cannot be made safe without shifting speech."""


MAX_TEMPO = 1.12
DEFAULT_TAIL_MS = 120
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def select_overflow_utterance_ids(tts_qc: Mapping[str, object]) -> list[str]:
    """Derive the current rewrite set from the authoritative TTS QC artifact."""

    rows = tts_qc.get("rows")
    if not isinstance(rows, list):
        raise SourceLockedTimingError("TTS QC rows are missing")
    result: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping) or row.get("reviewReason") != "TTS_TIMING_OVERFLOW":
            continue
        identifier = str(row.get("utteranceId") or "").strip()
        if not identifier or identifier in result:
            raise SourceLockedTimingError("TTS QC overflow identity is invalid")
        result.append(identifier)
    return result


def build_delivery_rewrite_payload(
    voice_events: Sequence[Mapping[str, object]],
    overflow_ids: Sequence[str],
    *,
    attempt: str,
    model: str,
    measured_durations_ms: Mapping[str, int] | None = None,
    scene_memory: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build one bounded contextual batch for overflow rows only."""

    wanted = {str(identifier) for identifier in overflow_ids}
    if not wanted or len(wanted) != len(list(overflow_ids)):
        raise SourceLockedTimingError("delivery rewrite identities are invalid")
    ordered = sorted(
        (row for row in voice_events if str(row.get("utteranceId")) in wanted),
        key=lambda row: (_integer(row, "baseStartMs"), str(row.get("utteranceId"))),
    )
    if {str(row.get("utteranceId")) for row in ordered} != wanted:
        raise SourceLockedTimingError("delivery rewrite set is not present in voice events")
    rows: list[dict[str, object]] = []
    for index, row in enumerate(ordered):
        identifier = str(row["utteranceId"])
        source_start = _integer(row, "baseStartMs")
        source_end = _integer(row, "baseEndMs")
        next_start = _next_start(ordered, index)
        limit = min(source_end + DEFAULT_TAIL_MS, next_start) if next_start is not None else source_end + DEFAULT_TAIL_MS
        measured = None if measured_durations_ms is None else measured_durations_ms.get(identifier)
        if measured is None:
            measured = row.get("finalTtsMs") or row.get("fittedDurationMs")
        _positive_int(measured, "measured TTS duration")
        rows.append({
            "utteranceId": identifier,
            "resolvedZh": row.get("resolvedZh"),
            "approvedNativeVi": row.get("nativeViApproved"),
            "currentDeliveryNativeVi": row.get("deliveryNativeVi"),
            "sourceStartMs": source_start,
            "sourceEndMs": source_end,
            "targetDeliveryDurationMs": max(1, limit - source_start),
            "measuredTtsMs": int(measured),
            "sourcePositions": list(row.get("sourcePositions") or []),
        })
    return {
        "operation": "source_locked_delivery_rewrite_v2",
        "attempt": attempt,
        "model": model,
        "rows": rows,
        "sceneMemory": dict(scene_memory or {}),
        "constraints": {
            "preserveApprovedNativeVi": True,
            "preserveSourceIdentity": True,
            "preserveEventAndRelationship": True,
            "naturalSpokenVietnamese": True,
            "noNewSlangOrJokes": True,
            "noCjkInDeliveryNativeVi": True,
            "oneResultPerUtterance": True,
        },
    }


def validate_delivery_rewrite_response(document: object, expected_ids: Sequence[str]) -> dict[str, str]:
    """Fail closed unless a strict, traceable Vietnamese batch is returned."""

    expected = [str(identifier) for identifier in expected_ids]
    if not isinstance(document, Mapping) or set(document) != {"translations"} or not isinstance(document.get("translations"), list):
        raise SourceLockedTimingError("delivery rewrite response schema is invalid")
    result: dict[str, str] = {}
    for item in document["translations"]:
        if not isinstance(item, Mapping) or set(item) != {"utteranceId", "deliveryNativeVi"}:
            raise SourceLockedTimingError("delivery rewrite row schema is invalid")
        identifier = str(item.get("utteranceId") or "").strip()
        text = item.get("deliveryNativeVi")
        if identifier not in expected or identifier in result or not isinstance(text, str) or not text.strip() or _CJK.search(text):
            raise SourceLockedTimingError("delivery rewrite identity or text is invalid")
        result[identifier] = text.strip()
    if set(result) != set(expected) or len(result) != len(expected):
        raise SourceLockedTimingError("delivery rewrite coverage is incomplete")
    return result


def apply_delivery_rewrites(
    voice_events: Sequence[Mapping[str, object]], rewrites: Mapping[str, str]
) -> list[dict[str, object]]:
    """Return a copy with delivery text changed only for explicitly named rows."""

    known = {str(row.get("utteranceId")) for row in voice_events}
    if not set(rewrites).issubset(known):
        raise SourceLockedTimingError("delivery rewrite references unknown utterance")
    result: list[dict[str, object]] = []
    for raw in voice_events:
        row = dict(raw)
        identifier = str(row.get("utteranceId"))
        if identifier in rewrites:
            text = rewrites[identifier]
            if not isinstance(text, str) or not text.strip() or _CJK.search(text):
                raise SourceLockedTimingError("delivery rewrite text is invalid")
            row["deliveryNativeVi"] = text.strip()
            row["deliveryRewriteApplied"] = True
        result.append(row)
    return result


def build_source_locked_timeline(
    voice_events: Sequence[Mapping[str, object]],
    subtitle_cues: Sequence[Mapping[str, object]],
    *,
    source_duration_ms: int,
    audio_duration_ms: Mapping[str, int] | None = None,
    audio_path_by_id: Mapping[str, str] | None = None,
    tail_ms: int = DEFAULT_TAIL_MS,
    max_tempo: float = MAX_TEMPO,
    max_extension_ms: int = 250,
) -> dict[str, object]:
    """Build V2 events whose starts are always the original source starts."""

    _positive_int(source_duration_ms, "source duration")
    if isinstance(tail_ms, bool) or not isinstance(tail_ms, int) or tail_ms < 0:
        raise SourceLockedTimingError("tail tolerance is invalid")
    if not isinstance(max_tempo, (int, float)) or max_tempo <= 0 or max_tempo > MAX_TEMPO:
        raise SourceLockedTimingError("tempo ceiling is invalid")
    candidates: list[Mapping[str, object]] = []
    omitted: list[str] = []
    seen: set[str] = set()
    for raw in voice_events:
        if not isinstance(raw, Mapping):
            raise SourceLockedTimingError("voice event is invalid")
        identifier = str(raw.get("utteranceId") or "").strip()
        if not identifier or identifier in seen:
            raise SourceLockedTimingError("voice event identity is invalid")
        seen.add(identifier)
        classification = str(raw.get("classification") or raw.get("type") or "").upper()
        if _is_source_uncertain(raw) or classification == "VISUAL_ONLY":
            omitted.append(identifier)
            continue
        if not raw.get("audioPath") and (audio_path_by_id is None or not audio_path_by_id.get(identifier)):
            raise SourceLockedTimingError(f"audio path is missing for {identifier}")
        candidates.append(raw)
    candidates.sort(key=lambda row: (_integer(row, "baseStartMs"), str(row.get("utteranceId"))))
    scheduled: list[dict[str, object]] = []
    for index, raw in enumerate(candidates):
        identifier = str(raw["utteranceId"])
        source_start = _integer(raw, "baseStartMs")
        source_end = _integer(raw, "baseEndMs")
        if source_end <= source_start:
            raise SourceLockedTimingError(f"source window is invalid for {identifier}")
        next_start = _next_start(candidates, index)
        delivery_limit = min(source_end + tail_ms, next_start) if next_start is not None else source_end + tail_ms
        if delivery_limit <= source_start:
            raise SourceLockedTimingError(f"source events overlap for {identifier}")
        duration = None if audio_duration_ms is None else audio_duration_ms.get(identifier)
        if duration is None:
            duration = raw.get("fittedDurationMs") or raw.get("finalTtsMs")
        _positive_int(duration, f"audio duration for {identifier}")
        delivery_end = source_start + int(duration)
        if delivery_end > delivery_limit:
            raise SourceLockedTimingError(f"audio cannot fit source window for {identifier}")
        available = delivery_limit - source_start
        tempo_required = round(int(duration) / available, 4)
        review = tempo_required > float(max_tempo)
        audio_path = (audio_path_by_id or {}).get(identifier) or str(raw.get("audioPath"))
        scheduled.append({
            "timingPolicyVersion": 2,
            "utteranceId": identifier,
            "sourcePositions": list(raw.get("sourcePositions") or []),
            "resolvedZh": raw.get("resolvedZh"),
            "nativeViApproved": raw.get("nativeViApproved"),
            "deliveryNativeVi": raw.get("deliveryNativeVi"),
            "audioPath": audio_path,
            "sourceStartMs": source_start,
            "sourceEndMs": source_end,
            "deliveryStartMs": source_start,
            "deliveryEndMs": delivery_end,
            "scheduledStartMs": source_start,
            "scheduledEndMs": delivery_end,
            "deliveryEndLimitMs": delivery_limit,
            "availableMs": available,
            "measuredMs": int(duration),
            "tempoRequired": tempo_required,
            "startOffsetMs": 0,
            "reviewRequired": review,
            "reviewReason": "TTS_TIMING_OVERFLOW" if review else None,
            "deliveryRewriteApplied": bool(raw.get("deliveryRewriteApplied")),
        })
    event_by_id = {str(row["utteranceId"]): row for row in scheduled}
    retimed = retime_source_locked_subtitles(subtitle_cues, event_by_id)
    duration_ms = max(source_duration_ms, max((int(row["deliveryEndMs"]) for row in scheduled), default=0))
    if duration_ms > source_duration_ms + max_extension_ms:
        raise SourceLockedTimingError("source-locked preview would extend beyond the bounded tail")
    return {
        "schemaVersion": 2,
        "artifact": "source_locked_voice_timeline_v2",
        "sourceDurationMs": source_duration_ms,
        "durationMs": duration_ms,
        "tailToleranceMs": tail_ms,
        "maxTempo": float(max_tempo),
        "voiceEvents": scheduled,
        "subtitleCues": retimed,
        "includedUtteranceIds": [str(row["utteranceId"]) for row in scheduled],
        "omittedUtteranceIds": omitted,
    }


def retime_source_locked_subtitles(
    subtitle_cues: Sequence[Mapping[str, object]], event_by_id: Mapping[str, Mapping[str, object]]
) -> list[dict[str, object]]:
    """Retain cue identity while mapping each cue only inside its local event."""

    grouped: dict[str, list[Mapping[str, object]]] = {}
    for cue in subtitle_cues:
        if not isinstance(cue, Mapping):
            raise SourceLockedTimingError("subtitle cue is invalid")
        identifier = str(cue.get("voiceEventId") or cue.get("utteranceId") or "").strip()
        if identifier in event_by_id:
            grouped.setdefault(identifier, []).append(cue)
    result: list[dict[str, object]] = []
    for identifier, event in event_by_id.items():
        cues = grouped.get(identifier, [])
        delivery = str(event.get("deliveryNativeVi") or "").strip()
        approved = str(event.get("nativeViApproved") or "").strip()
        texts = [str(cue.get("translatedTextVi") or "") for cue in cues]
        if delivery != approved and cues:
            texts = _split_for_count(delivery, len(cues))
        source_start = int(event["sourceStartMs"])
        source_end = int(event["sourceEndMs"])
        source_span = max(1, source_end - source_start)
        delivery_start = int(event["deliveryStartMs"])
        delivery_end = int(event["deliveryEndMs"])
        for cue, text in zip(cues, texts):
            start = cue.get("startMs")
            end = cue.get("endMs")
            if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int) or end <= start:
                raise SourceLockedTimingError(f"subtitle cue range is invalid for {identifier}")
            left = max(source_start, min(start, source_end))
            right = max(left + 1, min(end, source_end))
            scheduled_start = delivery_start + round((left - source_start) * (delivery_end - delivery_start) / source_span)
            scheduled_end = delivery_start + round((right - source_start) * (delivery_end - delivery_start) / source_span)
            scheduled_start = max(delivery_start, min(scheduled_start, delivery_end - 1))
            scheduled_end = max(scheduled_start + 1, min(scheduled_end, delivery_end))
            result.append({
                "voiceEventId": identifier,
                "utteranceId": identifier,
                "position": cue.get("position"),
                "sourcePositions": list(cue.get("sourcePositions") or []),
                "translatedTextVi": text,
                "sourceStartMs": start,
                "sourceEndMs": end,
                "scheduledStartMs": scheduled_start,
                "scheduledEndMs": scheduled_end,
            })
    return result


def _is_source_uncertain(row: Mapping[str, object]) -> bool:
    if str(row.get("reviewReason") or "").upper() == "SOURCE_UNCERTAIN":
        return True
    flags = row.get("reviewFlags")
    return isinstance(flags, (list, tuple)) and any(str(flag).upper() == "SOURCE_UNCERTAIN" for flag in flags)


def _next_start(rows: Sequence[Mapping[str, object]], index: int) -> int | None:
    if index + 1 >= len(rows):
        return None
    return _integer(rows[index + 1], "baseStartMs")


def _integer(row: Mapping[str, object], key: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SourceLockedTimingError(f"{key} must be an integer")
    return value


def _positive_int(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SourceLockedTimingError(f"{label} must be a positive integer")


def _split_for_count(text: str, count: int) -> list[str]:
    if count <= 0:
        return []
    words = text.split()
    if len(words) <= count:
        return words + [""] * (count - len(words))
    result: list[str] = []
    for index in range(count):
        left = (len(words) * index) // count
        right = (len(words) * (index + 1)) // count
        result.append(" ".join(words[left:right]))
    return result
