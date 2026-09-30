"""Generic deterministic source-to-output piecewise timeline mapping."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


class TimelineMappingError(ValueError):
    """The source-to-output mapping cannot satisfy its invariants."""


@dataclass(frozen=True)
class TimelineSegment:
    source_start_ms: int
    source_end_ms: int
    output_start_ms: int
    output_end_ms: int
    speed: float
    reason: str
    utterance_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "sourceStartMs": self.source_start_ms,
            "sourceEndMs": self.source_end_ms,
            "outputStartMs": self.output_start_ms,
            "outputEndMs": self.output_end_ms,
            "speed": self.speed,
            "reason": self.reason,
            "utteranceIds": list(self.utterance_ids),
        }


class TimelineMap:
    def __init__(self, source_duration_ms: int, segments: Sequence[TimelineSegment]) -> None:
        self.source_duration_ms = source_duration_ms
        self.segments = tuple(segments)
        self.output_duration_ms = self.segments[-1].output_end_ms if self.segments else 0

    def map_timestamp(self, source_ms: int) -> int:
        if isinstance(source_ms, bool) or not isinstance(source_ms, int):
            raise TimelineMappingError("source timestamp must be an integer")
        if source_ms < 0 or source_ms > self.source_duration_ms:
            raise TimelineMappingError("source timestamp is outside the source")
        if source_ms == self.source_duration_ms:
            return self.output_duration_ms
        for segment in self.segments:
            if segment.source_start_ms <= source_ms < segment.source_end_ms:
                return segment.output_start_ms + round(
                    (source_ms - segment.source_start_ms) / segment.speed
                )
        raise TimelineMappingError("source timestamp has no mapping")

    def map_interval(self, source_start_ms: int, source_end_ms: int) -> tuple[int, int]:
        if source_end_ms <= source_start_ms:
            raise TimelineMappingError("source interval is invalid")
        start = self.map_timestamp(source_start_ms)
        end = self.map_timestamp(source_end_ms)
        if end <= start:
            raise TimelineMappingError("mapped interval is invalid")
        return start, end

    def to_dict(self) -> dict[str, object]:
        return {
            "schemaVersion": 1,
            "artifact": "timeline_map_v1",
            "sourceDurationMs": self.source_duration_ms,
            "outputDurationMs": self.output_duration_ms,
            "segments": [segment.to_dict() for segment in self.segments],
        }


def timeline_map_from_document(document: Mapping[str, object]) -> TimelineMap:
    """Reconstruct the accepted mapper document without inventing a timeline."""
    if not isinstance(document, Mapping) or document.get("schemaVersion") != 1:
        raise TimelineMappingError("accepted timeline document is invalid")
    source_duration = document.get("sourceDurationMs")
    output_duration = document.get("outputDurationMs")
    raw_segments = document.get("segments")
    if (
        isinstance(source_duration, bool) or not isinstance(source_duration, int) or source_duration <= 0
        or isinstance(output_duration, bool) or not isinstance(output_duration, int) or output_duration <= 0
        or not isinstance(raw_segments, list) or not raw_segments
    ):
        raise TimelineMappingError("accepted timeline document is malformed")
    segments: list[TimelineSegment] = []
    for raw in raw_segments:
        if not isinstance(raw, Mapping):
            raise TimelineMappingError("accepted timeline segment is malformed")
        bounds = [raw.get(key) for key in ("sourceStartMs", "sourceEndMs", "outputStartMs", "outputEndMs")]
        if any(isinstance(value, bool) or not isinstance(value, int) for value in bounds):
            raise TimelineMappingError("accepted timeline segment bounds are invalid")
        speed = raw.get("speed")
        utterance_ids = raw.get("utteranceIds", [])
        if not isinstance(speed, (int, float)) or isinstance(speed, bool) or not 0 < speed <= 1:
            raise TimelineMappingError("accepted timeline segment speed is invalid")
        if not isinstance(utterance_ids, list) or any(not isinstance(value, str) for value in utterance_ids):
            raise TimelineMappingError("accepted timeline segment identities are invalid")
        segments.append(TimelineSegment(*bounds, float(speed), str(raw.get("reason") or ""), tuple(utterance_ids)))
    _validate_segments(segments, source_duration)
    mapping = TimelineMap(source_duration, segments)
    if mapping.output_duration_ms != output_duration or mapping.to_dict()["segments"] != raw_segments:
        raise TimelineMappingError("accepted timeline document changed during reconstruction")
    return mapping


def build_timeline(
    requests: Sequence[Mapping[str, object]], *, source_duration_ms: int
) -> TimelineMap:
    """Build full source coverage, merging compatible overlapping slowdown requests."""

    if isinstance(source_duration_ms, bool) or not isinstance(source_duration_ms, int) or source_duration_ms <= 0:
        raise TimelineMappingError("source duration must be positive")
    normalized = _normalize_requests(requests, source_duration_ms)
    merged = _merge_requests(normalized)
    segments: list[TimelineSegment] = []
    source_cursor = 0
    output_cursor = 0
    for request in merged:
        start = int(request["sourceStartMs"])
        end = int(request["sourceEndMs"])
        if start > source_cursor:
            gap = _segment(source_cursor, start, output_cursor, 1.0, "SOURCE_1X")
            segments.append(gap)
            output_cursor = gap.output_end_ms
        segment = _segment(
            start,
            end,
            output_cursor,
            float(request["visualSpeed"]),
            str(request.get("reason") or request.get("action") or "PLANNED"),
            tuple(sorted(str(item) for item in request.get("utteranceIds", []))),
        )
        segments.append(segment)
        source_cursor = end
        output_cursor = segment.output_end_ms
    if source_cursor < source_duration_ms:
        tail = _segment(source_cursor, source_duration_ms, output_cursor, 1.0, "SOURCE_1X")
        segments.append(tail)
    if not segments:
        segments = [_segment(0, source_duration_ms, 0, 1.0, "SOURCE_1X")]
    _validate_segments(segments, source_duration_ms)
    return TimelineMap(source_duration_ms, segments)


def _normalize_requests(
    requests: Sequence[Mapping[str, object]], source_duration_ms: int
) -> list[dict[str, object]]:
    normalized = []
    for request in requests:
        start = request.get("sourceStartMs")
        end = request.get("sourceEndMs")
        speed = request.get("visualSpeed", request.get("speed", 1.0))
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or end <= start
            or start < 0
            or end > source_duration_ms
        ):
            raise TimelineMappingError("source span is invalid")
        if not isinstance(speed, (int, float)) or isinstance(speed, bool) or not 0 < speed <= 1:
            raise TimelineMappingError("visual speed is invalid")
        normalized.append({
            "sourceStartMs": start,
            "sourceEndMs": end,
            "visualSpeed": float(speed),
            "reason": request.get("reason") or request.get("action") or "PLANNED",
            "utteranceIds": list(request.get("utteranceIds") or ([request["utteranceId"]] if request.get("utteranceId") else [])),
            "compatible": request.get("compatible", True),
        })
    normalized.sort(key=lambda item: (int(item["sourceStartMs"]), int(item["sourceEndMs"])))
    return normalized


def _merge_requests(requests: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for request in requests:
        if not result or int(request["sourceStartMs"]) >= int(result[-1]["sourceEndMs"]):
            result.append(dict(request))
            continue
        previous = result[-1]
        if request.get("compatible") is False or previous.get("compatible") is False:
            raise TimelineMappingError("overlapping retime requests are incompatible")
        previous["sourceEndMs"] = max(int(previous["sourceEndMs"]), int(request["sourceEndMs"]))
        previous["visualSpeed"] = min(float(previous["visualSpeed"]), float(request["visualSpeed"]))
        previous["reason"] = "MERGED_OVERLAPPING_RETIME"
        previous["utteranceIds"] = list(previous.get("utteranceIds", [])) + list(request.get("utteranceIds", []))
    return result


def _segment(
    source_start: int,
    source_end: int,
    output_start: int,
    speed: float,
    reason: str,
    utterance_ids: tuple[str, ...] = (),
) -> TimelineSegment:
    duration = max(1, math.ceil((source_end - source_start) / speed))
    return TimelineSegment(
        source_start,
        source_end,
        output_start,
        output_start + duration,
        speed,
        reason,
        utterance_ids,
    )


def _validate_segments(segments: Sequence[TimelineSegment], source_duration_ms: int) -> None:
    if segments[0].source_start_ms != 0 or segments[-1].source_end_ms != source_duration_ms:
        raise TimelineMappingError("source coverage is incomplete")
    for previous, current in zip(segments, segments[1:]):
        if previous.source_end_ms != current.source_start_ms:
            raise TimelineMappingError("source spans contain a gap or overlap")
        if previous.output_end_ms != current.output_start_ms:
            raise TimelineMappingError("output spans contain a gap or overlap")
    for segment in segments:
        if segment.source_start_ms >= segment.source_end_ms or segment.output_start_ms >= segment.output_end_ms:
            raise TimelineMappingError("timeline segment is empty")
