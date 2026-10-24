"""Generic bounded source-to-output adaptive render planning.

This module contains no provider calls and does not alter localization text. It
projects measured, already-synthesized utterances onto a validated piecewise
video timeline, then builds a local FFmpeg command with dubbed audio only.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

from timeline_mapper import TimelineMap, TimelineMappingError, build_timeline


class AdaptiveRenderError(RuntimeError):
    """The measured delivery plan cannot be rendered without losing timing authority."""


def build_adaptive_timeline(
    voice_events: Sequence[Mapping[str, object]],
    subtitle_cues: Sequence[Mapping[str, object]],
    delivery_plans: Sequence[Mapping[str, object]],
    *,
    source_duration_ms: int,
    audio_duration_ms: Mapping[str, int] | None = None,
) -> dict[str, object]:
    """Map valid local voice events onto a full, gap-free output timeline."""

    if not _positive_int(source_duration_ms):
        raise AdaptiveRenderError("source duration must be positive")
    plans = _plans_by_id(delivery_plans)
    events = _ordered_events(voice_events)
    if set(plans) != {str(event["utteranceId"]) for event in events}:
        raise AdaptiveRenderError("delivery plan and voice event identities differ")

    requests: list[dict[str, object]] = []
    for event in events:
        identifier = str(event["utteranceId"])
        plan = plans[identifier]
        start, end = _event_range(event)
        if (start, end) != (_integer(plan, "sourceStartMs"), _integer(plan, "sourceEndMs")):
            raise AdaptiveRenderError(f"delivery plan source timing differs for {identifier}")
        speed = _speed(plan)
        if speed < 1.0:
            requests.append({
                "sourceStartMs": start,
                "sourceEndMs": end,
                "visualSpeed": speed,
                "reason": str(plan.get("action") or "SLOW_VISUAL"),
                "utteranceId": identifier,
            })
    try:
        mapping = build_timeline(requests, source_duration_ms=source_duration_ms)
    except TimelineMappingError as exc:
        raise AdaptiveRenderError("adaptive source timeline is invalid") from exc

    scheduled: list[dict[str, object]] = []
    for index, event in enumerate(events):
        identifier = str(event["utteranceId"])
        plan = plans[identifier]
        start, end = _event_range(event)
        next_start = _event_range(events[index + 1])[0] if index + 1 < len(events) else source_duration_ms
        borrowed = _nonnegative_integer(plan, "borrowedGapMs")
        delivery_limit_source = min(end + borrowed, next_start, source_duration_ms)
        if delivery_limit_source <= start:
            raise AdaptiveRenderError(f"delivery window is invalid for {identifier}")
        duration = _audio_duration(event, identifier, audio_duration_ms)
        scheduled_start = mapping.map_timestamp(start)
        scheduled_end = scheduled_start + duration
        delivery_limit_output = mapping.map_timestamp(delivery_limit_source)
        if scheduled_end > delivery_limit_output:
            raise AdaptiveRenderError(f"audio exceeds mapped delivery window for {identifier}")
        scheduled.append({
            "utteranceId": identifier,
            "sourcePositions": list(event.get("sourcePositions") or []),
            "nativeViApproved": event.get("nativeViApproved"),
            "deliveryNativeVi": event.get("deliveryNativeVi"),
            "audioPath": _text(event, "audioPath"),
            "sourceStartMs": start,
            "sourceEndMs": end,
            "deliveryLimitSourceMs": delivery_limit_source,
            "deliveryLimitOutputMs": delivery_limit_output,
            "scheduledStartMs": scheduled_start,
            "scheduledEndMs": scheduled_end,
            "durationMs": duration,
            "deliveryAction": plan.get("action"),
            "deliveryTempo": plan.get("tempo", 1.0),
            "visualSpeed": _speed(plan),
            "reviewRequired": bool(plan.get("reviewRequired")) or bool(event.get("reviewRequired")),
            "reviewReason": event.get("reviewReason"),
        })
    return {
        "schemaVersion": 1,
        "artifact": "adaptive_retime_timeline_v1",
        **mapping.to_dict(),
        "voiceEvents": scheduled,
        "subtitleCues": _retime_subtitles(subtitle_cues, scheduled),
        "includedUtteranceIds": [str(event["utteranceId"]) for event in scheduled],
    }


def build_adaptive_ffmpeg_command(
    source: Path,
    output: Path,
    ass_path: Path,
    voice_inputs: Sequence[tuple[Path, int]],
    *,
    timeline: Mapping[str, object],
    width: int,
    height: int,
    ffmpeg_binary: str = "ffmpeg",
    mask_overlay: Path | None = None,
) -> list[str]:
    """Build a portable H.264/AAC command with all source intervals rendered once."""

    if not voice_inputs:
        raise AdaptiveRenderError("adaptive render needs at least one voice input")
    if not _positive_int(width) or not _positive_int(height):
        raise AdaptiveRenderError("output dimensions must be positive")
    duration_ms = _integer(timeline, "outputDurationMs")
    if duration_ms <= 0:
        raise AdaptiveRenderError("output timeline duration is invalid")
    segments = timeline.get("segments")
    if not isinstance(segments, list) or not segments:
        raise AdaptiveRenderError("adaptive timeline segments are missing")
    _validate_render_segments(segments)
    if mask_overlay is not None and (not mask_overlay.is_file() or mask_overlay.stat().st_size == 0):
        raise AdaptiveRenderError("Page 1 mask overlay is not a regular file")

    filters: list[str] = []
    visual_labels: list[str] = []
    source_labels = [f"[source{index}]" for index in range(len(segments))]
    filters.append(f"[0:v]split={len(segments)}" + "".join(source_labels))
    for index, raw in enumerate(segments):
        start = _integer(raw, "sourceStartMs") / 1000.0
        end = _integer(raw, "sourceEndMs") / 1000.0
        speed = raw.get("speed")
        if not isinstance(speed, (float, int)) or isinstance(speed, bool) or not 0 < float(speed) <= 1:
            raise AdaptiveRenderError("adaptive render speed is invalid")
        label = f"v{index}"
        visual_labels.append(f"[{label}]")
        filters.append(
            f"{source_labels[index]}trim=start={start:.3f}:end={end:.3f},setpts=(PTS-STARTPTS)/{float(speed):.6f},"
            f"fps=30,scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p[{label}]"
        )
    filters.append("".join(visual_labels) + f"concat=n={len(visual_labels)}:v=1:a=0[vbase]")
    if mask_overlay is not None:
        filters.append("[vbase][1:v]overlay=x=0:y=0:eof_action=repeat:shortest=0[vmasked]")
        filters.append(f"[vmasked]subtitles='{_filter_path(ass_path)}'[vout]")
    else:
        filters.append(f"[vbase]subtitles='{_filter_path(ass_path)}'[vout]")

    audio_labels: list[str] = []
    command = [ffmpeg_binary, "-hide_banner", "-y", "-i", str(source)]
    if mask_overlay is not None:
        command.extend(["-loop", "1", "-i", str(mask_overlay)])
    audio_start = 2 if mask_overlay is not None else 1
    for index, (audio_path, start_ms) in enumerate(voice_inputs, start=audio_start):
        if not isinstance(start_ms, int) or isinstance(start_ms, bool) or start_ms < 0:
            raise AdaptiveRenderError("voice start must be a non-negative integer")
        command.extend(["-i", str(audio_path)])
        label = f"a{index}"
        audio_labels.append(f"[{label}]")
        filters.append(
            f"[{index}:a]aresample=48000,aformat=sample_fmts=fltp:sample_rates=48000:"
            f"channel_layouts=stereo,adelay={start_ms}|{start_ms}[{label}]"
        )
    filters.append(
        "".join(audio_labels)
        + f"amix=inputs={len(audio_labels)}:duration=longest:dropout_transition=0:normalize=0,"
        f"atrim=duration={duration_ms / 1000.0:.3f},asetpts=PTS-STARTPTS[aout]"
    )
    command.extend([
        "-filter_complex", ";".join(filters), "-map", "[vout]", "-map", "[aout]",
        "-t", f"{duration_ms / 1000.0:.3f}",
        "-c:v", "libx264", "-profile:v", "main", "-level:v", "4.1", "-pix_fmt", "yuv420p",
        "-preset", "veryfast", "-crf", "20", "-threads", "2",
        "-c:a", "aac", "-profile:a", "aac_low", "-b:a", "160k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart", str(output),
    ])
    return command


def render_adaptive(command: Sequence[str]) -> None:
    """Execute one local FFmpeg render and retain its detailed stderr at the caller boundary."""

    try:
        # The canonical child returns a small JSON receipt over stdout; FFmpeg's
        # progress must be drained here or the parent's pipe can block the render.
        args = list(command)
        if sum(len(part) + 1 for part in args) > 30000 and "-filter_complex" in args:
            with tempfile.TemporaryDirectory(prefix="dubvi-filter-") as directory:
                index = args.index("-filter_complex")
                script = Path(directory) / "graph.txt"
                script.write_text(args[index + 1], encoding="utf-8")
                args[index:index + 2] = ["-filter_complex_script", str(script)]
                subprocess.run(args, check=True, capture_output=True)
        else:
            subprocess.run(args, check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AdaptiveRenderError("adaptive FFmpeg render failed") from exc


def _plans_by_id(plans: Sequence[Mapping[str, object]]) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for plan in plans:
        if not isinstance(plan, Mapping):
            raise AdaptiveRenderError("delivery plan row is invalid")
        identifier = _text(plan, "utteranceId")
        if identifier in result:
            raise AdaptiveRenderError("delivery plan identity is duplicated")
        result[identifier] = plan
    return result


def _ordered_events(events: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    result: list[Mapping[str, object]] = []
    seen: set[str] = set()
    for event in events:
        if not isinstance(event, Mapping):
            raise AdaptiveRenderError("voice event is invalid")
        identifier = _text(event, "utteranceId")
        if identifier in seen:
            raise AdaptiveRenderError("voice event identity is duplicated")
        seen.add(identifier)
        _event_range(event)
        result.append(event)
    result.sort(key=lambda event: (*_event_range(event), str(event["utteranceId"])))
    for left, right in zip(result, result[1:]):
        if _event_range(left)[1] > _event_range(right)[0]:
            raise AdaptiveRenderError("voice source windows overlap")
    return result


def _event_range(event: Mapping[str, object]) -> tuple[int, int]:
    start = _integer(event, "baseStartMs")
    end = _integer(event, "baseEndMs")
    if end <= start:
        raise AdaptiveRenderError("voice source window is invalid")
    return start, end


def _audio_duration(event: Mapping[str, object], identifier: str, values: Mapping[str, int] | None) -> int:
    value = values.get(identifier) if values is not None else None
    if value is None:
        value = event.get("fittedDurationMs", event.get("finalTtsMs"))
    if not _positive_int(value):
        raise AdaptiveRenderError(f"audio duration is invalid for {identifier}")
    return int(value)


def _retime_subtitles(cues: Sequence[Mapping[str, object]], events: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    events_by_id = {str(event["utteranceId"]): event for event in events}
    result: list[dict[str, object]] = []
    for cue in cues:
        if not isinstance(cue, Mapping):
            raise AdaptiveRenderError("subtitle cue is invalid")
        identifier = str(cue.get("voiceEventId") or cue.get("utteranceId") or "").strip()
        event = events_by_id.get(identifier)
        if event is None:
            continue
        start = _integer(cue, "startMs")
        end = _integer(cue, "endMs")
        source_start = int(event["sourceStartMs"])
        source_end = int(event["sourceEndMs"])
        if end <= start:
            raise AdaptiveRenderError(f"subtitle timing is invalid for {identifier}")
        left = max(source_start, min(start, source_end))
        right = max(left + 1, min(end, source_end))
        scheduled_start = int(event["scheduledStartMs"]) + round(
            (left - source_start) * int(event["durationMs"]) / max(1, source_end - source_start)
        )
        scheduled_end = int(event["scheduledStartMs"]) + round(
            (right - source_start) * int(event["durationMs"]) / max(1, source_end - source_start)
        )
        scheduled_start = max(int(event["scheduledStartMs"]), min(scheduled_start, int(event["scheduledEndMs"]) - 1))
        scheduled_end = max(scheduled_start + 1, min(scheduled_end, int(event["scheduledEndMs"])))
        result.append({
            "voiceEventId": identifier,
            "utteranceId": identifier,
            "position": cue.get("position"),
            "sourcePositions": list(cue.get("sourcePositions") or []),
            "translatedTextVi": cue.get("translatedTextVi"),
            "sourceStartMs": start,
            "sourceEndMs": end,
            "scheduledStartMs": scheduled_start,
            "scheduledEndMs": scheduled_end,
        })
    return result


def _validate_render_segments(segments: Sequence[object]) -> None:
    previous_source_end = 0
    previous_output_end = 0
    for raw in segments:
        if not isinstance(raw, Mapping):
            raise AdaptiveRenderError("adaptive render segment is invalid")
        source_start = _integer(raw, "sourceStartMs")
        source_end = _integer(raw, "sourceEndMs")
        output_start = _integer(raw, "outputStartMs")
        output_end = _integer(raw, "outputEndMs")
        if source_start != previous_source_end or output_start != previous_output_end or source_end <= source_start or output_end <= output_start:
            raise AdaptiveRenderError("adaptive render segments are not contiguous")
        previous_source_end = source_end
        previous_output_end = output_end


def _speed(plan: Mapping[str, object]) -> float:
    value = plan.get("visualSpeed", 1.0)
    if not isinstance(value, (float, int)) or isinstance(value, bool) or not 0 < float(value) <= 1:
        raise AdaptiveRenderError("delivery visual speed is invalid")
    return float(value)


def _text(row: Mapping[str, object], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise AdaptiveRenderError(f"{field} must be nonblank text")
    return value.strip()


def _integer(row: Mapping[str, object], field: str) -> int:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise AdaptiveRenderError(f"{field} must be an integer")
    return value


def _nonnegative_integer(row: Mapping[str, object], field: str) -> int:
    value = _integer(row, field)
    if value < 0:
        raise AdaptiveRenderError(f"{field} must be non-negative")
    return value


def _positive_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _filter_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
