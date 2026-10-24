"""Bounded preview scheduling and compatible local video rendering.

This module is intentionally separate from the legacy worker.  It consumes
already-approved CP3A voice/subtitle artifacts and never synthesizes audio or
contacts a provider.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

from subtitle_layout import SubtitleLayoutPolicy, write_ass_subtitles as _write_layout_ass_subtitles


class PreviewRenderError(RuntimeError):
    """The cached preview inputs cannot be rendered safely."""


DEFAULT_GAP_MS = 60
MAX_PREVIEW_DURATION_MS = 30_000


def build_preview_timeline(
    voice_events: Sequence[Mapping[str, object]],
    subtitle_cues: Sequence[Mapping[str, object]],
    *,
    source_duration_ms: int,
    max_duration_ms: int = MAX_PREVIEW_DURATION_MS,
    gap_ms: int = DEFAULT_GAP_MS,
    audio_duration_ms: Mapping[str, int] | None = None,
) -> dict[str, object]:
    """Create a deterministic, non-overlapping projection of CP3A events."""

    _positive_int(source_duration_ms, "source duration")
    _positive_int(max_duration_ms, "maximum duration")
    if isinstance(gap_ms, bool) or not isinstance(gap_ms, int) or gap_ms < 0:
        raise PreviewRenderError("preview gap must be a non-negative integer")

    candidates: list[dict[str, object]] = []
    omitted: list[str] = []
    for raw in voice_events:
        if not isinstance(raw, Mapping):
            raise PreviewRenderError("voice event is not an object")
        identifier = _identifier(raw)
        if _is_source_uncertain(raw):
            omitted.append(identifier)
            continue
        audio_path = raw.get("audioPath")
        if not isinstance(audio_path, str) or not audio_path.strip():
            raise PreviewRenderError(f"cached audio is missing for utterance {identifier}")
        source_start = _integer(raw, "baseStartMs")
        source_end = _integer(raw, "baseEndMs")
        if source_end <= source_start:
            raise PreviewRenderError(f"invalid source window for utterance {identifier}")
        duration = None if audio_duration_ms is None else audio_duration_ms.get(identifier)
        if duration is None:
            duration = raw.get("fittedDurationMs")
        if duration is None:
            duration = raw.get("finalTtsMs")
        _positive_int(duration, f"audio duration for utterance {identifier}")
        candidates.append({
            "source": raw,
            "utteranceId": identifier,
            "audioPath": audio_path,
            "sourceStartMs": source_start,
            "sourceEndMs": source_end,
            "sourceWindowStartMs": _optional_integer(raw, "effectiveStartMs", source_start),
            "sourceWindowEndMs": _optional_integer(raw, "effectiveEndMs", source_end),
            "durationMs": int(duration),
        })

    candidates.sort(key=lambda item: (int(item["sourceStartMs"]), str(item["utteranceId"])))
    scheduled: list[dict[str, object]] = []
    previous_end: int | None = None
    for item in candidates:
        preferred_start = int(item["sourceStartMs"])
        start = preferred_start if previous_end is None else max(preferred_start, previous_end + gap_ms)
        end = start + int(item["durationMs"])
        raw = item["source"]
        scheduled.append({
            "utteranceId": item["utteranceId"],
            "sourcePositions": list(raw.get("sourcePositions") or []),
            "resolvedZh": raw.get("resolvedZh"),
            "nativeViApproved": raw.get("nativeViApproved"),
            "deliveryNativeVi": raw.get("deliveryNativeVi"),
            "audioPath": item["audioPath"],
            "sourceStartMs": item["sourceStartMs"],
            "sourceEndMs": item["sourceEndMs"],
            "sourceWindowStartMs": item["sourceWindowStartMs"],
            "sourceWindowEndMs": item["sourceWindowEndMs"],
            "scheduledStartMs": start,
            "scheduledEndMs": end,
            "durationMs": item["durationMs"],
            "reviewRequired": bool(raw.get("reviewRequired")),
            "reviewReason": raw.get("reviewReason"),
        })
        previous_end = end

    by_id = {str(row["utteranceId"]): row for row in scheduled}
    retimed_cues: list[dict[str, object]] = []
    for cue in subtitle_cues:
        if not isinstance(cue, Mapping):
            raise PreviewRenderError("subtitle cue is not an object")
        identifier = str(cue.get("voiceEventId") or cue.get("utteranceId") or "").strip()
        if identifier not in by_id:
            continue
        start = cue.get("startMs")
        end = cue.get("endMs")
        if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int) or end <= start:
            raise PreviewRenderError(f"invalid subtitle cue for utterance {identifier}")
        event = by_id[identifier]
        source_start = int(event["sourceWindowStartMs"])
        source_end = int(event["sourceWindowEndMs"])
        if source_end <= source_start:
            source_start = int(event["sourceStartMs"])
            source_end = int(event["sourceEndMs"])
        cue_start = max(source_start, min(start, source_end))
        cue_end = max(cue_start + 1, min(end, source_end))
        source_span = max(1, source_end - source_start)
        scheduled_start = int(event["scheduledStartMs"]) + round((cue_start - source_start) * int(event["durationMs"]) / source_span)
        scheduled_end = int(event["scheduledStartMs"]) + round((cue_end - source_start) * int(event["durationMs"]) / source_span)
        scheduled_start = max(int(event["scheduledStartMs"]), min(scheduled_start, int(event["scheduledEndMs"]) - 1))
        scheduled_end = max(scheduled_start + 1, min(scheduled_end, int(event["scheduledEndMs"])))
        retimed_cues.append({
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

    duration_ms = max(source_duration_ms, previous_end or 0)
    if duration_ms > max_duration_ms:
        raise PreviewRenderError(f"preview duration {duration_ms}ms exceeds {max_duration_ms}ms")
    return {
        "schemaVersion": 1,
        "artifact": "preview_voice_timeline_v1",
        "sourceDurationMs": source_duration_ms,
        "durationMs": duration_ms,
        "gapMs": gap_ms,
        "voiceEvents": scheduled,
        "subtitleCues": retimed_cues,
        "includedUtteranceIds": [str(row["utteranceId"]) for row in scheduled],
        "omittedUtteranceIds": omitted,
    }


def build_ffmpeg_command(
    source: Path,
    output: Path,
    ass_path: Path,
    voice_inputs: Sequence[tuple[Path, int]],
    *,
    duration_ms: int,
    source_duration_ms: int,
    width: int = 720,
    height: int = 1280,
    ffmpeg_binary: str = "ffmpeg",
) -> list[str]:
    """Build the preview-only compatible H.264/AAC command."""

    _positive_int(duration_ms, "preview duration")
    _positive_int(source_duration_ms, "source duration")
    if not voice_inputs:
        raise PreviewRenderError("preview needs at least one cached voice input")
    extension = max(0.0, (duration_ms - source_duration_ms) / 1000.0)
    subtitle_file = _filter_path(ass_path)
    filters = [
        f"[0:v]tpad=stop_mode=clone:stop_duration={extension:.3f},subtitles='{subtitle_file}'[vout]",
    ]
    labels: list[str] = []
    command = [ffmpeg_binary, "-hide_banner", "-y", "-i", str(source)]
    for index, (audio_path, start_ms) in enumerate(voice_inputs, start=1):
        if isinstance(start_ms, bool) or not isinstance(start_ms, int) or start_ms < 0:
            raise PreviewRenderError("voice start must be a non-negative integer")
        command.extend(["-i", str(audio_path)])
        delay = f"{start_ms}|{start_ms}"
        label = f"a{index}"
        labels.append(f"[{label}]")
        filters.append(
            f"[{index}:a]aresample=48000,aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,"
            f"adelay={delay}[{label}]"
        )
    filters.append(
        "".join(labels)
        + f"amix=inputs={len(labels)}:duration=longest:dropout_transition=0:normalize=0,"
        f"atrim=duration={duration_ms / 1000.0:.3f},asetpts=PTS-STARTPTS[aout]"
    )
    command.extend([
        "-filter_complex", ";".join(filters),
        "-map", "[vout]", "-map", "[aout]",
        "-t", f"{duration_ms / 1000.0:.3f}",
        "-c:v", "libx264", "-profile:v", "main", "-level:v", "4.1",
        "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20", "-threads", "2",
        "-c:a", "aac", "-profile:a", "aac_low", "-b:a", "160k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart", str(output),
    ])
    return command


def write_ass_subtitles(
    cues: Sequence[Mapping[str, object]],
    path: Path,
    *,
    width: int = 720,
    height: int = 1280,
    placement_evidence: Sequence[Mapping[str, object]] | None = None,
    source_to_output: object | None = None,
) -> dict[str, object]:
    """Write measured, bounded subtitle plates at the source's native resolution."""

    return _write_layout_ass_subtitles(
        cues,
        path,
        width=width,
        height=height,
        placement_evidence=placement_evidence,
        source_to_output=source_to_output,
    )


def render_preview(command: Sequence[str]) -> None:
    """Execute one local render command and fail closed on ffmpeg errors."""

    try:
        subprocess.run(list(command), check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PreviewRenderError("preview ffmpeg render failed") from exc


def write_json(path: Path, document: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _is_source_uncertain(row: Mapping[str, object]) -> bool:
    if str(row.get("reviewReason") or "").upper() == "SOURCE_UNCERTAIN":
        return True
    flags = row.get("reviewFlags")
    return isinstance(flags, (list, tuple)) and any(str(flag).upper() == "SOURCE_UNCERTAIN" for flag in flags)


def _identifier(row: Mapping[str, object]) -> str:
    value = str(row.get("utteranceId") or "").strip()
    if not value:
        raise PreviewRenderError("voice event identity is missing")
    return value


def _integer(row: Mapping[str, object], key: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise PreviewRenderError(f"{key} must be an integer")
    return value


def _optional_integer(row: Mapping[str, object], key: str, default: int) -> int:
    value = row.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value


def _positive_int(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PreviewRenderError(f"{label} must be a positive integer")


def _filter_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")


def _ass_time(value_ms: int) -> str:
    hours, remainder = divmod(max(0, value_ms), 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, centiseconds = divmod(remainder, 1000)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds // 10:02d}"
