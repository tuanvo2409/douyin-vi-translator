"""Bounded Groq vision adapter for AV-only provider recovery."""

from __future__ import annotations

import base64
import json
import os
import subprocess
from pathlib import Path
from typing import Callable, Mapping, Sequence

from llm_transport import ProviderError, StructuredResponseError, groq_json_transport


GROQ_VISION_MODEL = "qwen/qwen3.8-27b"
MAX_GROQ_FRAMES = 2
MAX_GROQ_COMPLETION_TOKENS = 1024


class GroqFrameExtractionError(RuntimeError):
    """The local source could not produce a bounded JPEG frame."""


def sample_window_timestamps(start_ms: int, end_ms: int, count: int = MAX_GROQ_FRAMES) -> list[int]:
    """Return deterministic evenly spaced interior timestamps."""
    if not isinstance(start_ms, int) or not isinstance(end_ms, int) or end_ms <= start_ms:
        raise ValueError("Groq vision window is invalid")
    count = max(1, min(MAX_GROQ_FRAMES, int(count)))
    span = end_ms - start_ms
    if span <= 2:
        return [start_ms + 1] if span == 2 else []
    result: list[int] = []
    for index in range(1, count + 1):
        timestamp = start_ms + round(span * index / (count + 1))
        timestamp = max(start_ms + 1, min(end_ms - 1, timestamp))
        if start_ms < timestamp < end_ms and timestamp not in result:
            result.append(timestamp)
    return result


def _ffmpeg_jpeg(media_path: Path, timestamp_ms: int) -> bytes:
    try:
        completed = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{timestamp_ms / 1000:.3f}", "-i", str(media_path),
                "-frames:v", "1", "-vf", "scale=768:-2", "-q:v", "3",
                "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
            ],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise GroqFrameExtractionError("Groq vision frame extraction failed") from error
    data = bytes(completed.stdout or b"")
    if not data.startswith(b"\xff\xd8"):
        raise GroqFrameExtractionError("Groq vision frame extraction returned no JPEG")
    return data


def extract_window_frames(
    media_path: Path,
    start_ms: int,
    end_ms: int,
    *,
    frame_extractor: Callable[[Path, int], bytes] | None = None,
) -> list[dict[str, object]]:
    """Extract at most two timestamped JPEGs, failing closed on media errors."""
    extractor = frame_extractor or _ffmpeg_jpeg
    frames: list[dict[str, object]] = []
    for timestamp_ms in sample_window_timestamps(start_ms, end_ms):
        frame = extractor(media_path, timestamp_ms)
        if not isinstance(frame, (bytes, bytearray)) or not bytes(frame).startswith(b"\xff\xd8"):
            raise GroqFrameExtractionError("Groq vision frame is not JPEG")
        frames.append({"timestampMs": timestamp_ms, "jpegBase64": base64.b64encode(bytes(frame)).decode("ascii")})
    if not frames:
        raise GroqFrameExtractionError("Groq vision window has no interior frame")
    return frames


_EVIDENCE_KEYS = (
    "position", "cue_id", "cueId", "utteranceId", "startMs", "endMs", "start_ms", "end_ms",
    "asrTextZh", "asr_text_zh", "sourceTextZh", "source_text_zh", "ocrTextZh", "ocr_text_zh",
    "textZh", "text", "classification", "source_confidence", "confidence", "kind",
)


def _record_bounds(record: Mapping[str, object]) -> tuple[int, int] | None:
    start = record.get("startMs", record.get("start_ms"))
    end = record.get("endMs", record.get("end_ms"))
    if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
        return None
    return int(start), int(end)


def _overlaps_window(record: Mapping[str, object], start_ms: int, end_ms: int) -> bool:
    bounds = _record_bounds(record)
    return bounds is not None and bounds[0] < end_ms and start_ms < bounds[1]


def _safe_evidence_record(record: Mapping[str, object]) -> dict[str, object]:
    return {
        key: record[key]
        for key in _EVIDENCE_KEYS
        if key in record and isinstance(record[key], (str, int, float, bool))
    }


def _iter_records(value: object) -> list[Mapping[str, object]]:
    if isinstance(value, Mapping):
        nested = value.get("items")
        if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes, bytearray)):
            return [item for item in nested if isinstance(item, Mapping)]
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def collect_window_evidence(
    cues: Sequence[Mapping[str, object]],
    unresolved: Mapping[str, object] | None,
    start_ms: int,
    end_ms: int,
) -> dict[str, list[dict[str, object]]]:
    """Select only bounded transcript/OCR records overlapping this AV window."""
    transcript: list[dict[str, object]] = []
    ocr: list[dict[str, object]] = []
    values: list[Mapping[str, object]] = list(cues) + _iter_records(unresolved)
    for record in values:
        if not _overlaps_window(record, start_ms, end_ms):
            continue
        safe = _safe_evidence_record(record)
        if not safe:
            continue
        if any(key in safe for key in ("asrTextZh", "asr_text_zh", "sourceTextZh", "source_text_zh")):
            transcript.append(safe)
        if any(key in safe for key in ("ocrTextZh", "ocr_text_zh", "textZh", "text")):
            ocr.append(safe)
    return {"transcript": transcript[:32], "ocr": ocr[:32]}


def build_groq_vision_payload(
    frames: Sequence[Mapping[str, object]],
    start_ms: int,
    end_ms: int,
    *,
    model: str = GROQ_VISION_MODEL,
    cues: Sequence[Mapping[str, object]] = (),
    unresolved: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build an OpenAI-compatible multimodal request with canonical AV instructions."""
    bounded_frames = list(frames)[:MAX_GROQ_FRAMES]
    evidence = collect_window_evidence(cues, unresolved, start_ms, end_ms)
    evidence_text = json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))
    prompt = (
        f"Analyze only source window {start_ms}-{end_ms} ms. Use only the supplied visual frames "
        "and transcript/OCR evidence; do not invent dialogue or unsupported speaker identity. "
        "Keep every timestamp within the requested window. Preserve source cue/utterance labels "
        "exactly where supplied. Visual-only evidence must remain in visualOnly and never become "
        "dialogue. Keep uncertain evidence uncertain. Return canonical JSON only using exactly "
        "windowStartMs, windowEndMs, scene, utterances, and visualOnly. Each utterance must include "
        "spokenZh, subtitleZh, speakerRef, intent, emotion, and confidence (HIGH/MEDIUM/LOW). "
        f"TRANSCRIPT_OCR_EVIDENCE={evidence_text}"
    )
    content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
    for frame in bounded_frames:
        timestamp = frame.get("timestampMs")
        encoded = frame.get("jpegBase64")
        if not isinstance(timestamp, int) or not isinstance(encoded, str):
            raise ValueError("Groq vision frame metadata is invalid")
        content.append({"type": "text", "text": f"Frame timestampMs={timestamp}"})
        content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded}})
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "reasoning_effort": "none",
        "max_tokens": MAX_GROQ_COMPLETION_TOKENS,
        "response_format": {"type": "json_object"},
    }


class GroqVisionAnalyzer:
    """One-shot Groq AV recovery path; canonical validation remains local."""

    def __init__(
        self,
        *,
        transport: object | None = None,
        model: str = GROQ_VISION_MODEL,
        frame_extractor: Callable[[Path, int], bytes] | None = None,
    ) -> None:
        self.model = model
        self.transport = transport or groq_json_transport(model=model)
        self.frame_extractor = frame_extractor
        self.last_recovery: dict[str, object] = {}

    def analyze_media(
        self,
        media_path: Path,
        *,
        start_ms: int,
        end_ms: int,
        cues: Sequence[Mapping[str, object]] = (),
        unresolved: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        import av_analyzer

        if not isinstance(start_ms, int) or not isinstance(end_ms, int) or end_ms <= start_ms or end_ms - start_ms > av_analyzer.MAX_WINDOW_MS:
            raise ValueError("Groq vision window is invalid")

        self.last_recovery = {
            "windowStartMs": start_ms,
            "windowEndMs": end_ms,
            "provider": "groq",
            "model": self.model,
            "frameCount": 0,
            "transcriptContextUsed": False,
            "schemaValidationStatus": "NOT_RUN",
            "normalizationApplied": False,
            "coordinateNormalizationApplied": False,
            "boundaryNormalizationApplied": False,
            "finalStatus": "RUNNING",
        }
        try:
            frames = extract_window_frames(
                media_path,
                start_ms,
                end_ms,
                frame_extractor=self.frame_extractor,
            )
            self.last_recovery["frameCount"] = len(frames)
            evidence = collect_window_evidence(cues, unresolved, start_ms, end_ms)
            self.last_recovery["transcriptContextUsed"] = bool(evidence["transcript"] or evidence["ocr"])
            payload = build_groq_vision_payload(
                frames,
                start_ms,
                end_ms,
                model=self.model,
                cues=cues,
                unresolved=unresolved,
            )
            try:
                raw = self.transport.call_raw(payload)
            except StructuredResponseError as error:
                self.last_recovery["finalStatus"] = "AV_SCHEMA_INVALID_FINAL"
                raise av_analyzer.AVResponseRecoveryError("PROVIDER_RESPONSE_ENVELOPE_INVALID") from error
            except ProviderError as error:
                transport = getattr(self.transport, "last_call", {})
                self.last_recovery.update({
                    "httpStatus": transport.get("httpStatus", error.http_status),
                    "finalTransportStatus": transport.get("finalTransportStatus", error.classification),
                    "rateLimitClassification": getattr(error, "rate_limit_classification", None),
                    "rateLimitMetadata": dict(getattr(error, "rate_limit_metadata", {})),
                })
                self.last_recovery["finalStatus"] = "AV_PROVIDER_ERROR"
                raise
            transport = getattr(self.transport, "last_call", {})
            self.last_recovery["transportAttempts"] = transport.get("transportAttempts", 1)
            self.last_recovery["httpStatus"] = transport.get("httpStatus", 200)
            self.last_recovery["finalTransportStatus"] = transport.get("finalTransportStatus", "HTTP_200")
            candidate, normalized = av_analyzer._normalize_structural_response(raw)
            candidate, coordinate_normalized = av_analyzer._normalize_window_coordinates(candidate, start_ms, end_ms)
            candidate, boundary_normalized = av_analyzer._normalize_boundary_utterance_window(candidate, start_ms, end_ms)
            self.last_recovery["normalizationApplied"] = bool(normalized)
            self.last_recovery["coordinateNormalizationApplied"] = bool(coordinate_normalized)
            self.last_recovery["boundaryNormalizationApplied"] = bool(boundary_normalized)
            try:
                result = av_analyzer.validate_av_analysis(candidate)
            except ValueError as error:
                failure_class = av_analyzer._failure_class(candidate, error)
                self.last_recovery["schemaValidationStatus"] = "INVALID"
                self.last_recovery["finalStatus"] = "AV_SCHEMA_INVALID_FINAL"
                raise av_analyzer.AVResponseRecoveryError(failure_class) from error
            self.last_recovery["schemaValidationStatus"] = "VALID"
            self.last_recovery["finalStatus"] = "AV_SCHEMA_VALID_GROQ_FALLBACK"
            return result
        except GroqFrameExtractionError:
            self.last_recovery["finalStatus"] = "AV_FRAME_EXTRACTION_FAILED"
            raise


__all__ = [
    "GROQ_VISION_MODEL",
    "MAX_GROQ_FRAMES",
    "MAX_GROQ_COMPLETION_TOKENS",
    "GroqFrameExtractionError",
    "GroqVisionAnalyzer",
    "build_groq_vision_payload",
    "collect_window_evidence",
    "extract_window_frames",
    "sample_window_timestamps",
]
