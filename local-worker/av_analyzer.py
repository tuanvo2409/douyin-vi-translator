"""Bounded multimodal analysis, deliberately separate from the legacy worker."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Mapping

from llm_transport import StructuredGeminiTransport, gemini_json_transport

MAX_WINDOW_MS = 30_000
WINDOW_OVERLAP_MS = 5_000
MAX_DIAGNOSTIC_BYTES = 64 * 1024
_AV_KEYS = {"windowStartMs", "windowEndMs", "scene", "utterances", "visualOnly"}
_SCENE_KEYS = {"summary", "tone", "register"}
_UTTERANCE_KEYS = {
    "startMs", "endMs", "spokenZh", "subtitleZh", "speakerRef", "intent", "emotion", "confidence",
}
_VISUAL_KEYS = {"startMs", "endMs", "textZh", "confidence"}
_CONFIDENCES = {"HIGH", "MEDIUM", "LOW"}


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _valid_interval(value: Mapping[str, object], start_key: str, end_key: str) -> None:
    start, end = value[start_key], value[end_key]
    if not _is_int(start) or not _is_int(end) or start < 0 or end <= start:
        raise ValueError("AV analysis interval is invalid")


def _optional_text(value: object) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError("AV analysis text field is invalid")


def validate_av_analysis(document: object) -> dict[str, object]:
    if not isinstance(document, dict) or set(document) != _AV_KEYS:
        raise ValueError("AV analysis schema is invalid")
    _valid_interval(document, "windowStartMs", "windowEndMs")
    window_start, window_end = document["windowStartMs"], document["windowEndMs"]
    scene = document["scene"]
    if not isinstance(scene, dict) or set(scene) != _SCENE_KEYS or any(
        not isinstance(scene[key], str) or not scene[key].strip() for key in _SCENE_KEYS
    ):
        raise ValueError("AV analysis scene is invalid")
    if not isinstance(document["utterances"], list) or not isinstance(document["visualOnly"], list):
        raise ValueError("AV analysis lists are invalid")
    for value in document["utterances"]:
        if not isinstance(value, dict) or set(value) != _UTTERANCE_KEYS:
            raise ValueError("AV analysis utterance is invalid")
        _valid_interval(value, "startMs", "endMs")
        if value["startMs"] < window_start or value["endMs"] > window_end:
            raise ValueError("AV analysis utterance is outside its window")
        if value["spokenZh"] is None and value["subtitleZh"] is None:
            raise ValueError("AV analysis utterance has no text evidence")
        for key in ("spokenZh", "subtitleZh", "speakerRef", "intent", "emotion"):
            _optional_text(value[key])
        if value["confidence"] not in _CONFIDENCES:
            raise ValueError("AV analysis confidence is invalid")
    for value in document["visualOnly"]:
        if not isinstance(value, dict) or set(value) != _VISUAL_KEYS:
            raise ValueError("AV analysis visual-only item is invalid")
        _valid_interval(value, "startMs", "endMs")
        if value["startMs"] < window_start or value["endMs"] > window_end:
            raise ValueError("AV analysis visual-only item is outside its window")
        if not isinstance(value["textZh"], str) or not value["textZh"].strip() or value["confidence"] not in _CONFIDENCES:
            raise ValueError("AV analysis visual-only values are invalid")
    return document


def persist_diagnostics(path: Path, document: Mapping[str, object]) -> None:
    safe = {key: value for key, value in document.items() if key != "debug"}
    validate_av_analysis(safe)
    payload = json.dumps(safe, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(payload) > MAX_DIAGNOSTIC_BYTES:
        raise ValueError("AV diagnostics exceed bounded size")
    path.write_bytes(payload)


def analysis_windows(duration_ms: int) -> list[tuple[int, int]]:
    if duration_ms <= 0:
        raise ValueError("AV duration must be positive")
    windows = []
    start = 0
    while start < duration_ms:
        end = min(duration_ms, start + MAX_WINDOW_MS)
        windows.append((start, end))
        if end == duration_ms:
            break
        start = end - WINDOW_OVERLAP_MS
    return windows


class GeminiAVAnalyzer:
    def __init__(self, transport: StructuredGeminiTransport | None = None, *, model: str | None = None) -> None:
        self.model = model or os.environ.get("DUBVI_AV_MODEL", "gemini-2.5-flash")
        self.transport = transport or gemini_json_transport(model=self.model)

    def analyze_media(self, media_path: Path, *, start_ms: int = 0, end_ms: int | None = None) -> dict[str, object]:
        if end_ms is None or end_ms - start_ms > MAX_WINDOW_MS:
            raise ValueError("AV analysis window must be at most 30 seconds")
        media = media_path.read_bytes()
        payload = {
            "contents": [{"role": "user", "parts": [
                {"text": "Return JSON only using exactly windowStartMs, windowEndMs, scene, utterances, and visualOnly. Each utterance must separately provide spokenZh and subtitleZh plus speakerRef, intent, emotion, and confidence HIGH/MEDIUM/LOW. Visual-only text belongs only in visualOnly and is never dialogue."},
                {"inline_data": {"mime_type": "video/mp4", "data": base64.b64encode(media).decode("ascii")}},
            ]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0},
        }
        return validate_av_analysis(self.transport.call(payload, required_keys=_AV_KEYS))

    def analyze_cached(self, media_path: Path, cache_path: Path, *, start_ms: int, end_ms: int) -> dict[str, object]:
        if cache_path.is_file():
            return validate_av_analysis(json.loads(cache_path.read_text(encoding="utf-8")))
        result = self.analyze_media(media_path, start_ms=start_ms, end_ms=end_ms)
        persist_diagnostics(cache_path, result)
        return result
