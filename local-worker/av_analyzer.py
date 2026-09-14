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


def validate_av_analysis(document: object) -> dict[str, object]:
    if not isinstance(document, dict) or set(document) != {"utterances", "visual_only"}:
        raise ValueError("AV analysis schema is invalid")
    for key in ("utterances", "visual_only"):
        values = document[key]
        if not isinstance(values, list):
            raise ValueError("AV analysis list is invalid")
        for value in values:
            if not isinstance(value, dict) or set(value) != {"start_ms", "end_ms", "text_zh"}:
                raise ValueError("AV analysis item is invalid")
            if not isinstance(value["start_ms"], int) or not isinstance(value["end_ms"], int) or value["start_ms"] < 0 or value["end_ms"] <= value["start_ms"] or not isinstance(value["text_zh"], str) or not value["text_zh"].strip():
                raise ValueError("AV analysis item values are invalid")
    return document


def persist_diagnostics(path: Path, document: Mapping[str, object]) -> None:
    safe = {key: value for key, value in document.items() if key != "debug"}
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
                {"text": "Return JSON only with utterances and visual_only. Each item needs start_ms,end_ms,text_zh. Visual-only is never dialogue."},
                {"inline_data": {"mime_type": "video/mp4", "data": base64.b64encode(media).decode("ascii")}},
            ]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0},
        }
        return validate_av_analysis(self.transport.call(payload, required_keys={"utterances", "visual_only"}))

    def analyze_cached(self, media_path: Path, cache_path: Path, *, start_ms: int, end_ms: int) -> dict[str, object]:
        if cache_path.is_file():
            return validate_av_analysis(json.loads(cache_path.read_text(encoding="utf-8")))
        result = self.analyze_media(media_path, start_ms=start_ms, end_ms=end_ms)
        persist_diagnostics(cache_path, result)
        return result
