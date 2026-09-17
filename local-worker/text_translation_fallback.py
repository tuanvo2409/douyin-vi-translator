"""Strict text-only Groq fallback for the canonical CP2 translation boundary."""

from __future__ import annotations

import json
import math
import time
from typing import Callable, Mapping, Sequence

from llm_transport import (
    ProviderError,
    StructuredGeminiTransport,
    groq_json_transport,
    is_transient_http_status,
)
from vn_localizer import validate_translation_document


GROQ_TEXT_MODEL = "qwen/qwen3.8-27b"
MAX_GROQ_TEXT_BATCH = 5
MAX_GROQ_TEXT_COMPLETION_TOKENS = 768
MAX_GROQ_TEXT_RETRY_DELAY_SECONDS = 60.0


def _translation_key(operation: str) -> str:
    if operation == "semantic":
        return "semanticVi"
    if operation == "native":
        return "nativeVi"
    raise ValueError("Groq text fallback supports only semantic and native translation")


def _source_rows(records: Sequence[Mapping[str, object]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for record in records:
        identifier = record.get("utteranceId")
        source = record.get("resolvedZh")
        if not isinstance(identifier, str) or not identifier.strip() or identifier in seen:
            raise ValueError("Groq text input identity is invalid")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("Groq text input source is invalid")
        seen.add(identifier)
        result.append({"utteranceId": identifier, "sourceText": source.strip()})
    return result


def groq_text_payload(
    records: Sequence[Mapping[str, object]],
    *,
    operation: str,
    model: str = GROQ_TEXT_MODEL,
    max_completion_tokens: int = MAX_GROQ_TEXT_COMPLETION_TOKENS,
) -> dict[str, object]:
    """Build one bounded OpenAI-compatible request containing text only."""

    key = _translation_key(operation)
    rows = _source_rows(records)
    if not rows or len(rows) > MAX_GROQ_TEXT_BATCH:
        raise ValueError("Groq text batch size is invalid")
    if not isinstance(max_completion_tokens, int) or isinstance(max_completion_tokens, bool) or not 1 <= max_completion_tokens <= MAX_GROQ_TEXT_COMPLETION_TOKENS:
        raise ValueError("Groq text completion limit is invalid")
    document = {"utterances": rows}
    instruction = (
        "Translate each sourceText from Chinese to natural spoken Vietnamese. "
        f"Return JSON only with exactly {{\"translations\":[{{\"utteranceId\":\"...\",\"{key}\":\"...\"}}]}}. "
        "Preserve every utteranceId exactly. Do not add, remove, merge, split, or reorder identity. "
        "Do not provide timing, OCR, scene, or explanatory fields."
    )
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": json.dumps(document, ensure_ascii=False, separators=(",", ":"))},
        ],
        "temperature": 0,
        "reasoning_effort": "none",
        "max_completion_tokens": max_completion_tokens,
        "response_format": {"type": "json_object"},
    }


def _retry_delay(failure_number: int, error: ProviderError) -> float:
    retry_after = error.retry_after
    if isinstance(retry_after, (int, float)) and not isinstance(retry_after, bool) and math.isfinite(float(retry_after)) and retry_after >= 0:
        return min(float(retry_after), MAX_GROQ_TEXT_RETRY_DELAY_SECONDS)
    return min(float(2 ** (failure_number - 1)), 10.0)


def groq_text_transport(
    *,
    request: Callable[[Mapping[str, object]], object] | None = None,
    model: str = GROQ_TEXT_MODEL,
    sleeper: Callable[[float], None] = time.sleep,
) -> StructuredGeminiTransport:
    """Create a sequential transport with at most one bounded 429 retry."""

    if request is not None:
        return StructuredGeminiTransport(
            request=request,
            max_transport_attempts=2,
            sleeper=sleeper,
            delay_calculator=_retry_delay,
        )
    return groq_json_transport(
        model=model,
        max_transport_attempts=2,
        sleeper=sleeper,
        delay_calculator=_retry_delay,
    )


class GroqTextTranslationAdapter:
    """Batch, validate, and reassemble Groq text translations by canonical ID."""

    def __init__(
        self,
        *,
        transport: object | None = None,
        model: str = GROQ_TEXT_MODEL,
        batch_size: int = MAX_GROQ_TEXT_BATCH,
        max_completion_tokens: int = MAX_GROQ_TEXT_COMPLETION_TOKENS,
    ) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or not 1 <= batch_size <= MAX_GROQ_TEXT_BATCH:
            raise ValueError("Groq text batch size is invalid")
        self.model = model
        self.batch_size = batch_size
        self.max_completion_tokens = max_completion_tokens
        self.transport = transport or groq_text_transport(model=model)
        self.batch_count = 0

    def __call__(self, payload: Mapping[str, object]) -> dict[str, object]:
        operation = str(payload.get("mode") or "")
        key = _translation_key(operation)
        records = payload.get("utterances")
        if not isinstance(records, list) or not all(isinstance(record, Mapping) for record in records):
            raise ValueError("Groq text utterance input is invalid")
        source_records = [dict(record) for record in records]
        _source_rows(source_records)
        translated: dict[str, str] = {}
        for start in range(0, len(source_records), self.batch_size):
            batch = source_records[start:start + self.batch_size]
            raw = self.transport.call_raw(
                groq_text_payload(
                    batch,
                    operation=operation,
                    model=self.model,
                    max_completion_tokens=self.max_completion_tokens,
                )
            )
            document = json.loads(raw) if isinstance(raw, str) else raw
            values = validate_translation_document(document, batch, key)
            translated.update(values)
            self.batch_count += 1
        expected = [str(record["utteranceId"]) for record in source_records]
        if set(translated) != set(expected) or len(translated) != len(expected):
            raise ValueError("Groq text translation coverage is invalid")
        return {"translations": [{"utteranceId": identifier, key: translated[identifier]} for identifier in expected]}


def _gemini_fallback_allowed(error: ProviderError) -> bool:
    if error.classification in {"NO_READY_GEMINI_KEY", "PROVIDER_UNAVAILABLE"}:
        return True
    return error.http_status is not None and (error.http_status == 429 or is_transient_http_status(error.http_status))


class GeminiPrimaryWithGroqTextFallback:
    """Keep Gemini primary; invoke Groq only for provider availability/quota failures."""

    def __init__(self, gemini: Callable[[Mapping[str, object]], object], groq: Callable[[Mapping[str, object]], object]) -> None:
        self.gemini = gemini
        self.groq = groq
        self.last_provider = "gemini"
        self.gemini_failure: str | None = None

    def __call__(self, payload: Mapping[str, object]) -> object:
        try:
            result = self.gemini(payload)
            self.last_provider = "gemini"
            return result
        except ProviderError as error:
            if not _gemini_fallback_allowed(error):
                raise
            self.gemini_failure = error.classification
            self.last_provider = "groq_text"
            return self.groq(payload)


__all__ = [
    "GROQ_TEXT_MODEL",
    "MAX_GROQ_TEXT_BATCH",
    "MAX_GROQ_TEXT_COMPLETION_TOKENS",
    "GeminiPrimaryWithGroqTextFallback",
    "GroqTextTranslationAdapter",
    "groq_text_payload",
    "groq_text_transport",
]
