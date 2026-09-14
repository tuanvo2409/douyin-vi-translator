"""Small, injectable structured Gemini transport for offline-testable callers."""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Mapping

import requests


class ProviderError(RuntimeError):
    def __init__(self, classification: str, *, transient: bool = False) -> None:
        super().__init__(classification)
        self.classification = classification
        self.transient = transient


class StructuredResponseError(ValueError):
    pass


def validate_structured_document(document: object, required_keys: set[str]) -> dict[str, object]:
    if not isinstance(document, dict) or set(document) != required_keys:
        raise StructuredResponseError("structured response schema is invalid")
    return dict(document)


class StructuredGeminiTransport:
    """Retries only explicitly transient, injectable requests; never logs payloads."""

    def __init__(self, *, request: Callable[[Mapping[str, object]], object], max_attempts: int = 2) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self._request = request
        self._max_attempts = max_attempts

    def call(self, payload: Mapping[str, object], *, required_keys: set[str]) -> dict[str, object]:
        for attempt in range(self._max_attempts):
            try:
                return validate_structured_document(self._request(payload), required_keys)
            except ProviderError as error:
                if not error.transient or attempt + 1 == self._max_attempts:
                    raise
        raise AssertionError("unreachable")


def gemini_json_transport(*, model: str, api_key: str | None = None) -> StructuredGeminiTransport:
    """Build the live provider adapter without exposing its credential in diagnostics."""
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ProviderError("PROVIDER_UNAVAILABLE")

    def request(payload: Mapping[str, object]) -> object:
        url = "https://generativelanguage.googleapis.com/v1beta/models/" + model + ":generateContent?key=" + key
        try:
            response = requests.post(url, json=dict(payload), timeout=60)
        except requests.RequestException as error:
            raise ProviderError("PROVIDER_UNAVAILABLE", transient=True) from error
        if response.status_code != 200:
            raise ProviderError("HTTP_" + str(response.status_code), transient=response.status_code == 429 or response.status_code >= 500)
        try:
            text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(text)
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise StructuredResponseError("provider structured response is invalid") from error

    return StructuredGeminiTransport(request=request)
