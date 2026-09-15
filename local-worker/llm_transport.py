"""Small, injectable structured Gemini transport for offline-testable callers."""
from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Callable, Mapping

import requests

from gemini_pool import gemini_pool


TRANSIENT_HTTP_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_RETRY_DELAY_SECONDS = 10.0


def is_transient_http_status(status_code: int) -> bool:
    return status_code in TRANSIENT_HTTP_STATUSES


class ProviderError(RuntimeError):
    def __init__(
        self,
        classification: str,
        *,
        transient: bool = False,
        http_status: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(classification)
        self.classification = classification
        self.transient = transient
        self.http_status = http_status
        self.retry_after = retry_after
        self.transport_attempts = 0
        self.transient_retry_used = False


class StructuredResponseError(ValueError):
    pass


def calculate_retry_delay(
    failure_number: int,
    error: ProviderError,
    *,
    max_delay: float = MAX_RETRY_DELAY_SECONDS,
) -> float:
    """Return bounded exponential backoff, preferring a bounded Retry-After."""
    if failure_number < 1 or max_delay < 0:
        raise ValueError("retry delay inputs are invalid")
    retry_after = error.retry_after
    if isinstance(retry_after, (int, float)) and math.isfinite(float(retry_after)) and retry_after >= 0:
        delay = float(retry_after)
    else:
        delay = float(2 ** (failure_number - 1))
    return min(delay, float(max_delay))


def validate_structured_document(document: object, required_keys: set[str]) -> dict[str, object]:
    if not isinstance(document, dict) or set(document) != required_keys:
        raise StructuredResponseError("structured response schema is invalid")
    return dict(document)


class StructuredGeminiTransport:
    """Retries only explicitly transient, injectable requests; never logs payloads."""

    def __init__(
        self,
        *,
        request: Callable[[Mapping[str, object]], object],
        max_attempts: int | None = None,
        max_transport_attempts: int | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        delay_calculator: Callable[[int, ProviderError], float] | None = None,
    ) -> None:
        if max_attempts is not None and max_transport_attempts is not None:
            raise ValueError("configure only one transport attempt limit")
        configured_attempts = (
            max_transport_attempts
            if max_transport_attempts is not None
            else max_attempts
            if max_attempts is not None
            else 3
        )
        if configured_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self._request = request
        self._max_transport_attempts = configured_attempts
        self._sleeper = sleeper
        self._delay_calculator = delay_calculator or calculate_retry_delay
        self.last_call: dict[str, object] = {}

    def call_raw(self, payload: Mapping[str, object], *, max_attempts: int | None = None) -> object:
        """Return one provider document after bounded transport-only retries."""
        transport_limit = self._max_transport_attempts if max_attempts is None else max_attempts
        if transport_limit < 1:
            raise ValueError("max_attempts must be positive")
        self.last_call = {}
        for attempt in range(transport_limit):
            try:
                result = self._request(payload)
            except ProviderError as error:
                error.transport_attempts = attempt + 1
                error.transient_retry_used = attempt > 0
                if not error.transient or attempt + 1 == transport_limit:
                    self.last_call = {
                        "transportAttempts": attempt + 1,
                        "httpStatus": error.http_status,
                        "finalTransportStatus": error.classification,
                        "transientRetryUsed": error.transient_retry_used,
                    }
                    raise
                self._sleeper(self._delay_calculator(attempt + 1, error))
            except requests.RequestException as cause:
                error = ProviderError("PROVIDER_UNAVAILABLE", transient=True)
                error.transport_attempts = attempt + 1
                error.transient_retry_used = attempt > 0
                if attempt + 1 == transport_limit:
                    self.last_call = {
                        "transportAttempts": attempt + 1,
                        "httpStatus": None,
                        "finalTransportStatus": error.classification,
                        "transientRetryUsed": error.transient_retry_used,
                    }
                    raise error from cause
                self._sleeper(self._delay_calculator(attempt + 1, error))
            else:
                self.last_call = {
                    "transportAttempts": attempt + 1,
                    "httpStatus": 200,
                    "finalTransportStatus": "HTTP_200",
                    "transientRetryUsed": attempt > 0,
                }
                return result
        raise AssertionError("unreachable")

    def call(self, payload: Mapping[str, object], *, required_keys: set[str]) -> dict[str, object]:
        return validate_structured_document(
            self.call_raw(payload),
            required_keys,
        )


def _retry_after_seconds(response: object) -> float | None:
    headers = getattr(response, "headers", None)
    if headers is None or not hasattr(headers, "get"):
        return None
    raw = headers.get("Retry-After")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def gemini_json_transport(
    *,
    model: str,
    api_key: str | None = None,
    max_transport_attempts: int = 3,
    sleeper: Callable[[float], None] = time.sleep,
    delay_calculator: Callable[[int, ProviderError], float] | None = None,
) -> StructuredGeminiTransport:
    """Build the live provider adapter without exposing its credential in diagnostics."""
    def request(payload: Mapping[str, object]) -> object:
        key = gemini_pool.get_key()
        if not key:
            if gemini_pool.keys:
                raise ProviderError("NO_READY_GEMINI_KEY")
            key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ProviderError("PROVIDER_UNAVAILABLE")
        url = "https://generativelanguage.googleapis.com/v1beta/models/" + model + ":generateContent?key=" + key
        try:
            response = requests.post(url, json=dict(payload), timeout=60)
        except requests.RequestException as error:
            raise ProviderError("PROVIDER_UNAVAILABLE", transient=True) from error
        if response.status_code != 200:
            gemini_pool.report_error(key, response.status_code)
            raise ProviderError(
                "HTTP_" + str(response.status_code),
                transient=is_transient_http_status(response.status_code),
                http_status=response.status_code,
                retry_after=_retry_after_seconds(response),
            )
        try:
            text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            try:
                return json.loads(text)
            except (TypeError, ValueError):
                # Let a higher-level canonical boundary safely normalize a
                # serialization wrapper or fail closed without losing it.
                return text
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise StructuredResponseError("provider structured response is invalid") from error

    return StructuredGeminiTransport(
        request=request,
        max_transport_attempts=max_transport_attempts,
        sleeper=sleeper,
        delay_calculator=delay_calculator,
    )
