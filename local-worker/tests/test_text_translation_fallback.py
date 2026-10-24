from __future__ import annotations

import json
import unittest
from unittest.mock import Mock

from llm_transport import ProviderError, StructuredResponseError

try:
    from text_translation_fallback import (
        GROQ_TEXT_MODEL,
        MAX_GROQ_TEXT_BATCH,
        MAX_GROQ_TEXT_COMPLETION_TOKENS,
        GeminiPrimaryWithGroqTextFallback,
        GroqTextTranslationAdapter,
        groq_text_payload,
        groq_text_transport,
    )
except ImportError:
    GROQ_TEXT_MODEL = None
    MAX_GROQ_TEXT_BATCH = None
    MAX_GROQ_TEXT_COMPLETION_TOKENS = None
    GeminiPrimaryWithGroqTextFallback = None
    GroqTextTranslationAdapter = None
    groq_text_payload = None
    groq_text_transport = None


def _records(count: int = 5) -> list[dict[str, object]]:
    return [
        {"utteranceId": f"u{index}", "resolvedZh": f"中文第{index}句", "source_confidence": "high", "kind": "dialogue"}
        for index in range(count)
    ]


def _response(records: list[dict[str, object]], key: str = "semanticVi") -> dict[str, object]:
    return {"translations": [{"utteranceId": str(item["utteranceId"]), key: f"Câu tiếng Việt {item['utteranceId']}"} for item in records]}


class _Transport:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def call_raw(self, payload: dict[str, object]) -> object:
        self.calls.append(payload)
        return self.responses.pop(0)


class TextTranslationFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        if GroqTextTranslationAdapter is None:
            self.fail("text translation fallback module is not implemented")

    def test_gemini_success_remains_primary_and_does_not_call_groq(self) -> None:
        gemini = Mock(return_value={"translations": [{"utteranceId": "u0", "semanticVi": "Xin chào"}]})
        groq = Mock()
        request = GeminiPrimaryWithGroqTextFallback(gemini, groq)
        result = request({"mode": "semantic", "utterances": _records(1)})
        self.assertEqual("Xin chào", result["translations"][0]["semanticVi"])
        gemini.assert_called_once()
        groq.assert_not_called()

    def test_gemini_quota_falls_back_to_sequential_bounded_groq_batches(self) -> None:
        gemini = Mock(side_effect=ProviderError("HTTP_429", transient=True, http_status=429))
        records = _records(5)
        transport = _Transport([_response(records[:2]), _response(records[2:4]), _response(records[4:])])
        groq = GroqTextTranslationAdapter(transport=transport, batch_size=2)
        result = GeminiPrimaryWithGroqTextFallback(gemini, groq)({"mode": "semantic", "utterances": records})
        self.assertEqual([f"u{i}" for i in range(5)], [item["utteranceId"] for item in result["translations"]])
        self.assertEqual(3, len(transport.calls))
        self.assertEqual([2, 2, 1], [len(json.loads(call["messages"][1]["content"])["utterances"]) for call in transport.calls])
        for call in transport.calls:
            encoded = json.dumps(call, ensure_ascii=False).lower()
            self.assertNotIn("image", encoded)
            self.assertNotIn("video", encoded)
            self.assertEqual(GROQ_TEXT_MODEL, call["model"])
            self.assertEqual("none", call["reasoning_effort"])
            self.assertLessEqual(int(call["max_completion_tokens"]), 768)

    def test_gemini_schema_failure_does_not_silently_fallback(self) -> None:
        failure = ValueError("CP2_RESPONSE_SCHEMA_INVALID")
        gemini = Mock(side_effect=failure)
        groq = Mock()
        with self.assertRaisesRegex(ValueError, "CP2_RESPONSE_SCHEMA_INVALID"):
            GeminiPrimaryWithGroqTextFallback(gemini, groq)({"mode": "semantic", "utterances": _records(1)})
        groq.assert_not_called()

    def test_gemini_cp2_schema_failure_falls_back_to_groq_when_authorized(self) -> None:
        records = _records(1)
        gemini = Mock(side_effect=StructuredResponseError("CP2_RESPONSE_SCHEMA_INVALID"))
        groq = Mock(return_value=_response(records))
        adapter = GeminiPrimaryWithGroqTextFallback(gemini, groq)

        result = adapter(
            {"mode": "semantic", "utterances": records}
        )

        self.assertEqual("groq_text", adapter.last_provider)
        self.assertEqual("Câu tiếng Việt u0", result["translations"][0]["semanticVi"])
        groq.assert_called_once()

    def test_groq_rejects_unknown_duplicate_and_missing_ids(self) -> None:
        records = _records(2)
        for document in (
            {"translations": [{"utteranceId": "unknown", "semanticVi": "Câu"}, {"utteranceId": "u1", "semanticVi": "Câu"}]},
            {"translations": [{"utteranceId": "u0", "semanticVi": "Câu"}, {"utteranceId": "u0", "semanticVi": "Câu khác"}]},
            {"translations": [{"utteranceId": "u0", "semanticVi": "Câu"}]},
        ):
            with self.subTest(document=document):
                adapter = GroqTextTranslationAdapter(transport=_Transport([document]), batch_size=MAX_GROQ_TEXT_BATCH)
                with self.assertRaises(ValueError):
                    adapter({"mode": "semantic", "utterances": records})

    def test_groq_payload_is_text_only_and_batch_bounded(self) -> None:
        payload = groq_text_payload(_records(2), operation="native")
        self.assertEqual(GROQ_TEXT_MODEL, payload["model"])
        self.assertEqual("none", payload["reasoning_effort"])
        self.assertLessEqual(int(payload["max_completion_tokens"]), MAX_GROQ_TEXT_COMPLETION_TOKENS)
        self.assertEqual({"utteranceId", "sourceText"}, set(json.loads(payload["messages"][1]["content"])["utterances"][0]))
        self.assertNotIn("image_url", json.dumps(payload))
        self.assertNotIn("video", json.dumps(payload).lower())

    def test_groq_transport_retries_one_429_using_retry_after(self) -> None:
        calls = []
        sleeps = []

        def request(_payload):
            calls.append(True)
            if len(calls) == 1:
                raise ProviderError("HTTP_429", transient=True, http_status=429, retry_after=2.5)
            return _response(_records(1))

        transport = groq_text_transport(request=request, sleeper=sleeps.append)
        result = GroqTextTranslationAdapter(transport=transport, batch_size=1)({"mode": "semantic", "utterances": _records(1)})
        self.assertEqual("Câu tiếng Việt u0", result["translations"][0]["semanticVi"])
        self.assertEqual(2, len(calls))
        self.assertEqual([2.5], sleeps)


if __name__ == "__main__":
    unittest.main()
