from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import av_analyzer
import llm_transport
from groq_vision import GROQ_VISION_MODEL, GroqVisionAnalyzer, sample_window_timestamps


def _canonical_av(start_ms: int = 0, end_ms: int = 1_000) -> dict[str, object]:
    return {
        "windowStartMs": start_ms,
        "windowEndMs": end_ms,
        "scene": {"summary": "room", "tone": "neutral", "register": "colloquial"},
        "utterances": [{
            "startMs": start_ms + 100,
            "endMs": min(start_ms + 900, end_ms),
            "spokenZh": "所以",
            "subtitleZh": "所以",
            "speakerRef": "speaker-1",
            "intent": "statement",
            "emotion": "neutral",
            "confidence": "HIGH",
        }],
        "visualOnly": [],
    }


class _FailingGemini:
    last_call = {"transportAttempts": 1, "finalTransportStatus": "HTTP_429", "httpStatus": 429}

    def call_raw(self, _payload):
        raise llm_transport.ProviderError("HTTP_429", transient=True, http_status=429)


class _NoReadyGemini:
    last_call = {"transportAttempts": 1, "finalTransportStatus": "NO_READY_GEMINI_KEY", "httpStatus": None}

    def call_raw(self, _payload):
        raise llm_transport.ProviderError("NO_READY_GEMINI_KEY")


class _GroqFake:
    def __init__(self, response: object):
        self.response = response
        self.calls: list[dict[str, object]] = []
        self.last_call = {"transportAttempts": 1, "httpStatus": 200, "finalTransportStatus": "HTTP_200"}

    def call_raw(self, payload):
        self.calls.append(dict(payload))
        return self.response


class GroqAvFallbackTests(unittest.TestCase):
    def _media(self, directory: str) -> Path:
        path = Path(directory) / "source.mp4"
        path.write_bytes(b"synthetic media")
        return path

    def _frames(self, _path: Path, timestamp_ms: int) -> bytes:
        return b"\xff\xd8jpeg-" + str(timestamp_ms).encode("ascii")

    def test_gemini_success_does_not_call_groq(self) -> None:
        groq = _GroqFake(_canonical_av())

        class _SuccessGemini:
            last_call = {"transportAttempts": 1, "httpStatus": 200, "finalTransportStatus": "HTTP_200"}

            def call_raw(self, _payload):
                return _canonical_av()

        analyzer = av_analyzer.GeminiAVAnalyzer(
            transport=_SuccessGemini(),
            groq_transport=groq,
            groq_frame_extractor=self._frames,
        )
        with tempfile.TemporaryDirectory() as directory:
            result = analyzer.analyze_media(self._media(directory), start_ms=0, end_ms=1_000)
        self.assertEqual(_canonical_av(), result)
        self.assertEqual([], groq.calls)

    def test_gemini_429_invokes_groq_fallback(self) -> None:
        groq = _GroqFake(_canonical_av())
        analyzer = av_analyzer.GeminiAVAnalyzer(
            transport=_FailingGemini(),
            groq_transport=groq,
            groq_frame_extractor=self._frames,
        )
        with tempfile.TemporaryDirectory() as directory:
            result = analyzer.analyze_media(self._media(directory), start_ms=0, end_ms=1_000)
        self.assertEqual(_canonical_av(), result)
        self.assertEqual(1, len(groq.calls))
        self.assertTrue(analyzer.last_recovery["fallbackInvoked"])
        self.assertEqual("groq", analyzer.last_recovery["fallbackProvider"])

    def test_no_ready_gemini_invokes_groq_fallback(self) -> None:
        groq = _GroqFake(_canonical_av())
        analyzer = av_analyzer.GeminiAVAnalyzer(
            transport=_NoReadyGemini(),
            groq_transport=groq,
            groq_frame_extractor=self._frames,
        )
        with tempfile.TemporaryDirectory() as directory:
            analyzer.analyze_media(self._media(directory), start_ms=0, end_ms=1_000)
        self.assertEqual(1, len(groq.calls))

    def test_subdivision_accounting_records_groq_as_the_validated_av_provider(self) -> None:
        groq = _GroqFake(_canonical_av())
        analyzer = av_analyzer.GeminiAVAnalyzer(
            transport=_NoReadyGemini(),
            groq_transport=groq,
            groq_frame_extractor=self._frames,
        )
        with tempfile.TemporaryDirectory() as directory:
            result = av_analyzer.analyze_media_windows_with_subdivision(
                analyzer,
                self._media(directory),
                1_000,
                diagnostics_path=Path(directory) / "subdivision.json",
            )
            report = json.loads((Path(directory) / "subdivision.json").read_text(encoding="utf-8"))
        self.assertEqual(2, result["providerCalls"])
        self.assertEqual("AV_PRIMARY_SCHEMA_VALID_GROQ_FALLBACK", result["diagnostics"][0]["status"])
        self.assertTrue(report["windows"][0]["fallbackInvoked"])

    def test_groq_payload_has_two_bounded_deterministic_frames_and_compact_budget(self) -> None:
        groq = _GroqFake(_canonical_av())
        analyzer = GroqVisionAnalyzer(transport=groq, frame_extractor=self._frames)
        with tempfile.TemporaryDirectory() as directory:
            analyzer.analyze_media(self._media(directory), start_ms=10_000, end_ms=20_000)
        payload = groq.calls[0]
        content = payload["messages"][0]["content"]
        images = [part for part in content if part.get("type") == "image_url"]
        self.assertEqual(2, len(images))
        timestamps = [int(part["text"].split("=")[-1]) for part in content if part.get("type") == "text" and "Frame timestampMs=" in part.get("text", "")]
        self.assertEqual(sample_window_timestamps(10_000, 20_000), timestamps)
        self.assertTrue(all(10_000 < timestamp < 20_000 for timestamp in timestamps))
        self.assertEqual(GROQ_VISION_MODEL, payload["model"])
        self.assertEqual("none", payload["reasoning_effort"])
        self.assertEqual(1024, payload["max_tokens"])

    def test_groq_payload_contains_overlapping_transcript_and_ocr_evidence(self) -> None:
        groq = _GroqFake(_canonical_av())
        analyzer = GroqVisionAnalyzer(transport=groq, frame_extractor=self._frames)
        cues = [{
            "position": 7, "startMs": 200, "endMs": 800,
            "asrTextZh": "这样做的目的是避免以后他们把这些坏的东西赖在你身上",
            "ocrTextZh": "所以",
        }, {"position": 8, "startMs": 2_000, "endMs": 3_000, "asrTextZh": "outside"}]
        with tempfile.TemporaryDirectory() as directory:
            analyzer.analyze_media(self._media(directory), start_ms=0, end_ms=1_000, cues=cues)
        prompt = groq.calls[0]["messages"][0]["content"][0]["text"]
        self.assertIn("这样做的目的是避免以后他们把这些坏的东西赖在你身上", prompt)
        self.assertIn("所以", prompt)
        self.assertNotIn("outside", prompt)

    def test_valid_groq_json_passes_the_existing_canonical_validator(self) -> None:
        groq = _GroqFake(json.dumps(_canonical_av(), ensure_ascii=False))
        analyzer = GroqVisionAnalyzer(transport=groq, frame_extractor=self._frames)
        with tempfile.TemporaryDirectory() as directory:
            result = analyzer.analyze_media(self._media(directory), start_ms=0, end_ms=1_000)
        self.assertEqual(_canonical_av(), result)
        self.assertEqual("VALID", analyzer.last_recovery["schemaValidationStatus"])

    def test_groq_far_outside_timestamps_fail_closed(self) -> None:
        invalid = _canonical_av()
        invalid["utterances"][0]["startMs"] = -10
        invalid["utterances"][0]["endMs"] = 900
        groq = _GroqFake(invalid)
        analyzer = GroqVisionAnalyzer(transport=groq, frame_extractor=self._frames)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(av_analyzer.AVResponseRecoveryError):
                analyzer.analyze_media(self._media(directory), start_ms=0, end_ms=1_000)

    def test_groq_relative_coordinates_use_the_existing_child_window_normalization(self) -> None:
        groq = _GroqFake(_canonical_av())
        analyzer = GroqVisionAnalyzer(transport=groq, frame_extractor=self._frames)
        with tempfile.TemporaryDirectory() as directory:
            result = analyzer.analyze_media(self._media(directory), start_ms=5_000, end_ms=6_000)
        self.assertEqual((5_000, 6_000), (result["windowStartMs"], result["windowEndMs"]))
        self.assertTrue(analyzer.last_recovery["coordinateNormalizationApplied"])
        self.assertEqual("AV_SCHEMA_VALID_GROQ_FALLBACK", analyzer.last_recovery["finalStatus"])

    def test_groq_edge_utterance_uses_the_existing_bounded_boundary_normalization(self) -> None:
        boundary = _canonical_av(5_000, 6_000)
        boundary["utterances"][0]["startMs"] = 4_900
        groq = _GroqFake(boundary)
        analyzer = GroqVisionAnalyzer(transport=groq, frame_extractor=self._frames)
        with tempfile.TemporaryDirectory() as directory:
            result = analyzer.analyze_media(self._media(directory), start_ms=5_000, end_ms=6_000)
        self.assertEqual(4_900, result["windowStartMs"])
        self.assertTrue(analyzer.last_recovery["boundaryNormalizationApplied"])

    def test_groq_malformed_json_fails_closed(self) -> None:
        analyzer = GroqVisionAnalyzer(transport=_GroqFake("not json"), frame_extractor=self._frames)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(av_analyzer.AVResponseRecoveryError):
                analyzer.analyze_media(self._media(directory), start_ms=0, end_ms=1_000)

    def test_internal_gemini_schema_failure_does_not_mask_with_groq(self) -> None:
        invalid = _canonical_av()
        invalid["scene"] = "bad-shape"
        groq = _GroqFake(_canonical_av())

        class _InvalidGemini:
            last_call = {"transportAttempts": 1, "httpStatus": 200, "finalTransportStatus": "HTTP_200"}

            def __init__(self):
                self.calls = 0

            def call_raw(self, _payload):
                self.calls += 1
                return invalid

        gemini = _InvalidGemini()
        analyzer = av_analyzer.GeminiAVAnalyzer(
            transport=gemini,
            groq_transport=groq,
            groq_frame_extractor=self._frames,
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(av_analyzer.AVResponseRecoveryError):
                analyzer.analyze_media(self._media(directory), start_ms=0, end_ms=1_000)
        self.assertEqual(0, len(groq.calls))
        self.assertEqual(2, gemini.calls)

    def test_missing_groq_key_keeps_gemini_provider_error_explicit(self) -> None:
        analyzer = av_analyzer.GeminiAVAnalyzer(transport=_NoReadyGemini())
        with patch.dict(os.environ, {}, clear=True), tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(llm_transport.ProviderError, "NO_READY_GEMINI_KEY"):
                analyzer.analyze_media(self._media(directory), start_ms=0, end_ms=1_000)

    def test_secret_is_not_serialized_in_groq_runtime_payload_or_recovery(self) -> None:
        secret = "GROQ_SECRET_TEST_VALUE"
        groq = _GroqFake(_canonical_av())
        analyzer = av_analyzer.GeminiAVAnalyzer(
            transport=_FailingGemini(),
            groq_transport=groq,
            groq_frame_extractor=self._frames,
        )
        with tempfile.TemporaryDirectory() as directory:
            recovery = Path(directory) / "recovery.json"
            with patch.dict(os.environ, {"GROQ_API_KEY": secret}, clear=False):
                analyzer.analyze_media(self._media(directory), start_ms=0, end_ms=1_000, recovery_metadata_path=recovery)
            serialized = json.dumps(groq.calls) + recovery.read_text(encoding="utf-8")
        self.assertNotIn(secret, serialized)

    def test_live_groq_transport_requires_key_without_network(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch.object(llm_transport.requests, "post") as post:
            transport = llm_transport.groq_json_transport(model=GROQ_VISION_MODEL)
            with self.assertRaisesRegex(llm_transport.ProviderError, "GROQ_API_KEY_MISSING"):
                transport.call_raw({"messages": []})
        post.assert_not_called()

    def test_live_groq_transport_parses_openai_compatible_json_without_logging_key(self) -> None:
        secret = "GROQ_SECRET_TEST_VALUE"
        response = type(
            "Response",
            (),
            {
                "status_code": 200,
                "headers": {},
                "json": lambda self: {"choices": [{"message": {"content": json.dumps(_canonical_av())}}]},
            },
        )()
        with patch.dict(os.environ, {"GROQ_API_KEY": secret}, clear=True), patch.object(
            llm_transport.requests, "post", return_value=response
        ) as post:
            result = llm_transport.groq_json_transport(model=GROQ_VISION_MODEL).call_raw({"messages": []})
        self.assertEqual(_canonical_av(), result)
        self.assertEqual("Bearer " + secret, post.call_args.kwargs["headers"]["Authorization"])

    def test_live_groq_transport_captures_safe_429_rate_limit_metadata(self) -> None:
        secret = "GROQ_SECRET_TEST_VALUE"
        response = type(
            "Response",
            (),
            {
                "status_code": 429,
                "headers": {
                    "Retry-After": "2",
                    "x-ratelimit-limit-requests": "10",
                    "x-ratelimit-limit-tokens": "1000",
                    "x-ratelimit-remaining-requests": "9",
                    "x-ratelimit-remaining-tokens": "0",
                    "x-ratelimit-reset-requests": "1m",
                    "x-ratelimit-reset-tokens": "2s",
                },
                "json": lambda self: {},
            },
        )()
        with patch.dict(os.environ, {"GROQ_API_KEY": secret}, clear=True), patch.object(
            llm_transport.requests, "post", return_value=response
        ):
            transport = llm_transport.groq_json_transport(model=GROQ_VISION_MODEL)
            with self.assertRaises(llm_transport.ProviderError) as raised:
                transport.call_raw({"messages": []})
        error = raised.exception
        self.assertEqual("TPM_LIMIT", error.rate_limit_classification)
        self.assertEqual("0", error.rate_limit_metadata["x-ratelimit-remaining-tokens"])
        self.assertEqual("2", error.rate_limit_metadata["retry-after"])
        self.assertNotIn(secret, json.dumps(error.rate_limit_metadata))


if __name__ == "__main__":
    unittest.main()
