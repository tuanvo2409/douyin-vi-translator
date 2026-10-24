from __future__ import annotations

import json
import os
from copy import deepcopy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import av_analyzer
import llm_transport


def _canonical_av(*, utterances=None, visual_only=None):
    return {
        "windowStartMs": 0,
        "windowEndMs": 1_000,
        "scene": {"summary": "blind test", "tone": "neutral", "register": "colloquial"},
        "utterances": list(utterances or [_utterance("原文")]),
        "visualOnly": list(visual_only or []),
    }


def _utterance(text: str) -> dict[str, object]:
    return {
        "startMs": 100,
        "endMs": 900,
        "spokenZh": text,
        "subtitleZh": text,
        "speakerRef": "speaker-1",
        "intent": "statement",
        "emotion": "neutral",
        "confidence": "HIGH",
    }


def _window_av(start_ms: int, end_ms: int, text: str = "原文") -> dict[str, object]:
    return {
        "windowStartMs": start_ms,
        "windowEndMs": end_ms,
        "scene": {"summary": "blind test", "tone": "neutral", "register": "colloquial"},
        "utterances": [{
            "startMs": start_ms + 100,
            "endMs": min(start_ms + 900, end_ms),
            "spokenZh": text,
            "subtitleZh": text,
            "speakerRef": "speaker-1",
            "intent": "statement",
            "emotion": "neutral",
            "confidence": "HIGH",
        }],
        "visualOnly": [],
    }


def _bounded_test_clip(_path: Path, start_ms: int, end_ms: int) -> bytes:
    return f"bounded:{start_ms}:{end_ms}".encode("ascii")


class AvResponseRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._ambient_groq_key = os.environ.pop("GROQ_API_KEY", None)

    def tearDown(self) -> None:
        if self._ambient_groq_key is not None:
            os.environ["GROQ_API_KEY"] = self._ambient_groq_key

    def _analyzer(self, responses):
        calls = []
        remaining = list(responses)

        def request(payload):
            calls.append(payload)
            value = remaining.pop(0)
            if isinstance(value, BaseException):
                raise value
            return value

        transport = llm_transport.StructuredGeminiTransport(request=request)
        return av_analyzer.GeminiAVAnalyzer(
            transport=transport, model="test-model", allow_cloud_upload=True, clip_extractor=_bounded_test_clip,
        ), calls

    def _media(self, directory: str) -> Path:
        path = Path(directory) / "source.mp4"
        path.write_bytes(b"synthetic media")
        return path

    def test_valid_canonical_response_passes_first_attempt(self) -> None:
        expected = _canonical_av()
        analyzer, calls = self._analyzer([expected])
        with tempfile.TemporaryDirectory() as directory:
            result = analyzer.analyze_media(self._media(directory), start_ms=0, end_ms=1_000)
        self.assertEqual(expected, result)
        self.assertEqual(1, len(calls))
        self.assertEqual("AV_SCHEMA_VALID_FIRST_PASS", analyzer.last_recovery["finalStatus"])
        self.assertEqual(1, analyzer.last_recovery["attemptCount"])

    def test_fenced_json_is_normalized_without_changing_semantic_values(self) -> None:
        expected = _canonical_av()
        fenced = "```json\n" + json.dumps(expected, ensure_ascii=False) + "\n```"
        analyzer, calls = self._analyzer([fenced])
        with tempfile.TemporaryDirectory() as directory:
            result = analyzer.analyze_media(self._media(directory), start_ms=0, end_ms=1_000)
        self.assertEqual(expected, result)
        self.assertEqual(1, len(calls))
        self.assertTrue(analyzer.last_recovery["normalizationApplied"])
        self.assertEqual("AV_SCHEMA_VALID_AFTER_NORMALIZATION", analyzer.last_recovery["finalStatus"])

    def test_single_structural_envelope_is_normalized(self) -> None:
        expected = _canonical_av()
        analyzer, _ = self._analyzer([{ "data": expected }])
        with tempfile.TemporaryDirectory() as directory:
            result = analyzer.analyze_media(self._media(directory), start_ms=0, end_ms=1_000)
        self.assertEqual(expected, result)

    def test_missing_semantic_field_is_not_invented(self) -> None:
        invalid = _canonical_av()
        del invalid["utterances"][0]["spokenZh"]
        analyzer, calls = self._analyzer([invalid, invalid])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(av_analyzer.AVResponseRecoveryError):
                analyzer.analyze_media(self._media(directory), start_ms=0, end_ms=1_000)
        self.assertEqual(2, len(calls))

    def test_wrong_semantic_type_is_not_silently_coerced(self) -> None:
        invalid = _canonical_av()
        invalid["scene"] = "not-an-object"
        analyzer, calls = self._analyzer([invalid, invalid])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(av_analyzer.AVResponseRecoveryError):
                analyzer.analyze_media(self._media(directory), start_ms=0, end_ms=1_000)
        self.assertEqual(2, len(calls))
        self.assertEqual("AV_SCHEMA_INVALID_FINAL", analyzer.last_recovery["finalStatus"])

    def test_invalid_first_response_uses_one_schema_enforced_repair_retry(self) -> None:
        invalid = _canonical_av()
        invalid["scene"] = "bad-shape"
        expected = _canonical_av()
        analyzer, calls = self._analyzer([invalid, expected])
        with tempfile.TemporaryDirectory() as directory:
            result = analyzer.analyze_media(self._media(directory), start_ms=0, end_ms=1_000)
        self.assertEqual(expected, result)
        self.assertEqual(2, len(calls))
        self.assertIn("responseSchema", calls[1]["generationConfig"])
        self.assertEqual(av_analyzer.AV_RESPONSE_SCHEMA, calls[1]["generationConfig"]["responseSchema"])
        self.assertEqual("PASS", av_analyzer.validate_av_schema_parity(calls[1]["generationConfig"]["responseSchema"])["status"])
        self.assertIn("previous", calls[1]["contents"][0]["parts"][0]["text"].lower())
        self.assertEqual("AV_SCHEMA_VALID_AFTER_RETRY", analyzer.last_recovery["finalStatus"])

    def test_each_attempt_declares_the_same_requested_window(self) -> None:
        invalid = _window_av(125, 875)
        invalid["scene"] = "bad-shape"
        analyzer, calls = self._analyzer([invalid, _window_av(125, 875)])
        with tempfile.TemporaryDirectory() as directory:
            analyzer.analyze_media(self._media(directory), start_ms=125, end_ms=875)
        for call in calls:
            self.assertIn("source window 125-875 ms", call["contents"][0]["parts"][0]["text"])

    def test_transient_provider_error_uses_the_remaining_bounded_attempt(self) -> None:
        expected = _canonical_av()
        analyzer, calls = self._analyzer([llm_transport.ProviderError("HTTP_429", transient=True), expected])
        with tempfile.TemporaryDirectory() as directory:
            result = analyzer.analyze_media(self._media(directory), start_ms=0, end_ms=1_000)
        self.assertEqual(expected, result)
        self.assertEqual(2, len(calls))
        self.assertFalse(analyzer.last_recovery["repairRetryUsed"])

    def test_invalid_repair_response_fails_closed_and_never_exceeds_two_attempts(self) -> None:
        invalid = _canonical_av()
        invalid["scene"] = "bad-shape"
        analyzer, calls = self._analyzer([invalid, invalid, _canonical_av()])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(av_analyzer.AVResponseRecoveryError):
                analyzer.analyze_media(self._media(directory), start_ms=0, end_ms=1_000)
        self.assertEqual(2, len(calls))
        self.assertEqual(2, analyzer.last_recovery["attemptCount"])

    def test_recovery_metadata_is_bounded_and_does_not_store_provider_content(self) -> None:
        invalid = _canonical_av()
        invalid["providerKey"] = "FAKE_PROVIDER_SECRET_SUFFIX"
        analyzer, _ = self._analyzer([invalid, invalid])
        with tempfile.TemporaryDirectory() as directory:
            diagnostics = Path(directory) / "recovery.json"
            with self.assertRaises(av_analyzer.AVResponseRecoveryError):
                analyzer.analyze_media(
                    self._media(directory),
                    start_ms=0,
                    end_ms=1_000,
                    recovery_metadata_path=diagnostics,
                )
            saved = diagnostics.read_text(encoding="utf-8")
        self.assertNotIn("FAKE_PROVIDER_SECRET_SUFFIX", saved)
        self.assertLessEqual(len(saved.encode("utf-8")), av_analyzer.MAX_DIAGNOSTIC_BYTES)

    def test_canonical_schema_contract_remains_strict(self) -> None:
        document = _canonical_av()
        self.assertEqual(document, av_analyzer.validate_av_analysis(document))
        with self.assertRaises(ValueError):
            av_analyzer.validate_av_analysis({**document, "unsupported": True})

    def test_real_page_a_boundary_utterance_is_adapted_without_timing_mutation(self) -> None:
        boundary = _window_av(25_000, 40_000)
        boundary["utterances"][0]["startMs"] = 24_900
        analyzer, _ = self._analyzer([boundary, boundary])
        with tempfile.TemporaryDirectory() as directory:
            result = analyzer.analyze_media(self._media(directory), start_ms=25_000, end_ms=40_000)
        self.assertEqual(24_900, result["utterances"][0]["startMs"])
        self.assertEqual(24_900, result["windowStartMs"])
        self.assertTrue(analyzer.last_recovery["boundaryNormalizationApplied"])

    def test_far_outside_nested_utterance_still_fails_closed(self) -> None:
        invalid = _window_av(25_000, 40_000)
        invalid["utterances"][0].update(startMs=19_000, endMs=20_000)
        analyzer, _ = self._analyzer([invalid, invalid])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(av_analyzer.AVResponseRecoveryError) as raised:
                analyzer.analyze_media(self._media(directory), start_ms=25_000, end_ms=40_000)
        self.assertEqual("NESTED_UTTERANCE_SCHEMA_MISMATCH", raised.exception.failure_class)

    def test_boundary_utterance_survives_subdivision_with_canonical_identity(self) -> None:
        invalid_parent = _window_av(0, 30_000)
        invalid_parent["scene"] = "bad-shape"
        first = _window_av(0, 15_000, "第一句")
        second = _window_av(12_000, 27_000, "跨边界")
        second["utterances"][0].update(startMs=11_900, endMs=13_000)
        third = _window_av(24_000, 30_000, "第三句")
        analyzer, _ = self._analyzer([invalid_parent, invalid_parent, first, second, third])
        with tempfile.TemporaryDirectory() as directory:
            result = av_analyzer.analyze_media_windows_with_subdivision(
                analyzer,
                self._media(directory),
                30_000,
            )
        av_utterances = result["av_analysis"]["utterances"]
        self.assertIn(11_900, [item["startMs"] for item in av_utterances])
        dialogue = result["reconstructed"]["dialogue"]
        identifiers = [item["utteranceId"] for item in dialogue]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        boundary = next(item for item in dialogue if item["startMs"] == 11_900)
        self.assertTrue(boundary["utteranceId"].startswith("utt_v1_d_00011900_"))


class AvSchemaParityTests(unittest.TestCase):
    def _schema(self) -> dict[str, object]:
        return deepcopy(av_analyzer.AV_RESPONSE_SCHEMA)

    def test_every_canonical_required_field_is_provider_required(self) -> None:
        report = av_analyzer.validate_av_schema_parity(self._schema())
        self.assertEqual("PASS", report["status"])
        self.assertEqual(20, report["canonicalRequiredFieldCount"])

    def test_every_canonical_nested_field_is_provider_required(self) -> None:
        schema = self._schema()
        self.assertEqual(
            ["summary", "tone", "register"],
            schema["properties"]["scene"]["required"],
        )
        self.assertEqual(
            [
                "startMs", "endMs", "spokenZh", "subtitleZh", "speakerRef",
                "intent", "emotion", "confidence",
            ],
            schema["properties"]["utterances"]["items"]["required"],
        )
        self.assertEqual(
            ["startMs", "endMs", "textZh", "confidence"],
            schema["properties"]["visualOnly"]["items"]["required"],
        )

    def test_utterance_timing_fields_are_nested_required(self) -> None:
        report = av_analyzer.validate_av_schema_parity(self._schema())
        self.assertIn("utterances[].startMs", report["canonicalRequiredPaths"])
        self.assertIn("utterances[].endMs", report["canonicalRequiredPaths"])
        self.assertNotIn("utterances[].startMs", report["missingPaths"])
        self.assertNotIn("utterances[].endMs", report["missingPaths"])

    def test_canonical_object_types_match_provider_schema(self) -> None:
        report = av_analyzer.validate_av_schema_parity(self._schema())
        self.assertEqual([], report["violations"])

    def test_scene_shape_matches_canonical_scene(self) -> None:
        report = av_analyzer.validate_av_schema_parity(self._schema())
        self.assertEqual("PASS", report["status"])
        self.assertEqual([], [item for item in report["violations"] if item["path"].startswith("scene")])

    def test_array_items_preserve_nested_requirements(self) -> None:
        report = av_analyzer.validate_av_schema_parity(self._schema())
        self.assertEqual("PASS", report["status"])
        self.assertNotIn("utterances[].confidence", report["missingPaths"])
        self.assertNotIn("visualOnly[].confidence", report["missingPaths"])

    def test_weakened_root_required_schema_fails_closed(self) -> None:
        schema = self._schema()
        schema["required"].remove("scene")
        report = av_analyzer.validate_av_schema_parity(schema)
        self.assertEqual("FAIL", report["status"])
        self.assertIn("CANONICAL_REQUIRED_MISSING_FROM_PROVIDER_SCHEMA", report["violationCodes"])

    def test_missing_nested_required_schema_fails_closed(self) -> None:
        schema = self._schema()
        schema["properties"]["utterances"]["items"]["required"].remove("startMs")
        report = av_analyzer.validate_av_schema_parity(schema)
        self.assertEqual("FAIL", report["status"])
        self.assertIn("NESTED_REQUIRED_MISSING", report["violationCodes"])
        self.assertIn("utterances[].startMs", report["missingPaths"])

    def test_nested_timing_type_mismatch_fails_closed(self) -> None:
        schema = self._schema()
        schema["properties"]["utterances"]["items"]["properties"]["startMs"]["type"] = "STRING"
        report = av_analyzer.validate_av_schema_parity(schema)
        self.assertEqual("FAIL", report["status"])
        self.assertIn("TYPE_MISMATCH", report["violationCodes"])

    def test_valid_schema_passes_without_provider_calls(self) -> None:
        calls = []

        def request(_payload):
            calls.append(True)
            raise AssertionError("schema parity must not call a provider")

        report = av_analyzer.validate_av_schema_parity(self._schema())
        self.assertEqual("PASS", report["status"])
        self.assertEqual([], calls)
        self.assertEqual(20, report["providerRequiredFieldCount"])

    def test_repair_retry_uses_same_hardened_canonical_equivalent_schema(self) -> None:
        analyzer = av_analyzer.GeminiAVAnalyzer(model="test-model")
        payload = analyzer._payload(b"synthetic", 0, 1_000, repair=True)
        schema = payload["generationConfig"]["responseSchema"]
        self.assertEqual(av_analyzer.AV_RESPONSE_SCHEMA, schema)
        self.assertEqual("PASS", av_analyzer.validate_av_schema_parity(schema)["status"])

    def test_transport_payload_preserves_nested_required(self) -> None:
        seen = []

        def request(payload):
            seen.append(payload)
            return _canonical_av()

        transport = llm_transport.StructuredGeminiTransport(request=request)
        schema = self._schema()
        transport.call_raw({"generationConfig": {"responseSchema": schema}})
        self.assertEqual(
            schema["properties"]["utterances"]["items"]["required"],
            seen[0]["generationConfig"]["responseSchema"]["properties"]["utterances"]["items"]["required"],
        )

    def test_unexpected_nullability_fails_closed(self) -> None:
        schema = self._schema()
        schema["properties"]["utterances"]["items"]["properties"]["spokenZh"]["nullable"] = False
        report = av_analyzer.validate_av_schema_parity(schema)
        self.assertEqual("FAIL", report["status"])
        self.assertIn("UNEXPECTED_NULLABILITY", report["violationCodes"])

    def test_enum_mismatch_fails_closed(self) -> None:
        schema = self._schema()
        schema["properties"]["visualOnly"]["items"]["properties"]["confidence"]["enum"] = ["HIGH"]
        report = av_analyzer.validate_av_schema_parity(schema)
        self.assertEqual("FAIL", report["status"])
        self.assertIn("ENUM_MISMATCH", report["violationCodes"])

    def test_array_container_shape_mismatch_fails_closed(self) -> None:
        schema = self._schema()
        del schema["properties"]["utterances"]["items"]
        report = av_analyzer.validate_av_schema_parity(schema)
        self.assertEqual("FAIL", report["status"])
        self.assertIn("CONTAINER_SHAPE_MISMATCH", report["violationCodes"])

    def test_production_logic_contains_no_fixture_specific_strings(self) -> None:
        for name in ("av_analyzer.py", "llm_transport.py"):
            source = (Path(__file__).parents[1] / name).read_text(encoding="utf-8")
            self.assertNotIn("BLIND_FIXTURE_02", source)
            self.assertNotIn("7654533698251586161", source)


class AvWindowSubdivisionTests(unittest.TestCase):
    def _analyzer(self, responses):
        calls = []
        remaining = list(responses)

        def request(payload):
            calls.append(payload)
            value = remaining.pop(0)
            if isinstance(value, BaseException):
                raise value
            return value

        transport = llm_transport.StructuredGeminiTransport(request=request)
        return av_analyzer.GeminiAVAnalyzer(
            transport=transport, model="test-model", allow_cloud_upload=True, clip_extractor=_bounded_test_clip,
        ), calls

    def _media(self, directory: str) -> Path:
        path = Path(directory) / "source.mp4"
        path.write_bytes(b"synthetic media")
        return path

    def test_valid_parent_window_never_subdivides(self) -> None:
        analyzer, calls = self._analyzer([_window_av(0, 30_000)])
        with tempfile.TemporaryDirectory() as directory:
            result = av_analyzer.analyze_media_windows_with_subdivision(
                analyzer, self._media(directory), 30_000
            )
        self.assertEqual(1, len(calls))
        self.assertEqual("AV_PRIMARY_SCHEMA_VALID_FIRST_PASS", result["diagnostics"][0]["status"])

    def test_parent_passing_bounded_repair_never_subdivides(self) -> None:
        invalid = _window_av(0, 30_000)
        invalid["scene"] = "bad-shape"
        analyzer, calls = self._analyzer([invalid, _window_av(0, 30_000)])
        with tempfile.TemporaryDirectory() as directory:
            result = av_analyzer.analyze_media_windows_with_subdivision(
                analyzer, self._media(directory), 30_000
            )
        self.assertEqual(2, len(calls))
        self.assertEqual("AV_PRIMARY_SCHEMA_VALID_AFTER_RETRY", result["diagnostics"][0]["status"])

    def test_parent_final_schema_failure_triggers_one_subdivision(self) -> None:
        invalid = _window_av(0, 30_000)
        invalid["scene"] = "bad-shape"
        analyzer, calls = self._analyzer([
            invalid, invalid,
            _window_av(0, 15_000), _window_av(12_000, 27_000), _window_av(24_000, 30_000),
        ])
        with tempfile.TemporaryDirectory() as directory:
            result = av_analyzer.analyze_media_windows_with_subdivision(
                analyzer, self._media(directory), 30_000
            )
        self.assertEqual(5, len(calls))
        self.assertEqual("AV_WINDOW_SUBDIVISION_SUCCEEDED", result["diagnostics"][0]["status"])

    def test_subdivision_policy_has_deterministic_full_coverage(self) -> None:
        self.assertEqual(
            [(0, 15_000), (12_000, 27_000), (24_000, 30_000)],
            av_analyzer.subdivision_windows(0, 30_000),
        )

    def test_child_windows_overlap_by_configured_amount(self) -> None:
        windows = av_analyzer.subdivision_windows(0, 30_000)
        self.assertEqual(3_000, windows[0][1] - windows[1][0])
        self.assertEqual(3_000, windows[1][1] - windows[2][0])

    def test_final_child_reaches_exact_parent_end(self) -> None:
        self.assertEqual(30_000, av_analyzer.subdivision_windows(0, 30_000)[-1][1])

    def test_no_child_has_zero_or_negative_duration(self) -> None:
        for start, end in av_analyzer.subdivision_windows(0, 30_000):
            self.assertGreater(end, start)

    def test_no_source_interval_inside_parent_is_uncovered(self) -> None:
        windows = av_analyzer.subdivision_windows(0, 30_000)
        self.assertEqual(0, windows[0][0])
        self.assertEqual(30_000, windows[-1][1])
        for previous, current in zip(windows, windows[1:]):
            self.assertLessEqual(current[0], previous[1])

    def test_subdivision_depth_never_exceeds_one(self) -> None:
        self.assertEqual([], av_analyzer.subdivision_windows(0, 30_000, depth=1))

    def test_each_child_still_has_at_most_two_provider_attempts(self) -> None:
        invalid_parent = _window_av(0, 30_000)
        invalid_parent["scene"] = "bad-shape"
        invalid_child = _window_av(0, 15_000)
        invalid_child["scene"] = "bad-shape"
        analyzer, calls = self._analyzer([
            invalid_parent, invalid_parent,
            invalid_child, _window_av(0, 15_000),
            _window_av(12_000, 27_000), _window_av(24_000, 30_000),
        ])
        with tempfile.TemporaryDirectory() as directory:
            result = av_analyzer.analyze_media_windows_with_subdivision(
                analyzer, self._media(directory), 30_000
            )
        self.assertEqual(6, len(calls))
        self.assertEqual(2, result["diagnostics"][0]["children"][0]["attemptCount"])
        self.assertTrue(all(item["attemptCount"] <= 2 for item in result["diagnostics"][0]["children"]))

    def test_failed_child_causes_fail_closed_parent_result(self) -> None:
        invalid_parent = _window_av(0, 30_000)
        invalid_parent["scene"] = "bad-shape"
        invalid_child = _window_av(0, 15_000)
        invalid_child["scene"] = "bad-shape"
        analyzer, calls = self._analyzer([invalid_parent, invalid_parent, invalid_child, invalid_child])
        reconciled = []
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(av_analyzer.AVResponseRecoveryError):
                av_analyzer.analyze_media_windows_with_subdivision(
                    analyzer,
                    self._media(directory),
                    30_000,
                    reconciliation=lambda *_args: reconciled.append(True),
                )
        self.assertEqual(4, len(calls))
        self.assertEqual([], reconciled)

    def test_all_valid_children_proceed_to_reconciliation(self) -> None:
        invalid = _window_av(0, 30_000)
        invalid["scene"] = "bad-shape"
        analyzer, _ = self._analyzer([
            invalid, invalid,
            _window_av(0, 15_000), _window_av(12_000, 27_000), _window_av(24_000, 30_000),
        ])
        seen = []
        with tempfile.TemporaryDirectory() as directory:
            av_analyzer.analyze_media_windows_with_subdivision(
                analyzer,
                self._media(directory),
                30_000,
                reconciliation=lambda _cues, document, _unresolved: seen.append(document) or document,
            )
        self.assertEqual(1, len(seen))
        self.assertEqual(3, len(seen[0]["utterances"]))

    def test_only_canonical_valid_child_documents_reach_reconciliation(self) -> None:
        invalid = _window_av(0, 30_000)
        invalid["scene"] = "bad-shape"
        invalid_child = _window_av(0, 15_000)
        invalid_child["scene"] = "bad-shape"
        analyzer, _ = self._analyzer([invalid, invalid, invalid_child, invalid_child])
        seen = []
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(av_analyzer.AVResponseRecoveryError):
                av_analyzer.analyze_media_windows_with_subdivision(
                    analyzer,
                    self._media(directory),
                    30_000,
                    reconciliation=lambda *_args: seen.append(True),
                )
        self.assertEqual([], seen)

    def test_child_relative_timestamps_are_converted_once(self) -> None:
        analyzer, _ = self._analyzer([_window_av(0, 15_000)])
        with tempfile.TemporaryDirectory() as directory:
            result = analyzer.analyze_media(self._media(directory), start_ms=12_000, end_ms=27_000)
        self.assertEqual(12_000, result["windowStartMs"])
        self.assertEqual(27_000, result["windowEndMs"])
        self.assertEqual(12_100, result["utterances"][0]["startMs"])
        self.assertTrue(analyzer.last_recovery["coordinateNormalizationApplied"])

    def test_already_absolute_timestamps_are_not_double_offset(self) -> None:
        analyzer, _ = self._analyzer([_window_av(12_000, 27_000)])
        with tempfile.TemporaryDirectory() as directory:
            result = analyzer.analyze_media(self._media(directory), start_ms=12_000, end_ms=27_000)
        self.assertEqual(12_100, result["utterances"][0]["startMs"])
        self.assertFalse(analyzer.last_recovery["coordinateNormalizationApplied"])

    def test_overlap_duplicates_are_reconciled_through_existing_reconciler(self) -> None:
        invalid = _window_av(0, 30_000)
        invalid["scene"] = "bad-shape"
        duplicate = _window_av(12_000, 27_000)
        duplicate["utterances"][0].update(startMs=13_000, endMs=14_000)
        first = _window_av(0, 15_000)
        first["utterances"][0].update(startMs=13_000, endMs=14_000)
        analyzer, _ = self._analyzer([invalid, invalid, first, duplicate, _window_av(24_000, 30_000)])
        with tempfile.TemporaryDirectory() as directory:
            result = av_analyzer.analyze_media_windows_with_subdivision(
                analyzer, self._media(directory), 30_000
            )
        duplicate_groups = [
            item for item in result["reconstructed"]["dialogue"] if item["startMs"] == 13_000
        ]
        self.assertEqual(1, len(duplicate_groups))
        self.assertNotIn("AV_CONFLICT", duplicate_groups[0]["flags"])

    def test_conflicting_overlap_evidence_remains_observable(self) -> None:
        invalid = _window_av(0, 30_000)
        invalid["scene"] = "bad-shape"
        first = _window_av(0, 15_000, "甲")
        first["utterances"][0].update(startMs=13_000, endMs=14_000)
        second = _window_av(12_000, 27_000, "乙")
        second["utterances"][0].update(startMs=13_000, endMs=14_000)
        analyzer, _ = self._analyzer([invalid, invalid, first, second, _window_av(24_000, 30_000)])
        with tempfile.TemporaryDirectory() as directory:
            result = av_analyzer.analyze_media_windows_with_subdivision(
                analyzer, self._media(directory), 30_000
            )
        self.assertIn("AV_CONFLICT", result["reconstructed"]["dialogue"][0]["flags"])

    def test_unsupported_provider_failure_does_not_trigger_subdivision(self) -> None:
        analyzer, calls = self._analyzer([llm_transport.ProviderError("HTTP_401")])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(llm_transport.ProviderError):
                av_analyzer.analyze_media_windows_with_subdivision(
                    analyzer, self._media(directory), 30_000
                )
        self.assertEqual(1, len(calls))

    def test_provider_call_count_is_deterministically_bounded(self) -> None:
        invalid = _window_av(0, 30_000)
        invalid["scene"] = "bad-shape"
        analyzer, calls = self._analyzer([
            invalid, invalid,
            _window_av(0, 15_000), _window_av(12_000, 27_000), _window_av(24_000, 30_000),
        ])
        with tempfile.TemporaryDirectory() as directory:
            result = av_analyzer.analyze_media_windows_with_subdivision(
                analyzer, self._media(directory), 30_000
            )
        self.assertEqual(5, len(calls))
        self.assertEqual(5, result["providerCalls"])

    def test_default_primary_window_behavior_remains_unchanged(self) -> None:
        self.assertEqual([(0, 30_000), (25_000, 36_107)], av_analyzer.analysis_windows(36_107))

    def test_subdivision_diagnostics_are_bounded_and_safe(self) -> None:
        invalid = _window_av(0, 30_000)
        invalid["scene"] = "bad-shape"
        analyzer, _ = self._analyzer([
            invalid, invalid,
            _window_av(0, 15_000), _window_av(12_000, 27_000), _window_av(24_000, 30_000),
        ])
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "subdivision.json"
            av_analyzer.analyze_media_windows_with_subdivision(
                analyzer, self._media(directory), 30_000, diagnostics_path=report_path
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report_size = report_path.stat().st_size
        self.assertEqual("AV_WINDOW_SUBDIVISION_SUCCEEDED", report["windows"][0]["status"])
        self.assertLessEqual(report_size, av_analyzer.MAX_DIAGNOSTIC_BYTES)
        self.assertNotIn("responseSchema", json.dumps(report))


class AvTransportRetryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._ambient_groq_key = os.environ.pop("GROQ_API_KEY", None)

    def tearDown(self) -> None:
        if self._ambient_groq_key is not None:
            os.environ["GROQ_API_KEY"] = self._ambient_groq_key

    def _analyzer(self, responses, sleeps=None, *, max_transport_attempts=3):
        calls = []
        remaining = list(responses)

        def request(payload):
            calls.append(payload)
            value = remaining.pop(0)
            if isinstance(value, BaseException):
                raise value
            return value

        transport = llm_transport.StructuredGeminiTransport(
            request=request,
            max_transport_attempts=max_transport_attempts,
            sleeper=(sleeps.append if sleeps is not None else (lambda _delay: None)),
        )
        return av_analyzer.GeminiAVAnalyzer(
            transport=transport, model="test-model", allow_cloud_upload=True, clip_extractor=_bounded_test_clip,
        ), calls

    def _media(self, directory: str) -> Path:
        path = Path(directory) / "source.mp4"
        path.write_bytes(b"synthetic media")
        return path

    def test_transient_503_then_200_is_one_logical_generation(self) -> None:
        sleeps = []
        analyzer, calls = self._analyzer([
            llm_transport.ProviderError("HTTP_503", transient=True),
            _canonical_av(),
        ], sleeps)
        with tempfile.TemporaryDirectory() as directory:
            analyzer.analyze_media(self._media(directory), start_ms=0, end_ms=1_000)
        self.assertEqual(2, len(calls))
        self.assertEqual(1, analyzer.last_recovery["attemptCount"])
        self.assertEqual(2, analyzer.last_recovery["transportAttempts"])
        self.assertTrue(analyzer.last_recovery["transientRetryUsed"])
        self.assertEqual([1.0], sleeps)

    def test_two_transient_503s_then_200_is_one_logical_generation(self) -> None:
        sleeps = []
        analyzer, calls = self._analyzer([
            llm_transport.ProviderError("HTTP_503", transient=True),
            llm_transport.ProviderError("HTTP_503", transient=True),
            _canonical_av(),
        ], sleeps)
        with tempfile.TemporaryDirectory() as directory:
            analyzer.analyze_media(self._media(directory), start_ms=0, end_ms=1_000)
        self.assertEqual(3, len(calls))
        self.assertEqual(1, analyzer.last_recovery["attemptCount"])
        self.assertEqual(3, analyzer.last_recovery["transportAttempts"])
        self.assertEqual([1.0, 2.0], sleeps)

    def test_three_transient_503s_are_provider_temporarily_unavailable(self) -> None:
        sleeps = []
        analyzer, calls = self._analyzer([
            llm_transport.ProviderError("HTTP_503", transient=True),
            llm_transport.ProviderError("HTTP_503", transient=True),
            llm_transport.ProviderError("HTTP_503", transient=True),
        ], sleeps)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(llm_transport.ProviderError) as raised:
                analyzer.analyze_media(self._media(directory), start_ms=0, end_ms=1_000)
        self.assertEqual("AV_PROVIDER_TEMPORARILY_UNAVAILABLE", analyzer.last_recovery["finalStatus"])
        self.assertEqual(1, analyzer.last_recovery["attemptCount"])
        self.assertEqual(3, analyzer.last_recovery["transportAttempts"])
        self.assertEqual("HTTP_503", raised.exception.classification)
        self.assertEqual(3, len(calls))

    def test_transport_exhaustion_does_not_invoke_logical_repair(self) -> None:
        analyzer, calls = self._analyzer([
            llm_transport.ProviderError("HTTP_503", transient=True),
            llm_transport.ProviderError("HTTP_503", transient=True),
            llm_transport.ProviderError("HTTP_503", transient=True),
        ])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(llm_transport.ProviderError):
                analyzer.analyze_media(self._media(directory), start_ms=0, end_ms=1_000)
        self.assertEqual(3, len(calls))
        self.assertTrue(all("responseSchema" not in call["generationConfig"] for call in calls))

    def test_transport_exhaustion_does_not_trigger_subdivision(self) -> None:
        analyzer, calls = self._analyzer([
            llm_transport.ProviderError("HTTP_503", transient=True),
            llm_transport.ProviderError("HTTP_503", transient=True),
            llm_transport.ProviderError("HTTP_503", transient=True),
        ])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(llm_transport.ProviderError):
                av_analyzer.analyze_media_windows_with_subdivision(
                    analyzer, self._media(directory), 30_000,
                )
        self.assertEqual(3, len(calls))

    def test_malformed_initial_consumes_one_logical_generation_before_repair(self) -> None:
        invalid = _canonical_av()
        invalid["scene"] = "bad-shape"
        analyzer, calls = self._analyzer([invalid, _canonical_av()])
        with tempfile.TemporaryDirectory() as directory:
            analyzer.analyze_media(self._media(directory), start_ms=0, end_ms=1_000)
        self.assertEqual(2, len(calls))
        self.assertEqual(2, analyzer.last_recovery["attemptCount"])
        self.assertEqual(1, analyzer.last_recovery["transportAttempts"])
        self.assertTrue(analyzer.last_recovery["repairGenerationUsed"])

    def test_repair_generation_has_its_own_transport_retry_budget(self) -> None:
        invalid = _canonical_av()
        invalid["scene"] = "bad-shape"
        sleeps = []
        analyzer, calls = self._analyzer([
            invalid,
            llm_transport.ProviderError("HTTP_503", transient=True),
            _canonical_av(),
        ], sleeps)
        with tempfile.TemporaryDirectory() as directory:
            analyzer.analyze_media(self._media(directory), start_ms=0, end_ms=1_000)
        self.assertEqual(3, len(calls))
        self.assertEqual(2, analyzer.last_recovery["attemptCount"])
        self.assertEqual(2, analyzer.last_recovery["transportAttempts"])
        self.assertEqual([1.0], sleeps)

    def test_malformed_initial_then_repair_transport_recovery_succeeds(self) -> None:
        invalid = _canonical_av()
        invalid["scene"] = "bad-shape"
        sleeps = []
        analyzer, calls = self._analyzer([
            invalid,
            llm_transport.ProviderError("HTTP_503", transient=True),
            llm_transport.ProviderError("HTTP_503", transient=True),
            _canonical_av(),
        ], sleeps)
        with tempfile.TemporaryDirectory() as directory:
            result = analyzer.analyze_media(self._media(directory), start_ms=0, end_ms=1_000)
        self.assertEqual("原文", result["utterances"][0]["spokenZh"])
        self.assertEqual(4, len(calls))
        self.assertEqual(2, analyzer.last_recovery["attemptCount"])
        self.assertEqual(3, analyzer.last_recovery["transportAttempts"])
        self.assertEqual([1.0, 2.0], sleeps)

    def test_malformed_initial_then_exhausted_repair_is_provider_unavailable(self) -> None:
        invalid = _canonical_av()
        invalid["scene"] = "bad-shape"
        analyzer, calls = self._analyzer([
            invalid,
            llm_transport.ProviderError("HTTP_503", transient=True),
            llm_transport.ProviderError("HTTP_503", transient=True),
            llm_transport.ProviderError("HTTP_503", transient=True),
        ])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(llm_transport.ProviderError):
                analyzer.analyze_media(self._media(directory), start_ms=0, end_ms=1_000)
        self.assertEqual(4, len(calls))
        self.assertEqual("AV_PROVIDER_TEMPORARILY_UNAVAILABLE", analyzer.last_recovery["finalStatus"])
        self.assertEqual(2, analyzer.last_recovery["attemptCount"])
        self.assertEqual(3, analyzer.last_recovery["transportAttempts"])
        self.assertEqual("NOT_RUN", analyzer.last_recovery["schemaValidationStatus"])

    def test_non_retryable_http_statuses_fail_without_transient_retry(self) -> None:
        for status in (400, 401, 403, 404):
            calls = []

            def request(_payload, status=status):
                calls.append(True)
                raise llm_transport.ProviderError("HTTP_" + str(status), transient=False)

            transport = llm_transport.StructuredGeminiTransport(
                request=request,
                sleeper=lambda _delay: self.fail("non-retryable status slept"),
            )
            with self.assertRaises(llm_transport.ProviderError):
                transport.call_raw({"contents": []})
            self.assertEqual(1, len(calls))

    def test_transient_status_policy_is_exactly_bounded(self) -> None:
        for status in (429, 500, 502, 503, 504):
            self.assertTrue(llm_transport.is_transient_http_status(status))
        for status in (400, 401, 403, 404):
            self.assertFalse(llm_transport.is_transient_http_status(status))

    def test_transport_attempts_never_exceed_configured_maximum(self) -> None:
        calls = []

        def request(_payload):
            calls.append(True)
            raise llm_transport.ProviderError("HTTP_503", transient=True)

        transport = llm_transport.StructuredGeminiTransport(
            request=request,
            max_transport_attempts=2,
            sleeper=lambda _delay: None,
        )
        with self.assertRaises(llm_transport.ProviderError):
            transport.call_raw({"contents": []})
        self.assertEqual(2, len(calls))

    def test_delay_calculation_is_deterministic_and_bounded(self) -> None:
        error = llm_transport.ProviderError("HTTP_503", transient=True)
        self.assertEqual(1.0, llm_transport.calculate_retry_delay(1, error))
        self.assertEqual(2.0, llm_transport.calculate_retry_delay(2, error))
        self.assertEqual(10.0, llm_transport.calculate_retry_delay(20, error))

    def test_retry_after_is_respected_but_bounded(self) -> None:
        short = llm_transport.ProviderError("HTTP_503", transient=True, retry_after=4.5)
        long = llm_transport.ProviderError("HTTP_503", transient=True, retry_after=99.0)
        self.assertEqual(4.5, llm_transport.calculate_retry_delay(1, short))
        self.assertEqual(10.0, llm_transport.calculate_retry_delay(1, long))

    def test_live_transport_uses_bounded_retry_after_without_exposing_request_data(self) -> None:
        class Pool:
            keys = ["TEST_KEY"]

            def get_key(self):
                return "TEST_KEY"

            def report_error(self, _key, _status):
                return None

        first = type("Response", (), {"status_code": 503, "headers": {"Retry-After": "4.5"}})()
        second = type(
            "Response",
            (),
            {
                "status_code": 200,
                "headers": {},
                "json": lambda self: {"candidates": [{"content": {"parts": [{"text": '{"ok":true}'}]}}]},
            },
        )()
        sleeps = []
        with patch.object(llm_transport, "gemini_pool", Pool()), patch.object(
            llm_transport.requests, "post", side_effect=[first, second]
        ) as post:
            result = llm_transport.gemini_json_transport(
                model="fake",
                sleeper=sleeps.append,
            ).call_raw({"contents": []})
        self.assertEqual({"ok": True}, result)
        self.assertEqual(2, post.call_count)
        self.assertEqual([4.5], sleeps)

    def test_recovery_diagnostics_keep_logical_and_transport_fields_separate(self) -> None:
        invalid = _canonical_av()
        invalid["scene"] = "bad-shape"
        analyzer, _ = self._analyzer([
            invalid,
            llm_transport.ProviderError("HTTP_503", transient=True),
            llm_transport.ProviderError("HTTP_503", transient=True),
            llm_transport.ProviderError("HTTP_503", transient=True),
        ])
        with tempfile.TemporaryDirectory() as directory:
            diagnostics = Path(directory) / "recovery.json"
            with self.assertRaises(llm_transport.ProviderError):
                analyzer.analyze_media(
                    self._media(directory),
                    start_ms=0,
                    end_ms=1_000,
                    recovery_metadata_path=diagnostics,
                )
            saved = json.loads(diagnostics.read_text(encoding="utf-8"))
        self.assertEqual(2, saved["attemptCount"])
        self.assertEqual(3, saved["transportAttempts"])
        self.assertEqual("HTTP_503", saved["httpStatus"])
        self.assertEqual("NOT_RUN", saved["schemaValidationStatus"])
        self.assertNotIn("TEST_KEY", json.dumps(saved))

    def test_subdivision_accounting_separates_logical_and_http_counts(self) -> None:
        invalid = _window_av(0, 30_000)
        invalid["scene"] = "bad-shape"
        sleeps = []
        analyzer, calls = self._analyzer([
            invalid, invalid,
            llm_transport.ProviderError("HTTP_503", transient=True),
            _window_av(0, 15_000),
            _window_av(12_000, 27_000),
            _window_av(24_000, 30_000),
        ], sleeps)
        with tempfile.TemporaryDirectory() as directory:
            result = av_analyzer.analyze_media_windows_with_subdivision(
                analyzer,
                self._media(directory),
                30_000,
                diagnostics_path=Path(directory) / "subdivision.json",
            )
        self.assertEqual(6, len(calls))
        self.assertEqual(5, result["logicalAvGenerations"])
        self.assertEqual(6, result["httpRequests"])
        self.assertEqual(6, result["providerCalls"])
        self.assertEqual(1, result["schemaRepairGenerations"])
        self.assertEqual(3, result["subdivisionChildLogicalGenerations"])
        self.assertEqual(2, result["diagnostics"][0]["attemptCount"])
        self.assertEqual(2, result["diagnostics"][0]["transportAttempts"])
        self.assertEqual(2, result["diagnostics"][0]["children"][0]["transportAttempts"])

    def test_transport_failure_diagnostic_preserves_resume_safe_accounting(self) -> None:
        analyzer, calls = self._analyzer([
            llm_transport.ProviderError("HTTP_503", transient=True),
            llm_transport.ProviderError("HTTP_503", transient=True),
            llm_transport.ProviderError("HTTP_503", transient=True),
        ])
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "subdivision.json"
            with self.assertRaises(llm_transport.ProviderError):
                av_analyzer.analyze_media_windows_with_subdivision(
                    analyzer,
                    self._media(directory),
                    30_000,
                    diagnostics_path=report_path,
                )
            report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(3, len(calls))
        self.assertEqual(1, report["logicalAvGenerations"])
        self.assertEqual(3, report["httpRequests"])
        self.assertEqual("AV_PARENT_PROVIDER_ERROR", report["finalReconciliationStatus"])
        self.assertEqual(3, report["windows"][0]["transportAttempts"])
        self.assertEqual("AV_PROVIDER_TEMPORARILY_UNAVAILABLE", report["windows"][0]["finalStatus"])


if __name__ == "__main__":
    unittest.main()
