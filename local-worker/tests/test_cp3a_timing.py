from __future__ import annotations

import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from cue_aligner import CueAlignmentError, build_alignment, write_alignment_artifacts
from tts_timing import run_cp3a, run_tts_timing


def _record(
    identifier: str = "u1",
    *,
    text: str = "Xin chào, mình đi nhé.",
    resolved: str = "你好，我走了",
    positions: list[int] | None = None,
    start: int = 0,
    end: int = 3000,
    classification: str = "RESOLVED_DIALOGUE",
    flags: list[str] | None = None,
    native: str | None = None,
) -> dict[str, object]:
    return {
        "utteranceId": identifier,
        "start_ms": start,
        "end_ms": end,
        "resolvedZh": resolved,
        "nativeVi": text if native is None else native,
        "positions": [0] if positions is None else positions,
        "classification": classification,
        "type": "DIALOGUE" if classification == "RESOLVED_DIALOGUE" else classification,
        "source_flags": flags or [],
    }


class CueAlignmentTests(unittest.TestCase):
    def test_mapping_preserves_identity_positions_and_time_range(self) -> None:
        result = build_alignment([_record(positions=[22], start=1200, end=1800)])
        row = result["dialogue"][0]
        self.assertEqual("u1", row["utteranceId"])
        self.assertEqual([22], row["sourcePositions"])
        self.assertEqual((1200, 1800), (row["startMs"], row["endMs"]))
        cue = row["alignedCues"][0]
        self.assertEqual("u1", cue["utteranceId"])
        self.assertEqual([22], cue["sourcePositions"])

    def test_one_semantic_utterance_can_span_multiple_original_cues(self) -> None:
        result = build_alignment([_record(text="Tớ đã dành dụm tiền để mua cặp sách mới.", positions=[1, 2, 3])])
        row = result["dialogue"][0]
        self.assertGreaterEqual(len(row["alignedCues"]), 2)
        self.assertEqual([1, 2, 3], row["sourcePositions"])
        self.assertEqual("u1", row["alignedCues"][0]["utteranceId"])

    def test_without_a_semantic_boundary_multiple_positions_are_grouped_not_word_sliced(self) -> None:
        result = build_alignment([_record(text="Ngươi nghe rõ chưa?", positions=[1, 2, 3])])
        cues = result["dialogue"][0]["alignedCues"]
        self.assertEqual(1, len(cues))
        self.assertEqual([1, 2, 3], cues[0]["sourcePositions"])
        self.assertEqual("Ngươi nghe rõ chưa?", cues[0]["translatedTextVi"])

    def test_text_uses_punctuation_and_phrase_boundaries_not_character_slices(self) -> None:
        result = build_alignment([_record(text="Tớ gom tiền, rồi mua cặp mới nhé.", positions=[1, 2, 3])])
        cues = result["dialogue"][0]["alignedCues"]
        joined = " ".join(cue["translatedTextVi"] for cue in cues)
        self.assertEqual("Tớ gom tiền, rồi mua cặp mới nhé.", joined)
        self.assertTrue(any("," in cue["translatedTextVi"] for cue in cues))
        self.assertTrue(all(" " not in cue["translatedTextVi"][:1] for cue in cues))

    def test_visual_only_and_unresolved_rows_are_diagnostic_not_tts_dialogue(self) -> None:
        result = build_alignment([
            _record("visual", classification="VISUAL_ONLY", text="", resolved=""),
            _record("unknown", classification="UNRESOLVED_DIALOGUE", text="", resolved=""),
            _record("spoken"),
        ])
        self.assertEqual(["spoken"], [row["utteranceId"] for row in result["dialogue"]])
        self.assertEqual({"visual", "unknown"}, {row["utteranceId"] for row in result["excluded"]})

    def test_visual_only_without_positions_remains_a_diagnostic_exclusion(self) -> None:
        visual = _record("visual", classification="VISUAL_ONLY", text="", resolved="")
        visual.pop("positions")
        result = build_alignment([visual, _record("spoken")])
        self.assertEqual([], result["excluded"][0]["sourcePositions"])

    def test_dialogue_without_positions_uses_one_whole_utterance_cue(self) -> None:
        """Missing cue positions must not invent granular source evidence."""

        result = build_alignment([_record("spoken", positions=[], start=4000, end=4800)])
        row = result["dialogue"][0]
        self.assertEqual([], row["sourcePositions"])
        self.assertEqual(1, len(row["alignedCues"]))
        cue = row["alignedCues"][0]
        self.assertIsNone(cue["position"])
        self.assertEqual([], cue["sourcePositions"])
        self.assertEqual((4000, 4800), (cue["startMs"], cue["endMs"]))
        self.assertEqual("Xin chào, mình đi nhé.", cue["translatedTextVi"])
        self.assertEqual("utterance_range_no_cues", cue["diagnostics"]["timingBasis"])

    def test_source_uncertain_is_represented_but_remains_review_required(self) -> None:
        result = build_alignment([_record(flags=["SOURCE_UNCERTAIN"])])
        row = result["dialogue"][0]
        self.assertTrue(row["reviewRequired"])
        self.assertIn("SOURCE_UNCERTAIN", row["reviewFlags"])
        self.assertFalse(row["ttsEligible"])

    def test_alignment_reconciles_every_accepted_dialogue_without_duplicate_or_loss(self) -> None:
        result = build_alignment([_record("a"), _record("b", positions=[1, 2])])
        self.assertEqual({"a", "b"}, set(result["coverage"]["alignedUtteranceIds"]))
        self.assertEqual(2, result["coverage"]["alignedUtteranceCount"])
        for row in result["dialogue"]:
            self.assertEqual(row["nativeVi"], " ".join(cue["translatedTextVi"] for cue in row["alignedCues"]))

    def test_alignment_fails_closed_on_missing_or_extra_translation_ids(self) -> None:
        with self.assertRaises(CueAlignmentError):
            build_alignment([_record("u1")], accepted=[])
        with self.assertRaises(CueAlignmentError):
            build_alignment([_record("u1")], accepted=[_record("extra")])

    def test_source_timing_and_positions_remain_authoritative_over_accepted_payload(self) -> None:
        source = _record("u1", positions=[9], start=100, end=900)
        accepted = {**source, "positions": [99], "start_ms": 1, "end_ms": 2, "nativeVi": source["nativeVi"]}
        result = build_alignment([source], accepted=[accepted])
        row = result["dialogue"][0]
        self.assertEqual([9], row["sourcePositions"])
        self.assertEqual((100, 900), (row["startMs"], row["endMs"]))

    def test_alignment_artifact_has_required_table_and_preserves_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            paths = write_alignment_artifacts(
                build_alignment([_record(), _record("visual", classification="VISUAL_ONLY", text="", resolved="")]),
                Path(temp),
            )
            self.assertEqual({"cue_alignment_v3.json", "cue_alignment_v3.md"}, {path.name for path in paths})
            markdown = (Path(temp) / "cue_alignment_v3.md").read_text(encoding="utf-8")
            self.assertIn("| Utterance | Positions | Source time | Native VI | Aligned cue(s) | Review flags |", markdown)
            self.assertIn("visual", markdown)


class TtsTimingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)

    def _alignment(self, record: dict[str, object] | None = None) -> dict[str, object]:
        return build_alignment([record or _record(positions=[0], start=0, end=1000)])

    def test_measured_duration_and_tempo_ceiling_are_persisted(self) -> None:
        calls: list[str] = []
        durations = iter([1100])

        def synthesize(text: str, voice: str, target: Path) -> None:
            calls.append(text)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"audio")

        def fit_audio(raw: Path, fitted: Path, slot_ms: int, max_tempo: float) -> tuple[int, bool]:
            fitted.write_bytes(raw.read_bytes())
            return 1000, False

        manifest = run_tts_timing(
            self._alignment(), self.output, synthesizer=synthesize,
            measure_duration=lambda path: next(durations), fit_audio=fit_audio,
        )
        row = manifest["rows"][0]
        self.assertEqual(1100, row["measuredMsBefore"])
        self.assertEqual(1.1, row["tempoRequired"])
        self.assertLessEqual(row["tempoRequired"], 1.12)
        self.assertEqual(["Xin chào, mình đi nhé."], calls)

    def test_timing_repair_only_runs_after_overflow_and_only_once(self) -> None:
        syntheses: list[str] = []
        repairs: list[dict[str, object]] = []
        durations = iter([1500, 900])
        fit_results = iter([(1500, True), (900, False)])

        def synthesize(text: str, voice: str, target: Path) -> None:
            syntheses.append(text)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"audio")

        def repair(text: str, context: dict[str, object]) -> str:
            repairs.append(context)
            return "Xin chào, mình đi."

        def fit_audio(raw: Path, fitted: Path, slot_ms: int, max_tempo: float) -> tuple[int, bool]:
            fitted.write_bytes(raw.read_bytes())
            return next(fit_results)

        manifest = run_tts_timing(
            self._alignment(), self.output, synthesizer=synthesize,
            measure_duration=lambda path: next(durations), fit_audio=fit_audio,
            timing_repair=repair,
        )
        row = manifest["rows"][0]
        self.assertEqual(["Xin chào, mình đi nhé.", "Xin chào, mình đi."], syntheses)
        self.assertEqual(1, len(repairs))
        self.assertEqual("u1", repairs[0]["utteranceId"])
        self.assertEqual("你好，我走了", repairs[0]["resolvedZh"])
        self.assertEqual([0], repairs[0]["sourcePositions"])
        self.assertTrue(row["timingRepairAttempted"])
        self.assertEqual(1500, row["measuredMsBefore"])
        self.assertEqual(900, row["measuredMsAfter"])
        self.assertFalse(row["reviewRequired"])

    def test_timing_repair_rejects_semantic_drop_without_resynthesis(self) -> None:
        syntheses: list[str] = []

        def synthesize(text: str, voice: str, target: Path) -> None:
            syntheses.append(text)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"audio")

        def fit_audio(raw: Path, fitted: Path, slot_ms: int, max_tempo: float) -> tuple[int, bool]:
            fitted.write_bytes(raw.read_bytes())
            return 1500, True

        result = run_tts_timing(
            self._alignment(_record(text="Tớ đã dành dụm rất nhiều tiền để mua cặp mới.", resolved="我攒了重金买了新书包")),
            self.output,
            synthesizer=synthesize,
            measure_duration=lambda path: 1500,
            fit_audio=fit_audio,
            timing_repair=lambda text, context: "Mua cặp đắt tiền.",
        )
        row = result["rows"][0]
        self.assertEqual(["Tớ đã dành dụm rất nhiều tiền để mua cặp mới."], syntheses)
        self.assertTrue(row["timingRepairAttempted"])
        self.assertEqual("Tớ đã dành dụm rất nhiều tiền để mua cặp mới.", row["nativeViAfterTiming"])
        self.assertTrue(row["reviewRequired"])
        self.assertIsNone(row["measuredMsAfter"])

    def test_timing_repair_cannot_change_source_identity(self) -> None:
        alignment = self._alignment()
        captured: list[dict[str, object]] = []

        def synthesize(text: str, voice: str, target: Path) -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"audio")

        def fit_audio(raw: Path, fitted: Path, slot_ms: int, max_tempo: float) -> tuple[int, bool]:
            fitted.write_bytes(raw.read_bytes())
            return (1500, True)

        def repair(text: str, context: dict[str, object]) -> str:
            captured.append(context)
            return "Một câu ngắn."

        result = run_tts_timing(
            alignment, self.output, synthesizer=synthesize,
            measure_duration=lambda path: 1500, fit_audio=fit_audio,
            timing_repair=repair,
        )
        row = result["rows"][0]
        self.assertEqual("u1", row["utteranceId"])
        self.assertEqual("你好，我走了", row["resolvedZh"])
        self.assertEqual([0], row["alignedPositions"])
        self.assertEqual([0], captured[0]["sourcePositions"])

    def test_failed_tts_row_remains_review_required(self) -> None:
        def fail(text: str, voice: str, target: Path) -> None:
            raise RuntimeError("synthetic TTS failure")

        result = run_tts_timing(self._alignment(), self.output, synthesizer=fail)
        row = result["rows"][0]
        self.assertTrue(row["reviewRequired"])
        self.assertEqual("TTS_FAILED", row["reviewReason"])
        self.assertEqual("u1", row["utteranceId"])
        self.assertEqual("RuntimeError: synthetic TTS failure", row["providerError"])

    def test_failed_cache_event_is_retried_while_valid_audio_is_reused(self) -> None:
        records = build_alignment([
            _record("good", positions=[0], start=0, end=1000),
            _record("missing", text="Câu thiếu.", native="Câu thiếu.", positions=[1], start=1000, end=2000),
        ])
        first_calls: list[str] = []

        def first_synthesize(text: str, voice: str, target: Path) -> None:
            first_calls.append(text)
            if text == "Xin chào, mình đi nhé.":
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"audio")
                return
            raise RuntimeError("transient provider failure")

        fit_audio = lambda raw, fitted, slot_ms, max_tempo: (fitted.write_bytes(raw.read_bytes()) or 500, False)
        first = run_tts_timing(
            records, self.output, synthesizer=first_synthesize,
            measure_duration=lambda path: 500, fit_audio=fit_audio,
        )
        self.assertEqual(2, len(first["voiceEvents"]))
        self.assertEqual("TTS_FAILED", first["voiceEvents"][1]["reviewReason"])

        retry_calls: list[str] = []

        def retry_synthesize(text: str, voice: str, target: Path) -> None:
            retry_calls.append(text)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"recovered-audio")

        second = run_tts_timing(
            records, self.output, synthesizer=retry_synthesize,
            measure_duration=lambda path: 500, fit_audio=fit_audio,
        )
        self.assertEqual(1, second["cacheHits"])
        self.assertEqual(["Câu thiếu."], retry_calls)
        self.assertEqual(["good", "missing"], [row["utteranceId"] for row in second["voiceEvents"]])
        self.assertTrue(all(row.get("audioPath") for row in second["voiceEvents"]))

    def test_source_uncertain_row_is_not_sent_to_tts(self) -> None:
        calls: list[str] = []
        record = _record(flags=["SOURCE_UNCERTAIN"])

        def synthesize(text: str, voice: str, target: Path) -> None:
            calls.append(text)

        result = run_tts_timing(self._alignment(record), self.output, synthesizer=synthesize)
        self.assertEqual([], calls)
        self.assertTrue(result["rows"][0]["reviewRequired"])
        self.assertEqual("SOURCE_UNCERTAIN", result["rows"][0]["reviewReason"])

    def test_source_uncertain_takes_precedence_over_partial_cue_reason(self) -> None:
        record = _record(flags=["PARTIAL_CUE", "SOURCE_UNCERTAIN"])
        result = run_tts_timing(self._alignment(record), self.output, synthesizer=lambda text, voice, target: None)
        self.assertEqual("SOURCE_UNCERTAIN", result["rows"][0]["reviewReason"])

    def test_valid_audio_manifest_is_reused_without_another_tts_call(self) -> None:
        calls: list[str] = []

        def synthesize(text: str, voice: str, target: Path) -> None:
            calls.append(text)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"audio")

        def fit_audio(raw: Path, fitted: Path, slot_ms: int, max_tempo: float) -> tuple[int, bool]:
            fitted.write_bytes(raw.read_bytes())
            return 500, False

        first = run_tts_timing(
            self._alignment(), self.output, synthesizer=synthesize,
            measure_duration=lambda path: 500, fit_audio=fit_audio,
        )
        second = run_tts_timing(
            self._alignment(), self.output,
            synthesizer=lambda text, voice, target: (_ for _ in ()).throw(AssertionError("cache miss")),
            measure_duration=lambda path: 9999, fit_audio=fit_audio,
        )
        self.assertEqual(1, len(calls))
        self.assertEqual(1, second["cacheHits"])
        self.assertTrue(second["rows"][0]["cacheHit"])
        self.assertEqual(first["rows"][0]["audioPath"], second["rows"][0]["audioPath"])

    def test_cache_collision_cannot_reuse_audio_for_another_utterance_identity(self) -> None:
        def synthesize(text: str, voice: str, target: Path) -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"audio")

        fit_audio = lambda raw, fitted, slot_ms, max_tempo: (fitted.write_bytes(raw.read_bytes()) or 5, False)
        run_tts_timing(
            self._alignment(_record("first")), self.output, synthesizer=synthesize,
            measure_duration=lambda path: 500, fit_audio=fit_audio,
        )
        calls: list[str] = []
        second = run_tts_timing(
            self._alignment(_record("second")), self.output,
            synthesizer=lambda text, voice, target: (calls.append(text), target.write_bytes(b"new-audio")),
            measure_duration=lambda path: 500, fit_audio=fit_audio,
        )
        self.assertEqual(["Xin chào, mình đi nhé."], calls)
        self.assertEqual("second", second["rows"][0]["utteranceId"])
        self.assertFalse(second["rows"][0]["cacheHit"])

    def test_qc_review_count_includes_review_rows_reused_from_cache(self) -> None:
        def synthesize(text: str, voice: str, target: Path) -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"audio")

        def fit_audio(raw: Path, fitted: Path, slot_ms: int, max_tempo: float) -> tuple[int, bool]:
            fitted.write_bytes(raw.read_bytes())
            return 1500, True

        run_tts_timing(
            self._alignment(), self.output, synthesizer=synthesize,
            measure_duration=lambda path: 1500, fit_audio=fit_audio,
        )
        second = run_tts_timing(
            self._alignment(), self.output,
            synthesizer=lambda text, voice, target: (_ for _ in ()).throw(AssertionError("cache miss")),
            measure_duration=lambda path: 1, fit_audio=fit_audio,
        )
        qc = json.loads((self.output / "tts_qc_v4.json").read_text(encoding="utf-8"))
        self.assertEqual(1, qc["reviewCount"])
        self.assertEqual("TTS_TIMING_OVERFLOW", qc["reviewRequired"][0]["reason"])
        self.assertTrue(second["rows"][0]["reviewRequired"])

    def test_manifest_is_machine_readable_and_contains_qc_artifact(self) -> None:
        def synthesize(text: str, voice: str, target: Path) -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"audio")

        run_tts_timing(
            self._alignment(), self.output, synthesizer=synthesize,
            measure_duration=lambda path: 500,
            fit_audio=lambda raw, fitted, slot_ms, max_tempo: (fitted.write_bytes(raw.read_bytes()) or 500, False),
        )
        manifest = json.loads((self.output / "tts_manifest_v4.json").read_text(encoding="utf-8"))
        qc = json.loads((self.output / "tts_qc_v4.json").read_text(encoding="utf-8"))
        self.assertEqual(1, manifest["schemaVersion"])
        self.assertEqual(1, qc["schemaVersion"])
        self.assertIn("measuredMsBefore", manifest["rows"][0])
        self.assertIn("reviewRequired", qc["rows"][0])


class VoiceEventTimingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)

    def _alignment(self, *records: dict[str, object]) -> dict[str, object]:
        return build_alignment(list(records))

    def _synthesizer(self, calls: list[str]):
        def synthesize(text: str, voice: str, target: Path) -> None:
            calls.append(text)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"audio")

        return synthesize

    def _fit(self, calls: list[int], duration: int = 900, review: bool = False):
        def fit_audio(raw: Path, fitted: Path, slot_ms: int, max_tempo: float) -> tuple[int, bool]:
            calls.append(slot_ms)
            fitted.write_bytes(raw.read_bytes())
            return duration, review

        return fit_audio

    def test_one_semantic_utterance_produces_one_voice_event_and_one_tts_call(self) -> None:
        record = _record(text="Tớ gom tiền, rồi mua cặp mới nhé.", positions=[1, 2, 3], start=0, end=3000)
        calls: list[str] = []
        result = run_tts_timing(
            self._alignment(record), self.output, synthesizer=self._synthesizer(calls),
            measure_duration=lambda path: 900, fit_audio=self._fit([]), timing_repair=None,
        )
        self.assertEqual([record["nativeVi"]], calls)
        self.assertEqual(1, len(result["voiceEvents"]))
        self.assertEqual(1, len(result["rows"]))
        self.assertGreaterEqual(len(result["subtitleCues"]), 2)
        self.assertEqual([1, 2, 3], sorted({position for cue in result["subtitleCues"] for position in cue["sourcePositions"]}))

    def test_subtitle_cue_splitting_does_not_create_provider_calls(self) -> None:
        record = _record(text="Tớ gom tiền, rồi mua cặp mới nhé.", positions=[1, 2, 3], start=0, end=3000)
        calls: list[str] = []
        result = run_tts_timing(
            self._alignment(record), self.output, synthesizer=self._synthesizer(calls),
            measure_duration=lambda path: 900, fit_audio=self._fit([]), timing_repair=None,
        )
        self.assertEqual(1, result["providerCalls"])
        self.assertEqual([1, 2, 3], sorted({position for cue in result["subtitleCues"] for position in cue["sourcePositions"]}))
        self.assertTrue(all("audioPath" not in cue for cue in result["subtitleCues"]))

    def test_full_utterance_window_and_positions_remain_available(self) -> None:
        record = _record(text="Tớ gom tiền, rồi mua cặp mới nhé.", positions=[1, 2, 3], start=1200, end=4200)
        fit_slots: list[int] = []
        result = run_tts_timing(
            self._alignment(record), self.output, synthesizer=self._synthesizer([]),
            measure_duration=lambda path: 900, fit_audio=self._fit(fit_slots), timing_repair=None,
        )
        event = result["voiceEvents"][0]
        self.assertEqual((1200, 4200), (event["baseStartMs"], event["baseEndMs"]))
        self.assertEqual([3000], fit_slots)
        self.assertEqual([1, 2, 3], event["sourcePositions"])
        self.assertEqual([1, 2, 3], sorted({position for cue in result["subtitleCues"] for position in cue["sourcePositions"]}))
        self.assertEqual("u1", result["subtitleCues"][0]["utteranceId"])

    def test_visual_only_and_unresolved_dialogue_never_enter_tts(self) -> None:
        visual = _record("visual", classification="VISUAL_ONLY", text="", resolved="")
        visual.pop("positions")
        unresolved = _record("unknown", classification="UNRESOLVED_DIALOGUE", text="", resolved="")
        calls: list[str] = []
        result = run_tts_timing(
            self._alignment(visual, unresolved, _record("spoken")), self.output,
            synthesizer=self._synthesizer(calls), measure_duration=lambda path: 500,
            fit_audio=self._fit([]), timing_repair=None,
        )
        self.assertEqual(["Xin chào, mình đi nhé."], calls)
        self.assertEqual(["spoken"], [event["utteranceId"] for event in result["voiceEvents"]])

    def test_source_uncertain_stays_review_required_without_audio(self) -> None:
        record = _record(flags=["SOURCE_UNCERTAIN"])
        calls: list[str] = []
        result = run_tts_timing(
            self._alignment(record), self.output, synthesizer=self._synthesizer(calls),
            measure_duration=lambda path: 500, fit_audio=self._fit([]), timing_repair=None,
        )
        event = result["voiceEvents"][0]
        self.assertEqual([], calls)
        self.assertTrue(event["reviewRequired"])
        self.assertEqual("SOURCE_UNCERTAIN", event["reviewReason"])
        self.assertIsNone(event["audioPath"])

    def test_timing_repair_receives_full_native_line_and_context(self) -> None:
        record = _record(
            text="Tớ gom tiền, rồi mua cặp mới nhé.", positions=[1, 2, 3], start=0, end=1000,
        )
        record.update({"intent": "request", "roleplay": "palace", "register": "formal"})
        calls: list[str] = []
        repair_contexts: list[dict[str, object]] = []

        def repair(text: str, context: dict[str, object]) -> str:
            repair_contexts.append(context)
            return "Gom tiền rồi mua cặp mới nhé."

        result = run_tts_timing(
            self._alignment(record), self.output, synthesizer=self._synthesizer(calls),
            measure_duration=lambda path: 1500, fit_audio=self._fit([], 900, True), timing_repair=repair,
        )
        self.assertEqual([record["nativeVi"], "Gom tiền rồi mua cặp mới nhé."], calls)
        context = repair_contexts[0]
        self.assertEqual(record["nativeVi"], context["approvedNativeVi"])
        self.assertEqual([1, 2, 3], context["sourcePositions"])
        self.assertEqual("request", context["intent"])
        self.assertEqual("palace", context["roleplay"])
        self.assertEqual("formal", context["register"])
        self.assertEqual("Gom tiền rồi mua cặp mới nhé.", result["voiceEvents"][0]["deliveryNativeVi"])

    def test_only_one_linguistic_repair_is_allowed(self) -> None:
        repairs: list[str] = []

        def repair(text: str, context: dict[str, object]) -> str:
            repairs.append(text)
            return "Gom tiền rồi mua cặp mới nhé."

        result = run_tts_timing(
            self._alignment(_record(text="Tớ gom tiền, rồi mua cặp mới nhé.")), self.output,
            synthesizer=self._synthesizer([]), measure_duration=lambda path: 1500,
            fit_audio=self._fit([], 1500, True), timing_repair=repair,
        )
        self.assertEqual(1, len(repairs))
        self.assertTrue(result["voiceEvents"][0]["timingRepairAttempted"])
        self.assertTrue(result["voiceEvents"][0]["reviewRequired"])

    def test_subtitle_alignment_after_repair_does_not_resynthesize_cues(self) -> None:
        calls: list[str] = []
        fit_results = iter([(1500, True), (900, False)])

        def fit_audio(raw: Path, fitted: Path, slot_ms: int, max_tempo: float) -> tuple[int, bool]:
            fitted.write_bytes(raw.read_bytes())
            return next(fit_results)

        result = run_tts_timing(
            self._alignment(_record(text="Tớ gom tiền, rồi mua cặp mới nhé.", positions=[1, 2, 3])), self.output,
            synthesizer=self._synthesizer(calls), measure_duration=lambda path: 1500,
            fit_audio=fit_audio, timing_repair=lambda text, context: "Gom tiền rồi mua cặp mới nhé.",
        )
        self.assertEqual(2, len(calls))
        self.assertGreater(len(result["subtitleCues"]), 1)
        self.assertEqual(1, len(result["voiceEvents"]))

    def test_whole_utterance_cache_reuse_is_identity_safe(self) -> None:
        calls: list[str] = []
        alignment = self._alignment(_record(text="Tớ gom tiền, rồi mua cặp mới nhé.", positions=[1, 2, 3]))
        first = run_tts_timing(
            alignment, self.output, synthesizer=self._synthesizer(calls), measure_duration=lambda path: 500,
            fit_audio=self._fit([]), timing_repair=None,
        )
        second = run_tts_timing(
            alignment, self.output, synthesizer=lambda text, voice, target: (_ for _ in ()).throw(AssertionError("cache miss")),
            measure_duration=lambda path: 9999, fit_audio=self._fit([]), timing_repair=None,
        )
        self.assertEqual(1, len(calls))
        self.assertEqual(1, second["cacheHits"])
        self.assertEqual(first["voiceEvents"][0]["audioPath"], second["voiceEvents"][0]["audioPath"])

    def test_review_debt_is_counted_by_utterance_not_subtitle_fragment(self) -> None:
        record = _record(text="Tớ gom tiền, rồi mua cặp mới nhé.", positions=[1, 2, 3])
        result = run_tts_timing(
            self._alignment(record), self.output, synthesizer=self._synthesizer([]), measure_duration=lambda path: 1500,
            fit_audio=self._fit([], 1500, True), timing_repair=None,
        )
        qc = json.loads((self.output / "tts_qc_v4.json").read_text(encoding="utf-8"))
        self.assertEqual(1, qc["reviewCount"])
        self.assertEqual(["u1"], [item["utteranceId"] for item in qc["reviewRequired"]])
        self.assertEqual(1, len(result["voiceEvents"]))

    def test_visual_gap_is_borrowed_only_before_next_spoken_utterance(self) -> None:
        first = _record("first", text="Câu đầu.", positions=[0], start=0, end=1000)
        visual = _record("visual", classification="VISUAL_ONLY", text="", resolved="", start=1000, end=1500)
        visual.pop("positions")
        second = _record("second", text="Câu sau.", positions=[1], start=1500, end=2500)
        result = run_tts_timing(
            self._alignment(first, visual, second), self.output, synthesizer=self._synthesizer([]),
            measure_duration=lambda path: 500, fit_audio=self._fit([]), timing_repair=None,
        )
        events = result["voiceEvents"]
        self.assertEqual(1500, events[0]["effectiveEndMs"])
        self.assertEqual(500, events[0]["borrowedGapMs"])
        self.assertEqual(0, events[1]["borrowedGapMs"])
        self.assertLessEqual(events[0]["effectiveEndMs"], events[1]["baseStartMs"])

    def test_gap_without_next_spoken_utterance_is_not_arbitrarily_extended(self) -> None:
        first = _record("first", text="Câu đầu.", positions=[0], start=0, end=1000)
        visual = _record("visual", classification="VISUAL_ONLY", text="", resolved="", start=1000, end=2000)
        visual.pop("positions")
        result = run_tts_timing(
            self._alignment(first, visual), self.output, synthesizer=self._synthesizer([]),
            measure_duration=lambda path: 500, fit_audio=self._fit([]), timing_repair=None,
        )
        self.assertEqual(1000, result["voiceEvents"][0]["effectiveEndMs"])
        self.assertEqual(0, result["voiceEvents"][0]["borrowedGapMs"])

    def test_v4_artifacts_have_one_row_per_utterance_and_required_qc_table(self) -> None:
        run_tts_timing(
            self._alignment(_record(text="Tớ gom tiền, rồi mua cặp mới nhé.", positions=[1, 2, 3])), self.output,
            synthesizer=self._synthesizer([]), measure_duration=lambda path: 500,
            fit_audio=self._fit([]), timing_repair=None,
        )
        expected = {
            "voice_events_v4.json", "subtitle_alignment_v4.json", "tts_manifest_v4.json",
            "tts_qc_v4.json", "tts_qc_v4.md",
        }
        self.assertEqual(expected, {path.name for path in self.output.iterdir() if path.is_file()})
        markdown = (self.output / "tts_qc_v4.md").read_text(encoding="utf-8")
        self.assertIn("| ID | Source time | Effective window | Borrowed gap | Approved VI | Delivery VI | Raw TTS ms | Tempo | Repair | Final TTS ms | Review |", markdown)

    def test_run_cp3a_resolves_existing_default_repair_hook_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cp2 = root / "cp2.json"
            record = _record(text="Câu ngắn.", positions=[0], start=0, end=1000)
            cp2.write_text(json.dumps({"records": [record], "accepted": [record]}), encoding="utf-8")
            calls: list[str] = []
            result = run_cp3a(
                cp2, root / "output", synthesizer=self._synthesizer(calls), measure_duration=lambda path: 500,
                fit_audio=self._fit([]),
            )
            self.assertEqual(["Câu ngắn."], calls)
            self.assertEqual(1, len(result["voiceEvents"]))

    def test_tts_uses_existing_lazy_capcut_synthesizer_adapter(self) -> None:
        calls: list[str] = []

        def capcut(text: str, voice: str, target: Path) -> None:
            calls.append(text)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"audio")

        with patch("dubvi_worker.capcut_synthesize", side_effect=capcut):
            result = run_tts_timing(
                self._alignment(_record(text="Câu ngắn.")), self.output,
                measure_duration=lambda path: 500, fit_audio=self._fit([]), timing_repair=None,
            )
        self.assertEqual(["Câu ngắn."], calls)
        self.assertEqual(1, result["providerCalls"])


if __name__ == "__main__":
    unittest.main()
