from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cue_aligner import build_alignment, write_alignment_artifacts
from tts_timing import run_tts_timing


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
        self.assertTrue(result["rows"][0]["reviewRequired"])
        self.assertEqual("TTS_FAILED", result["rows"][0]["reviewReason"])

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
        qc = json.loads((self.output / "tts_qc_v3.json").read_text(encoding="utf-8"))
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
        manifest = json.loads((self.output / "tts_manifest_v3.json").read_text(encoding="utf-8"))
        qc = json.loads((self.output / "tts_qc_v3.json").read_text(encoding="utf-8"))
        self.assertEqual(1, manifest["schemaVersion"])
        self.assertEqual(1, qc["schemaVersion"])
        self.assertIn("measuredMsBefore", manifest["rows"][0])
        self.assertIn("reviewRequired", qc["rows"][0])


if __name__ == "__main__":
    unittest.main()
