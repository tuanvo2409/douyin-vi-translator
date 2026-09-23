from __future__ import annotations

import copy
import unittest

from source_locked_timing import (
    SourceLockedTimingError,
    apply_delivery_rewrites,
    build_delivery_rewrite_payload,
    build_source_locked_timeline,
    select_overflow_utterance_ids,
    validate_delivery_rewrite_response,
)


def _event(
    identifier: str,
    start: int,
    end: int,
    duration: int,
    *,
    text: str | None = None,
    classification: str = "RESOLVED_DIALOGUE",
    review_reason: str | None = None,
    audio: str | None = None,
) -> dict[str, object]:
    return {
        "utteranceId": identifier,
        "sourcePositions": [identifier],
        "resolvedZh": f"source-{identifier}",
        "nativeViApproved": text or f"Approved {identifier}",
        "deliveryNativeVi": text or f"Approved {identifier}",
        "baseStartMs": start,
        "baseEndMs": end,
        "effectiveStartMs": start,
        "effectiveEndMs": end,
        "fittedDurationMs": duration,
        "audioPath": audio or f"audio/{identifier}.mp3",
        "reviewRequired": review_reason is not None,
        "reviewReason": review_reason,
        "classification": classification,
    }


def _cue(identifier: str, start: int, end: int, text: str) -> dict[str, object]:
    return {
        "voiceEventId": identifier,
        "utteranceId": identifier,
        "position": identifier,
        "sourcePositions": [identifier],
        "startMs": start,
        "endMs": end,
        "translatedTextVi": text,
    }


class SourceLockedTimingTests(unittest.TestCase):
    def test_voice_start_is_locked_to_local_source_start(self) -> None:
        result = build_source_locked_timeline(
            [_event("a", 0, 1000, 900), _event("b", 1200, 2200, 600)], [],
            source_duration_ms=2500,
        )
        self.assertEqual([0, 1200], [row["deliveryStartMs"] for row in result["voiceEvents"]])
        self.assertEqual([0, 1200], [row["scheduledStartMs"] for row in result["voiceEvents"]])

    def test_long_audio_never_cumulatively_shifts_later_events(self) -> None:
        result = build_source_locked_timeline(
            [_event("a", 0, 1000, 1100), _event("b", 1200, 2200, 900), _event("c", 2400, 3000, 500)],
            [], source_duration_ms=3500,
        )
        self.assertEqual([0, 1200, 2400], [row["deliveryStartMs"] for row in result["voiceEvents"]])

    def test_overflow_selection_creates_timing_repair_pressure(self) -> None:
        qc = {"rows": [
            {"utteranceId": "a", "reviewReason": "TTS_TIMING_OVERFLOW"},
            {"utteranceId": "b", "reviewReason": None},
        ]}
        self.assertEqual(["a"], select_overflow_utterance_ids(qc))

    def test_subtitle_timing_follows_final_local_voice_event(self) -> None:
        result = build_source_locked_timeline(
            [_event("a", 1000, 2000, 800)],
            [_cue("a", 1000, 1400, "Một"), _cue("a", 1400, 2000, "câu")],
            source_duration_ms=2500,
        )
        event = result["voiceEvents"][0]
        for cue in result["subtitleCues"]:
            self.assertGreaterEqual(cue["scheduledStartMs"], event["deliveryStartMs"])
            self.assertLessEqual(cue["scheduledEndMs"], event["deliveryEndMs"])

    def test_source_uncertain_is_excluded_from_canonical_speech(self) -> None:
        result = build_source_locked_timeline(
            [_event("known", 0, 500, 300), _event("unknown", 600, 1000, 300, review_reason="SOURCE_UNCERTAIN", audio=None)],
            [_cue("unknown", 600, 1000, "không được đọc")], source_duration_ms=1200,
        )
        self.assertEqual(["known"], result["includedUtteranceIds"])
        self.assertEqual(["unknown"], result["omittedUtteranceIds"])
        self.assertEqual([], result["subtitleCues"])

    def test_visual_only_is_excluded_even_if_an_audio_path_is_present(self) -> None:
        result = build_source_locked_timeline(
            [_event("visual", 0, 500, 300, classification="VISUAL_ONLY")], [], source_duration_ms=1000,
        )
        self.assertEqual([], result["voiceEvents"])
        self.assertEqual(["visual"], result["omittedUtteranceIds"])

    def test_already_fitting_audio_keeps_existing_delivery_and_audio(self) -> None:
        original = _event("good", 100, 900, 500, text="Giữ nguyên.", audio="audio/good.mp3")
        before = copy.deepcopy(original)
        result = build_source_locked_timeline([original], [], source_duration_ms=1000)
        row = result["voiceEvents"][0]
        self.assertEqual("Giữ nguyên.", row["deliveryNativeVi"])
        self.assertEqual("audio/good.mp3", row["audioPath"])
        self.assertEqual(before, original)

    def test_only_overflow_rows_are_eligible_for_delivery_rewrite(self) -> None:
        events = [_event("overflow", 0, 1000, 900), _event("good", 1200, 2200, 500)]
        rewrites = {"overflow": "Câu ngắn."}
        result = apply_delivery_rewrites(events, rewrites)
        self.assertEqual("Câu ngắn.", result[0]["deliveryNativeVi"])
        self.assertEqual("Approved good", result[1]["deliveryNativeVi"])

    def test_rewrite_cannot_change_neighbor_identity_or_text(self) -> None:
        events = [_event("a", 0, 1000, 900), _event("b", 1200, 2200, 500)]
        result = apply_delivery_rewrites(events, {"a": "Câu mới."})
        self.assertEqual(["a", "b"], [row["utteranceId"] for row in result])
        self.assertEqual("Approved b", result[1]["nativeViApproved"])
        self.assertEqual("Approved b", result[1]["deliveryNativeVi"])

    def test_approved_native_vietnamese_remains_immutable_and_delivery_is_separate(self) -> None:
        result = apply_delivery_rewrites([_event("a", 0, 1000, 500, text="Bản duyệt.")], {"a": "Bản đọc ngắn."})
        self.assertEqual("Bản duyệt.", result[0]["nativeViApproved"])
        self.assertEqual("Bản đọc ngắn.", result[0]["deliveryNativeVi"])
        self.assertNotEqual(result[0]["nativeViApproved"], result[0]["deliveryNativeVi"])

    def test_tempo_budget_is_explicit_and_never_above_ceiling(self) -> None:
        result = build_source_locked_timeline(
            [_event("a", 0, 1000, 1100)], [], source_duration_ms=1500,
        )
        self.assertLessEqual(result["voiceEvents"][0]["tempoRequired"], 1.12)

    def test_source_timing_identity_is_unchanged(self) -> None:
        result = build_source_locked_timeline(
            [_event("a", 700, 1700, 500)], [], source_duration_ms=2000,
        )
        row = result["voiceEvents"][0]
        self.assertEqual(700, row["sourceStartMs"])
        self.assertEqual(1700, row["sourceEndMs"])
        self.assertEqual(0, row["startOffsetMs"])

    def test_preview_duration_stays_near_source_without_cumulative_extension(self) -> None:
        result = build_source_locked_timeline(
            [_event("a", 0, 1000, 900), _event("b", 1200, 2200, 900)], [], source_duration_ms=2500,
        )
        self.assertEqual(2500, result["durationMs"])
        self.assertLess(result["durationMs"], 29000)

    def test_scheduler_fails_when_audio_cannot_fit_local_window(self) -> None:
        with self.assertRaises(SourceLockedTimingError):
            build_source_locked_timeline([_event("a", 0, 1000, 1500)], [], source_duration_ms=2000)

    def test_delivery_rewrite_payload_is_one_bounded_batch(self) -> None:
        payload = build_delivery_rewrite_payload(
            [_event("a", 0, 1000, 1500), _event("b", 1200, 2200, 1500)], ["a", "b"],
            attempt="initial", model="gemini-2.5-flash",
        )
        self.assertEqual("gemini-2.5-flash", payload["model"])
        self.assertEqual("initial", payload["attempt"])
        self.assertEqual(["a", "b"], [row["utteranceId"] for row in payload["rows"]])

    def test_delivery_rewrite_response_is_strict_and_traceable(self) -> None:
        response = {"translations": [
            {"utteranceId": "a", "deliveryNativeVi": "Câu ngắn."},
            {"utteranceId": "b", "deliveryNativeVi": "Câu khác."},
        ]}
        result = validate_delivery_rewrite_response(response, ["a", "b"])
        self.assertEqual({"a": "Câu ngắn.", "b": "Câu khác."}, result)
        with self.assertRaises(SourceLockedTimingError):
            validate_delivery_rewrite_response({"translations": [{"utteranceId": "a", "deliveryNativeVi": "中文"}]}, ["a"])


if __name__ == "__main__":
    unittest.main()
