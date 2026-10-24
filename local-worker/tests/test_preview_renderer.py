from __future__ import annotations

import copy
import unittest
from pathlib import Path

from preview_renderer import PreviewRenderError, build_ffmpeg_command, build_preview_timeline


def _event(
    identifier: str,
    start: int,
    end: int,
    duration: int,
    *,
    audio: str | None = None,
    review_reason: str | None = None,
) -> dict[str, object]:
    return {
        "utteranceId": identifier,
        "sourcePositions": [identifier],
        "resolvedZh": f"source-{identifier}",
        "nativeViApproved": f"VI {identifier}",
        "deliveryNativeVi": f"VI {identifier}",
        "baseStartMs": start,
        "baseEndMs": end,
        "effectiveStartMs": start,
        "effectiveEndMs": end,
        "fittedDurationMs": duration,
        "audioPath": audio if audio is not None else f"audio/{identifier}.mp3",
        "reviewRequired": review_reason is not None,
        "reviewReason": review_reason,
    }


class PreviewTimelineTests(unittest.TestCase):
    def test_preserves_source_order_and_schedules_without_overlap(self) -> None:
        events = [_event("b", 500, 1000, 700), _event("a", 0, 500, 600)]
        result = build_preview_timeline(events, [], source_duration_ms=2500, gap_ms=60)
        scheduled = result["voiceEvents"]
        self.assertEqual(["a", "b"], [row["utteranceId"] for row in scheduled])

    def test_scheduled_voice_events_have_the_required_gap(self) -> None:
        result = build_preview_timeline(
            [_event("a", 0, 500, 600), _event("b", 500, 1000, 700)],
            [], source_duration_ms=2500, gap_ms=60,
        )
        scheduled = result["voiceEvents"]
        self.assertGreaterEqual(scheduled[1]["scheduledStartMs"], scheduled[0]["scheduledEndMs"] + 60)

    def test_rejects_timeline_that_exceeds_max_duration(self) -> None:
        with self.assertRaises(PreviewRenderError):
            build_preview_timeline(
                [_event("long", 1000, 2000, 2500)], [], source_duration_ms=2000, max_duration_ms=3000,
            )

    def test_source_identity_is_immutable(self) -> None:
        events = [_event("u1", 100, 900, 500)]
        before = copy.deepcopy(events)
        result = build_preview_timeline(events, [], source_duration_ms=1000)
        self.assertEqual(before, events)
        scheduled = result["voiceEvents"][0]
        self.assertEqual("u1", scheduled["utteranceId"])
        self.assertEqual(["u1"], scheduled["sourcePositions"])
        self.assertEqual("source-u1", scheduled["resolvedZh"])
        self.assertEqual(100, scheduled["sourceStartMs"])
        self.assertEqual(900, scheduled["sourceEndMs"])

    def test_source_uncertain_has_no_voice_or_subtitle(self) -> None:
        events = [_event("known", 0, 500, 300), _event("uncertain", 500, 900, 300, audio=None, review_reason="SOURCE_UNCERTAIN")]
        events[1]["audioPath"] = None
        cues = [{"voiceEventId": "known", "startMs": 0, "endMs": 500, "translatedTextVi": "VI known"},
                {"voiceEventId": "uncertain", "startMs": 500, "endMs": 900, "translatedTextVi": "should not speak"}]
        result = build_preview_timeline(events, cues, source_duration_ms=1000)
        self.assertEqual(["known"], result["includedUtteranceIds"])
        self.assertEqual(["uncertain"], result["omittedUtteranceIds"])
        self.assertEqual(["known"], [row["voiceEventId"] for row in result["subtitleCues"]])

    def test_subtitle_timings_follow_scheduled_voice_event(self) -> None:
        events = [_event("u1", 1000, 2000, 800)]
        cues = [
            {"voiceEventId": "u1", "startMs": 1000, "endMs": 1400, "translatedTextVi": "VI one"},
            {"voiceEventId": "u1", "startMs": 1400, "endMs": 2000, "translatedTextVi": "VI two"},
        ]
        result = build_preview_timeline(events, cues, source_duration_ms=2500)
        event = result["voiceEvents"][0]
        for cue in result["subtitleCues"]:
            self.assertEqual("u1", cue["voiceEventId"])
            self.assertGreaterEqual(cue["scheduledStartMs"], event["scheduledStartMs"])
            self.assertLessEqual(cue["scheduledEndMs"], event["scheduledEndMs"])

    def test_ffmpeg_command_enforces_broad_compatibility_encoding(self) -> None:
        command = build_ffmpeg_command(
            Path("source.mp4"), Path("output.mp4"), Path("subtitles.ass"),
            [(Path("voice.mp3"), 500)], duration_ms=25000, source_duration_ms=20000,
        )
        self.assertIn("-pix_fmt", command)
        self.assertEqual("yuv420p", command[command.index("-pix_fmt") + 1])
        self.assertEqual("libx264", command[command.index("-c:v") + 1])
        self.assertEqual("main", command[command.index("-profile:v") + 1])
        self.assertEqual("aac", command[command.index("-c:a") + 1])
        self.assertEqual("48000", command[command.index("-ar") + 1])
        self.assertEqual("2", command[command.index("-ac") + 1])
        self.assertIn("+faststart", command)
        self.assertNotIn("yuv444p", command)
        self.assertNotIn("96000", command)


if __name__ == "__main__":
    unittest.main()
