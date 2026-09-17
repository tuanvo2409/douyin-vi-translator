from __future__ import annotations

import unittest
from pathlib import Path

from adaptive_renderer import AdaptiveRenderError, build_adaptive_ffmpeg_command, build_adaptive_timeline


def _event(identifier: str, start: int, end: int, duration: int) -> dict[str, object]:
    return {
        "utteranceId": identifier,
        "sourcePositions": [],
        "nativeViApproved": f"Approved {identifier}",
        "deliveryNativeVi": f"Delivered {identifier}",
        "baseStartMs": start,
        "baseEndMs": end,
        "fittedDurationMs": duration,
        "audioPath": f"audio/{identifier}.mp3",
    }


def _plan(identifier: str, start: int, end: int, *, action: str = "KEEP", speed: float = 1.0, borrowed: int = 0) -> dict[str, object]:
    return {
        "utteranceId": identifier,
        "sourceStartMs": start,
        "sourceEndMs": end,
        "effectiveWindowMs": end - start + borrowed,
        "borrowedGapMs": borrowed,
        "visualSpeed": speed,
        "action": action,
    }


class AdaptiveRendererTests(unittest.TestCase):
    def test_piecewise_map_covers_source_once_and_locks_voice_to_mapped_source(self) -> None:
        timeline = build_adaptive_timeline(
            [_event("a", 0, 1000, 1500), _event("b", 1000, 2000, 600)],
            [
                {"utteranceId": "a", "voiceEventId": "a", "startMs": 0, "endMs": 1000, "translatedTextVi": "Delivered a"},
                {"utteranceId": "b", "voiceEventId": "b", "startMs": 1000, "endMs": 2000, "translatedTextVi": "Delivered b"},
            ],
            [_plan("a", 0, 1000, action="SLOW_VISUAL", speed=0.5), _plan("b", 1000, 2000)],
            source_duration_ms=2500,
        )
        segments = timeline["segments"]
        self.assertEqual(0, segments[0]["sourceStartMs"])
        self.assertEqual(2500, segments[-1]["sourceEndMs"])
        for left, right in zip(segments, segments[1:]):
            self.assertEqual(left["sourceEndMs"], right["sourceStartMs"])
            self.assertEqual(left["outputEndMs"], right["outputStartMs"])
        voices = timeline["voiceEvents"]
        self.assertEqual(0, voices[0]["scheduledStartMs"])
        self.assertEqual(2000, voices[1]["scheduledStartMs"])
        self.assertLessEqual(voices[0]["scheduledEndMs"], 2000)
        self.assertEqual(2000, voices[0]["deliveryLimitOutputMs"])

    def test_subtitles_follow_final_voice_window_not_source_clock(self) -> None:
        timeline = build_adaptive_timeline(
            [_event("a", 1000, 2000, 800)],
            [{"utteranceId": "a", "voiceEventId": "a", "startMs": 1000, "endMs": 2000, "translatedTextVi": "Delivered a"}],
            [_plan("a", 1000, 2000)],
            source_duration_ms=2500,
        )
        voice = timeline["voiceEvents"][0]
        cue = timeline["subtitleCues"][0]
        self.assertEqual(voice["scheduledStartMs"], cue["scheduledStartMs"])
        self.assertEqual(voice["scheduledEndMs"], cue["scheduledEndMs"])

    def test_ffmpeg_command_uses_piecewise_video_and_dub_only_audio(self) -> None:
        timeline = build_adaptive_timeline(
            [_event("a", 0, 1000, 900)], [], [_plan("a", 0, 1000)], source_duration_ms=1200,
        )
        command = build_adaptive_ffmpeg_command(
            Path("source.mp4"), Path("output.mp4"), Path("subtitles.ass"),
            [(Path("audio/a.mp3"), 0)], timeline=timeline, width=720, height=960,
        )
        filter_graph = command[command.index("-filter_complex") + 1]
        self.assertIn("split=1", filter_graph)
        self.assertIn("trim=", filter_graph)
        self.assertIn("setpts=", filter_graph)
        self.assertIn("concat=", filter_graph)
        self.assertIn("amix=", filter_graph)
        self.assertEqual("libx264", command[command.index("-c:v") + 1])
        self.assertEqual("yuv420p", command[command.index("-pix_fmt") + 1])
        self.assertEqual("aac", command[command.index("-c:a") + 1])
        self.assertEqual("48000", command[command.index("-ar") + 1])
        self.assertEqual("2", command[command.index("-ac") + 1])

    def test_rejects_audio_that_exceeds_mapped_delivery_window(self) -> None:
        with self.assertRaises(AdaptiveRenderError):
            build_adaptive_timeline(
                [_event("a", 0, 1000, 1300)], [], [_plan("a", 0, 1000)], source_duration_ms=1200,
            )


if __name__ == "__main__":
    unittest.main()
