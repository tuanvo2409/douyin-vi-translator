from __future__ import annotations

import unittest

from timeline_mapper import TimelineMappingError, build_timeline


def _span(start: int, end: int, speed: float = 1.0, action: str = "KEEP", **extra: object) -> dict[str, object]:
    value: dict[str, object] = {
        "sourceStartMs": start,
        "sourceEndMs": end,
        "visualSpeed": speed,
        "action": action,
    }
    value.update(extra)
    return value


class TimelineMapperTests(unittest.TestCase):
    def test_mapping_is_monotonic(self) -> None:
        timeline = build_timeline([_span(200, 400, 0.5, "SLOW_VISUAL")], source_duration_ms=1000)
        starts = [segment["outputStartMs"] for segment in timeline.to_dict()["segments"]]
        ends = [segment["outputEndMs"] for segment in timeline.to_dict()["segments"]]
        self.assertEqual(starts, sorted(starts))
        self.assertEqual(ends, sorted(ends))

    def test_source_spans_have_no_overlap(self) -> None:
        segments = build_timeline([_span(100, 300)], source_duration_ms=500).segments
        for previous, current in zip(segments, segments[1:]):
            self.assertEqual(previous.source_end_ms, current.source_start_ms)

    def test_full_visual_coverage_fills_source_gaps(self) -> None:
        timeline = build_timeline([_span(200, 400, 0.5, "SLOW_VISUAL")], source_duration_ms=600)
        self.assertEqual((0, 600), (timeline.segments[0].source_start_ms, timeline.segments[-1].source_end_ms))
        self.assertEqual([200, 400], [segment.source_start_ms for segment in timeline.segments[1:]])

    def test_one_x_mapping_preserves_duration(self) -> None:
        timeline = build_timeline([_span(0, 1000)], source_duration_ms=1000)
        segment = timeline.segments[0]
        self.assertEqual(1000, segment.output_end_ms - segment.output_start_ms)

    def test_slowdown_expands_expected_output_duration(self) -> None:
        timeline = build_timeline([_span(0, 1000, 0.5, "SLOW_VISUAL")], source_duration_ms=1000)
        self.assertEqual(2000, timeline.output_duration_ms)

    def test_voice_and_subtitle_use_same_transform(self) -> None:
        timeline = build_timeline([_span(0, 1000, 0.5, "SLOW_VISUAL")], source_duration_ms=1000)
        voice_start = timeline.map_timestamp(250)
        subtitle_start, subtitle_end = timeline.map_interval(250, 500)
        self.assertEqual(voice_start, subtitle_start)
        self.assertEqual(500, subtitle_end - subtitle_start)

    def test_overlapping_compatible_requests_merge_with_strongest_slowdown(self) -> None:
        timeline = build_timeline(
            [_span(100, 500, 0.8, "SLOW_VISUAL"), _span(400, 700, 0.6, "SLOW_VISUAL")],
            source_duration_ms=1000,
        )
        covered = [segment for segment in timeline.segments if segment.source_start_ms == 100]
        self.assertEqual(0.6, covered[0].speed)
        self.assertEqual(700, covered[0].source_end_ms)

    def test_incompatible_overlap_fails_closed(self) -> None:
        with self.assertRaises(TimelineMappingError):
            build_timeline(
                [_span(100, 500, 0.8, "SLOW_VISUAL", compatible=False), _span(400, 700, 0.6, "SLOW_VISUAL")],
                source_duration_ms=1000,
            )

    def test_final_duration_is_deterministic(self) -> None:
        spans = [_span(0, 1000), _span(1500, 2000, 0.75, "SLOW_VISUAL")]
        first = build_timeline(spans, source_duration_ms=2500).to_dict()
        second = build_timeline(spans, source_duration_ms=2500).to_dict()
        self.assertEqual(first, second)

    def test_timestamp_mapping_preserves_segment_boundaries(self) -> None:
        timeline = build_timeline([_span(0, 1000), _span(1000, 2000, 0.5, "SLOW_VISUAL")], source_duration_ms=2000)
        self.assertEqual(1000, timeline.map_timestamp(1000))
        self.assertEqual(2000, timeline.map_timestamp(1500))

    def test_invalid_source_overlap_is_rejected_when_not_mergeable(self) -> None:
        with self.assertRaises(TimelineMappingError):
            build_timeline([_span(300, 200)], source_duration_ms=500)


if __name__ == "__main__":
    unittest.main()
