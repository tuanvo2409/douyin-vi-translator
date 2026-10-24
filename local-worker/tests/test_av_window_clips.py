from __future__ import annotations

import base64
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import av_analyzer
from llm_transport import ProviderError


def _document(start_ms: int, end_ms: int) -> dict[str, object]:
    return {
        "windowStartMs": start_ms,
        "windowEndMs": end_ms,
        "scene": {"summary": "scene", "tone": "neutral", "register": "spoken"},
        "utterances": [{
            "startMs": start_ms + 100,
            "endMs": min(start_ms + 900, end_ms),
            "spokenZh": "原文",
            "subtitleZh": "原文",
            "speakerRef": "speaker-1",
            "intent": "statement",
            "emotion": "neutral",
            "confidence": "HIGH",
        }],
        "visualOnly": [],
    }


class _Transport:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.payloads: list[dict[str, object]] = []
        self.last_call: dict[str, object] = {"transportAttempts": 1}

    def call_raw(self, payload: dict[str, object]) -> object:
        self.payloads.append(payload)
        self.last_call = {"transportAttempts": 1, "httpStatus": 200, "finalTransportStatus": "HTTP_200"}
        return self.responses.pop(0)


class AVWindowClipTests(unittest.TestCase):
    def _media(self, directory: str) -> Path:
        path = Path(directory) / "source.mp4"
        path.write_bytes(b"whole source bytes that must not be uploaded")
        return path

    def test_parent_window_upload_uses_bounded_clip_bytes_and_absolute_prompt(self) -> None:
        transport = _Transport([_document(12_000, 27_000)])
        extracted: list[tuple[int, int]] = []

        def clip(_path: Path, start_ms: int, end_ms: int) -> bytes:
            extracted.append((start_ms, end_ms))
            return f"clip:{start_ms}:{end_ms}".encode()

        with tempfile.TemporaryDirectory() as directory:
            result = av_analyzer.GeminiAVAnalyzer(
                transport=transport,
                allow_cloud_upload=True,
                clip_extractor=clip,
            ).analyze_media(self._media(directory), start_ms=12_000, end_ms=27_000)

        self.assertEqual(_document(12_000, 27_000), result)
        self.assertEqual([(12_000, 27_000)], extracted)
        payload = transport.payloads[0]
        prompt = payload["contents"][0]["parts"][0]["text"]
        self.assertIn("12000-27000 ms", prompt)
        encoded = payload["contents"][0]["parts"][1]["inline_data"]["data"]
        self.assertEqual(b"clip:12000:27000", base64.b64decode(encoded))
        self.assertNotIn(b"whole source", base64.b64decode(encoded))

    def test_schema_repair_reuses_one_clip_for_both_logical_attempts(self) -> None:
        invalid = _document(0, 1_000)
        invalid["scene"] = "bad-shape"
        transport = _Transport([invalid, _document(0, 1_000)])
        extracted: list[tuple[int, int]] = []

        with tempfile.TemporaryDirectory() as directory:
            analyzer = av_analyzer.GeminiAVAnalyzer(
                transport=transport,
                allow_cloud_upload=True,
                clip_extractor=lambda _path, start, end: extracted.append((start, end)) or b"one bounded clip",
            )
            analyzer.analyze_media(self._media(directory), start_ms=0, end_ms=1_000)

        self.assertEqual([(0, 1_000)], extracted)
        self.assertEqual(2, len(transport.payloads))
        first = transport.payloads[0]["contents"][0]["parts"][1]["inline_data"]["data"]
        second = transport.payloads[1]["contents"][0]["parts"][1]["inline_data"]["data"]
        self.assertEqual(first, second)

    def test_subdivision_uploads_each_child_clip_with_absolute_child_window(self) -> None:
        invalid = _document(0, 30_000)
        invalid["scene"] = "bad-shape"
        child_windows = av_analyzer.subdivision_windows(0, 30_000)
        transport = _Transport([invalid, invalid] + [_document(start, end) for start, end in child_windows])
        extracted: list[tuple[int, int]] = []

        with tempfile.TemporaryDirectory() as directory:
            result = av_analyzer.analyze_media_windows_with_subdivision(
                av_analyzer.GeminiAVAnalyzer(
                    transport=transport,
                    allow_cloud_upload=True,
                    clip_extractor=lambda _path, start, end: extracted.append((start, end)) or f"clip:{start}:{end}".encode(),
                ),
                self._media(directory),
                30_000,
            )

        self.assertEqual([(0, 30_000)] + child_windows, extracted)
        self.assertEqual(len(child_windows) + 2, len(transport.payloads))
        for (start, end), payload in zip([(0, 30_000), (0, 30_000)] + child_windows, transport.payloads):
            prompt = payload["contents"][0]["parts"][0]["text"]
            self.assertIn(f"{start}-{end}", prompt)
        self.assertEqual("AV_WINDOW_SUBDIVISION_SUCCEEDED", result["diagnostics"][0]["status"])

    def test_live_transport_requires_explicit_cloud_upload_opt_in(self) -> None:
        transport = _Transport([_document(0, 1_000)])
        analyzer = av_analyzer.GeminiAVAnalyzer(
            transport=transport,
            allow_cloud_upload=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ProviderError, "CLOUD_AV_UPLOAD_CONSENT_REQUIRED"):
                analyzer.analyze_media(self._media(directory), start_ms=0, end_ms=1_000)
        self.assertEqual([], transport.payloads)
        self.assertEqual("AV_CLOUD_UPLOAD_CONSENT_REQUIRED", analyzer.last_recovery["finalStatus"])
        self.assertLessEqual(len(json.dumps(analyzer.last_recovery)), av_analyzer.MAX_DIAGNOSTIC_BYTES)
        self.assertNotIn("whole source", json.dumps(analyzer.last_recovery))

    def test_clip_exactly_at_cloud_cap_is_read_successfully(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            source.write_bytes(b"source")

            def fake_run(command, **_kwargs):
                Path(command[-1]).write_bytes(b"x" * av_analyzer.MAX_CLOUD_CLIP_BYTES)

            with patch("av_analyzer.subprocess.run", side_effect=fake_run):
                data = av_analyzer._extract_bounded_clip_bytes(source, 0, 1_000)

        self.assertEqual(av_analyzer.MAX_CLOUD_CLIP_BYTES, len(data))

    def test_oversize_extracted_clip_is_rejected_before_read_bytes(self) -> None:
        transport = _Transport([_document(0, 1_000)])
        analyzer = av_analyzer.GeminiAVAnalyzer(transport=transport, allow_cloud_upload=True)
        with tempfile.TemporaryDirectory() as directory:
            source = self._media(directory)

            def fake_run(command, **_kwargs):
                Path(command[-1]).write_bytes(b"x" * (av_analyzer.MAX_CLOUD_CLIP_BYTES + 1))

            with patch("av_analyzer.subprocess.run", side_effect=fake_run), patch.object(
                Path, "read_bytes", side_effect=AssertionError("oversize clip must not be read")
            ) as read_bytes:
                with self.assertRaisesRegex(av_analyzer.AVClipExtractionError, "AV_CLIP_TOO_LARGE"):
                    analyzer.analyze_media(source, start_ms=0, end_ms=1_000)

        read_bytes.assert_not_called()
        self.assertEqual([], transport.payloads)
        diagnostic = json.dumps(analyzer.last_recovery)
        self.assertEqual("AV_CLIP_TOO_LARGE", analyzer.last_recovery["finalStatus"])
        self.assertLessEqual(len(diagnostic), av_analyzer.MAX_DIAGNOSTIC_BYTES)
        self.assertNotIn("whole source bytes", diagnostic)

    def test_oversize_injected_clip_is_rejected_before_transport_and_base64(self) -> None:
        transport = _Transport([_document(0, 1_000)])
        analyzer = av_analyzer.GeminiAVAnalyzer(
            transport=transport,
            allow_cloud_upload=True,
            clip_extractor=lambda _path, _start, _end: b"x" * (av_analyzer.MAX_CLOUD_CLIP_BYTES + 1),
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "av_analyzer.base64.b64encode", side_effect=AssertionError("oversize clip must not be encoded")
        ) as b64encode:
            with self.assertRaisesRegex(av_analyzer.AVClipExtractionError, "AV_CLIP_TOO_LARGE"):
                analyzer.analyze_media(self._media(directory), start_ms=0, end_ms=1_000)

        b64encode.assert_not_called()
        self.assertEqual([], transport.payloads)
        diagnostic = json.dumps(analyzer.last_recovery)
        self.assertEqual("AV_CLIP_TOO_LARGE", analyzer.last_recovery["finalStatus"])
        self.assertLessEqual(len(diagnostic), av_analyzer.MAX_DIAGNOSTIC_BYTES)
        self.assertNotIn("whole source bytes", diagnostic)

    def test_ffmpeg_clip_cleanup_only_removes_verified_clip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            source.write_bytes(b"source")

            def fake_run(command, **_kwargs):
                Path(command[-1]).write_bytes(b"bounded mp4")

            with patch("av_analyzer.subprocess.run", side_effect=fake_run) as run:
                data = av_analyzer._extract_bounded_clip_bytes(source, 5_000, 20_000)

        self.assertEqual(b"bounded mp4", data)
        command = run.call_args.args[0]
        self.assertIn("-ss", command)
        self.assertIn("-t", command)
        self.assertIn("5.000", command)
        self.assertIn("15.000", command)
        self.assertLess(command.index("-i"), command.index("-ss"))
        self.assertIn("libx264", command)

    def test_stalled_ffmpeg_times_out_and_removes_partial_clip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            source.write_bytes(b"source")
            workspace = root / "owned-ffmpeg-temp"
            workspace.mkdir()

            def stalled_run(command, **_kwargs):
                Path(command[-1]).write_bytes(b"partial clip")
                raise subprocess.TimeoutExpired(command[0], 1)

            with patch("av_analyzer.tempfile.mkdtemp", return_value=str(workspace)), patch(
                "av_analyzer.subprocess.run", side_effect=stalled_run
            ) as run:
                with self.assertRaisesRegex(av_analyzer.AVClipExtractionError, "timeout"):
                    av_analyzer._extract_bounded_clip_bytes(source, 0, 1_000)

            self.assertTrue(run.call_args.kwargs["timeout"] > 0)
            self.assertFalse(workspace.exists())

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg/FFprobe required")
    def test_long_gop_clip_is_accurately_trimmed_to_requested_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "long-gop.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc=size=160x90:rate=10",
                    "-f", "lavfi", "-i", "sine=frequency=880:sample_rate=8000",
                    "-t", "12", "-c:v", "libx264", "-g", "80", "-keyint_min", "80",
                    "-sc_threshold", "0", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
                    str(source),
                ],
                check=True,
                capture_output=True,
            )
            clip = av_analyzer._extract_bounded_clip_bytes(source, 4_250, 7_250)
            clip_path = root / "window.mp4"
            clip_path.write_bytes(clip)
            probe = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1", str(clip_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertLessEqual(float(probe.stdout.strip()), 3.15)


if __name__ == "__main__":
    unittest.main()
