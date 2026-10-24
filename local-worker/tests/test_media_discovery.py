from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


WORKER_DIR = Path(__file__).resolve().parents[1]
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from dubvi_worker import scan_multi_channel_raw


class MediaDiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.media_dir = Path(self.temp_dir.name) / "media"
        self.output_dir = Path(self.temp_dir.name) / "output"
        self.model_dir = Path(self.temp_dir.name) / "models"
        self.log_dir = Path(self.temp_dir.name) / "logs"
        self.env = {
            "DUBVI_MEDIA_DIR": str(self.media_dir),
            "DUBVI_OUTPUT_DIR": str(self.output_dir),
            "DUBVI_MODEL_DIR": str(self.model_dir),
            "DUBVI_LOG_DIR": str(self.log_dir),
        }

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_video(self, directory: Path, name: str) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        video = directory / name
        video.write_bytes(b"test-video")
        return video

    def test_default_discovery_uses_configured_media_directory(self) -> None:
        self._write_video(self.media_dir, "manual.mp4")

        with patch.dict(os.environ, self.env, clear=False):
            items = scan_multi_channel_raw()

        self.assertEqual(1, len(items))
        self.assertEqual(self.media_dir.resolve(), items[0]["video_path"].parent.resolve())
        self.assertNotIn("vmath", str(items[0]["video_path"]).lower())

    def test_complete_versioned_sidecar_preserves_handoff_metadata(self) -> None:
        profile_dir = self.media_dir / "channel-a"
        video = self._write_video(profile_dir, "processed_clip.mp4")
        sidecar = profile_dir / f"{video.stem}.meta.json"
        sidecar.write_text(
            json.dumps({
                "handoff_schema_version": 1,
                "handoff_status": "complete",
                "channel_profile": "channel-a",
                "target_platform": "shorts",
                "original_filename": "source.mp4",
                "output_filename": video.name,
                "original_md5": "original-md5",
                "output_md5": "output-md5",
                "vpdq_similarity_percent": 12.5,
                "vpdq_status": "PASSED",
                "duration_seconds": 42.0,
                "zoom_factor": 1.03,
                "encoder_used": "libx264",
                "layout_mode": "crop_fill",
            }),
            encoding="utf-8",
        )

        items = scan_multi_channel_raw([self.media_dir])

        self.assertEqual(1, len(items))
        item = items[0]
        self.assertEqual("channel-a", item["channel_profile"])
        self.assertEqual("shorts", item["target_platform"])
        self.assertEqual(12.5, item["vpdq_similarity"])
        self.assertEqual(42.0, item["duration_seconds"])
        self.assertEqual(1.03, item["zoom_factor"])
        self.assertEqual(1, item["handoff_schema_version"])
        self.assertEqual("complete", item["handoff_status"])

    def test_incomplete_versioned_sidecar_is_not_discovered(self) -> None:
        video = self._write_video(self.media_dir / "channel-a", "processed_clip.mp4")
        (video.parent / f"{video.stem}.meta.json").write_text(
            json.dumps({"handoff_schema_version": 1, "handoff_status": "staging"}),
            encoding="utf-8",
        )

        self.assertEqual([], scan_multi_channel_raw([self.media_dir]))

    def test_legacy_manual_video_without_sidecar_remains_discoverable(self) -> None:
        video = self._write_video(self.media_dir / "manual-channel", "manual.mp4")

        items = scan_multi_channel_raw([self.media_dir])

        self.assertEqual(1, len(items))
        self.assertEqual(video, items[0]["video_path"])
        self.assertFalse(items[0]["has_meta"])
        self.assertIsNone(items[0]["handoff_schema_version"])

    def test_canonical_p1c_reup_media_without_sidecar_is_not_discovered(self) -> None:
        self._write_video(
            self.media_dir / "channel-a",
            "reup-22222222-2222-4222-8222-222222222222.mp4",
        )

        self.assertEqual([], scan_multi_channel_raw([self.media_dir]))

    def test_canonical_p1c_reup_v2_handoff_is_not_discovered(self) -> None:
        profile_dir = self.media_dir / "channel-a"
        video = self._write_video(
            profile_dir,
            "reup-22222222-2222-4222-8222-222222222222.mp4",
        )
        (profile_dir / f"{video.stem}.meta.json").write_text(
            json.dumps({"handoff_schema_version": 2, "handoff_status": "complete"}),
            encoding="utf-8",
        )

        self.assertEqual([], scan_multi_channel_raw([self.media_dir]))


if __name__ == "__main__":
    unittest.main()
