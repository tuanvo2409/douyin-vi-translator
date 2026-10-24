from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from p1c_config import P1CConfigurationError, load_p1c_settings


class P1CConfigTests(unittest.TestCase):
    def test_loading_is_side_effect_free_and_requires_canonical_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            values = {
                "DUBVI_TRANSLATOR_JOB_DIR": str(root / "jobs"),
                "DUBVI_MEDIA_DIR": str(root / "media"),
                "DUBVI_ENGINE_STATUS_DIR": str(root / "status"),
                "DUBVI_OUTPUT_DIR": str(root / "output"),
            }
            settings = load_p1c_settings(values)
            self.assertEqual(root / "jobs", settings.translator_job_dir)
            self.assertFalse((root / "jobs").exists())
            with self.assertRaises(P1CConfigurationError):
                load_p1c_settings({key: value for key, value in values.items() if key != "DUBVI_MEDIA_DIR"})

    def test_control_plane_root_is_only_required_for_canonical_execution(self) -> None:
        values = {
            "DUBVI_TRANSLATOR_JOB_DIR": "jobs",
            "DUBVI_MEDIA_DIR": "media",
            "DUBVI_ENGINE_STATUS_DIR": "status",
            "DUBVI_OUTPUT_DIR": "output",
        }
        self.assertIsNone(load_p1c_settings(values).control_plane_root)
        with self.assertRaises(P1CConfigurationError):
            load_p1c_settings(values, require_control_plane=True)


if __name__ == "__main__":
    unittest.main()
