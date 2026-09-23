from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "local-worker"))

from dubvi_worker import whisper_runtime_options  # noqa: E402


class ExecutionLimitTests(unittest.TestCase):
    def test_cpu_whisper_uses_control_plane_profile_limits(self) -> None:
        with patch.dict(
            os.environ,
            {
                "DUBVI_WHISPER_COMPUTE_TYPE": "int8",
                "DUBVI_WHISPER_CPU_THREADS": "2",
                "DUBVI_WHISPER_DECODE_WORKERS": "1",
            },
            clear=False,
        ):
            self.assertEqual(
                {"compute_type": "int8", "cpu_threads": 2, "num_workers": 1},
                whisper_runtime_options("cpu"),
            )


if __name__ == "__main__":
    unittest.main()
