from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dubvi_worker import Settings
from p1c_processing import build_canonical_runtime_config


class TtsProviderConfigTests(unittest.TestCase):
    def _environment(self, root: Path, **overrides: str) -> dict[str, str]:
        values = {
            "USERPROFILE": str(root),
            "HOME": str(root),
            "DUBVI_MEDIA_DIR": str(root / "media"),
            "DUBVI_OUTPUT_DIR": str(root / "output"),
            "DUBVI_MODEL_DIR": str(root / "models"),
            "DUBVI_LOG_DIR": str(root / "logs"),
        }
        values.update(overrides)
        return values

    def test_settings_default_to_cpu_piper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("dubvi_worker.load_dotenv", return_value=False), patch.dict(
                os.environ, self._environment(root), clear=True
            ):
                settings = Settings.from_env()

        self.assertEqual("piper", settings.tts_provider)
        self.assertEqual("fast", settings.piper_speed)
        self.assertTrue(str(settings.piper_model).endswith("piper-vietnamese\\pretrained_vi.onnx"))

    def test_explicit_capcut_provider_remains_available(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("dubvi_worker.load_dotenv", return_value=False), patch.dict(
                os.environ,
                self._environment(root, DUBVI_TTS_PROVIDER="capcut"),
                clear=True,
            ):
                self.assertEqual("capcut", Settings.from_env().tts_provider)

    def test_invalid_provider_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("dubvi_worker.load_dotenv", return_value=False), patch.dict(
                os.environ,
                self._environment(root, DUBVI_TTS_PROVIDER="remote"),
                clear=True,
            ):
                with self.assertRaises(ValueError):
                    Settings.from_env()

    def test_invalid_piper_speed_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("dubvi_worker.load_dotenv", return_value=False), patch.dict(
                os.environ,
                self._environment(root, DUBVI_PIPER_SPEED="turbo"),
                clear=True,
            ):
                with self.assertRaises(ValueError):
                    Settings.from_env()

    def test_runtime_config_uses_explicit_provider_without_changing_persona(self) -> None:
        document = {
            "channel_slug": "demo-channel",
            "target_platform": "tiktok",
            "localization_profile": "page_goc_tro_bat_on",
        }
        with patch.dict(os.environ, {"DUBVI_TTS_PROVIDER": "piper"}, clear=False):
            runtime = build_canonical_runtime_config(document)

        self.assertEqual("piper", runtime["ttsProvider"])
        self.assertEqual(
            "sarcastic_dramatic_roommate_landlord_student_housing",
            runtime["persona"],
        )


if __name__ == "__main__":
    unittest.main()
