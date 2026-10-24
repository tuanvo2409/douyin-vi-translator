from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from piper_tts import (
    PiperTtsError,
    PiperTtsRuntime,
    _runtime_for_model,
    speed_length_scale,
    write_wav,
)


class PiperTtsTests(unittest.TestCase):
    def test_speed_values_are_bounded(self) -> None:
        self.assertEqual(1.0, speed_length_scale("normal"))
        self.assertEqual(0.6, speed_length_scale("fast"))
        self.assertEqual(0.4, speed_length_scale("very_fast"))
        with self.assertRaises(ValueError):
            speed_length_scale("turbo")

    def test_missing_model_fails_with_setup_path(self) -> None:
        with self.assertRaises(PiperTtsError) as raised:
            PiperTtsRuntime(Path("missing-pretrained_vi.onnx"))
        self.assertIn("pretrained_vi.onnx", str(raised.exception))

    def test_missing_model_config_fails_before_runtime_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "pretrained_vi.onnx"
            model.write_bytes(b"not-an-onnx-model")
            with self.assertRaises(PiperTtsError) as raised:
                PiperTtsRuntime(model)
        self.assertIn("pretrained_vi.onnx.json", str(raised.exception))

    def test_write_wav_produces_mono_22050_hz_pcm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "voice.wav"
            write_wav(target, [0.0, 0.5, -0.5])
            with wave.open(str(target), "rb") as audio:
                self.assertEqual(1, audio.getnchannels())
                self.assertEqual(2, audio.getsampwidth())
                self.assertEqual(22050, audio.getframerate())
                self.assertEqual(3, audio.getnframes())

    def test_runtime_cache_reuses_one_runtime_per_model_path(self) -> None:
        model = Path("C:/models/pretrained_vi.onnx")
        first = object()
        with patch("piper_tts.PiperTtsRuntime", return_value=first) as constructor:
            self.assertIs(first, _runtime_for_model(model))
            self.assertIs(first, _runtime_for_model(model))
        constructor.assert_called_once_with(model.resolve())


if __name__ == "__main__":
    unittest.main()
