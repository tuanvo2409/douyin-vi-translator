"""Small local Vietnamese Piper/ONNX runtime used by the Translator worker."""

from __future__ import annotations

import json
import os
import threading
import wave
from array import array
from pathlib import Path
from typing import Iterable


_SPEED_LENGTH_SCALES = {"normal": 1.0, "fast": 0.6, "very_fast": 0.4}
_RUNTIME_CACHE: dict[Path, "PiperTtsRuntime"] = {}
_RUNTIME_LOCK = threading.Lock()


class PiperTtsError(RuntimeError):
    """Piper cannot safely produce the requested local audio."""


def speed_length_scale(speed: str) -> float:
    value = str(speed).strip().lower()
    try:
        return _SPEED_LENGTH_SCALES[value]
    except KeyError as exc:
        raise ValueError(f"Unsupported Piper speed: {speed!r}") from exc


def write_wav(target: Path, samples: Iterable[float], sample_rate: int = 22050) -> None:
    """Write normalized mono float samples as signed 16-bit PCM WAV."""

    values = [float(sample) for sample in samples]
    if not values:
        raise PiperTtsError("Piper returned empty audio")
    peak = max(abs(value) for value in values)
    scale = 32767.0 / max(0.01, peak)
    pcm = array(
        "h",
        (
            max(-32768, min(32767, int(round(value * scale))))
            for value in values
        ),
    )
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_suffix(f"{target.suffix}.part")
    try:
        with wave.open(str(staging), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(int(sample_rate))
            output.writeframes(pcm.tobytes())
        staging.replace(target)
    finally:
        staging.unlink(missing_ok=True)


class PiperTtsRuntime:
    """One loaded ONNX model and its Vietnamese phoneme configuration."""

    def __init__(self, model_path: Path):
        self.model_path = Path(model_path).expanduser().resolve()
        self.config_path = Path(f"{self.model_path}.json")
        if not self.model_path.is_file():
            raise PiperTtsError(f"Piper model is missing: {self.model_path}")
        if not self.config_path.is_file():
            raise PiperTtsError(f"Piper model config is missing: {self.config_path}")
        try:
            self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise PiperTtsError(f"Piper model config is invalid: {self.config_path}") from exc
        self._session = self._load_session()

    def _load_session(self):
        try:
            import onnxruntime
            from piper.config import PiperConfig
            from piper.voice import PiperVoice
        except ImportError as exc:
            raise PiperTtsError(
                "Thiếu piper-tts/onnxruntime. Chạy `pip install -r requirements.txt`."
            ) from exc
        options = onnxruntime.SessionOptions()
        try:
            options.intra_op_num_threads = max(
                1, int(os.getenv("DUBVI_TTS_CPU_THREADS", os.getenv("DUBVI_WHISPER_CPU_THREADS", "2")))
            )
        except ValueError as exc:
            raise PiperTtsError("DUBVI_TTS_CPU_THREADS phải là số nguyên dương") from exc
        options.inter_op_num_threads = 1
        try:
            config = PiperConfig.from_dict(self.config)
            session = onnxruntime.InferenceSession(
                str(self.model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            return PiperVoice(session=session, config=config)
        except Exception as exc:
            raise PiperTtsError(f"Không thể load Piper ONNX model: {self.model_path}") from exc

    def synthesize(self, text: str, target: Path, speed: str) -> None:
        text = str(text).strip()
        if not text:
            raise PiperTtsError("Piper không nhận được text rỗng")
        try:
            import numpy as np
        except ImportError as exc:
            raise PiperTtsError("Thiếu numpy cho Piper ONNX runtime") from exc
        try:
            from piper.config import SynthesisConfig

            chunks = list(
                self._session.synthesize(
                    text,
                    SynthesisConfig(
                        length_scale=speed_length_scale(speed),
                        noise_scale=0.667,
                        noise_w_scale=0.8,
                        normalize_audio=True,
                    ),
                )
            )
            if not chunks:
                raise PiperTtsError("Piper không tạo được audio chunk")
            samples = np.concatenate([chunk.audio_float_array for chunk in chunks])
            write_wav(Path(target), samples.tolist(), chunks[0].sample_rate)
        except PiperTtsError:
            raise
        except Exception as exc:
            raise PiperTtsError(f"Piper inference thất bại cho: {text[:80]!r}") from exc


def _runtime_for_model(model_path: Path) -> PiperTtsRuntime:
    resolved = Path(model_path).expanduser().resolve()
    with _RUNTIME_LOCK:
        runtime = _RUNTIME_CACHE.get(resolved)
        if runtime is None:
            runtime = PiperTtsRuntime(resolved)
            _RUNTIME_CACHE[resolved] = runtime
        return runtime


def clear_runtime_cache() -> None:
    with _RUNTIME_LOCK:
        _RUNTIME_CACHE.clear()
