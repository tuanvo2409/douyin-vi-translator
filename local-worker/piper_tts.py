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
        self._input_names = {item.name for item in self._session.get_inputs()}

    def _load_session(self):
        try:
            import onnxruntime
        except ImportError as exc:
            raise PiperTtsError(
                "Thiếu onnxruntime. Chạy `pip install -r requirements.txt`."
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
            return onnxruntime.InferenceSession(str(self.model_path), sess_options=options, providers=["CPUExecutionProvider"])
        except Exception as exc:
            raise PiperTtsError(f"Không thể load Piper ONNX model: {self.model_path}") from exc

    def _phonemize(self, text: str) -> list[list[str]]:
        try:
            from piper_phonemize import phonemize_codepoints, phonemize_espeak
        except ImportError as exc:
            raise PiperTtsError(
                "Thiếu piper-phonemize. Chạy `pip install -r requirements.txt`."
            ) from exc
        phoneme_type = str(self.config.get("phoneme_type") or "text").lower()
        if phoneme_type == "text":
            return phonemize_codepoints(text)
        if phoneme_type == "espeak":
            voice = str((self.config.get("espeak") or {}).get("voice") or "vi")
            return phonemize_espeak(text, voice)
        raise PiperTtsError(f"Piper model dùng phoneme_type không hỗ trợ: {phoneme_type}")

    def _phonemes_to_ids(self, phonemes: list[str]) -> list[int]:
        id_map = self.config.get("phoneme_id_map")
        if not isinstance(id_map, dict):
            raise PiperTtsError("Piper model config thiếu phoneme_id_map")
        bos = id_map.get("^")
        pad = id_map.get("_")
        eos = id_map.get("$")
        if not isinstance(bos, list) or not isinstance(pad, list) or not isinstance(eos, list):
            raise PiperTtsError("Piper model config thiếu token BOS/PAD/EOS")
        result = list(bos)
        for phoneme in phonemes:
            ids = id_map.get(phoneme)
            if isinstance(ids, list):
                result.extend(ids)
                result.extend(pad)
        result.extend(eos)
        return result

    def synthesize(self, text: str, target: Path, speed: str) -> None:
        text = str(text).strip()
        if not text:
            raise PiperTtsError("Piper không nhận được text rỗng")
        try:
            import numpy as np
        except ImportError as exc:
            raise PiperTtsError("Thiếu numpy cho Piper ONNX runtime") from exc
        groups = self._phonemize(text)
        phoneme_ids = []
        for group in groups:
            phoneme_ids.extend(self._phonemes_to_ids(group) + [0, 0, 0])
        if not phoneme_ids:
            raise PiperTtsError("Piper không tạo được phoneme từ text")
        input_ids = np.expand_dims(np.asarray(phoneme_ids, dtype=np.int64), 0)
        input_lengths = np.asarray([input_ids.shape[1]], dtype=np.int64)
        scales = np.asarray([0.667, speed_length_scale(speed), 0.8], dtype=np.float32)
        inputs = {
            "input": input_ids,
            "input_lengths": input_lengths,
            "scales": scales,
        }
        if "sid" in self._input_names:
            speakers = int(self.config.get("num_speakers") or 1)
            inputs["sid"] = np.asarray([0], dtype=np.int64) if speakers > 1 else None
        try:
            audio = self._session.run(None, inputs)[0]
            samples = np.asarray(audio).squeeze().tolist()
            sample_rate = int((self.config.get("audio") or {}).get("sample_rate") or 22050)
            write_wav(Path(target), samples, sample_rate)
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
