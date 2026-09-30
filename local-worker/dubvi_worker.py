#!/usr/bin/env python3
"""DUBVI local media worker.

The worker never accepts a public inbound connection and never uploads input
video files. It only polls the signed-out public worker RPC endpoints with a
pairing token, processes jobs using local CPU/GPU resources, and reports JSON
metadata plus local output paths back to the web control plane.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import requests
try:
    from dotenv import load_dotenv
except ImportError:  # Allows `verify` to explain the environment before the virtualenv is complete.
    def load_dotenv() -> bool:
        return False

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
_CANONICAL_REUP_MEDIA_RE = re.compile(
    r"^reup-[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\.mp4$"
)
_nllb_components: tuple[Any, Any] | None = None
_whisper_model: Any | None = None          # Cache WhisperModel giữa các job
_whisper_model_key: tuple[str, str, str | None] | None = None
_rapidocr_engine: Any | None = None        # Cache RapidOCR engine giữa các segment


class _OfflineModelLoadError(RuntimeError):
    """Sanitized error for a model that is unavailable in the local cache."""


TTS_PROVIDERS = frozenset({"piper", "capcut", "edge"})
PIPER_SPEEDS = frozenset({"normal", "fast", "very_fast"})

# Giới hạn thread cho numpy/torch/onnxruntime ngay khi import — phải set TRƯỚC khi import các lib đó
# Trên i3-1115G4 (2 core / 4 thread) dùng 2 thread là optimal, tránh context-switch thrash
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def validate_tts_provider(value: str) -> str:
    provider = str(value).strip().lower()
    if provider not in TTS_PROVIDERS:
        raise ValueError(f"DUBVI_TTS_PROVIDER must be one of: {', '.join(sorted(TTS_PROVIDERS))}")
    return provider


def validate_piper_speed(value: str) -> str:
    speed = str(value).strip().lower()
    if speed not in PIPER_SPEEDS:
        raise ValueError(f"DUBVI_PIPER_SPEED must be one of: {', '.join(sorted(PIPER_SPEEDS))}")
    return speed


DOUYIN_OUTRO_CTA_KEYWORDS = [
    "住小帮", "设计我家", "左下角", "搜小区", "小区名",
    "免费生成", "专属你家", "点赞关注", "关注我", "主页",
    "私信", "点击链接", "点击下方", "下载", "一键匹配",
    "户型图", "免费领取", "同款设计", "立即预约", "小房子"
]


def detect_outro_cta(
    segments: list[dict[str, Any]],
    video_duration_s: float,
    tail_window_s: float = 20.0
) -> tuple[int | None, str | None, int | None]:
    """Smart Outro CTA Trimmer: detect Douyin outro promo CTA within tail_window_s."""
    if not segments or video_duration_s <= 0:
        return None, None, None

    threshold_ms = int(max(0.0, video_duration_s - tail_window_s) * 1000)
    first_cta_idx = None
    matched_keyword = None

    for i in range(len(segments) - 1, -1, -1):
        seg = segments[i]
        st_ms = seg.get("startMs", 0)
        if st_ms < threshold_ms:
            break

        text = (seg.get("asrTextZh") or seg.get("chinese_text") or seg.get("text") or "").strip()
        for kw in DOUYIN_OUTRO_CTA_KEYWORDS:
            if kw in text:
                first_cta_idx = i
                matched_keyword = kw
                break

    if first_cta_idx is not None:
        cutoff_ms = segments[first_cta_idx]["startMs"]
        return cutoff_ms, matched_keyword, first_cta_idx

    return None, None, None


@dataclass
class Settings:
    api_base: str
    token: str
    media_dir: Path
    output_dir: Path
    model_dir: Path
    log_dir: Path
    device: str
    whisper_model: str
    poll_seconds: int
    tts_provider: str        # "piper", "edge" hoặc "capcut"
    capcut_voice: str        # voice_type trong Voice.json, VD: "BV421_vivn_streaming"
    piper_model: Path
    piper_speed: str
    translation_engine: str = "llm"  # "llm" hoặc "marian"
    llm_provider: str = "gemini"     # "gemini", "deepseek", "openai"
    gemini_api_key: str | None = None
    deepseek_api_key: str | None = None
    openai_api_key: str | None = None
    policy_mode: str = "off"
    policy_enable_ai_judge: bool = False
    policy_min_duration_seconds: float = 35.0
    policy_min_speech_density: float = 0.60
    policy_judge_threshold: float = 8.0

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        api_base = os.getenv("DUBVI_API_BASE", "http://localhost:8080").rstrip("/")
        token = os.getenv("DUBVI_WORKER_TOKEN", "standalone_studio_token")
        media_dir_value = os.getenv("DUBVI_MEDIA_DIR", str(Path.home() / "Videos" / "douyin"))
        media_dir = Path(media_dir_value).expanduser().resolve()
        media_dir.mkdir(parents=True, exist_ok=True)
        output_dir = Path(os.getenv("DUBVI_OUTPUT_DIR", str(media_dir / "dubvi-output"))).expanduser().resolve()
        model_dir = Path(os.getenv("DUBVI_MODEL_DIR", str(media_dir / "dubvi-model-cache"))).expanduser().resolve()
        log_dir = Path(os.getenv("DUBVI_LOG_DIR", str(media_dir / "dubvi-logs"))).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(model_dir / "huggingface"))
        os.environ.setdefault("XDG_CACHE_HOME", str(model_dir / "xdg"))
        policy_mode = os.getenv("DUBVI_POLICY_MODE", "off").strip().lower()
        if policy_mode not in {"off", "review", "enforce"}:
            raise ValueError("DUBVI_POLICY_MODE must be one of: off, review, enforce")
        return cls(
            api_base=api_base,
            token=token,
            media_dir=media_dir,
            output_dir=output_dir,
            model_dir=model_dir,
            log_dir=log_dir,
            device=os.getenv("DUBVI_DEVICE", "cpu"),
            whisper_model=os.getenv("DUBVI_WHISPER_MODEL", "small"),
            poll_seconds=max(3, int(os.getenv("DUBVI_POLL_SECONDS", "10"))),
            tts_provider=validate_tts_provider(os.getenv("DUBVI_TTS_PROVIDER", "piper")),
            capcut_voice=os.getenv("DUBVI_CAPCUT_VOICE", "BV421_vivn_streaming"),
            piper_model=Path(
                os.getenv(
                    "DUBVI_PIPER_MODEL",
                    str(model_dir / "piper-vietnamese" / "pretrained_vi.onnx"),
                )
            ).expanduser().resolve(),
            piper_speed=validate_piper_speed(os.getenv("DUBVI_PIPER_SPEED", "fast")),
            translation_engine=os.getenv("DUBVI_TRANSLATION_ENGINE", "llm").lower(),
            llm_provider=os.getenv("DUBVI_LLM_PROVIDER", "gemini").lower(),
            gemini_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            policy_mode=policy_mode,
            policy_enable_ai_judge=_env_flag("DUBVI_POLICY_ENABLE_AI_JUDGE"),
            policy_min_duration_seconds=float(os.getenv("DUBVI_POLICY_MIN_DURATION_SECONDS", "35")),
            policy_min_speech_density=float(os.getenv("DUBVI_POLICY_MIN_SPEECH_DENSITY", "0.60")),
            policy_judge_threshold=float(os.getenv("DUBVI_POLICY_JUDGE_THRESHOLD", "8.0")),
        )


CHANNEL_DEFAULTS = {
    "page_giai_cuu_chuong_lon": {
        "channel_name": "Giải Cứu Chuồng Lợn",
        "default_voice": "BV421_vivn_streaming",
        "style_tag": "🐷 Before-After / Review Gia Dụng",
        "color": "#FFE600",  # Vàng Douyin
    },
    "page_goc_tro_bat_on": {
        "channel_name": "Góc Trọ Bất Ổn",
        "default_voice": "BV007_streaming",
        "style_tag": "🏠 Drama KTX / Ở Chung",
        "color": "#00FFFF",  # Xanh Cyan Drama
    }
}


def scan_multi_channel_raw(base_dirs: list[Path] | None = None) -> list[dict[str, Any]]:
    """Discover local media and its optional reup sidecar metadata.

    Callers can still provide explicit scan roots. The default is intentionally
    limited to the configured DUBVI media directory; it never scans a whole
    user profile or another account's directories.
    """
    if base_dirs is None:
        base_dirs = [Settings.from_env().media_dir]
        
    discovered: list[dict[str, Any]] = []
    seen = set()
    
    for base in base_dirs:
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            if p.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            if _CANONICAL_REUP_MEDIA_RE.fullmatch(p.name):
                continue
            if not p.is_file() or p.name.startswith("_") or str(p.resolve()) in seen:
                continue
            seen.add(str(p.resolve()))
            
            meta_data: dict[str, Any] = {}
            # 1. Tìm file companion .meta.json cùng tên (VD: processed_1_tiktok.meta.json)
            meta_path = p.parent / f"{p.stem}.meta.json"
            if meta_path.is_file():
                try:
                    meta_data = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            elif p.name.startswith("processed_"):
                alt_meta = p.parent / f"{p.name}.meta.json"
                if alt_meta.is_file():
                    try:
                        meta_data = json.loads(alt_meta.read_text(encoding="utf-8"))
                    except Exception:
                        pass

            # Versioned bridge sidecars are a readiness contract. Legacy/manual
            # files remain usable without a sidecar, but an explicit incomplete
            # handoff is never shown as a completed DUBVI input.
            if meta_data.get("handoff_schema_version") == 2:
                continue
            if "handoff_schema_version" in meta_data and meta_data.get("handoff_status") != "complete":
                continue
                        
            channel_profile = meta_data.get("channel_profile") or p.parent.name
            target_platform = meta_data.get("target_platform") or "tiktok"
            vpdq_status = meta_data.get("vpdq_status") or (
                "PASSED" if meta_data.get("vpdq_similarity_percent") is not None else "UNKNOWN"
            )
            
            # Gán style kịch bản và giọng đọc mặc định theo kênh
            cleaned_channel = channel_profile.lower().replace(" ", "_").replace("-", "_")
            if "chuong_lon" in cleaned_channel or "giai_cuu" in cleaned_channel:
                cfg = CHANNEL_DEFAULTS["page_giai_cuu_chuong_lon"]
            elif "goc_tro" in cleaned_channel or "bat_on" in cleaned_channel:
                cfg = CHANNEL_DEFAULTS["page_goc_tro_bat_on"]
            else:
                cfg = {
                    "channel_name": channel_profile,
                    "default_voice": "BV421_vivn_streaming",
                    "style_tag": "🎬 Review Đời Thường",
                    "color": "#FFFFFF"
                }
                
            discovered.append({
                "video_path": p,
                "video_name": p.name,
                "parent_dir": p.parent.name,
                "channel_profile": channel_profile,
                "channel_name": cfg["channel_name"],
                "target_platform": target_platform,
                "vpdq_status": vpdq_status,
                "vpdq_similarity": meta_data.get("vpdq_similarity_percent"),
                "zoom_factor": meta_data.get("zoom_factor"),
                "duration_seconds": meta_data.get("duration_seconds"),
                "handoff_schema_version": meta_data.get("handoff_schema_version"),
                "handoff_status": meta_data.get("handoff_status"),
                "original_filename": meta_data.get("original_filename"),
                "output_filename": meta_data.get("output_filename"),
                "original_md5": meta_data.get("original_md5"),
                "output_md5": meta_data.get("output_md5"),
                "encoder_used": meta_data.get("encoder_used"),
                "layout_mode": meta_data.get("layout_mode"),
                "default_voice": cfg["default_voice"],
                "style_tag": cfg["style_tag"],
                "color": cfg["color"],
                "size_mb": round(p.stat().st_size / (1024 * 1024), 1),
                "mtime": p.stat().st_mtime,
                "has_meta": bool(meta_data)
            })
            
    discovered.sort(key=lambda x: x["mtime"], reverse=True)
    return discovered


def setup_logging(settings: Settings) -> logging.Logger:
    logger = logging.getLogger("dubvi-worker")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(settings.log_dir / "worker.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


class RpcClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.session = requests.Session()

    def mutate(self, procedure: str, payload: dict[str, Any]) -> Any:
        response = self.session.post(
            f"{self.settings.api_base}/api/trpc/{procedure}",
            json={"json": payload},
            headers={"content-type": "application/json"},
            timeout=45,
        )
        response.raise_for_status()
        parsed = response.json()
        if "error" in parsed:
            raise RuntimeError(parsed["error"].get("json", {}).get("message", "Worker RPC thất bại"))
        return parsed.get("result", {}).get("data", {}).get("json")

    def heartbeat(self, inventory: list[dict[str, Any]]) -> Any:
        return self.mutate("worker.heartbeat", {"token": self.settings.token, "inventory": inventory})

    def claim(self) -> dict[str, Any] | None:
        return self.mutate("worker.claim", {"token": self.settings.token})

    def report(self, job_id: str, stage: str, progress: int, status: str | None = None, output_path: str | None = None, error: str | None = None) -> None:
        payload: dict[str, Any] = {"token": self.settings.token, "jobId": job_id, "stage": stage, "progress": progress}
        if status:
            payload["status"] = status
        if output_path:
            payload["outputLocalPath"] = output_path
        if error:
            payload["errorMessage"] = error[:8000]
        self.mutate("worker.report", payload)

    def replace_segments(self, job_id: str, segments: list[dict[str, Any]]) -> None:
        self.mutate("worker.replaceSegments", {"token": self.settings.token, "jobId": job_id, "segments": segments})

    def correct_ocr(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.mutate("worker.correctOcr", {"token": self.settings.token, **payload})


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Không tìm thấy `{name}`. Hãy cài FFmpeg và bảo đảm binary nằm trong PATH.")


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True)


def ffprobe_duration(path: Path) -> float:
    result = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)])
    return float(result.stdout.strip())


def extract_audio(source: Path, target: Path) -> None:
    run(["ffmpeg", "-y", "-i", str(source), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(target)])


def scan_inventory(media_dir: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in media_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        stat = path.stat()
        result.append({
            "key": str(path.relative_to(media_dir)).replace("\\", "/"),
            "name": path.name,
            "sizeBytes": stat.st_size,
            "modifiedAtMs": int(stat.st_mtime * 1000),
        })
    return sorted(result, key=lambda item: item["modifiedAtMs"], reverse=True)[:500]


def whisper_runtime_options(device: str) -> dict[str, Any]:
    """Read bounded Whisper worker limits from the Control Plane child environment."""

    if device == "cpu":
        compute_type = os.getenv("DUBVI_WHISPER_COMPUTE_TYPE", "int8")
    else:
        compute_type = "float16"
    return {
        "compute_type": compute_type,
        "cpu_threads": int(os.getenv("DUBVI_WHISPER_CPU_THREADS", "2")),
        "num_workers": int(os.getenv("DUBVI_WHISPER_DECODE_WORKERS", "1")),
    }


def ensure_whisper_model(model_name: str, device: str) -> Any:
    """Resolve one selected Whisper model from the local cache before media work."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("Thiếu faster-whisper. Chạy `pip install -r requirements.txt`.") from exc
    global _whisper_model, _whisper_model_key
    cache_key = (str(model_name), str(device), os.environ.get("HF_HOME"))
    options = whisper_runtime_options(device)
    # Cache model giữa các job — tránh load lại model (~1-2 phút) mỗi lần
    if _whisper_model is None or _whisper_model_key != cache_key:
        try:
            model = WhisperModel(
                model_name,
                device=device,
                local_files_only=True,
                **options,
            )
        except Exception:
            raise RuntimeError(
                f"Whisper model '{model_name}' không có hoặc không tải được từ local cache offline; "
                "hãy cài sẵn model vào cache local trước khi chạy worker."
            ) from None
        _whisper_model = model
        _whisper_model_key = cache_key
    return _whisper_model


def transcribe(audio_path: Path, model_name: str, device: str) -> list[dict[str, Any]]:
    model = ensure_whisper_model(model_name, device)
    # beam_size=1 (greedy) nhanh hơn ~2x so với beam_size=5 mặc định, đủ dùng với model tiny
    # vad_filter=False: Ngăn Silero VAD nuốt nhầm câu thoại khi có nhạc nền BGM
    parts, _ = model.transcribe(
        str(audio_path),
        language="zh",
        vad_filter=False,
        word_timestamps=True,
        beam_size=1,
    )
    output: list[dict[str, Any]] = []
    for index, part in enumerate(parts):
        start_ms = int(part.start * 1000)
        end_ms = max(start_ms + 200, int(part.end * 1000))
        text = part.text.strip()
        if text:
            confidence = max(0, min(100, int((getattr(part, "avg_logprob", -1) + 1) * 100)))
            output.append({"position": index, "startMs": start_ms, "endMs": end_ms, "asrTextZh": text, "confidence": confidence})
    return output


def sample_times(start_ms: int, end_ms: int, count: int) -> Iterable[float]:
    duration = max(0.2, (end_ms - start_ms) / 1000)
    for index in range(max(1, count)):
        yield start_ms / 1000 + duration * (index + 0.5) / max(1, count)


def ocr_consensus_details(source: Path, segment: dict[str, Any], roi: dict[str, Any], samples: int) -> dict[str, Any]:
    try:
        import cv2
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise RuntimeError("Thiếu OpenCV hoặc RapidOCR. Chạy `pip install -r requirements.txt`.") from exc
    global _rapidocr_engine
    # Cache engine — tránh khởi tạo lại ONNX session cho mỗi segment (~200ms/lần)
    if _rapidocr_engine is None:
        _rapidocr_engine = RapidOCR()
    engine = _rapidocr_engine
    # Giới hạn số frame sample theo env var để tiết kiệm thời gian trên CPU chậm
    max_samples = int(os.getenv("DUBVI_OCR_MAX_SAMPLES", "0"))
    if max_samples > 0:
        samples = min(samples, max_samples)
    evidence: dict[str, list[float]] = {}
    detections: list[dict[str, Any]] = []
    for moment in sample_times(segment["startMs"], segment["endMs"], samples):
        frame_path = source.parent / f"_dubvi_ocr_{segment['position']}_{int(moment * 1000)}.jpg"
        try:
            run(["ffmpeg", "-y", "-ss", f"{moment:.3f}", "-i", str(source), "-frames:v", "1", "-q:v", "2", str(frame_path)])
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue
            height, width = frame.shape[:2]
            x = int(width * roi["xPercent"] / 100)
            y = int(height * roi["yPercent"] / 100)
            w = int(width * roi["widthPercent"] / 100)
            h = int(height * roi["heightPercent"] / 100)
            crop = frame[y:y + h, x:x + w]
            enlarged = cv2.resize(crop, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
            normalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
            response, _ = engine(normalized)
            if not response:
                continue
            from auto_roi import contains_cjk_text, map_ocr_quad_to_source
            for item in response:
                if len(item) < 3:
                    continue
                quad, item_text, item_confidence = item[0], str(item[1]).strip(), float(item[2])
                if item_confidence < 0.45 or not contains_cjk_text(item_text):
                    continue
                mapped_quad = map_ocr_quad_to_source(quad, (x, y), 2.5, (width, height))
                detections.append({
                    "text": item_text,
                    "confidence": round(item_confidence, 3),
                    "quad": mapped_quad,
                    "bbox": (min(point[0] for point in mapped_quad), min(point[1] for point in mapped_quad), max(point[0] for point in mapped_quad), max(point[1] for point in mapped_quad)),
                    "timestampMs": int(moment * 1000),
                    "sourceWidth": width,
                    "sourceHeight": height,
                })
            text = "".join(item[1] for item in response if len(item) >= 3 and float(item[2]) >= 0.45).strip()
            confidence = sum(float(item[2]) for item in response if len(item) >= 3) / max(1, len(response))
            if text:
                evidence.setdefault(text, []).append(confidence)
        finally:
            frame_path.unlink(missing_ok=True)
    if not evidence:
        return {"text": None, "confidence": 0, "frameAgreement": 0, "candidates": [], "detections": detections[:64]}
    best_text, scores = max(evidence.items(), key=lambda item: (len(item[1]), sum(item[1])))
    total_hits = sum(len(values) for values in evidence.values())
    candidates = [
        {"text": text, "hits": len(values), "confidence": int(100 * sum(values) / len(values))}
        for text, values in evidence.items()
    ]
    candidates.sort(key=lambda candidate: (candidate["hits"], candidate["confidence"]), reverse=True)
    return {
        "text": best_text,
        "confidence": int(100 * sum(scores) / len(scores)),
        "frameAgreement": int(100 * len(scores) / max(1, total_hits)),
        "candidates": candidates[:8],
        "detections": detections[:64],
    }


def ocr_consensus(source: Path, segment: dict[str, Any], roi: dict[str, Any], samples: int) -> tuple[str | None, int]:
    details = ocr_consensus_details(source, segment, roi, samples)
    return details["text"], details["confidence"]


def translate_to_vietnamese(text: str, slot_ms: int) -> tuple[str, bool]:
    try:
        import argostranslate.translate
        translated = argostranslate.translate.translate(text, "zh", "vi")
    except Exception as exc:
        # Argos packages do not always ship a direct Chinese-Vietnamese pair.
        # Marian downloads once into DUBVI_MODEL_DIR, then translates locally.
        try:
            global _nllb_components
            if _nllb_components is None:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
                model_name = os.getenv("DUBVI_MT_MODEL", "Helsinki-NLP/opus-mt-zh-vi")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
                except Exception:
                    raise _OfflineModelLoadError(
                        f"Translation model '{model_name}' không có hoặc không tải được từ local cache offline; "
                        "hãy cài sẵn model vào cache local trước khi chạy worker."
                    ) from None
                _nllb_components = (tokenizer, model)
            tokenizer, model = _nllb_components
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            generated = model.generate(**inputs, max_new_tokens=128)
            translated = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        except _OfflineModelLoadError:
            raise
        except Exception:
            raise RuntimeError(
                "Dịch máy local không khả dụng. Cài một lựa chọn trong "
                "requirements-optional.txt (argostranslate hoặc transformers) "
                "và model phù hợp, hoặc cấu hình LLM translation."
            ) from None
    if not translated or translated == text:
        raise RuntimeError("Dịch không trả về tiếng Việt; hãy thử lại hoặc kiểm tra model/kết nối mạng.")
    translated = translated.strip()
    char_budget = max(18, int(slot_ms / 1000 * 14))
    if len(translated) <= char_budget:
        return translated, False
    clauses = [clause.strip() for clause in re.split(r"(?<=[,;:])\s+", translated) if clause.strip()]
    shortened = ""
    for clause in clauses:
        candidate = f"{shortened} {clause}".strip()
        if len(candidate) > char_budget:
            break
        shortened = candidate
    if shortened and len(shortened) >= max(12, int(char_budget * 0.55)):
        return shortened, True
    return translated, True


async def edge_synthesize(text: str, voice: str, target: Path) -> None:
    try:
        import edge_tts
    except ImportError as exc:
        raise RuntimeError("Thiếu edge-tts. Chạy `pip install -r requirements.txt`.") from exc
    staging = target.with_suffix(f"{target.suffix}.part")
    last_error: Exception | None = None
    for attempt in range(1, 4):
        staging.unlink(missing_ok=True)
        try:
            communicator = edge_tts.Communicate(text, voice=voice)
            await communicator.save(str(staging))
            if staging.is_file() and staging.stat().st_size > 1024:
                staging.replace(target)
                return
            raise RuntimeError("Edge TTS tạo audio rỗng")
        except Exception as exc:
            last_error = exc
            await asyncio.sleep(attempt * 1.5)
    staging.unlink(missing_ok=True)
    raise RuntimeError("TTS Edge không tạo được audio sau 3 lần thử. Kiểm tra Internet hoặc dùng provider Piper local.") from last_error


def normalize_tts_provider_text(text: str) -> str:
    """Use the proven CapCut-safe Unicode form without changing canonical text."""
    return unicodedata.normalize("NFD", text)


def capcut_synthesize(text: str, voice: str, target: Path) -> None:
    """Tạo audio bằng CapCut TTS API (giọng chất lượng cao hơn Edge TTS).

    Tham số ``voice`` là voice_type trong Voice.json, ví dụ 'BV421_vivn_streaming'.
    CapCut API trả về URL MP3 tạm thời — hàm này tải file về rồi lưu vào ``target``.
    Retry tối đa 3 lần nếu gặp lỗi mạng.
    """
    try:
        from capcut_tts_api.client import CapCutClient
    except ImportError as exc:
        raise RuntimeError(
            "Thiếu capcut_tts_api. Hãy chắc chắn thư mục capcut_tts_api/ nằm trong local-worker/."
        ) from exc

    client = CapCutClient()
    provider_text = normalize_tts_provider_text(text)
    last_error: Exception | None = None

    for attempt in range(1, 4):
        staging = target.with_name(f".{target.name}.attempt-{attempt}.part")
        staging.unlink(missing_ok=True)
        try:
            result = client.generate_speech(
                texts=provider_text,
                voice=voice,
                timeout=45.0,
            )
            tasks = (result.get("data") or {}).get("tasks") or []
            if not tasks:
                raise RuntimeError(f"CapCut TTS không trả về task: {result}")
            payload_raw = tasks[0].get("payload", "{}")
            import json as _json
            payload = _json.loads(payload_raw) if isinstance(payload_raw, str) else payload_raw
            
            # Trích xuất speech_url từ danh sách audio_subtitles của CapCut API
            audio_url = None
            if isinstance(payload, dict):
                subtitles = payload.get("audio_subtitles") or []
                if subtitles and isinstance(subtitles, list):
                    audio_url = subtitles[0].get("speech_url")
                if not audio_url:
                    audio_url = (
                        payload.get("audio_url")
                        or payload.get("result_url")
                        or payload.get("url")
                        or (payload.get("data") or {}).get("audio")
                    )
            if not audio_url:
                raise RuntimeError("CapCut response did not contain an audio URL")
            # Download MP3 về target
            import requests as _req
            resp = _req.get(audio_url, timeout=30)
            resp.raise_for_status()
            if len(resp.content) < 1024:
                raise RuntimeError("CapCut TTS trả về file audio rỗng")
            staging.write_bytes(resp.content)
            if not staging.is_file() or staging.stat().st_size < 1024:
                raise RuntimeError("CapCut TTS staging audio is unusable")
            staging.replace(target)
            return
        except Exception as exc:
            last_error = exc
            staging.unlink(missing_ok=True)
            time.sleep(attempt * 1.5)

    error_type = type(last_error).__name__ if last_error is not None else "UnknownError"
    raise RuntimeError(
        f"CapCut TTS thất bại sau 3 lần thử ({error_type})"
    ) from None


def piper_synthesize(text: str, voice: str, target: Path, settings: Settings) -> None:
    """Tạo voice tiếng Việt local bằng một Piper ONNX session đã cache."""
    from piper_tts import _runtime_for_model

    _runtime_for_model(settings.piper_model).synthesize(text, target, settings.piper_speed)


def validate_capcut_voice(voice: str) -> None:
    """Reject an unresolved CapCut voice before a batch starts."""
    if not isinstance(voice, str) or not voice.strip():
        raise ValueError("CapCut voice must be nonblank")
    try:
        from capcut_tts_api.client import CapCutClient
        CapCutClient().resolve_voice(voice=voice.strip())
    except Exception as exc:
        raise ValueError(f"CapCut voice is not resolvable: {voice!r}") from exc


def synthesize(text: str, voice: str, target: Path, settings: Settings) -> None:
    """Synthesize strictly through the configured provider branch."""
    provider = validate_tts_provider(str(settings.tts_provider).strip().lower())
    if provider == "piper":
        piper_synthesize(text, voice or settings.capcut_voice, target, settings)
        return
    if provider == "capcut":
        actual_voice = voice or settings.capcut_voice
        validate_capcut_voice(actual_voice)
        capcut_synthesize(text, actual_voice, target)
        return
    edge_voice = voice or "vi-VN-HoaiMyNeural"
    if not edge_voice.startswith("vi-"):
        raise ValueError(f"Edge voice must use a vi-* voice name: {edge_voice!r}")
    asyncio.run(edge_synthesize(text, edge_voice, target))


def fit_voice(source_audio: Path, target_audio: Path, slot_ms: int, max_tempo: float) -> tuple[int, bool]:
    original_ms = int(ffprobe_duration(source_audio) * 1000)
    if original_ms <= slot_ms:
        shutil.copy2(source_audio, target_audio)
        return original_ms, False
    required_tempo = original_ms / max(1, slot_ms)
    if required_tempo > max_tempo:
        # Preserve the natural audio for manual editing instead of silently producing robotic speech.
        shutil.copy2(source_audio, target_audio)
        return original_ms, True
    run(["ffmpeg", "-y", "-i", str(source_audio), "-filter:a", f"atempo={required_tempo:.4f}", str(target_audio)])
    return int(ffprobe_duration(target_audio) * 1000), False


def round_rect_ass(w: int, h: int, r: int = 20) -> str:
    """Tạo ASS Vector path cho hình chữ nhật bo tròn từ (0, 0) đến (w, h)."""
    r = min(r, w // 2, h // 2)
    k = int(r * 0.55228)
    return (
        f"m {r} 0 "
        f"l {w - r} 0 "
        f"b {w - r + k} 0 {w} {r - k} {w} {r} "
        f"l {w} {h - r} "
        f"b {w} {h - r + k} {w - r + k} {h} {w - r} {h} "
        f"l {r} {h} "
        f"b {r - k} {h} 0 {h - r + k} 0 {h - r} "
        f"l 0 {r} "
        f"b 0 {r - k} {r - k} 0 {r} 0"
    )


def clamp_and_bridge_audio_gaps(
    audio_clips: list[dict[str, Any]],
    max_gap_ms: int = 600,
    min_pause_ms: int = 250
) -> list[dict[str, Any]]:
    """
    THUẬT TOÁN SMART GAP CLAMPING & PACING:
    Khống chế mọi khoảng câm giữa 2 câu thoại không vượt quá max_gap_ms (mặc định 600ms = 0.6s).
    Tự động tịnh tiến timeline câu kế tiếp về khoảng nghỉ tự nhiên 300-450ms.
    """
    if not audio_clips:
        return []

    adjusted_clips = []
    for i, clip in enumerate(audio_clips):
        c_copy = dict(clip)
        dur_sec = max(0.2, c_copy.get("durationSec") or (c_copy.get("voiceDurationMs", 1000) / 1000.0))
        c_copy["durationSec"] = dur_sec
        
        if i == 0:
            c_copy["startMs"] = 0
            c_copy["endMs"] = int(dur_sec * 1000)
            adjusted_clips.append(c_copy)
            continue

        prev = adjusted_clips[-1]
        prev_end_speech_ms = prev["startMs"] + int(prev["durationSec"] * 1000)
        current_raw_start = c_copy["startMs"]
        gap = current_raw_start - prev_end_speech_ms

        if gap > max_gap_ms:
            new_start = prev_end_speech_ms + min(gap, 450)
            c_copy["startMs"] = new_start
            c_copy["endMs"] = new_start + int(dur_sec * 1000)
        elif gap < min_pause_ms:
            new_start = prev_end_speech_ms + min_pause_ms
            c_copy["startMs"] = new_start
            c_copy["endMs"] = new_start + int(dur_sec * 1000)

        adjusted_clips.append(c_copy)
    return adjusted_clips


def preserve_source_timeline(audio_clips: list[dict[str, Any]]) -> list[tuple[int, Any]]:
    """Canonical jobs retain source cue starts; legacy gap clamping stays opt-in elsewhere."""
    return [(int(clip["startMs"]), clip["fitted_voice"]) for clip in audio_clips]


def draw_ass(
    segments: list[dict[str, Any]],
    video_width: int,
    video_height: int,
    roi: dict[str, Any],
    path: Path,
    font_name: str = "Segoe UI",
    font_size: int | None = None,
    font_color: str = "#111111"
) -> None:
    """Vẽ phụ đề ASS Thẻ Trắng Bo Góc Đơn (Single White Pill Subtitle Card) chuẩn 1080p."""
    if video_height >= video_width:
        target_w, target_h = 1080, 1920
    else:
        target_w, target_h = 1920, 1080

    center_x = target_w // 2
    actual_font_size = font_size if font_size else 48
    
    # Tính tâm Y theo ROI hoặc mặc định 70%
    if roi and "yPercent" in roi:
        center_y = int(target_h * (roi.get("yPercent", 65.0) + roi.get("heightPercent", 10.0) / 2.0) / 100)
    else:
        center_y = int(target_h * 0.70)

    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {target_w}", f"PlayResY: {target_h}",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: CleanWhiteBox,Arial,10,&H00FFFFFF,&H00000000,&H00FFFFFF,&H00FFFFFF,0,0,0,0,100,100,0,0,1,0,0,5,0,0,0,1",
        f"Style: CleanBlackText,{font_name},{actual_font_size},&H00111111,&H00000000,&H00FFFFFF,&H00000000,-1,0,0,0,100,100,0,0,1,0,0,5,0,0,0,1",
        "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
    ]

    def ass_time(value_ms: int) -> str:
        hours, remainder = divmod(value_ms, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        seconds, milliseconds = divmod(remainder, 1000)
        return f"{hours}:{minutes:02d}:{seconds:02d}.{milliseconds // 10:02d}"

    for seg in segments:
        st_ms = int(seg.get("startMs", 0))
        et_ms = int(seg.get("endMs", st_ms + 1500))
        vi_txt = seg.get("translatedTextVi", "").strip()
        if not vi_txt:
            continue

        words = vi_txt.split()
        if len(vi_txt) > 22 and len(words) > 3:
            mid = len(words) // 2
            formatted_vi = " ".join(words[:mid]) + r"\N" + " ".join(words[mid:])
            num_lines = 2
            max_line_len = max(len(" ".join(words[:mid])), len(" ".join(words[mid:])))
        else:
            formatted_vi = vi_txt
            num_lines = 1
            max_line_len = len(vi_txt)

        box_w = max(950, int(max_line_len * actual_font_size * 0.58 + 80))
        box_h = 135 if num_lines == 1 else 158

        st_str = ass_time(st_ms)
        et_str = ass_time(et_ms)

        clean_text = formatted_vi.replace("{", "\\{").replace("}", "\\}").replace("\n", "\\N")
        lines.append(f"Dialogue: 0,{st_str},{et_str},CleanWhiteBox,,0,0,0,,{{\\pos({center_x},{center_y})\\p1}}{round_rect_ass(box_w, box_h, r=20)}{{\\p0}}")
        lines.append(f"Dialogue: 1,{st_str},{et_str},CleanBlackText,,0,0,0,,{{\\pos({center_x},{center_y})}}{clean_text}")

    path.write_text("\n".join(lines), encoding="utf-8")


def draw_srt(segments: list[dict[str, Any]], path: Path) -> None:
    def srt_time(value: int) -> str:
        hours, remainder = divmod(value, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        seconds, milliseconds = divmod(remainder, 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
    blocks = []
    for index, segment in enumerate(segments, start=1):
        blocks.append(f"{index}\n{srt_time(int(segment['startMs']))} --> {srt_time(int(segment['endMs']))}\n{segment.get('translatedTextVi') or ''}")
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def ffprobe_dimensions(source: Path) -> tuple[int, int]:
    result = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(source)])
    width, height = result.stdout.strip().split("x")
    return int(width), int(height)


def ffprobe_has_audio(source: Path) -> bool:
    """Kiểm tra video gốc có chứa luồng âm thanh (audio stream) hay không."""
    try:
        res = run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(source)])
        return bool(res.stdout.strip())
    except Exception:
        return False


def render_video(
    source: Path,
    output: Path,
    ass_path: Path,
    audio_segments: list[tuple[Path, int]],
    roi: dict[str, Any],
    audio_mode: str,
    clean_bgm_path: Path | None = None,
    **kwargs: Any
) -> None:
    """Render video Master 1080p với Thẻ Trắng Bo Góc Đơn (Single White Pill) và hòa âm LUFS -14dB."""
    orig_w, orig_h = ffprobe_dimensions(source)
    
    # Chuẩn hóa độ phân giải xuất ra Full HD 1080p (Chuẩn TikTok/Reels/Shorts)
    if orig_h >= orig_w:
        scale_filter = "scale=1080:1920"
    else:
        scale_filter = "scale=1920:1080"

    ass_escaped = str(ass_path).replace("\\", "/").replace(":", "\\:")
    video_filter = f"[0:v]{scale_filter},subtitles='{ass_escaped}'[video]"
    
    use_clean_bgm = clean_bgm_path and clean_bgm_path.is_file()
    has_orig_audio = ffprobe_has_audio(source)
    command = ["ffmpeg", "-y", "-i", str(source)]
    if use_clean_bgm:
        command.extend(["-i", str(clean_bgm_path)])
        
    audio_labels: list[str] = []
    audio_offset = 2 if use_clean_bgm else 1
    for index, (audio_path, offset_ms) in enumerate(audio_segments, start=audio_offset):
        command.extend(["-i", str(audio_path)])
        idx_name = index - audio_offset + 1
        audio_labels.append(f"[{index}:a]adelay={offset_ms}|{offset_ms}[v{idx_name}]")

    filters = [video_filter]
    if audio_labels:
        inputs = "".join(f"[v{i}]" for i in range(1, len(audio_segments) + 1))
        if len(audio_segments) == 1:
            mix_voice = f"{audio_labels[0]};[v1]anull[voice]"
        else:
            mix_voice = f"{';'.join(audio_labels)};{inputs}amix=inputs={len(audio_segments)}:duration=longest:normalize=0[voice]"

        if use_clean_bgm:
            # Dùng nhạc nền AI tách sạch tiếng Trung + chuẩn hóa âm lượng Mobile LUFS -14dB
            filters.append(f"{mix_voice};[1:a]volume=0.85[bgm];[bgm][voice]amix=inputs=2:duration=first:normalize=0,loudnorm=I=-14:TP=-1.5:LRA=11[finalaudio]")
        elif audio_mode == "duck" and has_orig_audio:
            # Đệm nhạc nền nhẹ 8-10% lọc bớt giọng nói cũ + chuẩn hóa âm lượng Mobile LUFS -14dB
            filters.append(f"{mix_voice};[0:a]volume=0.08,lowpass=f=3000[bg];[bg][voice]amix=inputs=2:duration=longest:normalize=0,loudnorm=I=-14:TP=-1.5:LRA=11[finalaudio]")
        elif audio_mode == "keep" and has_orig_audio:
            filters.append("[0:a]loudnorm=I=-14:TP=-1.5:LRA=11[finalaudio]")
        else:
            # Replace: chỉ dùng voice + chuẩn hóa âm lượng Mobile LUFS -14dB
            filters.append(f"{mix_voice};[voice]loudnorm=I=-14:TP=-1.5:LRA=11[finalaudio]")
    else:
        if has_orig_audio:
            filters.append("[0:a]loudnorm=I=-14:TP=-1.5:LRA=11[finalaudio]")
        else:
            filters.append("aevalsrc=0:d=1[finalaudio]")

    command.extend([
        "-filter_complex", ";".join(filters),
        "-map", "[video]", "-map", "[finalaudio]",
        "-c:v", "libx264", "-preset", "veryfast", "-threads", "2", "-crf", "20",
        "-c:a", "aac",
    ])
    effective_duration = float(kwargs.get("max_duration_s") or ffprobe_duration(source))
    command.extend(build_render_command(str(source), str(output), str(ass_path), audio_segments, audio_mode, effective_duration))
    run(command)


def build_render_command(
    source: str, output: str, ass_path: str, audio_segments: list[tuple[Path, int]] | list[Any], audio_mode: str, source_duration: float,
) -> list[str]:
    """Return the bounded output tail: source visual duration always wins over short speech."""
    del source, ass_path, audio_segments, audio_mode
    return ["-t", f"{max(0.0, source_duration):.3f}", "-movflags", "+faststart", output]


def generate_video_thumbnail(
    video_path: Path,
    hook_text: str,
    output_thumb_path: Path,
    timestamp_s: float = 2.0
) -> bool:
    """Tự động trích xuất frame chất lượng cao và đè chữ Hook to nổi bật làm ảnh bìa (Thumbnail)."""
    try:
        temp_frame = output_thumb_path.parent / f"temp_{output_thumb_path.stem}.png"
        run([
            "ffmpeg", "-y", "-ss", f"{timestamp_s:.2f}", "-i", str(video_path),
            "-vframes", "1", "-q:v", "1", str(temp_frame)
        ])
        if not temp_frame.is_file():
            return False

        wrapped = wrap_subtitle_text(hook_text, max_chars=20).replace(r"\N", "\n")
        
        try:
            from PIL import Image, ImageDraw, ImageFont
            img = Image.open(temp_frame).convert("RGBA")
            w, h = img.size
            draw = ImageDraw.Draw(img)
            
            # Chọn font hệ thống Windows
            font_size = int(h * 0.045)  # ~86px trên 1080x1920
            font = None
            font_paths = []
            configured_font = os.getenv("DUBVI_THUMBNAIL_FONT")
            if configured_font:
                font_paths.append(Path(configured_font).expanduser())
            windows_font_dir = os.getenv("WINDIR")
            if windows_font_dir:
                font_root = Path(windows_font_dir) / "Fonts"
                font_paths.extend(font_root / name for name in ("arialbd.ttf", "tahomabd.ttf", "segoeuib.ttf"))
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        font = ImageFont.truetype(font_path, font_size)
                        break
                    except Exception:
                        pass
            if font is None:
                font = ImageFont.load_default()

            # Tọa độ ở 1/3 trên của màn hình
            lines = wrapped.split("\n")
            line_height = int(font_size * 1.3)
            start_y = int(h * 0.26)
            
            for idx, line in enumerate(lines):
                bbox = draw.textbbox((0, 0), line, font=font)
                text_w = bbox[2] - bbox[0]
                x_pos = (w - text_w) // 2
                y_pos = start_y + idx * line_height
                
                # Viền đen 5px
                for dx in range(-5, 6):
                    for dy in range(-5, 6):
                        if dx*dx + dy*dy <= 25:
                            draw.text((x_pos + dx, y_pos + dy), line, font=font, fill=(0, 0, 0, 255))
                # Chữ Vàng Neon Douyin
                draw.text((x_pos, y_pos), line, font=font, fill=(255, 230, 0, 255))

            img.convert("RGB").save(output_thumb_path, quality=95)
            if temp_frame.is_file():
                temp_frame.unlink(missing_ok=True)
            return True
        except Exception:
            shutil.copy2(temp_frame, output_thumb_path)
            if temp_frame.is_file():
                temp_frame.unlink(missing_ok=True)
            return True
    except Exception as ex:
        logging.getLogger("dubvi-worker").warning(f"Lỗi tạo thumbnail: {ex}")
        return False


def process_job(
    settings: Settings,
    rpc: RpcClient,
    job: dict[str, Any],
    *,
    policy_observer: Callable[[Any], Any] | None = None,
    canonical_policy_outcomes: bool = False,
) -> None:
    job_id = job["id"]
    config = job["configJson"]
    source = (settings.media_dir / job["sourceKey"]).resolve()
    if settings.media_dir not in source.parents or not source.is_file():
        raise RuntimeError(f"Không tìm thấy file nguồn trong DUBVI_MEDIA_DIR: {job['sourceKey']}")
    job_dir = settings.output_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    job_log = job_dir / "job.log"
    def log_job(message: str) -> None:
        with job_log.open("a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {message}\n")
    log_job(f"Bắt đầu job: {source.name}")
    rpc.report(job_id, "probe", 5, status="processing")
    require_binary("ffmpeg")
    require_binary("ffprobe")
    if job.get("resumeStage") == "render_from_review":
        reviewed_segments = job.get("segments") or []
        if not reviewed_segments:
            raise RuntimeError("Job render-from-review không có segment đã duyệt.")
        voice_inputs: list[tuple[Path, int]] = []
        review_needed = False
        for segment in reviewed_segments:
            raw_voice = job_dir / f"voice_reviewed_{segment['position']:03d}_raw.mp3"
            fitted_voice = job_dir / f"voice_reviewed_{segment['position']:03d}.mp3"
            synthesize(segment.get("translatedTextVi") or "", config["voice"]["name"], raw_voice, settings)
            duration_ms, needs_review = fit_voice(raw_voice, fitted_voice, int(segment["endMs"]) - int(segment["startMs"]), float(config["voice"]["maxTempo"]))
            segment["voicePath"] = str(fitted_voice)
            segment["voiceDurationMs"] = duration_ms
            segment["needsReview"] = needs_review
            review_needed = review_needed or needs_review
            voice_inputs.append((fitted_voice, int(segment["startMs"])))
        rpc.replace_segments(job_id, reviewed_segments)
        if review_needed:
            log_job("Voice sau khi duyệt vẫn dài hơn slot; trả về awaiting_review")
            rpc.report(job_id, "awaiting_review", 82, status="awaiting_review", output_path=str(job_dir / "segments.json"))
            return
        rpc.report(job_id, "render_approved", 84, status="processing")
        ass_path = job_dir / "subtitles_vi_reviewed.ass"
        srt_path = job_dir / "subtitles_vi_reviewed.srt"
        visual_source = job_dir / "masked_source_repaired.mkv"
        if not visual_source.is_file():
            visual_source = job_dir / "masked_source.mkv"
        if not visual_source.is_file():
            visual_source = source
        width, height = ffprobe_dimensions(visual_source)
        draw_ass(reviewed_segments, width, height, config["roi"], ass_path)
        draw_srt(reviewed_segments, srt_path)
        output = job_dir / f"{source.stem}_vi_reviewed.mp4"
        render_video(visual_source, output, ass_path, voice_inputs, config["roi"], config["audioMode"])
        if visual_source != source:
            visual_source.unlink(missing_ok=True)
        log_job(f"Render từ segment đã duyệt hoàn tất: {output.name}")
        rpc.report(job_id, "complete", 100, status="complete", output_path=str(output))
        return
    audio_wav = job_dir / "source.wav"
    extract_audio(source, audio_wav)
    log_job("Đã trích audio WAV 16 kHz")

    rpc.report(job_id, "asr_zh", 18)
    segments = transcribe(audio_wav, settings.whisper_model, settings.device)
    if not segments:
        raise RuntimeError("ASR không tìm thấy lời thoại tiếng Trung. Hãy thử model Whisper lớn hơn hoặc chỉnh audio nguồn.")

    rpc.report(job_id, "ocr_consensus", 35)
    for segment in segments:
        audit: dict[str, Any] = {"originalText": None, "ocrConfidence": 0, "frameAgreement": 0, "candidates": []}
        if config["ocr"]["enabled"]:
            details = ocr_consensus_details(source, segment, config["roi"], config["ocr"]["sampleFrames"])
            text = details["text"]
            confidence = details["confidence"]
            audit = {
                "originalText": text, "ocrConfidence": confidence, "frameAgreement": details["frameAgreement"],
                "candidates": details["candidates"], "detections": details["detections"],
            }
            segment["ocrTextZh"] = text
            segment["ocrAuditJson"] = audit
            asr_confidence = int(segment.get("confidence", 0))
            should_call_llm = bool(text) and config["ocr"].get("llmCorrection", False) and (
                confidence < 96 or details["frameAgreement"] < 86 or (asr_confidence >= 65 and segment.get("asrTextZh") != text)
            )
            if should_call_llm:
                try:
                    correction = rpc.correct_ocr({
                        "ocrText": text,
                        "ocrConfidence": confidence,
                        "frameAgreement": details["frameAgreement"],
                        "candidates": details["candidates"],
                        "asrTextZh": segment.get("asrTextZh"),
                        "asrConfidence": asr_confidence,
                    })
                    audit["llmCorrection"] = correction
                    if correction.get("accepted"):
                        segment["sourceTextZh"] = correction["correctedText"]
                    if correction.get("needsReview"):
                        segment["needsReview"] = True
                except Exception as exc:
                    audit["llmCorrection"] = {"attempted": True, "model": None, "correctedText": text, "accepted": False, "modelConfidence": 0, "rationale": f"Worker không gọi được LLM: {exc}", "needsReview": True}
                    segment["needsReview"] = True
            if text and confidence >= config["ocr"]["minConfidence"]:
                segment.setdefault("sourceTextZh", text)
                segment["confidence"] = confidence
            else:
                segment.setdefault("sourceTextZh", segment["asrTextZh"])
        else:
            segment["sourceTextZh"] = segment["asrTextZh"]
            segment["ocrAuditJson"] = audit
    log_job(f"Đã tạo {len(segments)} segment ASR/OCR")

    # Repair only OCR-confirmed CJK pixels before the Vietnamese overlay.  This is
    # deliberately local: a failed repair becomes review work, never a broad blur.
    visual_source = source
    mask_review_needed = False
    if config["ocr"]["enabled"]:
        from auto_roi import sparse_whole_frame_discovery
        from localized_masking import (
            apply_localized_masks, build_mask_plan, build_mask_qc, expand_affected_plan, stratified_sample_indices,
        )

        detections = [
            detection
            for segment in segments
            for detection in segment.get("ocrAuditJson", {}).get("detections", [])
            if isinstance(detection, dict)
        ]
        detections.extend(sparse_whole_frame_discovery(source, int(config["ocr"].get("discoveryFrames", 8))))
        width, height = ffprobe_dimensions(source)
        mask_plan = build_mask_plan(detections, (width, height))
        mask_qc = build_mask_qc(plan_count=len(mask_plan), applied_count=0, residual_count=0)
        if mask_plan:
            masked_source = job_dir / "masked_source.mkv"
            try:
                mask_qc = apply_localized_masks(source, masked_source, mask_plan)
                # Stratify repair QC across the complete affected timeline, not its head.
                affected = [segment for segment in segments if segment.get("ocrAuditJson", {}).get("detections")]
                def sampled_residual(path: Path) -> list[dict[str, Any]]:
                    residual_items: list[dict[str, Any]] = []
                    for sample_index in stratified_sample_indices(len(affected), 12):
                        residual_items.extend(ocr_consensus_details(path, affected[sample_index], config["roi"], 1).get("detections", []))
                    residual_items.extend(sparse_whole_frame_discovery(path, int(config["ocr"].get("discoveryFrames", 8))))
                    return residual_items

                residual = sampled_residual(masked_source)
                retry_plan = expand_affected_plan(mask_plan, residual, (width, height))
                if retry_plan:
                    repaired_source = job_dir / "masked_source_repaired.mkv"
                    retry_qc = apply_localized_masks(masked_source, repaired_source, retry_plan)
                    masked_source.unlink(missing_ok=True)
                    masked_source = repaired_source
                    residual = sampled_residual(masked_source)
                    mask_qc = build_mask_qc(
                        plan_count=len(mask_plan),
                        applied_count=int(mask_qc["appliedRegions"]) + int(retry_qc["appliedRegions"]),
                        residual_count=len(residual),
                    )
                else:
                    mask_qc = build_mask_qc(
                        plan_count=len(mask_plan), applied_count=int(mask_qc["appliedRegions"]), residual_count=len(residual),
                    )
                visual_source = masked_source
                mask_review_needed = bool(mask_qc["needsReview"])
            except Exception as exc:
                mask_review_needed = True
                mask_qc = build_mask_qc(plan_count=len(mask_plan), applied_count=0, residual_count=1)
                log_job(f"Localized CJK repair requires review: {type(exc).__name__}")
        (job_dir / "mask_qc.json").write_text(json.dumps(mask_qc, ensure_ascii=False, indent=2), encoding="utf-8")

    if settings.policy_mode != "off":
        from content_policy import (
            ContentPolicyConfig,
            ContentPolicyInput,
            PolicyDecision,
            PrefilterConfig,
            evaluate_content_policy,
        )

        source_metadata = config.get("sourceMetadata")
        if not isinstance(source_metadata, dict):
            source_metadata = {}
        policy_transcript = " ".join(
            str(segment.get("sourceTextZh") or segment.get("ocrTextZh") or segment.get("asrTextZh") or "")
            for segment in segments
        ).strip()
        policy_result = evaluate_content_policy(
            ContentPolicyInput(
                duration_seconds=ffprobe_duration(source),
                segments=segments,
                transcript=policy_transcript,
                title=str(config.get("sourceTitle") or job.get("sourceName") or source.stem),
                description=str(config.get("sourceDescription") or ""),
                metadata=source_metadata,
                author=str(config.get("sourceAuthor") or ""),
                channel_profile=str(config.get("channelProfile") or ""),
            ),
            ContentPolicyConfig(
                prefilter=PrefilterConfig(
                    min_duration_seconds=settings.policy_min_duration_seconds,
                    min_speech_density=settings.policy_min_speech_density,
                ),
                enable_ai_judge=settings.policy_enable_ai_judge,
                judge_threshold=settings.policy_judge_threshold,
                judge_api_key=settings.gemini_api_key,
            ),
        )
        if policy_observer is not None:
            policy_observer(policy_result)
        policy_path = job_dir / "policy.json"
        if canonical_policy_outcomes:
            policy_document = {
                "policy_schema_version": 1,
                "mode": settings.policy_mode,
                "overall_decision": policy_result.overall_decision.value,
                "reason_codes": list(policy_result.reason_codes)[:32],
                "needs_review": policy_result.needs_review,
                "evaluated_at": policy_result.evaluated_at,
            }
        else:
            policy_document = policy_result.to_dict()
        policy_path.write_text(
            json.dumps(policy_document, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log_job(
            f"Content policy: {policy_result.overall_decision.value} "
            f"({', '.join(policy_result.reason_codes)})"
        )
        if settings.policy_mode == "enforce" and not canonical_policy_outcomes:
            if policy_result.overall_decision == PolicyDecision.REJECT:
                raise RuntimeError(f"Content policy rejected: {', '.join(policy_result.reason_codes)}")
            if policy_result.needs_review:
                rpc.report(job_id, "policy_review", 48, status="awaiting_review", output_path=str(policy_path))
                return
    elif canonical_policy_outcomes and policy_observer is not None:
        # Canonical mode makes disabled policy explicit; it must never become
        # a synthetic pass or fabricate deep-policy evidence.
        policy_observer(None)

    rpc.report(job_id, "translate_vi", 52)
    if settings.translation_engine == "llm":
        try:
            from llm_translator import translate_segments_native
            segments = translate_segments_native(
                segments,
                provider=settings.llm_provider,
                gemini_key=settings.gemini_api_key,
                deepseek_key=settings.deepseek_api_key,
                openai_key=settings.openai_api_key,
                channel_profile=str(config.get("channelProfile") or ""),
                context_card=config.get("localizationContext") if isinstance(config.get("localizationContext"), dict) else None,
                glossary=config.get("localizationGlossary"),
            )
        except Exception as exc:
            log_job(f"LLM Transcreation lỗi: {exc}, fallback sang dịch máy local...")

    for segment in segments:
        if not segment.get("translatedTextVi"):
            translated, translation_review = translate_to_vietnamese(segment["sourceTextZh"], segment["endMs"] - segment["startMs"])
            segment["translatedTextVi"] = translated
            segment["needsReview"] = bool(segment.get("needsReview")) or translation_review

    rpc.report(job_id, "tts_fit", 68)
    raw_clips: list[dict[str, Any]] = []
    review_needed = mask_review_needed
    timing_pass_b: list[dict[str, Any]] = []
    from llm_translator import normalize_context_card, normalize_glossary
    timing_context = normalize_context_card(config.get("localizationContext"))
    timing_glossary = normalize_glossary(config.get("localizationGlossary"))
    for segment_index, segment in enumerate(segments):
        raw_voice = job_dir / f"voice_{segment['position']:03d}_raw.mp3"
        fitted_voice = job_dir / f"voice_{segment['position']:03d}.mp3"
        synthesize(segment["translatedTextVi"], config["voice"]["name"], raw_voice, settings)
        duration_ms, needs_review = fit_voice(raw_voice, fitted_voice, segment["endMs"] - segment["startMs"], float(config["voice"]["maxTempo"]))
        if needs_review and settings.translation_engine == "llm":
            from llm_translator import refine_for_tts_overflow

            replacement = refine_for_tts_overflow(
                segment, api_key=settings.gemini_api_key, measured_ms=duration_ms,
                channel_profile=str(config.get("channelProfile") or ""),
                previous=segments[max(0, segment_index - 2):segment_index],
                following=segments[segment_index + 1:segment_index + 3],
                context_card=timing_context, glossary=timing_glossary,
            )
            timing_pass_b.append({"position": int(segment["position"]), "attempted": True, "accepted": bool(replacement)})
            if replacement:
                segment["translatedTextVi"] = replacement
                synthesize(replacement, config["voice"]["name"], raw_voice, settings)
                duration_ms, needs_review = fit_voice(
                    raw_voice, fitted_voice, segment["endMs"] - segment["startMs"], float(config["voice"]["maxTempo"]),
                )
        segment["voicePath"] = str(fitted_voice)
        segment["voiceDurationMs"] = duration_ms
        segment["needsReview"] = bool(segment.get("needsReview")) or needs_review
        review_needed = review_needed or bool(segment["needsReview"])
        raw_clips.append({
            "segment": segment,
            "fitted_voice": fitted_voice,
            "startMs": segment["startMs"],
            "endMs": segment["endMs"],
            "durationSec": duration_ms / 1000.0,
        })

    # P1C preserves the source timeline; legacy Studio/watchdog may still opt
    # into gap clamping, but canonical delivery never rewrites cue positions.
    voice_inputs = [(Path(path), start_ms) for start_ms, path in preserve_source_timeline(raw_clips)]

    rpc.replace_segments(job_id, segments)
    manifest_path = job_dir / "segments.json"
    manifest_path.write_text(json.dumps({"jobId": job_id, "source": str(source), "config": config, "segments": segments}, ensure_ascii=False, indent=2), encoding="utf-8")
    localization_qc = {
        "schemaVersion": 1,
        "segmentCount": min(512, len(segments)),
        "needsReview": review_needed,
        "tempoCeiling": float(config["voice"]["maxTempo"]),
        "timingPassB": timing_pass_b[:64],
    }
    (job_dir / "localization_qc.json").write_text(json.dumps(localization_qc, ensure_ascii=False, indent=2), encoding="utf-8")
    srt_path = job_dir / "subtitles_vi.srt"
    draw_srt(segments, srt_path)
    log_job(f"Đã xuất manifest và SRT: {manifest_path.name}, {srt_path.name}")
    if review_needed:
        rpc.report(job_id, "awaiting_review", 78, status="awaiting_review", output_path=str(manifest_path))
        return

    rpc.report(job_id, "render", 84)
    # Tách BGM sạch bằng Demucs AI nếu khả dụng
    clean_bgm_wav = job_dir / "clean_bgm.wav"
    if config.get("audioMode") != "keep":
        try:
            log_job("Bắt đầu tách giọng tiếng Trung và giữ nhạc nền BGM bằng Demucs AI...")
            demucs_dir = job_dir / "demucs_out"
            run([sys.executable, "-m", "demucs.separate", "--two-stems=vocals", "-n", "htdemucs", "-o", str(demucs_dir), str(audio_wav)])
            demucs_no_vocals = demucs_dir / "htdemucs" / audio_wav.stem / "no_vocals.wav"
            if demucs_no_vocals.is_file():
                shutil.copy2(demucs_no_vocals, clean_bgm_wav)
                log_job("Đã tách BGM sạch 100% bằng Demucs.")
        except Exception as exc:
            log_job(f"Demucs AI BGM không khả dụng ({exc}), dùng audio ducking thông thường.")

    ass_path = job_dir / "subtitles_vi.ass"
    width, height = ffprobe_dimensions(visual_source)
    draw_ass(segments, width, height, config["roi"], ass_path)
    output = job_dir / f"{source.stem}_vi.mp4"
    render_video(
        visual_source, output, ass_path, voice_inputs, config["roi"], config["audioMode"],
        clean_bgm_path=clean_bgm_wav if clean_bgm_wav.is_file() else None
    )
    if visual_source != source:
        visual_source.unlink(missing_ok=True)
    log_job(f"Render hoàn tất: {output.name}")
    rpc.report(job_id, "complete", 100, status="complete", output_path=str(output))


def check_environment(settings: Settings) -> None:
    require_binary("ffmpeg")
    require_binary("ffprobe")
    hf_home = os.environ.setdefault("HF_HOME", str(settings.model_dir / "huggingface"))
    try:
        ensure_whisper_model(settings.whisper_model, settings.device)
    except Exception:
        raise RuntimeError(
            f"Whisper model '{settings.whisper_model}' không có hoặc không tải được từ local cache offline; "
            f"HF_HOME hiệu lực: {Path(hf_home).expanduser().resolve()}"
        ) from None
    print("✓ FFmpeg và FFprobe sẵn sàng")
    print(f"✓ Media directory: {settings.media_dir}")
    print(f"✓ Output directory: {settings.output_dir}")
    print(f"✓ Model cache: {settings.model_dir}")
    print(f"✓ Hugging Face cache: {Path(hf_home).expanduser().resolve()}")
    print(f"✓ Worker logs: {settings.log_dir}")
    print(f"✓ Whisper device/model: {settings.device}/{settings.whisper_model}")
    print("! edge-tts gọi Microsoft TTS qua Internet; dùng Piper/Coqui nếu cần TTS hoàn toàn offline.")


def run_once(settings: Settings, rpc: RpcClient) -> bool:
    rpc.heartbeat(scan_inventory(settings.media_dir))
    job = rpc.claim()
    if not job:
        print("Không có job đang chờ.")
        return False
    print(f"Nhận job {job['id']} — {job['sourceName']}")
    try:
        process_job(settings, rpc, job)
        print("Đã xử lý xong job.")
    except Exception as exc:
        rpc.report(job["id"], "failed", 100, status="failed", error=str(exc))
        print(f"Job thất bại: {exc}", file=sys.stderr)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="DUBVI local video worker")
    parser.add_argument("command", choices=["verify", "heartbeat", "once", "watch"])
    args = parser.parse_args()
    settings = Settings.from_env()
    logger = setup_logging(settings)
    rpc = RpcClient(settings)
    if args.command == "verify":
        check_environment(settings)
        return
    if args.command == "heartbeat":
        reply = rpc.heartbeat(scan_inventory(settings.media_dir))
        print(json.dumps(reply, ensure_ascii=False, indent=2))
        return
    if args.command == "once":
        logger.info("Chạy worker một lần")
        run_once(settings, rpc)
        return
    while True:
        logger.info("Polling job mới")
        run_once(settings, rpc)
        time.sleep(settings.poll_seconds)




# ===========================================================================
# Studio Autopilot Helper Integrations (ASR Harness, OCR Tracking, CTA/Logo)
# ===========================================================================

def aggregate_whisper_segments(
    raw_segments: list[dict[str, Any]],
    max_gap_ms: int = 350,
    max_dur_ms: int = 4200,
    min_dur_ms: int = 1200
) -> list[dict[str, Any]]:
    """Hợp nhất các mẩu thoại vi mô của Whisper thành các câu hoàn chỉnh có nghĩa."""
    merged: list[dict[str, Any]] = []
    curr: dict[str, Any] | None = None

    for s in raw_segments:
        text = (s.get("text") or s.get("asrTextZh") or s.get("chinese_text") or "").strip()
        if not text:
            continue
        st = s.get("startMs", 0)
        et = s.get("endMs", 0)

        if curr is None:
            curr = {
                "startMs": st,
                "endMs": et,
                "texts": [text],
                "sub_segments": [s]
            }
            continue

        gap = st - curr["endMs"]
        cur_dur = et - curr["startMs"]

        has_punctuation = any(curr["texts"][-1].endswith(p) for p in ("。", "！", "？", "!", "?", "；"))
        is_too_long = cur_dur > max_dur_ms
        is_long_pause = gap > max_gap_ms and (curr["endMs"] - curr["startMs"]) >= min_dur_ms

        if is_too_long or has_punctuation or is_long_pause:
            merged.append({
                "position": len(merged),
                "startMs": curr["startMs"],
                "endMs": curr["endMs"],
                "duration_s": round((curr["endMs"] - curr["startMs"]) / 1000.0, 2),
                "asrTextZh": "".join(curr["texts"]),
                "chinese_text": "".join(curr["texts"]),
                "confidence": 95,
            })
            curr = {
                "startMs": st,
                "endMs": et,
                "texts": [text],
                "sub_segments": [s]
            }
        else:
            curr["endMs"] = et
            curr["texts"].append(text)
            curr["sub_segments"].append(s)

    if curr and curr["texts"]:
        merged.append({
            "position": len(merged),
            "startMs": curr["startMs"],
            "endMs": curr["endMs"],
            "duration_s": round((curr["endMs"] - curr["startMs"]) / 1000.0, 2),
            "asrTextZh": "".join(curr["texts"]),
            "chinese_text": "".join(curr["texts"]),
            "confidence": 95,
        })

    return merged


_rapidocr_engine = None


def auto_detect_subtitle_roi(video_path: Path, sample_count: int = 8) -> dict[str, Any]:
    """Quét tự động vị trí Y và kích thước chữ phụ đề gốc tiếng Trung."""
    try:
        import cv2
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        return {
            "has_hardsub": False,
            "yPercent": 75.0,
            "yCenterPx": 1440,
            "avgHeightPx": 60,
            "boxBlurY": 1380,
            "boxBlurH": 120
        }

    global _rapidocr_engine
    if _rapidocr_engine is None:
        try:
            _rapidocr_engine = RapidOCR()
        except Exception:
            return {
                "has_hardsub": False,
                "yPercent": 75.0,
                "yCenterPx": 1440,
                "avgHeightPx": 60,
                "boxBlurY": 1380,
                "boxBlurH": 120
            }
    engine = _rapidocr_engine

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1080
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1920

    if total_frames <= 0:
        cap.release()
        return {"has_hardsub": False, "yPercent": 75.0, "yCenterPx": int(height * 0.75), "avgHeightPx": 60, "boxBlurY": int(height * 0.75 - 60), "boxBlurH": 120}

    step = max(1, total_frames // (sample_count + 1))
    detected_boxes = []

    for i in range(1, sample_count + 1):
        frame_idx = i * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        crop_y_start = int(height * 0.50)
        crop_bottom = frame[crop_y_start:, :]

        try:
            result, _ = engine(crop_bottom)
        except Exception:
            continue
        if not result:
            continue

        for item in result:
            box = item[0]
            text = item[1]
            score = float(item[2])

            if score < 0.48 or len(text) < 2:
                continue

            xs = [p[0] for p in box]
            ys = [p[1] + crop_y_start for p in box]

            box_x = min(xs)
            box_y = min(ys)
            box_w = max(xs) - min(xs)
            box_h = max(ys) - min(ys)

            if 20 <= box_h <= 160 and box_w >= 100 and (box_y / height) >= 0.52:
                detected_boxes.append({
                    "text": text,
                    "x": box_x,
                    "y": box_y,
                    "w": box_w,
                    "h": box_h,
                    "y_center": box_y + box_h / 2.0,
                    "y_percent": (box_y + box_h / 2.0) / height * 100.0,
                    "score": score
                })

    cap.release()

    if not detected_boxes:
        return {
            "has_hardsub": False,
            "yPercent": 75.0,
            "yCenterPx": int(height * 0.75),
            "avgHeightPx": 60,
            "boxBlurY": int(height * 0.75 - 60),
            "boxBlurH": 120
        }

    avg_y_center = sum(b["y_center"] for b in detected_boxes) / len(detected_boxes)
    avg_h = sum(b["h"] for b in detected_boxes) / len(detected_boxes)
    avg_y_percent = (avg_y_center / height) * 100.0

    blur_h = int(avg_h + 40)
    blur_y = max(0, int(avg_y_center - blur_h / 2.0))

    return {
        "has_hardsub": True,
        "sampleCount": len(detected_boxes),
        "yPercent": round(avg_y_percent, 2),
        "yCenterPx": int(avg_y_center),
        "avgHeightPx": int(avg_h),
        "boxBlurY": blur_y,
        "boxBlurH": blur_h
    }


def attach_dynamic_subtitle_coordinates(
    video_path: Path,
    segments: list[dict[str, Any]],
    roi: dict[str, Any]
) -> list[dict[str, Any]]:
    """Định vị tọa độ Y thực tế của phụ đề gốc từng câu thoại."""
    try:
        import cv2
        import re
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        return segments

    global _rapidocr_engine
    if _rapidocr_engine is None:
        try:
            _rapidocr_engine = RapidOCR()
        except Exception:
            return segments
    ocr = _rapidocr_engine

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = total_frames / fps if fps > 0 else 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1920
    default_y = int(roi.get("yCenterPx", int(height * 0.75)))
    crop_y_start = int(height * 0.40)

    for seg in segments:
        st_s = seg.get("startMs", 0) / 1000.0
        et_s = seg.get("endMs", 0) / 1000.0
        t_mid = (st_s + et_s) / 2.0
        if t_mid >= dur:
            t_mid = max(0.0, dur - 0.2)

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_mid * fps))
        ret, frame = cap.read()
        if not ret or frame is None:
            seg["yCenterPx"] = default_y
            continue

        crop = frame[crop_y_start:, :]
        try:
            res, _ = ocr(crop)
        except Exception:
            res = None

        has_zh = res and any(re.search(r'[一-鿿]', item[1]) for item in res)
        if not has_zh:
            crop_flipped = cv2.flip(crop, 1)
            try:
                res_flipped, _ = ocr(crop_flipped)
            except Exception:
                res_flipped = None
            if res_flipped and any(re.search(r'[一-鿿]', item[1]) for item in res_flipped):
                res = res_flipped

        if not res:
            seg["yCenterPx"] = None
            continue

        zh_boxes = []
        for item in res:
            text = item[1].strip()
            score = float(item[2])
            if score < 0.40 or len(text) < 1:
                continue
            if not re.search(r'[一-鿿]', text):
                continue
            ys = [p[1] + crop_y_start for p in item[0]]
            min_y = min(ys)
            max_y = max(ys)
            h = max_y - min_y
            if 20 <= h <= 180:
                center_y = int((min_y + max_y) / 2.0)
                zh_boxes.append((score, center_y, h))

        if zh_boxes:
            zh_boxes.sort(key=lambda b: -b[0])
            best_score, best_cy, best_h = zh_boxes[0]
            seg["yCenterPx"] = best_cy
            seg["boxHeightPx"] = best_h
        else:
            seg["yCenterPx"] = None

    cap.release()

    valid_ys = [s["yCenterPx"] for s in segments if s.get("yCenterPx") is not None]
    if valid_ys:
        median_y = int(sorted(valid_ys)[len(valid_ys) // 2])
        for s in segments:
            if s.get("yCenterPx") is None:
                s["yCenterPx"] = median_y
    else:
        fallback_y = int(roi.get("yCenterPx", int(height * 0.50)))
        if fallback_y > int(height * 0.65):
            fallback_y = int(height * 0.50)
        for s in segments:
            if s.get("yCenterPx") is None:
                s["yCenterPx"] = fallback_y

    return segments


def generate_channel_logo_badge(
    avatar_path: Path | str | None,
    output_path: Path,
    size_px: int = 200,
    opacity: float = 0.70,
    border_w: int = 5
) -> Path | None:
    """Tự động cắt tròn và làm mờ Avatar kênh theo chuẩn nhận diện thương hiệu."""
    if not avatar_path or not Path(avatar_path).exists():
        return None
    try:
        from PIL import Image, ImageDraw, ImageOps
        img = Image.open(str(avatar_path)).convert("RGBA")
        size = (size_px, size_px)
        img = img.resize(size, Image.Resampling.LANCZOS)

        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, size[0], size[1]), fill=255)

        circular_img = ImageOps.fit(img, size, centering=(0.5, 0.5))
        circular_img.putalpha(mask)

        border_img = Image.new("RGBA", (size[0] + border_w * 2, size[1] + border_w * 2), (0, 0, 0, 0))
        draw_b = ImageDraw.Draw(border_img)
        draw_b.ellipse((0, 0, border_img.width - 1, border_img.height - 1), fill=(255, 255, 255, 255))
        border_img.paste(circular_img, (border_w, border_w), circular_img)

        for x in range(border_img.width):
            for y in range(border_img.height):
                r, g, b, a = border_img.getpixel((x, y))
                if a > 0:
                    border_img.putpixel((x, y), (r, g, b, int(a * opacity)))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        border_img.save(str(output_path), "PNG")
        return output_path
    except Exception as exc:
        logger.warning("Lỗi sinh avatar badge: %s", exc)
        return None


CTA_VARIATIONS: dict[str, list[str]] = {
    "page_goc_tro_bat_on": [
        "Xem cuốn thế này mà tiếc cái nút follow hả?!",
        "Không follow lẹ là trọ bất ổn giống vậy đó nha!",
        "Xem chùa không follow là tối trọ bất ổn nha!"
    ],
    "page_giai_cuu_chuong_lon": [
        "Không follow coi chừng phòng thành chuồng lợn nha!",
        "Xem cuốn thế này mà tiếc nút follow hả trời?!",
        "Bấm follow lẹ đi, tiếc gì một cái click hả?!"
    ]
}


def get_raw_cta_video_path(channel_profile: str) -> Path | None:
    return None


def build_dynamic_cta_clip(
    channel_profile: str,
    voice_name: str,
    settings: Settings,
    output_path: Path,
    variation_idx: int | None = None
) -> tuple[Path | None, float]:
    """Tạo video CTA động nếu có raw CTA video."""
    raw_cta = get_raw_cta_video_path(channel_profile)
    if not raw_cta or not Path(raw_cta).exists():
        return None, 0.0

    key = "page_goc_tro_bat_on" if "goc_tro" in channel_profile.lower() else "page_giai_cuu_chuong_lon"
    var_list = CTA_VARIATIONS.get(key, CTA_VARIATIONS["page_giai_cuu_chuong_lon"])
    if variation_idx is not None:
        cta_text = var_list[variation_idx % len(var_list)]
    else:
        import random
        cta_text = random.choice(var_list)

    temp_audio = output_path.parent / f"temp_cta_voice_{output_path.stem}.mp3"
    synthesize(cta_text, voice_name, temp_audio, settings)
    if not temp_audio.is_file() or temp_audio.stat().st_size == 0:
        return None, 0.0

    dur_voice = ffprobe_duration(temp_audio)
    target_dur = max(1.8, min(dur_voice + 0.15, 2.2))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-t", f"{target_dur:.3f}", "-i", str(raw_cta),
        "-i", str(temp_audio),
        "-filter_complex",
        f"[0:v]scale=1080:1920,fps=30[v];[1:a]apad,atrim=0:{target_dur:.3f},loudnorm=I=-14:TP=-1.5:LRA=11[a]",
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-movflags", "+faststart",
        str(output_path)
    ]
    run(cmd)
    temp_audio.unlink(missing_ok=True)

    actual_dur = ffprobe_duration(output_path) if output_path.is_file() else target_dur
    return output_path, actual_dur


def splice_cta_into_video(
    main_video: Path,
    cta_video: Path | None,
    output_path: Path,
    insert_ratio: float = 0.70,
    insert_sec: float | None = None
) -> Path:
    """Chèn clip CTA vào video chính tại thời điểm ~70% thời lượng."""
    if not cta_video or not Path(cta_video).is_file():
        shutil.copy2(main_video, output_path)
        return output_path

    total_dur = ffprobe_duration(main_video)
    if insert_sec is not None:
        actual_insert = insert_sec
    else:
        actual_insert = round(total_dur * insert_ratio, 2)

    if actual_insert >= total_dur - 2.0:
        actual_insert = max(0.0, total_dur - 2.0)

    if actual_insert <= 0.5 or actual_insert >= total_dur:
        filter_str = (
            "[0:v]scale=1080:1920,setsar=1,fps=30,setpts=PTS-STARTPTS[v0];"
            "[0:a]aresample=44100,asetpts=PTS-STARTPTS[a0];"
            "[1:v]scale=1080:1920,setsar=1,fps=30,setpts=PTS-STARTPTS[v1];"
            "[1:a]aresample=44100,asetpts=PTS-STARTPTS[a1];"
            "[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]"
        )
    else:
        filter_str = (
            f"[0:v]trim=0:{actual_insert},scale=1080:1920,setsar=1,fps=30,setpts=PTS-STARTPTS[v1];"
            f"[0:a]atrim=0:{actual_insert},aresample=44100,asetpts=PTS-STARTPTS[a1];"
            f"[1:v]scale=1080:1920,setsar=1,fps=30,setpts=PTS-STARTPTS[v2];"
            f"[1:a]aresample=44100,asetpts=PTS-STARTPTS[a2];"
            f"[0:v]trim=start={actual_insert},scale=1080:1920,setsar=1,fps=30,setpts=PTS-STARTPTS[v3];"
            f"[0:a]atrim=start={actual_insert},aresample=44100,asetpts=PTS-STARTPTS[a3];"
            f"[v1][a1][v2][a2][v3][a3]concat=n=3:v=1:a=1[v][a]"
        )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(main_video),
        "-i", str(cta_video),
        "-filter_complex", filter_str,
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20", "-threads", "2",
        "-c:a", "aac", "-movflags", "+faststart",
        str(output_path)
    ]
    run(cmd)
    return output_path


if __name__ == "__main__":
    main()
