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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import requests
try:
    from dotenv import load_dotenv
except ImportError:  # Allows `verify` to explain the environment before the virtualenv is complete.
    def load_dotenv() -> bool:
        return False

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
_nllb_components: tuple[Any, Any] | None = None
_whisper_model: Any | None = None          # Cache WhisperModel giữa các job
_rapidocr_engine: Any | None = None        # Cache RapidOCR engine giữa các segment

# Giới hạn thread cho numpy/torch/onnxruntime ngay khi import — phải set TRƯỚC khi import các lib đó
# Trên i3-1115G4 (2 core / 4 thread) dùng 2 thread là optimal, tránh context-switch thrash
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")


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
    tts_provider: str        # "edge" hoặc "capcut"
    capcut_voice: str        # voice_type trong Voice.json, VD: "BV421_vivn_streaming"
    translation_engine: str = "llm"  # "llm" hoặc "marian"
    llm_provider: str = "gemini"     # "gemini", "deepseek", "openai"
    gemini_api_key: str | None = None
    deepseek_api_key: str | None = None
    openai_api_key: str | None = None

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
            tts_provider=os.getenv("DUBVI_TTS_PROVIDER", "edge").lower(),
            capcut_voice=os.getenv("DUBVI_CAPCUT_VOICE", "BV421_vivn_streaming"),
            translation_engine=os.getenv("DUBVI_TRANSLATION_ENGINE", "llm").lower(),
            llm_provider=os.getenv("DUBVI_LLM_PROVIDER", "gemini").lower(),
            gemini_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
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
    """Quét đa kênh tự động từ Giai đoạn 1 (video reup raw/...) và đọc file .meta.json tương ứng."""
    if base_dirs is None:
        base_dirs = [
            Path(r"C:\Users\vmath\Downloads\douyinnnnnnnnnnn\video reup raw"),
            Path(r"C:\Users\vmath\Videos\douyin"),
            Path(r"C:\Users\vmath\Downloads\video douyin raw"),
            Path(r"C:\Users\vmath\Downloads")
        ]
        
    discovered: list[dict[str, Any]] = []
    seen = set()
    
    for base in base_dirs:
        if not base.is_dir():
            continue
        for p in list(base.rglob("*.mp4")) + list(base.rglob("*.mov")) + list(base.rglob("*.mkv")):
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
                        
            channel_profile = meta_data.get("channel_profile") or p.parent.name
            target_platform = meta_data.get("target_platform") or "tiktok"
            vpdq_status = meta_data.get("vpdq_status") or ("PASSED" if meta_data.get("vpdq_similarity_percent") else "UNKNOWN")
            
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


def transcribe(audio_path: Path, model_name: str, device: str) -> list[dict[str, Any]]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("Thiếu faster-whisper. Chạy `pip install -r requirements.txt`.") from exc
    global _whisper_model
    compute_type = "int8" if device == "cpu" else "float16"
    # Cache model giữa các job — tránh load lại model (~1-2 phút) mỗi lần
    if _whisper_model is None:
        _whisper_model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            cpu_threads=2,       # i3-1115G4: 2 core vật lý, 2 thread là optimal
            num_workers=1,       # 1 worker decode, không cần nhiều hơn trên CPU
        )
    model = _whisper_model
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
            text = "".join(item[1] for item in response if len(item) >= 3 and float(item[2]) >= 0.45).strip()
            confidence = sum(float(item[2]) for item in response if len(item) >= 3) / max(1, len(response))
            if text:
                evidence.setdefault(text, []).append(confidence)
        finally:
            frame_path.unlink(missing_ok=True)
    if not evidence:
        return {"text": None, "confidence": 0, "frameAgreement": 0, "candidates": []}
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
    }


def ocr_consensus(source: Path, segment: dict[str, Any], roi: dict[str, Any], samples: int) -> tuple[str | None, int]:
    details = ocr_consensus_details(source, segment, roi, samples)
    return details["text"], details["confidence"]


def translate_to_vietnamese(text: str, slot_ms: int) -> tuple[str, bool]:
    try:
        import argostranslate.translate
        translated = argostranslate.translate.translate(text, "zh", "vi")
    except Exception:
        # Argos packages do not always ship a direct Chinese-Vietnamese pair.
        # Marian downloads once into DUBVI_MODEL_DIR, then translates locally.
        try:
            global _nllb_components
            if _nllb_components is None:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
                model_name = os.getenv("DUBVI_MT_MODEL", "Helsinki-NLP/opus-mt-zh-vi")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                _nllb_components = (tokenizer, model)
            tokenizer, model = _nllb_components
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            generated = model.generate(**inputs, max_new_tokens=128)
            translated = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        except Exception as exc:
            raise RuntimeError("Dịch thất bại. Hãy cài model Argos phù hợp hoặc để Marian tải model local trong DUBVI_MODEL_DIR.") from exc
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
    last_error: Exception | None = None

    for attempt in range(1, 4):
        try:
            result = client.generate_speech(
                texts=text,
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
                raise RuntimeError(f"Không tìm thấy audio_url trong CapCut response: {payload}")
            # Download MP3 về target
            import requests as _req
            resp = _req.get(audio_url, timeout=30)
            resp.raise_for_status()
            if len(resp.content) < 1024:
                raise RuntimeError("CapCut TTS trả về file audio rỗng")
            target.write_bytes(resp.content)
            return
        except Exception as exc:
            last_error = exc
            time.sleep(attempt * 1.5)

    raise RuntimeError(
        f"CapCut TTS thất bại sau 3 lần thử. Lỗi cuối: {last_error}"
    ) from last_error


def synthesize(text: str, voice: str, target: Path, settings: Settings) -> None:
    """Wrapper chọn TTS provider theo voice được chọn hoặc DUBVI_TTS_PROVIDER trong .env.

    - Giọng bắt đầu bằng `BV` (ví dụ `BV421_vivn_streaming`, `BV007_streaming`) → CapCut TTS
    - Giọng bắt đầu bằng `vi-` (ví dụ `vi-VN-HoaiMyNeural`, `vi-VN-NamMinhNeural`) → Edge TTS
    - Nếu CapCut gặp sự cố, tự động fallback sang Edge TTS.
    """
    actual_voice = voice if voice else settings.capcut_voice
    if actual_voice.startswith("BV") or (settings.tts_provider == "capcut" and not actual_voice.startswith("vi-")):
        try:
            capcut_synthesize(text, actual_voice, target)
            return
        except Exception as exc:
            logging.getLogger("dubvi-worker").warning(
                "CapCut TTS lỗi (%s), fallback sang Edge TTS.", exc
            )
            actual_voice = "vi-VN-HoaiMyNeural"

    # Edge TTS
    edge_voice = actual_voice if actual_voice.startswith("vi-") else "vi-VN-HoaiMyNeural"
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
        "-c:a", "aac", "-shortest", "-movflags", "+faststart", str(output)
    ])
    run(command)


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
            for font_path in [r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\tahomabd.ttf", r"C:\Windows\Fonts\segoeuib.ttf"]:
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


def process_job(settings: Settings, rpc: RpcClient, job: dict[str, Any]) -> None:
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
        width, height = ffprobe_dimensions(source)
        draw_ass(reviewed_segments, width, height, config["roi"], ass_path)
        draw_srt(reviewed_segments, srt_path)
        output = job_dir / f"{source.stem}_vi_reviewed.mp4"
        render_video(source, output, ass_path, voice_inputs, config["roi"], config["audioMode"])
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
            audit = {"originalText": text, "ocrConfidence": confidence, "frameAgreement": details["frameAgreement"], "candidates": details["candidates"]}
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
    review_needed = False
    for segment in segments:
        raw_voice = job_dir / f"voice_{segment['position']:03d}_raw.mp3"
        fitted_voice = job_dir / f"voice_{segment['position']:03d}.mp3"
        synthesize(segment["translatedTextVi"], config["voice"]["name"], raw_voice, settings)
        duration_ms, needs_review = fit_voice(raw_voice, fitted_voice, segment["endMs"] - segment["startMs"], float(config["voice"]["maxTempo"]))
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

    # Áp dụng thuật toán Smart Gap Clamping để khống chế khoảng câm <= 600ms
    clamped_clips = clamp_and_bridge_audio_gaps(raw_clips, max_gap_ms=600, min_pause_ms=250)
    voice_inputs: list[tuple[Path, int]] = []
    for c in clamped_clips:
        seg = c["segment"]
        seg["startMs"] = c["startMs"]
        seg["endMs"] = c["endMs"]
        voice_inputs.append((c["fitted_voice"], c["startMs"]))

    rpc.replace_segments(job_id, segments)
    manifest_path = job_dir / "segments.json"
    manifest_path.write_text(json.dumps({"jobId": job_id, "source": str(source), "config": config, "segments": segments}, ensure_ascii=False, indent=2), encoding="utf-8")
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
    width, height = ffprobe_dimensions(source)
    draw_ass(segments, width, height, config["roi"], ass_path)
    output = job_dir / f"{source.stem}_vi.mp4"
    render_video(
        source, output, ass_path, voice_inputs, config["roi"], config["audioMode"],
        clean_bgm_path=clean_bgm_wav if clean_bgm_wav.is_file() else None
    )
    log_job(f"Render hoàn tất: {output.name}")
    rpc.report(job_id, "complete", 100, status="complete", output_path=str(output))


def check_environment(settings: Settings) -> None:
    require_binary("ffmpeg")
    require_binary("ffprobe")
    print("✓ FFmpeg và FFprobe sẵn sàng")
    print(f"✓ Media directory: {settings.media_dir}")
    print(f"✓ Output directory: {settings.output_dir}")
    print(f"✓ Model cache: {settings.model_dir}")
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


if __name__ == "__main__":
    main()
