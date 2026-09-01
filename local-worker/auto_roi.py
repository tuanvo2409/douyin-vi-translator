import re
import cv2
import subprocess
import tempfile
from pathlib import Path
from rapidocr_onnxruntime import RapidOCR

_OCR_INSTANCE = None

def get_ocr_instance():
    global _OCR_INSTANCE
    if _OCR_INSTANCE is None:
        _OCR_INSTANCE = RapidOCR()
    return _OCR_INSTANCE

def get_video_duration(video_path: Path) -> float:
    """Lấy thời lượng video chính xác bằng ffprobe."""
    try:
        res = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)
        ], capture_output=True, text=True)
        return float(res.stdout.strip())
    except Exception:
        return 60.0

def auto_detect_subtitle_roi(video_path: Path, sample_seconds=None):
    """Chỉ dò và che dải phụ đề khớp với giọng nói (Voice Subtitle) ở dải dưới (64%-82%), bỏ qua caption hook ở giữa."""
    ocr = get_ocr_instance()
    candidate_subtitles = []
    tmp_dir = Path(tempfile.gettempdir())
    
    dur = get_video_duration(video_path)
    if sample_seconds is None:
        # Lấy các mốc phân bổ thông minh theo thời lượng video
        sample_seconds = [round(dur * pct, 1) for pct in [0.08, 0.18, 0.32, 0.50, 0.70, 0.85] if dur * pct >= 0.5]
    if not sample_seconds:
        sample_seconds = [2.0, 4.0, 8.0]
    
    for sec in sample_seconds:
        frame_file = tmp_dir / f"_dubvi_roi_{int(sec*10)}.jpg"
        res = subprocess.run([
            "ffmpeg", "-y", "-ss", str(sec), "-i", str(video_path),
            "-vf", "scale=1080:-2", "-frames:v", "1", "-q:v", "3", str(frame_file)
        ], capture_output=True)
        
        if not frame_file.is_file():
            continue
            
        img = cv2.imread(str(frame_file))
        frame_file.unlink(missing_ok=True)
        if img is None:
            continue
            
        h_frame, w_frame = img.shape[:2]
        res_ocr, _ = ocr(img)
        if not res_ocr:
            continue
            
        for item in res_ocr:
            box, text, score = item[0], item[1], float(item[2])
            ymin = min(p[1] for p in box)
            ymax = max(p[1] for p in box)
            xmin = min(p[0] for p in box)
            xmax = max(p[0] for p in box)
            
            # Chỉ lấy phụ đề giọng nói ở dải dưới (63% -> 85%), bỏ qua caption tò mò/hook ở giữa
            has_chinese = bool(re.search(r'[\u4e00-\u9fff]', text))
            clean_txt = text.strip()
            if has_chinese and len(clean_txt) >= 3 and ymin >= h_frame * 0.63 and ymax <= h_frame * 0.85:
                candidate_subtitles.append({
                    "text": clean_txt,
                    "ymin_pct": ymin / h_frame * 100,
                    "ymax_pct": ymax / h_frame * 100,
                })

    if not candidate_subtitles:
        return {"xPercent": 2.0, "yPercent": 67.0, "widthPercent": 96.0, "heightPercent": 7.8, "blurPx": 24}

    min_y = min(c["ymin_pct"] for c in candidate_subtitles)
    max_y = max(c["ymax_pct"] for c in candidate_subtitles)
    
    # Khoảng đệm thở 0.8% trên và dưới để che vừa khít dải phụ đề nói
    final_ymin_pct = max(63.0, round(min_y - 0.8, 1))
    final_ymax_pct = min(85.0, round(max_y + 0.8, 1))
    h_pct = max(6.5, min(round(final_ymax_pct - final_ymin_pct, 1), 8.0))

    return {
        "xPercent": 2.0,
        "yPercent": final_ymin_pct,
        "widthPercent": 96.0,
        "heightPercent": h_pct,
        "blurPx": 24
    }


def scan_silent_subtitles(
    video_path: Path,
    roi: dict,
    existing_asr_segs: list[dict],
    step_s: float = 1.4,
    min_chars_for_tts: int = 3
) -> tuple[list[dict], list[tuple[float, float]]]:
    """
    Quét tìm các đoạn phụ đề tiếng Trung xuất hiện trên màn hình nhưng KHÔNG có tiếng nói.
    - Câu có nghĩa >= min_chars_for_tts (3-4 chữ): Trả về danh sách segment để AI dịch và lồng tiếng đọc.
    - Câu ngắn / Icon / Nhãn (< min_chars_for_tts): Trả về khoảng thời gian để CHỈ BẬT KÍNH MỜ (không đọc).
    """
    dur = get_video_duration(video_path)
    if dur <= 1.0:
        return [], []

    ocr = get_ocr_instance()
    tmp_dir = Path(tempfile.gettempdir())
    
    asr_intervals = [
        (s.get("startMs", 0) / 1000.0, s.get("endMs", 0) / 1000.0)
        for s in existing_asr_segs if s.get("endMs", 0) > s.get("startMs", 0)
    ]

    def is_in_asr(t: float) -> bool:
        return any(st - 0.3 <= t <= et + 0.3 for st, et in asr_intervals)

    num_steps = max(1, int(dur / step_s))
    timestamps = [round(i * step_s, 2) for i in range(num_steps) if round(i * step_s, 2) < dur]

    detected_silent_frames = []
    
    for t in timestamps:
        if is_in_asr(t):
            continue

        frame_file = tmp_dir / f"_dubvi_silent_{int(t*100)}.jpg"
        # Crop dải phụ đề dưới đáy màn hình
        res = subprocess.run([
            "ffmpeg", "-y", "-ss", str(t), "-i", str(video_path),
            "-vf", f"scale=1080:1920,crop=1080:{int(1920*0.25)}:0:{int(1920*0.62)}",
            "-frames:v", "1", "-q:v", "3", str(frame_file)
        ], capture_output=True)

        if not frame_file.is_file():
            continue

        img = cv2.imread(str(frame_file))
        frame_file.unlink(missing_ok=True)
        if img is None:
            continue

        res_ocr, _ = ocr(img)
        if not res_ocr:
            continue

        valid_texts = []
        for item in res_ocr:
            text = item[1].strip()
            if bool(re.search(r'[\u4e00-\u9fff]', text)) and len(text) >= 1:
                valid_texts.append(text)

        if valid_texts:
            combined_text = " ".join(valid_texts)
            detected_silent_frames.append((t, combined_text))

    if not detected_silent_frames:
        return [], []

    # Gom các frame liên tục thành các khoảng thời gian
    clusters = []
    cur_st, cur_text = detected_silent_frames[0]
    cur_et = cur_st + step_s

    for t, txt in detected_silent_frames[1:]:
        if t - cur_et <= step_s * 1.5 and (txt == cur_text or len(txt) == len(cur_text)):
            cur_et = t + step_s
        else:
            clusters.append((cur_st, cur_et, cur_text))
            cur_st, cur_text = t, txt
            cur_et = t + step_s
    clusters.append((cur_st, cur_et, cur_text))

    silent_dub_segments: list[dict] = []
    mask_only_intervals: list[tuple[float, float]] = []

    for idx, (st, et, text) in enumerate(clusters):
        # Đếm số ký tự tiếng Trung thực tế
        zh_chars = re.findall(r'[\u4e00-\u9fff]', text)
        if len(zh_chars) >= min_chars_for_tts:
            # Câu dài >= 3-4 chữ: Lồng tiếng đọc + bật mask
            st_ms = int(st * 1000)
            et_ms = max(st_ms + 1500, int(et * 1000))
            silent_dub_segments.append({
                "startMs": st_ms,
                "endMs": et_ms,
                "sourceTextZh": text,
                "asrTextZh": "",
                "isSilentSubtitle": True
            })
        else:
            # Câu ngắn / icon / sticker < 3 chữ: CHỈ BẬT MASK (không đọc)
            mask_only_intervals.append((st, et))

    return silent_dub_segments, mask_only_intervals


def merge_asr_and_ocr_segments(asr_segs: list[dict], silent_ocr_segs: list[dict]) -> list[dict]:
    """Hợp nhất các câu thoại ASR và các đoạn sub câm OCR thành 1 danh sách duy nhất theo đúng thứ tự thời gian."""
    combined = list(asr_segs) + list(silent_ocr_segs)
    combined.sort(key=lambda s: s.get("startMs", 0))
    for idx, seg in enumerate(combined):
        seg["position"] = idx
    return combined
