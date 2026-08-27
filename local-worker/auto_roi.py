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

if __name__ == "__main__":
    pass
