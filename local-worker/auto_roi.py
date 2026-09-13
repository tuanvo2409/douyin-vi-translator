import re
import cv2
import subprocess
import tempfile
from pathlib import Path
from rapidocr_onnxruntime import RapidOCR

_OCR_INSTANCE = None
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def contains_cjk_text(text: object) -> bool:
    """Return whether untrusted OCR text contains a Han character."""
    return isinstance(text, str) and bool(_CJK_RE.search(text))


def _clamp_bbox(bbox: tuple[int, int, int, int], frame_size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = frame_size
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(width, int(x1))), max(0, min(height, int(y1))),
        max(0, min(width, int(x2))), max(0, min(height, int(y2))),
    )


def map_ocr_quad_to_source(
    quad: list[list[float]] | tuple[tuple[float, float], ...],
    crop_origin: tuple[int, int],
    enlargement: float,
    frame_size: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    """Map a quadrilateral from an enlarged OCR crop back to source pixels."""
    if enlargement <= 0 or len(quad) != 4:
        raise ValueError("OCR quadrilateral and enlargement are required")
    origin_x, origin_y = crop_origin
    width, height = frame_size
    mapped = []
    for point in quad:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("invalid OCR quadrilateral")
        x = max(0, min(width, int(round(float(point[0]) / enlargement + origin_x))))
        y = max(0, min(height, int(round(float(point[1]) / enlargement + origin_y))))
        mapped.append((x, y))
    return tuple(mapped)


def _bbox_from_quad(quad: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    return min(x for x, _ in quad), min(y for _, y in quad), max(x for x, _ in quad), max(y for _, y in quad)


def dilate_region(bbox: tuple[int, int, int, int], frame_size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Expand a detected subtitle region proportionally without leaving the frame."""
    x1, y1, x2, y2 = _clamp_bbox(bbox, frame_size)
    height = max(1, y2 - y1)
    horizontal = max(4, int(round(height * 0.15)))
    vertical = max(5, int(round(height * 0.25)))
    return _clamp_bbox((x1 - horizontal, y1 - vertical, x2 + horizontal, y2 + vertical), frame_size)


def group_cjk_regions(detections: list[dict], frame_size: tuple[int, int]) -> list[dict]:
    """Combine neighboring CJK OCR boxes into small subtitle masks, never a broad band."""
    eligible = []
    for detection in detections:
        bbox = detection.get("bbox") if isinstance(detection, dict) else None
        if not contains_cjk_text(detection.get("text")) or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        clamped = _clamp_bbox(tuple(int(value) for value in bbox), frame_size)
        if clamped[2] > clamped[0] and clamped[3] > clamped[1]:
            eligible.append({**detection, "bbox": clamped})
    eligible.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    groups: list[dict] = []
    for detection in eligible:
        x1, y1, x2, y2 = detection["bbox"]
        merged = None
        for group in groups:
            gx1, gy1, gx2, gy2 = group["bbox"]
            line_height = max(y2 - y1, gy2 - gy1)
            same_line = min(y2, gy2) > max(y1, gy1) and x1 - gx2 <= max(20, int(line_height * 5.5))
            stacked_lines = min(x2, gx2) > max(x1, gx1) and y1 - gy2 <= max(12, int(line_height * 1.5))
            if same_line or stacked_lines:
                merged = group
                group["bbox"] = _clamp_bbox((min(x1, gx1), min(y1, gy1), max(x2, gx2), max(y2, gy2)), frame_size)
                group["detections"].append(detection)
                break
        if merged is None:
            groups.append({"bbox": (x1, y1, x2, y2), "detections": [detection]})
    return groups


class TemporalCJKTracker:
    """Keep short OCR misses stable while prohibiting cross-scene mask reuse."""

    def __init__(self, hold_ms: int, frame_size: tuple[int, int]) -> None:
        self.hold_ms = max(0, int(hold_ms))
        self.frame_size = frame_size
        self._tracks: list[dict] = []

    def observe(self, timestamp_ms: int, regions: list[tuple[int, int, int, int]], scene_id: int) -> None:
        self._tracks = [track for track in self._tracks if track["scene_id"] == scene_id]
        for region in regions:
            clamped = _clamp_bbox(region, self.frame_size)
            matched = next((track for track in self._tracks if _overlaps(track["bbox"], clamped)), None)
            if matched is None:
                self._tracks.append({"bbox": clamped, "scene_id": scene_id, "last_seen_ms": int(timestamp_ms)})
            else:
                matched["bbox"] = clamped
                matched["last_seen_ms"] = int(timestamp_ms)

    def active_regions(self, timestamp_ms: int, scene_id: int) -> list[tuple[int, int, int, int]]:
        return [
            track["bbox"] for track in self._tracks
            if track["scene_id"] == scene_id and int(timestamp_ms) - track["last_seen_ms"] <= self.hold_ms
        ]


def _overlaps(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> bool:
    return min(left[2], right[2]) > max(left[0], right[0]) and min(left[3], right[3]) > max(left[1], right[1])


def discover_cjk_regions(samples: list[dict], frame_size: tuple[int, int]) -> list[dict]:
    """Normalize sparse whole-frame OCR records without assuming a bottom band."""
    discoveries: list[dict] = []
    for sample in samples:
        timestamp = sample.get("timestampMs") if isinstance(sample, dict) else None
        items = sample.get("items") if isinstance(sample, dict) else None
        if not isinstance(timestamp, int) or not isinstance(items, list):
            continue
        for group in group_cjk_regions(items, frame_size):
            discoveries.append({
                "timestampMs": timestamp, "bbox": group["bbox"],
                "text": "".join(str(item.get("text") or "") for item in group["detections"]),
            })
    return sorted(discoveries, key=lambda item: (item["timestampMs"], item["bbox"][1], item["bbox"][0]))


def sparse_whole_frame_discovery(video_path: Path, sample_count: int = 8) -> list[dict]:
    """Use a bounded number of downscaled full-frame OCR samples to find non-ROI CJK."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return []
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    samples: list[dict] = []
    try:
        for index in range(max(1, min(12, sample_count))):
            frame_index = int((frame_count - 1) * index / max(1, min(12, sample_count) - 1))
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))
            ok, frame = capture.read()
            if not ok:
                continue
            scale = min(1.0, 960.0 / max(1, frame.shape[1]))
            scanned = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1.0 else frame
            response, _ = get_ocr_instance()(scanned)
            items = []
            for item in response or []:
                if len(item) >= 3 and float(item[2]) >= 0.45 and contains_cjk_text(item[1]):
                    quad = map_ocr_quad_to_source(item[0], (0, 0), scale, (width, height))
                    items.append({"text": str(item[1]), "bbox": _bbox_from_quad(quad), "confidence": float(item[2])})
            samples.append({"timestampMs": int(frame_index * 1000 / fps), "items": items})
    finally:
        capture.release()
    return discover_cjk_regions(samples, (width, height))

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


def extract_frame_image(video_path: Path, timestamp_s: float, vf: str = "scale=1080:1920") -> cv2.Mat | None:
    """Helper an toàn trích xuất frame hình ảnh từ video mà không để lại rác disk."""
    tmp_dir = Path(tempfile.gettempdir())
    frame_file = tmp_dir / f"_dubvi_f_{int(timestamp_s * 1000)}.jpg"
    try:
        subprocess.run([
            "ffmpeg", "-y", "-ss", f"{timestamp_s:.2f}", "-i", str(video_path),
            "-vf", vf, "-frames:v", "1", "-q:v", "2", str(frame_file)
        ], capture_output=True)
        if frame_file.is_file():
            img = cv2.imread(str(frame_file))
            frame_file.unlink(missing_ok=True)
            return img
    except Exception:
        pass
    finally:
        frame_file.unlink(missing_ok=True)
    return None


def analyze_3s_hook(video_path: Path, asr_segs: list[dict]) -> dict:
    """
    Phân tích 3 giây đầu của video:
    - Phát hiện xem có giọng nói và chữ tiếng Trung trong 3s đầu hay không.
    - Trích xuất toạ độ dòng chữ Trung lớn nhất trong 3s đầu để cấy Thẻ Trắng và Hook.
    """
    ocr = get_ocr_instance()
    has_voice = any(s.get("startMs", 0) < 3000 and len((s.get("asrTextZh") or "").strip()) >= 2 for s in asr_segs)
    
    img = extract_frame_image(video_path, 1.5, vf="scale=1080:1920")
    detected_y_pct = 70.0
    box_w = 950
    if img is not None:
        res_ocr, _ = ocr(img)
        if res_ocr:
            for item in res_ocr:
                box, text = item[0], item[1]
                if len(text.strip()) >= 2 and re.search(r'[\u4e00-\u9fff]', text):
                    ymin = min(p[1] for p in box)
                    ymax = max(p[1] for p in box)
                    xmin = min(p[0] for p in box)
                    xmax = max(p[0] for p in box)
                    cy = (ymin + ymax) / 2 / 1920.0 * 100
                    if 55.0 <= cy <= 85.0:
                        detected_y_pct = round(cy, 1)
                        box_w = max(box_w, int(xmax - xmin + 60))
                        break

    return {
        "hasVoiceIn3s": has_voice,
        "firstSegEndMs": asr_segs[0].get("endMs", 3500) if asr_segs else 3500,
        "yPercent": detected_y_pct,
        "boxWidth": box_w
    }


def auto_detect_subtitle_roi(video_path: Path, sample_seconds=None):
    """Chỉ dò và che dải phụ đề khớp với giọng nói (Voice Subtitle) ở dải dưới (64%-82%), bỏ qua caption hook ở giữa."""
    ocr = get_ocr_instance()
    candidate_subtitles = []
    
    dur = get_video_duration(video_path)
    if sample_seconds is None:
        sample_seconds = [round(dur * pct, 1) for pct in [0.08, 0.18, 0.32, 0.50, 0.70, 0.85] if dur * pct >= 0.5]
    if not sample_seconds:
        sample_seconds = [2.0, 4.0, 8.0]
    
    for sec in sample_seconds:
        img = extract_frame_image(video_path, sec, vf="scale=1080:-2")
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
            
            has_chinese = bool(re.search(r'[\u4e00-\u9fff]', text))
            clean_txt = text.strip()
            if has_chinese and len(clean_txt) >= 3 and ymin >= h_frame * 0.63 and ymax <= h_frame * 0.85:
                candidate_subtitles.append({
                    "text": clean_txt,
                    "ymin_pct": ymin / h_frame * 100,
                    "ymax_pct": ymax / h_frame * 100,
                })

    if not candidate_subtitles:
        return {"xPercent": 2.0, "yPercent": 64.5, "widthPercent": 96.0, "heightPercent": 14.0, "blurPx": 24}

    min_y = min(c["ymin_pct"] for c in candidate_subtitles)
    max_y = max(c["ymax_pct"] for c in candidate_subtitles)

    # Khoảng đệm thở rộng trên và dưới để che phủ 100% mọi chân chữ, dấu câu và emoji
    final_ymin_pct = max(62.5, round(min_y - 2.0, 1))
    final_ymax_pct = min(86.0, round(max_y + 4.5, 1))
    h_pct = max(13.8, min(round(final_ymax_pct - final_ymin_pct, 1), 16.0))

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
