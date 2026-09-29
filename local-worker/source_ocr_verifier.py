"""Conservative, local ASR/OCR source cross-check for recurring dialogue captions."""
from __future__ import annotations

import re
import subprocess
import tempfile
from difflib import SequenceMatcher
from pathlib import Path
from typing import Mapping, Sequence

import cv2

from auto_roi import get_ocr_instance

_HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def _han(value: object) -> str:
    return "".join(_HAN.findall(value if isinstance(value, str) else ""))


def infer_stable_subtitle_roi(
    detections: Sequence[Mapping[str, object]], frame_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """Use recurring lower-frame CJK detections; never infer from one frame."""
    width, height = frame_size
    if width <= 0 or height <= 0:
        return None
    candidates = []
    for item in detections:
        if str(item.get("category") or "UNKNOWN").upper() in {"STORY_TEXT", "WATERMARK", "PERSISTENT_WATERMARK"}:
            continue
        bbox = item.get("bbox")
        timestamp = item.get("timestampMs")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or not isinstance(timestamp, int) or not _han(item.get("text")):
            continue
        try:
            left, top, right, bottom = (int(value) for value in bbox)
        except (TypeError, ValueError):
            continue
        if 0 <= left < right <= width and height * 0.40 <= top < bottom <= height:
            candidates.append((timestamp, top, bottom, _han(item.get("text"))))
    if not candidates:
        return None
    center = sorted((top + bottom) / 2 for _, top, bottom, _ in candidates)[len(candidates) // 2]
    stable = [(time_ms, top, bottom, text) for time_ms, top, bottom, text in candidates
              if abs((top + bottom) / 2 - center) <= height * 0.06]
    if len({time_ms for time_ms, _, _, _ in stable}) < 3 or len({text for _, _, _, text in stable}) < 3:
        return None
    y1 = max(0, min(top for _, top, _, _ in stable) - int(height * 0.09))
    y2 = min(height, max(bottom for _, _, bottom, _ in stable) + int(height * 0.05))
    return (int(width * 0.1), y1, int(width * 0.9), y2)


def scan_cue_ocr(
    video_path: Path, cues: Sequence[Mapping[str, object]], roi: tuple[int, int, int, int],
    *, detections: Sequence[Mapping[str, object]] | None = None,
) -> dict[int, list[dict[str, object]]]:
    """Read two color frames per cue with one shared RapidOCR instance."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("source video cannot be opened for cue OCR")
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 576
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1024
    engine = get_ocr_instance()
    result: dict[int, list[dict[str, object]]] = {}
    x1, y1, x2, y2 = roi
    try:
        for cue in cues:
            position = int(cue["position"])
            start, end = int(cue["start_ms"]), int(cue["end_ms"])
            readings = []
            for fraction in (0.35, 0.65):
                time_ms = int(start + (end - start) * fraction)
                capture.set(cv2.CAP_PROP_POS_MSEC, time_ms)
                ok, frame = capture.read()
                if not ok:
                    continue
                h_f, w_f = frame.shape[:2]
                f_w, f_h = (w_f or frame_width), (h_f or frame_height)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                image = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                response, _ = engine(image)
                candidates = []
                for item in response or []:
                    if len(item) < 3 or not _han(item[1]):
                        continue
                    text = str(item[1]).strip()
                    conf = float(item[2])
                    quad = item[0]
                    if isinstance(quad, (list, tuple)) and len(quad) == 4:
                        try:
                            bx1 = max(0, min(f_w, int(round(min(p[0] for p in quad) / 3.0 + x1))))
                            by1 = max(0, min(f_h, int(round(min(p[1] for p in quad) / 3.0 + y1))))
                            bx2 = max(0, min(f_w, int(round(max(p[0] for p in quad) / 3.0 + x1))))
                            by2 = max(0, min(f_h, int(round(max(p[1] for p in quad) / 3.0 + y1))))
                            bbox = [bx1, by1, bx2, by2]
                        except Exception:
                            bbox = [x1, y1, x2, y2]
                    else:
                        bbox = [x1, y1, x2, y2]

                    category = "UNKNOWN"
                    is_story_or_watermark = False
                    if detections:
                        for d in detections:
                            d_cat = str(d.get("category") or "").upper()
                            if d_cat in {"STORY_TEXT", "WATERMARK", "PERSISTENT_WATERMARK"}:
                                d_box = d.get("bbox")
                                if isinstance(d_box, (list, tuple)) and len(d_box) == 4:
                                    if max(bbox[0], d_box[0]) < min(bbox[2], d_box[2]) and max(bbox[1], d_box[1]) < min(bbox[3], d_box[3]):
                                        is_story_or_watermark = True
                                        break
                    if not is_story_or_watermark:
                        if bbox[1] >= y1 - 25 and bbox[3] <= y2 + 25 and bbox[1] >= int(f_h * 0.35):
                            category = "DIALOGUE_SUBTITLE"

                    candidates.append({
                        "text": text,
                        "confidence": conf,
                        "timestampMs": time_ms,
                        "cue_start_ms": start,
                        "cue_end_ms": end,
                        "bbox": bbox,
                        "category": category,
                    })
                if candidates:
                    readings.append(max(candidates, key=lambda item: len(_han(item["text"])) * item["confidence"]))
            result[position] = readings
    finally:
        capture.release()
    return result


def scan_second_asr(
    audio_path: Path, cues: Sequence[Mapping[str, object]],
    readings: Mapping[int, Sequence[Mapping[str, object]]], exe: Path, model: Path,
) -> dict[int, str]:
    """Use the optional local SenseVoice model only for cue-sized OCR/ASR disputes."""
    if not audio_path.is_file() or not exe.is_file() or not model.is_file():
        raise FileNotFoundError("SenseVoice audio, executable, or model is missing")
    result: dict[int, str] = {}
    with tempfile.TemporaryDirectory(prefix="dubvi-source-ocr-") as directory:
        clip = Path(directory) / "cue.wav"
        for cue in cues:
            position = int(cue["position"])
            pair = readings.get(position, [])
            if len(pair) != 2 or _han(pair[0].get("text")) != _han(pair[1].get("text")):
                continue
            if not decide_source(str(cue["asr_text_zh"]), pair)["needsReview"]:
                continue
            start = max(0, int(cue["start_ms"]) / 1000 - 0.15)
            duration = max(0.1, (int(cue["end_ms"]) - int(cue["start_ms"])) / 1000 + 0.3)
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(start),
                 "-t", str(duration), "-i", str(audio_path), "-ac", "1", "-ar", "16000", str(clip)],
                check=True, timeout=30, capture_output=True,
            )
            response = subprocess.run(
                [str(exe), "-m", str(model), "-a", str(clip), "--backend", "cpu"],
                check=True, timeout=30, capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            result[position] = response.stdout.strip()
    return result


def decide_source(
    asr_text: str, readings: Sequence[Mapping[str, object]], *, second_asr: str | None = None,
    cue_timing: tuple[int, int] | None = None, roi: tuple[int, int, int, int] | None = None,
) -> dict[str, object]:
    """Promote OCR only when independent ASR evidence or verified repeated dialogue OCR supports it."""
    normalized_asr = _han(asr_text)
    valid = [item for item in readings if _han(item.get("text")) and isinstance(item.get("confidence"), (int, float))]
    if not valid:
        return {"resolvedZh": asr_text, "ocrZh": "", "decision": "ASR_ONLY", "needsReview": True}
    normalized = [_han(item["text"]) for item in valid]
    best = max(valid, key=lambda item: len(_han(item["text"])) * float(item["confidence"]))
    ocr_text = str(best["text"])
    repeated = len(valid) == 2 and normalized[0] == normalized[1] and all(float(item["confidence"]) >= 0.9 for item in valid)
    digits_match = re.findall(r"\d", ocr_text) == re.findall(r"\d", asr_text)
    if repeated and normalized[0] == normalized_asr and digits_match:
        return {"resolvedZh": asr_text, "ocrZh": ocr_text, "decision": "ASR_OCR_AGREE", "needsReview": False}
    if repeated and normalized_asr and digits_match and _han(second_asr) == normalized[0]:
        ratio = len(normalized[0]) / len(normalized_asr)
        similarity = SequenceMatcher(None, normalized_asr, normalized[0]).ratio()
        if 0.7 <= ratio <= 1.3 and similarity >= 0.55:
            return {"resolvedZh": ocr_text, "ocrZh": ocr_text, "decision": "OCR_SECOND_ASR_AGREE", "needsReview": False}

    if repeated and digits_match:
        r0, r1 = valid[0], valid[1]
        t0 = r0.get("timestampMs")
        t1 = r1.get("timestampMs")
        has_distinct_timestamps = (
            isinstance(t0, int) and isinstance(t1, int)
            and not isinstance(t0, bool) and not isinstance(t1, bool)
            and abs(t0 - t1) >= 10
        )
        timestamps_in_cue = has_distinct_timestamps
        if has_distinct_timestamps:
            if cue_timing is not None:
                c_start, c_end = cue_timing
                if not (c_start <= t0 <= c_end and c_start <= t1 <= c_end):
                    timestamps_in_cue = False
            else:
                c_start0 = r0.get("cue_start_ms")
                c_end0 = r0.get("cue_end_ms")
                if isinstance(c_start0, int) and isinstance(c_end0, int):
                    if not (c_start0 <= t0 <= c_end0 and c_start0 <= t1 <= c_end0):
                        timestamps_in_cue = False

        cat0 = str(r0.get("category") or r0.get("role") or "").strip().upper()
        cat1 = str(r1.get("category") or r1.get("role") or "").strip().upper()
        role_is_dialogue = (cat0 == "DIALOGUE_SUBTITLE" and cat1 == "DIALOGUE_SUBTITLE")

        b0 = r0.get("bbox")
        b1 = r1.get("bbox")
        valid_bbox = (
            isinstance(b0, (list, tuple)) and len(b0) == 4
            and isinstance(b1, (list, tuple)) and len(b1) == 4
            and b0[2] > b0[0] and b0[3] > b0[1]
            and b1[2] > b1[0] and b1[3] > b1[1]
        )
        if valid_bbox and roi is not None:
            rx1, ry1, rx2, ry2 = roi
            if not (b0[1] >= ry1 - 25 and b0[3] <= ry2 + 25 and b1[1] >= ry1 - 25 and b1[3] <= ry2 + 25):
                valid_bbox = False

        len_ocr = len(normalized[0])
        len_asr = len(normalized_asr)
        ratio_ok = True
        similarity_ok = True
        if len_asr > 0:
            ratio = len_ocr / len_asr
            if ratio < 0.65 or ratio > 1.8:
                ratio_ok = False
            similarity = SequenceMatcher(None, normalized_asr, normalized[0]).ratio()
            if similarity < 0.40:
                similarity_ok = False
        elif len_ocr == 0:
            ratio_ok = False

        if (
            has_distinct_timestamps
            and timestamps_in_cue
            and role_is_dialogue
            and valid_bbox
            and ratio_ok
            and similarity_ok
        ):
            return {"resolvedZh": ocr_text, "ocrZh": ocr_text, "decision": "OCR_REPEAT", "needsReview": False}

    return {"resolvedZh": asr_text, "ocrZh": ocr_text, "decision": "SOURCE_REVIEW_REQUIRED", "needsReview": True}
