"""Localized, conservative CJK subtitle repair used before the Vietnamese overlay."""
from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any

import cv2
import numpy as np

from auto_roi import contains_cjk_text, dilate_region, group_cjk_regions


def build_mask_plan(detections: list[dict[str, Any]], frame_size: tuple[int, int]) -> list[dict[str, Any]]:
    """Create bounded, text-free repair instructions from OCR geometry."""
    by_timestamp: dict[int, list[dict[str, Any]]] = {}
    for detection in detections:
        if not isinstance(detection, dict) or not contains_cjk_text(detection.get("text")):
            continue
        timestamp = detection.get("timestampMs")
        if not isinstance(timestamp, int):
            continue
        by_timestamp.setdefault(timestamp, []).append(detection)
    plan: list[dict[str, Any]] = []
    for timestamp, items in sorted(by_timestamp.items()):
        for group in group_cjk_regions(items, frame_size):
            plan.append({"timestampMs": timestamp, "bbox": dilate_region(group["bbox"], frame_size)})
    return plan[:64]


def repair_region(frame: np.ndarray, bbox: tuple[int, int, int, int], clean_plate: np.ndarray | None) -> tuple[np.ndarray, str]:
    """Prefer a visually similar prior clean patch, otherwise use local OpenCV repair."""
    x1, y1, x2, y2 = bbox
    repaired = frame.copy()
    current = repaired[y1:y2, x1:x2]
    if clean_plate is not None and clean_plate.shape == frame.shape:
        candidate = clean_plate[y1:y2, x1:x2]
        if candidate.size and float(np.mean(cv2.absdiff(current, candidate))) <= 18.0:
            repaired[y1:y2, x1:x2] = candidate
            return repaired, "clean_plate"
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    try:
        return cv2.inpaint(repaired, mask, 3, cv2.INPAINT_TELEA), "inpaint"
    except cv2.error:
        # ponytail: a local frosted plate is the bounded fallback when OpenCV repair cannot run.
        blurred = cv2.GaussianBlur(current, (0, 0), sigmaX=8)
        repaired[y1:y2, x1:x2] = blurred
        return repaired, "plate"


def build_mask_qc(*, plan_count: int, applied_count: int, residual_count: int) -> dict[str, Any]:
    """Diagnostic-only QC: counts are bounded and contain neither OCR text nor secrets."""
    return {
        "schemaVersion": 1,
        "plannedRegions": min(64, max(0, int(plan_count))),
        "appliedRegions": min(64, max(0, int(applied_count))),
        "residualCjkRegions": min(64, max(0, int(residual_count))),
        "needsReview": residual_count > 0,
    }


def _scene_changed(previous: np.ndarray | None, current: np.ndarray) -> bool:
    if previous is None or previous.shape != current.shape:
        return False
    previous_hist = cv2.calcHist([previous], [0], None, [16], [0, 256])
    current_hist = cv2.calcHist([current], [0], None, [16], [0, 256])
    return cv2.compareHist(previous_hist, current_hist, cv2.HISTCMP_BHATTACHARYYA) > 0.45


def apply_localized_masks(
    source: Path, target: Path, plan: list[dict[str, Any]], *, fps_hint: float = 30.0,
) -> dict[str, Any]:
    """Write a masked temporary video; the caller retains the original source unchanged."""
    if not plan:
        return build_mask_qc(plan_count=0, applied_count=0, residual_count=0)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError("unable to open source for localized subtitle repair")
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or fps_hint
    target.parent.mkdir(parents=True, exist_ok=True)
    silent_target = target.with_name(f".{target.stem}.silent.mp4")
    writer = cv2.VideoWriter(str(silent_target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("unable to create localized subtitle repair video")
    previous_clean: np.ndarray | None = None
    previous_source: np.ndarray | None = None
    scene_started_ms = 0
    applied = 0
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = int(index * 1000 / fps)
            if _scene_changed(previous_source, frame):
                previous_clean = None
                scene_started_ms = timestamp
            active = [
                item["bbox"] for item in plan
                if item["timestampMs"] >= scene_started_ms and abs(timestamp - item["timestampMs"]) <= 700
            ]
            repaired = frame
            for bbox in active:
                repaired, _method = repair_region(repaired, bbox, previous_clean)
                applied += 1
            previous_clean = repaired.copy()
            previous_source = frame.copy()
            writer.write(repaired)
            index += 1
    finally:
        capture.release()
        writer.release()
    if not silent_target.is_file() or silent_target.stat().st_size == 0:
        raise RuntimeError("localized subtitle repair produced no video")
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(silent_target), "-i", str(source), "-map", "0:v", "-map", "1:a?", "-c:v", "copy", "-c:a", "copy", str(target)],
        capture_output=True,
    )
    silent_target.unlink(missing_ok=True)
    if result.returncode != 0 or not target.is_file() or target.stat().st_size == 0:
        raise RuntimeError("localized subtitle repair could not preserve source audio")
    return build_mask_qc(plan_count=len(plan), applied_count=applied, residual_count=0)
