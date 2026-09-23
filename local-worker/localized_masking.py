"""Localized, conservative CJK subtitle repair used before the Vietnamese overlay."""
from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any

import cv2
import numpy as np

from auto_roi import TemporalCJKTracker, contains_cjk_text, dilate_region, group_cjk_regions

_MASK_CATEGORIES = {"DIALOGUE_SUBTITLE", "WATERMARK"}
_CATEGORY_ALIASES = {
    "CHINESE_DIALOGUE_SUBTITLE": "DIALOGUE_SUBTITLE",
    "PERSISTENT_WATERMARK": "WATERMARK",
}


def _normalize_category(value: object) -> str | None:
    if value is None:
        return None
    category = str(value).strip().upper()
    return _CATEGORY_ALIASES.get(category, category)


def build_mask_plan(detections: list[dict[str, Any]], frame_size: tuple[int, int]) -> list[dict[str, Any]]:
    """Create bounded, text-free repair instructions from OCR geometry."""
    by_timestamp: dict[int, list[dict[str, Any]]] = {}
    for detection in detections:
        if (
            not isinstance(detection, dict)
            or not contains_cjk_text(detection.get("text"))
            or _normalize_category(detection.get("category")) not in (None, *_MASK_CATEGORIES)
        ):
            continue
        timestamp = detection.get("timestampMs")
        if not isinstance(timestamp, int):
            continue
        by_timestamp.setdefault(timestamp, []).append(detection)
    plan: list[dict[str, Any]] = []
    for timestamp, items in sorted(by_timestamp.items()):
        for group in group_cjk_regions(items, frame_size):
            item: dict[str, Any] = {"timestampMs": timestamp, "bbox": dilate_region(group["bbox"], frame_size)}
            categories = {_normalize_category(detection.get("category")) for detection in group["detections"]}
            if len(categories) == 1 and None not in categories:
                item["category"] = categories.pop()
            plan.append(item)
    return plan


def stabilize_mask_plan(
    plan: list[dict[str, Any]], frame_size: tuple[int, int], *, lead_ms: int = 250, tail_ms: int = 500,
    max_gap_ms: int = 750,
) -> list[dict[str, Any]]:
    """Turn nearby matching OCR snapshots into clipped, short-lived mask intervals."""
    if min(lead_ms, tail_ms, max_gap_ms) < 0:
        raise ValueError("mask stabilization values must be non-negative")
    tracks: list[dict[str, Any]] = []
    for item in sorted(plan, key=lambda value: int(value["timestampMs"])):
        category = _normalize_category(item.get("category"))
        if category not in (None, *_MASK_CATEGORIES):
            continue
        timestamp = _mask_time(item, "timestampMs")
        bbox = _mask_bbox(item, frame_size)
        track = next((candidate for candidate in tracks if (
            candidate["category"] == category
            and timestamp - candidate["lastMs"] <= max_gap_ms
            and _intersects(candidate["bbox"], bbox)
        )), None)
        if track is None:
            tracks.append({"firstMs": timestamp, "lastMs": timestamp, "bbox": bbox, "category": category})
            continue
        track["lastMs"] = timestamp
        track["bbox"] = _clip_bbox((
            min(track["bbox"][0], bbox[0]), min(track["bbox"][1], bbox[1]),
            max(track["bbox"][2], bbox[2]), max(track["bbox"][3], bbox[3]),
        ), frame_size)
    stabilized: list[dict[str, Any]] = []
    for track in sorted(tracks, key=lambda value: (value["firstMs"], value["bbox"], value["category"] or "")):
        item = {
            "startMs": max(0, track["firstMs"] - lead_ms),
            "endMs": track["lastMs"] + tail_ms,
            "bbox": track["bbox"],
        }
        if track["category"] is not None:
            item["category"] = track["category"]
        stabilized.append(item)
    return stabilized


def _build_text_repair_mask(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Find bright text-like pixels inside one evidence box, never the whole box."""
    x1, y1, x2, y2 = bbox
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return np.zeros(frame.shape[:2], dtype=np.uint8)
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    local_background = cv2.GaussianBlur(gray, (0, 0), sigmaX=3)
    contrast = gray.astype(np.int16) - local_background.astype(np.int16)
    # OCR already supplied the evidence box; local contrast finds dim or compressed
    # glyphs without treating the complete box as a solid repair region.
    candidate = contrast >= max(10.0, float(np.percentile(contrast, 75)))
    mask = (candidate.astype(np.uint8) * 255)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), dtype=np.uint8))
    outline_radius = max(2, min(7, int(round((y2 - y1) * 0.25))))
    mask = cv2.dilate(mask, np.ones((outline_radius * 2 + 1, outline_radius * 2 + 1), dtype=np.uint8), iterations=1)
    full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = mask
    return full_mask


def repair_region(
    frame: np.ndarray, bbox: tuple[int, int, int, int], clean_plate: np.ndarray | None,
    *, category: str | None = None,
) -> tuple[np.ndarray, str]:
    """Use an explicit clean plate only when supplied; normal masking repairs text pixels locally."""
    x1, y1, x2, y2 = _clip_bbox(bbox, (frame.shape[1], frame.shape[0]))
    repaired = frame.copy()
    if x2 <= x1 or y2 <= y1:
        return repaired, "unchanged"
    current = repaired[y1:y2, x1:x2]
    if clean_plate is not None and clean_plate.shape == frame.shape:
        candidate = clean_plate[y1:y2, x1:x2]
        if candidate.size and float(np.mean(cv2.absdiff(current, candidate))) <= 18.0:
            repaired[y1:y2, x1:x2] = candidate
            return repaired, "clean_plate"
    normalized_category = _normalize_category(category)
    if normalized_category == "WATERMARK":
        padding = max(4, min(8, int(round((y2 - y1) * 0.2))))
        wx1, wy1, wx2, wy2 = _clip_bbox(
            (x1 - padding, y1 - padding, x2 + padding, y2 + padding),
            (frame.shape[1], frame.shape[0]),
        )
        context = padding
        cx1, cy1, cx2, cy2 = _clip_bbox(
            (wx1 - context, wy1 - context, wx2 + context, wy2 + context),
            (frame.shape[1], frame.shape[0]),
        )
        crop = repaired[cy1:cy2, cx1:cx2]
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        mask[wy1 - cy1:wy2 - cy1, wx1 - cx1:wx2 - cx1] = 255
        try:
            repaired[cy1:cy2, cx1:cx2] = cv2.inpaint(crop, mask, 30, cv2.INPAINT_TELEA)
            return repaired, "inpaint"
        except cv2.error:
            blurred = cv2.GaussianBlur(crop, (0, 0), sigmaX=8)
            crop[mask > 0] = blurred[mask > 0]
            repaired[cy1:cy2, cx1:cx2] = crop
            return repaired, "plate"
    else:
        mask = _build_text_repair_mask(repaired, (x1, y1, x2, y2))
    if not np.any(mask):
        return repaired, "unchanged"
    try:
        radius = 30 if normalized_category == "WATERMARK" else 5
        return cv2.inpaint(repaired, mask, radius, cv2.INPAINT_TELEA), "inpaint"
    except cv2.error:
        # ponytail: a local frosted plate is the bounded fallback when OpenCV repair cannot run.
        blurred = cv2.GaussianBlur(repaired, (0, 0), sigmaX=8)
        repaired[mask > 0] = blurred[mask > 0]
        return repaired, "plate"


def build_mask_qc(*, plan_count: int, applied_count: int, residual_count: int) -> dict[str, Any]:
    """Diagnostic-only QC: counts are bounded and contain neither OCR text nor secrets."""
    return {
        "schemaVersion": 1,
        "plannedRegions": max(0, int(plan_count)),
        "appliedRegions": max(0, int(applied_count)),
        "residualCjkRegions": max(0, int(residual_count)),
        "needsReview": residual_count > 0,
    }


def _scene_changed(previous: np.ndarray | None, current: np.ndarray) -> bool:
    if previous is None or previous.shape != current.shape:
        return False
    previous_hist = cv2.calcHist([previous], [0], None, [16], [0, 256])
    current_hist = cv2.calcHist([current], [0], None, [16], [0, 256])
    return cv2.compareHist(previous_hist, current_hist, cv2.HISTCMP_BHATTACHARYYA) > 0.45


def stratified_sample_indices(total: int, limit: int) -> list[int]:
    if total <= 0 or limit <= 0:
        return []
    count = min(total, limit)
    return sorted({int(round((total - 1) * index / max(1, count - 1))) for index in range(count)})


def expand_affected_plan(plan: list[dict[str, Any]], residual: list[dict[str, Any]], frame_size: tuple[int, int]) -> list[dict[str, Any]]:
    """Build one conservative retry plan only for residual CJK tracks."""
    retry: list[dict[str, Any]] = []
    for item in build_mask_plan(residual, frame_size):
        x1, y1, x2, y2 = item["bbox"]
        retry.append({"timestampMs": item["timestampMs"], "bbox": dilate_region((x1 - 4, y1 - 4, x2 + 4, y2 + 4), frame_size)})
    return retry


def lossless_transport_command(source: str, target: str, width: int, height: int, fps: float) -> list[str]:
    return [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-video_size", f"{width}x{height}",
        "-framerate", f"{fps:.6f}", "-i", "pipe:0", "-i", source, "-map", "0:v", "-map", "1:a?",
        "-c:v", "ffv1", "-level", "3", "-c:a", "copy", target,
    ]


def apply_localized_masks(
    source: Path, target: Path, plan: list[dict[str, Any]], *, fps_hint: float = 30.0,
) -> dict[str, Any]:
    """Write a masked temporary video; the caller retains the original source unchanged."""
    if not plan:
        return build_mask_qc(plan_count=0, applied_count=0, residual_count=0)
    point_plan, interval_plan = _partition_mask_plan(plan)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError("unable to open source for localized subtitle repair")
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or fps_hint
    target.parent.mkdir(parents=True, exist_ok=True)
    previous_source: np.ndarray | None = None
    tracker = TemporalCJKTracker(hold_ms=900, frame_size=(width, height))
    plan_index = 0
    ordered_plan = sorted(point_plan, key=lambda item: _mask_time(item, "timestampMs"))
    scene_id = 0
    applied = 0
    index = 0
    transport: subprocess.Popen[bytes] | None = None
    try:
        transport = subprocess.Popen(
            lossless_transport_command(str(source), str(target), width, height, fps), stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if transport.stdin is None:
            raise RuntimeError("unable to create localized subtitle repair transport")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = int(index * 1000 / fps)
            if _scene_changed(previous_source, frame):
                scene_id += 1
            while plan_index < len(ordered_plan) and int(ordered_plan[plan_index]["timestampMs"]) <= timestamp:
                item = ordered_plan[plan_index]
                tracker.observe(int(item["timestampMs"]), [item["bbox"]], scene_id)
                plan_index += 1
            active = [(bbox, None) for bbox in tracker.active_regions(timestamp, scene_id)] + [
                (item["bbox"], item.get("category"))
                for item in interval_plan if item["startMs"] <= timestamp <= item["endMs"]
            ]
            repaired = frame
            for bbox, category in _unique_regions(active):
                repaired, _method = repair_region(repaired, bbox, None, category=category)
                applied += 1
            previous_source = frame.copy()
            transport.stdin.write(repaired.tobytes())
            index += 1
    finally:
        capture.release()
        if transport is not None and transport.stdin is not None:
            transport.stdin.close()
    if transport is None or transport.wait() != 0:
        raise RuntimeError("localized subtitle repair produced no video")
    if not target.is_file() or target.stat().st_size == 0:
        raise RuntimeError("localized subtitle repair produced no video")
    return build_mask_qc(plan_count=len(plan), applied_count=applied, residual_count=0)


def _partition_mask_plan(plan: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    points: list[dict[str, Any]] = []
    intervals: list[dict[str, Any]] = []
    for item in plan:
        if not isinstance(item, dict):
            raise ValueError("mask plan item must be an object")
        category = _normalize_category(item.get("category"))
        if category not in (None, *_MASK_CATEGORIES):
            continue
        bbox = item.get("bbox")
        if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
            raise ValueError("mask plan bbox must be a four-value tuple")
        if "timestampMs" in item:
            point = {"timestampMs": _mask_time(item, "timestampMs"), "bbox": tuple(bbox)}
            if category is not None:
                point["category"] = category
            points.append(point)
        elif "startMs" in item and "endMs" in item:
            start, end = _mask_time(item, "startMs"), _mask_time(item, "endMs")
            if end < start:
                raise ValueError("mask interval end must not precede start")
            interval = {"startMs": start, "endMs": end, "bbox": tuple(bbox)}
            if category is not None:
                interval["category"] = category
            intervals.append(interval)
        else:
            raise ValueError("mask plan needs timestampMs or startMs/endMs")
    return points, intervals


def _mask_time(item: dict[str, Any], field: str) -> int:
    value = item.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _mask_bbox(item: dict[str, Any], frame_size: tuple[int, int]) -> tuple[int, int, int, int]:
    bbox = item.get("bbox")
    if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
        raise ValueError("mask plan bbox must be a four-value tuple")
    clipped = _clip_bbox(bbox, frame_size)
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError("mask plan bbox must have area")
    return clipped


def _clip_bbox(bbox: tuple[int, int, int, int], frame_size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = frame_size
    return (
        max(0, min(width, int(bbox[0]))), max(0, min(height, int(bbox[1]))),
        max(0, min(width, int(bbox[2]))), max(0, min(height, int(bbox[3]))),
    )


def _intersects(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> bool:
    return min(left[2], right[2]) > max(left[0], right[0]) and min(left[3], right[3]) > max(left[1], right[1])


def _unique_regions(
    regions: list[tuple[tuple[int, int, int, int], str | None]],
) -> list[tuple[tuple[int, int, int, int], str | None]]:
    return list(dict.fromkeys(regions))
