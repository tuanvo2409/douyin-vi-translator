"""Small P1C boundary adapter for the existing modern Translator chain."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from canonical_profile import CanonicalLocalizationProfile


class ModernP1CAdapterError(RuntimeError):
    """Modern canonical processing cannot safely continue."""


class SourceReviewRequiredError(ModernP1CAdapterError):
    """Source evidence is insufficient; fail closed with actionable review."""


_IDENTITY_FIELDS = (
    "translator_job_id",
    "dispatch_id",
    "reup_job_id",
    "handoff_id",
    "candidate_id",
    "schedule_id",
)


def _validated_identity(identity: Mapping[str, object]) -> dict[str, str]:
    if not isinstance(identity, Mapping):
        raise ModernP1CAdapterError("modern P1C identity is incomplete")
    normalized = {field: identity.get(field) for field in _IDENTITY_FIELDS}
    if any(not isinstance(value, str) or not value.strip() for value in normalized.values()):
        raise ModernP1CAdapterError("modern P1C identity is incomplete")
    return {field: str(value) for field, value in normalized.items()}


def _owned_audio_path(output_dir: Path, audio_path: object) -> Path:
    if not isinstance(audio_path, str) or not audio_path.strip():
        raise ModernP1CAdapterError("modern TTS audio path is invalid")
    relative = Path(audio_path)
    if relative.is_absolute() or relative.drive or ".." in relative.parts:
        raise ModernP1CAdapterError("modern TTS audio path escapes attempt output")
    root = output_dir.resolve(strict=True)
    candidate = output_dir / relative
    if candidate.is_symlink() or not candidate.is_file():
        raise ModernP1CAdapterError("modern TTS audio path is not an owned regular file")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise ModernP1CAdapterError("modern TTS audio path escapes attempt output") from error
    if resolved.is_symlink() or not resolved.is_file():
        raise ModernP1CAdapterError("modern TTS audio path is not an owned regular file")
    return resolved


def _normalise_source_text(value: object) -> str:
    import string
    import unicodedata

    if not isinstance(value, str):
        return ""
    punctuation = set(string.punctuation) | set("，。！？、；：‘’“”【】（）《》…—·")
    return "".join(
        character.casefold()
        for character in unicodedata.normalize("NFKC", value)
        if not character.isspace() and character not in punctuation
    )


def _page2_placement_evidence(
    detections: Sequence[Mapping[str, object]],
    dialogue: Sequence[Mapping[str, object]],
    frame_size: tuple[int, int],
) -> list[dict[str, object]]:
    """Keep typed OCR geometry only when it matches accepted AV subtitle text/time."""
    width, height = frame_size
    if width <= 0 or height <= 0:
        raise ModernP1CAdapterError("Page 2 OCR frame dimensions are invalid")
    scale = min(720 / width, 1280 / height)
    pad_x = (720 - width * scale) / 2
    pad_y = (1280 - height * scale) / 2
    evidence: list[dict[str, object]] = []
    typed_anchors = {
        (detection.get("timestampMs"), _normalise_source_text(detection.get("text")))
        for detection in detections
        if isinstance(detection, Mapping)
        and _ocr_dialogue_category(detection.get("category")) == "DIALOGUE_SUBTITLE"
    }
    for detection in detections:
        if not isinstance(detection, Mapping):
            continue
        category = _ocr_dialogue_category(detection.get("category"))
        if category not in {"UNKNOWN", "DIALOGUE_SUBTITLE"}:
            continue
        timestamp = detection.get("timestampMs")
        text = _normalise_source_text(detection.get("text"))
        if category == "UNKNOWN" and (timestamp, text) in typed_anchors:
            continue
        bbox = detection.get("bbox")
        confidence = detection.get("confidence")
        if (
            isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0
            or not text or not isinstance(bbox, (list, tuple)) or len(bbox) != 4
            or isinstance(confidence, bool) or not isinstance(confidence, (int, float))
            or not 0.65 <= float(confidence) <= 1.0
        ):
            continue
        try:
            left, top, right, bottom = (int(value) for value in bbox)
        except (TypeError, ValueError):
            continue
        if not 0 <= left < right <= width or not 0 <= top < bottom <= height:
            continue
        for item in dialogue:
            start = item.get("startMs")
            end = item.get("endMs")
            subtitle = _normalise_source_text(item.get("subtitleZh"))
            if (
                isinstance(start, bool) or not isinstance(start, int)
                or isinstance(end, bool) or not isinstance(end, int)
                or end <= start or timestamp < start or timestamp > end
                or not subtitle or text != subtitle
            ):
                continue
            row: dict[str, object] = {
                "category": "DIALOGUE_SUBTITLE",
                "text": str(detection.get("text") or "").strip(),
                "subtitleZh": str(item.get("subtitleZh") or "").strip(),
                "startMs": start,
                "endMs": end,
                "bbox": [round(pad_x + left * scale), round(pad_y + top * scale),
                         round(pad_x + right * scale), round(pad_y + bottom * scale)],
            }
            track_id = detection.get("trackId")
            if track_id is not None and str(track_id).strip():
                row["trackId"] = str(track_id).strip()
            row["confidence"] = "HIGH"
            row["ocrConfidence"] = float(confidence)
            evidence.append(row)
            break
    return evidence


def _ocr_dialogue_category(value: object) -> str:
    category = str(value or "UNKNOWN").strip().upper()
    return "DIALOGUE_SUBTITLE" if category == "CHINESE_DIALOGUE_SUBTITLE" else category


def _page2_mask_detections(
    detections: Sequence[Mapping[str, object]],
    dialogue: Sequence[Mapping[str, object]],
    frame_size: tuple[int, int],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Promote OCR only with validated AV text/time evidence before masking."""
    from auto_roi import contains_cjk_text

    _, height = frame_size
    mask_detections: list[dict[str, object]] = []
    review: list[dict[str, object]] = []
    for detection in detections:
        if not isinstance(detection, Mapping):
            continue
        item = dict(detection)
        category = _ocr_dialogue_category(item.get("category"))
        if category in {"STORY_TEXT", "WATERMARK", "PERSISTENT_WATERMARK"}:
            item["category"] = "WATERMARK" if category == "PERSISTENT_WATERMARK" else category
            mask_detections.append(item)
            continue
        if category not in {"UNKNOWN", "DIALOGUE_SUBTITLE"}:
            mask_detections.append(item)
            continue
        timestamp = item.get("timestampMs")
        text = _normalise_source_text(item.get("text"))
        bbox = item.get("bbox")
        lower_third = isinstance(bbox, (list, tuple)) and len(bbox) == 4 and int(bbox[1]) >= int(height * 0.55)
        match = None
        if isinstance(timestamp, int) and not isinstance(timestamp, bool) and text:
            for candidate in dialogue:
                start = candidate.get("startMs")
                end = candidate.get("endMs")
                spoken = _normalise_source_text(candidate.get("spokenZh"))
                subtitle = _normalise_source_text(candidate.get("subtitleZh"))
                if (
                    isinstance(start, int) and not isinstance(start, bool)
                    and isinstance(end, int) and not isinstance(end, bool)
                    and start <= timestamp <= end
                    and spoken and subtitle
                    and text == subtitle
                    and (text == spoken or text in spoken or spoken in text)
                ):
                    match = candidate
                    break
        if match is not None:
            item["category"] = "DIALOGUE_SUBTITLE"
            item["subtitleZh"] = str(match.get("subtitleZh") or "").strip()
            mask_detections.append(item)
        elif lower_third and contains_cjk_text(str(item.get("text") or "")):
            review.append({
                "timestampMs": timestamp,
                "classification": category,
                "text": str(item.get("text") or "").strip(),
                "bbox": list(bbox) if isinstance(bbox, (list, tuple)) else None,
                "reason": "OCR_DIALOGUE_HAS_NO_VALIDATED_AV_MATCH",
            })
    return mask_detections, review


def _unresolved_visual_dialogue(
    detections: Sequence[Mapping[str, object]], asr_cues: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Return typed OCR dialogue that has no nearby canonical ASR evidence."""
    unresolved: list[dict[str, object]] = []
    from auto_roi import contains_cjk_text

    for detection in detections:
        if not isinstance(detection, Mapping) or _ocr_dialogue_category(detection.get("category")) != "DIALOGUE_SUBTITLE":
            continue
        timestamp = detection.get("timestampMs")
        text = str(detection.get("text") or "").strip()
        if not isinstance(timestamp, int) or isinstance(timestamp, bool) or not contains_cjk_text(text):
            continue
        overlaps = any(
            int(cue["start_ms"]) - 500 <= timestamp <= int(cue["end_ms"]) + 500
            for cue in asr_cues
        )
        if overlaps:
            continue
        bbox = detection.get("bbox")
        evidence: dict[str, object] = {
            "timestampMs": timestamp,
            "classification": "DIALOGUE_SUBTITLE",
            "text": text,
        }
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                evidence["bbox"] = [int(value) for value in bbox]
            except (TypeError, ValueError):
                pass
        unresolved.append(evidence)
    return unresolved


def _page1_mask_geometry(
    detections: Sequence[Mapping[str, object]], frame_size: tuple[int, int],
    *, asr_cues: Sequence[tuple[int, int]] | None = None,
) -> tuple[tuple[int, int, int, int], int] | None:
    """Bound the recurring lower dialogue-subtitle band with one static box."""
    width, height = frame_size
    if width <= 0 or height <= 0:
        raise ModernP1CAdapterError("Page 1 mask frame dimensions are invalid")
    from auto_roi import contains_cjk_text

    candidates: list[tuple[int, tuple[int, int, int, int], str, str]] = []
    for detection in detections:
        if not isinstance(detection, Mapping):
            continue
        text = detection.get("text")
        if not contains_cjk_text(text):
            continue
        timestamp = detection.get("timestampMs")
        bbox = detection.get("bbox")
        category = str(detection.get("category") or "UNKNOWN").strip().upper()
        if category == "CHINESE_DIALOGUE_SUBTITLE":
            category = "DIALOGUE_SUBTITLE"
        if category in {"STORY_TEXT", "WATERMARK", "PERSISTENT_WATERMARK"}:
            if not isinstance(timestamp, int) or isinstance(timestamp, bool) or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                raise ModernP1CAdapterError("MASK_REVIEW_REQUIRED: Page 1 OCR candidate is malformed")
            try:
                story_y1 = int(bbox[1])
            except (TypeError, ValueError):
                raise ModernP1CAdapterError("MASK_REVIEW_REQUIRED: Page 1 OCR candidate is malformed")
            if story_y1 >= int(height * 0.55):
                raise ModernP1CAdapterError("MASK_REVIEW_REQUIRED: Page 1 OCR evidence is not dialogue")
            continue
        if category not in {"DIALOGUE_SUBTITLE", "UNKNOWN"}:
            raise ModernP1CAdapterError("MASK_REVIEW_REQUIRED: Page 1 OCR evidence category is ambiguous")
        if not isinstance(timestamp, int) or isinstance(timestamp, bool) or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ModernP1CAdapterError("MASK_REVIEW_REQUIRED: Page 1 OCR candidate is malformed")
        try:
            x1, y1, x2, y2 = (int(value) for value in bbox)
        except (TypeError, ValueError):
            raise ModernP1CAdapterError("MASK_REVIEW_REQUIRED: Page 1 OCR candidate is malformed")
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        box_width, box_height = x2 - x1, y2 - y1
        if y1 < int(height * 0.35):
            if category == "DIALOGUE_SUBTITLE":
                raise ModernP1CAdapterError("MASK_REVIEW_REQUIRED: Page 1 dialogue geometry is invalid")
            continue
        if box_width <= 0 or box_height <= 0:
            raise ModernP1CAdapterError("MASK_REVIEW_REQUIRED: Page 1 OCR candidate is malformed")
        confidence = detection.get("confidence")
        if isinstance(confidence, (int, float)) and float(confidence) < 0.65:
            raise ModernP1CAdapterError("MASK_REVIEW_REQUIRED: Page 1 OCR candidate is low confidence")
        if box_width * box_height > width * height * 0.20 or box_height > height * 0.18:
            raise ModernP1CAdapterError("MASK_REVIEW_REQUIRED: Page 1 dialogue geometry is too large")
        candidates.append((timestamp, (x1, y1, x2, y2), str(text).strip(), category))

    if not candidates:
        return None
    if len({item[3] for item in candidates}) > 1:
        raise ModernP1CAdapterError("MASK_REVIEW_REQUIRED: Page 1 OCR evidence mixes classified and raw tracks")
    if asr_cues is not None:
        def overlaps_asr(timestamp: int) -> bool:
            return any(start - 500 <= timestamp <= end + 500 for start, end in asr_cues)

        if (
            len({item[0] for item in candidates}) < 2
            or len({item[2] for item in candidates}) < 2
            or any(not overlaps_asr(item[0]) for item in candidates)
        ):
            raise ModernP1CAdapterError("MASK_REVIEW_REQUIRED: Page 1 dialogue OCR is not reconciled with ASR")
    elif any(item[3] == "UNKNOWN" for item in candidates):
        raise ModernP1CAdapterError("MASK_REVIEW_REQUIRED: Page 1 raw OCR lacks ASR promotion evidence")

    # ponytail: OCR samples are capped at 12; an O(n²) band scan avoids a
    # tracker. If discovery becomes dense, replace this with track grouping.
    tolerance = max(24, round(height * 0.06))
    best: list[tuple[int, tuple[int, int, int, int], str, str]] = []
    for _, anchor, _, _ in candidates:
        center = (anchor[1] + anchor[3]) / 2
        band = [item for item in candidates if (
            abs((item[1][1] + item[1][3]) / 2 - center) <= tolerance
            and item[1][0] < anchor[2] and item[1][2] > anchor[0]
        )]
        if len({item[0] for item in band}) > len({item[0] for item in best}):
            best = band
    if not best or len({item[0] for item in best}) < 2 or len(best) != len(candidates):
        raise ModernP1CAdapterError("MASK_REVIEW_REQUIRED: Page 1 mask geometry is ambiguous")

    x1 = min(item[1][0] for item in best)
    y1 = min(item[1][1] for item in best)
    x2 = max(item[1][2] for item in best)
    y2 = max(item[1][3] for item in best)
    box_width, box_height = x2 - x1, y2 - y1
    if box_width * box_height > width * height * 0.20 or box_height > height * 0.18:
        raise ModernP1CAdapterError("Page 1 dialogue band is too large for a fixed mask")
    horizontal = max(8, int(round(box_height * 0.15)))
    vertical = max(8, int(round(box_height * 0.20)))
    result = (max(0, x1 - horizontal), max(0, y1 - vertical), min(width, x2 + horizontal), min(height, y2 + vertical))
    if (result[2] - result[0]) * (result[3] - result[1]) > width * height * 0.25:
        raise ModernP1CAdapterError("Page 1 dialogue mask exceeds the safe frame area")
    return result, min(item[0] for item in best)


def _write_page1_mask_overlay(
    overlay_path: Path, source_bbox: tuple[int, int, int, int], source_size: tuple[int, int],
    output_size: tuple[int, int], evidence_time_ms: int,
) -> dict[str, object]:
    """Create one transparent output-sized rounded mask and persist its evidence."""
    from PIL import Image, ImageDraw

    source_width, source_height = source_size
    output_width, output_height = output_size
    scale = min(output_width / source_width, output_height / source_height)
    pad_x = (output_width - source_width * scale) / 2
    pad_y = (output_height - source_height * scale) / 2
    x1, y1, x2, y2 = source_bbox
    output_bbox = tuple(round(value) for value in (
        pad_x + x1 * scale, pad_y + y1 * scale, pad_x + x2 * scale, pad_y + y2 * scale,
    ))
    if output_bbox[2] <= output_bbox[0] or output_bbox[3] <= output_bbox[1]:
        raise ModernP1CAdapterError("Page 1 mask geometry is empty after output scaling")
    image = Image.new("RGBA", (output_width, output_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    radius = max(8, min(24, round((output_bbox[3] - output_bbox[1]) * 0.18)))
    draw.rounded_rectangle(output_bbox, radius=radius, fill=(255, 255, 255, 255))
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(overlay_path, format="PNG")
    if not overlay_path.is_file() or overlay_path.stat().st_size == 0:
        raise ModernP1CAdapterError("Page 1 mask overlay was not written")
    return {
        "schemaVersion": 1,
        "sourceFrameTimeMs": evidence_time_ms,
        "sourceFrameSize": [source_width, source_height],
        "sourceBbox": list(source_bbox),
        "outputSize": [output_width, output_height],
        "outputBbox": list(output_bbox),
        "overlayPath": str(overlay_path),
    }


def _canonical_av_analysis(result: object) -> Mapping[str, object]:
    """Extract the producer's canonical AV document without reshaping it."""
    if isinstance(result, Mapping):
        inner = result.get("av_analysis")
    else:
        inner = getattr(result, "av_analysis", None)
    if not isinstance(inner, Mapping):
        raise ModernP1CAdapterError("AV analysis result has no valid av_analysis document")
    return inner


def _presentation_render_config(profile: CanonicalLocalizationProfile) -> dict[str, object]:
    """Project canonical presentation metadata into the renderer boundary."""

    presentation = dict(profile.subtitle_presentation)
    color = presentation.get("color")
    return {
        "presentation": presentation,
        "presentation_color": color if isinstance(color, str) and color.strip() else None,
    }


_LOCALIZATION_STAGE_PROFILES = {
    "loc-v1": "neutral_faithful",
    "page_goc_tro_bat_on": "vn_native_light_comedy",
    "page_giai_cuu_chuong_lon": "vn_native_light_comedy",
}


def _localization_stage_profile(runtime_config: Mapping[str, object]) -> str:
    profile_id = runtime_config.get("localizationProfile")
    if not isinstance(profile_id, str) or profile_id not in _LOCALIZATION_STAGE_PROFILES:
        raise ModernP1CAdapterError("canonical localization profile is unsupported")
    return _LOCALIZATION_STAGE_PROFILES[profile_id]


_SOURCE_EVIDENCE_SCHEMA_VERSION = 1
_SOURCE_EVIDENCE_STAGE_VERSION = "local-asr-ocr-v2"
_SOURCE_DECISION_VERSION = "ocr-repeat-v1"
_SOURCE_TRUTH_DECISIONS = {"ASR_OCR_AGREE", "OCR_SECOND_ASR_AGREE", "OCR_REPEAT"}


def _source_sha256(source: Path) -> str:
    import hashlib

    if source.is_symlink() or not source.is_file():
        raise ModernP1CAdapterError("source media must be an owned regular file")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _source_provenance(
    source_sha256: str,
    duration_ms: int,
    model: str,
    device: str,
    runtime_config: Mapping[str, object],
) -> dict[str, object]:
    from dubvi_worker import whisper_runtime_options

    roi = runtime_config.get("roi")
    ocr = runtime_config.get("ocr")
    return {
        "stageVersion": _SOURCE_EVIDENCE_STAGE_VERSION,
        "rawSha256": source_sha256,
        "durationMs": duration_ms,
        "model": model,
        "compute": {"device": device, "type": whisper_runtime_options(device)["compute_type"]},
        "roi": dict(roi) if isinstance(roi, Mapping) else None,
        "ocr": dict(ocr) if isinstance(ocr, Mapping) else {},
        "decisionVersion": _SOURCE_DECISION_VERSION,
    }


def _cached_source_evidence(
    source_sha256: str,
    output_dir: Path,
    runtime_config: Mapping[str, object],
    duration_ms: int,
    model: str,
    device: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]] | None:
    import json

    path = output_dir / "source_evidence.json"
    if path.is_symlink() or not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(document, Mapping):
        return None
    if document.get("schemaVersion") != _SOURCE_EVIDENCE_SCHEMA_VERSION or document.get("mode") != "local_asr_text_only":
        return None
    if document.get("status") != "complete":
        return None
    expected = _source_provenance(source_sha256, duration_ms, model, device, runtime_config)
    if document.get("provenance") != expected:
        return None
    cues = document.get("cues")
    records = document.get("records")
    if not isinstance(cues, list) or not isinstance(records, list) or not cues or len(cues) != len(records):
        return None

    previous_position = -1
    for cue, record in zip(cues, records):
        if not isinstance(cue, Mapping) or not isinstance(record, Mapping):
            return None
        position = cue.get("position")
        start_ms = cue.get("start_ms")
        end_ms = cue.get("end_ms")
        if (
            not isinstance(position, int) or isinstance(position, bool) or position <= previous_position
            or not isinstance(start_ms, int) or isinstance(start_ms, bool)
            or not isinstance(end_ms, int) or isinstance(end_ms, bool)
            or start_ms < 0 or end_ms <= start_ms or end_ms > duration_ms
        ):
            return None
        previous_position = position
        if record.get("positions") != [position] or record.get("cue_ids") != [cue.get("cue_id")]:
            return None
        if record.get("startMs") != start_ms or record.get("endMs") != end_ms:
            return None
        if not isinstance(cue.get("asr_text_zh"), str) or not cue["asr_text_zh"].strip():
            return None
        if not isinstance(cue.get("ocr_text_zh"), str) or not cue["ocr_text_zh"].strip():
            return None
        if cue.get("needs_review"):
            return None
        decision = record.get("decision")
        if decision not in _SOURCE_TRUTH_DECISIONS or record.get("classification") != "RESOLVED_DIALOGUE":
            return None
        evidence = record.get("evidence")
        if not isinstance(evidence, Mapping) or not str(evidence.get("ocr") or "").strip():
            return None
        expected_text = cue["asr_text_zh"] if record["decision"] == "ASR_OCR_AGREE" else cue["ocr_text_zh"]
        if (
            record.get("resolvedZh") != expected_text
            or record.get("spokenZh") != cue["asr_text_zh"]
            or evidence.get("asr") != cue["asr_text_zh"]
            or evidence.get("ocr") != cue["ocr_text_zh"]
        ):
            return None
        if decision == "OCR_REPEAT":
            readings = cue.get("ocr_readings")
            if not isinstance(readings, list) or len(readings) < 2:
                return None
            r0, r1 = readings[0], readings[1]
            if not isinstance(r0, Mapping) or not isinstance(r1, Mapping):
                return None
            t0, t1 = r0.get("timestampMs"), r1.get("timestampMs")
            if not isinstance(t0, int) or not isinstance(t1, int) or isinstance(t0, bool) or isinstance(t1, bool) or abs(t0 - t1) < 10:
                return None
            if not (start_ms <= t0 <= end_ms and start_ms <= t1 <= end_ms):
                return None
            b0, b1 = r0.get("bbox"), r1.get("bbox")
            if not isinstance(b0, (list, tuple)) or len(b0) != 4 or not isinstance(b1, (list, tuple)) or len(b1) != 4:
                return None
            cat0 = str(r0.get("category") or r0.get("role") or "").strip().upper()
            cat1 = str(r1.get("category") or r1.get("role") or "").strip().upper()
            if cat0 != "DIALOGUE_SUBTITLE" or cat1 != "DIALOGUE_SUBTITLE":
                return None
            if float(r0.get("confidence", 0)) < 0.90 or float(r1.get("confidence", 0)) < 0.90:
                return None
    if document.get("unresolvedPositions") or document.get("unresolvedVisualDialogue"):
        return None
    return [dict(record) for record in records], [dict(cue) for cue in cues]


def _prepare_page1_mask(
    source: Path,
    output_dir: Path,
    state: dict[str, object],
    runtime_paths: dict[str, Path],
    cues: Sequence[Mapping[str, object]],
    detections: Sequence[Mapping[str, object]],
) -> None:
    from dubvi_worker import ffprobe_dimensions

    try:
        frame_size = ffprobe_dimensions(source)
        active_detections = detections
        roi = state.get("sourceOcrRoi")
        if isinstance(roi, (list, tuple)) and len(roi) == 4:
            rx1, ry1, rx2, ry2 = roi
            active_detections = [
                d for d in detections
                if not (
                    isinstance(d.get("bbox"), (list, tuple)) and len(d["bbox"]) == 4
                    and (d["bbox"][3] < ry1 - 25 or d["bbox"][1] > ry2 + 25)
                )
            ]
        geometry = _page1_mask_geometry(
            active_detections, frame_size,
            asr_cues=[(int(cue["start_ms"]), int(cue["end_ms"])) for cue in cues],
        )
        if geometry is None:
            return
        bbox, evidence_time_ms = geometry
        import json
        evidence = _write_page1_mask_overlay(
            output_dir / "page1_mask_overlay.png", bbox, frame_size, (720, 1280), evidence_time_ms,
        )
        (output_dir / "page1_mask.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        runtime_paths["page1_mask_overlay"] = output_dir / "page1_mask_overlay.png"
        state["page1MaskBBox"] = tuple(evidence["outputBbox"])
    except ModernP1CAdapterError:
        raise
    except Exception as exc:
        raise ModernP1CAdapterError("Page 1 OCR/mask evidence could not be prepared") from exc


def _local_asr_source(
    source: Path,
    output_dir: Path,
    runtime_config: Mapping[str, object],
    state: dict[str, object],
    runtime_paths: dict[str, Path],
) -> Sequence[Mapping[str, object]]:
    """Build canonical source rows from local ASR without entering the AV boundary."""

    import os

    from cue_aligner import _atomic_json
    from dubvi_worker import ensure_whisper_model, extract_audio, ffprobe_duration, transcribe
    from source_reconciler import assign_canonical_utterance_ids

    model = str(runtime_config.get("whisperModel") or os.environ.get("DUBVI_WHISPER_MODEL") or "small")
    device = str(os.environ.get("DUBVI_DEVICE") or "cpu")
    duration = round(ffprobe_duration(source) * 1000)
    source_sha256 = _source_sha256(source) if source.is_file() and not source.is_symlink() else None
    runtime_paths.update(source=source, output_dir=output_dir, render_source=source)
    cached = (
        _cached_source_evidence(source_sha256, output_dir, runtime_config, duration, model, device)
        if source_sha256 is not None else None
    )
    if cached is not None:
        canonical, cached_cues = cached
        state["sourceDurationMs"] = duration
        state["sourceCues"] = cached_cues
        state["sourceEvidenceMode"] = "local_asr_text_only"
        if runtime_config.get("page1MaskMode") != "none":
            mask_png = output_dir / "page1_mask_overlay.png"
            mask_json = output_dir / "page1_mask.json"
            if mask_png.is_file() and mask_json.is_file() and not mask_png.is_symlink() and not mask_json.is_symlink():
                import json
                try:
                    evidence = json.loads(mask_json.read_text(encoding="utf-8"))
                    runtime_paths["page1_mask_overlay"] = mask_png
                    state["page1MaskBBox"] = tuple(evidence["outputBbox"])
                except Exception:
                    mask_png = None
            if runtime_paths.get("page1_mask_overlay") is None:
                from auto_roi import sparse_whole_frame_discovery

                detections = sparse_whole_frame_discovery(
                    source, int((runtime_config.get("ocr") or {}).get("discoveryFrames", 8))
                )
                state["localOcrDetections"] = len(detections)
                _prepare_page1_mask(source, output_dir, state, runtime_paths, cached_cues, detections)
        return canonical

    ensure_whisper_model(model, device)
    state["sourceDurationMs"] = duration
    audio = output_dir / "source.wav"
    extract_audio(source, audio)
    rows = list(transcribe(audio, model, device))
    if not rows:
        raise ModernP1CAdapterError("local ASR produced no source utterances")
    # ponytail: Whisper's log-probability-derived score is not a calibrated
    # probability. Keep a separate, conservative ASR floor; OCR has its own scale.
    min_confidence = int(runtime_config.get("asrMinConfidence", 50))
    high_confidence_threshold = int(runtime_config.get("asrHighConfidence", 65))
    if not 0 <= min_confidence <= high_confidence_threshold <= 100:
        raise ModernP1CAdapterError("Page 1 ASR confidence thresholds are invalid")
    cues: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        position = row.get("position", index)
        start_ms = row.get("startMs")
        end_ms = row.get("endMs")
        text = str(row.get("asrTextZh") or "").strip()
        confidence = int(row.get("confidence", 0))
        if not isinstance(position, int) or isinstance(position, bool):
            raise ModernP1CAdapterError("local ASR source position is invalid")
        if not isinstance(start_ms, int) or isinstance(start_ms, bool) or not isinstance(end_ms, int) or isinstance(end_ms, bool) or end_ms <= start_ms:
            raise ModernP1CAdapterError("local ASR source timing is invalid")
        if not text:
            continue
        cue_id = f"asr-{position}"
        resolved = confidence >= min_confidence
        confidence_label = "HIGH" if confidence >= high_confidence_threshold else "MEDIUM" if resolved else "LOW"
        flags = [] if resolved else ["SOURCE_UNCERTAIN"]
        cues.append({
            "cue_id": cue_id,
            "position": position,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "asr_text_zh": text,
            "ocr_text_zh": "",
            "ocr_readings": [],
            "decision": "LOCAL_ASR",
            "needs_review": not resolved,
        })
        classification = "RESOLVED_DIALOGUE" if resolved else "UNRESOLVED_DIALOGUE"
        records.append({
            "startMs": start_ms,
            "endMs": end_ms,
            "spokenZh": text,
            "subtitleZh": "",
            "resolvedZh": text,
            "confidence": confidence_label,
            "source_confidence": confidence_label.lower(),
            "positions": [position],
            "cue_ids": [cue_id],
            "evidence": {"asr": text, "ocr": "", "avSpoken": "", "avSubtitle": ""},
            "decision": "LOCAL_ASR",
            "flags": flags,
            "source_flags": flags,
            "kind": "dialogue",
            "type": "DIALOGUE" if resolved else classification,
            "classification": classification,
        })
    if not records:
        raise ModernP1CAdapterError("local ASR produced no usable source utterances")
    canonical = assign_canonical_utterance_ids(records)
    state["sourceCues"] = cues
    state["sourceEvidenceMode"] = "local_asr_text_only"

    def write_source_evidence(
        unresolved: Sequence[Mapping[str, object]],
        *,
        status: str = "complete",
        failure: str | None = None,
    ) -> None:
        document: dict[str, object] = {
            "schemaVersion": _SOURCE_EVIDENCE_SCHEMA_VERSION,
            "mode": "local_asr_text_only",
            "status": status,
            "durationMs": duration,
            "cues": cues,
            "records": canonical,
            "unresolvedPositions": [int(cue["position"]) for cue in cues if cue["needs_review"]],
            "unresolvedVisualDialogue": [dict(item) for item in unresolved],
        }
        if source_sha256 is not None:
            document["provenance"] = _source_provenance(source_sha256, duration, model, device, runtime_config)
        if failure is not None:
            document["failure"] = failure
        _atomic_json(output_dir / "source_evidence.json", document)

    from auto_roi import sparse_whole_frame_discovery
    from dubvi_worker import ffprobe_dimensions

    try:
        detections = sparse_whole_frame_discovery(source, int((runtime_config.get("ocr") or {}).get("discoveryFrames", 8)))
    except Exception as exc:
        write_source_evidence([], status="failed", failure="SOURCE_OCR_DISCOVERY_FAILED")
        raise ModernP1CAdapterError("Page 1 OCR discovery failed") from exc
    state["localOcrDetections"] = len(detections)
    unresolved_visual_dialogue = _unresolved_visual_dialogue(detections, cues)
    if unresolved_visual_dialogue:
        write_source_evidence(unresolved_visual_dialogue, status="review_required", failure="SOURCE_EVIDENCE_UNRESOLVED")
        raise SourceReviewRequiredError(
            "SOURCE_EVIDENCE_UNRESOLVED: typed OCR dialogue has no ASR overlap; "
            "MASK_REVIEW_REQUIRED"
        )
    verify_ocr = runtime_config.get("sourceOcrVerify")
    if verify_ocr is True or (verify_ocr is None and os.environ.get("DUBVI_SOURCE_OCR_VERIFY") == "1"):
        from source_ocr_verifier import decide_source, infer_stable_subtitle_roi, scan_cue_ocr, scan_second_asr
        try:
            roi = infer_stable_subtitle_roi(detections, ffprobe_dimensions(source))
            if roi is not None:
                readings = scan_cue_ocr(source, cues, roi, detections=detections)
                exe = runtime_config.get("sensevoiceExe") or os.environ.get("DUBVI_SENSEVOICE_EXE")
                model_path = runtime_config.get("sensevoiceModel") or os.environ.get("DUBVI_SENSEVOICE_MODEL")
                if bool(exe) != bool(model_path):
                    raise ValueError("SenseVoice executable and model must be configured together")
                second_asr = scan_second_asr(audio, cues, readings, Path(str(exe)), Path(str(model_path))) if exe else {}
                for cue, record in zip(cues, canonical):
                    position = int(cue["position"])
                    cue_readings = readings.get(position, [])
                    cue_timing = (int(cue["start_ms"]), int(cue["end_ms"]))
                    decision = decide_source(
                        str(cue["asr_text_zh"]),
                        cue_readings,
                        second_asr=second_asr.get(position),
                        cue_timing=cue_timing,
                        roi=roi,
                    )
                    cue["ocr_readings"] = cue_readings
                    cue["ocr_text_zh"] = decision["ocrZh"]
                    cue["second_asr_text_zh"] = second_asr.get(position, "")
                    cue["decision"] = decision["decision"]
                    cue["needs_review"] = bool(decision["needsReview"])
                    record["evidence"]["ocr"] = decision["ocrZh"]
                    record["evidence"]["secondAsr"] = second_asr.get(position, "")
                    record["evidence"]["readings"] = cue_readings
                    record["resolvedZh"] = decision["resolvedZh"]
                    record["decision"] = decision["decision"]
                    if decision["decision"] in _SOURCE_TRUTH_DECISIONS:
                        record["subtitleZh"] = decision["ocrZh"]
                        record["classification"] = "RESOLVED_DIALOGUE"
                        record["type"] = "DIALOGUE"
                        record["confidence"] = "HIGH"
                        record["source_confidence"] = "high"
                        record["flags"] = [f for f in record["flags"] if f != "SOURCE_UNCERTAIN"]
                        record["source_flags"] = record["flags"]
                    if cue["needs_review"]:
                        record["confidence"] = "LOW"
                        record["source_confidence"] = "low"
                        record["classification"] = "UNRESOLVED_DIALOGUE"
                        record["type"] = "UNRESOLVED_DIALOGUE"
                        record["flags"] = list(dict.fromkeys([*record["flags"], "SOURCE_UNCERTAIN"]))
                        record["source_flags"] = record["flags"]
                state["sourceOcrRoi"] = roi
            else:
                for cue, record in zip(cues, canonical):
                    cue["needs_review"] = True
                    record["decision"] = "NO_STABLE_SUBTITLE_ROI"
                    record["classification"] = "UNRESOLVED_DIALOGUE"
                    record["type"] = "UNRESOLVED_DIALOGUE"
                    record["source_confidence"] = "low"
                    record["confidence"] = "LOW"
                    record["flags"] = list(dict.fromkeys([*record["flags"], "SOURCE_UNCERTAIN"]))
                    record["source_flags"] = record["flags"]
        except Exception as exc:
            write_source_evidence(unresolved_visual_dialogue, status="failed", failure="SOURCE_OCR_SCAN_FAILED")
            raise SourceReviewRequiredError("SOURCE_OCR_SCAN_FAILED") from exc
        unresolved_positions = [int(cue["position"]) for cue in cues if cue["needs_review"]]
        if unresolved_positions:
            write_source_evidence(unresolved_visual_dialogue, status="review_required", failure="SOURCE_EVIDENCE_UNRESOLVED")
            raise SourceReviewRequiredError(
                "SOURCE_EVIDENCE_UNRESOLVED: unverified Page 1 source positions "
                + ",".join(str(position) for position in unresolved_positions[:32])
            )
    write_source_evidence(unresolved_visual_dialogue)
    if runtime_config.get("page1MaskMode") == "none":
        return canonical
    _prepare_page1_mask(source, output_dir, state, runtime_paths, cues, detections)
    return canonical


def build_modern_stage_bundle(*, runtime_config: Mapping[str, object]) -> dict[str, Callable[..., object]]:
    """Construct the child-owned media graph; providers are resolved at runtime."""
    if not isinstance(runtime_config, Mapping):
        raise ModernP1CAdapterError("canonical runtime configuration is required")
    state: dict[str, object] = {}
    runtime_paths: dict[str, Path] = {}

    def source_stage(source: Path, output_dir: Path, **_: object) -> Sequence[Mapping[str, object]]:
        from dubvi_worker import ffprobe_dimensions, ffprobe_duration
        from localized_masking import apply_localized_masks, build_mask_plan
        from source_reconciler import reconcile_source
        from utterance_builder import build_utterances
        if runtime_config.get("sourceEvidenceMode") == "local_asr_text_only":
            return _local_asr_source(source, output_dir, runtime_config, state, runtime_paths)
        from av_analyzer import GeminiAVAnalyzer, analyze_media_windows_with_subdivision
        from auto_roi import sparse_whole_frame_discovery
        duration = round(ffprobe_duration(source) * 1000)
        state["sourceDurationMs"] = duration
        runtime_paths.update(source=source, output_dir=output_dir)
        analysis = analyze_media_windows_with_subdivision(
            GeminiAVAnalyzer(model=str(runtime_config.get("avModel", "gemini-2.5-flash"))),
            source, duration, diagnostics_path=output_dir / "av_subdivision_diagnostics.json",
        )
        canonical_analysis = _canonical_av_analysis(analysis)
        reconstructed = reconcile_source([], canonical_analysis)
        dialogue = build_utterances(reconstructed["dialogue"], [])
        try:
            detections = sparse_whole_frame_discovery(source, int(runtime_config.get("ocrDiscoveryFrames", 8)))
            frame_dimensions = ffprobe_dimensions(source)
            mask_detections, review = _page2_mask_detections(detections, dialogue, frame_dimensions)
            if review:
                state["page2SourceReview"] = review
                raise SourceReviewRequiredError(
                    "SOURCE_REVIEW_REQUIRED: Page 2 OCR dialogue lacks validated AV evidence"
                )
            mask_plan = build_mask_plan(mask_detections, frame_dimensions)
        except Exception as exc:
            if isinstance(exc, SourceReviewRequiredError):
                raise
            raise ModernP1CAdapterError("source OCR/mask plan preparation failed") from exc
        runtime_paths["render_source"] = source
        if mask_plan:
            masked_source = output_dir / "masked_source.mkv"
            try:
                apply_localized_masks(source, masked_source, mask_plan)
            except Exception as exc:
                raise ModernP1CAdapterError("localized source mask application failed") from exc
            if not masked_source.is_file() or masked_source.stat().st_size == 0:
                raise ModernP1CAdapterError("localized source mask produced no regular output file")
            runtime_paths["render_source"] = masked_source
        state["placementEvidence"] = _page2_placement_evidence(
            detections,
            dialogue,
            frame_dimensions,
        )
        return dialogue

    def localization_stage(records: Sequence[Mapping[str, object]], **kwargs: object) -> Sequence[Mapping[str, object]]:
        import os
        from cue_aligner import _atomic_json, build_alignment, write_alignment_artifacts
        from scene_memory import build_scene_memory, build_terminology_evidence
        from vn_localizer import localize
        provider = str(runtime_config.get("localizationProvider") or os.getenv("DUBVI_LOCALIZATION_PROVIDER", "gemini")).strip().lower()
        profile_name = _localization_stage_profile(runtime_config)
        output_dir = Path(kwargs.get("output_dir") or runtime_paths.get("output_dir") or ".")
        provider_artifact = output_dir / "translation_provider.json"
        alignment_artifact = output_dir / "cue_alignment_v3.json"

        # Check stage-specific localization cache
        cached_alignment = None
        if provider_artifact.is_file() and alignment_artifact.is_file() and not provider_artifact.is_symlink() and not alignment_artifact.is_symlink():
            try:
                import json
                prov_doc = json.loads(provider_artifact.read_text(encoding="utf-8"))
                align_doc = json.loads(alignment_artifact.read_text(encoding="utf-8"))
                expected_provider_path = {"gemini": "gemini", "groq_text": "groq", "deepseek": "deepseek"}.get(provider)
                if prov_doc.get("providerPath") == expected_provider_path:
                    dialogue_rows = align_doc.get("dialogue")
                    if isinstance(dialogue_rows, list) and len(dialogue_rows) == len(records):
                        match_all = True
                        for row, rec in zip(dialogue_rows, records):
                            if (
                                str(row.get("utteranceId")) != str(rec.get("utteranceId"))
                                or str(row.get("resolvedZh") or "") != str(rec.get("resolvedZh") or "")
                                or not str(row.get("nativeVi") or "").strip()
                                or row.get("reviewRequired")
                            ):
                                match_all = False
                                break
                        if match_all:
                            cached_alignment = align_doc
            except Exception:
                cached_alignment = None

        if cached_alignment is not None:
            state["translationProviderPath"] = provider
            state["alignment"] = cached_alignment
            return [
                {
                    **item,
                    "semanticStartMs": int(item["startMs"]),
                    "semanticEndMs": int(item["endMs"]),
                    "approvedNativeVi": item["nativeVi"],
                    "deliveryNativeVi": item["nativeVi"],
                    "sourcePositions": list(item.get("sourcePositions") or []),
                }
                for item in cached_alignment["dialogue"]
            ]

        memory = build_scene_memory({"terminologyEvidence": build_terminology_evidence(records), "summary": "", "tone": "", "register": ""})
        if provider == "deepseek":
            from cp2_deepseek_adapter import CP2DeepSeekAdapter
            adapter = CP2DeepSeekAdapter()
        elif provider == "gemini":
            from cp2_gemini_adapter import live_cp2_gemini_adapter
            from text_translation_fallback import GeminiPrimaryWithGroqTextFallback, GroqTextTranslationAdapter
            gemini = live_cp2_gemini_adapter(model=str(runtime_config.get("localizationModel", "gemini-2.5-flash")))
            adapter = GeminiPrimaryWithGroqTextFallback(gemini, GroqTextTranslationAdapter())
        else:
            raise ModernP1CAdapterError("localization provider is unsupported")

        text_only_records = [
            {
                "utteranceId": str(item["utteranceId"]),
                "resolvedZh": str(item.get("resolvedZh") or ""),
                "source_confidence": str(item.get("source_confidence") or ""),
                "kind": str(item.get("kind") or "dialogue"),
                "classification": str(item.get("classification") or item.get("type") or ""),
                "type": str(item.get("type") or ""),
                "flags": list(item.get("flags") or []),
                "source_flags": list(item.get("source_flags") or item.get("sourceFlags") or item.get("flags") or []),
                "decision": item.get("decision"),
                "evidence": dict(item.get("evidence") or {}),
            }
            for item in records
        ]
        provider_artifact.unlink(missing_ok=True)
        localized = localize(text_only_records, scene_memory=memory, request=adapter, profile=profile_name)
        state["translationProviderPath"] = adapter.last_provider
        if adapter.last_provider not in {None, "gemini", "groq_text", "deepseek"}:
            raise ModernP1CAdapterError("translation provider path is invalid")
        provider_path = {"gemini": "gemini", "groq_text": "groq", "deepseek": "deepseek"}.get(adapter.last_provider)
        _atomic_json(
            provider_artifact,
            {
                "schemaVersion": 1,
                "providerPath": provider_path,
                "fallbackClassification": getattr(adapter, "gemini_failure", None),
            },
        )
        if not localized.get("accepted") and localized.get("review_required"):
            raise SourceReviewRequiredError(
                "SOURCE_REVIEW_REQUIRED: source uncertainty left no translatable records"
            )
        alignment = build_alignment(
            records,
            accepted=localized["accepted"],
            review_required=localized["review_required"],
            source_cues=state.get("sourceCues"),
        )
        source_by_id = {str(item["utteranceId"]): item for item in records}
        for row in [*alignment["dialogue"], *alignment["excluded"]]:
            source = source_by_id.get(str(row["utteranceId"]))
            if source is None:
                continue
            for field in ("decision", "evidence"):
                if field in source:
                    row[field] = source[field]
            source_flags = source.get("source_flags") or source.get("sourceFlags") or source.get("flags")
            if isinstance(source_flags, (list, tuple)):
                row["sourceFlags"] = list(source_flags)
                row["reviewFlags"] = list(dict.fromkeys([*row.get("reviewFlags", []), *map(str, source_flags)]))
        write_alignment_artifacts(alignment, output_dir)
        state["alignment"] = alignment
        return [
            {
                **item,
                "semanticStartMs": int(item["startMs"]),
                "semanticEndMs": int(item["endMs"]),
                "approvedNativeVi": item["nativeVi"],
                "deliveryNativeVi": item["nativeVi"],
                "sourcePositions": list(item.get("sourcePositions") or []),
            }
            for item in alignment["dialogue"]
        ]

    def tts_stage(records: Sequence[Mapping[str, object]], output_dir: Path, **_: object) -> Mapping[str, object]:
        from dubvi_worker import Settings, fit_voice, ffprobe_duration, synthesize
        from tts_timing import run_tts_timing
        settings = Settings.from_env()
        provider = str(runtime_config.get("ttsProvider") or settings.tts_provider).strip().lower()
        settings.tts_provider = provider
        voice_name = str((runtime_config.get("voice") or {}).get("name") or "").strip()
        if provider == "capcut":
            from dubvi_worker import validate_capcut_voice
            validate_capcut_voice(voice_name or settings.capcut_voice)

        if provider == "piper":
            piper_model = Path(settings.piper_model).expanduser().resolve()
            manifest_voice = f"piper:{piper_model.name}"
            resource_identity = {"provider": "piper", "model": str(piper_model)}
        elif provider == "capcut":
            manifest_voice = voice_name or settings.capcut_voice
            resource_identity = {"provider": "capcut", "voice": manifest_voice}
        else:
            manifest_voice = voice_name or "vi-VN-HoaiMyNeural"
            resource_identity = {"provider": "edge", "voice": manifest_voice}

        def configured_synthesizer(text: str, voice: str, target: Path) -> None:
            synthesize(text, voice, target, settings)

        alignment = {
            "dialogue": [dict(item) for item in records],
            "excluded": list((state.get("alignment") or {}).get("excluded", [])) if isinstance(state.get("alignment"), Mapping) else [],
        }
        return run_tts_timing(
            alignment,
            output_dir,
            voice=manifest_voice,
            tts_provider=provider,
            resource_identity=resource_identity,
            synthesizer=configured_synthesizer,
            measure_duration=lambda p: round(ffprobe_duration(p) * 1000),
            fit_audio=fit_voice,
        )

    def delivery_stage(records: Sequence[Mapping[str, object]], *, voice_events: Sequence[Mapping[str, object]], **_: object) -> Sequence[Mapping[str, object]]:
        from delivery_planner import DeliveryPolicy, plan_delivery
        measured = {str(item["utteranceId"]): item for item in voice_events}
        enriched: list[dict[str, object]] = []
        for item in records:
            identifier = str(item["utteranceId"])
            event = measured.get(identifier)
            if not isinstance(event, Mapping):
                raise ModernP1CAdapterError(f"modern TTS event is missing for {identifier}")
            if not _is_renderable_tts_event(event):
                raise ModernP1CAdapterError(f"modern TTS event is not renderable for {identifier}")
            enriched.append({
                **item,
                "ttsMeasuredMs": int(event["fittedDurationMs"]),
                "audioPath": str(event["audioPath"]),
                "nextSpeechStartMs": int(event["effectiveEndMs"]),
            })
        return list(plan_delivery(enriched, policy=DeliveryPolicy(max_tempo=1.0))["plans"])

    def timeline_stage(voices, cues, plans, **_: object):
        from adaptive_renderer import build_adaptive_timeline
        from timeline_mapper import timeline_map_from_document
        timeline = build_adaptive_timeline(voices, cues, plans, source_duration_ms=int(state["sourceDurationMs"]))
        state["timeline"] = timeline
        state["sourceToOutput"] = timeline_map_from_document(timeline)
        return timeline

    def subtitle_stage(cues, *, profile: CanonicalLocalizationProfile, **_: object):
        from adaptive_renderer import build_adaptive_ffmpeg_command, render_adaptive
        from subtitle_layout import write_ass_subtitles
        timeline = state.get("timeline")
        if not isinstance(timeline, Mapping):
            raise ModernP1CAdapterError("adaptive timeline is missing at render seam")
        ass_path = runtime_paths["output_dir"] / "subtitles.ass"
        presentation_config = _presentation_render_config(profile)
        result = write_ass_subtitles(
            cues,
            ass_path,
            width=720,
            height=1280,
            presentation_color=presentation_config["presentation_color"],
            placement_evidence=state.get("placementEvidence"),
            source_to_output=state.get("sourceToOutput"),
            page1_mask_bbox=(
                tuple(state["page1MaskBBox"])
                if runtime_paths.get("page1_mask_overlay") is not None
                and isinstance(state.get("page1MaskBBox"), tuple)
                else None
            ),
        )
        result["presentation"] = presentation_config["presentation"]
        voice_inputs = [
            (_owned_audio_path(runtime_paths["output_dir"], event.get("audioPath")), int(event["scheduledStartMs"]))
            for event in timeline["voiceEvents"]
        ]
        output = runtime_paths["output_dir"] / "translated.mp4"
        command = build_adaptive_ffmpeg_command(
            runtime_paths["render_source"], output, ass_path, voice_inputs,
            timeline=timeline, width=720, height=1280,
            mask_overlay=runtime_paths.get("page1_mask_overlay"),
        )
        render_adaptive(command)
        result["outputPath"] = str(output)
        return result

    return {"source": source_stage, "localization": localization_stage, "tts": tts_stage, "delivery": delivery_stage, "timeline": timeline_stage, "subtitle": subtitle_stage}


def run_modern_p1c(
    source: Path,
    output_dir: Path,
    identity: Mapping[str, object],
    profile: CanonicalLocalizationProfile,
    *,
    source_stage: Callable[..., Sequence[Mapping[str, object]]],
    localization_stage: Callable[..., Sequence[Mapping[str, object]]],
    tts_stage: Callable[..., Mapping[str, object]],
    delivery_stage: Callable[..., Sequence[Mapping[str, object]]],
    timeline_stage: Callable[..., Mapping[str, object]],
    subtitle_stage: Callable[..., Mapping[str, object]],
) -> dict[str, object]:
    """Compose modern stages while carrying identity and canonical utterance IDs."""

    canonical_identity = _validated_identity(identity)
    records = list(source_stage(source, output_dir, profile=profile, identity=dict(canonical_identity)))
    localized = list(localization_stage(records, profile=profile, identity=dict(canonical_identity)))
    tts = dict(tts_stage(localized, output_dir, profile=profile, identity=dict(canonical_identity)))
    voices = tts.get("voiceEvents")
    if not isinstance(voices, list):
        raise ModernP1CAdapterError("modern TTS stage returned no voiceEvents")
    _validate_tts_events(localized, voices)
    plans = list(delivery_stage(localized, voice_events=voices, profile=profile, identity=dict(canonical_identity)))
    cues = _subtitle_cues(voices)
    timeline = dict(timeline_stage(voices, cues, plans, profile=profile, identity=dict(canonical_identity)))
    subtitles = dict(subtitle_stage(timeline.get("subtitleCues", cues), profile=profile, identity=dict(canonical_identity)))
    return {
        "identity": dict(canonical_identity),
        "profile": profile.profile_id,
        "records": localized,
        "tts": tts,
        "delivery": plans,
        "timeline": timeline,
        "subtitles": subtitles,
        "outputPath": subtitles.get("outputPath"),
    }


def _is_renderable_tts_event(event: Mapping[str, object]) -> bool:
    """Return whether an event has success-only audio metadata."""

    duration = event.get("fittedDurationMs")
    audio_path = event.get("audioPath")
    return (
        event.get("reviewReason") != "TTS_FAILED"
        and isinstance(audio_path, str)
        and bool(audio_path.strip())
        and isinstance(duration, int)
        and not isinstance(duration, bool)
        and duration > 0
    )


def _validate_tts_events(
    records: Sequence[Mapping[str, object]], voice_events: Sequence[object]
) -> None:
    """Reject failed or malformed TTS events before delivery/timeline planning."""

    by_id: dict[str, Mapping[str, object]] = {}
    duplicate_ids: set[str] = set()
    for event in voice_events:
        if not isinstance(event, Mapping):
            raise ModernP1CAdapterError("modern TTS event is not an object")
        identifier = str(event.get("utteranceId") or "")
        if not identifier or identifier in by_id:
            duplicate_ids.add(identifier or "<missing>")
        by_id[identifier] = event
    if duplicate_ids:
        raise ModernP1CAdapterError(
            "modern TTS event identities are duplicated or missing: "
            + ", ".join(sorted(duplicate_ids))
        )

    invalid: list[str] = []
    for record in records:
        identifier = str(record.get("utteranceId") or "")
        event = by_id.get(identifier)
        if event is None:
            invalid.append(f"MISSING_EVENT for {identifier or '<missing>'}")
        elif not _is_renderable_tts_event(event):
            reason = str(event.get("reviewReason") or "INVALID_AUDIO_METADATA")
            invalid.append(f"{reason} for {identifier or '<missing>'}")
    if invalid:
        raise ModernP1CAdapterError(
            "modern TTS stage produced non-renderable events: " + "; ".join(invalid)
        )


def _subtitle_cues(voice_events: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    cues: list[dict[str, object]] = []
    for event in voice_events:
        identifier = event.get("utteranceId")
        if not identifier:
            continue
        cues.append({
            "voiceEventId": str(identifier),
            "utteranceId": str(identifier),
            "startMs": int(event.get("baseStartMs", event.get("effectiveStartMs", 0))),
            "endMs": int(event.get("baseEndMs", event.get("effectiveEndMs", 0))),
            "translatedTextVi": event.get("deliveryNativeVi", event.get("nativeViApproved")),
            "sourcePositions": list(event.get("sourcePositions") or []),
        })
    return cues
