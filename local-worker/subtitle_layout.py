"""Measured, bounded viewer subtitle layout for ASS rendering."""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


class SubtitleLayoutError(ValueError):
    """A subtitle cannot be represented without violating the display contract."""


@dataclass(frozen=True)
class SubtitleLayoutPolicy:
    frame_width: int = 720
    frame_height: int = 960
    font_name: str = "Arial"
    font_path: str | None = None
    font_size: int = 38
    min_font_size: int = 28
    font_step: int = 2
    max_box_width_ratio: float = 0.88
    horizontal_safe_margin_ratio: float = 0.06
    padding_x_px: int = 18
    padding_top_px: int = 10
    padding_bottom_px: int = 10
    outline_px: int = 4
    shadow_px: int = 2
    line_spacing_px: int = 4
    bottom_safe_margin_px: int = 48
    minimum_display_duration_ms: int = 350
    source_anchor_min_ratio: float = 0.55
    source_anchor_max_ratio: float = 0.90
    source_anchor_max_spread_ratio: float = 0.25

    def __post_init__(self) -> None:
        if self.frame_width <= 0 or self.frame_height <= 0:
            raise SubtitleLayoutError("frame dimensions must be positive")
        if self.font_size <= 0 or self.min_font_size <= 0 or self.min_font_size > self.font_size:
            raise SubtitleLayoutError("font size range is invalid")
        if self.font_step <= 0:
            raise SubtitleLayoutError("font step must be positive")
        if not 0 < self.max_box_width_ratio <= 1:
            raise SubtitleLayoutError("maximum box width ratio is invalid")
        if not 0 <= self.horizontal_safe_margin_ratio < 0.5:
            raise SubtitleLayoutError("horizontal safe margin ratio is invalid")
        if self.max_box_width_ratio > 1 - 2 * self.horizontal_safe_margin_ratio:
            raise SubtitleLayoutError("box width exceeds horizontal safe area")
        if not 0 <= self.source_anchor_min_ratio <= self.source_anchor_max_ratio <= 1:
            raise SubtitleLayoutError("source anchor vertical range is invalid")
        if self.source_anchor_max_spread_ratio < 0:
            raise SubtitleLayoutError("source anchor spread cannot be negative")
        if min(
            self.padding_x_px,
            self.padding_top_px,
            self.padding_bottom_px,
            self.outline_px,
            self.shadow_px,
            self.line_spacing_px,
            self.bottom_safe_margin_px,
            self.minimum_display_duration_ms,
        ) < 0:
            raise SubtitleLayoutError("layout measurements cannot be negative")


def layout_subtitles(
    cues: Sequence[Mapping[str, object]],
    *,
    policy: SubtitleLayoutPolicy | None = None,
    placement_evidence: Sequence[Mapping[str, object]] | None = None,
    source_to_output: object | None = None,
) -> dict[str, object]:
    """Lay out display-only cues without mutating timing or source data."""

    policy = policy or SubtitleLayoutPolicy()
    display_cues: list[dict[str, object]] = []
    split_ids: set[str] = set()
    placements: dict[str, dict[str, object]] = {}
    for raw in cues:
        if not isinstance(raw, Mapping):
            raise SubtitleLayoutError("subtitle cue must be an object")
        identifier = _identifier(raw)
        text = _text(raw)
        start = _integer(raw, "scheduledStartMs")
        end = _integer(raw, "scheduledEndMs")
        if end <= start:
            raise SubtitleLayoutError(f"subtitle interval is invalid for {identifier}")
        chunks = _display_chunks(text, policy)
        if len(chunks) > 1:
            split_ids.add(identifier)
        layouts = [_best_layout(chunk, policy) for chunk in chunks]
        intervals = _weighted_intervals(start, end, layouts, policy)
        placement = placements.get(identifier)
        if placement is None:
            placement = _source_placement(
                identifier,
                start,
                end,
                placement_evidence,
                source_to_output,
                policy,
            )
            placements[identifier] = placement
        for index, (layout, (output_start, output_end)) in enumerate(zip(layouts, intervals), start=1):
            positioned = _position_layout(layout, placement, policy)
            display_cues.append({
                "utteranceId": identifier,
                "voiceEventId": str(raw.get("voiceEventId") or identifier),
                "displayCueIndex": index,
                "text": layout["text"],
                "lines": layout["lines"],
                "lineCount": layout["lineCount"],
                "fontSize": layout["fontSize"],
                "lineWidthsPx": layout["lineWidthsPx"],
                "lineHeightPx": layout["lineHeightPx"],
                "boxWidthPx": layout["boxWidthPx"],
                "boxHeightPx": layout["boxHeightPx"],
                "boxLeftPx": positioned["boxLeftPx"],
                "boxTopPx": positioned["boxTopPx"],
                "boxRightPx": positioned["boxRightPx"],
                "boxBottomPx": positioned["boxBottomPx"],
                "anchorSource": placement["anchorSource"],
                "sourceDialogueTrackId": placement["sourceDialogueTrackId"],
                "sourceAnchorXPx": placement["sourceAnchorXPx"],
                "sourceAnchorYPx": placement["sourceAnchorYPx"],
                "finalAnchorXPx": positioned["finalAnchorXPx"],
                "finalAnchorYPx": positioned["finalAnchorYPx"],
                "fallbackUsed": placement["fallbackUsed"],
                "fallbackReason": placement["fallbackReason"],
                "outputStartMs": output_start,
                "outputEndMs": output_end,
                "splitFromLongCue": len(chunks) > 1,
            })
    return {
        "schemaVersion": 1,
        "artifact": "subtitle_layout_v6_source_aware"
        if placement_evidence is not None or source_to_output is not None
        else "subtitle_layout_v5",
        "policy": _policy_document(policy),
        "displayCues": display_cues,
        "maximumLineCount": max((int(cue["lineCount"]) for cue in display_cues), default=0),
        "minimumFontSizeUsed": min((int(cue["fontSize"]) for cue in display_cues), default=0),
        "splitUtteranceCount": len(split_ids),
        "maximumBoxWidthRatio": max(
            (int(cue["boxWidthPx"]) / policy.frame_width for cue in display_cues),
            default=0.0,
        ),
        "sourceAnchoredCueCount": sum(
            1 for cue in display_cues if cue["anchorSource"] == "DIALOGUE_SUBTITLE"
        ),
        "fallbackCueCount": sum(1 for cue in display_cues if cue["fallbackUsed"]),
        "minimumBoxTopPx": min((int(cue["boxTopPx"]) for cue in display_cues), default=0),
        "maximumBoxBottomPx": max((int(cue["boxBottomPx"]) for cue in display_cues), default=0),
        "verticalClampCount": sum(
            1 for cue in display_cues
            if cue["anchorSource"] == "DIALOGUE_SUBTITLE"
            and cue["sourceAnchorYPx"] != cue["finalAnchorYPx"]
        ),
    }


def render_ass_document(
    display_cues: Sequence[Mapping[str, object]],
    policy: SubtitleLayoutPolicy | None = None,
) -> str:
    """Render measured display cues as explicit ASS text and box events."""

    policy = policy or SubtitleLayoutPolicy()
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {policy.frame_width}",
        f"PlayResY: {policy.frame_height}",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: SubtitleText,{policy.font_name},{policy.font_size},&H00FFFFFF,&H00FFFFFF,&H00101010,&H00000000,0,0,0,0,100,100,0,0,1,{policy.outline_px},{policy.shadow_px},5,0,0,0,1",
        f"Style: SubtitleBox,{policy.font_name},1,&H99000000,&H99000000,&H99000000,&H99000000,0,0,0,0,100,100,0,0,1,0,0,5,0,0,0,1",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for cue in display_cues:
        start = _ass_time(_integer(cue, "outputStartMs"))
        end = _ass_time(_integer(cue, "outputEndMs"))
        box_width = _integer(cue, "boxWidthPx")
        box_height = _integer(cue, "boxHeightPx")
        center_x = _render_anchor_x(cue, policy, box_width)
        center_y = _render_anchor(cue, policy, box_height)
        box_path = f"m 0 0 l {box_width} 0 l {box_width} {box_height} l 0 {box_height}"
        lines.append(
            f"Dialogue: 0,{start},{end},SubtitleBox,,0,0,0,,{{\\an5\\pos({center_x},{center_y})\\p1}}{box_path}{{\\p0}}"
        )
        text = r"\N".join(_ass_escape_text(str(line)) for line in cue["lines"])
        lines.append(
            f"Dialogue: 1,{start},{end},SubtitleText,,0,0,0,,{{\\an5\\pos({center_x},{center_y})\\fs{_integer(cue, 'fontSize')}}}{text}"
        )
    return "\n".join(lines) + "\n"


def write_ass_subtitles(
    cues: Sequence[Mapping[str, object]],
    path: Path,
    *,
    width: int = 720,
    height: int = 1280,
    policy: SubtitleLayoutPolicy | None = None,
    placement_evidence: Sequence[Mapping[str, object]] | None = None,
    source_to_output: object | None = None,
) -> dict[str, object]:
    """Write a measured ASS overlay and return its diagnostics document."""

    if policy is None:
        policy = SubtitleLayoutPolicy(frame_width=width, frame_height=height)
    elif policy.frame_width != width or policy.frame_height != height:
        raise SubtitleLayoutError("policy frame dimensions do not match ASS dimensions")
    result = layout_subtitles(
        cues,
        policy=policy,
        placement_evidence=placement_evidence,
        source_to_output=source_to_output,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_ass_document(result["displayCues"], policy), encoding="utf-8")
    return result


_DIALOGUE_CATEGORIES = frozenset({"DIALOGUE_SUBTITLE", "CHINESE_DIALOGUE_SUBTITLE"})


def _source_placement(
    identifier: str,
    output_start: int,
    output_end: int,
    evidence: Sequence[Mapping[str, object]] | None,
    source_to_output: object | None,
    policy: SubtitleLayoutPolicy,
) -> dict[str, object]:
    del identifier
    fallback = {
        "anchorSource": "V5_FALLBACK_BOTTOM",
        "sourceDialogueTrackId": None,
        "sourceAnchorXPx": None,
        "sourceAnchorYPx": None,
        "fallbackUsed": True,
        "fallbackReason": "NO_SOURCE_DIALOGUE_EVIDENCE",
    }
    if evidence is None:
        return fallback
    if source_to_output is None:
        fallback["fallbackReason"] = "SOURCE_TIMELINE_MISSING"
        return fallback

    candidates: list[dict[str, object]] = []
    malformed = False
    for raw in evidence:
        if not isinstance(raw, Mapping):
            malformed = True
            continue
        category = str(raw.get("category") or "").strip().upper()
        if category not in _DIALOGUE_CATEGORIES:
            continue
        confidence = str(raw.get("confidence") or "").strip().upper()
        if confidence in {"LOW", "UNKNOWN", "UNAVAILABLE"}:
            continue
        interval = _evidence_interval(raw)
        bbox = _evidence_bbox(raw, policy)
        if interval is None or bbox is None:
            malformed = True
            continue
        try:
            mapped_start, mapped_end = _map_source_interval(source_to_output, *interval)
        except (SubtitleLayoutError, TypeError, ValueError, AttributeError):
            malformed = True
            continue
        overlap = max(0, min(mapped_end, output_end) - max(mapped_start, output_start))
        if overlap <= 0:
            continue
        candidates.append({
            "trackId": _track_id(raw),
            "sourceAnchorX": (bbox[0] + bbox[2]) / 2,
            "sourceAnchorY": (bbox[1] + bbox[3]) / 2,
            "overlap": overlap,
        })
    if not candidates:
        fallback["fallbackReason"] = (
            "MALFORMED_OR_UNSAFE_DIALOGUE_EVIDENCE" if malformed
            else "NO_OVERLAPPING_DIALOGUE_EVIDENCE"
        )
        return fallback

    x_values = [float(candidate["sourceAnchorX"]) for candidate in candidates]
    y_values = [float(candidate["sourceAnchorY"]) for candidate in candidates]
    if max(y_values) - min(y_values) > policy.frame_height * policy.source_anchor_max_spread_ratio:
        fallback["fallbackReason"] = "AMBIGUOUS_DIALOGUE_ANCHOR"
        return fallback
    source_anchor = _median(y_values)
    source_anchor_x = _median(x_values)
    if not policy.frame_height * policy.source_anchor_min_ratio <= source_anchor <= policy.frame_height * policy.source_anchor_max_ratio:
        fallback["fallbackReason"] = "UNSAFE_DIALOGUE_ANCHOR"
        return fallback
    chosen = min(
        candidates,
        key=lambda candidate: (
            abs(float(candidate["sourceAnchorY"]) - source_anchor),
            -int(candidate["overlap"]),
            str(candidate["trackId"] or ""),
        ),
    )
    return {
        "anchorSource": "DIALOGUE_SUBTITLE",
        "sourceDialogueTrackId": chosen["trackId"],
        "sourceAnchorXPx": int(round(source_anchor_x)),
        "sourceAnchorYPx": int(round(source_anchor)),
        "fallbackUsed": False,
        "fallbackReason": None,
    }


def _position_layout(
    layout: Mapping[str, object],
    placement: Mapping[str, object],
    policy: SubtitleLayoutPolicy,
) -> dict[str, object]:
    positioned = dict(layout)
    box_width = int(layout["boxWidthPx"])
    box_height = int(layout["boxHeightPx"])
    fallback_anchor_x = policy.frame_width // 2
    fallback_anchor = policy.frame_height - policy.bottom_safe_margin_px - box_height // 2
    requested_x = placement.get("sourceAnchorXPx")
    requested = placement.get("sourceAnchorYPx")
    if placement.get("anchorSource") == "DIALOGUE_SUBTITLE" and isinstance(requested_x, int):
        anchor_x = requested_x
    else:
        anchor_x = fallback_anchor_x
    if placement.get("anchorSource") == "DIALOGUE_SUBTITLE" and isinstance(requested, int):
        anchor = requested
    else:
        anchor = fallback_anchor
    minimum = box_height // 2
    maximum = policy.frame_height - policy.bottom_safe_margin_px - box_height // 2
    final_anchor = max(minimum, min(anchor, maximum))
    horizontal_margin = policy.frame_width * policy.horizontal_safe_margin_ratio
    minimum_x = math.ceil(horizontal_margin + box_width / 2)
    maximum_x = math.floor(policy.frame_width - horizontal_margin - box_width / 2)
    final_anchor_x = max(minimum_x, min(anchor_x, maximum_x))
    positioned.update({
        "boxLeftPx": final_anchor_x - box_width // 2,
        "boxRightPx": final_anchor_x + box_width // 2,
        "boxTopPx": final_anchor - box_height // 2,
        "boxBottomPx": final_anchor + box_height // 2,
        "finalAnchorXPx": final_anchor_x,
        "finalAnchorYPx": final_anchor,
    })
    return positioned


def _render_anchor(cue: Mapping[str, object], policy: SubtitleLayoutPolicy, box_height: int) -> int:
    value = cue.get("finalAnchorYPx")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return policy.frame_height - policy.bottom_safe_margin_px - box_height // 2


def _render_anchor_x(cue: Mapping[str, object], policy: SubtitleLayoutPolicy, box_width: int) -> int:
    value = cue.get("finalAnchorXPx")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return policy.frame_width // 2


def _evidence_interval(row: Mapping[str, object]) -> tuple[int, int] | None:
    start = row.get("startMs", row.get("sourceStartMs"))
    end = row.get("endMs", row.get("sourceEndMs"))
    if (
        isinstance(start, bool) or isinstance(end, bool)
        or not isinstance(start, int) or not isinstance(end, int)
        or end <= start or start < 0
    ):
        return None
    return start, end


def _evidence_bbox(row: Mapping[str, object], policy: SubtitleLayoutPolicy) -> tuple[float, float, float, float] | None:
    value = row.get("bbox", row.get("bboxPx"))
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        return None
    left, top, right, bottom = (float(item) for item in value)
    if not 0 <= left < right <= policy.frame_width or not 0 <= top < bottom <= policy.frame_height:
        return None
    return left, top, right, bottom


def _track_id(row: Mapping[str, object]) -> str | None:
    value = row.get("trackId", row.get("sourceDialogueTrackId"))
    if value is None or not str(value).strip():
        return None
    return str(value).strip()


def _map_source_interval(source_to_output: object, start: int, end: int) -> tuple[int, int]:
    mapper = getattr(source_to_output, "map_interval", None)
    if not callable(mapper):
        raise SubtitleLayoutError("source timeline must provide the accepted TimelineMap API")
    mapped = mapper(start, end)
    if not isinstance(mapped, tuple) or len(mapped) != 2:
        raise SubtitleLayoutError("source timeline returned an invalid interval")
    return int(mapped[0]), int(mapped[1])


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _display_chunks(text: str, policy: SubtitleLayoutPolicy) -> list[str]:
    tokens = _tokens(text)
    if not tokens:
        raise SubtitleLayoutError("subtitle text is blank")
    if _best_layout(" ".join(tokens), policy, allow_split=False) is not None:
        return [" ".join(tokens)]
    remaining = tokens[:]
    chunks: list[str] = []
    while remaining:
        if _best_layout(" ".join(remaining), policy, allow_split=False) is not None:
            chunks.append(" ".join(remaining))
            break
        viable = [
            index for index in range(1, len(remaining))
            if _best_layout(" ".join(remaining[:index]), policy, allow_split=False) is not None
        ]
        if not viable:
            raise SubtitleLayoutError("a subtitle word cannot fit at minimum font size")
        punctuation = [index for index in viable if _ends_phrase(remaining[index - 1])]
        split_at = max(punctuation or viable)
        chunks.append(" ".join(remaining[:split_at]))
        remaining = remaining[split_at:]
    return chunks


def _best_layout(text: str, policy: SubtitleLayoutPolicy, *, allow_split: bool = True) -> dict[str, object] | None:
    del allow_split
    sizes = list(range(policy.font_size, policy.min_font_size - 1, -policy.font_step))
    if sizes[-1] != policy.min_font_size:
        sizes.append(policy.min_font_size)
    for size in sizes:
        font = _font(policy, size)
        available = _available_line_width(policy)
        tokens = _tokens(text)
        one_line = " ".join(tokens)
        if _measure(font, one_line) <= available:
            lines = [one_line]
        else:
            candidates: list[tuple[float, float, int, list[str], list[int]]] = []
            for index in range(1, len(tokens)):
                candidate_lines = [" ".join(tokens[:index]), " ".join(tokens[index:])]
                widths = [_measure(font, line) for line in candidate_lines]
                if max(widths) <= available:
                    candidates.append((abs(widths[0] - widths[1]), max(widths), index, candidate_lines, widths))
            if not candidates:
                continue
            _difference, _maximum, _index, lines, _widths = min(candidates)
        widths = [_measure(font, line) for line in lines]
        line_height = _line_height(font)
        box_width = _even(math.ceil(max(widths) + 2 * policy.padding_x_px + 2 * policy.outline_px + 2 * policy.shadow_px))
        box_height = _even(math.ceil(
            len(lines) * line_height
            + max(0, len(lines) - 1) * policy.line_spacing_px
            + policy.padding_top_px
            + policy.padding_bottom_px
            + 2 * policy.outline_px
            + 2 * policy.shadow_px
        ))
        box_left = (policy.frame_width - box_width) // 2
        box_right = box_left + box_width
        box_bottom = policy.frame_height - policy.bottom_safe_margin_px
        box_top = box_bottom - box_height
        if box_width <= _box_width_limit(policy) and box_left >= 0 and box_right <= policy.frame_width and box_top >= 0:
            return {
                "text": "\n".join(lines),
                "lines": lines,
                "lineCount": len(lines),
                "fontSize": size,
                "lineWidthsPx": [int(math.ceil(width)) for width in widths],
                "lineHeightPx": line_height,
                "boxWidthPx": box_width,
                "boxHeightPx": box_height,
                "boxLeftPx": box_left,
                "boxTopPx": box_top,
                "boxRightPx": box_right,
                "boxBottomPx": box_bottom,
            }
    return None


def _weighted_intervals(
    start: int,
    end: int,
    layouts: Sequence[Mapping[str, object]],
    policy: SubtitleLayoutPolicy,
) -> list[tuple[int, int]]:
    if len(layouts) == 1:
        return [(start, end)]
    total = end - start
    weights = [max(1, sum(int(width) for width in layout["lineWidthsPx"])) for layout in layouts]
    boundaries = [start]
    accumulated = 0
    for weight in weights[:-1]:
        accumulated += weight
        boundaries.append(start + round(total * accumulated / sum(weights)))
    boundaries.append(end)
    result: list[tuple[int, int]] = []
    for index in range(len(layouts)):
        left = boundaries[index]
        right = boundaries[index + 1]
        remaining = len(layouts) - index - 1
        if right - left < policy.minimum_display_duration_ms and total >= policy.minimum_display_duration_ms * len(layouts):
            right = left + policy.minimum_display_duration_ms
        if right > end - remaining * policy.minimum_display_duration_ms and total >= policy.minimum_display_duration_ms * len(layouts):
            right = end - remaining * policy.minimum_display_duration_ms
        if right <= left:
            raise SubtitleLayoutError("display split interval is too short")
        result.append((left, right))
        if index + 1 < len(boundaries):
            boundaries[index + 1] = right
    return result


def _policy_document(policy: SubtitleLayoutPolicy) -> dict[str, object]:
    return {
        "frameWidthPx": policy.frame_width,
        "frameHeightPx": policy.frame_height,
        "fontName": policy.font_name,
        "fontSizePx": policy.font_size,
        "minimumFontSizePx": policy.min_font_size,
        "maxBoxWidthRatio": policy.max_box_width_ratio,
        "horizontalSafeMarginRatio": policy.horizontal_safe_margin_ratio,
        "paddingXPx": policy.padding_x_px,
        "paddingTopPx": policy.padding_top_px,
        "paddingBottomPx": policy.padding_bottom_px,
        "outlinePx": policy.outline_px,
        "shadowPx": policy.shadow_px,
        "lineSpacingPx": policy.line_spacing_px,
        "bottomSafeMarginPx": policy.bottom_safe_margin_px,
        "sourceAnchorMinRatio": policy.source_anchor_min_ratio,
        "sourceAnchorMaxRatio": policy.source_anchor_max_ratio,
        "sourceAnchorMaxSpreadRatio": policy.source_anchor_max_spread_ratio,
    }


def _box_width_limit(policy: SubtitleLayoutPolicy) -> float:
    return min(
        policy.frame_width * policy.max_box_width_ratio,
        policy.frame_width * (1 - 2 * policy.horizontal_safe_margin_ratio),
    )


def _available_line_width(policy: SubtitleLayoutPolicy) -> float:
    return _box_width_limit(policy) - 2 * (
        policy.padding_x_px + policy.outline_px + policy.shadow_px
    )


def _font(policy: SubtitleLayoutPolicy, size: int):
    try:
        from PIL import ImageFont
    except ImportError as exc:
        raise SubtitleLayoutError("Pillow is required for actual subtitle glyph measurement") from exc
    path = policy.font_path or _default_font_path()
    try:
        return ImageFont.truetype(path, size=size)
    except OSError:
        try:
            return ImageFont.truetype(policy.font_name, size=size)
        except OSError as exc:
            raise SubtitleLayoutError("configured subtitle font cannot be loaded") from exc


def _default_font_path() -> str:
    windir = os.environ.get("WINDIR")
    candidates = [
        Path(windir) / "Fonts" / "arial.ttf" if windir else None,
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return str(candidate)
    return "Arial"


def _measure(font, text: str) -> float:
    if hasattr(font, "getlength"):
        return float(font.getlength(text))
    left, _top, right, _bottom = font.getbbox(text)
    return float(right - left)


def _line_height(font) -> int:
    try:
        ascent, descent = font.getmetrics()
        return max(1, int(ascent + descent))
    except AttributeError:
        top = font.getbbox("Agjp")[3]
        return max(1, int(top))


def _tokens(text: str) -> list[str]:
    return re.findall(r"\S+", text.replace("\r", " ").replace("\n", " "))


def _ends_phrase(token: str) -> bool:
    return bool(re.search(r"[.!?,;:。！？；，、]$", token))


def _text(row: Mapping[str, object]) -> str:
    value = row.get("translatedTextVi")
    if not isinstance(value, str) or not value.strip():
        raise SubtitleLayoutError("translatedTextVi must be nonblank text")
    return " ".join(_tokens(value))


def _identifier(row: Mapping[str, object]) -> str:
    value = str(row.get("utteranceId") or row.get("voiceEventId") or "").strip()
    if not value:
        raise SubtitleLayoutError("subtitle identity is missing")
    return value


def _integer(row: Mapping[str, object], key: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SubtitleLayoutError(f"{key} must be an integer")
    return value


def _even(value: int) -> int:
    return value if value % 2 == 0 else value + 1


def _ass_escape_text(text: str) -> str:
    return text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def _ass_time(value_ms: int) -> str:
    hours, remainder = divmod(max(0, value_ms), 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, centiseconds = divmod(remainder, 1000)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds // 10:02d}"
