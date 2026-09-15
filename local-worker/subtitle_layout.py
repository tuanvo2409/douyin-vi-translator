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
) -> dict[str, object]:
    """Lay out display-only cues without mutating timing or source data."""

    policy = policy or SubtitleLayoutPolicy()
    display_cues: list[dict[str, object]] = []
    split_ids: set[str] = set()
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
        for index, (layout, (output_start, output_end)) in enumerate(zip(layouts, intervals), start=1):
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
                "boxLeftPx": layout["boxLeftPx"],
                "boxTopPx": layout["boxTopPx"],
                "boxRightPx": layout["boxRightPx"],
                "boxBottomPx": layout["boxBottomPx"],
                "outputStartMs": output_start,
                "outputEndMs": output_end,
                "splitFromLongCue": len(chunks) > 1,
            })
    return {
        "schemaVersion": 1,
        "artifact": "subtitle_layout_v5",
        "policy": _policy_document(policy),
        "displayCues": display_cues,
        "maximumLineCount": max((int(cue["lineCount"]) for cue in display_cues), default=0),
        "minimumFontSizeUsed": min((int(cue["fontSize"]) for cue in display_cues), default=0),
        "splitUtteranceCount": len(split_ids),
        "maximumBoxWidthRatio": max(
            (int(cue["boxWidthPx"]) / policy.frame_width for cue in display_cues),
            default=0.0,
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
        center_x = policy.frame_width // 2
        center_y = policy.frame_height - policy.bottom_safe_margin_px - box_height // 2
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
) -> dict[str, object]:
    """Write a measured ASS overlay and return its diagnostics document."""

    if policy is None:
        policy = SubtitleLayoutPolicy(frame_width=width, frame_height=height)
    elif policy.frame_width != width or policy.frame_height != height:
        raise SubtitleLayoutError("policy frame dimensions do not match ASS dimensions")
    result = layout_subtitles(cues, policy=policy)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_ass_document(result["displayCues"], policy), encoding="utf-8")
    return result


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
