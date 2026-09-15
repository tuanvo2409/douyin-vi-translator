"""Deterministic semantic-utterance to timing-cue alignment for CP3A.

This module is deliberately independent of the legacy rendering worker.  It
turns the accepted CP2 rows into a traceable timing projection and keeps
visual-only and unresolved evidence out of the TTS input.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Mapping, Sequence


class CueAlignmentError(ValueError):
    """The accepted CP2 rows cannot be aligned without losing evidence."""


_SPACE = re.compile(r"\s+")
_SENTENCE = re.compile(r"(?<=[.!?;:,…])\s+")
_PHRASE = re.compile(
    r"\s+(?=(?:và|nhưng|mà|nên|để|vì|khi|rồi|thì|còn|giúp|suốt|cho|cả)\b)",
    re.IGNORECASE,
)


def build_alignment(
    records: Sequence[Mapping[str, object]],
    *,
    accepted: Sequence[Mapping[str, object]] | None = None,
    qa_issues: Sequence[Mapping[str, object]] = (),
    review_required: Sequence[Mapping[str, object]] = (),
    source_cues: Mapping[object, Mapping[str, object]] | Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Build a lossless alignment document from CP2 source and accepted rows.

    ``accepted`` is optional for focused callers that already merged source and
    Vietnamese fields.  When supplied, source rows remain authoritative for
    identity, timing, type, and positions; accepted rows only contribute the
    reviewed Vietnamese text and matching evidence.
    """

    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise CueAlignmentError("alignment source records must be a sequence")
    source_by_id: dict[str, Mapping[str, object]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise CueAlignmentError("alignment source row is invalid")
        identifier = _identifier(record)
        if identifier in source_by_id:
            raise CueAlignmentError("alignment source identity is duplicated")
        source_by_id[identifier] = record

    accepted_by_id: dict[str, Mapping[str, object]] = {}
    if accepted is not None:
        for item in accepted:
            if not isinstance(item, Mapping):
                raise CueAlignmentError("accepted localization row is invalid")
            identifier = _identifier(item)
            if identifier in accepted_by_id:
                raise CueAlignmentError("accepted localization identity is duplicated")
            accepted_by_id[identifier] = item

    qa_by_id: dict[str, list[str]] = {}
    for issue in qa_issues:
        if not isinstance(issue, Mapping) or "utteranceId" not in issue:
            continue
        identifier = str(issue["utteranceId"])
        code = str(issue.get("code") or "").strip()
        if code:
            qa_by_id.setdefault(identifier, []).append(code)
    review_ids = {
        str(item.get("utteranceId"))
        for item in review_required
        if isinstance(item, Mapping) and item.get("utteranceId") is not None
    }

    dialogue: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    seen_dialogue: set[str] = set()
    for source in records:
        identifier = _identifier(source)
        classification = _classification(source)
        merged = dict(source)
        if identifier in accepted_by_id:
            accepted_row = accepted_by_id[identifier]
            for field in ("nativeVi", "semanticVi", "style"):
                if field in accepted_row:
                    merged[field] = accepted_row[field]
        if classification == "RESOLVED_DIALOGUE":
            if identifier not in accepted_by_id and not _text(merged, "nativeVi"):
                raise CueAlignmentError(f"accepted dialogue {identifier} has no native Vietnamese")
            row = _align_dialogue_row(
                merged,
                qa_flags=qa_by_id.get(identifier, []),
                explicit_review=identifier in review_ids,
                source_cues=source_cues,
            )
            dialogue.append(row)
            seen_dialogue.add(identifier)
        else:
            excluded.append(_diagnostic_row(merged, classification, qa_by_id.get(identifier, [])))

    if accepted_by_id and set(accepted_by_id) != seen_dialogue:
        missing = sorted(set(accepted_by_id) - seen_dialogue)
        extra = sorted(seen_dialogue - set(accepted_by_id))
        raise CueAlignmentError(f"accepted dialogue coverage mismatch: missing={missing}, extra={extra}")

    aligned_ids = [str(row["utteranceId"]) for row in dialogue]
    if len(aligned_ids) != len(set(aligned_ids)):
        raise CueAlignmentError("aligned utterance identity is duplicated")
    source_counts = {_classification(record): 0 for record in records}
    for record in records:
        source_counts[_classification(record)] = source_counts.get(_classification(record), 0) + 1
    return {
        "schemaVersion": 1,
        "artifact": "cue_alignment_v3",
        "dialogue": dialogue,
        "excluded": excluded,
        "coverage": {
            "sourceRecords": len(records),
            "resolvedDialogue": source_counts.get("RESOLVED_DIALOGUE", 0),
            "unresolvedDialogue": source_counts.get("UNRESOLVED_DIALOGUE", 0),
            "visualOnly": source_counts.get("VISUAL_ONLY", 0),
            "alignedUtteranceCount": len(aligned_ids),
            "alignedUtteranceIds": aligned_ids,
            "ttsEligibleCount": sum(1 for row in dialogue if row["ttsEligible"]),
        },
    }


def alignment_from_cp2_artifact(artifact: Mapping[str, object]) -> dict[str, object]:
    """Build CP3A alignment from the persisted authoritative CP2 v3 artifact."""

    if not isinstance(artifact, Mapping):
        raise CueAlignmentError("CP2 artifact is not an object")
    records = artifact.get("records")
    accepted = artifact.get("accepted")
    if not isinstance(records, list) or not isinstance(accepted, list):
        raise CueAlignmentError("CP2 artifact is missing source or accepted rows")
    return build_alignment(
        records,
        accepted=accepted,
        qa_issues=artifact.get("qa", ()) if isinstance(artifact.get("qa", ()), list) else (),
        review_required=artifact.get("review_required", ()) if isinstance(artifact.get("review_required", ()), list) else (),
        source_cues=artifact.get("source_cues") if isinstance(artifact.get("source_cues"), (dict, list)) else None,
    )


def write_alignment_artifacts(alignment: Mapping[str, object], output_dir: Path) -> tuple[Path, Path]:
    """Persist the JSON and operator-readable CP3A alignment artifacts atomically."""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "cue_alignment_v3.json"
    markdown_path = output_dir / "cue_alignment_v3.md"
    _atomic_json(json_path, alignment)
    _atomic_text(markdown_path, _markdown(alignment))
    return json_path, markdown_path


def _align_dialogue_row(
    record: Mapping[str, object],
    *,
    qa_flags: Sequence[str],
    explicit_review: bool,
    source_cues: Mapping[object, Mapping[str, object]] | Sequence[Mapping[str, object]] | None,
) -> dict[str, object]:
    identifier = _identifier(record)
    native = _text(record, "nativeVi")
    resolved = _text(record, "resolvedZh")
    positions = _positions(record)
    start_ms, end_ms = _time_range(record)
    flags = _ordered_unique([*_flags(record), *qa_flags])
    if "SOURCE_UNCERTAIN" in flags:
        explicit_review = True
    cues: list[dict[str, object]] = []
    if not positions:
        cues.append({
            "position": None,
            "startMs": start_ms,
            "endMs": end_ms,
            "translatedTextVi": native,
            "utteranceId": identifier,
            "sourcePositions": [],
            "alignmentReason": "whole_utterance_without_source_cues",
            "diagnostics": {
                "chunkIndex": 0,
                "chunkCount": 1,
                "timingBasis": "utterance_range_no_cues",
            },
        })
    else:
        chunks = _split_for_slots(native, len(positions))
        ranges = _slot_ranges(start_ms, end_ms, positions, source_cues)
        groups = _group_positions(positions, ranges, len(chunks))
        for index, (chunk, group) in enumerate(zip(chunks, groups)):
            group_positions, group_range = group
            cue_start, cue_end = group_range
            cues.append({
                "position": group_positions[0],
                "startMs": cue_start,
                "endMs": cue_end,
                "translatedTextVi": chunk,
                "utteranceId": identifier,
                "sourcePositions": list(group_positions),
                "alignmentReason": "semantic_phrase_boundary_and_cue_duration",
                "diagnostics": {
                    "chunkIndex": index,
                    "chunkCount": len(chunks),
                    "timingBasis": "source_cue_ranges" if source_cues is not None else "utterance_range_partition",
                },
            })
    if _normalize_join(cues) != _normalize(native):
        raise CueAlignmentError(f"alignment text coverage failed for utterance {identifier}")
    result = {
        "utteranceId": identifier,
        "resolvedZh": resolved,
        "nativeVi": native,
        "sourcePositions": positions,
        "startMs": start_ms,
        "endMs": end_ms,
        "reviewRequired": bool(explicit_review),
        "reviewFlags": flags,
        "ttsEligible": not explicit_review,
        "alignedCues": cues,
    }
    for field in ("intent", "roleplay", "register", "style"):
        if field in record:
            result[field] = record[field]
    return result


def _diagnostic_row(record: Mapping[str, object], classification: str, qa_flags: Sequence[str]) -> dict[str, object]:
    identifier = _identifier(record)
    start_ms, end_ms = _time_range(record)
    flags = _ordered_unique([*_flags(record), *qa_flags, classification])
    return {
        "utteranceId": identifier,
        "type": classification,
        "resolvedZh": _text(record, "resolvedZh"),
        "nativeVi": _text(record, "nativeVi"),
        "sourcePositions": _optional_positions(record),
        "startMs": start_ms,
        "endMs": end_ms,
        "reviewRequired": True,
        "reviewFlags": flags,
        "ttsEligible": False,
        "alignedCues": [],
    }


def _classification(record: Mapping[str, object]) -> str:
    value = str(record.get("classification") or record.get("type") or "").upper()
    if value in {"RESOLVED_DIALOGUE", "UNRESOLVED_DIALOGUE", "VISUAL_ONLY"}:
        return value
    if str(record.get("kind") or "").lower() == "visual_only":
        return "VISUAL_ONLY"
    if str(record.get("source_confidence") or "").lower() in {"low", "unresolved"}:
        return "UNRESOLVED_DIALOGUE"
    return "RESOLVED_DIALOGUE"


def _identifier(record: Mapping[str, object]) -> str:
    value = record.get("utteranceId")
    if value is None or not str(value).strip():
        raise CueAlignmentError("utteranceId is required")
    return str(value)


def _text(record: Mapping[str, object], key: str) -> str:
    value = record.get(key, "")
    if not isinstance(value, str) or not value.strip():
        return ""
    return _normalize(value)


def _positions(record: Mapping[str, object]) -> list[object]:
    raw = record.get("positions", record.get("sourcePositions", record.get("cuePositions", [])))
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        raise CueAlignmentError("source positions must be a sequence")
    result = list(raw)
    if any(item is None for item in result):
        raise CueAlignmentError("source positions cannot contain null")
    return result


def _optional_positions(record: Mapping[str, object]) -> list[object]:
    raw = record.get("positions", record.get("sourcePositions", record.get("cuePositions", [])))
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        raise CueAlignmentError("source positions must be a sequence")
    return list(raw)


def _time_range(record: Mapping[str, object]) -> tuple[int, int]:
    start = record.get("startMs", record.get("start_ms"))
    end = record.get("endMs", record.get("end_ms"))
    if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int) or end <= start:
        raise CueAlignmentError(f"utterance {record.get('utteranceId')} has invalid time range")
    return start, end


def _flags(record: Mapping[str, object]) -> list[str]:
    raw = record.get("source_flags", record.get("sourceFlags", []))
    return [str(item) for item in raw] if isinstance(raw, (list, tuple)) else []


def _split_for_slots(text: str, slot_count: int) -> list[str]:
    if slot_count <= 0:
        raise CueAlignmentError("slot count must be positive")
    chunks = [part.strip() for sentence in _SENTENCE.split(_normalize(text)) if sentence.strip() for part in _PHRASE.split(sentence) if part.strip()]
    chunks = chunks or [_normalize(text)]
    target = min(slot_count, len(chunks))
    while len(chunks) > target:
        index = min(range(len(chunks) - 1), key=lambda item: len(chunks[item].split()) + len(chunks[item + 1].split()))
        chunks[index:index + 2] = [f"{chunks[index]} {chunks[index + 1]}".strip()]
    return chunks


def _slot_ranges(
    start_ms: int,
    end_ms: int,
    positions: Sequence[object],
    source_cues: Mapping[object, Mapping[str, object]] | Sequence[Mapping[str, object]] | None,
) -> list[tuple[int, int]]:
    lookup: dict[str, Mapping[str, object]] = {}
    if isinstance(source_cues, Mapping):
        lookup = {str(key): value for key, value in source_cues.items() if isinstance(value, Mapping)}
    elif isinstance(source_cues, Sequence):
        lookup = {str(item.get("position")): item for item in source_cues if isinstance(item, Mapping) and item.get("position") is not None}
    if lookup:
        candidate: list[tuple[int, int]] = []
        for position in positions:
            cue = lookup.get(str(position))
            cue_start, cue_end = (cue.get("startMs"), cue.get("endMs")) if cue else (None, None)
            if not isinstance(cue_start, int) or not isinstance(cue_end, int) or cue_end <= cue_start:
                candidate = []
                break
            candidate.append((max(start_ms, cue_start), min(end_ms, cue_end)))
        if candidate and all(end > start for start, end in candidate):
            return candidate
    total = end_ms - start_ms
    count = len(positions)
    return [(start_ms + (total * index) // count, start_ms + (total * (index + 1)) // count) for index in range(count)]


def _group_positions(
    positions: Sequence[object],
    ranges: Sequence[tuple[int, int]],
    group_count: int,
) -> list[tuple[list[object], tuple[int, int]]]:
    if group_count <= 0 or group_count > len(positions):
        raise CueAlignmentError("invalid aligned cue count")
    result = []
    for index in range(group_count):
        left = (len(positions) * index) // group_count
        right = (len(positions) * (index + 1)) // group_count
        selected = list(positions[left:right])
        selected_ranges = list(ranges[left:right])
        result.append((selected, (min(item[0] for item in selected_ranges), max(item[1] for item in selected_ranges))))
    return result


def _normalize(value: str) -> str:
    return _SPACE.sub(" ", value.strip())


def _normalize_join(cues: Sequence[Mapping[str, object]]) -> str:
    return _normalize(" ".join(str(cue.get("translatedTextVi") or "") for cue in cues))


def _ordered_unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _markdown(alignment: Mapping[str, object]) -> str:
    lines = [
        "# Cue Alignment v3",
        "",
        "| Utterance | Positions | Source time | Native VI | Aligned cue(s) | Review flags |",
        "|---|---|---|---|---|---|",
    ]
    for row in alignment.get("dialogue", []):
        cues = row.get("alignedCues", [])
        cue_text = "<br>".join(
            f"{cue.get('position')} [{cue.get('startMs')}–{cue.get('endMs')}]: {_cell(cue.get('translatedTextVi'))}"
            for cue in cues
        )
        lines.append(
            f"| {_cell(row.get('utteranceId'))} | {json.dumps(row.get('sourcePositions', []), ensure_ascii=False)} | "
            f"{row.get('startMs')}–{row.get('endMs')} | {_cell(row.get('nativeVi'))} | {cue_text} | "
            f"{_cell(', '.join(row.get('reviewFlags', [])))} |"
        )
    lines.extend([
        "",
        "## DIAGNOSTIC EXCLUSIONS",
        "",
        "| Utterance | Type | Source time | Review flags |",
        "|---|---|---|---|",
    ])
    for row in alignment.get("excluded", []):
        lines.append(f"| {_cell(row.get('utteranceId'))} | {row.get('type')} | {row.get('startMs')}–{row.get('endMs')} | {_cell(', '.join(row.get('reviewFlags', [])))} |")
    lines.extend(["", "## COVERAGE", "", "```json", json.dumps(alignment.get("coverage", {}), ensure_ascii=False, indent=2), "```", ""])
    return "\n".join(lines)


def _cell(value: object) -> str:
    return str(value if value is not None else "").replace("|", "\\|").replace("\n", " ")


def _atomic_json(path: Path, document: object) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
