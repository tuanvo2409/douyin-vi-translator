"""Deterministic CP1 source reconstruction and text-only A/B artifact writer."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from av_analyzer import persist_diagnostics, validate_av_analysis
from source_reconciler import reconcile_source
from utterance_builder import build_utterances


def _write_json(path: Path, document: object) -> Path:
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _metrics(reconstructed: Mapping[str, object]) -> dict[str, int]:
    dialogue = list(reconstructed["dialogue"])
    visual_only = list(reconstructed["visual_only"])
    unresolved = [item for item in dialogue if "UNRESOLVED" in item.get("flags", [])]
    return {
        "total": len(dialogue) + len(visual_only),
        "resolved_dialogue": len(dialogue) - len(unresolved),
        "unresolved_dialogue": len(unresolved),
        "visual_only": len(visual_only),
    }


def _ordered_records(reconstructed: Mapping[str, object]) -> list[dict[str, object]]:
    records = [dict(item, type="UNRESOLVED_DIALOGUE" if "UNRESOLVED" in item.get("flags", []) else "DIALOGUE") for item in reconstructed["dialogue"]]
    records.extend(dict(item) for item in reconstructed["visual_only"])
    return sorted(records, key=lambda item: (int(item["startMs"]), int(item["endMs"]), str(item.get("type", ""))))


def _cue_text(cues: Sequence[Mapping[str, object]], positions: Sequence[object], key: str) -> str:
    values = []
    wanted = {int(position) for position in positions if isinstance(position, int)}
    for cue in cues:
        if int(cue.get("position", -1)) in wanted:
            value = str(cue.get(key) or "").strip()
            if value and value not in values:
                values.append(value)
    return "; ".join(values)


def _ab_markdown(records: Sequence[Mapping[str, object]], cues: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "# Multimodal source A/B",
        "",
        "| # | Time | Positions | Current sourceTextZh | ASR | OCR | AV spoken | AV subtitle | Resolved Chinese | Confidence | Flags |",
        "|---:|---|---|---|---|---|---|---|---|---|---|",
    ]
    for number, item in enumerate(records, 1):
        start, end = int(item["startMs"]), int(item["endMs"])
        evidence = item.get("evidence", {})
        positions = ", ".join(str(position) for position in item.get("positions", [])) or "—"
        current = _cue_text(cues, item.get("positions", []), "source_text_zh")
        asr = str(evidence.get("asr", ""))
        ocr = str(evidence.get("ocr", ""))
        if not current:
            current = asr or ocr
        resolved = str(item.get("resolvedZh") or "")
        flags = "; ".join(str(flag) for flag in item.get("flags", []))
        lines.append(
            f"| {number} | {start / 1000:.3f}–{end / 1000:.3f} | {positions} | {current} | {asr} | {ocr} | "
            f"{evidence.get('avSpoken', '')} | {evidence.get('avSubtitle', '')} | {resolved} | {item.get('confidence', '')} | {flags} |"
        )
    return "\n".join(lines) + "\n"


def run_fixture(fixture: Mapping[str, object], output_dir: Path) -> list[Path]:
    """Run one canonical, provider-free fixture through the CP1 source path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    av_analysis = validate_av_analysis(fixture["av_analysis"])
    cues = fixture["cues"]
    reconstructed = reconcile_source(cues, av_analysis, fixture.get("unresolved"))
    reconstructed["dialogue"] = build_utterances(reconstructed["dialogue"], cues)
    reconstructed["metrics"] = _metrics(reconstructed)
    records = _ordered_records(reconstructed)
    persist_diagnostics(output_dir / "av_analysis.json", av_analysis)
    source_document = {**reconstructed, "records": records}
    source_path = _write_json(output_dir / "source_reconstruction.json", source_document)
    ab = {
        "fixture_version": fixture["fixture_version"],
        "before": [{"cue_id": cue["cue_id"], "text_zh": cue.get("asr_text_zh", "")} for cue in cues],
        "after": records,
        "visual_only": reconstructed["visual_only"],
        "unresolved": reconstructed["unresolved"],
        "metrics": reconstructed["metrics"],
    }
    ab_json = _write_json(output_dir / "multimodal_source_ab.json", ab)
    ab_md = output_dir / "multimodal_source_ab.md"
    ab_md.write_text(_ab_markdown(records, cues), encoding="utf-8")
    return [output_dir / "av_analysis.json", source_path, ab_json, ab_md]


def _canonical_from_persisted(document: Mapping[str, object]) -> dict[str, object]:
    utterances = []
    visual_only = []
    offsets = (0, 8_000, 16_500)
    chunks = document["avChunks"]
    if not isinstance(chunks, list) or len(chunks) != len(offsets):
        raise ValueError("CP1 requires exactly three persisted AV chunks")
    for offset, chunk in zip(offsets, chunks):
        for item in chunk["items"]:
            start = offset + round(float(item["startSec"]) * 1000)
            end = offset + round(float(item["endSec"]) * 1000)
            spoken = item.get("spokenZh")
            subtitle = item.get("subtitleZh")
            confidence = str(item.get("confidence", "LOW")).upper()
            if spoken is None:
                visual_only.append({"startMs": start, "endMs": end, "textZh": subtitle, "confidence": confidence})
            else:
                utterances.append({
                    "startMs": start,
                    "endMs": end,
                    "spokenZh": spoken,
                    "subtitleZh": subtitle,
                    "speakerRef": None,
                    "intent": None,
                    "emotion": None,
                    "confidence": confidence,
                })
    return {
        "windowStartMs": 0,
        "windowEndMs": int(document["durationMs"]),
        "scene": {"summary": "classroom comedy", "tone": "comic", "register": "colloquial"},
        "utterances": utterances,
        "visualOnly": visual_only,
    }


def _cues_from_persisted(document: Mapping[str, object]) -> list[dict[str, object]]:
    result = []
    for segment in document["segments"]:
        result.append({
            "cue_id": f"cue-{int(segment['position'])}",
            "position": int(segment["position"]),
            "start_ms": int(segment["startMs"]),
            "end_ms": int(segment["endMs"]),
            "asr_text_zh": segment.get("asrTextZh", ""),
            "ocr_text_zh": segment.get("ocrTextZh", ""),
            "source_text_zh": segment.get("sourceTextZh", ""),
            "needs_review": bool(segment.get("needsReview", False)),
        })
    return result


def run_persisted_runtime(fixture_path: Path, output_dir: Path) -> dict[str, object]:
    """Load persisted legacy/AV evidence only; never reads expected output as input."""
    document = json.loads(fixture_path.read_text(encoding="utf-8"))
    return _run_persisted_document(document, output_dir)


def _run_persisted_document(document: Mapping[str, object], output_dir: Path) -> dict[str, object]:
    fixture = {
        "fixture_version": 2,
        "cues": _cues_from_persisted(document),
        "av_analysis": _canonical_from_persisted(document),
    }
    run_fixture(fixture, output_dir)
    return json.loads((output_dir / "source_reconstruction.json").read_text(encoding="utf-8"))


def run_persisted_files(segments_path: Path, chunk_paths: Sequence[Path], output_dir: Path) -> dict[str, object]:
    """Run the current persisted AV chunks plus legacy segments without provider calls."""
    segments_document = json.loads(segments_path.read_text(encoding="utf-8"))
    chunks = [json.loads(path.read_text(encoding="utf-8")) for path in chunk_paths]
    segments = segments_document.get("segments", segments_document)
    segment_end = max((int(segment.get("endMs", 0)) for segment in segments), default=0)
    chunk_end = max(
        (offset + round(float(item["endSec"]) * 1000) for offset, chunk in zip((0, 8_000, 16_500), chunks) for item in chunk.get("items", [])),
        default=0,
    )
    duration = int(segments_document.get("durationMs") or max(segment_end, chunk_end))
    return _run_persisted_document({"durationMs": duration, "segments": segments, "avChunks": chunks}, output_dir)
