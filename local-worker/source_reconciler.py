"""Evidence-first source reconciliation with no OCR-confidence authority shortcut."""
from __future__ import annotations

from typing import Mapping, Sequence


def _overlaps(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    return int(left["start_ms"]) < int(right["end_ms"]) and int(right["start_ms"]) < int(left["end_ms"])


def reconcile_source(cues: Sequence[Mapping[str, object]], av_windows: Sequence[Mapping[str, object]], unresolved: Mapping[str, object] | None = None) -> dict[str, list[dict[str, object]]]:
    av_utterances = [item for window in av_windows for item in window.get("spoken", [])]
    visual_only = [dict(item, kind="visual_only") for window in av_windows for item in window.get("visual_only", [])]
    dialogue = []
    covered_ids: set[str] = set()
    for item in av_utterances:
        linked = [str(cue["cue_id"]) for cue in cues if _overlaps(item, cue)]
        covered_ids.update(linked)
        dialogue.append(dict(item, evidence="av", confidence="high", cue_ids=linked))
    for cue in cues:
        cue_id = str(cue["cue_id"])
        if cue_id in covered_ids:
            continue
        asr = str(cue.get("asr_text_zh") or "").strip()
        ocr = str(cue.get("ocr_text_zh") or "").strip()
        if asr and asr == ocr:
            dialogue.append({"start_ms": cue["start_ms"], "end_ms": cue["end_ms"], "text_zh": asr, "evidence": "asr_ocr_agree", "confidence": "low", "cue_ids": [cue_id]})
    unresolved_items = []
    if unresolved:
        unresolved_items.append(dict(unresolved, status="unresolved"))
    dialogue.sort(key=lambda item: (int(item["start_ms"]), int(item["end_ms"]), str(item["text_zh"])))
    visual_only.sort(key=lambda item: (int(item["start_ms"]), int(item["end_ms"])))
    return {"dialogue": dialogue, "visual_only": visual_only, "unresolved": unresolved_items}
