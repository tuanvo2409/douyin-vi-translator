"""Traceable, deterministic cue-to-semantic-utterance mapping."""
from __future__ import annotations

from typing import Mapping, Sequence


def build_utterances(dialogue: Sequence[Mapping[str, object]], cues: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    result = []
    for item in dialogue:
        start_ms = int(item.get("startMs", item.get("start_ms")))
        end_ms = int(item.get("endMs", item.get("end_ms")))
        cue_ids = [str(cue["cue_id"]) for cue in cues if start_ms < int(cue["end_ms"]) and int(cue["start_ms"]) < end_ms]
        positions = item.get("positions")
        if not isinstance(positions, list):
            positions = sorted({int(cue["position"]) for cue in cues if str(cue["cue_id"]) in cue_ids})
        result.append({**item, "cue_ids": cue_ids, "positions": positions})
    return result
