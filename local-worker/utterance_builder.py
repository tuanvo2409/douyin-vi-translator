"""Traceable, deterministic cue-to-semantic-utterance mapping."""
from __future__ import annotations

from typing import Mapping, Sequence


def build_utterances(dialogue: Sequence[Mapping[str, object]], cues: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    result = []
    for item in dialogue:
        cue_ids = [str(cue["cue_id"]) for cue in cues if int(item["start_ms"]) < int(cue["end_ms"]) and int(cue["start_ms"]) < int(item["end_ms"])]
        result.append({**item, "cue_ids": cue_ids})
    return result
