"""Text-only CP1 artifact writer; it never invokes the legacy processing path."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from av_analyzer import persist_diagnostics
from source_reconciler import reconcile_source
from utterance_builder import build_utterances


def _write_json(path: Path, document: object) -> Path:
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def run_fixture(fixture: Mapping[str, object], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    av_windows = fixture["av_windows"]
    cues = fixture["cues"]
    reconstructed = reconcile_source(cues, av_windows, fixture.get("unresolved"))
    reconstructed["dialogue"] = build_utterances(reconstructed["dialogue"], cues)
    av_path = output_dir / "av_analysis.json"
    persist_diagnostics(av_path, {"utterances": [item for window in av_windows for item in window["spoken"]], "visual_only": [item for window in av_windows for item in window["visual_only"]]})
    source_path = _write_json(output_dir / "source_reconstruction.json", reconstructed)
    ab = {
        "fixture_version": fixture["fixture_version"],
        "before": [{"cue_id": cue["cue_id"], "text_zh": cue.get("asr_text_zh", "")} for cue in cues],
        "after": reconstructed["dialogue"],
        "visual_only": reconstructed["visual_only"],
        "unresolved": reconstructed["unresolved"],
    }
    ab_json = _write_json(output_dir / "multimodal_source_ab.json", ab)
    lines = ["# Multimodal source A/B", "", "## After"]
    lines.extend("- " + item["text_zh"] + " ← " + ", ".join(item["cue_ids"]) for item in reconstructed["dialogue"])
    lines.extend(["", "## Visual only"])
    lines.extend("- " + item["text_zh"] for item in reconstructed["visual_only"])
    lines.extend(["", "## Unresolved"])
    lines.extend("- " + str(item.get("asr_text_zh", "")) + " / " + str(item.get("ocr_text_zh", "")) for item in reconstructed["unresolved"])
    ab_md = output_dir / "multimodal_source_ab.md"
    ab_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [av_path, source_path, ab_json, ab_md]
