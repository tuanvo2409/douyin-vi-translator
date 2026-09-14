"""Text-only CP2 A/B writer; it never invokes audio or rendering code."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Mapping, Sequence
from datetime import datetime, timezone

from scene_memory import build_scene_memory
from vn_localizer import localize
from vn_quality import apply_one_repair, deterministic_issues, llm_issues


def _write(path: Path, document: object) -> Path:
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def records_from_source_reconstruction(source: Mapping[str, object], old_benchmark: Mapping[str, object] | None = None) -> list[dict[str, object]]:
    """Convert a CP1 reconstruction document to lossless CP2 input rows."""
    utterances = source.get("utterances")
    if not isinstance(utterances, list):
        raise ValueError("source reconstruction utterances are invalid")
    segments = old_benchmark.get("segments", []) if isinstance(old_benchmark, Mapping) else []
    if not isinstance(segments, list):
        raise ValueError("old benchmark segments are invalid")
    result, seen = [], set()
    for item in utterances:
        if not isinstance(item, Mapping) or "utteranceId" not in item or not isinstance(item.get("startMs"), int) or not isinstance(item.get("endMs"), int):
            raise ValueError("source reconstruction utterance is invalid")
        identifier = str(item["utteranceId"])
        if not identifier or identifier in seen or item["endMs"] < item["startMs"]:
            raise ValueError("source reconstruction identity is invalid")
        seen.add(identifier)
        visual = item.get("visualOnly") is True
        unresolved = not visual and (str(item.get("currentSourceStatus", "")).upper() == "UNRESOLVED" or str(item.get("confidence", "")).lower() in {"low", "unresolved"})
        kind = "VISUAL_ONLY" if visual else "UNRESOLVED_DIALOGUE" if unresolved else "RESOLVED_DIALOGUE"
        old_vi = ""
        if not visual:
            overlaps = [segment for segment in segments if isinstance(segment, Mapping) and isinstance(segment.get("startMs"), int) and isinstance(segment.get("endMs"), int) and min(item["endMs"], segment["endMs"]) > max(item["startMs"], segment["startMs"])]
            if overlaps:
                old_vi = str(max(overlaps, key=lambda segment: min(item["endMs"], segment["endMs"]) - max(item["startMs"], segment["startMs"])).get("currentVi", ""))
        result.append({"utteranceId": identifier, "start_ms": item["startMs"], "end_ms": item["endMs"], "resolvedZh": str(item.get("resolvedZh", "")), "oldVi": old_vi, "source_confidence": str(item.get("confidence", "")).lower(), "kind": "visual_only" if visual else "dialogue", "type": "DIALOGUE" if kind == "RESOLVED_DIALOGUE" else kind, "classification": kind})
    return result


def _record_classification(record):
    if record.get("classification") in {"RESOLVED_DIALOGUE", "UNRESOLVED_DIALOGUE", "VISUAL_ONLY"}:
        return record["classification"]
    if record.get("type") in {"RESOLVED_DIALOGUE", "UNRESOLVED_DIALOGUE", "VISUAL_ONLY"}:
        return record["type"]
    if record.get("kind") == "visual_only":
        return "VISUAL_ONLY"
    if record.get("source_confidence") in {"low", "unresolved"}:
        return "UNRESOLVED_DIALOGUE"
    return "RESOLVED_DIALOGUE"


def _record_type(record):
    classification = _record_classification(record)
    return "DIALOGUE" if classification == "RESOLVED_DIALOGUE" else classification


def _coverage(records, accepted_records, review_required):
    accepted = {str(item["utteranceId"]) for item in accepted_records}
    review = {str(item["utteranceId"]) for item in review_required}
    types = [str(item.get("classification", "")) for item in records]
    resolved = [item for item in records if item.get("classification") == "RESOLVED_DIALOGUE"]
    unresolved = [item for item in records if item.get("classification") == "UNRESOLVED_DIALOGUE"]
    visual = [item for item in records if item.get("classification") == "VISUAL_ONLY"]
    if len({str(item.get("utteranceId")) for item in records}) != len(records) or set(str(item["utteranceId"]) for item in resolved) != accepted or set(str(item["utteranceId"]) for item in unresolved) != review or any(kind not in {"RESOLVED_DIALOGUE", "UNRESOLVED_DIALOGUE", "VISUAL_ONLY"} for kind in types):
        raise ValueError("CP2 coverage reconciliation failed")
    return {"sourceRecords": len(records), "dialogueResolved": len(resolved), "dialogueUnresolved": len(unresolved), "visualOnly": len(visual), "semanticTranslated": len(accepted_records), "nativeTranslated": len(accepted_records), "qaReviewed": len(accepted_records)}


def _markdown(records, memory, accepted_records, review_required, issues) -> str:
    accepted = {str(item["utteranceId"]): item for item in accepted_records}
    coverage = _coverage(records, accepted_records, review_required)
    lines = ["# Multimodal Vietnamese A/B", "", "| # | Time | Type | Resolved Chinese | Old DUBVI VI | Semantic VI | Native VI | QA | Source confidence |", "|---|---|---|---|---|---|---|---|---|"]
    for index, record in enumerate(records, start=1):
        item = accepted.get(str(record.get("utteranceId")), {})
        time = str(record.get("start_ms", "")) + "–" + str(record.get("end_ms", ""))
        qa = ", ".join(issue["code"] for issue in issues if issue["utteranceId"] == str(record.get("utteranceId")))
        lines.append("| " + str(index) + " | " + time + " | " + str(record.get("type", "")) + " | " + str(record.get("resolvedZh", "")) + " | " + str(record.get("oldVi", "")) + " | " + str(item.get("semanticVi", "")) + " | " + str(item.get("nativeVi", "")) + " | " + qa + " | " + str(record.get("source_confidence", "")) + " |")
    lines.extend(["", "## RECONCILE COUNTS", *[key + ": " + str(value) for key, value in coverage.items()], "", "## SCENE SUMMARY", str(memory["summary"]), "", "## TONE", str(memory["tone"]), "", "## REGISTER", str(memory["register"]), "", "## SPEAKER/RELATIONSHIP STATE", json.dumps({"speakers": memory["speakers"], "relationships": memory["relationships"]}, ensure_ascii=False), "", "## ROLEPLAY TERMS", json.dumps(memory["roleplayTerms"], ensure_ascii=False), "", "## RUNNING JOKES", json.dumps(memory["runningJokes"], ensure_ascii=False), "", "## UNCERTAINTIES", json.dumps(review_required, ensure_ascii=False)])
    return "\n".join(lines) + "\n"


class _RunObserver:
    def __init__(self, output_dir: Path, model: str) -> None:
        self.output_dir = output_dir
        self.phase = "start"
        self.state = {"status": "running", "phase": self.phase, "startedAt": _now(), "finishedAt": None, "providerCallsAttempted": 0, "providerCallsCompleted": 0, "calls": [], "artifacts": [], "lastError": None, "model": model}
        self._persist()

    def _persist(self) -> None:
        _write(self.output_dir / "cp2_run_state.json", self.state)

    def call(self, operation: str, request: Callable[[dict[str, object]], object], payload: dict[str, object]) -> object:
        self.phase = operation
        self.state["phase"] = operation
        call = {"operation": operation, "model": self.state["model"], "attempt": 1, "state": "attempted", "httpStatus": None, "errorType": None}
        self.state["providerCallsAttempted"] += 1
        self.state["calls"].append(call)
        self._persist()
        try:
            result = request(payload)
        except Exception as error:
            call["state"] = "failed"
            call["errorType"] = type(error).__name__
            self._persist()
            raise
        call["state"] = "completed"
        self.state["providerCallsCompleted"] += 1
        self._persist()
        return result

    def artifact(self, path: Path) -> None:
        self.state["artifacts"].append(path.name)
        self._persist()

    def complete(self) -> None:
        self.state.update({"status": "complete", "phase": "complete", "finishedAt": _now()})
        self._persist()

    def fail(self, error: Exception) -> None:
        self.state.update({"status": "failed", "finishedAt": _now(), "lastError": {"type": type(error).__name__, "message": self.phase + " failed"}})
        self._persist()
        _write(self.output_dir / "cp2_failure.json", {"phase": self.phase, "errorType": type(error).__name__, "message": self.phase + " failed", "providerCallsAttempted": self.state["providerCallsAttempted"], "providerCallsCompleted": self.state["providerCallsCompleted"], "artifactsCreated": self.state["artifacts"]})


def run_text_ab(records: Sequence[Mapping[str, object]], output_dir: Path, *, scene_evidence: Mapping[str, object], request: Callable[[dict[str, object]], object]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [{**record, "type": _record_type(record), "classification": _record_classification(record)} for record in records]
    memory = build_scene_memory(scene_evidence)
    result = localize(records, scene_memory=memory, request=request, profile="vn_native_light_comedy")
    issues = deterministic_issues(result["accepted"]) + llm_issues(result["accepted"], request)
    artifact = {"records": [dict(record) for record in records], "scene_memory": memory, "accepted": result["accepted"], "review_required": result["review_required"], "qa": issues, "coverage": _coverage(records, result["accepted"], result["review_required"])}
    json_path = _write(output_dir / "multimodal_vi_ab_25s.json", artifact)
    md_path = output_dir / "multimodal_vi_ab_25s.md"
    temporary = md_path.with_name("." + md_path.name + ".tmp")
    temporary.write_text(_markdown(records, memory, result["accepted"], result["review_required"], issues), encoding="utf-8")
    os.replace(temporary, md_path)
    return [json_path, md_path]


def run_observed_text_ab(records: Sequence[Mapping[str, object]], output_dir: Path, *, scene_request: Callable[[dict[str, object]], object], request: Callable[[dict[str, object]], object], model: str) -> list[Path]:
    """Run CP2 with durable call accounting before every provider boundary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    observer = _RunObserver(output_dir, model)
    try:
        scene_evidence = observer.call("scene", scene_request, {"mode": "scene", "utterances": [dict(item) for item in records if item.get("kind") == "dialogue"]})
        created = run_text_ab(records, output_dir, scene_evidence=scene_evidence, request=lambda payload: observer.call(str(payload["mode"]), request, payload))
        for path in created:
            observer.artifact(path)
        observer.complete()
        return created
    except Exception as error:
        observer.fail(error)
        raise


def repair_completed_run(output_dir: Path, *, request: Callable[[dict[str, object]], object]) -> None:
    """Apply at most one post-QA repair batch without rerunning prior calls."""
    state_path = output_dir / "cp2_run_state.json"
    artifact_path = output_dir / "multimodal_vi_ab_25s.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if state.get("status") != "complete" or any(call.get("operation") == "repair" for call in state.get("calls", [])):
        raise ValueError("repair is unavailable")
    flagged = [item for item in artifact["accepted"] if any(issue["utteranceId"] == item["utteranceId"] for issue in artifact["qa"])]
    if not flagged:
        return
    observer = object.__new__(_RunObserver)
    observer.output_dir, observer.state, observer.phase = output_dir, state, "repair"
    observer.state.update({"status": "running", "phase": "repair", "finishedAt": None})
    observer._persist()
    try:
        document = observer.call("repair", request, {"mode": "repair", "utterances": flagged, "qa_flags": artifact["qa"]})
        translations = document.get("translations") if isinstance(document, dict) else None
        if not isinstance(translations, list) or {item.get("utteranceId") for item in translations if isinstance(item, dict)} != {item["utteranceId"] for item in flagged}:
            raise ValueError("repair response identity is invalid")
        repairs = {item["utteranceId"]: item for item in translations if isinstance(item, dict)}
        accepted = [apply_one_repair(item, repairs[item["utteranceId"]]) if item["utteranceId"] in repairs else item for item in artifact["accepted"]]
        artifact["qa_history"] = artifact["qa"]
        artifact["qa"] = deterministic_issues(accepted)
        artifact["accepted"] = accepted
        _write(artifact_path, artifact)
        md_path = output_dir / "multimodal_vi_ab_25s.md"
        temporary = md_path.with_name("." + md_path.name + ".tmp")
        records = artifact.get("records")
        if not isinstance(records, list):
            raise ValueError("repair artifact is missing source rows")
        temporary.write_text(_markdown(records, artifact["scene_memory"], accepted, artifact["review_required"], artifact["qa"]), encoding="utf-8")
        os.replace(temporary, md_path)
        observer.complete()
    except Exception as error:
        observer.fail(error)
        raise
