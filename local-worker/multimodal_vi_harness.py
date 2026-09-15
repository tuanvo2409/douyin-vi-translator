"""Text-only CP2 A/B writer; it never invokes audio or rendering code."""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
from typing import Callable, Mapping, Sequence
from datetime import datetime, timezone

from scene_memory import build_scene_memory, build_terminology_evidence
from vn_localizer import localize
from llm_transport import StructuredResponseError
from vn_quality import apply_one_repair, deterministic_issues, llm_issues, validate_qa_document
from cp2_gemini_adapter import validate_scene_document


DEFAULT_LOCALIZATION_MODEL = "gemini-2.5-flash"
DEFAULT_QA_MODEL = "gemini-2.5-flash"


def model_for_operation(operation: str) -> str:
    """Resolve explicit CP2 stage models without an implicit Lite downgrade."""
    if operation == "qa":
        return os.environ.get("DUBVI_QA_MODEL", "").strip() or DEFAULT_QA_MODEL
    return os.environ.get("DUBVI_LOCALIZATION_MODEL", "").strip() or DEFAULT_LOCALIZATION_MODEL


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
        raise ValueError("canonical utterance list is required")
    if "schemaVersion" in source and source.get("schemaVersion") != 2:
        raise ValueError("unsupported canonical source schema version")
    if "identityVersion" in source and source.get("identityVersion") != 1:
        raise ValueError("unsupported canonical identity version")
    segments = old_benchmark.get("segments", []) if isinstance(old_benchmark, Mapping) else []
    if not isinstance(segments, list):
        raise ValueError("old benchmark segments are invalid")
    result, seen = [], set()
    for item in utterances:
        if not isinstance(item, Mapping) or "utteranceId" not in item or item.get("utteranceId") is None or not str(item.get("utteranceId")).strip():
            raise ValueError("canonical utterance identity is required")
        if not isinstance(item.get("startMs"), int) or isinstance(item.get("startMs"), bool) or not isinstance(item.get("endMs"), int) or isinstance(item.get("endMs"), bool):
            raise ValueError("canonical utterance timing is required")
        identifier = str(item["utteranceId"])
        if not identifier or identifier in seen or item["endMs"] < item["startMs"]:
            raise ValueError("canonical utterance identity is invalid")
        seen.add(identifier)
        visual = item.get("visualOnly") is True
        flags = item.get("flags", item.get("source_flags", []))
        if not isinstance(flags, list):
            raise ValueError("canonical source flags are invalid")
        unresolved = not visual and ("UNRESOLVED" in {str(flag).upper() for flag in flags} or str(item.get("currentSourceStatus", "")).upper() == "UNRESOLVED" or str(item.get("confidence", "")).lower() in {"low", "unresolved"})
        kind = "VISUAL_ONLY" if visual else "UNRESOLVED_DIALOGUE" if unresolved else "RESOLVED_DIALOGUE"
        old_vi = ""
        if not visual:
            overlaps = [segment for segment in segments if isinstance(segment, Mapping) and isinstance(segment.get("startMs"), int) and isinstance(segment.get("endMs"), int) and min(item["endMs"], segment["endMs"]) > max(item["startMs"], segment["startMs"])]
            if overlaps:
                old_vi = str(max(overlaps, key=lambda segment: min(item["endMs"], segment["endMs"]) - max(item["startMs"], segment["startMs"])).get("currentVi", ""))
        positions = item.get("positions", item.get("cuePositions", []))
        if positions is None:
            positions = []
        if not isinstance(positions, list):
            raise ValueError("canonical source positions are invalid")
        result.append({"utteranceId": identifier, "start_ms": item["startMs"], "end_ms": item["endMs"], "resolvedZh": str(item.get("resolvedZh", "")), "spokenZh": str(item.get("spokenZh", "")), "subtitleZh": str(item.get("subtitleZh", "")), "oldVi": old_vi, "source_confidence": str(item.get("confidence", "")).lower(), "positions": positions, "evidence": item.get("evidence", {}), "source_flags": list(flags), "kind": "visual_only" if visual else "dialogue", "type": "DIALOGUE" if kind == "RESOLVED_DIALOGUE" else kind, "classification": kind})
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
    lines = ["# Multimodal Vietnamese A/B", "", "| # | Time | Positions | Type | Resolved Chinese | Old DUBVI VI | CP2-v1 Native VI | CP2-v2 Semantic VI | CP2-v2 Native VI | QA |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for index, record in enumerate(records, start=1):
        item = accepted.get(str(record.get("utteranceId")), {})
        time = str(record.get("start_ms", "")) + "–" + str(record.get("end_ms", ""))
        qa = ", ".join(issue["code"] for issue in issues if issue["utteranceId"] == str(record.get("utteranceId")))
        positions = json.dumps(record.get("positions", []), ensure_ascii=False, separators=(",", ":"))
        lines.append("| " + str(index) + " | " + time + " | " + positions + " | " + str(record.get("type", "")) + " | " + str(record.get("resolvedZh", "")) + " | " + str(record.get("oldVi", "")) + " | " + str(item.get("oldNativeVi", item.get("oldVi", record.get("oldVi", "")))) + " | " + str(item.get("semanticVi", "")) + " | " + str(item.get("nativeVi", "")) + " | " + qa + " |")
    lines.extend(["", "## RECONCILE COUNTS", *[key + ": " + str(value) for key, value in coverage.items()], "", "## SCENE SUMMARY", str(memory["summary"]), "", "## TONE", str(memory["tone"]), "", "## REGISTER", str(memory["register"]), "", "## SPEAKER/RELATIONSHIP STATE", json.dumps({"speakers": memory["speakers"], "relationships": memory["relationships"]}, ensure_ascii=False), "", "## ROLEPLAY TERMS", json.dumps(memory["roleplayTerms"], ensure_ascii=False), "", "## RUNNING JOKES", json.dumps(memory["runningJokes"], ensure_ascii=False), "", "## UNCERTAINTIES", json.dumps(review_required, ensure_ascii=False)])
    return "\n".join(lines) + "\n"


_CORE_STAGES = ("scene", "semantic", "native", "qa")


def _model_pair(model: str | Mapping[str, str]) -> tuple[str, str]:
    if isinstance(model, Mapping):
        localization = str(model.get("localizationModel") or model.get("localization") or DEFAULT_LOCALIZATION_MODEL)
        qa = str(model.get("qaModel") or model.get("qa") or localization)
        return localization, qa
    value = str(model)
    return value, value


def _input_contract(records: Sequence[Mapping[str, object]], model: str | Mapping[str, str], profile: str, pipeline_version: str) -> dict[str, str]:
    encoded = json.dumps([dict(record) for record in records], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    localization_model, qa_model = _model_pair(model)
    return {
        "sourceFingerprint": hashlib.sha256(encoded).hexdigest(),
        "profile": profile,
        "localizationModel": localization_model,
        "qaModel": qa_model,
        "pipelineVersion": pipeline_version,
    }


def _valid_translation_payload(payload: object, key: str, expected_ids: set[str]) -> bool:
    if not isinstance(payload, list):
        return False
    seen = set()
    for item in payload:
        if not isinstance(item, dict) or set(item) != {"utteranceId", key} or not isinstance(item.get("utteranceId"), str) or item["utteranceId"] not in expected_ids or item["utteranceId"] in seen or not isinstance(item.get(key), str) or not item[key].strip():
            return False
        seen.add(item["utteranceId"])
    return seen == expected_ids


def _valid_qa_payload(payload: object, records: Sequence[Mapping[str, object]]) -> bool:
    if not isinstance(payload, list):
        return False
    base_keys = {"utteranceId", "code", "severity", "reason", "evidence"}
    documents = []
    for item in payload:
        if not isinstance(item, dict) or not base_keys <= set(item) or set(item) - base_keys - {"originalCode"}:
            return False
        documents.append({key: item[key] for key in base_keys})
    try:
        validate_qa_document({"issues": documents}, records)
    except ValueError:
        return False
    return True


def _state_is_compatible(state: object, input_contract: Mapping[str, str], records: Sequence[Mapping[str, object]]) -> bool:
    if not isinstance(state, dict) or state.get("version") != 2 or state.get("input") != dict(input_contract):
        return False
    stages = state.get("stages")
    if not isinstance(stages, dict):
        return False
    expected_ids = {str(item["utteranceId"]) for item in records if _record_classification(item) == "RESOLVED_DIALOGUE"}
    for name in _CORE_STAGES:
        stage = stages.get(name)
        if not isinstance(stage, dict) or stage.get("status") not in {"pending", "completed", "failed"}:
            return False
        if stage.get("status") == "completed" and stage.get("payload") is None:
            return False
        if stage.get("status") == "completed":
            payload = stage.get("payload")
            if name == "scene":
                try:
                    validate_scene_document(payload)
                except ValueError:
                    return False
            if name == "semantic" and not _valid_translation_payload(payload, "semanticVi", expected_ids):
                return False
            if name == "native" and not _valid_translation_payload(payload, "nativeVi", expected_ids):
                return False
            if name == "qa" and not _valid_qa_payload(payload, [item for item in records if _record_classification(item) == "RESOLVED_DIALOGUE"]):
                return False
    return True


def _archive_incompatible_state(state_path: Path) -> None:
    if not state_path.is_file():
        return
    archive = state_path.with_name("cp2_run_state_superseded.json")
    index = 1
    while archive.exists():
        archive = state_path.with_name("cp2_run_state_superseded_" + str(index) + ".json")
        index += 1
    os.replace(state_path, archive)


def _load_compatible_state(output_dir: Path, input_contract: Mapping[str, str], records: Sequence[Mapping[str, object]]) -> dict[str, object] | None:
    state_path = output_dir / "cp2_run_state.json"
    if not state_path.is_file():
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        _archive_incompatible_state(state_path)
        return None
    if not _state_is_compatible(state, input_contract, records):
        _archive_incompatible_state(state_path)
        return None
    return state


def _cached_stage_document(operation: str, payload: object) -> object:
    if operation in {"semantic", "native"}:
        if not isinstance(payload, list):
            raise ValueError("persisted localization stage is invalid")
        return {"translations": payload}
    if operation == "qa":
        if not isinstance(payload, list):
            raise ValueError("persisted QA stage is invalid")
        return {"issues": payload}
    raise ValueError("unsupported persisted stage")


class _RunObserver:
    def __init__(self, output_dir: Path, model: str | Mapping[str, str], input_contract: Mapping[str, str], existing: dict[str, object] | None = None) -> None:
        self.output_dir = output_dir
        self.phase = "start"
        localization_model, qa_model = _model_pair(model)
        self.state = existing or {
            "version": 2,
            "status": "running",
            "phase": self.phase,
            "startedAt": _now(),
            "finishedAt": None,
            "providerCallsAttempted": 0,
            "providerCallsCompleted": 0,
            "calls": [],
            "artifacts": [],
            "lastError": None,
            "model": localization_model if localization_model == qa_model else {"localization": localization_model, "qa": qa_model},
            "input": dict(input_contract),
            "stages": {name: {"status": "pending", "payload": None} for name in _CORE_STAGES},
            "qaSerializationRetries": 0,
        }
        if existing is not None:
            self.state.update({"status": "running", "phase": self.phase, "finishedAt": None, "lastError": None})
        self._persist()

    def _persist(self) -> None:
        _write(self.output_dir / "cp2_run_state.json", self.state)

    def call(self, operation: str, request: Callable[[dict[str, object]], object], payload: dict[str, object]) -> object:
        self.phase = operation
        self.state["phase"] = operation
        model = self.state["input"]["qaModel"] if operation == "qa" else self.state["input"]["localizationModel"]
        call = {"operation": operation, "model": model, "attempt": 1, "state": "attempted", "httpStatus": None, "errorType": None}
        self.state["providerCallsAttempted"] += 1
        self.state["calls"].append(call)
        self._persist()
        try:
            result = request(payload)
        except Exception as error:
            call["state"] = "failed"
            call["errorType"] = type(error).__name__
            self.state.setdefault("stages", {})[operation] = {"status": "failed", "payload": None}
            self._persist()
            raise
        call["state"] = "completed"
        self.state["providerCallsCompleted"] += 1
        self._persist()
        return result

    def stage(self, operation: str, payload: object) -> None:
        self.state.setdefault("stages", {})[operation] = {"status": "completed", "payload": payload}
        self._persist()

    def artifact(self, path: Path) -> None:
        if path.name not in self.state["artifacts"]:
            self.state["artifacts"].append(path.name)
        self._persist()

    def complete(self) -> None:
        self.state.update({"status": "complete", "phase": "complete", "finishedAt": _now()})
        self._persist()

    def fail(self, error: Exception) -> None:
        if self.phase in self.state.get("stages", {}) and self.state["stages"][self.phase].get("status") != "completed":
            self.state["stages"][self.phase] = {"status": "failed", "payload": None}
        self.state.update({"status": "failed", "finishedAt": _now(), "lastError": {"type": type(error).__name__, "message": self.phase + " failed"}})
        self._persist()
        _write(self.output_dir / "cp2_failure.json", {"phase": self.phase, "errorType": type(error).__name__, "message": self.phase + " failed", "providerCallsAttempted": self.state["providerCallsAttempted"], "providerCallsCompleted": self.state["providerCallsCompleted"], "artifactsCreated": self.state["artifacts"]})


def run_text_ab(records: Sequence[Mapping[str, object]], output_dir: Path, *, scene_evidence: Mapping[str, object], request: Callable[[dict[str, object]], object], artifact_stem: str = "multimodal_vi_ab_25s", profile: str = "vn_native_light_comedy", on_stage=None) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [{**record, "type": _record_type(record), "classification": _record_classification(record)} for record in records]
    memory = build_scene_memory(scene_evidence)
    memory["terminologyEvidence"] = build_terminology_evidence(records)
    result = localize(records, scene_memory=memory, request=request, profile=profile, on_stage=on_stage)
    provider_issues = llm_issues(result["accepted"], request, scene_memory=memory, on_validated=(lambda payload: on_stage("qa", payload) if on_stage is not None else None))
    issues = deterministic_issues(result["accepted"], scene_memory=memory) + provider_issues
    artifact = {"records": [dict(record) for record in records], "scene_memory": memory, "accepted": result["accepted"], "review_required": result["review_required"], "qa": issues, "coverage": _coverage(records, result["accepted"], result["review_required"])}
    json_path = _write(output_dir / (artifact_stem + ".json"), artifact)
    md_path = output_dir / (artifact_stem + ".md")
    temporary = md_path.with_name("." + md_path.name + ".tmp")
    temporary.write_text(_markdown(records, memory, result["accepted"], result["review_required"], issues), encoding="utf-8")
    os.replace(temporary, md_path)
    return [json_path, md_path]


def run_observed_text_ab(records: Sequence[Mapping[str, object]], output_dir: Path, *, scene_request: Callable[[dict[str, object]], object], request: Callable[[dict[str, object]], object], model: str | Mapping[str, str], artifact_stem: str = "multimodal_vi_ab_25s", profile: str = "vn_native_light_comedy", pipeline_version: str = "cp2-v2") -> list[Path]:
    """Run CP2 with durable call accounting before every provider boundary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_records = [{**record, "type": _record_type(record), "classification": _record_classification(record)} for record in records]
    input_contract = _input_contract(normalized_records, model, profile, pipeline_version)
    observer = _RunObserver(output_dir, model, input_contract, _load_compatible_state(output_dir, input_contract, normalized_records))
    try:
        scene_stage = observer.state["stages"]["scene"]
        if scene_stage.get("status") == "completed":
            scene_evidence = scene_stage["payload"]
        else:
            scene_evidence = observer.call("scene", scene_request, {"mode": "scene", "utterances": [dict(item) for item in normalized_records if item.get("kind") == "dialogue"]})
            scene_evidence = validate_scene_document(scene_evidence)
            observer.stage("scene", scene_evidence)

        def observed_request(payload):
            operation = str(payload["mode"])
            stage = observer.state["stages"].get(operation, {})
            if stage.get("status") == "completed":
                return _cached_stage_document(operation, stage.get("payload"))
            try:
                return observer.call(operation, request, payload)
            except StructuredResponseError:
                if operation != "qa" or int(observer.state.get("qaSerializationRetries", 0)) >= 1:
                    raise
                observer.state["qaSerializationRetries"] = int(observer.state.get("qaSerializationRetries", 0)) + 1
                observer._persist()
                return observer.call(operation, request, payload)

        created = run_text_ab(normalized_records, output_dir, scene_evidence=scene_evidence, request=observed_request, artifact_stem=artifact_stem, profile=profile, on_stage=observer.stage)
        for path in created:
            observer.artifact(path)
        observer.complete()
        return created
    except Exception as error:
        observer.fail(error)
        raise


_NON_REPAIRABLE_CODES = {"SOURCE_UNCERTAIN", "UNRESOLVED_SOURCE"}


def _repairable_issues(issues: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [dict(issue) for issue in issues if str(issue.get("code", "")) not in _NON_REPAIRABLE_CODES]


def _serious_repairable_issues(issues: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    serious_codes = {"TERM_MISTRANSLATION", "SEMANTIC_DRIFT", "CONTEXT_LOSS", "ROLEPLAY_INCONSISTENCY"}
    return [
        dict(issue)
        for issue in issues
        if str(issue.get("code", "")) not in _NON_REPAIRABLE_CODES
        and (str(issue.get("code", "")) in serious_codes or str(issue.get("severity", "")).upper() == "HIGH")
    ]


def _repair_context(records: Sequence[Mapping[str, object]], accepted: Sequence[Mapping[str, object]], flagged: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    ordered = [dict(record) for record in records]
    accepted_by_id = {str(item["utteranceId"]): dict(item) for item in accepted}
    contexts = []
    for item in flagged:
        identifier = str(item["utteranceId"])
        try:
            index = next(index for index, record in enumerate(ordered) if str(record.get("utteranceId")) == identifier)
        except StopIteration:
            raise ValueError("repair source identity is invalid")
        context = dict(accepted_by_id[identifier])
        context["neighbors"] = [dict(record) for record in ordered[max(0, index - 2):index + 3] if str(record.get("utteranceId")) != identifier]
        contexts.append(context)
    return contexts


def _v1_native_by_id(output_dir: Path, artifact_stem: str) -> dict[str, str]:
    path = output_dir / (artifact_stem + ".json")
    if not path.is_file():
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    accepted = document.get("accepted", []) if isinstance(document, dict) else []
    return {str(item["utteranceId"]): str(item.get("nativeVi", "")) for item in accepted if isinstance(item, dict) and "utteranceId" in item}


def _markdown_v3(records, v1_native, v2_accepted, v3_accepted, issues) -> str:
    v2_by_id = {str(item["utteranceId"]): item for item in v2_accepted}
    v3_by_id = {str(item["utteranceId"]): item for item in v3_accepted}
    lines = [
        "# Multimodal Vietnamese A/B v3",
        "",
        "| # | Time | Type | Resolved Chinese | V1 Native VI | V2 Native VI | V3 Native VI | QA |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for index, record in enumerate(records, start=1):
        identifier = str(record.get("utteranceId", ""))
        qa = ", ".join(str(issue["code"]) for issue in issues if str(issue.get("utteranceId")) == identifier)
        lines.append(
            "| " + " | ".join([
                str(index),
                str(record.get("start_ms", "")) + "–" + str(record.get("end_ms", "")),
                str(record.get("type", "")),
                str(record.get("resolvedZh", "")),
                str(v1_native.get(identifier, "")),
                str(v2_by_id.get(identifier, {}).get("nativeVi", "")),
                str(v3_by_id.get(identifier, {}).get("nativeVi", "")),
                qa,
            ]) + "|"
        )
    return "\n".join(lines) + "\n"


def rejudge_completed_run(output_dir: Path, *, request: Callable[[dict[str, object]], object], artifact_stem: str = "multimodal_vi_ab_25s_v2", output_artifact_stem: str = "multimodal_vi_ab_25s_v3", v1_artifact_stem: str = "multimodal_vi_ab_25s") -> list[Path]:
    """Rejudge cached localization and repair only provider-flagged rows once."""
    state_path = output_dir / "cp2_run_state.json"
    artifact_path = output_dir / (artifact_stem + ".json")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    source_artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if state.get("status") != "complete" or not isinstance(source_artifact, dict):
        raise ValueError("rejudge is unavailable")
    records = source_artifact.get("records")
    accepted = source_artifact.get("accepted")
    if not isinstance(records, list) or not isinstance(accepted, list):
        raise ValueError("rejudge artifact is invalid")
    resolved = [item for item in accepted if isinstance(item, dict)]
    memory = dict(source_artifact.get("scene_memory", {}))
    memory["terminologyEvidence"] = build_terminology_evidence(records)
    observer = object.__new__(_RunObserver)
    observer.output_dir, observer.state, observer.phase = output_dir, state, "qa"
    observer.state.update({"status": "running", "phase": "qa", "finishedAt": None, "lastError": None})
    observer._persist()
    try:
        qa_document = observer.call("qa", request, {
            "mode": "qa",
            "utterances": [{key: item.get(key, "") for key in ("utteranceId", "resolvedZh", "semanticVi", "nativeVi", "source_confidence", "source_flags", "neighbors")} for item in resolved],
            "scene_memory": memory,
            "terminology_evidence": memory["terminologyEvidence"],
            "term_check_guidance": [
                "Compare source term meaning in scene context with the Vietnamese term, not grammar alone.",
                "A fluent line can still be TERM_MISTRANSLATION when context supports a non-name sense.",
            ],
        })
        provider_issues = validate_qa_document(qa_document, resolved)
        observer.stage("qa", provider_issues)
        issues = deterministic_issues(accepted, scene_memory=memory) + provider_issues
        repair_issues = _serious_repairable_issues(issues)
        flagged_ids = {str(issue["utteranceId"]) for issue in repair_issues}
        flagged = [item for item in accepted if str(item.get("utteranceId")) in flagged_ids]
        repaired = [dict(item) for item in accepted]
        if flagged:
            repair_document = observer.call("repair", request, {
                "mode": "repair",
                "utterances": _repair_context(records, accepted, flagged),
                "qa_flags": [issue for issue in repair_issues if str(issue.get("utteranceId")) in flagged_ids],
                "scene_memory": memory,
                "terminology_evidence": memory["terminologyEvidence"],
            })
            translations = repair_document.get("translations") if isinstance(repair_document, dict) else None
            expected = {str(item["utteranceId"]) for item in flagged}
            if not isinstance(translations, list) or {str(item.get("utteranceId")) for item in translations if isinstance(item, dict)} != expected:
                raise ValueError("repair response identity is invalid")
            observer.stage("repair", translations)
            repairs = {str(item["utteranceId"]): item for item in translations if isinstance(item, dict)}
            repaired = [apply_one_repair(item, repairs[str(item["utteranceId"])]) if str(item.get("utteranceId")) in repairs else item for item in repaired]
        final_issues = deterministic_issues(repaired, scene_memory=memory)
        v3_artifact = json.loads(json.dumps(source_artifact, ensure_ascii=False))
        v3_artifact["scene_memory"] = memory
        v3_artifact["accepted"] = repaired
        v3_artifact["qa_history"] = list(source_artifact.get("qa", [])) + issues
        v3_artifact["qa"] = final_issues
        v3_artifact["coverage"] = _coverage(records, repaired, source_artifact.get("review_required", []))
        v3_artifact["artifactVersion"] = "v3"
        v3_artifact["comparison"] = {
            "v1NativeVi": _v1_native_by_id(output_dir, v1_artifact_stem),
            "v2Accepted": accepted,
        }
        v3_json = _write(output_dir / (output_artifact_stem + ".json"), v3_artifact)
        v3_md = output_dir / (output_artifact_stem + ".md")
        temporary = v3_md.with_name("." + v3_md.name + ".tmp")
        temporary.write_text(_markdown_v3(records, _v1_native_by_id(output_dir, v1_artifact_stem), accepted, repaired, final_issues), encoding="utf-8")
        os.replace(temporary, v3_md)
        observer.artifact(v3_json)
        observer.artifact(v3_md)
        observer.complete()
        return [v3_json, v3_md]
    except Exception as error:
        observer.fail(error)
        raise


def repair_completed_run(output_dir: Path, *, request: Callable[[dict[str, object]], object], artifact_stem: str = "multimodal_vi_ab_25s") -> None:
    """Apply at most one post-QA repair batch without rerunning prior calls."""
    state_path = output_dir / "cp2_run_state.json"
    artifact_path = output_dir / (artifact_stem + ".json")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if state.get("status") != "complete" or any(call.get("operation") == "repair" for call in state.get("calls", [])):
        raise ValueError("repair is unavailable")
    records = artifact.get("records")
    if not isinstance(records, list):
        raise ValueError("repair artifact is missing source rows")
    memory = dict(artifact.get("scene_memory", {}))
    memory["terminologyEvidence"] = build_terminology_evidence(records)
    current_issues = deterministic_issues(artifact["accepted"], scene_memory=memory)
    repair_issues = _repairable_issues(current_issues + list(artifact.get("qa", [])))
    flagged_ids = {str(issue["utteranceId"]) for issue in repair_issues}
    flagged = [item for item in artifact["accepted"] if str(item.get("utteranceId")) in flagged_ids]
    if not flagged:
        return
    previous_accepted = [dict(item) for item in artifact["accepted"]]
    is_v3 = artifact.get("artifactVersion") == "v3" or artifact_stem.endswith("_v3")
    observer = object.__new__(_RunObserver)
    observer.output_dir, observer.state, observer.phase = output_dir, state, "repair"
    observer.state.update({"status": "running", "phase": "repair", "finishedAt": None})
    observer._persist()
    try:
        document = observer.call("repair", request, {"mode": "repair", "utterances": _repair_context(records, artifact["accepted"], flagged), "qa_flags": [issue for issue in repair_issues if str(issue.get("utteranceId")) in flagged_ids], "scene_memory": memory, "terminology_evidence": memory["terminologyEvidence"]})
        translations = document.get("translations") if isinstance(document, dict) else None
        if not isinstance(translations, list) or {item.get("utteranceId") for item in translations if isinstance(item, dict)} != {item["utteranceId"] for item in flagged}:
            raise ValueError("repair response identity is invalid")
        observer.stage("repair", translations)
        repairs = {item["utteranceId"]: item for item in translations if isinstance(item, dict)}
        accepted = [apply_one_repair(item, repairs[item["utteranceId"]]) if item["utteranceId"] in repairs else item for item in artifact["accepted"]]
        artifact["qa_history"] = current_issues + list(artifact.get("qa", []))
        artifact["qa"] = deterministic_issues(accepted, scene_memory=memory)
        artifact["accepted"] = accepted
        artifact["scene_memory"] = memory
        _write(artifact_path, artifact)
        md_path = output_dir / (artifact_stem + ".md")
        temporary = md_path.with_name("." + md_path.name + ".tmp")
        if is_v3:
            comparison = artifact.get("comparison", {})
            v1_native = comparison.get("v1NativeVi", {}) if isinstance(comparison, dict) and isinstance(comparison.get("v1NativeVi", {}), dict) else {}
            v2_accepted = comparison.get("v2Accepted", previous_accepted) if isinstance(comparison, dict) else previous_accepted
            temporary.write_text(_markdown_v3(records, v1_native, v2_accepted, accepted, artifact["qa"]), encoding="utf-8")
        else:
            temporary.write_text(_markdown(records, artifact["scene_memory"], accepted, artifact["review_required"], artifact["qa"]), encoding="utf-8")
        os.replace(temporary, md_path)
        observer.complete()
    except Exception as error:
        observer.fail(error)
        raise
