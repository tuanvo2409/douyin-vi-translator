# Page 1 Source-Lock Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Repository policy overrides generic subagent test/commit steps: Luna writes bounded code/tests only; parent runs all tests, QA, renders, and commits.

**Goal:** Make new Page 1 raws use the proven evidence-led `source_locked` behavior through the canonical Translator path, then optimize valid cache hits and remove only proven dead code.

**Architecture:** Extend the existing source OCR decision and evidence cache rather than transplanting the row-specific runtime scripts. Keep Page 1 source adjudication upstream of existing DeepSeek → delivery → CapCut BV075 → timeline → mask/render stages. Use four prior renders as immutable regression references, not as a generalized production cache.

**Tech Stack:** Python, unittest, local Whisper/RapidOCR, optional SenseVoice, DeepSeek, CapCut, FFmpeg; no new dependency.

**Spec:** `docs/superpowers/specs/2026-09-29-page1-source-lock-parity.md` (addendum to `docs/superpowers/specs/2026-09-28-round2-audit-remediation.md`).

## Global Constraints

- Translator-only production changes; retain six-ID canonical contract and DeepSeek-only text localization.
- Page 1 voice `BV075_streaming`; no silent fallback or audio reused for changed text.
- Preserve every raw, four accepted outputs, row 5, dirty user changes, and the two untracked `source_locked_timing` prototypes.
- Parent alone runs checks, real providers, render/QA and commits; worker ownership is disjoint; at most one heavy media process.
- No production fixture ID/text/timestamp, no broad mask, no guessed source, no product-acceptance claim.

## Review Focus

1. Two identical OCR strings from one duplicated/decode-failed frame: Task 1 test must reject.
2. Story text, watermark, or clipped dialogue in the ROI: Task 1 test must reject without broadening the mask.
3. Traditional/simplified or digit disagreement: Task 1 test must not silently normalize away a material conflict.
4. Valid-looking cache from different raw or decision version: Task 2 test must miss before localization.
5. Same cue identity but revised Vietnamese delivery text: Task 4 test must synthesize new BV075 audio.

---

### Task 0: Freeze baseline and caller map (parent, read-only)

**Files:** No production edits; record evidence in this plan's execution notes or a runtime report outside Git.

**Interfaces:** Produces four raw SHA-256/output SHA-256 pairs, source-lock decision counts, current branch/HEAD/dirty paths, and `rg` caller map for `decide_source`, `_cached_source_evidence`, `source_locked_timing`, old providers and cache helpers.

- [ ] Read both `AGENTS.md`, the spec, `PAGE1_FIVE_20260928/report.json`, old `lock_sources.py`/`run_one_locked.py`, and the four `source_locked.json` artifacts; record the exact evidence gaps, especially old `ASR_ONLY`/manual overrides.
- [ ] Run read-only `git status --short`, `git rev-parse HEAD`, `rg -n` caller searches, and hashes; confirm old outputs and row 5 will not be written.
- [ ] Gate: no code edit until baseline and ownership are documented. If a relevant file has overlapping user changes, integrate deliberately rather than reset/stage broadly.

### Task 1: Make repeated dialogue OCR a conservative source decision

**Files:** Modify `local-worker/source_ocr_verifier.py`; modify `local-worker/tests/test_source_ocr_verifier.py`. Inspect `local-worker/auto_roi.py` and `local-worker/modern_p1c_adapter.py` read-only for detection schema.

**Interfaces:** Keep `decide_source(asr_text: str, readings: Sequence[Mapping[str, object]], *, second_asr: str | None = None) -> dict[str, object]`; extend `scan_cue_ocr(video_path, cues, roi, *, detections)` so readings carry distinct `timestampMs`, source-frame `bbox`, and evidence-derived `category`. Derive `DIALOGUE_SUBTITLE` from the recurring/changing lower-frame ROI pattern plus in-band position; do not copy RapidOCR's default `UNKNOWN` into an accepted decision. Accepted `OCR_REPEAT` returns `resolvedZh`, `ocrZh`, `decision`, `needsReview=False`; otherwise existing review contract remains.

- [ ] Write failing tests: former Page 1 disputed cue with two complete in-cue `DIALOGUE_SUBTITLE` readings at >=0.90 accepts the matching OCR; duplicate timestamp/frame, missing bbox/type, `STORY_TEXT`, unresolved `UNKNOWN`, low confidence, conflicting text, digit change, clipped line, and cross-cue sample each return review. Preserve the existing test where repeated OCR *without* typed/timestamped evidence cannot overrule ASR. Existing ASR/OCR and second-ASR tests remain green.
- [ ] Parent runs `python -B -m unittest tests.test_source_ocr_verifier -v` from `local-worker`; verify new cases fail for the intended reason (RED).
- [ ] Implement the smallest read/decision change. Validate crop-to-frame bbox and timestamps, preserve actual reading evidence, and avoid expanding OCR acceptance to visual/story text.
- [ ] Parent reruns the focused suite (GREEN); check one known old dispute gives `OCR_REPEAT`, while old ASR-only/manual cues are still not auto-accepted.

### Task 2: Bind accepted decisions to source cache provenance

**Files:** Modify `local-worker/modern_p1c_adapter.py` around `_source_provenance`, `_cached_source_evidence`, `_local_asr_source`; modify `local-worker/tests/test_modern_source_ocr_route.py` and `local-worker/tests/test_modern_p1c_adapter.py`. Avoid a new cache module unless the existing functions cannot express validation safely.

**Interfaces:** `source_evidence.json` remains the canonical output. Advance its schema/stage version; each cue records underlying OCR readings and decision. `_cached_source_evidence(...) -> tuple[canonical_records, source_cues] | None` returns only a complete, ordered, provenance-matched, source-resolved artifact; `OCR_REPEAT` is valid only with Task 1 evidence. A rejected cache returns `None`, not a partial source.

- [ ] Write failing tests for valid `OCR_REPEAT` warm hit skipping transcription/OCR, changed raw SHA, duration, model/compute, ROI/OCR config, decision version, cue time/ID, missing reading, unresolved cue, and old `ASR_ONLY`/manual lock. Assert rejected evidence never reaches localization.
- [ ] Parent runs focused `python -B -m unittest tests.test_modern_source_ocr_route tests.test_modern_p1c_adapter -v` (RED).
- [ ] Wire Task 1 readings into cue/record evidence, broaden only the verified decision allowlist, and validate it again on cache load; retain atomic JSON writes and review-required failure output.
- [ ] Parent reruns focused tests (GREEN), confirming valid cache skips expensive source work while mask geometry still derives from that raw's detections.

### Task 3: Prove canonical Page 1 delivery parity, without the old runner

**Files:** Modify only if tests reveal a contract gap: `local-worker/modern_p1c_adapter.py`, `local-worker/p1c_processing.py`; tests `local-worker/tests/test_modern_deepseek_route.py`, `local-worker/tests/test_p1c_processing.py`, `local-worker/tests/test_source_locked_timing.py` (do not edit the preserved prototype test; add an owned regression test instead if needed).

**Interfaces:** `build_canonical_runtime_config(...)` retains `sourceOcrVerify=True`, DeepSeek, and the Page 1 profile; `build_modern_stage_bundle(...)["source_stage"]` yields only resolved records. Existing localization, TTS, delivery, timeline, subtitles and mask hooks consume canonical cue IDs and the same accepted timeline.

- [ ] Write an integration-style fake-provider test: `OCR_REPEAT` resolved Chinese enters DeepSeek, faithful Vietnamese remains separate from delivery rewrite, BV075 receives the final delivery text, and subtitle/voice/picture timing uses one transform. A review-required cue stops before provider calls. Include mask geometry per raw and two-line subtitle bound assertions.
- [ ] Parent runs focused suites (RED only for a demonstrated gap); if they already pass, record parity and make no production edit.
- [ ] Fix only the failing shared function/caller, never import or monkeypatch `PAGE1_FIVE_20260928` scripts; keep old four render files untouched.
- [ ] Parent reruns focused suites (GREEN). Do not call old `ASR_ONLY` source locks verified simply because their previous videos rendered.

### Task 4: Stage-specific cache and cleanup, only after parity

**Files:** Inspect first, then limit edits to the owning existing helpers in `local-worker/modern_p1c_adapter.py`, `local-worker/cp2_deepseek_adapter.py`, `local-worker/capcut_tts_api/`, and focused tests (`test_modern_deepseek_route.py`, `test_capcut_tts_provider_input.py`, `test_tts_provider_config.py`). Touch `local-worker/README.md` only to document proven entry points/cache semantics. No broad deletions.

**Interfaces:** Source cache key is Task 2 provenance; localization cache identity includes locked Chinese/cue IDs plus DeepSeek model/prompt/config; CapCut audio identity includes exact delivery text, `BV075_streaming`, provider settings and audio validity. On a changed key, recompute only that stage and its dependents.

- [ ] Inventory all cache readers/writers and production entry points via `rg`; mark each suspected dead branch with a caller result before deletion. If an existing helper already supplies an identity check, reuse it.
- [ ] Write failing tests for cold/warm call counts, changed Chinese invalidating localization/TTS, changed Vietnamese invalidating only TTS/downstream, corrupt/missing audio miss, and same text+voice valid hit. Tests must use fakes, not live API/media.
- [ ] Parent runs focused tests (RED), implement minimum cache-key/validation correction, then reruns (GREEN). If no defect is found, do not add a new cache layer.
- [ ] Remove only truly unreachable duplicate code with passing focused tests and `rg` caller proof; leave historic provider modules needed by supported entry points untouched. Do not touch the two preserved prototype files or stage unrelated dirty files.

### Task 5: Parent integration, measurement and handoff

**Files:** No required production changes; diagnostics/output under a new `C:\IDE_WORKSPACE\DUBVI_RUNTIME\P1_SOURCE_LOCK_PARITY_*` directory. Update plan checkboxes/evidence only.

**Interfaces:** A result is `technical_pass`, `source_review_required`, `mask_review_required`, or provider/render failure; subjective quality remains Tuấn's decision.

- [ ] Parent runs focused tests once after integration, then `python -B -m unittest discover -s tests -p 'test_*.py' -q`, `python -B -m compileall -q .`, and `git diff --check` from `local-worker`/repo as appropriate. Fix only actual failures and rerun affected checks. Verify baseline SHA-256 pairs unchanged.
- [ ] Pick one eligible *new* Page 1 raw from the provided Page 1 list, verify its local raw SHA, and run a sequential non-publishing 10-second disposable smoke through the canonical path. Do not select a replacement without recording why. If source/mask review is required, preserve evidence and stop before DeepSeek/CapCut/render.
- [ ] If source clears, check output with `ffprobe` for H.264/AAC/duration, full FFmpeg decode, no voice overlap/out-of-window events, subtitle/timeline equality, <=2 lines, and tight mask bounds preserving `STORY_TEXT`; record actual cold wall time. Repeat same raw/config for warm timing, proving ASR/OCR call omission. Do not rerender the four reference videos.
- [ ] Parent reviews exact diff and unrelated dirty files; commit only task-owned validated changes if attribution is clear. Report changed files, tests, cold/warm timings, source/mask warnings and artifact link. Use `READY FOR TUAN REVIEW` only for a technically verified new artifact; never claim Tuấn approved its content.

## Execution ordering

Tasks 0→1→2→3 are sequential because they share the source contract. Task 4 follows parity, not ahead of it. Parent may assign Luna a disjoint read-only caller/cache inventory while doing Task 1, but never a concurrent edit of `modern_p1c_adapter.py`; workers do not run tests, render, QA or commit. Page 2 and row 5 are separate follow-ups, not shortcuts to close this plan.
