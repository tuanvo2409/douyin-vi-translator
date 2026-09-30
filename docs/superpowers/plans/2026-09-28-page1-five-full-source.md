# Page 1 Five Full-Source Renders Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. In this workspace Luna acts as parent; `dubvi_worker` delegates, if any, obey both `AGENTS.md` files and never run tests or product QA.

**Goal:** Deliver evidence-grounded, full-source Vietnamese renders of the first five Page 1 CSV videos with fresh CapCut voice and safe, fixed per-video subtitle boxes.

**Architecture:** Reuse the existing local ASR, OCR verifier, DeepSeek localization, CapCut provider and adaptive renderer. Lock each cue's source evidence before semantic translation and separate the humor delivery layer; drive media and subtitle output from one accepted timeline. This is primarily an artifact run, with only generic code fixes when a reproducible shared defect blocks it.

**Tech Stack:** Python, faster-whisper-small CPU/int8, RapidOCR, existing DeepSeek adapter, CapCut `BV075_streaming`, FFmpeg/ffprobe, pytest.

**Spec:** `docs/superpowers/specs/2026-09-28-page1-five-full-source.md`

## Global Constraints

- CSV order and the five IDs in the spec are authoritative; no substitute raw.
- Full source; retain all raw and previous render files. Runtime outputs under `C:\IDE_WORKSPACE\DUBVI_RUNTIME` only.
- Fresh `BV075_streaming` for changed delivery text; no silent voice fallback or stale audio.
- Faithful source/localization and humorous delivery are separate, traceable fields.
- Fixed rounded white dialogue box per video, Vietnamese maximum two visible lines; preserve `STORY_TEXT` and `UNKNOWN` unless evidence justifies cleanup.
- One accepted source→output timeline; one heavy media task at a time; reuse valid caches and avoid unnecessary API calls.
- Luna parent owns tests, integration, commits if any, final automated verification, and handoff. The user alone judges quality.

## Review Focus

1. Same video ID but changed raw bytes: invalidate source, timing, translation and audio caches; Task 1 pins raw hash identity.
2. ASR homophone or OCR line shared by adjacent cues: do not blindly replace by positional OCR; Task 2 pins ambiguous and split-line examples.
3. Opening story title inside an otherwise valid dialogue band: fixed box must not hide it; Task 4 pins row-2 failure.
4. Moving story text and no stable safe dialogue band: fail closed with a concrete blocker, not a broad box; Task 4 pins row-5 condition.
5. Long new TTS utterance under adaptive retiming: voice, picture and subtitle must still share one map and never overlap/outlive output; Task 5 pins timeline invariants.

---

### Task 1: Lock input identity and inventory reusable artifacts

**Files:**
- Read: `C:\IDE_WORKSPACE\DUBVI_COMPARE\AGENTS.md`, `AGENTS.md`, `local-worker/modern_p1c_adapter.py`, `local-worker/dubvi_worker.py`.
- Create: `C:\IDE_WORKSPACE\DUBVI_RUNTIME\PAGE1_FIVE_20260928\manifest.json`.
- Modify only if a shared cache-key defect is proven: `local-worker/dubvi_worker.py`, `local-worker/tests/test_local_model_resolution.py`.

**Interfaces:** Consumes the Page 1 CSV, the five raw paths and existing runtime artifacts from the spec. Produces a manifest keyed by `(CSV row, Douyin ID, raw SHA-256)` with raw path, ffprobe duration/dimensions, valid cache paths and invalidation reasons; later tasks use this manifest, never array-position-only joins.

- [ ] Confirm branch/HEAD/dirty files and CSV rows 1–5, raw file existence, IDs, SHA-256 and ffprobe metadata; abort an identity mismatch rather than substituting media.
- [ ] Compare existing ASR/OCR/alignment/translation/audio artifacts to raw hash, source text, cue IDs and timings. Record `REUSE` or explicit invalidation reason per stage. In particular invalidate row-3 stale cue alignment.
- [ ] Verify `faster-whisper-small` resolves offline in the actual worker environment; the old `tiny` runner config is not evidence of `small` availability. If `_whisper_model` singleton reuses a different model, first add a focused failing test, then make the smallest generic model-keyed fix, then parent runs that focused test.
- [ ] Write and validate `manifest.json`; ensure all five IDs are present exactly once and no old output is targeted for overwrite.

### Task 2: Reconcile Chinese SRT with visual evidence, cue by cue

**Files:**
- Read: `local-worker/source_ocr_verifier.py`, `local-worker/modern_p1c_adapter.py`, `local-worker/tests/test_source_ocr_verifier.py`.
- Reuse: `C:\IDE_WORKSPACE\DUBVI_RUNTIME\PAGE1_ASR_SMALL_20260928\{03,04,05}_<id>\source_zh_small.srt/json`, existing row-3/4 `source_ocr_crosscheck.json`.
- Create: `C:\IDE_WORKSPACE\DUBVI_RUNTIME\PAGE1_FIVE_20260928\<row>_<id>\source_locked.json` and `source_zh_reviewed.srt` per unresolved row.
- Modify only for a generic reproducible defect: `local-worker/source_ocr_verifier.py`, `local-worker/tests/test_source_ocr_verifier.py`.

**Interfaces:** `source_locked.json` records `cue_id`, source start/end milliseconds, ASR text, timestamp-matched OCR text/role, chosen Chinese text, evidence/confidence, and `review_required` reason. Task 3 consumes only resolved dialogue cues; unresolved/source-visual-only material remains explicit.

- [ ] Parse SRT with ascending timestamps in raw duration. Reuse small-ASR artifacts only if manifest identity matches; otherwise rerun local small CPU/int8 for that raw, sequentially. Do not accept high ASR logprob as proof of correct characters.
- [ ] Reuse row-3/4 OCR cross-checks and inspect low-similarity or conflicting cues against timestamped frames/audio. Account for OCR lines spanning adjacent ASR cues and timing lag; classify `DIALOGUE_SUBTITLE`, `STORY_TEXT`, `WATERMARK`, `UNKNOWN` before choosing text.
- [ ] For row 5, run bounded/sampled OCR across the frame, not one presumed lower subtitle ROI; map moving visual text separately from spoken dialogue. Keep non-dialogue story text in source evidence but do not synthesize speech from it.
- [ ] If generic reconciliation logic must change, write a focused failing test for adjacent split OCR, homophone conflict and `UNKNOWN` preservation; parent runs test failing, implements minimal fix, then runs it passing. Otherwise leave production untouched.
- [ ] Export reviewed SRT and `source_locked.json`; assert every chosen text has specific evidence and every unresolved cue is marked rather than guessed. A material unresolved speech cue blocks that row's translation/render.

### Task 3: Translate faithfully, then write bounded humor delivery

**Files:**
- Read: `local-worker/cp2_deepseek_adapter.py`, `local-worker/vn_localizer.py`, `C:\IDE_WORKSPACE\DUBVI_RUNTIME\P1_HUMOR_AUDITION_20260927\audition.json`, the existing runtime `render_page1_five_humor.py`.
- Create: `C:\IDE_WORKSPACE\DUBVI_RUNTIME\PAGE1_FIVE_20260928\<row>_<id>\localization.json` per row.
- Modify only if a proven generic bug blocks this route: existing DeepSeek/localization module and its corresponding focused test; never insert five-video IDs/text in production.

**Interfaces:** For each canonical cue ID, `localization.json` keeps Chinese source and provenance, faithful Vietnamese, separate concise delivery Vietnamese, unchanged source interval, and reason for any comic rewrite. Task 5 synthesizes exact delivery text from this artifact.

- [ ] Reuse valid existing translations where source cue ID/text/timing hash matches. Send only new/changed resolved dialogue cues to the configured DeepSeek route; no Gemini/Groq fallback unless explicitly configured and disclosed.
- [ ] Preserve literal meaning/relationships/instructions in faithful Vietnamese. Make an occasional scene-grounded twist only in delivery text; do not inject new facts or make every line a joke. Check length against available scene/timing before TTS.
- [ ] Assert one-to-one canonical cue IDs, no missing translation, no stale text reuse, and no delivery text silently written back into canonical Vietnamese. Log source ambiguities and editorial changes for handoff.
- [ ] For a reproducible generic routing defect, add its focused failing test, fix only the shared caller/function, and have Luna run the focused test; otherwise no production edit.

### Task 4: Prove fixed mask geometry per source without deleting story text

**Files:**
- Read: `local-worker/modern_p1c_adapter.py` (`_page1_mask_geometry`), `local-worker/source_ocr_verifier.py`, `local-worker/localized_masking.py`, `local-worker/subtitle_layout.py`.
- Create: `C:\IDE_WORKSPACE\DUBVI_RUNTIME\PAGE1_FIVE_20260928\<row>_<id>\mask_evidence.json` and sampled frame contact sheet per row.
- Modify only for a generic fix: appropriate mask/layout module and `local-worker/tests/test_modern_source_ocr_route.py` or `test_localized_masking.py`.

**Interfaces:** `mask_evidence.json` contains source dimensions, candidate fixed rectangle, evidence timestamps, covered dialogue detections, intersected non-dialogue detections, and an `accepted` Boolean/reason. Task 5 may render only accepted geometry.

- [ ] Sample first, middle and late dialogue frames plus every detected story-title transition. Estimate the tight fixed dialogue subtitle envelope separately for each raw; apply actual source→output scaling. Keep rounded box visible throughout as requested; Vietnamese text within, maximum two lines.
- [ ] Reject a candidate rectangle if it intersects `STORY_TEXT`, unless the overlapping text is preserved by an evidence-backed Vietnamese visual replacement cue kept separate from spoken dialogue. For row 2, translate the opening title as a non-spoken visual cue and fit it with opening dialogue within the 2-line limit; record its source role/timing and translation. Do not widen the box just to cover all Chinese text.
- [ ] If there is a generic geometry/classification defect, write focused failing tests for those two conditions, prove failure, fix shared logic once, and parent verifies passing tests. Do not hardcode IDs/timestamps/coordinates in production.
- [ ] If no fixed rectangle satisfies dialogue coverage and story preservation for a row, stop **that row** with annotated evidence and ask Tuấn to choose a changed visual contract; do not silently switch to dynamic masks or deliver a knowingly obscured story title. Other ready rows may continue.

### Task 5: Sequential full renders and parent verification

**Files:**
- Read: `local-worker/adaptive_renderer.py`, `local-worker/timeline_mapper.py`, `local-worker/tts_timing.py`, `local-worker/tests/test_adaptive_renderer.py`.
- Reuse or minimally revise runtime-only `C:\IDE_WORKSPACE\DUBVI_RUNTIME\SOURCE_REPAIR_PROBE_20260925\render_page1_five_humor.py` for manifest-driven input (never hardcode five IDs in production).
- Create: `C:\IDE_WORKSPACE\DUBVI_RUNTIME\PAGE1_FIVE_20260928\<row>_<id>\final.mp4`, validation logs, and `report.json`.

**Interfaces:** Consumes accepted `source_locked.json`, `localization.json`, `mask_evidence.json` and raw identity. Produces a final MP4 and machine-readable validation per row; no row is marked ready until all gates pass.

- [ ] Reverify row 1's existing full candidate against current spec; reuse only if all gates pass. Row 2 must be repaired because its prior opening story title is hidden. Preserve both old files regardless.
- [ ] Render eligible rows one at a time, full source, using fresh `BV075_streaming` for exact new delivery text; hard-fail a CapCut/voice mismatch or unavailable provider. Record requested text hash and returned audio hash per cue.
- [ ] Build exactly one accepted retiming map and feed video, voice, masking and Vietnamese subtitle positions through it. If a TTS clip overruns, use the existing adaptive retiming policy; never concatenate voice on an independent schedule.
- [ ] After each row, parent runs ffprobe H.264/AAC/duration checks and full FFmpeg decode; assert ordered/non-overlapping voice events, all voice/sub intervals within output, subtitle map equality/tolerance per existing contract, max two lines, no glyph/box overflow and accepted mask evidence. Add one focused regression test for any newly fixed shared timing bug, run it failing then passing.
- [ ] Run relevant focused tests once after integration, `git diff --check`, and full suite only if generic production behavior changed enough to warrant it. Stage/commit only task-owned source/tests after reviewing the dirty diff; leave runtime artifacts outside Git.
- [ ] Produce five playable absolute links and a compact report with raw name, source/output duration, processing time, source/retiming/mask warnings. Mark `READY FOR TUAN REVIEW` only when five final files pass deterministic gates; otherwise report completed rows plus exact blocked rows and decisions needed, without claiming five are ready.
