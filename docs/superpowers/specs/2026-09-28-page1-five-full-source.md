# Page 1 — five full-source Vietnamese renders

The user wants the first five rows, in CSV order, of `C:\Users\idebox\Downloads\DANH SÁCH VIDEO RAW - Page_1_Chuan_90_Video_Drama_KTX.csv`. Preserve raw files and all prior renders. Render the **full source** of each row, not a 60-second cut. Final handoff is five playable video links with raw name, source/output duration, processing time and warnings; status `READY FOR TUAN REVIEW`, not a claim of user approval.

## Editorial and media contract

- Ground Chinese source understanding in audio ASR plus frame OCR. Keep unresolved words marked; do not invent dialogue, relationships, instructions or timing. A timed SRT is an intermediate, not proof of semantic accuracy.
- Keep faithful Vietnamese translation separate from a delivery-specific, light situational-humor edit. Humor is occasional, concise and appropriate to the scene; use `C:\IDE_WORKSPACE\DUBVI_RUNTIME\P1_HUMOR_AUDITION_20260927\humor_preview.mp4` and `audition.json` as style references, not source evidence.
- Generate **fresh** CapCut `BV075_streaming` audio for new delivery text. Never silently use another voice or stale text/audio cache.
- Keep one accepted source→output timeline for picture, TTS and subtitle. No independent voice concatenation or cumulative scene drift. Vietnamese subtitle maximum two visible lines inside a fixed white rounded box that tightly covers the Chinese dialogue-subtitle band, positioned separately per video. Preserve `STORY_TEXT`; do not broadly erase `UNKNOWN`.
- Run only one heavy media task at a time on the laptop. Reuse source/OCR/translation artifacts only when identity, raw hash, text and timing are still valid. Do not call APIs without a need. Save artifacts under `C:\IDE_WORKSPACE\DUBVI_RUNTIME`, not over raw/old renders.
- Before delivery, require H.264/AAC, full FFmpeg decode, no voice overlap, all voice/sub events within output, subtitle timing on the accepted timeline, and a check that the fixed box does not hide story text.

## Source identities and existing evidence

| CSV row | Douyin ID | Raw | Current state |
|---|---|---|---|
| 1 | `7219647632271101221` | `C:\Users\idebox\Downloads\SaveTik.io_7219647632271101221.mp4` | Full candidate `C:\IDE_WORKSPACE\DUBVI_RUNTIME\PAGE1_HUMOR_FIRST5_20260927\01_7219647632271101221\full_humor.mp4`; reverify before reuse. |
| 2 | `7605550910287298502` | `C:\Users\idebox\Downloads\DUBVI_auto_test_7605550910287298502.mp4` | Full candidate `C:\IDE_WORKSPACE\DUBVI_RUNTIME\PAGE1_HUMOR_ROW2_MASKED_20260928\02_7605550910287298502\translated.mp4`; its fixed box hides the opening story title, so not deliverable unchanged. |
| 3 | `7344246568251297087` | `C:\Users\idebox\Downloads\DUBVI_auto_raw_7344246568251297087.mp4` | `PAGE1_ASR_SMALL_20260928\03_7344246568251297087\source_zh_small.srt/json` and `source_ocr_crosscheck.json`; old cue-alignment cache conflicts with new ASR. |
| 4 | `7539091038532144444` | `C:\Users\idebox\Downloads\DUBVI_auto_raw_7539091038532144444.mp4` | `PAGE1_ASR_SMALL_20260928\04_7539091038532144444\source_zh_small.srt/json` and `source_ocr_crosscheck.json`; prior mask-source reconciliation failed. |
| 5 | `6972145211812105487` | `C:\Users\idebox\Downloads\DUBVI_auto_raw_6972145211812105487.mp4` | `PAGE1_ASR_SMALL_20260928\05_6972145211812105487\source_zh_small.srt/json`; moving story text makes a safe fixed dialogue box uncertain. |

All relative `PAGE1_ASR_SMALL_20260928` paths above are under `C:\IDE_WORKSPACE\DUBVI_RUNTIME`. Check files and source hashes at execution time; this table is context, not permission to trust caches blindly. For rows 3–4 the OCR cross-check reports many string disagreements (47/64 and 41/63 respectively); neither ASR nor OCR is automatically authoritative. Row 5 needs visual text-role classification before any masking.

## Coordination

Luna is the parent implementer for this run. It reads workspace and repository `AGENTS.md`, owns integration, tests, render and reporting. Any `dubvi_worker` is bounded to disjoint file ownership; workers do not run tests, commit or perform product/demo QA. Do not stage or commit unrelated dirty-worktree changes. No production hardcodes for these five IDs.
