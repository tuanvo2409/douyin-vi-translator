# Page 1 source-lock parity — 2026-09-29

## Decision and scope

The four Page 1 renders in `C:\IDE_WORKSPACE\DUBVI_RUNTIME\PAGE1_FIVE_20260928` are the immutable user-liked reference, not evidence that the canonical pipeline already works. Restore their generalizable source-to-delivery behavior in the canonical Translator path before optimizing or deleting anything. This addendum supersedes the Page 1 rule in `2026-09-28-round2-audit-remediation.md` that *every* ASR/OCR disagreement must stop; other round-2 requirements, including Page 2, remain in force. No Control Plane or Reup contract change is planned.

The old `lock_sources.py` and `run_one_locked.py` are evidence/prototypes, not production dependencies: they contain row/position-specific overrides and monkeypatches. The four prior `source_locked.json` artifacts may seed read-only regression cases but may not become automatic verified cache hits. Old `ASR_ONLY` or manually reconciled cues are not implicitly verified under the new rule.

## Source truth contract

- Preserve separate Chinese ASR, timestamped OCR readings, decision, resolved Chinese, confidence/uncertainty, source cue identity, and provenance. Canonical source, faithful Vietnamese translation, and optional humorous delivery are separate fields/stages.
- `ASR_OCR_AGREE` remains accepted when independently sampled OCR agrees with ASR. `OCR_SECOND_ASR_AGREE` remains accepted under its existing safeguards.
- Add `OCR_REPEAT` only when two distinct in-cue frames each produce the same complete Chinese dialogue reading at confidence >= 0.90, with valid timestamp, source-ROI bounds, and `DIALOGUE_SUBTITLE` classification. The role may be derived from recurring, changing lower-frame subtitle evidence and in-band geometry; raw RapidOCR's default `UNKNOWN` label alone is insufficient. Neither `STORY_TEXT`, `WATERMARK`, unresolved `UNKNOWN`, a duplicated single frame, a clipped/partial line, nor contradictory visible OCR may pass. Preserve a digit or meaning-changing punctuation conflict as review, not silent OCR promotion. A production cue may use OCR to correct erroneous ASR only under these conditions; repeated OCR is not a claim that the spoken words independently corroborate the text.
- OCR crossing ASR cue boundaries or split across frames needs explicit alignment evidence; if current evidence cannot establish it, mark `SOURCE_REVIEW_REQUIRED` rather than guessing/merging. No stable ROI, missing/unreadable OCR, ambiguous source, `ASR_ONLY`, or essential visual-only text stays review-required. Manual reconciliation, where needed, must be an explicit traceable input tied to raw/cue/reviewer/evidence, never a production hardcode or silent pass.
- A verified source cache must bind raw SHA-256, duration, ASR model/compute identity, OCR/ROI sampling configuration, decision algorithm version, ordered cue IDs/times, underlying readings and accepted decision. A mismatch or incomplete decision invalidates that stage. A valid hit skips ASR/OCR; a miss must not overwrite accepted renders.

## Delivery and safety

- Keep the canonical Page 1 profile: DeepSeek localization, new CapCut `BV075_streaming` audio for new delivery text, source-locked retiming, one accepted timeline for picture/voice/subtitles, at most two Vietnamese subtitle lines, and per-raw tight fixed mask that preserves story text. No Gemini/Groq fallback and no voice reuse when text differs.
- One heavy media task at a time on this laptop. Preserve raw files, four reference outputs, and the withheld fifth video. No real provider calls, downloads, or renders in automated tests. Runtime smoke is disposable, sequential and non-publishing.
- The user alone accepts subjective quality. Technical checks can prove source provenance, H.264/AAC decode, timing, mask bounds, and cache behavior, not that a new video is aesthetically approved.

## Acceptance

1. A fixture based on a former Page 1 ASR/OCR dispute yields `OCR_REPEAT` with full evidence; removing one frame or changing its text/type/timestamp returns review-required. The old four renders and raws remain byte-identical.
2. The canonical Page 1 entry point—not a monkeypatched runner—passes the verified Chinese source to DeepSeek, then fresh BV075 delivery and shared timeline. It does not localize review-required cues.
3. A valid warm source hit omits ASR/OCR. Altered raw/config/ROI/timing/decision version or partial evidence misses. Invalid downstream localization/TTS cache invalidates only that stage; changed text never selects stale audio.
4. Parent-owned automated tests and one sequential disposable smoke on an eligible new Page 1 raw pass technical checks; cold/warm elapsed times are reported separately. If no raw clears source/mask gates, report the blocker rather than weakening a gate or claiming `READY FOR TUAN REVIEW`.
5. Dead-code/cache cleanup is limited to callers proven unused and separately tested. The four accepted outputs, row 5, and Page 2 remain untouched.
