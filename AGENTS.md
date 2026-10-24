# DUBVI Multi-Agent Fast Execution Policy

## Purpose

Primary goal: minimize wall-clock execution time without weakening repository
contracts or automated verification.

Use multi-agent execution when work can be split into genuinely independent,
non-overlapping workstreams.

Default orchestration:

- Parent: Terra
- Worker role: `dubvi_worker`
- Worker model: Luna
- Maximum useful concurrent workers: 3
- Automated verification: parent only
- Final product/demo acceptance: user only

The USER is the sole final authority for subjective product quality.

---

## Parent Agent Responsibility

The parent agent owns:

- reading applicable repository instructions
- understanding current authority / branch / HEAD
- architecture and invariants
- task decomposition
- deciding whether parallelism is useful
- assigning worker ownership
- integration
- conflict resolution
- automated verification
- commits
- runtime artifact/demo creation
- consolidated final report

The parent MUST NOT delegate final product acceptance.

The parent MUST NOT claim that a video, demo, render, or product is visually
approved by the user.

After producing the requested artifact, stop at:

`READY FOR TUAN REVIEW`

unless the user explicitly asks for further implementation.

---

## Subagent Role

Use the configured:

`dubvi_worker`

for bounded parallel work.

Subagents are execution workers, not independent project owners.

A subagent MAY:

- inspect assigned code
- trace assigned call paths
- inspect runtime artifacts and logs
- implement clearly bounded production changes
- add or update regression tests when explicitly required by its implementation
- perform read-only static review
- identify contract violations
- report blockers and assumptions

A subagent MUST NOT:

- run test suites
- run regression suites
- run full decode validation
- render product demos unless explicitly owned as an implementation artifact
- inspect demo videos for product acceptance
- perform subjective visual QA
- perform subjective audio QA
- declare product PASS / FAIL
- commit
- modify files outside assigned ownership
- duplicate another worker's task
- spawn additional subagents unless the parent explicitly authorizes it
- change repository authority or architecture outside its assigned scope

---

## No Test / QA Subagents

Do NOT create:

- test agents
- QA agents
- verification agents
- demo-review agents

Subagents do not execute tests.

Subagents may WRITE or MODIFY tests when needed for the implementation they own,
but execution is centralized at the parent level.

Do not use parallelism like:

- worker implements
- second worker tests worker
- third worker reviews worker
- fourth worker tests again

That increases wall-clock time and duplicates work.

Preferred flow:

parallel implementation / audit
→ single parent integration
→ single automated verification phase
→ artifact
→ user review

---

## File Ownership Before Spawn

Before spawning any worker, the parent MUST define explicit ownership.

For every worker state:

- task
- files/modules it may modify
- files/modules it may inspect read-only
- forbidden scope

Two workers MUST NOT edit the same production file concurrently.

If two workstreams require the same file:

- assign that file to exactly one worker
- make other workers read-only for that file
- or serialize that portion after parallel work completes

Never allow competing patches against the same production ownership area.

---

## Parallelism Policy

Use subagents only when there are at least two genuinely independent
workstreams.

Prefer:

2 or 3 workers.

Do NOT spawn a worker for a tiny task that the parent can complete faster than
delegation overhead.

Good example:

Parent
├─ Worker A: implementation module A
├─ Worker B: implementation module B
└─ Worker C: read-only audit of related call paths

Bad example:

Parent
├─ Worker A: implementation
├─ Worker B: test Worker A
├─ Worker C: review Worker A
└─ Worker D: run tests again

Favor useful concurrency, not maximum agent count.

---

## Worker Assignment Format

When spawning a worker, use a compact assignment containing:

ROLE:
`dubvi_worker`

MODE:
`WRITE` or `READ-ONLY`

TASK:
one clearly bounded goal

OWNERSHIP:
exact files/modules the worker may modify

READ-ONLY SCOPE:
optional files/modules it may inspect

MUST PRESERVE:
important contracts/invariants

MUST NOT:
explicit forbidden actions

RETURN:
- files inspected
- files changed
- implementation/findings
- blockers
- assumptions

Workers should not return long repository histories or duplicate parent context.

---

## Fast Execution Sequence

For substantial tasks, the parent should follow this sequence:

1. Read applicable `AGENTS.md`.
2. Verify branch / HEAD / working tree.
3. Understand current authority and invariants.
4. Identify independent workstreams.
5. Assign explicit non-overlapping ownership.
6. Spawn useful `dubvi_worker` workers concurrently.
7. Continue parent-owned work that does not conflict with workers when useful.
8. Collect worker results.
9. Integrate once.
10. Resolve integration conflicts centrally.
11. Run relevant focused automated checks once.
12. Run the full regression suite once when warranted.
13. Run required static / compile / diff validation.
14. Commit coherent source changes.
15. Produce the requested runtime artifact/demo.
16. Report `READY FOR TUAN REVIEW`.
17. USER performs final product/demo evaluation.

Do not add artificial intermediate approval gates unless there is a genuine
architectural ambiguity or hard blocker.

---

## Automated Verification Policy

Automated verification is centralized in the parent.

Workers MUST NOT execute the repository test suites.

After integration, the parent may run:

- focused tests relevant to changed behavior
- full regression suite when warranted
- compileall
- AST/import smoke
- git diff --check
- ffprobe
- FFmpeg full decode
- deterministic artifact integrity checks

Avoid repeated cycles such as:

focused tests
→ full tests
→ reviewer full tests
→ final full tests again

unless a failure required another code change.

The intent is:

verify once after integration,
not independently inside every worker.

---

## Product Acceptance Policy

Automated correctness and product acceptance are separate.

Agents may verify:

- file existence
- codec contract
- decode success
- timing invariants
- schema integrity
- test results
- deterministic artifact properties

Agents MUST NOT make the user's final subjective judgment about:

- visual quality
- subtitle aesthetics
- voice naturalness
- perceived lip sync
- masking appearance
- pacing feel
- overall production quality

The user personally reviews the final artifact.

Before user review, use:

`READY FOR TUAN REVIEW`

Do NOT use:

`PRODUCT ACCEPTED`
`DEMO APPROVED`
`VISUALLY APPROVED`

unless the user has explicitly reviewed and accepted the artifact.

---

## Demo-First Rule

When prerequisites are satisfied and the task requires a demo:

produce the demo as early as technically safe.

Do not create unnecessary architecture checkpoints after the execution path is
known.

Do not stop after every internal layer merely to request approval.

Continue until:

- the requested artifact exists and automated validation completes, or
- a true hard blocker prevents completion

Examples that are NOT automatic stop conditions:

- a bounded generic module requires correction
- a source cue lacks optional evidence
- video duration changes due to valid adaptive retiming
- subtitle display requires safe wrapping
- masking requires a generic geometry correction

Examples of real blockers:

- corrupt required media
- invalid canonical source ambiguity that cannot be resolved safely
- provider unavailable after bounded policy
- unrecoverable timeline invalidity
- renderer cannot produce valid media after bounded generic correction

---

## DUBVI Architecture Preservation

Workers and parent MUST preserve current authority boundaries.

Do not:

- invent semantic source facts
- fabricate timing evidence
- fabricate source positions
- mutate canonical utterance identity casually
- map provider output by array position where canonical IDs exist
- hard-code fixture IDs/text/timestamps in production logic
- weaken fail-closed contracts merely to produce a demo
- silently replace canonical localized text with delivery-specific text

Keep distinct:

- canonical source understanding
- canonical Vietnamese localization
- delivery-specific Vietnamese
- speech timing
- delivery timeline
- viewer-facing subtitle layout

Technical identity may be deterministic/system-owned.

Semantic truth must remain evidence-backed.

---

## Timeline Authority

When adaptive retiming exists, use the accepted source→output timeline as the
single timing authority.

Do not independently calculate competing output timelines in:

- renderer
- subtitle placement
- masking
- voice scheduling

Video, Vietnamese voice, subtitle timing, and source-mask timing must map
through the same accepted transform where applicable.

Never reintroduce audio-only downstream serialization that causes cumulative
scene drift.

---

## Subtitle Policy

Viewer-facing Vietnamese subtitles should follow the accepted generic subtitle
layout system.

Preserve these contracts unless the task explicitly changes them:

- maximum 2 visible lines per display cue
- no glyph overflow
- no outline/shadow overflow
- subtitle background box remains inside frame bounds
- actual rendered glyph metrics drive layout
- display-only splitting may be used when a long utterance cannot fit safely
- display splitting must not alter TTS or canonical localization
- reliable `DIALOGUE_SUBTITLE` source evidence may anchor subtitle placement
- `WATERMARK`, `STORY_TEXT`, and `UNKNOWN` must not become dialogue subtitle
  placement anchors

---

## Source Text Cleanup Policy

Source cleanup should distinguish at least:

- `DIALOGUE_SUBTITLE`
- `WATERMARK`
- `STORY_TEXT`
- `UNKNOWN`

Automatic cleanup targets:

- `DIALOGUE_SUBTITLE`
- `WATERMARK`

Preserve:

- `STORY_TEXT`

Do not aggressively erase `UNKNOWN` without sufficient evidence.

Prefer tight evidence-bounded cleanup over broad horizontal masking.

Do not use moving-body clean plates that create visible anatomy smears.

Vietnamese subtitles should be rendered after source-text cleanup.

---

## Repository Hygiene

Workers MUST NOT commit.

Only parent commits.

Before commit, parent verifies intended tracked changes.

Never stage unrelated files.

Preserve known pre-existing untracked files unless a task explicitly owns them.

Current known preserved prototypes:

- `local-worker/source_locked_timing.py`
- `local-worker/tests/test_source_locked_timing.py`

Do not stage, modify, delete, reset, or commit those files unless explicitly
authorized.

---

## Commit Policy

Commit only after:

- integration completes
- required automated validation passes
- repository diff is understood
- unrelated files are excluded

Prefer coherent commits describing the actual generic change.

Runtime media and diagnostics remain outside the repository unless repository
policy explicitly requires tracking them.

---

## Reporting Policy

Subagent reports should be concise.

Each worker returns only:

- FILES INSPECTED
- FILES CHANGED
- IMPLEMENTATION / FINDINGS
- BLOCKERS
- ASSUMPTIONS

The parent returns one consolidated report containing relevant items such as:

- AUTHORITY
- WORKERS USED
- OWNERSHIP
- FILES CHANGED
- IMPLEMENTATION
- AUTOMATED VERIFICATION
- COMMIT SHA
- REPOSITORY STATUS
- ARTIFACT PATH
- KNOWN TECHNICAL DEBT
- `READY FOR TUAN REVIEW`

Do not repeat the same information across multiple sections.

---

## Performance Principle

Optimize for useful work completed per wall-clock minute.

Parallelize:

- independent implementation
- independent code inspection
- independent runtime/log analysis
- read-only architecture tracing

Do not parallelize:

- conflicting edits
- duplicate testing
- duplicate QA
- duplicate repository archaeology
- repeated full-suite validation

Use the smallest number of agents that materially reduces execution time.
