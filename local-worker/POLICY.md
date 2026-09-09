# DUBVI content policy

`content_policy` is a local-first, JSON-serializable policy layer owned by the translator repository. It has no browser, cloud-write, upload, deletion, or move side effects.

## Stages

- `prefilter` checks explicit duration, photo/slideshow metadata, optional photo text signals, and de-overlapped ASR speech density.
- `trust` evaluates centralized Chinese and Vietnamese commercial/trust signals plus explicit commerce metadata.
- `ai_script_judge` is optional. It scores hook, conflict, pacing, and relatability through the existing Gemini key-pool path when configured.

## Result schema

Every result is JSON serializable and has this shape:

```json
{
  "policy_schema_version": 1,
  "overall_decision": "pass|reject|review|error|unavailable",
  "stage_results": [
    {
      "stage": "prefilter|trust|ai_script_judge",
      "decision": "pass|reject|review|error|unavailable",
      "reason_codes": ["..."],
      "human_summary": "...",
      "confidence": 0.0,
      "metrics": {},
      "details": {},
      "needs_review": false
    }
  ],
  "reason_codes": ["..."],
  "human_summary": "...",
  "needs_review": false,
  "evaluated_at": "UTC ISO-8601 timestamp"
}
```

Decision precedence is `reject > error > review > unavailable > pass`. An enabled but unavailable AI judge turns an otherwise passing policy result into `review`; it never silently becomes `pass`.

## Worker configuration

`DUBVI_POLICY_MODE=off` is the default and preserves the existing worker behavior.

- `off`: no policy evaluation.
- `review`: writes `<job output>/policy.json` and logs the decision, but continues processing.
- `enforce`: confident rejects fail the job; `review`, `error`, or `unavailable` route it to `awaiting_review`.

`DUBVI_POLICY_ENABLE_AI_JUDGE=false` is the default. Set it to `true` only when optional provider-based judging is intended. Missing keys, quota, timeout, provider error, or malformed JSON return a visible non-pass result and do not call a fallback acceptance path.

## Future P1B consumption

P1B orchestration can store the same result in a handoff sidecar, editorial state, or UI without translating its meaning. It should call the pure local policy stages before expensive work when candidate metadata and ASR evidence are available, then respect the `overall_decision` and `needs_review` fields rather than duplicating keyword logic.
