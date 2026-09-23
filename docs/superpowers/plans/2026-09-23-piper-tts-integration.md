# Piper TTS Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make local Vietnamese Piper/ONNX TTS the default provider for both Translator execution paths on the user's CPU laptop, while preserving CapCut and Edge as explicit alternatives.

**Architecture:** Keep provider selection at the existing `dubvi_worker.synthesize` boundary. Add one lazy, process-level Piper runtime that loads the ONNX model once, emits WAV files, and leaves the existing `fit_voice`, timing, subtitle, and render stages unchanged. The legacy worker and modern P1C adapter will both pass through the same provider boundary instead of calling CapCut directly.

**Tech Stack:** Python 3.12, `onnxruntime`, `piper-phonemize`, the Vietnamese VITS ONNX model from `phatjkk/vits-tts-vietnamese`, existing FFmpeg timing/render helpers, and the repository's `unittest` suite.

**Spec:** The approved bounded design in chat for this task; no separate architectural specification is required.

## Global Constraints

- The target machine is CPU-only for this workflow: Intel Core i3-1115G4, 2 physical cores / 4 logical processors, about 20 GB RAM, Intel UHD graphics.
- `piper` is the default provider and `fast` is the default speech speed; `capcut` and `edge` remain selectable through `DUBVI_TTS_PROVIDER`.
- Do not commit model binaries, Hugging Face caches, generated WAV/MP3 files, or video artifacts.
- Load the Piper ONNX session once per process; never recreate the session for every utterance.
- Piper output must continue through the existing `fit_voice` timing policy; no independent audio timeline is introduced.
- Local TTS must not make a network request or consume API/model tokens after the model files are installed. Translation behavior is out of scope.
- Preserve the current canonical utterance identity, subtitle timing, masking, and render contracts.
- Keep the existing working-tree checkpoint committed as `b16611d`; the Piper work starts in a new commit.

## Review Focus

- Missing or incomplete ONNX/config files must fail with a clear setup error instead of silently falling back to CapCut.
- Selecting `piper` while the configured legacy voice still has a `BV...` name must route to Piper, not CapCut.
- Legacy and modern P1C paths must both honor the same provider setting.
- Piper output longer than its delivery window must retain the existing `fit_voice` / review behavior.
- Repeated utterances in one process must reuse one model session and must not reload the ONNX model per segment.

### Task 1: Add provider configuration and dependency contract

**Files:**
- Modify: `local-worker/dubvi_worker.py:Settings`, `Settings.from_env`, and provider constants.
- Modify: `local-worker/p1c_processing.py:build_canonical_runtime_config` and `_legacy_settings`.
- Modify: `local-worker/requirements.txt`.
- Test: `local-worker/tests/test_tts_provider_config.py`.

**Interfaces:**
- `Settings` exposes `tts_provider`, `piper_model`, and `piper_speed`.
- `DUBVI_TTS_PROVIDER` accepts `piper`, `capcut`, or `edge`, defaulting to `piper`.
- `DUBVI_PIPER_MODEL` overrides the model path; otherwise use `<DUBVI_MODEL_DIR>/piper-vietnamese/pretrained_vi.onnx`.
- `DUBVI_PIPER_SPEED` accepts `normal`, `fast`, or `very_fast`, defaulting to `fast`.
- Runtime config carries the selected provider as `ttsProvider`; profile persona and voice identity remain unchanged.

- [ ] **Step 1: Write the failing tests.** Add tests that assert the default provider is `piper`, an explicit `capcut` override is preserved, invalid provider/speed values fail with `ValueError`, and `build_canonical_runtime_config` reflects `DUBVI_TTS_PROVIDER` without changing the profile persona.

```python
def test_settings_default_to_cpu_piper(self):
    with patch.dict(os.environ, {}, clear=True):
        settings = Settings.from_env()
    self.assertEqual("piper", settings.tts_provider)
    self.assertEqual("fast", settings.piper_speed)
    self.assertTrue(str(settings.piper_model).endswith("piper-vietnamese\\pretrained_vi.onnx"))

def test_explicit_capcut_provider_remains_available(self):
    with patch.dict(os.environ, {"DUBVI_TTS_PROVIDER": "capcut"}, clear=False):
        self.assertEqual("capcut", Settings.from_env().tts_provider)
```

- [ ] **Step 2: Run the focused tests and verify RED.**

  Run from `local-worker`:

  ```powershell
  python -m unittest tests.test_tts_provider_config
  ```

  Expected: FAIL because the new settings fields/defaults and test module do not exist yet.

- [ ] **Step 3: Implement the minimal configuration.** Add a shared provider validator, the two Piper settings, the `piper` default, and the model/speed environment parsing. Make `build_canonical_runtime_config` choose `DUBVI_TTS_PROVIDER` when present and use `piper` as the local default. Add `onnxruntime>=1.15,<2` and `piper-phonemize==1.1.0` to the worker runtime requirements.

- [ ] **Step 4: Run the focused tests and verify GREEN.**

  ```powershell
  python -m unittest tests.test_tts_provider_config
  ```

  Expected: all configuration tests pass.

- [ ] **Step 5: Commit the configuration boundary.**

  ```powershell
  git add local-worker/dubvi_worker.py local-worker/p1c_processing.py local-worker/requirements.txt local-worker/tests/test_tts_provider_config.py
  git commit -m "feat: configure local piper tts provider"
  ```

### Task 2: Implement the lazy Vietnamese Piper runtime

**Files:**
- Create: `local-worker/piper_tts.py`.
- Modify: `local-worker/dubvi_worker.py:synthesize` and provider imports.
- Test: `local-worker/tests/test_piper_tts.py`.

**Interfaces:**
- `PiperTtsError(RuntimeError)` reports setup, phonemization, inference, and output errors.
- `PiperTtsRuntime(model_path: Path)` owns one `onnxruntime.InferenceSession` and exposes `synthesize(text: str, target: Path, speed: str) -> None`.
- `piper_synthesize(text: str, voice: str, target: Path, settings: Settings) -> None` uses a process-level runtime cache keyed by resolved model path and ignores CapCut-only `BV...` voice names.
- The generated file is 22,050 Hz, mono, signed 16-bit PCM WAV, which FFmpeg and `fit_voice` already accept.

- [ ] **Step 1: Write the failing tests.** Cover speed-to-length-scale mapping, missing model/config diagnostics, Vietnamese WAV output contract through a small local test model fixture or a deterministic runtime seam, and reuse of the same runtime for two calls.

```python
def test_missing_model_fails_with_setup_path(self):
    with self.assertRaises(PiperTtsError) as raised:
        PiperTtsRuntime(Path("missing-pretrained_vi.onnx"))
    self.assertIn("pretrained_vi.onnx", str(raised.exception))

def test_speed_values_are_bounded(self):
    self.assertEqual(1.0, speed_length_scale("normal"))
    self.assertEqual(0.6, speed_length_scale("fast"))
    self.assertEqual(0.4, speed_length_scale("very_fast"))
    with self.assertRaises(ValueError):
        speed_length_scale("turbo")
```

- [ ] **Step 2: Run the focused tests and verify RED.**

  ```powershell
  python -m unittest tests.test_piper_tts
  ```

  Expected: FAIL because `piper_tts.py` and its runtime interfaces do not exist.

- [ ] **Step 3: Implement the smallest runtime.** Read the model JSON, phonemize Vietnamese codepoints with `piper_phonemize`, create the ONNX session with CPU thread limits inherited from the worker environment, run inference with the selected length scale, normalize the output to int16, and write WAV through the standard-library `wave` module. Cache the runtime by resolved model path. Reject empty text and missing model/config files before inference.

- [ ] **Step 4: Route `synthesize` to Piper before voice-prefix checks.** When `settings.tts_provider == "piper"`, call `piper_synthesize` even if `voice` is `BV007_streaming` or another CapCut identifier. Keep the existing CapCut and Edge branches unchanged for explicit provider selections; do not silently fall back from a missing Piper model to an online provider.

- [ ] **Step 5: Run the focused runtime tests and verify GREEN.**

  ```powershell
  python -m unittest tests.test_piper_tts tests.test_execution_limits
  ```

  Expected: all runtime and CPU-limit tests pass without network access.

- [ ] **Step 6: Commit the runtime.**

  ```powershell
  git add local-worker/piper_tts.py local-worker/dubvi_worker.py local-worker/tests/test_piper_tts.py
  git commit -m "feat: add cached vietnamese piper runtime"
  ```

### Task 3: Connect the modern P1C TTS path

**Files:**
- Modify: `local-worker/tts_timing.py:_validate_options` and the default synthesizer boundary.
- Modify: `local-worker/modern_p1c_adapter.py:tts_stage`.
- Modify: `local-worker/p1c_processing.py` if runtime provider propagation needs the same helper used by Task 1.
- Test: `local-worker/tests/test_tts_timing.py` or the existing `local-worker/tests/test_cp3a_timing.py` and `local-worker/tests/test_modern_p1c_adapter.py`.

**Interfaces:**
- `run_tts_timing(..., tts_provider="piper")` is accepted and includes the provider in its existing cache key.
- `modern_p1c_adapter.tts_stage` passes the selected runtime provider and a three-argument synthesizer closure that invokes `dubvi_worker.synthesize` with one loaded `Settings` object.
- Explicit `capcut` CP3A tests keep using `capcut_synthesize` and continue to pass.

- [ ] **Step 1: Write the failing tests.** Add one test that accepts `piper` in `_validate_options`, one that proves the modern stage no longer hard-codes `capcut_synthesize`, and one that proves a `BV...` voice still reaches the Piper branch when the runtime provider is `piper`.

```python
def test_tts_timing_accepts_piper_provider(self):
    _validate_options("piper", "BV421_vivn_streaming", 1.12)

def test_modern_tts_stage_uses_runtime_provider(self):
    stages = build_modern_stage_bundle(
        runtime_config={"ttsProvider": "piper", "voice": {"name": "BV421_vivn_streaming"}, "localizationProfile": "loc-v1"}
    )
    self.assertTrue(callable(stages["tts"]))
```

- [ ] **Step 2: Run the focused tests and verify RED.**

  ```powershell
  python -m unittest tests.test_cp3a_timing tests.test_modern_p1c_adapter
  ```

  Expected: the new Piper assertions fail because CP3A only accepts CapCut and the modern adapter passes CapCut directly.

- [ ] **Step 3: Implement the provider handoff.** Allow `piper`, `capcut`, and `edge` in `tts_timing`; include the provider in existing cache identity; replace the direct `capcut_synthesize` callback in the modern adapter with the configured `synthesize` closure; pass `runtime_config["ttsProvider"]` into `run_tts_timing`.

- [ ] **Step 4: Run the focused tests and verify GREEN.**

  ```powershell
  python -m unittest tests.test_cp3a_timing tests.test_modern_p1c_adapter tests.test_tts_provider_config
  ```

  Expected: the new tests and existing explicit-CapCut tests all pass.

- [ ] **Step 5: Commit the pipeline handoff.**

  ```powershell
  git add local-worker/tts_timing.py local-worker/modern_p1c_adapter.py local-worker/p1c_processing.py local-worker/tests/test_cp3a_timing.py local-worker/tests/test_modern_p1c_adapter.py local-worker/tests/test_tts_provider_config.py
  git commit -m "feat: route modern translator tts through provider boundary"
  ```

### Task 4: Install the model, document setup, and run an audio smoke test

**Files:**
- Modify: `local-worker/README.md`.
- Do not add model files to Git; store them under the configured local `DUBVI_MODEL_DIR`.
- Test artifact: local WAV smoke output outside the repository.

**Interfaces:**
- Model files live at `<DUBVI_MODEL_DIR>/piper-vietnamese/pretrained_vi.onnx` and its adjacent `.json` config.
- `.env` values are `DUBVI_TTS_PROVIDER=piper`, `DUBVI_PIPER_SPEED=fast`, and optional `DUBVI_PIPER_MODEL=<absolute path>`.
- The smoke test accepts Vietnamese text, produces a non-empty WAV, and reports generation time, audio duration, and real-time factor.

- [ ] **Step 1: Add setup documentation.** Document the exact Windows install command, model directory layout, environment variables, and the fact that the model is intentionally untracked. Include the model source repository and its license/usage caveat without embedding a binary in the project.

- [ ] **Step 2: Install the two runtime packages in the existing Python 3.12 environment.** Run:

  ```powershell
  python -m pip install "onnxruntime>=1.15,<2" "piper-phonemize==1.1.0"
  ```

  If the pinned phonemizer has no compatible Windows/Python 3.12 wheel, stop at this step and report the installer error before changing the runtime design.

- [ ] **Step 3: Download the ONNX model and adjacent JSON config to the local model cache.** Verify both files exist and their sizes are non-zero; do not place them under a tracked repository directory.

- [ ] **Step 4: Run the real one-sentence smoke test.**

  ```powershell
  $env:DUBVI_TTS_PROVIDER = "piper"
  $env:DUBVI_PIPER_SPEED = "fast"
  python -c "from pathlib import Path; from dubvi_worker import Settings, synthesize, ffprobe_duration; p=Path('piper_smoke.wav'); s=Settings.from_env(); synthesize('Phòng trọ này đúng là một câu chuyện không thể tin nổi.', 'BV421_vivn_streaming', p, s); print(p, p.stat().st_size, round(ffprobe_duration(p), 3))"
  ```

  Expected: a non-empty local WAV, no CapCut/Edge request, and a duration measurable by FFprobe.

- [ ] **Step 5: Run the relevant automated checks once after integration.**

  ```powershell
  python -m unittest discover -s tests -p "test_*.py"
  python -m compileall -q .
  git diff --check HEAD~3..HEAD
  git status --short
  ```

  Record any pre-existing FFmpeg/Groq fixture failures separately from Piper results; never hide them by weakening tests.

- [ ] **Step 6: Commit documentation and finish at user review.**

  ```powershell
  git add local-worker/README.md
  git commit -m "docs: document local piper tts setup"
  ```

  Produce a short 10-second preview using the existing fixed pill mask pipeline, report the measured TTS real-time factor, and stop at `READY FOR TUAN REVIEW`. The user remains the final judge of voice naturalness and subtitle feel.

## Handoff

Execute the tasks in order. The recommended execution method is **Native**: the changes are tightly coupled across three existing Python modules, and one parent-run integration test is cheaper than coordinating workers that cannot run QA or the full suite. Do not begin implementation until the plan is reviewed and the execution method is confirmed.
