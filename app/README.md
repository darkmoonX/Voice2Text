# Voice2Text App Runtime

Windows live subtitle overlay runtime (Python main process + C++ capture bridge).

## Stack

- UI: `PySide6`
- Capture: Python capture adapters + C++ bridge (`WASAPI loopback`, `Application Loopback Capture`)
- STT providers: `whisperx` (default) and optional `whispercpp` (resident whisper.cpp Vulkan server backend, with subprocess fallback). Legacy persisted names such as `whisper` / `faster-whisper` are normalized to `whisperx` for compatibility.
- Optional translation: `Argos Translate` or offline NLLB (`CTranslate2` + `transformers`)

## Runtime Structure

```text
app/
  src/
    app/
      capture/      # capture factory + cpp bridge adapter
      pipeline/     # subtitle assembler, delta logger, runtime recovery
      settings/     # i18n + mapping + schema
      stt/          # STT registry/factory/health-check, WhisperX, whisper.cpp, downloads, diarization, speaker identity
    runtime_bin/    # bridge executable output
  native/
    audio_bridge/   # C++ bridge source + build script
  scripts/
    diagnostics/    # manual diagnostics scripts
```

## Architecture Seams

- `src/app/bootstrap.py`: entry orchestration only.
- `src/app/bootstrap_args.py`: CLI interface and listing commands.
- `src/app/bootstrap_config.py`: argparse -> `RuntimeConfig` mapping.
- `src/app/bootstrap_runtime.py`: Qt runtime wiring and settings-apply restart flow.
- `src/app/pipeline/transcription_loop.py`: capture/STT loop state machine.
- `src/app/pipeline/audio_windowing.py`: segment/hop byte-alignment helper.
- `src/app/pipeline/segment_artifacts.py`: latest raw/STT segment wav debug snapshots.
- `src/app/pipeline/gpu_telemetry.py`: debug-mode CUDA / nvidia-smi telemetry reporter.
- `src/app/settings/source_selection_dialog.py`: reusable source picker dialog.
- `src/app/settings/presenter.py`: settings view-model helpers and alignment suggestion rules.
- `src/app/settings/widgets.py`: reusable combo/widget builders.
- `src/app/capture/session_match.py`: app-session token normalization/matching utilities.
- `src/app/capture/mixer_utils.py`: Python capture mixing/resampling utilities.
- `src/app/capture/bridge_probe.py`: C++ bridge executable health/capability probes.

## Setup

```powershell
cd app
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional WhisperX diarization / speaker-identity dependencies:

```powershell
pip install -r requirements-stt-extra.txt
```

Optional offline NLLB translation dependencies:

```powershell
pip install -r requirements-translation-extra.txt
```

## Build Native Capture Bridge

```powershell
cd app\native\audio_bridge
.\build_bridge.ps1
```

Output executable:
- `app/src/runtime_bin/voice2text_capture_bridge.exe`

Notes:
- In debug mode, `app/src/segments/latest_segment_cpp_bridge.wav` is a rolling-tail artifact from the C++ bridge.
- Its window length now follows `segment_seconds` (same setting used by Python STT windows).

Advanced build parameters and MSVC/MinGW presets:
- [app/native/audio_bridge/README.md](/D:/Voice2Text/app/native/audio_bridge/README.md)

## Optional whisper.cpp Vulkan Backend

`whispercpp` is an alternative ASR backend for non-CUDA GPU acceleration through whisper.cpp's Vulkan path.
WhisperX remains the default and is still the only backend with forced alignment, diarization, and speaker labels.
The default `server` mode starts a local resident `whisper-server.exe` child process on `127.0.0.1`, warms it before
capture starts, and reuses the loaded model for every live window. The older `subprocess` mode still exists as a
fallback/offline path and runs `whisper-cli.exe` per window.

The whisper.cpp backend emits segment timestamps only; Voice2Text synthesizes per-token timestamps from those
segment spans and marks `alignment_enabled=False`, reusing the same rolling subtitle merge path as WhisperX
alignment-off mode.

Build/copy the local Vulkan binaries and runtime DLLs:

```powershell
cd app
.\build_whispercpp.ps1
```

Expected output:
- `app/src/runtime_bin/whispercpp/whisper-cli.exe`
- `app/src/runtime_bin/whispercpp/whisper-server.exe`
- colocated `whisper.dll`, `ggml*.dll`, `ggml-vulkan.dll`
- `app/src/models/whispercpp/ggml-silero-v5.1.2.bin` for whisper-server VAD

`vulkan-1.dll` is supplied by the system Vulkan loader and is not bundled. Install the Vulkan runtime/SDK if the
driver does not provide it.

Run with the bundled binary and auto-downloaded ggml model:

```powershell
cd app\src
python main.py --stt-provider whispercpp --whispercpp-model-size medium
```

`medium` is the recommended live model — realtime (~0.45x at `seg10/hop2`) and cross-window stable, so the rolling
merge dedups cleanly. `large-v2` is higher-accuracy but, in local testing, transcribes the same audio differently
across overlapping windows, which the text-keyed merge cannot dedup → visible duplication; use it for
**offline / file-replay** runs, not live. `large-v3` is not recommended at all (slower and more hallucination-prone).
Note: very low `--hop-seconds` (large overlap) over-stresses the merge even for `medium`; keep the default `hop 2.0`.

Useful overrides:

```powershell
$env:VOICE2TEXT_WHISPERCPP_BIN="D:\path\to\whisper-cli.exe"
$env:VOICE2TEXT_WHISPERCPP_SERVER_BIN="D:\path\to\whisper-server.exe"
$env:VOICE2TEXT_WHISPERCPP_VAD_MODEL="D:\path\to\ggml-silero-v5.1.2.bin"
python main.py --stt-provider whispercpp --whispercpp-model-path D:\models\ggml-medium.bin
python main.py --stt-provider whispercpp --whispercpp-mode subprocess  # 0032 CLI fallback path
python main.py --stt-provider whispercpp --stt-variant cpu  # adds -ng
python main.py --stt-provider whispercpp --whispercpp-server-vad  # opt-in; off by default
python main.py --stt-provider whispercpp --whispercpp-server-max-len 32  # optional segment-length cap
```

Model cache:
- default directory: `app/src/models/whispercpp/`
- filename pattern: `ggml-<size>.bin`
- ASR source repo: `ggerganov/whisper.cpp`
- VAD source repo: `ggml-org/whisper-vad`, default filename `ggml-silero-v5.1.2.bin`
- approximate disk size: `medium` ~1.5GB, `large-v2`/`large-v3` ~3.1GB

Server VAD is opt-in because current `whisper-server` builds can crash on windows with zero VAD speech segments.
When enabled, Voice2Text always passes `--vad-model`; it never starts whisper-server with a bare `--vad`.
Server mode uses real `verbose_json` `words[]` timestamps when available, falling back to synthesized segment
timestamps only when word timings are absent. `--whispercpp-server-max-len` is optional and defaults to `0`.

## Run

```powershell
cd app\src
python main.py
```

### Settings Persistence

- Runtime settings changed from tray `Settings` are saved to `app/src/runtime_settings.json`.
- On next launch, the app restores these settings before capture/STT startup.
- The settings dialog now includes a bottom-left `Reset defaults` button that resets all visible options back to built-in defaults (apply with `OK`).
- `Audio preprocess` is now a direct on/off switch in Settings and persists across restart.
- Pre-STT VAD Gate has been removed. WhisperX internal VAD (`silero-vad` / `pyannote`) is the only speech gate.
- Speaker-profile embedding backend is now selectable in Settings (`pyannote` / `speechbrain-ecapa` / `nemo-titanet`).
- Advanced speaker-profile options remain config-driven: `whisperx_speaker_profile_enabled`, `whisperx_speaker_profile_model`, `whisperx_speaker_speechbrain_model`, `whisperx_speaker_nemo_model`, `whisperx_speaker_profile_match_threshold`, `whisperx_speaker_profile_min_seconds`, `whisperx_speaker_profile_store_path`.
- Speaker-profile learn-path quality gate (`whisperx_speaker_profile_quality_gate_enabled`, default off; CLI `--speaker-profile-quality-gate`): when on, a low-quality speaker clip (empty / music-sound tag / `♪` / degenerate repetition / mean word-score below `whisperx_speaker_profile_quality_min_confidence`) can still match an existing profile for display but never updates or creates an embedding centroid, so gibberish and music tails do not pollute speaker identities. The displayed speaker label for a span is unaffected — only profile *learning* is gated.
- Transcript export is available in Settings:
  - enable/disable export
  - formats (`txt,srt,json`)
  - include timestamps
  - include speaker labels
- Manual `Export Subtitle...` exports the most recent runtime interval: from the last runtime start to the latest pause/stop, or to the export moment if capture is still running.
  - exports are written on runtime stop.

#### JSON export schema — confidence / stability fields

The `json` export carries optional confidence/stability metrics derived from WhisperX alignment word scores
(gated by `transcript_export_include_confidence`, default on; set off to get the pre-0021 byte-identical json).
A token is counted *stable* when its alignment `score >= 0.60` and its duration is in `[0.02, 1.2]s` (mirrors the
provider's `stability_ratio`). The fields:

- **`summary.mean_confidence`** — mean alignment `score` over all ingested tokens (`0.0` when there are none).
- **`summary.stable_token_ratio`** — stable tokens / total tokens.
- **`cues[].confidence` / `min_score` / `stable_ratio`** — mean / minimum token score and stable fraction *within
  that cue*. Present only on cues built from tokens; text/event-only fallback cues omit them.
- **`events[].stability_ratio` / `stable_token_count`** — the per-window stability the provider reported for that
  decode window (alongside the existing `token_count`).

`txt` and `srt` are unaffected — they only render `text/speaker/start/end`, so the extra cue keys never reach them.

#### JSON export schema — separated speaker labels (round 0027)

A subtitle line has three distinct speaker labels that are normally conflated; the `json` export keeps them
separate per cue (gated by `transcript_export_include_speaker`, default on) so a debug/export diff shows where
they agree and diverge:

- **`cues[].speaker`** — the *effective* label (profile-preferred), exactly what the SRT/TXT marker renders
  (unchanged).
- **`cues[].visible_speaker`** — the rendered `[spk_xxx]` marker for that cue (equals `speaker` in the export
  view; surfaced explicitly so the field set is self-describing).
- **`cues[].profile_speaker`** — the cross-window speaker-profile (centroid) identity, dominant within the cue.
- **`cues[].raw_speaker`** — the local per-window diarization label (`local_speaker`), dominant within the cue.

These are observability fields only: the effective `speaker`, the dedup key, SRT/TXT text, and the live overlay
are byte-identical (additive, metric-neutral). In `--debug-mode`/trace runs the provider also emits a per-window
`[speaker-labels] visible=… raw=… profile=…` line so live divergence (e.g. a marker lagging one segment, or a
raw label the profile remapped) is visible in the log.

#### Pre-run health check + model/alignment cache

`python main.py --stt-health-check` runs structured, actionable checks and exits (scope via
`--stt-health-check-scope active|all`). Beyond the existing model/flag detail lines it now reports
`check[ok|warn] <id>` rows for **CUDA/cuBLAS, FFmpeg, HuggingFace token, the C++ capture bridge, and the
model/alignment cache**, each with a `fix` hint when not ok (the HF token is reported presence-only — never
echoed). The cache check uses `stt/model_cache.py`, a headless scanner of `models/whisperx/` (`stt/<model>` +
`align/<bucket>/<lang>/<model>`) that reports per-folder size + readiness, totals (`cache_summary`), and a
root-guarded `delete_cache_entry`. (A Qt wizard + cache-manager dialog on top of these is the round-0022 Phase B
follow-up.)

#### CPU / no-GPU realtime (the `cpu` preset)

On a machine without CUDA, run live subtitles with `--preset cpu` (also selectable in Settings as `cpu`). It
bundles the levers that keep the pipeline realtime on CPU: `stt_variant=cpu`, `model=small`, `compute_type=int8`,
`beam_size=3`, diarization/speaker-profile off, and **forced alignment off** — alignment is the dominant CPU cost
(~8× slower) and unaffordable live (beam size, by contrast, is nearly free on CPU — ~6% rtf from 1→5 — since the
model forward pass dominates, so the preset uses beam 3 for a small accuracy gain). Because the rolling-window de-duplication is normally driven by word
timestamps (which alignment produces), the provider **synthesizes per-word timestamps from each segment's span**
when alignment is off (round 0024), so overlapping windows still de-duplicate instead of piling up repeated text.

Trade-offs: no word-level timestamps, so exported SRT/JSON has segment-level (not precise per-word) timing and no
speaker word-attribution; best for non-CJK speech (CJK leans harder on the dropped accuracy levers). Tune CPU
parallelism with `--cpu-threads N` (0 = CTranslate2 default); raise it on multi-core CPUs.

```powershell
python main.py --preset cpu                  # non-CUDA realtime
python main.py --preset cpu --cpu-threads 8  # use more CPU cores
```

### Common Commands

```powershell
python main.py --list-devices
python main.py --list-app-sessions
python main.py --source-mode app --app-names msedge.exe

python main.py --model small --stt-variant gpu
python main.py --stt-provider whisperx --model large-v2 --no-whisperx-vad
python main.py --model large-v2 --segment-seconds 6.0 --hop-seconds 1.5
python main.py --model large-v2 --whisperx-alignment-device cpu

python main.py --source-language zh-hant --cjk-no-space-gap-seconds 0.2
python main.py --debug-mode
```

### Optional Runtime Parameters

```powershell
python main.py --ffmpeg-dll-dir D:\FFmpeg\ffmpeg-7.1.1-full_build-shared\bin
python main.py --cublas-source-dll D:\CUDA\bin\x64\cublas64_13.dll
python main.py --ui-language en
```

### WhisperX Model Download Behavior

- WhisperX model auto-download now uses direct HTTP streaming (byte-based progress) by default.
- Progress is emitted as `[download] ...` status lines and is visible in overlay + `app/src/logs/voice2text.log`.
- Download progress now preflights remote file sizes (when available) and reports `current/total MB` with `%` bar.
- If total size is unknown (external/internal downloader path), progress should show bytes only and must not show fake `100%` intermediate lines.
- Alignment downloads triggered through torch-hub/torchaudio now bridge into app logs with the same rule; when remote headers expose total size, app log shows `x/y MB` totals instead of unbounded growth lines.
- When torch-hub file-level progress is active for a transfer, generic fallback progress for that same transfer is fully suppressed (including completion) to avoid duplicate trailing lines after `100%`.
- If direct download fails, startup reports explicit failure instead of silently hanging.
- Downloaded files are now size-validated against HF manifest; mismatched files are deleted and retried once with a clean download.
- If local cache is still incomplete (for example truncated `model.bin`), startup now fails fast with an explicit `auto-repair did not finish` reason.
- Snapshot fallback is disabled by default; enable only when needed:
  - PowerShell: `$env:VOICE2TEXT_WHISPER_USE_SNAPSHOT='1'`
- WhisperX STT auto-download now follows the same direct byte-progress path (no file-count `Fetching N files` progress bars).
- WhisperX snapshot fallback is also disabled by default; enable only when needed:
  - PowerShell: `$env:VOICE2TEXT_WHISPERX_USE_SNAPSHOT='1'`

## Diagnostics

App-source visual probe:

```powershell
cd app
python scripts/diagnostics/app_source_visual_probe.py --show-sessions --app-names msedge.exe --seconds 45
```

Post-fix validation matrix:

```powershell
cd app
.\run_post_fix_validation.ps1
```

Segment-arrival status-routing regression test:

```powershell
cd app
.\.venv\Scripts\python.exe .\scripts\diagnostics\segment_arrival_regression_test.py
```

WhisperX alignment stability test (about 1 minute):

```powershell
cd app
$env:VOICE2TEXT_TRACE_WHISPERX='1'
.\.venv\Scripts\python.exe .\scripts\diagnostics\whisperx_alignment_stability_test.py --duration-seconds 65 --source-wav .\src\segments\latest_segment_stt.wav --alignment-device cuda --model medium --language en
```

WhisperX diarization readiness check (foreground):

```powershell
cd app
.\.venv\Scripts\python.exe .\scripts\diagnostics\whisperx_diarization_readiness_check.py
```

WhisperX diarization foreground probe (with proxy cleared only for this run):

```powershell
cd app
.\.venv\Scripts\python.exe .\scripts\diagnostics\whisperx_diarization_stability_test.py --duration-seconds 70 --clear-proxy --source-wav .\src\segments\latest_segment_stt.wav --model medium --language en
```

Compare pack accuracy test (incremental project flow vs direct WhisperX full-file):

```powershell
cd app
.\.venv\Scripts\python.exe .\scripts\diagnostics\compare_test_data_whisperx.py `
  --input ".\src\tests\compare_whisperx_test\input\YTDown_YouTube_Media_aXqBRYQSGp0_008_128k.m4a" `
  --device cuda `
  --profile fast `
  --segment-seconds 12 `
  --hop-seconds 2.4 `
  --direct-group-seconds 30 `
  --realtime-compare-one-line `
  --export-formats txt,srt,json
```

Chinese alignment comparison example:

```powershell
cd app
.\.venv\Scripts\python.exe .\scripts\diagnostics\compare_test_data_whisperx.py `
  --input ".\src\tests\compare_whisperx_test\input\your_zh_audio.m4a" `
  --model medium `
  --language zh `
  --align-language follow-source `
  --align-model WAV2VEC2_ASR_LARGE_LV60K_960H `
  --align-device cpu `
  --force-alignment on `
  --profile accurate `
  --device cuda
```

Notes:

- Script prints text progress bars, audio-seconds processed, processing speed, and ETA for both direct and realtime phases. Realtime progress also includes the latest window `runtime_timing` breakdown (`raw`, `preprocess`, `stt_artifact`, `transcribe`, `language`, `timestamp`, `merge`, `payload`, `total`).
- Debug-mode main runtime and compare realtime replay emit `[window-timing]` status lines per rolling window with the same timing fields, including empty/no-speech windows.
- `--device cuda` defaults to strict GPU requirement (if fallback to CPU occurs, the case fails early).
- Use `--allow-cpu-fallback` only when you explicitly want CPU comparison.
- In this environment, `torch` is CPU-only, but ASR can still run on CUDA through CTranslate2 in `--profile fast` (ASR-only mode).
- `realtime_project` export writes the final history-style subtitle snapshot after file replay; overlapping per-window token metadata stays in `realtime_debug_trace.jsonl` for diagnostics instead of being stacked into exported subtitles.
- Accurate compare reports include speaker-profile diagnostics in `compare.json` and `[speaker_profile_diagnostics.*]` sections in `compare.txt`, showing profile duration, sample count, weight, and observed local labels.
- Compare text now defaults to exported full subtitle text (`realtime_project.txt` / `direct_whisperx.txt`) before normalization, so metrics no longer depend on the incremental in-memory tail text.
- `direct_whisperx.txt` line breaks are generated by transcript-export cue grouping from token timestamps. Cues split on speaker change or pauses greater than 2 seconds, not punctuation or length-only limits; ASCII fragments such as `1 1 6` / `l o c a l` become `116` / `local`.
- Compare mode options:
  - `--direct-group-seconds 30`: regroup direct transcript into 30-second bins for compare text.
  - `--direct-chunk-seconds 0`: single full-file WhisperX pass (no project-side slicing; lets WhisperX do its own VAD-based segmentation — most complete reference). Positive values (`30`, `60`, `120`) slice the audio into hard, non-overlapping project-side chunks.
  - Note: with `--language auto`, any `--direct-chunk-seconds` larger than `--direct-language-subchunk-seconds` (default 30s) is further hard-sliced into 30s subchunks. Hard cuts have no overlap, so a sentence straddling a boundary can be dropped by WhisperX VAD at the edge.
  - `--realtime-compare-one-line`: flatten realtime transcript to one line for compare text.
- Default compare pack location:
  - inputs: `app/src/tests/compare_whisperx_test/input`
  - outputs: `app/src/tests/compare_whisperx_test/output/<timestamp>`
- Full workflow and output structure are documented in `app/src/tests/compare_whisperx_test/WORKFLOW.md` (docs mirror: `docs/test-data-whisperx-compare.md`).

CUDA/GPU telemetry in debug mode:

- When `--debug-mode` and CUDA device are active, runtime emits periodic `[gpu-telemetry]` lines into `app/src/logs/voice2text.log`.
- Reported metrics include:
  - PyTorch process memory: `alloc`, `reserved`, `max_alloc`, `total`
  - Device-level usage from `nvidia-smi`: `util`, `mem_util`, `vram used/total`

Session-end observability summaries (round 0029):

- On stop/EOF the loop emits, in addition to the per-window timings:
  - `[timing-summary]` — session window count, `realtime_factor` (`mean(window_total)/hop`; > 1.0 = falling behind), `window_total` p50/p95/max, and the dominant stage.
  - `[timing-stages]` — the **full per-stage breakdown** (`name=p50/p95/max s (n=N)`) for every window and WhisperX provider sub-stage (`transcribe`, `wx_asr`, `wx_align`, `wx_diarization`, `wx_speaker_profile`, `merge`, ...), sorted by p50 descending, so the dominant cost is visible without re-instrumenting.
  - `[gpu-telemetry-summary]` — VRAM-used and GPU-util p50/p95/max plus peak torch `max_alloc`, aggregated from the periodic `[gpu-telemetry]` ticks. Only emitted when samples were collected (i.e. a `--debug-mode` CUDA session); silent no-op otherwise.
- These are log-only diagnostics (suppressed from the overlay) and do not change subtitle/translation behavior.

Debug window log visibility:

- Main overlay now only shows curated important statuses (startup/capture/runtime-critical/downloading), while noisy diagnostics stay in logs.
- In debug mode, debug window loads recent runtime log history from `app/src/logs/voice2text.log*` and keeps streaming all new logger lines in real time.
- Third-party library warnings (for example `whisperx.vads.pyannote: No active speech found in audio`) are captured to `voice2text.log` and the debug window, and startup removes pre-existing console stream handlers so these expected warnings do not reach PowerShell.
- Debug window still writes structured event traces to `debug_trace_YYYYMMDD.jsonl`.

Settings window behavior:

- The Settings dialog is opened as an independent dialog instead of an owned child of the translucent overlay window, preventing a transient blank overlay-owned window/tab from flashing while Settings initializes.
- Settings initialization also guards source-row visibility updates until widgets are attached to the dialog layout, preventing unparented Qt widgets from briefly becoming top-level blank windows.

## WhisperX Warmup Behavior

- When forced alignment is enabled, startup warmup preloads the resolved WhisperX alignment model (for example Chinese alignment for `zh-hant`) before first live speech.
- When WhisperX diarization is enabled, startup warmup now also preloads diarization pipeline/resources before first live speech.
- This avoids first-utterance stalls caused by deferred `alignment model loading`.

## WhisperX Crash Diagnostics

- Runtime now enables Python `faulthandler`; native crash traces are written to `app/src/logs/python_crash_trace.log`.
- In debug mode, WhisperX emits per-segment stage markers (`[whisperx-trace] start/asr-done/align-done/text-done`) into `app/src/logs/voice2text.log`.
- In debug mode, WhisperX now also emits alignment micro-benchmark lines (`[align-bench]`) with per-segment elapsed ms, running average, running max, and sample count.
- Alignment device can now be configured in `Settings -> WhisperX Alignment device` and is applied immediately (runtime restart is automatic).
- CLI override is also available: `--whisperx-alignment-device {auto|cpu|cuda}`.
- Windows safety guard: when alignment resolves to `cuda`, runtime now downgrades to `cpu` by default on Windows to avoid known `torchaudio/wav2vec2` access-violation crashes. Override only for diagnostics:
  - **Runtime switch (round 0028):** `whisperx_align_guard` config / `--whisperx-align-guard {safe,unsafe-cuda}` CLI / settings-dialog control. Default `safe` is byte-identical to the previous downgrade behavior. `unsafe-cuda` keeps CUDA alignment on Windows and emits a loud one-line warning in the runtime log; the settings dialog shows an inline warning and a one-click **Revert to safe** button. The guard is consulted first; the env var below stays as a back-compat override; on non-Windows the guard is a no-op.
  - `VOICE2TEXT_WHISPERX_ALLOW_UNSAFE_CUDA_ALIGN=1`
- Environment variable fallback is still supported when config is `auto`:
  - `VOICE2TEXT_WHISPERX_ALIGN_DEVICE=auto` (default heuristic): when ASR uses CUDA and model is `large*`, alignment runs on CPU to reduce VRAM pressure.
  - `VOICE2TEXT_WHISPERX_ALIGN_DEVICE=cpu`: always run alignment on CPU.
  - `VOICE2TEXT_WHISPERX_ALIGN_DEVICE=cuda`: force alignment on CUDA.
- Alignment CUDA stability hardening:
  - Alignment input segments are clamped to valid audio duration.
  - Audio passed into alignment is normalized to contiguous `float32` and cropped to active segment span.
  - CUDA alignment path now runs with explicit sync/inference-mode boundaries to reduce Windows native crash risk.
  - In debug mode, each `latest_segment_stt.wav` write now emits `[segment-artifact] ...` status lines (path/bytes/sample-rate/channels/duration) for pre-crash localization.
  - `alignment_language=follow-source` now falls back to ASR detected language when source language is `auto`/empty, so alignment word timestamps still populate `meta.token_timestamps` and subtitle partial/stable states.

## WhisperX Diarization Behavior

- When `WhisperX Diarization` is enabled, runtime now emits overlay-visible `[download] whisperx-diarization ...` lines for:
  - preparing manifest
  - cache hit
  - ready
  - failure reason
- Diarization predownload now prefers byte-based direct file download progress before snapshot fallback.
- WhisperX direct HF downloader now includes bearer auth headers for gated artifact streaming and size probes, so file-level gated assets can be fetched when token access is granted.
- Diarization bootstrap now pins HF cache to `app/src/models/whisperx/hf-home` (project-local writable path) instead of relying on user profile cache defaults.
- Diarization dependency repos are stored under `app/src/models/whisperx/diarization_deps/` (`_deps` means dependency models referenced by diarization pipeline config).
- On startup, optional diarization/dependency predownload now performs a local readiness check and emits `cache hit` when complete, instead of re-running manifest/download progress every launch.
- Before diarization bootstrap, runtime now auto-cleans stale partial HF cache entries (`*.incomplete`, `*.lock`, `tmp_*`) under the project-local cache to reduce Windows permission/lock collisions.
- If proxy env is set to known dead local placeholders (for example `127.0.0.1:9`), diarization bootstrap now clears those proxy keys in-process and continues without requiring manual `--clear-proxy`.
- If Hugging Face access/token/network is unavailable, runtime emits explicit `[download] ... failed/skipped` reasons and safely falls back to no-diarization text flow.
- If diarization initialization fails once (for example gated repo/token issue), runtime disables diarization re-initialization for the current session to avoid repeated heavy retries/noisy logs.
- When diarization speaker labels are available, transcript output inserts `>>` markers at speaker-turn boundaries so subtitle lines visibly mark speaker changes.
- Speaker-turn markers force a new line on speaker change (instead of continuing on the same line).
- Speaker-turn switching now uses anti-jitter hysteresis (2 consecutive confirmations + minimum 0.18s hold) for both raw diarized text and token-merged subtitles.
- Merge output now prefers `history + raw` overlap composition (while still maintaining token-level state internally), reducing duplicated tail text in rolling subtitles.
- Speaker labels are propagated into `meta.token_timestamps` and rolling subtitle merge re-applies speaker markers on change, preventing marker loss when incremental token-based merge is active.
- Diarization local labels are now mapped through persistent speaker profiles (`SPK_xxx`, embedding centroid matching) so speaker identity remains more stable across rolling STT windows.
- Runtime logs now emit `[speaker-turn] diarization summary` (segment-turn count / marker count / token count / speaker set / pending-switch state) for speaker-turn diagnostics.
- Runtime now skips diarization forward calls for empty/ultra-short/near-silent windows to reduce non-actionable NumPy empty-slice warnings and avoid needless compute.
- Runtime performs one-time CUDA context warmup before first diarization forward call to reduce first-call `cublasLt` fallback warning noise.
- Validation (2026-05-26): foreground 70s probe successfully initialized diarization and observed speaker-turn marker output after access grant + cache/auth fixes.

## Translation Backends (round 0026/0030)

Translation is pluggable and runs through a `TranslationEngine` (`app/src/voice2text/translation/`) that wraps the
selected backend so a slow/hanging backend can never stall the subtitle loop.

- **Backend selection** — `--translation-backend {argos,nllb,llm,cloud}` (default `argos`).
  - `argos`: light offline backend using per-source/target Argos models.
  - `nllb`: offline multilingual backend using a local CTranslate2 NLLB model. It is CPU + int8 by default and
    only uses CUDA if explicitly configured via `translation_nllb_device='cuda'`.
  - `llm`/`cloud`: reserved registry slots that resolve to a disabled stub with a clear "not yet implemented"
    status. No cloud translation service is used.
  An unknown name degrades to `argos` with a warning.
- **NLLB model/dependencies** — install `requirements-translation-extra.txt` for `transformers`/`sentencepiece`.
  The default cache path is `app/src/models/translation/nllb/`. On first use, the backend can download the
  configured PyTorch NLLB model (`facebook/nllb-200-distilled-600M`, roughly 2.4GB) and convert it once into a
  local CTranslate2 int8 folder (roughly 600MB). The intermediate PyTorch download is **removed after a successful
  conversion** (only when the backend downloaded it; a user-supplied local source path is left untouched), so the
  ~2.4GB is transient. Disable auto-conversion with `--no-translation-nllb-auto-convert` when you want to provide a
  pre-converted CT2 model path/repo yourself.
  Download/status messages use byte/MB totals; the local conversion step emits `[convert] nllb: ...` stage markers
  instead of fake percentages. If conversion/download fails, the backend stays disabled and subtitles fall back to
  source-only.
- **Mixed-language routing** — when STT source language is `auto`, the display language hint still uses the stable
  session lock, but translation now routes each window through a corroborated detected source language when the
  text script and stability/token counts agree. Short/noisy windows or script mismatches fall back to the session
  lock, so subtitle source text remains unchanged while genuine code-switch translation improves.
- **Recommended NLLB policy** — because NLLB is CPU-heavy, use a non-zero translation queue for live sessions, for
  example `--translation-queue-max 4 --translation-timeout 8.0`. The default remains `--translation-queue-max 0`
  for back-compatible inline behavior.
- **Off-thread policy** — by default the engine is in **inline passthrough** mode (`--translation-queue-max 0`),
  byte-identical to the historical direct Argos call. Set `--translation-queue-max N` (> 0) to move translation
  onto a bounded background worker with:
  - `--translation-timeout <seconds>` — per-request timeout; on timeout the subtitle emits source-only and the
    loop moves on (never blocked beyond the timeout).
  - `--translation-max-retries <n>` — bounded retry with backoff for a failed request.
  - a **drop-oldest** queue: when full, the oldest pending request is discarded (translation is best-effort).
- **No-translation mode** remains a working fallback; disabling translation or using a disabled backend yields
  source-only subtitles unchanged.
- **Credential redaction** — `redact_config_snapshot` (used by the crash bundle / session manifest) redacts any
  key containing `token` / `api_key` / `secret` / `password`, so future cloud/LLM credentials never reach a bundle.

## Startup Import Behavior

- CLI/bootstrap argument parsing does not import heavy STT runtimes (`torch`, `ctranslate2`) anymore.
- Heavy WhisperX runtime imports are deferred until actual transcriber creation, reducing startup interruptions before capture begins.
- CUDA compatibility alias preparation is now also deferred to STT bootstrap (async) instead of blocking in pre-Qt startup, so first launch reaches UI more reliably.
- Terminal interrupt (`Ctrl+C`) now exits cleanly with code `130` without printing a full Python traceback.

## Additional Docs

- Operator runbook: [docs/build-and-run.md](/D:/Voice2Text/docs/build-and-run.md)
- Repo context (consolidated): [docs/context/CONTEXT.md](/D:/Voice2Text/docs/context/CONTEXT.md)
- AI collaboration workflow: [docs/ai/AI_WORKFLOW.md](/D:/Voice2Text/docs/ai/AI_WORKFLOW.md)
- Changelog: [docs/changelog.md](/D:/Voice2Text/docs/changelog.md)
