# Voice2Text App Runtime

Windows live subtitle overlay runtime (Python main process + C++ capture bridge).

## Stack

- UI: `PySide6`
- Capture: Python capture adapters + C++ bridge (`WASAPI loopback`, `Application Loopback Capture`)
- STT providers: `whisper`, `whisperx`, optional `vosk` / `sherpa-onnx` / `riva` / `funasr`
- Optional translation: `Argos Translate`

## Runtime Structure

```text
app/
  src/
    app/
      capture/      # capture factory + cpp bridge adapter
      pipeline/     # subtitle assembler, delta logger, runtime recovery
      settings/     # i18n + mapping + schema
      stt/          # provider registry/factory/health-check/providers
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

Optional provider dependencies:

```powershell
pip install -r requirements-stt-extra.txt
```

## Build Native Capture Bridge

```powershell
cd app\native\audio_bridge
.\build_bridge.ps1
```

Output executable:
- `app/src/runtime_bin/voice2text_capture_bridge.exe`

Advanced build parameters and MSVC/MinGW presets:
- [app/native/audio_bridge/README.md](/D:/Voice2Text/app/native/audio_bridge/README.md)

## Run

```powershell
cd app\src
python main.py
```

### Settings Persistence

- Runtime settings changed from tray `Settings` are saved to `app/src/runtime_settings.json`.
- On next launch, the app restores these settings before capture/STT startup.
- The settings dialog now includes a bottom-left `Reset defaults` button that resets all visible options back to built-in defaults (apply with `OK`).

### Common Commands

```powershell
python main.py --list-devices
python main.py --list-app-sessions
python main.py --source-mode app --app-names msedge.exe

python main.py --stt-provider whisper --model small --stt-variant gpu
python main.py --stt-provider whisperx --model large-v2 --no-whisperx-vad
python main.py --stt-provider whisperx --model large-v2 --segment-seconds 6.0 --hop-seconds 1.5
python main.py --stt-provider whisperx --model large-v2 --whisperx-alignment-device cpu

python main.py --source-language zh-hant --cjk-no-space-gap-seconds 0.2
python main.py --debug-mode
```

### Optional Runtime Parameters

```powershell
python main.py --ffmpeg-dll-dir D:\FFmpeg\ffmpeg-7.1.1-full_build-shared\bin
python main.py --cublas-source-dll D:\CUDA\bin\x64\cublas64_13.dll
python main.py --ui-language en
```

### Whisper Model Download Behavior

- Whisper model auto-download now uses direct HTTP streaming (byte-based progress) by default.
- Progress is emitted as `[download] ...` status lines and is visible in overlay + `app/src/logs/voice2text.log`.
- Download progress now preflights remote file sizes (when available) and reports `current/total MB` with `%` bar.
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

CUDA/GPU telemetry in debug mode:

- When `--debug-mode` and CUDA device are active, runtime emits periodic `[gpu-telemetry]` lines into `app/src/logs/voice2text.log`.
- Reported metrics include:
  - PyTorch process memory: `alloc`, `reserved`, `max_alloc`, `total`
  - Device-level usage from `nvidia-smi`: `util`, `mem_util`, `vram used/total`

Debug window log visibility:

- Main overlay now only shows curated important statuses (startup/capture/runtime-critical/downloading), while noisy diagnostics stay in logs.
- In debug mode, debug window loads recent runtime log history from `app/src/logs/voice2text.log*` and keeps streaming all new logger lines in real time.
- Debug window still writes structured event traces to `debug_trace_YYYYMMDD.jsonl`.

## WhisperX Warmup Behavior

- When `--stt-provider whisperx` and forced alignment are enabled, startup warmup now preloads the resolved alignment model (for example Chinese alignment for `zh-hant`) before first live speech.
- This avoids first-utterance stalls caused by deferred `alignment model loading`.

## WhisperX Crash Diagnostics

- Runtime now enables Python `faulthandler`; native crash traces are written to `app/src/logs/python_crash_trace.log`.
- In debug mode, WhisperX emits per-segment stage markers (`[whisperx-trace] start/asr-done/align-done/text-done`) into `app/src/logs/voice2text.log`.
- Alignment device can now be configured in `Settings -> WhisperX Alignment device` and is applied immediately (runtime restart is automatic).
- CLI override is also available: `--whisperx-alignment-device {auto|cpu|cuda}`.
- Windows safety guard: when alignment resolves to `cuda`, runtime now downgrades to `cpu` by default on Windows to avoid known `torchaudio/wav2vec2` access-violation crashes. Override only for diagnostics:
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

## Startup Import Behavior

- CLI/bootstrap argument parsing does not import heavy STT runtimes (`torch`, `ctranslate2`) anymore.
- Heavy provider imports are deferred until actual transcriber creation, reducing startup interruptions before capture begins.
- CUDA compatibility alias preparation is now also deferred to STT bootstrap (async) instead of blocking in pre-Qt startup, so first launch reaches UI more reliably.
- Terminal interrupt (`Ctrl+C`) now exits cleanly with code `130` without printing a full Python traceback.

## Additional Docs

- Operator runbook: [docs/build-and-run.md](/D:/Voice2Text/docs/build-and-run.md)
- Shared context: [docs/context/CONTEXT.md](/D:/Voice2Text/docs/context/CONTEXT.md)
- App context: [app/CONTEXT.md](/D:/Voice2Text/app/CONTEXT.md)
- Changelog: [docs/changelog.md](/D:/Voice2Text/docs/changelog.md)
