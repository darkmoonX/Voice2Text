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

python main.py --source-language zh-hant --cjk-no-space-gap-seconds 0.2
python main.py --debug-mode
```

### Optional Runtime Parameters

```powershell
python main.py --ffmpeg-dll-dir D:\FFmpeg\ffmpeg-7.1.1-full_build-shared\bin
python main.py --cublas-source-dll D:\CUDA\bin\x64\cublas64_13.dll
python main.py --ui-language en
```

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

CUDA/GPU telemetry in debug mode:

- When `--debug-mode` and CUDA device are active, runtime emits periodic `[gpu-telemetry]` lines into `app/src/logs/voice2text.log`.
- Reported metrics include:
  - PyTorch process memory: `alloc`, `reserved`, `max_alloc`, `total`
  - Device-level usage from `nvidia-smi`: `util`, `mem_util`, `vram used/total`

## WhisperX Warmup Behavior

- When `--stt-provider whisperx` and forced alignment are enabled, startup warmup now preloads the resolved alignment model (for example Chinese alignment for `zh-hant`) before first live speech.
- This avoids first-utterance stalls caused by deferred `alignment model loading`.

## Additional Docs

- Operator runbook: [docs/build-and-run.md](/D:/Voice2Text/docs/build-and-run.md)
- Shared context: [docs/context/CONTEXT.md](/D:/Voice2Text/docs/context/CONTEXT.md)
- App context: [app/CONTEXT.md](/D:/Voice2Text/app/CONTEXT.md)
- Changelog: [docs/changelog.md](/D:/Voice2Text/docs/changelog.md)
