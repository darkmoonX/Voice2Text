# Voice2Text

Python-first live subtitle overlay project for Windows system audio.

## Runtime Focus

- Primary runtime: `app/` (Python UI + STT/translation pipeline).
- Native capture bridge: `app/native/audio_bridge/` (C++ WASAPI/Application Loopback backend for Python).
- Historical external workspace: `D:\Voice2Text_cpp` (not the active runtime in this repo).
- WhisperX native-crash diagnostics: Python faulthandler writes traces to `app/src/logs/python_crash_trace.log` (details in `app/README.md`).
- WhisperX alignment device now supports runtime selection (`auto/cpu/cuda`) from Settings or CLI (`--whisperx-alignment-device`).
- WhisperX STT download now uses byte-based direct progress by default; snapshot fallback is opt-in via `VOICE2TEXT_WHISPERX_USE_SNAPSHOT=1`.
- WhisperX alignment safety guard now downgrades Windows CUDA alignment to CPU by default to avoid known `torchaudio/wav2vec2` access violations (override with `VOICE2TEXT_WHISPERX_ALLOW_UNSAFE_CUDA_ALIGN=1`).
- Segment-arrival status-routing regression check is available at `app/scripts/diagnostics/segment_arrival_regression_test.py`.

## Repository Layout

- `app/`: main runtime project.
- `docs/`: runbooks, context docs, and changelog.
- `task.md`: active implementation checklist.
- `ARCHITECTURE_NOTES.md`: active architecture constraints and decisions.

## Quick Start

1. Setup Python venv in `app/`.
2. Build capture bridge in `app/native/audio_bridge/`.
3. Run app from `app/src`.

Detailed steps are in [docs/build-and-run.md](/D:/Voice2Text/docs/build-and-run.md).

## Documentation

- Documentation index: [docs/README.md](/D:/Voice2Text/docs/README.md)
- App runtime guide: [app/README.md](/D:/Voice2Text/app/README.md)
- Bridge guide: [app/native/audio_bridge/README.md](/D:/Voice2Text/app/native/audio_bridge/README.md)
- Context map: [CONTEXT-MAP.md](/D:/Voice2Text/CONTEXT-MAP.md)
- Changelog: [docs/changelog.md](/D:/Voice2Text/docs/changelog.md)
