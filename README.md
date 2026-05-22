# Voice2Text

Python-first live subtitle overlay project for Windows system audio.

## Runtime Focus

- Primary runtime: `app/` (Python UI + STT/translation pipeline).
- Native capture bridge: `app/native/audio_bridge/` (C++ WASAPI/Application Loopback backend for Python).
- Historical external workspace: `D:\Voice2Text_cpp` (not the active runtime in this repo).

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
