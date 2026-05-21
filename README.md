# Voice2Text

Python-first live subtitle overlay project for Windows system audio.

## Runtime Focus
- Primary runtime: `app/` (Python UI + STT/translation pipeline).
- Native capture bridge: `app/native/audio_bridge/` (C++ WASAPI/Application Loopback capture backend for Python).
- External historical workspace: `D:\Voice2Text_cpp\cpp_app` (not the active runtime in this repository).

## Repository Layout
- `app/`: main project (Python runtime + in-repo native bridge project).
- `docs/`: shared architecture/context/ops docs.
- `README.md`: root orientation.
- `task.md`: implementation checklist and progress.
- `agent.md`: architecture decision log.

## Quick Start
1. Build/activate Python venv and install dependencies.
2. Run Python runtime from `app/src`.
3. Build bridge when capture backend changes: `app/native/audio_bridge/build_bridge.ps1`.

Detailed runtime flags and setup are in [app/README.md](/D:/Voice2Text/app/README.md).

## Documentation Index
- Architecture context map: [CONTEXT-MAP.md](/D:/Voice2Text/CONTEXT-MAP.md)
- Shared context: [docs/context/CONTEXT.md](/D:/Voice2Text/docs/context/CONTEXT.md)
- App context: [app/CONTEXT.md](/D:/Voice2Text/app/CONTEXT.md)
- Native bridge notes: [app/native/audio_bridge/README.md](/D:/Voice2Text/app/native/audio_bridge/README.md)
- Build/run guide: [docs/build-and-run.md](/D:/Voice2Text/docs/build-and-run.md)
- Historical change log: [docs/changelog.md](/D:/Voice2Text/docs/changelog.md)
