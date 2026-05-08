# Voice2Text

Dual-path live subtitle overlay project for Windows system audio:
- Python implementation: faster-whisper + pyaudiowpatch + PySide6 + Argos Translate
- C++ implementation: whisper.cpp + Qt + CUDA/cuBLAS

## Goals
- Semi-transparent rolling subtitles over any desktop app or browser.
- Capture web/app playback audio from loopback output.
- Support microphone capture and source selection.
- Keep translation as an optional stage that can be enabled later.

## Repository Layout
- `agent.md`: implementation intent and architecture guide.
- `task.md`: milestone checklist.
- `python_app/`: functional Python MVP.
- `cpp_app/`: C++/Qt + whisper.cpp scaffold.

## Quick Start
1. Start with the Python MVP for fastest iteration.
2. Move to C++ path for lower-level control and tighter deployment.
3. Reuse the same flow: capture -> STT -> optional translation -> rolling overlay.

Current highlights:
- Python: tray icon menu, redesigned settings dialog (multi-source selector + translation toggle/style + source language hint), source/translated text color split, edge-resizable overlay with width-based auto-wrap, status/error/STT/translation logs, CUDA->CPU fallback, low-latency streaming with incremental de-dup.
- C++: Qt overlay with edge resize + source/translated rendering style + width-based auto-wrap, source mode option wiring (`loopback`/`microphone`/`app`), source language + segment/hop CLI options, incremental transcript de-dup, status/error/STT/translation logging, and D-drive-aware build script with MinGW fallback when MSVC toolchain is unavailable.

Detailed setup instructions are inside each subfolder README.
