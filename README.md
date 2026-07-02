# Voice2Text

Python-first live subtitle overlay for Windows system audio: a Python UI + STT/translation
pipeline (`app/`) driven over a native C++ audio-capture bridge.

```
AudioSource -> SpeechRecognizer (WhisperX) -> (optional) TranslationEngine (Argos) -> SubtitleOverlay (PySide6)
```

## Runtime Focus

- Primary runtime: `app/` (Python UI + STT/translation pipeline).
- Native capture bridge: `app/native/audio_bridge/` (C++ WASAPI / Application Loopback backend).
- STT is **WhisperX-only**; legacy provider names (`whisper`, `faster-whisper`) are normalized to `whisperx`.
- Supports live capture (`loopback`/`microphone`/`app`) and imported-audio replay (`file`) through the same pipeline.

The granular behavior/decision log that previously lived here now lives in:
- `ARCHITECTURE_NOTES.md` — active architecture decisions and runtime constraints.
- `docs/changelog.md` — dated change timeline.
- `docs/history/task-archive.md` — full historical task log.

## Repository Layout

- `app/` — main runtime project (Python `app/src`, C++ bridge `app/native/audio_bridge`).
- `docs/` — documentation:
  - `docs/ai/` — Claude × Codex collaboration workflow and per-role rules.
  - `docs/tasks/` — per-round task specs.
  - `docs/context/` — consolidated repo context.
  - `docs/agents/` — issue-tracker / triage / git-workflow conventions.
  - `docs/history/` — archived task log and history.
- `task.md` — pointer to the task backlog (`docs/tasks/BACKLOG.md`).
- `ARCHITECTURE_NOTES.md` — active architecture constraints and decisions.
- `CLAUDE.md` / `AGENTS.md` — guidance for AI agents working in this repo.

## Quick Start

1. Set up the Python venv in `app/`.
2. Build the capture bridge in `app/native/audio_bridge/`.
3. Run the app from `app/src` (`python main.py`).

Detailed steps: [docs/build-and-run.md](/D:/project/Voice2Text/docs/build-and-run.md).

## Documentation

- Documentation index: [docs/README.md](/D:/project/Voice2Text/docs/README.md)
- App runtime guide: [app/README.md](/D:/project/Voice2Text/app/README.md)
- AI collaboration workflow: [docs/ai/AI_WORKFLOW.md](/D:/project/Voice2Text/docs/ai/AI_WORKFLOW.md)
- Repo context: [docs/context/CONTEXT.md](/D:/project/Voice2Text/docs/context/CONTEXT.md)
- Architecture constraints: [ARCHITECTURE_NOTES.md](/D:/project/Voice2Text/ARCHITECTURE_NOTES.md)
- Compare workflow: [app/src/tests/compare_whisperx_test/WORKFLOW.md](/D:/project/Voice2Text/app/src/tests/compare_whisperx_test/WORKFLOW.md)
- Bridge guide: [app/native/audio_bridge/README.md](/D:/project/Voice2Text/app/native/audio_bridge/README.md)
- Changelog: [docs/changelog.md](/D:/project/Voice2Text/docs/changelog.md)
