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

## Agent Skill Configuration
- Agent skills now read repository conventions from `docs/agents/`.
- Issue tracker is configured as GitHub Issues (`gh` CLI workflow).
- Triage label vocabulary uses defaults: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`.
- Domain-doc layout is configured as multi-context (`CONTEXT-MAP.md` + per-context docs).
- Python logging path handling was hardened: default and relative `--log-dir` now resolve under `python_app/src` to avoid writing logs to repo-root `logs/` by accident.
- Python architecture seam update: runtime callers now consume audio source discovery/factory via `python_app/src/app/capture/` instead of importing `audio_capture.py` directly.
- Python pipeline architecture update: subtitle incremental assembly is now encapsulated in `python_app/src/app/pipeline/subtitle_assembler.py`, reducing controller coupling.
- Python pipeline architecture update: controller now delegates incremental transcript logging and whisper runtime recovery to `app/pipeline/text_delta_logger.py` and `app/pipeline/runtime_recovery.py`.
- Settings architecture update: locale strings were extracted from `settings_dialog.py` to `python_app/src/app/settings/i18n.py` with explicit UI language normalization.
- Settings architecture update: dialog payload validation and RuntimeConfig update mapping now lives in `python_app/src/app/settings/mapping.py`.
- Settings architecture update: provider capability/schema rules are centralized in `python_app/src/app/settings/schema.py` and consumed by settings dialog.
- STT architecture update: provider alias normalization is centralized in `python_app/src/app/stt/registry.py` and reused by both factory and health-check flows.
- STT architecture update: provider-specific construction and health-check routing now use dispatch maps in factory/healthcheck to reduce branching and simplify provider extension.
- STT/settings architecture update: provider capabilities (GPU variant + source-language hint support) and variant normalization now share a single source of truth in `python_app/src/app/stt/registry.py`.
- Added low-risk architecture regression tests: `python_app/src/tests/test_stt_registry_schema.py` validates shared provider alias/capability/variant/schema consistency.
- Python subtitle merge strategies were streamlined to two mainstream modes (`stable-tail`, `commit-on-break`) with backward-compatible alias mapping for legacy CLI values.
- Python startup/settings status now includes effective model output (`model=<name-or-path>`).
- Python Windows runtime now sets explicit AppUserModelID so taskbar icon/grouping matches app tray icon instead of default Python host icon.
- Python translation adapter now lazy-loads Argos dependencies only when translation is enabled, preventing non-translation startup from being blocked by Argos/spaCy import side effects.
- Python subtitle merge was hardened for English/CJK overlap windows (safer stable-tail boundaries + stronger repeated-tail collapse).
- Python app-mode capture now enforces stricter target-dominance gating and auto-prefers VB-CABLE loopback in app mode (when target apps are set and no source device is specified).
- C++ path resumed and advanced toward Python parity: overlap merge strategy is now normalized to `stable-tail` / `commit-on-break` with legacy alias compatibility and stronger mixed-language de-dup behavior.
- C++ runtime status now includes effective model label after settings apply/restart.
- C++ architecture improvement: `RuntimeSettings` + merge-method normalization extracted into `cpp_app/src/runtime/runtime_settings.*` to decouple main runtime rules from settings UI.
- C++ second-round architecture follow-up: `main.cpp` audio/session discovery moved into `cpp_app/src/audio/discovery.*`, and runtime restart-decision mapping extracted into `cpp_app/src/runtime/runtime_update.*`.
- C++ app-mode isolation policy is now stricter: when process-loopback is unavailable, runtime prefers VB-CABLE endpoint; if no strict-isolation source exists, app capture aborts rather than falling back to mixed default loopback.
- C++ third-round architecture follow-up: settings value mapping moved into `cpp_app/src/settings/mapping.*`; settings i18n resource module introduced at `cpp_app/src/settings/i18n.*` as the next split seam for dialog localization cleanup.
- C++ fourth-round follow-up: settings dialog now consumes `settings/i18n.*` zh/en labels and `RuntimeSettings.uiLanguage` is persisted via settings mapping.
- C++ build follow-up: fixed `whisper_engine.cpp` compile-time `std::min` type ambiguity (Qt `qsizetype` with `int`) and validated successful `voice2text_cpp` Release build under elevated tool execution.
- C++ build script update: cpp_app/build.ps1 now supports explicit CPU mode (-CpuOnly) and deploys runtime dependencies for both multi-config (Release) and single-config fallback (*-nmake) output layouts.
- C++ settings enhancement: STT model is now switchable from settings (auto-discovered model files + manual path), and VAD controls are exposed (enabled, daptive, ms-threshold).
- C++ overlay enhancement: mouse wheel can browse older subtitle history, with a down-arrow jump button to return to latest lines instantly.
- Python overlay enhancement: jump-to-bottom button now force-resets both history and arrival scroll offsets for immediate latest-position recovery.
- Stable-tail merge hardening: lock/preserve ratio now adapts to overlap confidence to reduce bad early-token influence from new segment windows.
