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
- WhisperX warmup now also preloads diarization pipeline/resources at startup (when diarization is enabled), instead of waiting for first voiced segment.
- WhisperX diarization now reports overlay-visible `[download] whisperx-diarization ...` progress/failure status and adds speaker markers at detected speaker-turn boundaries.
- Third-party library warnings are now captured into `voice2text.log` and the debug window without using Python logging fallback or pre-existing console handlers to spam PowerShell; expected WhisperX VAD no-speech warnings stay out of the console.
- Speaker-turn markers now use `>>` line prefix on detected speaker changes in live subtitles; historical `Sx:`/`>>` traces remain normalization-compatible in merge paths.
- Speaker-turn identity is still stabilized across rolling STT windows by persistent speaker profiles (`SPK_xxx` embedding matching), but on-screen cue style is `>>` boundary markers.
- WhisperX speaker identity is now modularized with selectable embedding backend in Settings (`pyannote` / `speechbrain-ecapa` / `nemo-titanet`).
- WhisperX speaker profile store is now reset on each startup (old fingerprint records are cleared before new session begins).
- Runtime STT is now WhisperX-only. Legacy persisted provider names (`whisper`, `faster-whisper`) are normalized to `whisperx`, and the old project-level pre-STT VAD gate has been removed in favor of WhisperX internal VAD.
- WhisperX align cache path is now normalized into stable subfolders (`align/hf`, `align/torch`, `align/cache`, `align/custom`) shared by realtime runtime and compare harness; startup also cleans stale partial/lock temp artifacts and reports cleanup counters in logs.
- Session transcript export is now supported on runtime stop (`txt/srt/json`) with optional timestamps and speaker fields.
- Runtime now supports imported-audio replay with `--source-mode file --source-file <path>`, feeding decoded media through the same live transcription loop used by loopback/app capture.
- Settings now has an `Import Audio...` button beside `Export Subtitle...`; importing pauses the current live source, replays the selected file through the realtime subtitle pipeline, and continues even if the Settings dialog is accepted. Use the overlay stop/toggle control to interrupt replay; the previous source setting is restored after replay stops.
- Added offline comparison harness for compare-pack audio (`app/src/tests/compare_whisperx_test/input`): incremental project subtitle flow vs direct WhisperX full-file transcription, with normalized CER report output.
- Compare harness now uses project-exported `direct_whisperx.txt` and `realtime_project.txt` as primary compare sources, strips timeline/speaker prefixes, flattens to one-line plain text, and generates a marker-line diff (`.` same / `-` realtime extra / `+` realtime missing) against WhisperX reference.
- Direct/exported transcript cue grouping is now friendlier for CJK/mixed text: token timestamps no longer force a hard split at 4 seconds for Chinese/Japanese context, and ASCII fragments such as `1 1 6` / `l o c a l` are joined as `116` / `local`.
- Compare harness diff unit is now auto-routed by language/script: Chinese/Japanese compares use character-level; other languages use word-level.
- Compare harness now also writes `compare.html` per case with visual annotation (realtime extra = deep-red strikethrough, realtime missing = deep-green insertion).
- Compare harness realtime export now writes one final history-style snapshot (full-audio result) instead of per-window event stacking, so `realtime_project.*` reflects final subtitle state.
- Compare harness realtime phase now uses `source_mode=file` and `TranscriptionLoopEngine`, so startup padding, audio preprocessing, WhisperX transcription metadata enrichment, subtitle merge, and debug payloads follow the main runtime path.
- Compare harness fast-profile realtime output now accumulates runtime snapshots when WhisperX returns sparse/no token metadata, preventing `realtime_project` from collapsing to the final raw tail.
- Compare harness now emits per-case `realtime_debug_trace.jsonl` from the runtime debug-event payloads (disable with `--no-realtime-debug-trace`).
- Realtime transcription loop now injects startup leading-silence padding (`segment - hop`) into rolling windows so early speech can pass through multiple partial/stable merge rounds instead of appearing only once.
- Compare harness char-level HTML diff now preserves spaces explicitly (including `&nbsp;` output for space tokens), so mixed CJK/English snippets remain readable in `compare.html`.
- Compare harness now writes execution log to `compare_run.log` under the selected output root for each run.
- Compare harness now executes direct/realtime phases sequentially with explicit post-phase cleanup (transcriber object teardown + GC/CUDA cache release), so each phase writes artifacts first and then releases memory before the next phase starts.
- Compare harness now emits `[mem] ...` checkpoints (RSS + CUDA allocator stats when available) in run log for memory-peak diagnosis.
- Compare harness suppresses known non-actionable `pyannote/torchcodec` warning spam and, when source language is auto, locks incremental language from direct-pass detection to reduce short-window language-ID warning noise.
- Comparison harness now prints runtime progress (heartbeat + window progress + ETA), supports strict GPU requirement, and can keep WhisperX ASR on CUDA in ASR-only fast profile even when local torch build is CPU-only.
- Speaker-turn markers are now preserved through rolling subtitle merge (token-based merge no longer drops marker lines), and runtime logs include `[speaker-turn] diarization summary` diagnostics.
- Speaker-turn confirmation now persists pending-switch state across chunk boundaries, preventing single-segment rolling windows from getting stuck on the previous speaker.
- Speaker-turn switch decisions now prioritize local diarization labels (`speaker`) over profile labels (`profile_speaker`) to avoid cross-window profile matching hiding real turn boundaries.
- Speaker-turn hysteresis now supports an immediate-confirm path for one clearly long new-speaker segment, and emits `[speaker-turn] switch pending ...` diagnostics when a candidate did not pass thresholds.
- Debug trace now includes per-window `meta.speaker_profile_stats` and concise `[speaker-profile] window summary` logs (assigned/matched/created/skip counts + similarity rows) for speaker-identity troubleshooting.
- Speaker-turn markers are now newline-boundary stabilized during chunk stitching: inline marker joins are normalized to line-start labels, reducing speaker-line jitter in rolling merged output.
- Main overlay subtitle normalization now preserves line breaks (instead of flattening all whitespace), so speaker marker line breaks are rendered as actual new lines on screen.
- WhisperX gated-repo downloads now pass HF bearer auth on direct file streaming; diarization bootstrap uses project-local HF cache (`app/src/models/whisperx/hf-home`) plus startup cleanup of stale partial lock/temp files.
- WhisperX diarization bootstrap now auto-bypasses known invalid local proxy placeholders (`127.0.0.1:9` / `localhost:9`) in-process.
- WhisperX `alignment_language=follow-source` now falls back to detected ASR language when source language is `auto`/empty, preventing missing word timestamps in debug `meta`.
- C++ bridge debug segment mirror (`app/src/segments/latest_segment_cpp_bridge.wav`) now follows runtime `segment_seconds` and is explicitly a rolling-tail window.
- WhisperX download progress display is standardized: known totals must show `current/total MB` with `%`; unknown totals must show bytes only (no fake intermediate `100%` lines).
- Torch-hub/torchaudio alignment downloads are now bridged into app progress logs with remote-size probing, so `download.pytorch.org` model fetches can also report `x/y MB` totals when available.
- For the same torch-hub transfer, generic fallback progress is now suppressed through completion to avoid post-`100%` duplicate lines.
- Settings now provide a persistent `Audio preprocess` toggle; pre-STT VAD gating has been retired because WhisperX internal VAD is the only active speech gate.
- WhisperX optional diarization dependency predownload now short-circuits on local cache readiness (`cache hit`) to avoid repeated startup re-download progress noise.
- WhisperX debug trace mode now includes alignment micro-benchmark logs (`[align-bench]`) for direct CPU/CUDA timing comparison on this machine.
- WhisperX alignment auto mode now reuses existing language-scoped cache folders (for example `align/hf/zh`) before triggering fallback downloads.
- WhisperX alignment cache routing now uses language-scoped folders by mode: `follow-source` uses STT source language folder (`align/hf/{source_lang}`), and `auto` uses detected alignment language folder (`align/hf/{detected_lang}`).
- This language-folder routing also applies when `whisperx_alignment_model` is set to an explicit HF repo id; repo slug is no longer the default cache folder for `follow-source/auto`.
- WhisperX diarization speaker markers now break line on speaker change and merged subtitle dedupe is more tolerant to transient speaker-label jitter.
- Speaker-turn switch now includes anti-jitter hysteresis (2 consecutive confirmations + minimum 0.18s hold), and CJK merge output no longer collapses `>>` newline markers.
- Rolling subtitle composition now prefers `history + raw` overlap output (token state remains internal), reducing duplicated tail text during incremental updates.
- Foreground diagnostics are available for diarization readiness/probe under `app/scripts/diagnostics/whisperx_diarization_readiness_check.py` and `app/scripts/diagnostics/whisperx_diarization_stability_test.py`.
- Segment-arrival status-routing regression check is available at `app/scripts/diagnostics/segment_arrival_regression_test.py`.
- Settings transcript-export controls are moved off the main form: use the bottom-left `Export Subtitle...` button (next to `Reset defaults`) to open export settings first, then choose save path/type (`txt`/`srt`/`json`).
- Settings dialog main form is now split into left/right columns to reduce vertical crowding and avoid clipping on shorter screens.
- Settings dialog depends on `app.settings.schema` for STT variant choices; tray settings startup verifies this path by instantiating `SettingsDialog` in the project venv.
- Python tests should be run with the project virtualenv: from `app/src`, use `..\.venv\Scripts\python.exe -m unittest discover -s tests`.
- `app/src/logs/python_crash_trace.log` now records session timestamps, uncaught Python/Qt callback exceptions with timestamps, and native-crash heartbeat `last_alive` markers before faulthandler dumps.
- Translation runtime path now accepts per-window source-language hint (`source_code`) without repeated Argos module import in hot path, avoiding long stalls during runtime language-route updates.

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
- Test-data compare workflow: [app/src/tests/compare_whisperx_test/WORKFLOW.md](/D:/Voice2Text/app/src/tests/compare_whisperx_test/WORKFLOW.md)
- Bridge guide: [app/native/audio_bridge/README.md](/D:/Voice2Text/app/native/audio_bridge/README.md)
- Context map: [CONTEXT-MAP.md](/D:/Voice2Text/CONTEXT-MAP.md)
- Changelog: [docs/changelog.md](/D:/Voice2Text/docs/changelog.md)
