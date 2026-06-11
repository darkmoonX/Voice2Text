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
- Speaker-turn markers now default to `[spk_000]` style on detected speaker changes in live subtitles; historical `Sx:`/`>>` traces remain normalization-compatible in merge paths.
- Speaker-turn display now prioritizes local diarization labels for visible subtitle markers; speaker-profile matching remains available for diagnostics/identity stats but no longer overwrites the visible realtime speaker label.
- WhisperX speaker identity is now modularized with selectable embedding backend in Settings (`pyannote` / `speechbrain-ecapa` / `nemo-titanet`).
- If `pyannote/embedding` returns a Hugging Face gated 403 for speaker-profile embedding, runtime falls back to SpeechBrain ECAPA automatically; the main diarization pipeline can still be ready while this optional profile backend is denied.
- Settings update logs redact `whisperx_hf_token`; do not rely on runtime logs to recover the token value.
- WhisperX speaker profile store is now reset on each startup (old fingerprint records are cleared before new session begins).
- Runtime STT is now WhisperX-only. Legacy persisted provider names (`whisper`, `faster-whisper`) are normalized to `whisperx`, and the old project-level pre-STT VAD gate has been removed in favor of WhisperX internal VAD.
- WhisperX align cache path is now normalized into stable subfolders (`align/hf`, `align/torch`, `align/cache`, `align/custom`) shared by realtime runtime and compare harness; startup also cleans stale partial/lock temp artifacts and reports cleanup counters in logs.
- Session transcript export is now supported on runtime stop (`txt/srt/json`) with optional timestamps and speaker fields; manual export covers the current runtime interval from the last `start()` to the latest pause/stop, or to the export moment if runtime is still running.
- Runtime now supports imported-audio replay with `--source-mode file --source-file <path>`, feeding decoded media through the same live transcription loop used by loopback/app capture.
- Settings now has an `Import Audio...` button beside `Export Subtitle...`; importing pauses the current live source, replays the selected file through the realtime subtitle pipeline, and continues even if the Settings dialog is accepted. Use the overlay stop/toggle control to interrupt replay; the previous source setting is restored after replay stops.
- Added offline comparison harness for compare-pack audio (`app/src/tests/compare_whisperx_test/input`): incremental project subtitle flow vs direct WhisperX full-file transcription, with normalized CER report output.
- Compare harness now uses project-exported `direct_whisperx.txt` and `realtime_project.txt` as primary compare sources, normalizes speaker prefixes to `[spk_000]`, collapses repeated same-speaker markers, flattens to one-line text, and generates a marker-line diff (`.` same / `-` realtime extra / `+` realtime missing) against WhisperX reference.
- Direct/exported transcript cue grouping is now friendlier for CJK/mixed text: token timestamps no longer force a hard split at 4 seconds for Chinese/Japanese context, and ASCII fragments such as `1 1 6` / `l o c a l` are joined as `116` / `local`.
- Live merged Chinese subtitles use phrase spacing: token timestamp gaps above `cjk_no_space_gap_seconds` (default `0.2s`) insert one visible space; when aligned gaps are too small or timestamps cannot be matched, the CJK max-length fallback inserts spaces only and does not create artificial line breaks.
- Debug trace jsonl is compacted: full token timestamp/state arrays are replaced by summary counts, samples, and `assembler_summary.cjk_spacing` diagnostics.
- Compare harness diff unit is now auto-routed by language/script: Chinese/Japanese compares use character-level; other languages use word-level.
- Compare harness now also writes `compare.html` per case with visual annotation (realtime extra = deep-red strikethrough, realtime missing = deep-green insertion).
- Compare harness realtime export writes only the final history-style snapshot to `realtime_project.*`; overlapping per-window token metadata remains in `realtime_debug_trace.jsonl` for diagnostics so rolling-window tokens cannot stack duplicate CJK characters in exported subtitles.
- Compare harness realtime phase now uses `source_mode=file` and `TranscriptionLoopEngine`, so startup padding, audio preprocessing, WhisperX transcription metadata enrichment, subtitle merge, and debug payloads follow the main runtime path.
- Compare harness realtime phase now mirrors main WhisperX warmup before file replay (`prewarm` plus one second of silent audio) so first-window progress timing is not dominated by alignment/diarization/profile model initialization.
- Compare harness realtime language now stays in main-runtime parity by default: `--language auto` remains auto for realtime/file replay; use `--lock-realtime-language-from-direct` only when you intentionally want direct-pass language detection to stabilize compare output.
- Final snapshot transcript export now collapses consecutive same-speaker lines into one cue before assigning fallback timestamps, so visual line wraps do not create artificial fixed-duration subtitle rows.
- Compare harness realtime speaker normalization now applies session reconciliation and direct-speaker-count capping in the export post-process, avoiding a double-renumbering path that previously left noisy realtime speakers visible.
- Compare harness fast-profile realtime output now accumulates runtime snapshots when WhisperX returns sparse/no token metadata, preventing `realtime_project` from collapsing to the final raw tail.
- Compare harness fast profile now keeps forced alignment enabled; the intended main profile difference is diarization off for `fast` and diarization on for `accurate`.
- Compare harness accurate reports now include speaker-sequence diagnostics (`speaker_compare` in `compare.json`, `[speaker_compare]` in `compare.txt`, and speaker counts/error rate in `summary.txt/json`) to spot realtime speaker over-splitting against the direct WhisperX reference; fast profile marks speaker compare as disabled because diarization is intentionally off.
- Compare harness now emits per-case `realtime_debug_trace.jsonl` from the runtime debug-event payloads (disable with `--no-realtime-debug-trace`).
- Realtime transcription loop now injects startup leading-silence padding (`segment - hop`) into rolling windows so early speech can pass through multiple partial/stable merge rounds instead of appearing only once.
- Compare harness char-level HTML diff now preserves spaces explicitly (including `&nbsp;` output for space tokens), so mixed CJK/English snippets remain readable in `compare.html`; identical realtime text is explicitly rendered as normal-weight black, while only missing reference text is bold green.
- Compare harness HTML diff now groups adjacent same/extra/missing runs into one span per run, preserves exported subtitle cue line breaks, and renders accurate-profile speaker-sequence comparison directly in `compare.html`.
- Compare harness keeps per-window runtime details in `realtime_debug_trace.jsonl`, and `realtime_project.*` now avoids synthetic evenly-spaced final-snapshot cues when token timestamps are available.
- Transcript cue export now uses speaker changes and pauses greater than 2 seconds as line/cue boundaries; punctuation and length-only splitting are disabled. Each speaker-enabled cue line starts with normalized `[spk_000]` labels.
- Compare/export speaker remapping now performs a second same-speaker cue coalescing pass for `txt/srt/json`, so profile-cap remaps do not create repeated `[spk_000]` line breaks when there is no speaker change and no pause greater than 2 seconds.
- Transcript export token speaker labels prefer profile speaker identity for cross-window/chunk stability, while live speaker-turn detection still prefers local diarization labels so realtime turn boundaries are not hidden by profile smoothing.
- Direct WhisperX compare exports now run speaker-profile reconciliation after all direct chunks complete, remapping highly similar split profiles before writing `direct_whisperx.*`; this reduces repeated `[spk_000]`/reset artifacts caused by chunk-local diarization labels.
- Compare harness speaker labels are now renumbered by first visible occurrence for each exported artifact, so WhisperX Reference and Realtime both start at `[spk_000]`; known-speaker-count clips can use `--speaker-label-max-speakers N` to collapse short/noisy profile splits to the nearest dominant profiles.
- Compare harness now separates online speaker matching from final session reconciliation: `--speaker-profile-match-threshold` controls rolling-window matching, while `--speaker-profile-reconcile-threshold` (default `0.52`) merges similar profiles after each full direct/realtime pass. Accurate realtime exports are auto-capped to the direct reference speaker count unless `--speaker-label-max-speakers N` is explicitly provided.
- Compare harness `--direct-chunk-seconds 0` means a single full-file WhisperX pass. Use positive values such as `30` or `120` for project-side direct chunking.
- Compare harness direct auto-language mode now supports `--direct-language-subchunk-seconds` (default `30`) so long direct chunks can be internally split into smaller language-routing subchunks for mixed Chinese/English material.
- Realtime auto-language handling keeps WhisperX ASR in auto-detect mode while using a rolling language lock only as a downstream display/translation hint, so durable language switches can still be detected after startup.
- Realtime speaker markers and token transcript export now smooth tiny A-B-A speaker blips before visible cue/marker rendering, reducing 2-3 character false speaker splits while preserving raw diarization metadata in debug traces.
- Realtime speaker-profile matching now stages short new-speaker embeddings in an in-memory candidate pool before creating new `SPK_xxx` identities, reducing profile explosions from one-off diarization fragments while preserving raw local labels in debug traces.
- Realtime speaker marker/cue smoothing also folds tiny trailing speaker islands into the previous non-tiny speaker span, so 1-2 character false turns at window edges do not create new visible `[spk_###]` lines.
- Compare harness now writes execution log to `compare_run.log` under the selected output root for each run.
- Compare harness now executes direct/realtime phases sequentially with explicit post-phase cleanup (transcriber object teardown + GC/CUDA cache release), so each phase writes artifacts first and then releases memory before the next phase starts.
- Compare harness now emits `[mem] ...` checkpoints (RSS + CUDA allocator stats when available) in run log for memory-peak diagnosis.
- Compare harness suppresses known non-actionable `pyannote/torchcodec` warning spam; realtime language locking from direct-pass detection is opt-in via `--lock-realtime-language-from-direct`.
- Comparison harness now prints text progress bars with audio-seconds processed, speed, ETA, and latest realtime window timing; it supports strict GPU requirement and can keep WhisperX ASR on CUDA in ASR-only fast profile even when local torch build is CPU-only.
- Debug-mode main runtime and compare realtime replay now emit `[window-timing]` status lines per rolling window with raw artifact, preprocess, STT artifact, transcribe, language route, timestamp enrich, merge, payload, and total seconds for main-vs-replay latency comparison.
- Debug-mode runtime also emits `[whisperx-timing]` provider substep lines (`asr`, alignment model/load/run, diarization load/run/assign, speaker-profile, metadata) and `[merge-timing]` subtitle assembly substep lines (state update, history render, spacing, overlap, state sizes) for bottleneck diagnosis.
- Comparison harness case reports now include `speaker_profile_diagnostics` in `compare.json` and `[speaker_profile_diagnostics.*]` sections in `compare.txt`, listing profile duration, sample count, weight, and observed local labels for direct and realtime phases.
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
- On Windows, if `voice2text.log` cannot rotate because another process holds the file, logging falls back to `voice2text.<date>.pid<PID>.log` to avoid PowerShell `--- Logging error ---` spam.
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
