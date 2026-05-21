# Python App (MVP)

Windows live subtitle overlay using:
- faster-whisper (STT)
- Vosk (STT, optional)
- sherpa-onnx (STT, optional)
- NVIDIA Riva gRPC (STT, optional)
- FunASR (STT, optional)
- pyaudiowpatch (WASAPI loopback capture)
- PySide6 (semi-transparent rolling subtitles)
- Argos Translate (optional translation stage)
- pycaw (optional app-session awareness)

## Architecture Map

Current Python structure after A->D refactor:

- `src/app/capture/`
	- AudioSource seam for device/session discovery and capture factory.
	- External callers should import from this package instead of `audio_capture.py`.
- `src/app/pipeline/`
	- `subtitle_assembler.py`: incremental subtitle merge + rolling state.
	- `text_delta_logger.py`: STT/translation incremental log delta/chunk emission.
	- `runtime_recovery.py`: whisper CUDA runtime recovery and CPU fallback policy.
- `src/app/settings/`
	- `i18n.py`: locale resources + UI language normalization.
	- `mapping.py`: settings payload validation/mapping to runtime update dict.
	- `schema.py`: provider capability/rule helpers consumed by settings UI.
- `src/app/stt/`
	- `registry.py`: provider alias normalization + shared capability metadata + variant normalization.
	- `factory.py`: provider transcriber creation via dispatch map.
	- `healthcheck.py`: provider diagnostics via dispatch map.

Practical extension points:

- Add a new STT provider:
	- add provider adapter in `src/app/stt/`
	- register aliases/capabilities in `src/app/stt/registry.py`
	- hook provider builder in `src/app/stt/factory.py`
	- hook provider checker in `src/app/stt/healthcheck.py`
- Add new settings rule:
	- add/adjust rule helper in `src/app/settings/schema.py`
	- keep UI rendering in `settings_dialog.py`
	- keep payload validation/mapping in `src/app/settings/mapping.py`

## Recommended Layout (App-First)

```text
app/
  src/                      # Python runtime entry + modules
    app/
      capture/              # capture seam; Python/C++ bridge adapter lives here
    runtime_bin/            # compiled native bridge executable output
    tests/                  # pytest-based automated tests only
  scripts/
    diagnostics/            # manual probe/smoke/runtime-matrix scripts
  native/
    audio_bridge/           # C++ capture bridge source and build script
      src/
      CMakeLists.txt
      build_bridge.ps1
```

Build native bridge into Python runtime bin:

```powershell
cd app\native\audio_bridge
.\build_bridge.ps1
```

Bridge subproject notes:

- `app/native/audio_bridge/README.md`
- `docs/build-and-run.md` (full build + startup workflow)

## 1. Setup

```powershell
cd app
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional STT backends:

```powershell
pip install -r requirements-stt-extra.txt
```

## 2. Download STT model files

The first run of faster-whisper downloads the selected model automatically.
Default STT provider is `whisper` with model `small`.

Model storage convention:
- faster-whisper models: `app/src/models/faster-whisper/<model-name>`
- vosk models: `app/src/models/vosk/<model-name>`
- sherpa-onnx models: `app/src/models/sherpa-onnx/<model-name>`
- funasr models: local path is supported via `--stt-model-path`; remote model IDs are also supported.
- custom path: set `--stt-model-path` to the exact model folder.

Provider-specific expectations:
- whisper: accepts built-in model aliases (`small`, `medium`, etc.) or model path.
- vosk: expects a Vosk model directory (or `app/src/models/vosk/<model-name>`).
- sherpa-onnx: expects a folder with `encoder.onnx`, `decoder.onnx`, `joiner.onnx`, `tokens.txt`.
- riva: does not require local model files; connect to a running Riva server.
- funasr: supports model ID (for example `iic/SenseVoiceSmall`, `FunAudioLLM/Fun-ASR-Nano-2512`) or local model path.

Auto-download behavior for missing models:
- whisper: built-in download path from Hugging Face (can be disabled with `--no-stt-auto-download`).
- vosk: built-in preset aliases can auto-download (`small`, `small-en-us`, `small-zh`).
- sherpa-onnx: built-in preset aliases can auto-download (`small` -> paraformer-zh, `zipformer-zh-en`).
- riva: N/A (server side).
- funasr: model ID is resolved by FunASR runtime via ModelScope/HuggingFace.

### 2.1 Model Download / Conversion Matrix

Whisper (faster-whisper):
- Download source: https://github.com/SYSTRAN/faster-whisper
- Conversion: OpenAI Whisper -> CTranslate2 via `ct2-transformers-converter`
- Conversion doc: https://github.com/SYSTRAN/faster-whisper#model-conversion

Vosk:
- Official model list/download: https://alphacephei.com/vosk/models
- Conversion: generally use Kaldi/Vosk model layout directly, no common one-click converter from Whisper/FunASR.

Sherpa-ONNX:
- Pretrained model index: https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html

NVIDIA Riva:
- Quickstart and deployment: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/resources/riva_quickstart
- Riva uses server-side deployed models, not this app local folder format.

FunASR:
- Project and model zoo: https://github.com/modelscope/FunASR

## 3. (Optional) Install Argos translation packages

Argos Translate requires a language model package to be installed.
If you want English -> Chinese translation, install an Argos package in advance.
If a required package is missing, the app now attempts one-time auto install when translation is enabled.

## 4. Run

```powershell
cd src
python main.py
```

Useful flags:

```powershell
python main.py --list-devices
python main.py --list-app-sessions
python main.py --source-mode loopback --source-devices 12
python main.py --source-mode microphone --source-devices 3
python main.py --source-mode app --app-names msedge.exe

python main.py --stt-provider whisper --model small --stt-variant gpu
python main.py --stt-provider whisperx --model large-v2 --no-whisperx-vad
python main.py --stt-provider whisperx --whisperx-forced-alignment --whisperx-vad
python main.py --stt-provider whisperx --whisperx-diarization --whisperx-hf-token <HF_TOKEN>

python main.py --stt-health-check --stt-health-check-scope all
python main.py --stt-health-check --stt-health-check-scope active --stt-provider whisperx

python main.py --model medium --segment-seconds 6.0 --hop-seconds 1.5 --overlap-merge-method stable-tail
python main.py --source-language zh-hant --cjk-no-space-gap-seconds 0.2

python main.py --preprocess-modules auto
python main.py --preprocess-modules webrtc-ns,webrtc-agc
python main.py --no-preprocess
python main.py --no-vad

python main.py --translate --from-lang en --to-lang zh --bilingual-style stacked
python main.py --translate --from-lang auto --to-lang zh --bilingual-style translation-only

python main.py --ui-language en
python main.py --debug-mode
python main.py --ffmpeg-dll-dir D:\FFmpeg\ffmpeg-7.1.1-full_build-shared\bin
python main.py --cublas-source-dll D:\CUDA\bin\x64\cublas64_13.dll

python scripts/diagnostics/app_source_visual_probe.py --show-sessions --app-names msedge.exe --seconds 30
python scripts/diagnostics/app_source_visual_probe.py --app-names msedge.exe,discord.exe --segment-seconds 6 --hop-seconds 1.5
```

### 4.2 App Source Diagnostic (Visual Probe)

If specified app capture seems silent, run:

```powershell
cd app
python scripts/diagnostics/app_source_visual_probe.py --show-sessions --app-names msedge.exe --seconds 45
```

This probe prints live visual meters for:
- chunk RMS / peak
- segment RMS / peak
- segment gate pass/skip
- app gate pass/drop
- selected/other session peak and match decision reason

### 4.1 Required Post-Fix Validation (60 seconds)

After each code fix, run the provider runtime matrix for 60 seconds per provider:

```powershell
cd app
.\run_post_fix_validation.ps1
```

Optional parameters:

```powershell
.\run_post_fix_validation.ps1 -DurationSeconds 60 -Providers "whisper,whisperx"
```

## 5. whisper_config

File location:
- `app/src/whisper_config.json`

Priority:
- CLI argument > `whisper_config.json` > built-in default.

Supported keys (JSON aliases are accepted):
- `max-context` (`max_context`, `mc`, `-mc`)
- `entropy-thold` (`entropy_thold`)
- `logprob-thold` (`logprob_thold`)
- `no-speech-thold` (`no_speech_thold`)
- `temperature`
- `beam-size` (`beam_size`)
- `best-of` (`best_of`)

Python backend mapping notes:
- `max-context` is mapped to faster-whisper `max_new_tokens`.
- `entropy-thold` is mapped to faster-whisper `compression_ratio_threshold`.

## 6. Overlap Merge Methods (Implementation Detail)

Rolling model:
- The controller keeps two buffers:
	- frozen text: locked historical prefix.
	- active text: mutable latest window region.
- Lock ratio uses `hop_seconds / segment_seconds`, clamped to `[0.05, 0.95]`.

Methods:
- `stable-tail` (recommended)
	- Keep a stable head from previous overlap tail, and replace only the mutable tail using overlap-aware reconciliation.
	- This mirrors mainstream live-caption behavior where interim text gets revised while earlier confirmed text remains stable.
	- Decision logic:
		- lock a prefix by `lock_ratio = clamp(hop_seconds / segment_seconds, 0.05..0.95)`
		- append locked prefix into frozen history
		- reconcile only mutable tail with exact/fuzzy suffix-prefix overlap
		- normalize output with repeated phrase + repeated char-span collapse (better for CJK)
- `commit-on-break`
	- Keep appending by exact overlap during a sentence, then rely on sentence/silence boundaries to freeze text.
	- This mirrors commit/final-result style pipelines used by streaming STT products.
	- Decision logic:
		- merge current window into rolling text via exact overlap
		- do not revise old text in-place
		- when silence/sentence break is detected, freeze current rolling sentence

Legacy aliases (for backward compatibility):
- `replace-window`, `suffix-overlap`, `fuzzy-overlap` -> `stable-tail`
- `append-only` -> `commit-on-break`

## 7. Notes

- CUDA missing `cublas64_12.dll` will now auto-fallback to CPU (`int8`) unless `--no-cpu-fallback` is set.
- The app will also try to create a compatibility alias from the source DLL path (`--cublas-source-dll`) by copying `cublas64_13.dll` to runtime `cublas64_12.dll`.
- CUDA compatibility fallback logic is only applied to the whisper provider.
- Sherpa-ONNX GPU requires CUDAExecutionProvider and a sherpa-onnx build compiled with `-DSHERPA_ONNX_ENABLE_GPU=ON`; otherwise runtime auto-falls back to CPU.
- FunASR GPU requires a CUDA-enabled torch build; if `torch.cuda.is_available()` is false, runtime auto-falls back to CPU.
- STT provider can be selected at startup with `--stt-provider` (`whisper`, `whisperx`, `vosk`, `sherpa-onnx`, `riva`, `funasr`).
- STT runtime variant can be selected with `--stt-variant` (`auto`, `cpu`, `gpu`).
- `--stt-health-check` can validate provider dependencies/model availability/connectivity before running UI.
- Audio preprocessing runs before VAD/STT. `--preprocess-modules auto` tries WebRTC NS/RNNoise if installed, otherwise uses built-in spectral-gate plus adaptive-gain. Explicit modules can include `webrtc-ns`, `webrtc-agc`, `webrtc-aec`, `rnnoise`, `spectral-gate`, and `adaptive-gain`.
- Adaptive VAD is enabled by default and raises/lowers the RMS gate from the observed environment noise floor. Use `--no-adaptive-vad` for the older fixed-threshold behavior.
- Edge resize is enabled on the overlay window (frameless, always on top).
- Status and error messages are persisted under `app/src/logs/voice2text.log` by default.
- `--log-dir` accepts absolute or relative paths; relative paths are resolved from `app/src` (not the shell working directory).
- Transcription and translation outputs are also persisted in log file (`STT:` and `TRANSLATE:` records).
- A tray icon is provided:
	- Left click toggles show/hide.
	- Right click menu: Show / Minimize / Settings / Exit.
- Settings dialog supports runtime updates for STT provider switching (including provider-specific fields), source mode, multi-source selection, source language hint (auto/en/zh-hant/zh-hans/ja/ko), translation toggle + style, translation target language, source/translated text colors, background, opacity, and low-latency parameters.
- Settings dialog supports UI language selection (`zh`/`en`) and source selectors now refresh device/app-session lists on open and via an in-dialog refresh button.
- Subtitle merge strategy is user-selectable (`stable-tail`, `commit-on-break`), with legacy aliases accepted for CLI compatibility.
- Startup and settings-applied status now include effective model information (`model=<name-or-path>`).
- WhisperX provider supports dedicated settings for phoneme-ASR path, forced alignment, internal VAD, and diarization.
- WhisperX model download behavior:
  - STT model: auto-fetched by WhisperX when missing (`--stt-auto-download` path).
  - Alignment model: auto-fetched on first forced-alignment use.
  - Diarization model: auto-fetched on first diarization use, but requires a valid Hugging Face token for pyannote access.
- On Windows, an explicit AppUserModelID is set so taskbar grouping/icon follows the app icon instead of the default Python icon.
- Argos translation modules are now loaded lazily only when translation is enabled, so normal STT startup does not depend on spaCy/Argos import chain.
- Subtitle rendering now keeps a rolling sentence and wraps only when width is exceeded.
- App source mode now prefers C++ Application Loopback Capture for selected process names (`--app-names`).
- MSVC bridge builds now expose process-loopback capability probe correctly (`--probe-process-loopback`) when Windows SDK headers are available.
- If C++ app-mode process-loopback APIs are unavailable in the current bridge build/toolchain, runtime now falls back to Python app-session capture backend automatically.
- If log reason shows bridge does not expose `--probe-process-loopback` (or binary older than source), rebuild/deploy `voice2text_capture_bridge.exe` before retrying app mode.
- If C++ bridge executable is not available, runtime falls back to the legacy Python capture backend.
- Python runtime expects bridge executable at `app/src/runtime_bin/voice2text_capture_bridge.exe` (or `VOICE2TEXT_CPP_CAPTURE_BRIDGE` env override).
- In `--debug-mode`, C++ bridge stream received by Python is mirrored to `app/src/segments/latest_segment_cpp_bridge.wav` for capture-path diagnostics.
- `--list-app-sessions` and settings app list now read from Windows volume mixer sessions (same source as the system mixer panel), rather than full process-details lists.
