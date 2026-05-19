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

## 1. Setup

```powershell
cd python_app
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
- faster-whisper models: `python_app/src/models/faster-whisper/<model-name>`
- vosk models: `python_app/src/models/vosk/<model-name>`
- sherpa-onnx models: `python_app/src/models/sherpa-onnx/<model-name>`
- funasr models: local path is supported via `--stt-model-path`; remote model IDs are also supported.
- custom path: set `--stt-model-path` to the exact model folder.

Provider-specific expectations:
- whisper: accepts built-in model aliases (`small`, `medium`, etc.) or model path.
- vosk: expects a Vosk model directory (or `python_app/src/models/vosk/<model-name>`).
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
python main.py --stt-health-check --stt-health-check-scope all
python main.py --stt-health-check --stt-health-check-scope active --stt-provider sherpa-onnx
python main.py --stt-provider whisper --model small
python main.py --stt-provider whisperx --model small --stt-auto-download
python main.py --stt-provider whisperx --model small --whisperx-forced-alignment --whisperx-vad
python main.py --stt-provider whisperx --model small --whisperx-diarization --whisperx-hf-token <HF_TOKEN>
python main.py --stt-provider whisper --stt-variant gpu --model small
python main.py --stt-provider whisper --stt-variant cpu --model small
python main.py --stt-provider vosk --stt-model-path C:\models\vosk-model-small-en-us-0.15
python main.py --stt-provider sherpa-onnx --stt-model-path C:\models\sherpa-transducer --sherpa-onnx-provider cpu
python main.py --stt-provider riva --riva-uri localhost:50051 --riva-language-code en-US
python main.py --stt-provider funasr --stt-model-path iic/SenseVoiceSmall --funasr-device cpu
python main.py --stt-provider funasr --stt-model-path FunAudioLLM/Fun-ASR-Nano-2512 --stt-variant gpu
python main.py --stt-provider vosk --model small --stt-auto-download
python main.py --stt-provider sherpa-onnx --model small --stt-auto-download
python main.py --stt-provider sherpa-onnx --model small --no-stt-auto-download
python main.py --model medium --segment-seconds 6.0 --hop-seconds 1.5
python main.py --source-language auto
python main.py --source-language zh-hant
python main.py --source-language zh-hans
python main.py --source-language ja
python main.py --preprocess-modules auto
python main.py --preprocess-modules webrtc-ns,webrtc-agc
python main.py --preprocess-modules rnnoise
python main.py --preprocess-modules spectral-gate,adaptive-gain
python main.py --no-preprocess
python main.py --vad-rms-threshold 0.010
python main.py --no-adaptive-vad
python main.py --no-vad
python main.py -mc 128 --entropy-thold 2.4 --logprob-thold -1.0 --no-speech-thold 0.6
python main.py --temperature 0.0 --beam-size 1 --best-of 1
python main.py --translate --from-lang en --to-lang zh
python main.py --overlap-merge-method stable-tail
python main.py --overlap-merge-method commit-on-break
python main.py --overlap-merge-method replace-window   # legacy alias -> stable-tail
python main.py --overlap-merge-method append-only      # legacy alias -> commit-on-break
python main.py --source-mode loopback --source-devices 12
python main.py --source-mode microphone --source-devices 3
python main.py --source-mode loopback --source-devices 12,31
python main.py --source-mode app --app-names chrome.exe,discord.exe
python main.py --ui-language zh
python main.py --ui-language en
python src/tests/app_source_visual_probe.py --show-sessions --app-names msedge.exe --seconds 30
python src/tests/app_source_visual_probe.py --app-names msedge.exe,discord.exe --segment-seconds 6 --hop-seconds 1.5
python main.py --bilingual-style stacked
python main.py --bilingual-style translation-only
python main.py --source-text-color "#F0F2F5" --translated-text-color "#FFD98A"
python main.py --cublas-source-dll D:\CUDA\bin\x64\cublas64_13.dll
```

### 4.2 App Source Diagnostic (Visual Probe)

If specified app capture seems silent, run:

```powershell
cd python_app
python src/tests/app_source_visual_probe.py --show-sessions --app-names msedge.exe --seconds 45
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
cd python_app
.\run_post_fix_validation.ps1
```

Optional parameters:

```powershell
.\run_post_fix_validation.ps1 -DurationSeconds 60 -Providers "whisper,vosk,sherpa-onnx,riva,funasr"
```

## 5. whisper_config

File location:
- `python_app/src/whisper_config.json`

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
- Status and error messages are persisted under `python_app/src/logs/voice2text.log` by default.
- `--log-dir` accepts absolute or relative paths; relative paths are resolved from `python_app/src` (not the shell working directory).
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
- App source mode now uses session-gated loopback by default. For strict per-process isolation, explicitly select a VB-CABLE loopback index via `--source-devices`.
- App source mode is now stricter in session gating: target app must dominate (or pass hold window after dominance), so "target present but mixed with louder non-target app" no longer passes.
- In app mode with target app names and no explicit `--source-devices`, runtime now auto-selects a detected VB-CABLE loopback endpoint (if available) for stricter isolation.
- `--list-app-sessions` and settings app list now read from Windows volume mixer sessions (same source as the system mixer panel), rather than full process-details lists.
