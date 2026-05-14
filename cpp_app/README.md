# C++ App (Qt + whisper.cpp)

Windows C++ path with:
- Qt (overlay UI)
- whisper.cpp (speech recognition)
- CUDA/cuBLAS acceleration path for RTX 3060

## 1. Prerequisites

- Visual Studio 2026/2022 with Desktop C++ workload (must provide `cl.exe`)
- CMake >= 3.22
- Qt 6 SDK (Core/Gui/Widgets)
- CUDA toolkit (for GPU acceleration, needs `nvcc` in PATH)

Important:
- `Qt6Config.cmake` must be discoverable via `Qt6_DIR` or `CMAKE_PREFIX_PATH`.
- Project now auto-detects Qt from `D:\Qt` and CUDA from `D:\CUDA` when possible.
- If `nvcc` is not found, CMake now auto-disables CUDA and falls back to CPU build.

## 2. Get whisper.cpp

Place whisper.cpp under:

`cpp_app/third_party/whisper.cpp`

Example:

```powershell
cd cpp_app
git clone https://github.com/ggerganov/whisper.cpp third_party/whisper.cpp
```

## 3. Configure and build

```powershell
cd cpp_app
.\build.ps1 -Gpu
```

Build outputs:
- MSVC GPU build: `build-vs-gpu\Release\voice2text_cpp.exe`
- MSVC CPU build: `build-vs-cpu\Release\voice2text_cpp.exe`

`build.ps1` now also:
- syncs whisper/ggml DLLs from `build-*\bin\Release` into `build-*\Release`
- runs `windeployqt` for Qt runtime deployment when available

Explicit path overrides:

```powershell
.\build.ps1 -Qt6Dir D:\Qt\6.11.0\msvc2022_64\lib\cmake\Qt6 -CudaRoot D:\CUDA -VsRoot "D:\Microsoft Visual Studio" -Gpu
```

If Visual Studio generator is unavailable, script auto-falls back to `VsDevCmd + NMake/JOM`.
If `cl.exe` is still not found, script now auto-falls back to MinGW CPU build (`build-mingw-cpu`) when Qt MinGW kit is available.

Manual configure example:

```powershell
cd cpp_app
cmake -S . -B build-vs -G "Visual Studio 18 2026" -A x64 -DQt6_DIR=D:\Qt\6.11.0\msvc2022_64\lib\cmake\Qt6 -DCUDAToolkit_ROOT=D:\CUDA -DENABLE_WHISPER_CUDA=ON -DWHISPER_CUDA_ARCHITECTURES=86
cmake --build build-vs --config Release
```

## 4. Run

```powershell
build-vs-gpu\Release\voice2text_cpp.exe --model-path C:\models\ggml-small.bin
```

Optional flags:

```powershell
build-vs-gpu\Release\voice2text_cpp.exe --model-path C:\models\ggml-medium.bin --translate --from-lang en --to-lang zh
build-vs-gpu\Release\voice2text_cpp.exe --source-mode microphone
build-vs-gpu\Release\voice2text_cpp.exe --source-mode app --source-apps chrome.exe,discord.exe
build-vs-gpu\Release\voice2text_cpp.exe --list-app-sessions
build-vs-gpu\Release\voice2text_cpp.exe --source-language ja
build-vs-gpu\Release\voice2text_cpp.exe --source-language zh-hant
build-vs-gpu\Release\voice2text_cpp.exe --source-language zh-hans
build-vs-gpu\Release\voice2text_cpp.exe --segment-seconds 6 --hop-seconds 1.5
build-vs-gpu\Release\voice2text_cpp.exe -mc 128 --entropy-thold 2.4 --logprob-thold -1.0 --no-speech-thold 0.6
build-vs-gpu\Release\voice2text_cpp.exe --temperature 0.0 --beam-size 1 --best-of 1
build-vs-gpu\Release\voice2text_cpp.exe --overlap-merge-method stable-tail
build-vs-gpu\Release\voice2text_cpp.exe --overlap-merge-method commit-on-break
build-vs-gpu\Release\voice2text_cpp.exe --overlap-merge-method replace-window   # legacy alias -> stable-tail
build-vs-gpu\Release\voice2text_cpp.exe --overlap-merge-method append-only      # legacy alias -> commit-on-break
build-vs-gpu\Release\voice2text_cpp.exe --bilingual-style translation-only
build-vs-gpu\Release\voice2text_cpp.exe --source-text-color "#F0F2F5" --translated-text-color "#FFD98A" --background-color "#0A101A" --overlay-opacity 0.8
```

If built via MinGW fallback, run with Qt/MinGW runtime paths:

```powershell
$env:Path = "D:\Qt\6.11.0\mingw_64\bin;D:\Qt\Tools\mingw1310_64\bin;$env:Path"
build-mingw-cpu\voice2text_cpp.exe --help
```

## 5. whisper_config

File location:
- `cpp_app/src/whisper_config.json`

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

Runtime binding in whisper.cpp:
- `max-context` -> `whisper_full_params.n_max_text_ctx` (when `> 0`, and `no_context=false`)
- `entropy-thold` -> `whisper_full_params.entropy_thold`
- `logprob-thold` -> `whisper_full_params.logprob_thold`
- `no-speech-thold` -> `whisper_full_params.no_speech_thold`
- `temperature` -> `whisper_full_params.temperature`
- `beam-size` / `best-of` -> decode strategy and beam/greedy fields

## 6. Overlap Merge Methods (Implementation Detail)

Rolling model:
- C++ keeps two buffers in `WhisperEngine`:
	- `frozenTranscript_`: locked historical prefix.
	- `activeWindowTranscript_`: mutable overlap window tail.
- Lock ratio uses `hopMs / windowMs`, clamped to `[0.05, 0.95]`.

Methods:
- `stable-tail` (recommended)
	- Locks a prefix from current rolling window and only reconciles mutable tail with incoming segment.
	- Includes extra overlap repair and repeated-tail collapse for mixed English/CJK transcripts.
- `commit-on-break`
	- Keeps incremental append behavior in active sentence and commits/freeze on sentence break timing.
	- Lower rewrite aggressiveness, useful when you prefer conservative transcript evolution.

Legacy aliases:
- `replace-window`, `suffix-overlap`, `fuzzy-overlap` -> `stable-tail`
- `append-only` -> `commit-on-break`

## 7. Current status

- Overlay UI: implemented (drag + edge resize + source/translated colors + auto-fit history).
- Capture mode: `loopback` / `microphone` / `app` implemented; `app` uses Windows Core Audio process loopback (first running match from `--source-apps`) when available.
- App-mode strict isolation fallback policy:
	- if process-loopback is unavailable, runtime auto-selects a VB-CABLE loopback endpoint when available.
	- if neither process-loopback nor VB-CABLE is available, app capture aborts (no mixed default-loopback fallback).
- whisper.cpp bridge: implemented with configurable segment/hop windows and selectable overlap merge methods:
	- `stable-tail`: lock stable prefix and reconcile mutable tail (recommended)
	- `commit-on-break`: conservative append model with sentence-break commit
- Startup/settings status now includes effective model label (`model=...`) after runtime apply.
- Architecture improvement: `RuntimeSettings` and merge-method normalization are now centralized in `src/runtime/runtime_settings.*` instead of being embedded in settings UI.
- Architecture improvement (round 2): audio/session discovery moved to `src/audio/discovery.*`; runtime restart-decision mapping moved to `src/runtime/runtime_update.*`.
- Architecture improvement (round 3):
	- settings payload mapping/validation moved to `src/settings/mapping.*`.
	- settings UI string resources moved to `src/settings/i18n.*` and wired into Settings dialog/tray for zh/en labels.
	- `RuntimeSettings` now carries `uiLanguage` and settings payload mapping persists this field.
- Translation stage: implemented via Python bridge (`tools/argos_translate_bridge.py`) to Argos Translate, with one-time package auto-install support and stacked/translation-only render style wiring.
- Status/error logging: persisted to `cpp_app/logs/voice2text_cpp.log`, including `STT:` and `TRANSLATE:` lines.
- Translation bridge runtime picks Python from `VOICE2TEXT_PYTHON`, then `python`, then `py -3`; if Argos is unavailable it degrades to STT-only and reports the reason in status logs.
