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
build-vs\Release\voice2text_cpp.exe --model-path C:\models\ggml-small.bin
```

Optional flags:

```powershell
build-vs\Release\voice2text_cpp.exe --model-path C:\models\ggml-medium.bin --translate --from-lang en --to-lang zh
build-vs\Release\voice2text_cpp.exe --source-mode microphone
build-vs\Release\voice2text_cpp.exe --source-mode app --source-apps chrome.exe,discord.exe
build-vs\Release\voice2text_cpp.exe --list-app-sessions
build-vs\Release\voice2text_cpp.exe --source-language ja
build-vs\Release\voice2text_cpp.exe --source-language zh-hant
build-vs\Release\voice2text_cpp.exe --source-language zh-hans
build-vs\Release\voice2text_cpp.exe --segment-seconds 6 --hop-seconds 1.5
build-vs\Release\voice2text_cpp.exe -mc 128 --entropy-thold 2.4 --logprob-thold -1.0 --no-speech-thold 0.6
build-vs\Release\voice2text_cpp.exe --temperature 0.0 --beam-size 1 --best-of 1
build-vs\Release\voice2text_cpp.exe --overlap-merge-method replace-window
build-vs\Release\voice2text_cpp.exe --overlap-merge-method suffix-overlap
build-vs\Release\voice2text_cpp.exe --overlap-merge-method fuzzy-overlap
build-vs\Release\voice2text_cpp.exe --overlap-merge-method append-only
build-vs\Release\voice2text_cpp.exe --bilingual-style translation-only
build-vs\Release\voice2text_cpp.exe --source-text-color "#F0F2F5" --translated-text-color "#FFD98A" --background-color "#0A101A" --overlay-opacity 0.8
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
- `append-only`
	- Keeps appending by exact overlap.
	- Most conservative, but may retain old recognition mistakes longer.
- `suffix-overlap`
	- Uses strict suffix/prefix exact overlap.
	- Best when recognition output is stable and deterministic.
- `fuzzy-overlap`
	- Uses approximate overlap matching (LCS similarity) when exact overlap misses.
	- Better resilience to minor wording drift.
- `replace-window`
	- Splits previous overlap tail into stable head + mutable tail, then reconciles mutable part with latest chunk.
	- Current factors:
		- `preserveRatio = clamp(lockRatio * 2.0, 0.22, 0.55)`
		- `keepChars = clamp(round(previousTailLen * preserveRatio), 10, previousTailLen)`
	- If overlap confidence is weak (`mergeByFuzzyOverlap` falls back to latest), engine skips about 18% leading chars of latest chunk before merge, reducing low-context head pollution at the start of new segments.

## 7. Current status

- Overlay UI: implemented (drag + edge resize + source/translated colors + auto-fit history).
- Capture mode: `loopback` / `microphone` / `app` implemented; `app` uses Windows Core Audio process loopback (first running match from `--source-apps`) when available, and auto-falls back to default render loopback on toolchains/SDKs that do not expose process-loopback APIs.
- whisper.cpp bridge: implemented with configurable segment/hop windows and selectable overlap merge methods:
	- `replace-window`: lock old prefix, replace recent overlap tail with latest window (recommended)
	- `suffix-overlap`: exact suffix/prefix merge on overlap tail
	- `fuzzy-overlap`: approximate overlap merge for minor recognition drift
	- `append-only`: conservative append behavior
- Translation stage: implemented via Python bridge (`tools/argos_translate_bridge.py`) to Argos Translate, with one-time package auto-install support and stacked/translation-only render style wiring.
- Status/error logging: persisted to `cpp_app/logs/voice2text_cpp.log`, including `STT:` and `TRANSLATE:` lines.
- Translation bridge runtime picks Python from `VOICE2TEXT_PYTHON`, then `python`, then `py -3`; if Argos is unavailable it degrades to STT-only and reports the reason in status logs.