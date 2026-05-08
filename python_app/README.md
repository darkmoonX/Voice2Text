# Python App (MVP)

Windows live subtitle overlay using:
- faster-whisper (STT)
- pyaudiowpatch (WASAPI loopback capture)
- PySide6 (semi-transparent rolling subtitles)
- Argos Translate (optional translation stage)
- pycaw (optional app-session awareness)

## 1. Setup

```powershell
cd python_app
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Download Whisper model (first run auto-download)

The first run of faster-whisper downloads the selected model automatically.
Default model is `small`.

Model storage convention:
- faster-whisper models are stored under `python_app/src/models/faster-whisper`.
- example: model "small" in faster-whisper is stored under `python_app/src/models/faster-whisper/small`
- If future libraries use additional models, place them under:
	- `python_app/src/models/<library-name>/<model-name>`

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
python main.py --model medium --segment-seconds 6.0 --hop-seconds 1.5
python main.py --source-language auto
python main.py --source-language zh-hant
python main.py --source-language zh-hans
python main.py --source-language ja
python main.py -mc 128 --entropy-thold 2.4 --logprob-thold -1.0 --no-speech-thold 0.6
python main.py --temperature 0.0 --beam-size 1 --best-of 1
python main.py --translate --from-lang en --to-lang zh
python main.py --overlap-merge-method replace-window
python main.py --overlap-merge-method suffix-overlap
python main.py --overlap-merge-method fuzzy-overlap
python main.py --overlap-merge-method append-only
python main.py --source-mode loopback --source-devices 12
python main.py --source-mode microphone --source-devices 3
python main.py --source-mode loopback --source-devices 12,31
python main.py --source-mode app --app-names chrome.exe,discord.exe
python main.py --bilingual-style stacked
python main.py --bilingual-style translation-only
python main.py --source-text-color "#F0F2F5" --translated-text-color "#FFD98A"
python main.py --cublas-source-dll D:\CUDA\bin\x64\cublas64_13.dll
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
- `append-only`
	- Keep appending by exact suffix/prefix overlap only.
	- Least destructive, but most likely to keep historical recognition errors.
- `suffix-overlap`
	- Use exact overlap between old tail and incoming head.
	- High precision when transcripts are stable; weaker against small drift.
- `fuzzy-overlap`
	- Use approximate overlap matching (SequenceMatcher) when exact overlap misses.
	- Better tolerance for minor wording shifts.
- `replace-window`
	- Keep a stable head from previous overlap tail, and replace only mutable tail with latest chunk.
	- Current factors:
		- `preserve_ratio = clamp(lock_ratio * 2.0, 0.22, 0.55)`
		- `keep_chars = clamp(round(len(previous_tail) * preserve_ratio), 10, len(previous_tail))`
	- If overlap is weak, the algorithm avoids trusting the most unstable leading tokens of latest chunk by skipping about 18% of latest head before reconciling.
	- This design reduces early-window context loss pollution at the start of each new segment.

## 7. Notes

- CUDA missing `cublas64_12.dll` will now auto-fallback to CPU (`int8`) unless `--no-cpu-fallback` is set.
- The app will also try to create a compatibility alias from the source DLL path (`--cublas-source-dll`) by copying `cublas64_13.dll` to runtime `cublas64_12.dll`.
- Edge resize is enabled on the overlay window (frameless, always on top).
- Status and error messages are persisted under `python_app/src/logs/voice2text.log` (or custom `--log-dir`).
- Transcription and translation outputs are also persisted in log file (`STT:` and `TRANSLATE:` records).
- A tray icon is provided:
	- Left click toggles show/hide.
	- Right click menu: Show / Minimize / Settings / Exit.
- Settings dialog supports runtime updates for source mode, multi-source selection, source language hint (auto/en/zh-hant/zh-hans/ja/ko), translation toggle + style, translation target language, source/translated text colors, background, opacity, and low-latency parameters.
- Subtitle merge strategy is user-selectable (`replace-window`, `suffix-overlap`, `fuzzy-overlap`, `append-only`) and now supports overlap correction for sliding windows.
- Subtitle rendering now keeps a rolling sentence and wraps only when width is exceeded.
- App source mode now uses session-gated loopback by default. For strict per-process isolation, explicitly select a VB-CABLE loopback index via `--source-devices`.
- `--list-app-sessions` and settings app list now read from Windows volume mixer sessions (same source as the system mixer panel), rather than full process-details lists.