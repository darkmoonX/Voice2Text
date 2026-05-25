"""CLI argument parser and simple listing helpers."""
from __future__ import annotations

import argparse

from .capture import list_active_app_sessions, list_audio_devices
from .whisper_config import WhisperRuntimeParams


def build_arg_parser(whisper_defaults: WhisperRuntimeParams) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live rolling subtitle overlay from Windows audio sources.")
    parser.add_argument("--stt-provider", choices=["whisper", "whisperx"], default="whisper", help="STT backend provider.")
    parser.add_argument("--stt-variant", choices=["auto", "cpu", "gpu"], default="auto", help="Execution variant hint for providers.")
    parser.add_argument("--stt-auto-download", dest="stt_auto_download", action="store_true", help="Allow provider presets to auto-download missing model files.")
    parser.add_argument("--no-stt-auto-download", dest="stt_auto_download", action="store_false", help="Disable provider preset auto-download behavior.")
    parser.add_argument("--model", default="small", help="Model name used by the selected STT provider.")
    parser.add_argument("--stt-model-path", default="", help="Optional model folder path for STT providers. Overrides --model when set.")
    parser.add_argument("--device", default="cuda", help="Whisper device: cuda or cpu")
    parser.add_argument("--compute-type", default="float16", help="Whisper compute type, e.g. float16, int8_float16, int8")
    parser.add_argument("--whisperx-phoneme-asr", dest="whisperx_phoneme_asr", action="store_true", help="Enable WhisperX phoneme-based ASR pipeline.")
    parser.add_argument("--no-whisperx-phoneme-asr", dest="whisperx_phoneme_asr", action="store_false", help="Disable WhisperX phoneme-based ASR pipeline.")
    parser.add_argument("--whisperx-forced-alignment", dest="whisperx_forced_alignment", action="store_true", help="Enable WhisperX forced alignment.")
    parser.add_argument("--no-whisperx-forced-alignment", dest="whisperx_forced_alignment", action="store_false", help="Disable WhisperX forced alignment.")
    parser.add_argument("--whisperx-vad", dest="whisperx_vad", action="store_true", help="Enable WhisperX internal VAD in transcription.")
    parser.add_argument("--no-whisperx-vad", dest="whisperx_vad", action="store_false", help="Disable WhisperX internal VAD in transcription.")
    parser.add_argument("--whisperx-diarization", dest="whisperx_diarization", action="store_true", help="Enable WhisperX diarization.")
    parser.add_argument("--no-whisperx-diarization", dest="whisperx_diarization", action="store_false", help="Disable WhisperX diarization.")
    parser.add_argument("--whisperx-alignment-model", default="", help="Optional WhisperX alignment model id/path.")
    parser.add_argument("--whisperx-alignment-language", choices=["auto", "follow-source", "en", "zh-hant", "zh-hans", "ja", "ko", "de", "fr", "es", "it", "pt", "ru"], default="auto", help="Alignment language override. auto=from ASR result, follow-source=use STT source language setting.")
    parser.add_argument("--whisperx-alignment-device", choices=["auto", "cpu", "cuda"], default="auto", help="Alignment device override. auto uses runtime heuristic.")
    parser.add_argument("--whisperx-diarization-model", default="pyannote/speaker-diarization-3.1", help="WhisperX diarization model id.")
    parser.add_argument("--whisperx-hf-token", default="", help="Hugging Face token for WhisperX diarization model download/access.")
    parser.add_argument("--stt-health-check", action="store_true", help="Run STT provider health checks and exit.")
    parser.add_argument("--stt-health-check-scope", choices=["all", "active"], default="all", help="Health-check scope when --stt-health-check is enabled.")
    parser.add_argument("--no-cpu-fallback", action="store_true", help="Disable automatic CPU fallback when CUDA initialization fails.")
    parser.add_argument("--cublas-source-dll", default="D:\\CUDA\\bin\\x64\\cublas64_13.dll", help="Path to cublas64_13.dll used to prepare cublas64_12.dll compatibility alias.")
    parser.add_argument("--ffmpeg-dll-dir", default="D:\\FFmpeg\\ffmpeg-7.1.1-full_build-shared\\bin", help="Windows FFmpeg shared-DLL directory used for torchcodec/pyannote dynamic loading.")
    parser.add_argument("--segment-seconds", type=float, default=6.0, help="Audio window length sent to STT.")
    parser.add_argument("--hop-seconds", type=float, default=1.5, help="Sliding hop interval for low-latency incremental updates.")
    parser.add_argument("--overlap-merge-method", choices=["stable-tail", "commit-on-break", "replace-window", "suffix-overlap", "fuzzy-overlap", "append-only"], default="stable-tail", help="Merge strategy for overlapped STT windows.")
    parser.add_argument("--no-preprocess", dest="preprocess_enabled", action="store_false", help="Disable audio preprocessing before VAD/STT.")
    parser.add_argument("--preprocess-modules", default="auto", help="Comma-separated preprocessing modules: auto, none, webrtc-ns, webrtc-agc, webrtc-aec, rnnoise, spectral-gate, adaptive-gain.")
    parser.add_argument("--no-vad", dest="vad_enabled", action="store_false", help="Disable pre-transcription VAD pipeline.")
    parser.add_argument("--vad-backend", choices=["silero", "adaptive-rms", "rms"], default="silero", help="General VAD backend used before STT.")
    parser.add_argument("--vad-rms-threshold", type=float, default=0.008, help="RMS threshold for modular VAD gate.")
    parser.add_argument("--no-adaptive-vad", dest="vad_adaptive_enabled", action="store_false", help="Use fixed RMS VAD threshold instead of adaptive environment-noise tracking.")
    parser.add_argument("--vad-adaptive-min-threshold", type=float, default=0.004, help="Lower bound for adaptive RMS VAD threshold.")
    parser.add_argument("--vad-adaptive-max-threshold", type=float, default=0.08, help="Upper bound for adaptive RMS VAD threshold.")
    parser.add_argument("--vad-adaptive-noise-multiplier", type=float, default=2.6, help="Noise-floor multiplier used by adaptive RMS VAD.")
    parser.add_argument("--vad-adaptive-margin", type=float, default=0.002, help="Extra RMS margin added above adaptive noise floor.")
    parser.add_argument("--source-language", choices=["auto", "en", "zh-hant", "zh-hans", "ja", "ko"], default="auto", help="STT language hint. auto uses multilingual detection.")
    parser.add_argument("--cjk-no-space-gap-seconds", type=float, default=0.2, help="When source language is Chinese, adjacent tokens within this gap are concatenated without spaces in stable/history text.")
    parser.add_argument("--max-context", "-mc", type=int, default=whisper_defaults.max_context, help="Whisper decode max context tokens (Python maps to faster-whisper max_new_tokens).")
    parser.add_argument("--entropy-thold", type=float, default=whisper_defaults.entropy_thold, help="Whisper entropy threshold (Python maps to compression_ratio_threshold).")
    parser.add_argument("--logprob-thold", type=float, default=whisper_defaults.logprob_thold, help="Whisper log probability threshold.")
    parser.add_argument("--no-speech-thold", type=float, default=whisper_defaults.no_speech_thold, help="Whisper no-speech threshold.")
    parser.add_argument("--temperature", type=float, default=whisper_defaults.temperature if whisper_defaults.temperature is not None else 0.0, help="Whisper decode temperature.")
    parser.add_argument("--beam-size", type=int, default=whisper_defaults.beam_size if whisper_defaults.beam_size is not None else 1, help="Whisper beam size.")
    parser.add_argument("--best-of", type=int, default=whisper_defaults.best_of if whisper_defaults.best_of is not None else 1, help="Whisper best-of samples.")
    parser.add_argument("--source-mode", choices=["loopback", "microphone", "app"], default="loopback", help="Audio source mode. app prefers C++ Application Loopback Capture for selected process names.")
    parser.add_argument("--ui-language", choices=["zh", "en"], default="zh", help="UI language for tray menu and settings dialog.")
    parser.add_argument("--source-devices", default="", help="Comma-separated source device indices, e.g. 12,35")
    parser.add_argument("--app-names", default="", help="Comma-separated app names for app source mode, e.g. chrome.exe,discord.exe")
    parser.add_argument("--device-index", type=int, default=None, help="Backward-compatible single source index.")
    parser.add_argument("--list-devices", action="store_true", help="List loopback and microphone capture devices and exit.")
    parser.add_argument("--list-app-sessions", action="store_true", help="List active app audio sessions and exit.")
    parser.add_argument("--translate", action="store_true", help="Enable Argos translation.")
    parser.add_argument("--from-lang", default="auto", help="Argos source language code. Use auto to infer from installed models.")
    parser.add_argument("--to-lang", default="zh", help="Argos target language code.")
    parser.add_argument("--bilingual-style", choices=["stacked", "translation-only"], default="stacked", help="How source and translated text should be rendered.")
    parser.add_argument("--hide-source-when-translated", action="store_true", help="Backward-compatible shortcut for --bilingual-style translation-only.")
    parser.add_argument("--overlay-width", type=int, default=1200)
    parser.add_argument("--overlay-height", type=int, default=320)
    parser.add_argument("--overlay-x", type=int, default=40)
    parser.add_argument("--overlay-y", type=int, default=700)
    parser.add_argument("--overlay-opacity", type=float, default=0.8)
    parser.add_argument("--font-size", type=int, default=18)
    parser.add_argument("--source-text-color", default="#F0F2F5")
    parser.add_argument("--translated-text-color", default="#FFD98A")
    parser.add_argument("--text-color", default="", help="Backward-compatible alias of --source-text-color")
    parser.add_argument("--background-color", default="#0A101A")
    parser.add_argument("--log-dir", default="", help="Directory for runtime log files.")
    parser.add_argument("--debug-mode", dest="debug_mode", action="store_true", help="Enable STT debug window with per-step trace.")
    parser.add_argument("--no-debug-mode", dest="debug_mode", action="store_false", help="Disable STT debug window.")
    parser.set_defaults(
        stt_auto_download=True,
        preprocess_enabled=True,
        vad_enabled=True,
        vad_adaptive_enabled=True,
        whisperx_phoneme_asr=True,
        whisperx_forced_alignment=True,
        whisperx_vad=False,
        whisperx_diarization=False,
        debug_mode=False,
    )
    return parser


def parse_int_csv(raw: str) -> list[int]:
    if not raw.strip():
        return []
    values: list[int] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        values.append(int(piece))
    return values


def parse_str_csv(raw: str) -> list[str]:
    if not raw.strip():
        return []
    values: list[str] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        values.append(piece)
    return values


def print_devices() -> int:
    devices = list_audio_devices()
    if not devices:
        print("No capture devices found.")
        return 1
    print("Available capture devices:")
    for dev in devices:
        print(f"[{dev.index}] {dev.kind:10s} | {dev.name} | ch={dev.max_input_channels} | rate={dev.default_sample_rate}")
    return 0


def print_app_sessions() -> int:
    sessions = list_active_app_sessions()
    if not sessions:
        print("No mixer app sessions detected (or pycaw is not installed).")
        return 0
    print("Mixer app sessions:")
    for name in sessions:
        print(f"- {name}")
    return 0
