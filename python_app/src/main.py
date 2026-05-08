from __future__ import annotations

import argparse
from pathlib import Path
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from app.audio_capture import list_active_app_sessions, list_audio_devices
from app.config import RuntimeConfig
from app.controller import TranscriptionController
from app.cuda_compat import ensure_cublas12_from_source
from app.logging_utils import configure_app_logger
from app.overlay_window import SubtitleOverlayWindow
from app.tray_controller import Voice2TextTrayController
from app.whisper_config import WhisperRuntimeParams, load_whisper_runtime_params


def build_arg_parser(whisper_defaults: WhisperRuntimeParams) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Live rolling subtitle overlay from Windows audio sources."
    )
    parser.add_argument("--model", default="small", help="faster-whisper model size")
    parser.add_argument("--device", default="cuda", help="Whisper device: cuda or cpu")
    parser.add_argument(
        "--compute-type",
        default="float16",
        help="Whisper compute type, e.g. float16, int8_float16, int8",
    )
    parser.add_argument(
        "--no-cpu-fallback",
        action="store_true",
        help="Disable automatic CPU fallback when CUDA initialization fails.",
    )
    parser.add_argument(
        "--cublas-source-dll",
        default=r"D:\CUDA\bin\x64\cublas64_13.dll",
        help="Path to cublas64_13.dll used to prepare cublas64_12.dll compatibility alias.",
    )

    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=6.0,
        help="Audio window length sent to STT.",
    )
    parser.add_argument(
        "--hop-seconds",
        type=float,
        default=1.5,
        help="Sliding hop interval for low-latency incremental updates.",
    )
    parser.add_argument(
        "--overlap-merge-method",
        choices=["replace-window", "suffix-overlap", "fuzzy-overlap", "append-only"],
        default="replace-window",
        help="Merge strategy for overlapped STT windows.",
    )
    parser.add_argument(
        "--source-language",
        choices=["auto", "en", "zh-hant", "zh-hans", "ja", "ko"],
        default="auto",
        help="STT language hint. auto uses multilingual detection.",
    )
    parser.add_argument(
        "--max-context",
        "-mc",
        type=int,
        default=whisper_defaults.max_context,
        help="Whisper decode max context tokens (Python maps to faster-whisper max_new_tokens).",
    )
    parser.add_argument(
        "--entropy-thold",
        type=float,
        default=whisper_defaults.entropy_thold,
        help="Whisper entropy threshold (Python maps to compression_ratio_threshold).",
    )
    parser.add_argument(
        "--logprob-thold",
        type=float,
        default=whisper_defaults.logprob_thold,
        help="Whisper log probability threshold.",
    )
    parser.add_argument(
        "--no-speech-thold",
        type=float,
        default=whisper_defaults.no_speech_thold,
        help="Whisper no-speech threshold.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=whisper_defaults.temperature if whisper_defaults.temperature is not None else 0.0,
        help="Whisper decode temperature.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=whisper_defaults.beam_size if whisper_defaults.beam_size is not None else 1,
        help="Whisper beam size.",
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=whisper_defaults.best_of if whisper_defaults.best_of is not None else 1,
        help="Whisper best-of samples.",
    )

    parser.add_argument(
        "--source-mode",
        choices=["loopback", "microphone", "app"],
        default="loopback",
        help="Audio source mode. app uses session-gated capture by default, or VB-CABLE if explicitly selected.",
    )
    parser.add_argument(
        "--source-devices",
        default="",
        help="Comma-separated source device indices, e.g. 12,35",
    )
    parser.add_argument(
        "--app-names",
        default="",
        help="Comma-separated app names for app source mode, e.g. chrome.exe,discord.exe",
    )

    parser.add_argument(
        "--device-index",
        type=int,
        default=None,
        help="Backward-compatible single source index.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List loopback and microphone capture devices and exit.",
    )
    parser.add_argument(
        "--list-app-sessions",
        action="store_true",
        help="List active app audio sessions and exit.",
    )

    parser.add_argument("--translate", action="store_true", help="Enable Argos translation.")
    parser.add_argument(
        "--from-lang",
        default="auto",
        help="Argos source language code. Use auto to infer from installed models.",
    )
    parser.add_argument("--to-lang", default="zh", help="Argos target language code.")
    parser.add_argument(
        "--bilingual-style",
        choices=["stacked", "translation-only"],
        default="stacked",
        help="How source and translated text should be rendered.",
    )
    parser.add_argument(
        "--hide-source-when-translated",
        action="store_true",
        help="Backward-compatible shortcut for --bilingual-style translation-only.",
    )

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

    parser.add_argument("--log-dir", default="logs", help="Directory for runtime log files.")
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
        print(
            f"[{dev.index}] {dev.kind:10s} | {dev.name} | "
            f"ch={dev.max_input_channels} | rate={dev.default_sample_rate}"
        )
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


def main() -> int:
    config_path = Path(__file__).resolve().with_name("whisper_config.json")
    whisper_defaults = load_whisper_runtime_params(config_path)
    args = build_arg_parser(whisper_defaults).parse_args()

    if args.list_devices:
        return print_devices()

    if args.list_app_sessions:
        return print_app_sessions()

    source_indices = parse_int_csv(args.source_devices)
    if args.device_index is not None and not source_indices:
        source_indices = [args.device_index]

    app_names = parse_str_csv(args.app_names)

    bilingual_style = args.bilingual_style
    if args.hide_source_when_translated:
        bilingual_style = "translation-only"

    source_text_color = args.source_text_color or args.text_color or "#F0F2F5"

    cfg = RuntimeConfig(
        model_size=args.model,
        model_device=args.device,
        compute_type=args.compute_type,
        cpu_fallback_on_cuda_error=not args.no_cpu_fallback,
        cuda_compat_source_dll=args.cublas_source_dll,
        segment_seconds=max(0.5, args.segment_seconds),
        hop_seconds=max(0.1, args.hop_seconds),
        source_language=None,
        source_mode=args.source_mode,
        source_device_indices=source_indices,
        source_mix_weights=[],
        source_app_name=app_names[0] if app_names else "",
        source_app_names=app_names,
        source_channel_mode="mono",
        overlap_merge_method=args.overlap_merge_method,
        whisper_max_context=args.max_context if args.max_context and args.max_context > 0 else None,
        whisper_entropy_thold=args.entropy_thold,
        whisper_logprob_thold=args.logprob_thold,
        whisper_no_speech_thold=args.no_speech_thold,
        whisper_temperature=args.temperature,
        whisper_beam_size=max(1, args.beam_size),
        whisper_best_of=max(1, args.best_of),
        overlay_width=max(480, args.overlay_width),
        overlay_height=max(160, args.overlay_height),
        overlay_x=max(0, args.overlay_x),
        overlay_y=max(0, args.overlay_y),
        overlay_opacity=min(1.0, max(0.2, args.overlay_opacity)),
        font_size=max(10, args.font_size),
        text_color=source_text_color,
        source_text_color=source_text_color,
        translated_text_color=args.translated_text_color,
        background_color=args.background_color,
        translation_enabled=args.translate,
        translation_from=args.from_lang,
        translation_to=args.to_lang,
        bilingual_style=bilingual_style,
        device_index=args.device_index,
        log_dir=args.log_dir,
    )

    if args.source_language != "auto":
        cfg.source_language = args.source_language

    logger = configure_app_logger(cfg.log_dir)
    logger.info("Voice2Text startup")

    if cfg.model_device.lower().startswith("cuda"):
        ok = ensure_cublas12_from_source(
            source_dll=cfg.cuda_compat_source_dll,
            on_status=lambda msg: logger.info(msg),
        )
        if ok:
            logger.info("CUDA compatibility preparation completed.")
        else:
            logger.warning(
                "CUDA compatibility preparation failed. Runtime may fallback to CPU."
            )

    app = QApplication(sys.argv)
    app.setApplicationName("Voice2Text Python")
    app.setQuitOnLastWindowClosed(False)

    overlay = SubtitleOverlayWindow(cfg)
    controller = TranscriptionController(cfg, logger=logger)

    controller.subtitle_ready.connect(overlay.push_subtitle)
    controller.status_message.connect(overlay.push_status)
    controller.error_message.connect(overlay.push_error)

    def apply_settings(updates: dict[str, object]) -> None:
        restart_keys = {
            "source_mode",
            "source_device_indices",
            "source_app_name",
            "source_app_names",
            "source_language",
            "segment_seconds",
            "hop_seconds",
            "overlap_merge_method",
            "translation_enabled",
            "bilingual_style",
            "translation_from",
            "translation_to",
        }

        requires_restart = False
        for key, value in updates.items():
            if not hasattr(cfg, key):
                continue
            if getattr(cfg, key) == value:
                continue
            setattr(cfg, key, value)
            if key in restart_keys:
                requires_restart = True

        overlay.apply_runtime_config(cfg)
        logger.info("Settings updated: %s", updates)

        if requires_restart:
            def restart_capture() -> None:
                try:
                    controller.restart()
                    overlay.push_status("Runtime settings applied. Capture restarted.")
                except Exception as exc:
                    logger.exception("Capture restart failed after settings update")
                    overlay.push_error(f"Capture restart failed: {exc}")

            QTimer.singleShot(0, restart_capture)
        else:
            overlay.push_status("UI settings applied.")

    tray = Voice2TextTrayController(
        app=app,
        overlay=overlay,
        config=cfg,
        on_settings_applied=apply_settings,
    )
    _ = tray

    app.aboutToQuit.connect(controller.stop)

    overlay.show()
    controller.start()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())