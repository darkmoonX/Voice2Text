"""Qt runtime bootstrap and settings-apply orchestration."""
from __future__ import annotations

import ctypes
import faulthandler
import logging
import os
from pathlib import Path
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from .config import RuntimeConfig
from .controller import TranscriptionController
from .debug_window import DebugWindowLogHandler, STTDebugWindow
from .logging_utils import configure_app_logger
from .overlay_window import SubtitleOverlayWindow
from .settings_persistence import apply_updates_to_config, load_persisted_updates, save_runtime_settings
from .tray_controller import Voice2TextTrayController

_FAULTHANDLER_FILE = None


def effective_model_label(cfg: RuntimeConfig) -> str:
    if cfg.stt_model_path.strip():
        return cfg.stt_model_path.strip()
    return (cfg.model_size or "").strip() or "unknown"


def set_windows_app_user_model_id() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("voice2text.python.overlay.1")
    except Exception:
        return


def configure_windows_dll_search_paths(cfg: RuntimeConfig, *, on_status) -> None:
    if sys.platform != "win32":
        return
    raw = str(getattr(cfg, "ffmpeg_dll_dir", "") or "").strip()
    if not raw:
        return
    dll_dir = Path(raw)
    if not dll_dir.exists() or (not dll_dir.is_dir()):
        on_status(f"FFmpeg DLL directory not found: {dll_dir}")
        return
    try:
        os.add_dll_directory(str(dll_dir))
        on_status(f"FFmpeg DLL directory registered: {dll_dir}")
    except Exception as exc:
        on_status(f"FFmpeg DLL directory registration failed: {exc}")


def build_restart_keys() -> set[str]:
    return {
        "stt_provider",
        "stt_variant",
        "stt_auto_download",
        "stt_model_path",
        "model_size",
        "model_device",
        "compute_type",
        "whisperx_enable_phoneme_asr",
        "whisperx_enable_forced_alignment",
        "whisperx_enable_vad",
        "whisperx_vad_method",
        "whisperx_enable_diarization",
        "whisperx_alignment_model",
        "whisperx_alignment_language",
        "whisperx_alignment_device",
        "whisperx_diarization_model",
        "whisperx_hf_token",
        "ffmpeg_dll_dir",
        "source_mode",
        "source_device_indices",
        "source_app_name",
        "source_app_names",
        "source_language",
        "segment_seconds",
        "hop_seconds",
        "overlap_merge_method",
        "preprocess_enabled",
        "preprocess_modules",
        "vad_enabled",
        "vad_rms_threshold",
        "vad_adaptive_enabled",
        "vad_adaptive_min_threshold",
        "vad_adaptive_max_threshold",
        "vad_adaptive_noise_multiplier",
        "vad_adaptive_margin",
        "translation_enabled",
        "bilingual_style",
        "translation_from",
        "translation_to",
        "debug_mode",
    }


def run_qt_app(cfg: RuntimeConfig) -> int:
    global _FAULTHANDLER_FILE
    logger = configure_app_logger(cfg.log_dir)
    logger.info("Voice2Text startup")
    try:
        crash_path = Path(cfg.log_dir).resolve().parent / "logs" / "python_crash_trace.log"
        crash_path.parent.mkdir(parents=True, exist_ok=True)
        _FAULTHANDLER_FILE = open(crash_path, "a", encoding="utf-8")
        faulthandler.enable(file=_FAULTHANDLER_FILE, all_threads=True)
        logger.info("Python faulthandler enabled: %s", crash_path)
    except Exception as exc:
        logger.warning("Failed to enable Python faulthandler: %s", exc)
    persisted_updates = load_persisted_updates()
    if persisted_updates:
        changed_keys = apply_updates_to_config(cfg, persisted_updates)
        if changed_keys:
            logger.info("Loaded persisted settings: %s", sorted(changed_keys))

    configure_windows_dll_search_paths(cfg, on_status=lambda msg: logger.info(msg))
    if bool(getattr(cfg, "debug_mode", False)):
        os.environ["VOICE2TEXT_TRACE_WHISPERX"] = "1"
    else:
        os.environ.pop("VOICE2TEXT_TRACE_WHISPERX", None)
    if cfg.model_device.lower().startswith("cuda"):
        logger.info("CUDA compatibility preparation deferred to STT bootstrap.")

    set_windows_app_user_model_id()
    app = QApplication(sys.argv)
    app.setApplicationName("Voice2Text Python")
    app.setQuitOnLastWindowClosed(False)
    overlay = SubtitleOverlayWindow(cfg)
    controller = TranscriptionController(cfg, logger=logger)
    debug_log_dir = str(Path(cfg.log_dir).resolve().parent / "debug_logs")
    runtime_log_dir = str(Path(cfg.log_dir).resolve())
    debug_window_holder: dict[str, STTDebugWindow | None] = {"window": None}
    debug_handler_holder: dict[str, DebugWindowLogHandler | None] = {"handler": None}

    def ensure_debug_window_state() -> None:
        enabled = bool(getattr(cfg, "debug_mode", False))
        window = debug_window_holder.get("window")
        handler = debug_handler_holder.get("handler")
        if enabled:
            if window is None:
                window = STTDebugWindow(debug_log_dir=debug_log_dir)
                controller.debug_event.connect(window.append_event)
                window.load_runtime_history(runtime_log_dir)
                debug_window_holder["window"] = window
            if handler is None:
                handler = DebugWindowLogHandler(window.append_log_line)
                handler.setFormatter(
                    logging.Formatter(
                        fmt="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                    )
                )
                logger.addHandler(handler)
                debug_handler_holder["handler"] = handler
            if not window.isVisible():
                window.show()
                window.raise_()
                window.activateWindow()
        else:
            if handler is not None:
                try:
                    logger.removeHandler(handler)
                except Exception:
                    pass
                debug_handler_holder["handler"] = None
            if window is not None and window.isVisible():
                window.hide()

    ensure_debug_window_state()
    controller.subtitle_ready.connect(overlay.push_subtitle)
    controller.status_message.connect(overlay.push_status)
    controller.error_message.connect(overlay.push_error)
    controller.runtime_state_changed.connect(overlay.set_runtime_running)

    def toggle_runtime() -> None:
        if controller.is_running():
            controller.stop()
            overlay.push_status("Capture stopped.")
        else:
            controller.start()
            overlay.push_status("Capture resumed.")

    overlay.toggle_runtime_requested.connect(toggle_runtime)
    restart_keys = build_restart_keys()
    tray_holder: dict[str, Voice2TextTrayController] = {}

    def apply_settings(updates: dict[str, object]) -> None:
        changed_keys = apply_updates_to_config(cfg, updates)
        requires_restart = any((key in restart_keys for key in changed_keys))
        overlay.apply_runtime_config(cfg)
        tray = tray_holder.get("tray")
        if tray is not None and "ui_language" in updates:
            tray.refresh_locale()
        try:
            settings_path = save_runtime_settings(cfg)
            logger.info("Persisted settings saved: %s", settings_path)
        except Exception:
            logger.exception("Failed to save persisted settings")
        logger.info("Settings updated: %s", updates)
        ensure_debug_window_state()
        if requires_restart:

            def restart_capture() -> None:
                try:
                    controller.restart()
                    overlay.push_status(f"Runtime settings applied. Capture restarted. model={effective_model_label(cfg)}")
                except Exception as exc:
                    logger.exception("Capture restart failed after settings update")
                    overlay.push_error(f"Capture restart failed: {exc}")

            QTimer.singleShot(0, restart_capture)
        else:
            overlay.push_status(f"UI settings applied. model={effective_model_label(cfg)}")

    tray_holder["tray"] = Voice2TextTrayController(app=app, overlay=overlay, config=cfg, on_settings_applied=apply_settings)
    def _cleanup_on_quit() -> None:
        controller.stop()
        handler = debug_handler_holder.get("handler")
        if handler is not None:
            try:
                logger.removeHandler(handler)
            except Exception:
                pass
            debug_handler_holder["handler"] = None

    app.aboutToQuit.connect(_cleanup_on_quit)
    overlay.show()
    controller.start()
    return app.exec()
