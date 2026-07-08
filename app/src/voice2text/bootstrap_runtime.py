"""Qt runtime bootstrap and settings-apply orchestration."""
from __future__ import annotations

import ctypes
from datetime import datetime
import faulthandler
import logging
import os
from pathlib import Path
import sys
import threading
import traceback

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from .config import RuntimeConfig
from .controller import TranscriptionController
from .debug_window import DebugWindowLogHandler, STTDebugWindow
from .logging_utils import configure_app_logger, suppress_third_party_console_logging
from .overlay_window import SubtitleOverlayWindow
from .settings_persistence import (
    apply_updates_to_config,
    load_persisted_updates,
    save_runtime_settings,
    seed_alignment_model_defaults,
)
from .tray_controller import Voice2TextTrayController

_FAULTHANDLER_FILE = None
_CRASH_TRACE_LOCK = threading.Lock()
_PREVIOUS_EXCEPTHOOK = None
_PREVIOUS_THREADING_EXCEPTHOOK = None
_PREVIOUS_UNRAISABLEHOOK = None
_CRASH_BUNDLE_WRITTEN = False


def _crash_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write_crash_trace_line(message: str) -> None:
    handle = _FAULTHANDLER_FILE
    if handle is None:
        return
    with _CRASH_TRACE_LOCK:
        try:
            handle.write(message.rstrip() + "\n")
            handle.flush()
        except Exception:
            return


def _write_crash_trace_exception(title: str, exc_type, exc_value, exc_tb) -> None:
    handle = _FAULTHANDLER_FILE
    if handle is None:
        return
    with _CRASH_TRACE_LOCK:
        try:
            handle.write(f"\n=== {title} at {_crash_timestamp()} ===\n")
            traceback.print_exception(exc_type, exc_value, exc_tb, file=handle)
            handle.flush()
        except Exception:
            return


def _write_auto_crash_bundle(cfg: RuntimeConfig, logger: logging.Logger, reason: str) -> None:
    """Best-effort, once-per-process diagnostics bundle on an uncaught top-level exception.

    Deliberately NOT hooked into threading/unraisable exceptions too: those can fire repeatedly
    for benign library warnings (e.g. numpy/pyannote runtime warnings promoted by some
    environments), and a bundle-per-occurrence would spam disk writes. The single top-level
    `sys.excepthook` case is the genuinely rare "the app is crashing" signal.
    """
    global _CRASH_BUNDLE_WRITTEN
    if _CRASH_BUNDLE_WRITTEN or not bool(getattr(cfg, "crash_bundle_on_uncaught_exception", True)):
        return
    _CRASH_BUNDLE_WRITTEN = True
    try:
        from .crash_bundle import create_crash_bundle

        path = create_crash_bundle(cfg, reason=reason)
        logger.info("Crash bundle written: %s", path)
    except Exception as exc:
        logger.warning("Failed to write auto crash bundle: %s", exc)


def _install_python_exception_hooks(logger: logging.Logger, cfg: RuntimeConfig | None = None) -> None:
    global _PREVIOUS_EXCEPTHOOK, _PREVIOUS_THREADING_EXCEPTHOOK, _PREVIOUS_UNRAISABLEHOOK
    if _PREVIOUS_EXCEPTHOOK is None:
        _PREVIOUS_EXCEPTHOOK = sys.excepthook
    if _PREVIOUS_UNRAISABLEHOOK is None:
        _PREVIOUS_UNRAISABLEHOOK = sys.unraisablehook
    if _PREVIOUS_THREADING_EXCEPTHOOK is None:
        _PREVIOUS_THREADING_EXCEPTHOOK = threading.excepthook

    def excepthook(exc_type, exc_value, exc_tb) -> None:
        _write_crash_trace_exception("Uncaught Python exception", exc_type, exc_value, exc_tb)
        logger.error("Uncaught Python exception", exc_info=(exc_type, exc_value, exc_tb))
        if cfg is not None:
            _write_auto_crash_bundle(cfg, logger, "uncaught Python exception")
        if _PREVIOUS_EXCEPTHOOK is not None and _PREVIOUS_EXCEPTHOOK is not excepthook:
            _PREVIOUS_EXCEPTHOOK(exc_type, exc_value, exc_tb)

    def threading_excepthook(args: threading.ExceptHookArgs) -> None:
        _write_crash_trace_exception(
            f"Uncaught thread exception ({getattr(args.thread, 'name', 'unknown')})",
            args.exc_type,
            args.exc_value,
            args.exc_traceback,
        )
        logger.error(
            "Uncaught thread exception (%s)",
            getattr(args.thread, "name", "unknown"),
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )
        if _PREVIOUS_THREADING_EXCEPTHOOK is not None and _PREVIOUS_THREADING_EXCEPTHOOK is not threading_excepthook:
            _PREVIOUS_THREADING_EXCEPTHOOK(args)

    def unraisablehook(unraisable) -> None:
        _write_crash_trace_exception(
            f"Unraisable Python exception ({getattr(unraisable, 'err_msg', '') or 'no message'})",
            unraisable.exc_type,
            unraisable.exc_value,
            unraisable.exc_traceback,
        )
        logger.error(
            "Unraisable Python exception: %s",
            getattr(unraisable, "err_msg", "") or "no message",
            exc_info=(unraisable.exc_type, unraisable.exc_value, unraisable.exc_traceback),
        )
        if _PREVIOUS_UNRAISABLEHOOK is not None and _PREVIOUS_UNRAISABLEHOOK is not unraisablehook:
            _PREVIOUS_UNRAISABLEHOOK(unraisable)

    sys.excepthook = excepthook
    threading.excepthook = threading_excepthook
    sys.unraisablehook = unraisablehook


def _start_crash_trace_heartbeat(stop_event: threading.Event, interval_seconds: float = 10.0) -> None:
    def run() -> None:
        while not stop_event.wait(interval_seconds):
            _write_crash_trace_line(f"[crash-heartbeat] last_alive={_crash_timestamp()}")

    thread = threading.Thread(target=run, name="crash-trace-heartbeat", daemon=True)
    thread.start()


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
        "stt_whispercpp_model_path",
        "stt_whispercpp_model_size",
        "stt_whispercpp_binary_path",
        "stt_whispercpp_server_path",
        "stt_whispercpp_mode",
        "stt_whispercpp_server_vad",
        "stt_whispercpp_vad_model_path",
        "stt_whispercpp_vad_model",
        "stt_whispercpp_server_max_len",
        "stt_whispercpp_request_timeout_seconds",
        "stt_whispercpp_no_speech_threshold",
        "stt_whispercpp_avg_logprob_min",
        "stt_whispercpp_repetition_similarity",
        "stt_whispercpp_boilerplate_phrases",
        "model_size",
        "model_device",
        "compute_type",
        "whisperx_enable_phoneme_asr",
        "whisperx_enable_forced_alignment",
        "whisperx_enable_vad",
        "whisperx_vad_method",
        "whisperx_enable_diarization",
        "whisperx_alignment_model",
        "whisperx_zh_align_wbbbbb",
        "whisperx_alignment_model_defaults",
        "whisperx_alignment_language",
        "whisperx_alignment_device",
        "whisperx_align_guard",
        "whisperx_diarization_device",
        "whisperx_diarization_model",
        "whisperx_diarization_min_speakers",
        "whisperx_diarization_max_speakers",
        "whisperx_hf_token",
        "whisperx_speaker_profile_enabled",
        "whisperx_speaker_profile_backend",
        "whisperx_speaker_profile_model",
        "whisperx_speaker_speechbrain_model",
        "whisperx_speaker_nemo_model",
        "whisperx_speaker_profile_match_threshold",
        "whisperx_speaker_profile_min_seconds",
        "whisperx_speaker_profile_reconcile_threshold",
        "whisperx_speaker_profile_store_path",
        "whisperx_speaker_count_hint_enabled",
        "whisperx_speaker_count_hint_seconds",
        "whisperx_speaker_count_hint_window_seconds",
        "whisperx_speaker_count_hint_sliver_floor_seconds",
        "whisperx_speaker_merge_grace_windows",
        "whisperx_speaker_merge_grace_relief",
        "whisperx_speaker_merge_preserve_centroid",
        "whisperx_speaker_profile_max_exemplars",
        "whisperx_speaker_profile_exemplar_diversity_threshold",
        "speaker_marker_style",
        "whisper_beam_size",
        "whisper_batch_size",
        # Round 0051: ASR-load-time and loop-start-time knobs need a pipeline restart to apply.
        "whisperx_asr_temperatures",
        "whisperx_asr_log_prob_threshold",
        "whisperx_asr_compression_ratio_threshold",
        "whisperx_asr_no_speech_threshold",
        "subtitle_commit_hold_seconds",
        "subtitle_relabel_enabled",
        "subtitle_relabel_async",
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
        "translation_enabled",
        "bilingual_style",
        "translation_from",
        "translation_to",
        "translation_backend",
        "translation_queue_max",
        "translation_request_timeout_seconds",
        "translation_max_retries",
        "translation_retry_backoff_seconds",
        "debug_mode",
        "transcript_export_enabled",
        "transcript_export_formats",
        "transcript_export_include_timestamps",
        "transcript_export_include_speaker",
        "transcript_export_dir",
        # Round 0076: both gate _build_capture()'s recorder wrapping / stop()-time relabel
        # kickoff, which only run at capture (re)start.
        "session_record_enabled",
        "session_finalize_direct_relabel_enabled",
    }


_SENSITIVE_SETTING_KEYS = {
    "whisperx_hf_token",
}


def sanitize_settings_for_log(updates: dict[str, object]) -> dict[str, object]:
    safe: dict[str, object] = {}
    for (key, value) in updates.items():
        if key in _SENSITIVE_SETTING_KEYS:
            safe[key] = "<redacted>" if str(value or "").strip() else ""
        else:
            safe[key] = value
    return safe


def run_qt_app(cfg: RuntimeConfig) -> int:
    global _FAULTHANDLER_FILE
    logger = configure_app_logger(cfg.log_dir)
    logger.info("Voice2Text startup")
    crash_heartbeat_stop = threading.Event()
    try:
        crash_path = Path(cfg.log_dir).resolve().parent / "logs" / "python_crash_trace.log"
        crash_path.parent.mkdir(parents=True, exist_ok=True)
        _FAULTHANDLER_FILE = open(crash_path, "a", encoding="utf-8", buffering=1)
        _write_crash_trace_line(f"\n=== Voice2Text crash trace session started at {_crash_timestamp()} ===")
        faulthandler.enable(file=_FAULTHANDLER_FILE, all_threads=True)
        _install_python_exception_hooks(logger, cfg)
        _start_crash_trace_heartbeat(crash_heartbeat_stop)
        logger.info("Python faulthandler enabled: %s", crash_path)
    except Exception as exc:
        logger.warning("Failed to enable Python faulthandler: %s", exc)
    persisted_updates = load_persisted_updates()
    if persisted_updates:
        changed_keys = apply_updates_to_config(cfg, persisted_updates)
        if changed_keys:
            logger.info("Loaded persisted settings: %s", sorted(changed_keys))

    # Round 0077: fold the legacy whisperx_zh_align_wbbbbb / whisperx_english_align_large
    # booleans (just restored from disk above, or set via CLI in build_runtime_config) into the
    # generalized whisperx_alignment_model_defaults map, so an upgrade preserves an existing
    # user's alignment preference in the new Settings-dialog-editable form.
    seeded_align_langs = seed_alignment_model_defaults(cfg)
    if seeded_align_langs:
        logger.info("Migrated legacy alignment-model flags into whisperx_alignment_model_defaults: %s", sorted(seeded_align_langs))

    # Seed per-language defaults (e.g. zh reconcile_threshold 0.88) for fields the user left at
    # the global default. Runs after CLI + persisted so explicit choices still win.
    from .language_defaults import apply_language_defaults
    lang_applied = apply_language_defaults(cfg)
    if lang_applied:
        logger.info("Applied per-language defaults (%s): %s", cfg.source_language, sorted(lang_applied))

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
    root_debug_handler_holder: dict[str, DebugWindowLogHandler | None] = {"handler": None}

    def ensure_debug_window_state() -> None:
        enabled = bool(getattr(cfg, "debug_mode", False))
        window = debug_window_holder.get("window")
        handler = debug_handler_holder.get("handler")
        root_handler = root_debug_handler_holder.get("handler")
        if enabled:
            suppress_third_party_console_logging()
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
            if root_handler is None:
                root_handler = DebugWindowLogHandler(window.append_log_line)
                root_handler.setFormatter(
                    logging.Formatter(
                        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                    )
                )
                logging.getLogger().addHandler(root_handler)
                root_debug_handler_holder["handler"] = root_handler
            if not window.isVisible():
                window.show()
                window.raise_()
                window.activateWindow()
        else:
            suppress_third_party_console_logging()
            if handler is not None:
                try:
                    logger.removeHandler(handler)
                except Exception:
                    pass
                debug_handler_holder["handler"] = None
            if root_handler is not None:
                try:
                    logging.getLogger().removeHandler(root_handler)
                except Exception:
                    pass
                root_debug_handler_holder["handler"] = None
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

    def export_transcript_now(output_path: str, export_format: str, include_timestamps: bool, include_speaker: bool) -> str:
        return controller.export_transcript_now(
            output_path=output_path,
            export_format=export_format,
            include_timestamps=bool(include_timestamps),
            include_speaker=bool(include_speaker),
        )

    def import_audio_file(file_path: str, mode: str = "replay") -> str:
        selected = str(mode or "replay").strip().lower()
        if selected == "direct":
            imported = controller.import_audio_file_direct(file_path)
            overlay.push_status(f"Imported audio direct transcription started: {imported}")
            return imported
        imported = controller.import_audio_file(file_path)
        overlay.push_status(f"Imported audio replay started: {imported}")
        return imported

    def apply_settings(updates: dict[str, object]) -> None:
        deferred_restart_updates: dict[str, object] = {}
        if controller.is_temporary_file_replay_active():
            deferred_restart_updates = {key: value for (key, value) in updates.items() if key in restart_keys}
            if deferred_restart_updates:
                updates = {key: value for (key, value) in updates.items() if key not in restart_keys}
                logger.info(
                    "Settings requiring runtime restart deferred during imported-audio replay: %s",
                    sorted(deferred_restart_updates),
                )
        changed_keys = apply_updates_to_config(cfg, updates)
        requires_restart = any((key in restart_keys for key in changed_keys))
        overlay.apply_runtime_config(cfg)
        tray = tray_holder.get("tray")
        if tray is not None and "ui_language" in updates:
            tray.refresh_locale()
        try:
            persist_cfg = cfg
            restore_source = controller.temporary_source_restore_values()
            if restore_source is not None:
                persist_cfg = RuntimeConfig(**cfg.__dict__)
                persist_cfg.source_mode = str(restore_source.get("source_mode") or "loopback")
                persist_cfg.source_file_path = str(restore_source.get("source_file_path") or "")
                persist_cfg.source_file_replay_speed = float(restore_source.get("source_file_replay_speed") or 0.0)
                persist_cfg.source_file_chunk_seconds = float(restore_source.get("source_file_chunk_seconds") or 0.25)
            settings_path = save_runtime_settings(persist_cfg)
            logger.info("Persisted settings saved: %s", settings_path)
        except Exception:
            logger.exception("Failed to save persisted settings")
        logger.info("Settings updated: %s", sanitize_settings_for_log(updates))
        ensure_debug_window_state()
        if deferred_restart_updates:
            overlay.push_status("Imported audio replay is still running. Runtime-restart settings were deferred until replay stops.")
            return
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

    tray_holder["tray"] = Voice2TextTrayController(
        app=app,
        overlay=overlay,
        config=cfg,
        on_settings_applied=apply_settings,
        on_export_transcript=export_transcript_now,
        on_import_audio=import_audio_file,
    )
    def _cleanup_on_quit() -> None:
        crash_heartbeat_stop.set()
        _write_crash_trace_line(f"=== Voice2Text crash trace session ended at {_crash_timestamp()} ===")
        controller.stop()
        handler = debug_handler_holder.get("handler")
        if handler is not None:
            try:
                logger.removeHandler(handler)
            except Exception:
                pass
        root_handler = root_debug_handler_holder.get("handler")
        if root_handler is not None:
            try:
                logging.getLogger().removeHandler(root_handler)
            except Exception:
                pass
            debug_handler_holder["handler"] = None

    app.aboutToQuit.connect(_cleanup_on_quit)
    overlay.show()
    import_direct_path = str(getattr(cfg, "import_direct_path", "") or "").strip()
    if import_direct_path:
        QTimer.singleShot(0, lambda: import_audio_file(import_direct_path, "direct"))
    else:
        controller.start()
    return app.exec()
