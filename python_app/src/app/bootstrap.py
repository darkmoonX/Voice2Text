"""CLI parsing, runtime config construction, health-check execution, and Qt app bootstrap."""
from __future__ import annotations
import argparse
import ctypes
import os
from pathlib import Path
import sys
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from .capture import list_active_app_sessions, list_audio_devices
from .config import RuntimeConfig
from .controller import TranscriptionController
from .debug_window import STTDebugWindow
from .cuda_compat import ensure_cublas12_from_source
from .logging_utils import configure_app_logger
from .overlay_window import SubtitleOverlayWindow
from .stt import has_failed_reports, run_provider_health_check, summarize_health_reports
from .stt.whisper_provider import WhisperRuntimeParams, load_whisper_runtime_params
from .tray_controller import Voice2TextTrayController


def _normalize_merge_method(raw_method: str) -> str:
    method = (raw_method or '').strip().lower()
    if method in {'stable-tail', 'replace-window', 'suffix-overlap', 'fuzzy-overlap'}:
        return 'stable-tail'
    if method in {'commit-on-break', 'append-only'}:
        return 'commit-on-break'
    return 'stable-tail'


def _effective_model_label(cfg: RuntimeConfig) -> str:
    if cfg.stt_model_path.strip():
        return cfg.stt_model_path.strip()
    return (cfg.model_size or '').strip() or 'unknown'


def _set_windows_app_user_model_id() -> None:
    if sys.platform != 'win32':
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('voice2text.python.overlay.1')
    except Exception:
        return


def _configure_windows_dll_search_paths(cfg: RuntimeConfig, *, on_status) -> None:
    if sys.platform != 'win32':
        return
    raw = str(getattr(cfg, 'ffmpeg_dll_dir', '') or '').strip()
    if not raw:
        return
    dll_dir = Path(raw)
    if not dll_dir.exists() or (not dll_dir.is_dir()):
        on_status(f'FFmpeg DLL directory not found: {dll_dir}')
        return
    try:
        os.add_dll_directory(str(dll_dir))
        on_status(f'FFmpeg DLL directory registered: {dll_dir}')
    except Exception as exc:
        on_status(f'FFmpeg DLL directory registration failed: {exc}')

def _default_log_dir() -> str:
    """Keep runtime logs inside python_app/src/logs regardless of launch cwd."""
    return str((Path(__file__).resolve().parents[1] / 'logs'))

def _resolve_log_dir(raw_path: str) -> str:
    candidate = Path(raw_path.strip()) if raw_path and raw_path.strip() else Path(_default_log_dir())
    if not candidate.is_absolute():
        candidate = (Path(__file__).resolve().parents[1] / candidate)
    return str(candidate)

def build_arg_parser(whisper_defaults: WhisperRuntimeParams) -> argparse.ArgumentParser:
    """Build CLI options used by main.py and tray-launched runtime."""
    parser = argparse.ArgumentParser(description='Live rolling subtitle overlay from Windows audio sources.')
    parser.add_argument('--stt-provider', choices=['whisper', 'whisperx'], default='whisper', help='STT backend provider.')
    parser.add_argument('--stt-variant', choices=['auto', 'cpu', 'gpu'], default='auto', help='Execution variant hint for providers.')
    parser.add_argument('--stt-auto-download', dest='stt_auto_download', action='store_true', help='Allow provider presets to auto-download missing model files.')
    parser.add_argument('--no-stt-auto-download', dest='stt_auto_download', action='store_false', help='Disable provider preset auto-download behavior.')
    parser.add_argument('--model', default='small', help='Model name used by the selected STT provider.')
    parser.add_argument('--stt-model-path', default='', help='Optional model folder path for STT providers. Overrides --model when set.')
    parser.add_argument('--device', default='cuda', help='Whisper device: cuda or cpu')
    parser.add_argument('--compute-type', default='float16', help='Whisper compute type, e.g. float16, int8_float16, int8')
    parser.add_argument('--whisperx-phoneme-asr', dest='whisperx_phoneme_asr', action='store_true', help='Enable WhisperX phoneme-based ASR pipeline.')
    parser.add_argument('--no-whisperx-phoneme-asr', dest='whisperx_phoneme_asr', action='store_false', help='Disable WhisperX phoneme-based ASR pipeline.')
    parser.add_argument('--whisperx-forced-alignment', dest='whisperx_forced_alignment', action='store_true', help='Enable WhisperX forced alignment.')
    parser.add_argument('--no-whisperx-forced-alignment', dest='whisperx_forced_alignment', action='store_false', help='Disable WhisperX forced alignment.')
    parser.add_argument('--whisperx-vad', dest='whisperx_vad', action='store_true', help='Enable WhisperX internal VAD in transcription.')
    parser.add_argument('--no-whisperx-vad', dest='whisperx_vad', action='store_false', help='Disable WhisperX internal VAD in transcription.')
    parser.add_argument('--whisperx-diarization', dest='whisperx_diarization', action='store_true', help='Enable WhisperX diarization.')
    parser.add_argument('--no-whisperx-diarization', dest='whisperx_diarization', action='store_false', help='Disable WhisperX diarization.')
    parser.add_argument('--whisperx-alignment-model', default='', help='Optional WhisperX alignment model id/path.')
    parser.add_argument('--whisperx-alignment-language', choices=['auto','follow-source','en','zh-hant','zh-hans','ja','ko','de','fr','es','it','pt','ru'], default='auto', help='Alignment language override. auto=from ASR result, follow-source=use STT source language setting.')
    parser.add_argument('--whisperx-diarization-model', default='pyannote/speaker-diarization-3.1', help='WhisperX diarization model id.')
    parser.add_argument('--whisperx-hf-token', default='', help='Hugging Face token for WhisperX diarization model download/access.')
    parser.add_argument('--stt-health-check', action='store_true', help='Run STT provider health checks and exit.')
    parser.add_argument('--stt-health-check-scope', choices=['all', 'active'], default='all', help='Health-check scope when --stt-health-check is enabled.')
    parser.add_argument('--no-cpu-fallback', action='store_true', help='Disable automatic CPU fallback when CUDA initialization fails.')
    parser.add_argument('--cublas-source-dll', default='D:\\CUDA\\bin\\x64\\cublas64_13.dll', help='Path to cublas64_13.dll used to prepare cublas64_12.dll compatibility alias.')
    parser.add_argument('--ffmpeg-dll-dir', default='D:\\FFmpeg\\ffmpeg-7.1.1-full_build-shared\\bin', help='Windows FFmpeg shared-DLL directory used for torchcodec/pyannote dynamic loading.')
    parser.add_argument('--segment-seconds', type=float, default=6.0, help='Audio window length sent to STT.')
    parser.add_argument('--hop-seconds', type=float, default=1.5, help='Sliding hop interval for low-latency incremental updates.')
    parser.add_argument('--overlap-merge-method', choices=['stable-tail', 'commit-on-break', 'replace-window', 'suffix-overlap', 'fuzzy-overlap', 'append-only'], default='stable-tail', help='Merge strategy for overlapped STT windows.')
    parser.add_argument('--no-preprocess', dest='preprocess_enabled', action='store_false', help='Disable audio preprocessing before VAD/STT.')
    parser.add_argument('--preprocess-modules', default='auto', help='Comma-separated preprocessing modules: auto, none, webrtc-ns, webrtc-agc, webrtc-aec, rnnoise, spectral-gate, adaptive-gain.')
    parser.add_argument('--no-vad', dest='vad_enabled', action='store_false', help='Disable pre-transcription VAD pipeline.')
    parser.add_argument('--vad-backend', choices=['silero', 'adaptive-rms', 'rms'], default='silero', help='General VAD backend used before STT.')
    parser.add_argument('--vad-rms-threshold', type=float, default=0.008, help='RMS threshold for modular VAD gate.')
    parser.add_argument('--no-adaptive-vad', dest='vad_adaptive_enabled', action='store_false', help='Use fixed RMS VAD threshold instead of adaptive environment-noise tracking.')
    parser.add_argument('--vad-adaptive-min-threshold', type=float, default=0.004, help='Lower bound for adaptive RMS VAD threshold.')
    parser.add_argument('--vad-adaptive-max-threshold', type=float, default=0.08, help='Upper bound for adaptive RMS VAD threshold.')
    parser.add_argument('--vad-adaptive-noise-multiplier', type=float, default=2.6, help='Noise-floor multiplier used by adaptive RMS VAD.')
    parser.add_argument('--vad-adaptive-margin', type=float, default=0.002, help='Extra RMS margin added above adaptive noise floor.')
    parser.add_argument('--source-language', choices=['auto', 'en', 'zh-hant', 'zh-hans', 'ja', 'ko'], default='auto', help='STT language hint. auto uses multilingual detection.')
    parser.add_argument('--cjk-no-space-gap-seconds', type=float, default=0.2, help='When source language is Chinese, adjacent tokens within this gap are concatenated without spaces in stable/history text.')
    parser.add_argument('--max-context', '-mc', type=int, default=whisper_defaults.max_context, help='Whisper decode max context tokens (Python maps to faster-whisper max_new_tokens).')
    parser.add_argument('--entropy-thold', type=float, default=whisper_defaults.entropy_thold, help='Whisper entropy threshold (Python maps to compression_ratio_threshold).')
    parser.add_argument('--logprob-thold', type=float, default=whisper_defaults.logprob_thold, help='Whisper log probability threshold.')
    parser.add_argument('--no-speech-thold', type=float, default=whisper_defaults.no_speech_thold, help='Whisper no-speech threshold.')
    parser.add_argument('--temperature', type=float, default=whisper_defaults.temperature if whisper_defaults.temperature is not None else 0.0, help='Whisper decode temperature.')
    parser.add_argument('--beam-size', type=int, default=whisper_defaults.beam_size if whisper_defaults.beam_size is not None else 1, help='Whisper beam size.')
    parser.add_argument('--best-of', type=int, default=whisper_defaults.best_of if whisper_defaults.best_of is not None else 1, help='Whisper best-of samples.')
    parser.add_argument('--source-mode', choices=['loopback', 'microphone', 'app'], default='loopback', help='Audio source mode. app uses session-gated capture by default, or VB-CABLE if explicitly selected.')
    parser.add_argument('--ui-language', choices=['zh', 'en'], default='zh', help='UI language for tray menu and settings dialog.')
    parser.add_argument('--source-devices', default='', help='Comma-separated source device indices, e.g. 12,35')
    parser.add_argument('--app-names', default='', help='Comma-separated app names for app source mode, e.g. chrome.exe,discord.exe')
    parser.add_argument('--device-index', type=int, default=None, help='Backward-compatible single source index.')
    parser.add_argument('--list-devices', action='store_true', help='List loopback and microphone capture devices and exit.')
    parser.add_argument('--list-app-sessions', action='store_true', help='List active app audio sessions and exit.')
    parser.add_argument('--translate', action='store_true', help='Enable Argos translation.')
    parser.add_argument('--from-lang', default='auto', help='Argos source language code. Use auto to infer from installed models.')
    parser.add_argument('--to-lang', default='zh', help='Argos target language code.')
    parser.add_argument('--bilingual-style', choices=['stacked', 'translation-only'], default='stacked', help='How source and translated text should be rendered.')
    parser.add_argument('--hide-source-when-translated', action='store_true', help='Backward-compatible shortcut for --bilingual-style translation-only.')
    parser.add_argument('--overlay-width', type=int, default=1200)
    parser.add_argument('--overlay-height', type=int, default=320)
    parser.add_argument('--overlay-x', type=int, default=40)
    parser.add_argument('--overlay-y', type=int, default=700)
    parser.add_argument('--overlay-opacity', type=float, default=0.8)
    parser.add_argument('--font-size', type=int, default=18)
    parser.add_argument('--source-text-color', default='#F0F2F5')
    parser.add_argument('--translated-text-color', default='#FFD98A')
    parser.add_argument('--text-color', default='', help='Backward-compatible alias of --source-text-color')
    parser.add_argument('--background-color', default='#0A101A')
    parser.add_argument('--log-dir', default=_default_log_dir(), help='Directory for runtime log files.')
    parser.add_argument('--debug-mode', dest='debug_mode', action='store_true', help='Enable STT debug window with per-step trace.')
    parser.add_argument('--no-debug-mode', dest='debug_mode', action='store_false', help='Disable STT debug window.')
    parser.set_defaults(stt_auto_download=True, preprocess_enabled=True, vad_enabled=True, vad_adaptive_enabled=True, whisperx_phoneme_asr=True, whisperx_forced_alignment=True, whisperx_vad=False, whisperx_diarization=False, debug_mode=False)
    return parser

def parse_int_csv(raw: str) -> list[int]:
    if not raw.strip():
        return []
    values: list[int] = []
    for piece in raw.split(','):
        piece = piece.strip()
        if not piece:
            continue
        values.append(int(piece))
    return values

def parse_str_csv(raw: str) -> list[str]:
    if not raw.strip():
        return []
    values: list[str] = []
    for piece in raw.split(','):
        piece = piece.strip()
        if not piece:
            continue
        values.append(piece)
    return values

def print_devices() -> int:
    devices = list_audio_devices()
    if not devices:
        print('No capture devices found.')
        return 1
    print('Available capture devices:')
    for dev in devices:
        print(f'[{dev.index}] {dev.kind:10s} | {dev.name} | ch={dev.max_input_channels} | rate={dev.default_sample_rate}')
    return 0

def print_app_sessions() -> int:
    sessions = list_active_app_sessions()
    if not sessions:
        print('No mixer app sessions detected (or pycaw is not installed).')
        return 0
    print('Mixer app sessions:')
    for name in sessions:
        print(f'- {name}')
    return 0

def build_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    """Map parsed CLI arguments into a RuntimeConfig consumed by controller/overlay."""
    source_indices = parse_int_csv(args.source_devices)
    if args.device_index is not None and (not source_indices):
        source_indices = [args.device_index]
    app_names = parse_str_csv(args.app_names)
    bilingual_style = args.bilingual_style
    if args.hide_source_when_translated:
        bilingual_style = 'translation-only'
    source_text_color = args.source_text_color or args.text_color or '#F0F2F5'
    cfg = RuntimeConfig(model_size=args.model, model_device=args.device, compute_type=args.compute_type, stt_provider=args.stt_provider, stt_variant=args.stt_variant, stt_auto_download=args.stt_auto_download, stt_model_path=args.stt_model_path, whisperx_enable_phoneme_asr=bool(args.whisperx_phoneme_asr), whisperx_enable_forced_alignment=bool(args.whisperx_forced_alignment), whisperx_enable_vad=bool(args.whisperx_vad), whisperx_enable_diarization=bool(args.whisperx_diarization), whisperx_alignment_model=args.whisperx_alignment_model, whisperx_alignment_language=args.whisperx_alignment_language, whisperx_diarization_model=args.whisperx_diarization_model, whisperx_hf_token=args.whisperx_hf_token, cpu_fallback_on_cuda_error=not args.no_cpu_fallback, cuda_compat_source_dll=args.cublas_source_dll, ffmpeg_dll_dir=args.ffmpeg_dll_dir, segment_seconds=max(0.5, args.segment_seconds), hop_seconds=max(0.1, args.hop_seconds), source_language=None, cjk_no_space_gap_seconds=max(0.0, float(args.cjk_no_space_gap_seconds)), source_mode=args.source_mode, ui_language=args.ui_language, source_device_indices=source_indices, source_mix_weights=[], source_app_name=app_names[0] if app_names else '', source_app_names=app_names, source_channel_mode='mono', overlap_merge_method=_normalize_merge_method(args.overlap_merge_method), preprocess_enabled=bool(args.preprocess_enabled), preprocess_modules=args.preprocess_modules, vad_enabled=bool(args.vad_enabled), vad_backend=str(args.vad_backend), vad_rms_threshold=max(0.0, float(args.vad_rms_threshold)), vad_adaptive_enabled=bool(args.vad_adaptive_enabled), vad_adaptive_min_threshold=max(0.0, float(args.vad_adaptive_min_threshold)), vad_adaptive_max_threshold=max(0.0, float(args.vad_adaptive_max_threshold)), vad_adaptive_noise_multiplier=max(1.0, float(args.vad_adaptive_noise_multiplier)), vad_adaptive_margin=max(0.0, float(args.vad_adaptive_margin)), whisper_max_context=args.max_context if args.max_context and args.max_context > 0 else None, whisper_entropy_thold=args.entropy_thold, whisper_logprob_thold=args.logprob_thold, whisper_no_speech_thold=args.no_speech_thold, whisper_temperature=args.temperature, whisper_beam_size=max(1, args.beam_size), whisper_best_of=max(1, args.best_of), overlay_width=max(480, args.overlay_width), overlay_height=max(160, args.overlay_height), overlay_x=max(0, args.overlay_x), overlay_y=max(0, args.overlay_y), overlay_opacity=min(1.0, max(0.2, args.overlay_opacity)), font_size=max(10, args.font_size), text_color=source_text_color, source_text_color=source_text_color, translated_text_color=args.translated_text_color, background_color=args.background_color, translation_enabled=args.translate, translation_from=args.from_lang, translation_to=args.to_lang, bilingual_style=bilingual_style, device_index=args.device_index, log_dir=_resolve_log_dir(args.log_dir), debug_mode=bool(args.debug_mode))
    if args.source_language != 'auto':
        cfg.source_language = args.source_language
    return cfg

def _build_restart_keys() -> set[str]:
    return {'stt_provider', 'stt_variant', 'stt_auto_download', 'stt_model_path', 'model_size', 'model_device', 'compute_type',  'whisperx_enable_phoneme_asr', 'whisperx_enable_forced_alignment', 'whisperx_enable_vad', 'whisperx_vad_method', 'whisperx_enable_diarization', 'whisperx_alignment_model', 'whisperx_alignment_language', 'whisperx_diarization_model', 'whisperx_hf_token', 'ffmpeg_dll_dir', 'source_mode', 'source_device_indices', 'source_app_name', 'source_app_names', 'source_language', 'segment_seconds', 'hop_seconds', 'overlap_merge_method', 'preprocess_enabled', 'preprocess_modules', 'vad_enabled', 'vad_rms_threshold', 'vad_adaptive_enabled', 'vad_adaptive_min_threshold', 'vad_adaptive_max_threshold', 'vad_adaptive_noise_multiplier', 'vad_adaptive_margin',  'translation_enabled', 'bilingual_style', 'translation_from', 'translation_to', 'debug_mode'}

def run_qt_app(cfg: RuntimeConfig) -> int:
    """Create overlay + controller + tray, wire signals, and run Qt event loop."""
    logger = configure_app_logger(cfg.log_dir)
    logger.info('Voice2Text startup')
    _configure_windows_dll_search_paths(cfg, on_status=lambda msg: logger.info(msg))
    if cfg.stt_provider == 'whisper' and cfg.model_device.lower().startswith('cuda'):
        ok = ensure_cublas12_from_source(source_dll=cfg.cuda_compat_source_dll, on_status=lambda msg: logger.info(msg))
        if ok:
            logger.info('CUDA compatibility preparation completed.')
        else:
            logger.warning('CUDA compatibility preparation failed. Runtime may fallback to CPU.')
    _set_windows_app_user_model_id()
    app = QApplication(sys.argv)
    app.setApplicationName('Voice2Text Python')
    app.setQuitOnLastWindowClosed(False)
    overlay = SubtitleOverlayWindow(cfg)
    controller = TranscriptionController(cfg, logger=logger)
    debug_log_dir = str(Path(cfg.log_dir).resolve().parent / "debug_logs")
    debug_window_holder: dict[str, STTDebugWindow | None] = {"window": None}

    def ensure_debug_window_state() -> None:
        enabled = bool(getattr(cfg, "debug_mode", False))
        window = debug_window_holder.get("window")
        if enabled:
            if window is None:
                window = STTDebugWindow(debug_log_dir=debug_log_dir)
                controller.debug_event.connect(window.append_event)
                debug_window_holder["window"] = window
            if not window.isVisible():
                window.show()
                window.raise_()
                window.activateWindow()
        else:
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
            overlay.push_status('Capture stopped.')
        else:
            controller.start()
            overlay.push_status('Capture resumed.')

    overlay.toggle_runtime_requested.connect(toggle_runtime)
    restart_keys = _build_restart_keys()
    tray_holder: dict[str, Voice2TextTrayController] = {}

    def apply_settings(updates: dict[str, object]) -> None:
        requires_restart = False
        for (key, value) in updates.items():
            if not hasattr(cfg, key):
                continue
            if getattr(cfg, key) == value:
                continue
            setattr(cfg, key, value)
            if key in restart_keys:
                requires_restart = True
        overlay.apply_runtime_config(cfg)
        tray = tray_holder.get('tray')
        if tray is not None and 'ui_language' in updates:
            tray.refresh_locale()
        logger.info('Settings updated: %s', updates)
        ensure_debug_window_state()
        if requires_restart:

            def restart_capture() -> None:
                try:
                    controller.restart()
                    overlay.push_status(f'Runtime settings applied. Capture restarted. model={_effective_model_label(cfg)}')
                except Exception as exc:
                    logger.exception('Capture restart failed after settings update')
                    overlay.push_error(f'Capture restart failed: {exc}')
            QTimer.singleShot(0, restart_capture)
        else:
            overlay.push_status(f'UI settings applied. model={_effective_model_label(cfg)}')
    tray_holder['tray'] = Voice2TextTrayController(app=app, overlay=overlay, config=cfg, on_settings_applied=apply_settings)
    app.aboutToQuit.connect(controller.stop)
    overlay.show()
    controller.start()
    return app.exec()

def main(argv: list[str] | None=None) -> int:
    """Application bootstrap entry used by src/main.py and package main()."""
    config_path = Path(__file__).resolve().parents[1] / 'whisper_config.json'
    whisper_defaults = load_whisper_runtime_params(config_path)
    args = build_arg_parser(whisper_defaults).parse_args(argv)
    if args.list_devices:
        return print_devices()
    if args.list_app_sessions:
        return print_app_sessions()
    cfg = build_runtime_config(args)
    if args.stt_health_check:
        reports = run_provider_health_check(cfg, scope=args.stt_health_check_scope)
        print(summarize_health_reports(reports))
        return 2 if has_failed_reports(reports) else 0
    return run_qt_app(cfg)










