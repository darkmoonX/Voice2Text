"""RuntimeConfig construction from parsed CLI args."""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import RuntimeConfig
from .bootstrap_args import parse_int_csv, parse_str_csv


def default_log_dir() -> str:
    return str((Path(__file__).resolve().parents[1] / "logs"))


def resolve_log_dir(raw_path: str) -> str:
    candidate = Path(raw_path.strip()) if raw_path and raw_path.strip() else Path(default_log_dir())
    if not candidate.is_absolute():
        candidate = Path(__file__).resolve().parents[1] / candidate
    return str(candidate)


def normalize_merge_method(raw_method: str) -> str:
    method = (raw_method or "").strip().lower()
    if method in {"stable-tail", "replace-window", "suffix-overlap", "fuzzy-overlap"}:
        return "stable-tail"
    if method in {"commit-on-break", "append-only"}:
        return "commit-on-break"
    return "stable-tail"


def build_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    source_indices = parse_int_csv(args.source_devices)
    if args.device_index is not None and (not source_indices):
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
        stt_provider="whisperx",
        stt_variant=args.stt_variant,
        stt_auto_download=args.stt_auto_download,
        stt_model_path=args.stt_model_path,
        whisperx_enable_phoneme_asr=bool(args.whisperx_phoneme_asr),
        whisperx_enable_forced_alignment=bool(args.whisperx_forced_alignment),
        whisperx_enable_vad=bool(args.whisperx_vad),
        whisperx_enable_diarization=bool(args.whisperx_diarization),
        whisperx_alignment_model=args.whisperx_alignment_model,
        whisperx_alignment_language=args.whisperx_alignment_language,
        whisperx_alignment_device=args.whisperx_alignment_device,
        whisperx_diarization_device=args.whisperx_diarization_device,
        whisperx_diarization_model=args.whisperx_diarization_model,
        whisperx_hf_token=args.whisperx_hf_token,
        cpu_fallback_on_cuda_error=not args.no_cpu_fallback,
        cuda_compat_source_dll=args.cublas_source_dll,
        ffmpeg_dll_dir=args.ffmpeg_dll_dir,
        segment_seconds=max(0.5, args.segment_seconds),
        hop_seconds=max(0.1, args.hop_seconds),
        source_language=None,
        cjk_no_space_gap_seconds=max(0.0, float(args.cjk_no_space_gap_seconds)),
        speaker_pause_break_seconds=max(0.0, float(args.speaker_pause_break_seconds)),
        subtitle_display_script=('' if str(getattr(args, 'subtitle_display_script', 'hant')) == 'off' else str(getattr(args, 'subtitle_display_script', 'hant'))),
        source_mode=args.source_mode,
        source_file_path=str(args.source_file or ""),
        source_file_replay_speed=max(0.0, float(args.source_file_replay_speed)),
        source_file_chunk_seconds=max(0.02, float(args.source_file_chunk_seconds)),
        ui_language=args.ui_language,
        source_device_indices=source_indices,
        source_mix_weights=[],
        source_app_name=app_names[0] if app_names else "",
        source_app_names=app_names,
        source_channel_mode="mono",
        overlap_merge_method=normalize_merge_method(args.overlap_merge_method),
        preprocess_enabled=bool(args.preprocess_enabled),
        preprocess_modules=args.preprocess_modules,
        whisper_max_context=args.max_context if args.max_context and args.max_context > 0 else None,
        whisper_entropy_thold=args.entropy_thold,
        whisper_logprob_thold=args.logprob_thold,
        whisper_no_speech_thold=args.no_speech_thold,
        whisper_temperature=args.temperature,
        whisper_beam_size=max(1, args.beam_size),
        whisper_batch_size=max(1, args.batch_size),
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
        log_dir=resolve_log_dir(args.log_dir),
        debug_mode=bool(args.debug_mode),
        transcript_export_enabled=bool(args.transcript_export_enabled),
        transcript_export_formats=str(args.transcript_export_formats or "txt,srt,json"),
        transcript_export_include_timestamps=bool(args.transcript_export_include_timestamps),
        transcript_export_include_speaker=bool(args.transcript_export_include_speaker),
        transcript_export_dir=str(args.transcript_export_dir or ""),
    )
    if args.source_language != "auto":
        cfg.source_language = args.source_language
    return cfg
