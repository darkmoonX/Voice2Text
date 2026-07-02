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
        cpu_threads=int(getattr(args, "cpu_threads", 0) or 0),
        stt_provider=str(getattr(args, "stt_provider", "whisperx") or "whisperx"),
        stt_variant=args.stt_variant,
        stt_auto_download=args.stt_auto_download,
        stt_model_path=args.stt_model_path,
        stt_whispercpp_model_path=str(getattr(args, "whispercpp_model_path", "") or ""),
        stt_whispercpp_model_size=str(getattr(args, "whispercpp_model_size", "medium") or "medium"),
        stt_whispercpp_binary_path=str(getattr(args, "whispercpp_binary_path", "") or ""),
        stt_whispercpp_server_path=str(getattr(args, "whispercpp_server_path", "") or ""),
        stt_whispercpp_mode=str(getattr(args, "whispercpp_mode", "server") or "server"),
        stt_whispercpp_server_vad=bool(getattr(args, "whispercpp_server_vad", False)),
        stt_whispercpp_vad_model_path=str(getattr(args, "whispercpp_vad_model_path", "") or ""),
        stt_whispercpp_vad_model=str(getattr(args, "whispercpp_vad_model", "ggml-silero-v5.1.2.bin") or "ggml-silero-v5.1.2.bin"),
        stt_whispercpp_server_max_len=max(0, int(getattr(args, "whispercpp_server_max_len", 0) or 0)),
        stt_whispercpp_request_timeout_seconds=max(0.5, float(getattr(args, "whispercpp_request_timeout", 30.0) or 30.0)),
        stt_whispercpp_no_speech_threshold=float(getattr(args, "whispercpp_no_speech_threshold", 0.85) or 0.85),
        stt_whispercpp_avg_logprob_min=float(getattr(args, "whispercpp_avg_logprob_min", -1.2) or -1.2),
        stt_whispercpp_repetition_similarity=float(getattr(args, "whispercpp_repetition_similarity", 0.92) or 0.92),
        stt_whispercpp_boilerplate_phrases=str(getattr(args, "whispercpp_boilerplate_phrases", "") or ""),
        whisperx_enable_phoneme_asr=bool(args.whisperx_phoneme_asr),
        whisperx_enable_forced_alignment=bool(args.whisperx_forced_alignment),
        whisperx_enable_vad=bool(args.whisperx_vad),
        whisperx_enable_diarization=bool(args.whisperx_diarization),
        whisperx_speaker_profile_enabled=bool(getattr(args, "whisperx_speaker_profile", True)),
        whisperx_speaker_realtime_refresh_seconds=max(0.0, float(getattr(args, "speaker_realtime_refresh_seconds", 0.0) or 0.0)),
        whisperx_speaker_realtime_refresh_alpha=max(
            0.0,
            min(
                1.0,
                float(
                    0.5 if getattr(args, "speaker_realtime_refresh_alpha", 0.5) is None
                    else getattr(args, "speaker_realtime_refresh_alpha", 0.5)
                ),
            ),
        ),
        whisperx_speaker_realtime_refresh_assign_threshold=max(
            0.0,
            min(
                0.999,
                float(
                    0.55 if getattr(args, "speaker_realtime_refresh_assign_threshold", 0.55) is None
                    else getattr(args, "speaker_realtime_refresh_assign_threshold", 0.55)
                ),
            ),
        ),
        whisperx_speaker_realtime_refresh_min_cluster_seconds=max(
            0.0,
            float(
                4.0 if getattr(args, "speaker_realtime_refresh_min_cluster_seconds", 4.0) is None
                else getattr(args, "speaker_realtime_refresh_min_cluster_seconds", 4.0)
            ),
        ),
        whisperx_speaker_realtime_refresh_merge=bool(getattr(args, "speaker_realtime_refresh_merge", True)),
        whisperx_speaker_realtime_refresh_match_mode=str(
            getattr(args, "speaker_realtime_refresh_match_mode", "argmax") or "argmax"
        ),
        whisperx_speaker_profile_quality_gate_enabled=bool(getattr(args, "whisperx_speaker_profile_quality_gate_enabled", False)),
        runtime_preset=str(getattr(args, "preset", "") or ""),
        whisperx_alignment_model=args.whisperx_alignment_model,
        whisperx_english_align_large=bool(getattr(args, "whisperx_english_align_large", True)),
        whisperx_zh_align_wbbbbb=bool(getattr(args, "whisperx_zh_align_wbbbbb", False)),
        whisperx_alignment_language=args.whisperx_alignment_language,
        whisperx_alignment_device=args.whisperx_alignment_device,
        whisperx_align_guard=str(getattr(args, "whisperx_align_guard", "safe") or "safe"),
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
        subtitle_commit_hold_seconds=max(0.0, float(getattr(args, 'subtitle_commit_hold_seconds', 0.0) or 0.0)),
        subtitle_reanchor_stabilization=str(getattr(args, 'subtitle_reanchor_stabilization', 'consecutive') or 'consecutive'),
        subtitle_relabel_enabled=bool(getattr(args, 'subtitle_relabel_enabled', False)),
        subtitle_relabel_window_seconds=max(1.0, float(getattr(args, 'subtitle_relabel_window_seconds', 20.0) or 20.0)),
        subtitle_relabel_sliver_floor_seconds=max(0.0, float(getattr(args, 'subtitle_relabel_sliver_floor_seconds', 1.5) or 0.0)),
        subtitle_relabel_assign_threshold=max(0.0, min(0.999, float(getattr(args, 'subtitle_relabel_assign_threshold', 0.65) or 0.65))),
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
        whisperx_rolling_prompt_chars=max(0, int(getattr(args, 'whisperx_rolling_prompt_chars', 0) or 0)),
        whisperx_asr_temperatures=str(getattr(args, 'asr_temperatures', '') or ''),
        whisperx_asr_log_prob_threshold=getattr(args, 'asr_log_prob_threshold', None),
        whisperx_asr_compression_ratio_threshold=getattr(args, 'asr_compression_ratio_threshold', None),
        whisperx_asr_no_speech_threshold=getattr(args, 'asr_no_speech_threshold', None),
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
        translation_backend=str(getattr(args, "translation_backend", "argos") or "argos"),
        translation_nllb_auto_convert=bool(getattr(args, "translation_nllb_auto_convert", True)),
        translation_queue_max=int(getattr(args, "translation_queue_max", 0) or 0),
        translation_request_timeout_seconds=float(getattr(args, "translation_timeout", 8.0) or 8.0),
        translation_max_retries=int(getattr(args, "translation_max_retries", 0) or 0),
        bilingual_style=bilingual_style,
        device_index=args.device_index,
        log_dir=resolve_log_dir(args.log_dir),
        debug_mode=bool(args.debug_mode),
        session_record_enabled=bool(getattr(args, "session_record_enabled", False)),
        session_finalize_direct_relabel_enabled=bool(
            getattr(args, "session_finalize_direct_relabel_enabled", False)
        ),
        transcript_export_enabled=bool(args.transcript_export_enabled),
        transcript_export_formats=str(args.transcript_export_formats or "txt,srt,json"),
        transcript_export_include_timestamps=bool(args.transcript_export_include_timestamps),
        transcript_export_include_speaker=bool(args.transcript_export_include_speaker),
        transcript_export_dir=str(args.transcript_export_dir or ""),
        import_direct_path=str(getattr(args, "import_direct", "") or ""),
        import_direct_chunk_seconds=max(0.0, float(getattr(args, "import_direct_chunk_seconds", 0.0) or 0.0)),
        import_direct_language_subchunk_seconds=max(
            0.0,
            float(getattr(args, "import_direct_language_subchunk_seconds", 30.0) or 30.0),
        ),
    )
    if args.source_language != "auto":
        cfg.source_language = args.source_language
    return cfg
