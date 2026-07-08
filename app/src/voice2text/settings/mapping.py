"""Mapping and validation helpers for settings dialog payload."""
from __future__ import annotations

from dataclasses import dataclass

from ..stt.registry import normalize_stt_provider


@dataclass
class SettingsPayloadInput:
    ui_language: str
    source_mode: str
    stt_provider: str
    stt_variant: str
    compute_type: str
    stt_model_path: str
    stt_auto_download: bool
    whisperx_enable_phoneme_asr: bool
    whisperx_enable_forced_alignment: bool
    whisperx_enable_vad: bool
    whisperx_vad_method: str
    whisperx_enable_diarization: bool
    whisperx_alignment_model: str
    whisperx_alignment_language: str
    whisperx_alignment_device: str
    whisperx_align_guard: str
    whisperx_diarization_device: str
    whisperx_diarization_model: str
    whisperx_diarization_expected_speakers: int
    whisperx_hf_token: str
    whisperx_speaker_profile_backend: str
    source_language: str
    translation_to: str
    segment_seconds: float
    hop_seconds: float
    selected_loopback_indices: list[int]
    selected_app_names: list[str]
    overlap_merge_method: str
    preprocess_enabled: bool
    preprocess_modules: str
    translation_enabled: bool
    translation_backend: str
    bilingual_style: str
    font_size: int
    overlay_opacity: float
    source_text_color: str
    translated_text_color: str
    background_color: str
    debug_mode: bool
    transcript_export_enabled: bool
    transcript_export_formats: str
    transcript_export_include_timestamps: bool
    transcript_export_include_speaker: bool
    # Preset-bundled runtime knobs now editable from the dialog (round 0015 Phase C).
    # Defaulted so older callers/tests that omit them keep working.
    model_size: str = "auto"
    whisper_beam_size: int = 5
    whisperx_speaker_profile_enabled: bool = True
    runtime_preset: str = ""
    # Round 0051: recently-shipped live knobs surfaced in the dialog (previously CLI/JSON-only).
    whisperx_zh_align_wbbbbb: bool = False
    whisperx_asr_temperatures: str = ""
    subtitle_commit_hold_seconds: float = 0.0
    asr_temperatures_invalid_message: str = "Invalid ASR temperature schedule."
    # Round 0076: session recording + finalize direct-relabel surfaced in the dialog
    # (previously CLI/JSON-only, round 0047).
    session_record_enabled: bool = False
    session_finalize_direct_relabel_enabled: bool = False


def build_settings_updates(payload: SettingsPayloadInput, *, lang: str, hop_gt_segment_message: str) -> dict[str, object]:
    stt_provider = normalize_stt_provider(payload.stt_provider)
    stt_model_path = payload.stt_model_path.strip()
    whispercpp_model_path = stt_model_path if (stt_provider == 'whispercpp' and _is_whispercpp_model_path(stt_model_path)) else ''

    source_lang_data = payload.source_language
    if source_lang_data == 'auto':
        translation_from = 'auto'
    elif source_lang_data in {'zh-hant', 'zh-hans'}:
        translation_from = 'zh'
    else:
        translation_from = str(source_lang_data)

    segment_seconds = float(payload.segment_seconds)
    hop_seconds = float(payload.hop_seconds)
    if hop_seconds > segment_seconds:
        raise ValueError(hop_gt_segment_message)

    # Round 0051: lightweight local validation of the temperature schedule string (mirrors the
    # provider's parser semantics without importing the heavy STT module into the Qt process).
    # Empty = library default; garbage is rejected here so the dialog can warn instead of the
    # provider silently ignoring it at model-load time.
    asr_temperatures = str(payload.whisperx_asr_temperatures or "").strip()
    if asr_temperatures:
        parts = [p.strip() for p in asr_temperatures.split(",") if p.strip()]
        try:
            values = [float(p) for p in parts]
        except ValueError:
            raise ValueError(payload.asr_temperatures_invalid_message) from None
        if not values or any((v < 0.0 or v > 1.0) for v in values):
            raise ValueError(payload.asr_temperatures_invalid_message)

    commit_hold_seconds = float(max(0.0, min(120.0, float(payload.subtitle_commit_hold_seconds or 0.0))))

    compute_type = (payload.compute_type or "float16").strip().lower()
    if compute_type not in {"float16", "int8_float16", "int8"}:
        compute_type = "float16"

    source_device_indices: list[int] = []
    source_app_names: list[str] = []
    if payload.source_mode == 'loopback':
        source_device_indices = list(payload.selected_loopback_indices)
    elif payload.source_mode == 'app':
        source_app_names = list(payload.selected_app_names)

    alignment_device = (payload.whisperx_alignment_device or "auto").strip().lower()
    if alignment_device not in {"auto", "cpu", "cuda"}:
        alignment_device = "auto"
    align_guard = (payload.whisperx_align_guard or "safe").strip().lower().replace("_", "-")
    if align_guard not in {"safe", "unsafe-cuda", "probe"}:
        align_guard = "safe"
    diarization_device = (payload.whisperx_diarization_device or "auto").strip().lower()
    if diarization_device not in {"auto", "cpu", "cuda"}:
        diarization_device = "auto"
    expected_speakers = max(0, min(20, int(payload.whisperx_diarization_expected_speakers or 0)))
    speaker_profile_backend = (payload.whisperx_speaker_profile_backend or "pyannote").strip().lower()
    if speaker_profile_backend not in {"pyannote", "wespeaker", "speechbrain_ecapa", "nemo_titanet"}:
        speaker_profile_backend = "pyannote"
    export_formats: list[str] = []
    for token in str(payload.transcript_export_formats or "").split(","):
        item = token.strip().lower()
        if item in {"txt", "srt", "json"} and item not in export_formats:
            export_formats.append(item)
    if not export_formats:
        export_formats = ["txt", "srt", "json"]

    return {
        'ui_language': lang,
        'stt_provider': stt_provider,
        'runtime_preset': (payload.runtime_preset or '').strip(),
        'stt_variant': payload.stt_variant or 'auto',
        'model_size': (payload.model_size or 'auto').strip() or 'auto',
        # whisper.cpp resolves ggml-<size>.bin from this literal name, so the whisperx-only
        # 'auto' sentinel must never leak through (it would look for ggml-auto.bin).
        'stt_whispercpp_model_size': _whispercpp_model_size(payload.model_size),
        'stt_whispercpp_model_path': whispercpp_model_path,
        'whisper_beam_size': max(1, int(payload.whisper_beam_size or 5)),
        'whisperx_speaker_profile_enabled': bool(payload.whisperx_speaker_profile_enabled),
        'compute_type': compute_type,
        'stt_auto_download': bool(payload.stt_auto_download),
        'stt_model_path': stt_model_path,
        'whisperx_enable_phoneme_asr': bool(payload.whisperx_enable_phoneme_asr),
        'whisperx_enable_forced_alignment': bool(payload.whisperx_enable_forced_alignment),
        'whisperx_enable_vad': True,
        'whisperx_vad_method': (payload.whisperx_vad_method.strip() or 'silero-vad'),
        'whisperx_enable_diarization': bool(payload.whisperx_enable_diarization),
        'whisperx_alignment_model': payload.whisperx_alignment_model.strip(),
        'whisperx_alignment_language': payload.whisperx_alignment_language.strip() or 'auto',
        'whisperx_alignment_device': alignment_device,
        'whisperx_align_guard': align_guard,
        'whisperx_diarization_device': diarization_device,
        'whisperx_diarization_model': payload.whisperx_diarization_model.strip() or 'pyannote/speaker-diarization-3.1',
        'whisperx_diarization_min_speakers': expected_speakers,
        'whisperx_diarization_max_speakers': expected_speakers,
        'whisperx_hf_token': payload.whisperx_hf_token.strip(),
        'whisperx_speaker_profile_backend': speaker_profile_backend,
        'source_mode': payload.source_mode,
        'source_device_indices': source_device_indices,
        'source_app_name': source_app_names[0] if source_app_names else '',
        'source_app_names': source_app_names,
        'source_language': None if payload.source_language == 'auto' else payload.source_language,
        'segment_seconds': segment_seconds,
        'hop_seconds': hop_seconds,
        'overlap_merge_method': payload.overlap_merge_method,
        'preprocess_enabled': bool(payload.preprocess_enabled),
        'preprocess_modules': payload.preprocess_modules.strip() or 'auto',
        'translation_enabled': bool(payload.translation_enabled),
        'translation_backend': _normalize_translation_backend(payload.translation_backend),
        'translation_from': translation_from,
        'translation_to': payload.translation_to,
        'bilingual_style': payload.bilingual_style,
        'font_size': int(payload.font_size),
        'overlay_opacity': float(payload.overlay_opacity),
        'text_color': payload.source_text_color.strip(),
        'source_text_color': payload.source_text_color.strip(),
        'translated_text_color': payload.translated_text_color.strip(),
        'background_color': payload.background_color.strip(),
        'debug_mode': bool(payload.debug_mode),
        'transcript_export_enabled': bool(payload.transcript_export_enabled),
        'transcript_export_formats': ",".join(export_formats),
        'transcript_export_include_timestamps': bool(payload.transcript_export_include_timestamps),
        'transcript_export_include_speaker': bool(payload.transcript_export_include_speaker),
        'whisperx_zh_align_wbbbbb': bool(payload.whisperx_zh_align_wbbbbb),
        'whisperx_asr_temperatures': asr_temperatures,
        'subtitle_commit_hold_seconds': commit_hold_seconds,
        'session_record_enabled': bool(payload.session_record_enabled),
        # Relabel is a no-op without recording; keep it off if recording isn't on so a
        # stale True in the payload can't silently activate later if recording is re-enabled.
        'session_finalize_direct_relabel_enabled': bool(
            payload.session_finalize_direct_relabel_enabled and payload.session_record_enabled
        ),
    }


def _normalize_translation_backend(value: str) -> str:
    token = str(value or "argos").strip().lower()
    if token in {"argos", "nllb", "llm", "cloud"}:
        return token
    return "argos"


def _whispercpp_model_size(value: str) -> str:
    """whisper.cpp size from the shared model field; 'auto' is whisperx-only (round 0072)."""
    token = str(value or "").strip()
    if not token or token.lower() == "auto":
        return "medium"
    return token


def _is_whispercpp_model_path(value: str) -> bool:
    token = str(value or "").strip()
    if not token:
        return False
    return token.lower().endswith(".bin") or any((ch in token for ch in ('/', '\\'))) or token.startswith('.')
