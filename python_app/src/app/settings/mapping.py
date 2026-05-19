"""Mapping and validation helpers for settings dialog payload."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SettingsPayloadInput:
    ui_language: str
    source_mode: str
    stt_provider: str
    stt_variant: str
    stt_model_path: str
    stt_auto_download: bool
    whisperx_enable_phoneme_asr: bool
    whisperx_enable_forced_alignment: bool
    whisperx_enable_vad: bool
    whisperx_vad_method: str
    whisperx_enable_diarization: bool
    whisperx_alignment_model: str
    whisperx_alignment_language: str
    whisperx_diarization_model: str
    whisperx_hf_token: str
    source_language: str
    translation_to: str
    segment_seconds: float
    hop_seconds: float
    selected_loopback_indices: list[int]
    selected_app_names: list[str]
    overlap_merge_method: str
    preprocess_enabled: bool
    preprocess_modules: str
    vad_adaptive_enabled: bool
    vad_rms_threshold: float
    translation_enabled: bool
    bilingual_style: str
    font_size: int
    overlay_opacity: float
    source_text_color: str
    translated_text_color: str
    background_color: str
    debug_mode: bool


def build_settings_updates(payload: SettingsPayloadInput, *, lang: str, hop_gt_segment_message: str) -> dict[str, object]:
    stt_provider = payload.stt_provider.strip() or 'whisper'
    stt_model_path = payload.stt_model_path.strip()

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

    source_device_indices: list[int] = []
    source_app_names: list[str] = []
    if payload.source_mode == 'loopback':
        source_device_indices = list(payload.selected_loopback_indices)
    elif payload.source_mode == 'app':
        source_app_names = list(payload.selected_app_names)

    return {
        'ui_language': lang,
        'stt_provider': stt_provider,
        'stt_variant': payload.stt_variant or 'auto',
        'stt_auto_download': bool(payload.stt_auto_download),
        'stt_model_path': stt_model_path,
        'whisperx_enable_phoneme_asr': bool(payload.whisperx_enable_phoneme_asr),
        'whisperx_enable_forced_alignment': bool(payload.whisperx_enable_forced_alignment),
        'whisperx_enable_vad': True,
        'whisperx_vad_method': (payload.whisperx_vad_method.strip() or 'silero-vad'),
        'whisperx_enable_diarization': bool(payload.whisperx_enable_diarization),
        'whisperx_alignment_model': payload.whisperx_alignment_model.strip(),
        'whisperx_alignment_language': payload.whisperx_alignment_language.strip() or 'auto',
        'whisperx_diarization_model': payload.whisperx_diarization_model.strip() or 'pyannote/speaker-diarization-3.1',
        'whisperx_hf_token': payload.whisperx_hf_token.strip(),
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
        'vad_adaptive_enabled': bool(payload.vad_adaptive_enabled),
        'vad_rms_threshold': float(payload.vad_rms_threshold),
        'translation_enabled': bool(payload.translation_enabled),
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
    }

