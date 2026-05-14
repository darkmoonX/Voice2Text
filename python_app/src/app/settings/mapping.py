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
    sherpa_onnx_provider: str
    riva_uri: str
    riva_use_ssl: bool
    riva_ssl_cert: str
    riva_language_code: str
    riva_api_key: str
    funasr_device: str
    funasr_vad_model: str
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


def build_settings_updates(payload: SettingsPayloadInput, *, lang: str, need_riva_uri_message: str, hop_gt_segment_message: str) -> dict[str, object]:
    stt_provider = payload.stt_provider.strip() or 'whisper'
    stt_model_path = payload.stt_model_path.strip()
    if stt_provider == 'riva' and not payload.riva_uri.strip():
        raise ValueError(need_riva_uri_message)

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
        'sherpa_onnx_provider': payload.sherpa_onnx_provider or 'cpu',
        'riva_uri': payload.riva_uri.strip() or 'localhost:50051',
        'riva_use_ssl': bool(payload.riva_use_ssl),
        'riva_ssl_cert': payload.riva_ssl_cert.strip(),
        'riva_language_code': payload.riva_language_code.strip() or 'en-US',
        'riva_api_key': payload.riva_api_key.strip(),
        'funasr_device': payload.funasr_device.strip() or 'cpu',
        'funasr_vad_model': payload.funasr_vad_model.strip(),
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
    }
