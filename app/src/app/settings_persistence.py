"""Persistent settings storage for runtime configuration restored across launches."""
from __future__ import annotations

import json
from pathlib import Path

from .config import RuntimeConfig

_PERSIST_FILE_NAME = "runtime_settings.json"
_PERSIST_KEYS = {
    "ui_language",
    "stt_provider",
    "stt_variant",
    "stt_auto_download",
    "stt_model_path",
    "whisperx_enable_phoneme_asr",
    "whisperx_enable_forced_alignment",
    "whisperx_enable_vad",
    "whisperx_vad_method",
    "whisperx_enable_diarization",
    "whisperx_alignment_model",
    "whisperx_alignment_language",
    "whisperx_alignment_device",
    "whisperx_diarization_device",
    "whisperx_diarization_model",
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
    "speaker_marker_style",
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
    "translation_from",
    "translation_to",
    "bilingual_style",
    "font_size",
    "overlay_opacity",
    "text_color",
    "source_text_color",
    "translated_text_color",
    "background_color",
    "debug_mode",
    "transcript_export_enabled",
    "transcript_export_formats",
    "transcript_export_include_timestamps",
    "transcript_export_include_speaker",
    "transcript_export_dir",
}


def settings_file_path() -> Path:
    return Path(__file__).resolve().parents[1] / _PERSIST_FILE_NAME


def load_persisted_updates(path: Path | None = None) -> dict[str, object]:
    target = path or settings_file_path()
    if not target.exists():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    updates: dict[str, object] = {}
    for key in _PERSIST_KEYS:
        if key in payload:
            updates[key] = payload[key]
    return updates


def apply_updates_to_config(cfg: RuntimeConfig, updates: dict[str, object]) -> set[str]:
    changed: set[str] = set()
    for (key, value) in updates.items():
        if key not in _PERSIST_KEYS:
            continue
        if not hasattr(cfg, key):
            continue
        if getattr(cfg, key) == value:
            continue
        setattr(cfg, key, value)
        changed.add(key)
    return changed


def save_runtime_settings(cfg: RuntimeConfig, path: Path | None = None) -> Path:
    target = path or settings_file_path()
    payload: dict[str, object] = {}
    for key in sorted(_PERSIST_KEYS):
        if hasattr(cfg, key):
            payload[key] = getattr(cfg, key)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
