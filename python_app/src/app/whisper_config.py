"""Compatibility re-export for legacy imports of Whisper runtime config parsing."""
from __future__ import annotations

from .stt.whisper_provider import WhisperRuntimeParams, load_whisper_runtime_params

__all__ = ["WhisperRuntimeParams", "load_whisper_runtime_params"]
