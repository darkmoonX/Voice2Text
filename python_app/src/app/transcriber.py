"""Compatibility re-export for legacy imports of the Whisper transcriber."""
from __future__ import annotations

from .stt.whisper_provider import FasterWhisperTranscriber

__all__ = ["FasterWhisperTranscriber"]

