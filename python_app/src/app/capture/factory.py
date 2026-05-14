"""Audio capture factory adapter.

The first architecture step keeps runtime behavior in `audio_capture` but routes
all consumers through this seam.
"""
from __future__ import annotations

from ..audio_capture import AudioCaptureBase, AudioChunk, build_capture_from_config

__all__ = ['AudioCaptureBase', 'AudioChunk', 'build_capture_from_config']

