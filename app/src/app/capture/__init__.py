"""AudioSource seam for discovery and capture factory.

External callers should import from this package instead of reaching into
`audio_capture.py` directly.
"""
from __future__ import annotations

from .discovery import AudioDevice, LoopbackDevice, list_active_app_sessions, list_audio_devices, list_loopback_devices
from .factory import AudioCaptureBase, AudioChunk, build_capture_from_config

__all__ = [
    'AudioCaptureBase',
    'AudioChunk',
    'AudioDevice',
    'LoopbackDevice',
    'build_capture_from_config',
    'list_active_app_sessions',
    'list_audio_devices',
    'list_loopback_devices',
]

