"""Audio device and session discovery adapters.

This module intentionally re-exports implementations from `audio_capture`
so callers can depend on a stable seam while internals are refactored.
"""
from __future__ import annotations

from ..audio_capture import AudioDevice, LoopbackDevice, list_active_app_sessions, list_audio_devices, list_loopback_devices

__all__ = [
    'AudioDevice',
    'LoopbackDevice',
    'list_active_app_sessions',
    'list_audio_devices',
    'list_loopback_devices',
]

