"""AudioSource seam for discovery and capture factory.

External callers should import from this package instead of reaching into
`audio_capture.py` directly.
"""
from __future__ import annotations

from typing import Any

__all__ = [
    "AudioCaptureBase",
    "AudioChunk",
    "AudioDevice",
    "LoopbackDevice",
    "build_capture_from_config",
    "list_active_app_sessions",
    "list_audio_devices",
    "list_loopback_devices",
]


def __getattr__(name: str) -> Any:
    # Lazy exports avoid circular imports while audio_capture initializes.
    if name in {"AudioCaptureBase", "AudioChunk", "build_capture_from_config"}:
        from .factory import AudioCaptureBase, AudioChunk, build_capture_from_config

        mapping = {
            "AudioCaptureBase": AudioCaptureBase,
            "AudioChunk": AudioChunk,
            "build_capture_from_config": build_capture_from_config,
        }
        return mapping[name]

    if name in {"AudioDevice", "LoopbackDevice", "list_active_app_sessions", "list_audio_devices", "list_loopback_devices"}:
        from .discovery import AudioDevice, LoopbackDevice, list_active_app_sessions, list_audio_devices, list_loopback_devices

        mapping = {
            "AudioDevice": AudioDevice,
            "LoopbackDevice": LoopbackDevice,
            "list_active_app_sessions": list_active_app_sessions,
            "list_audio_devices": list_audio_devices,
            "list_loopback_devices": list_loopback_devices,
        }
        return mapping[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
