"""Audio capture factory adapter.

Prefers the C++ capture bridge for loopback/app modes and falls back to the
legacy Python backend for unsupported scenarios.
"""
from __future__ import annotations

from typing import Callable, Optional

from ..audio_capture import AudioCaptureBase, AudioChunk, build_capture_from_config as _build_python_capture
from .cpp_backend import build_cpp_capture_from_config


def build_capture_from_config(config, on_error: Optional[Callable[[str], None]] = None, on_status: Optional[Callable[[str], None]] = None) -> AudioCaptureBase:
    cpp_capture = build_cpp_capture_from_config(config, on_error=on_error, on_status=on_status)
    if cpp_capture is not None:
        return cpp_capture
    return _build_python_capture(config, on_error=on_error, on_status=on_status)


__all__ = ["AudioCaptureBase", "AudioChunk", "build_capture_from_config"]
