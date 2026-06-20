"""Rules that decide which status lines should be surfaced in the overlay."""
from __future__ import annotations

_SUPPRESSED_OVERLAY_STATUS_PREFIXES: tuple[str, ...] = (
    "App session match not found for current targets.",
    "Audio preprocessing active:",
    "Audio preprocessing disabled.",
    "Translation disabled by config.",
    "Stream format changed:",
    "[capture-status] ",
    "[gpu-telemetry] ",
    "[gpu-telemetry-summary] ",
    "[timing-summary] ",
    "[timing-stages] ",
    "WhisperX alignment model loading:",
    "WhisperX alignment model ready:",
    "WhisperX language routing:",
    "WhisperX produced empty text after alignment/postprocess.",
    "WhisperX initialized:",
    "FFmpeg DLL directory registered:",
    "Loaded persisted settings:",
    "Persisted settings saved:",
    "Settings updated:",
    "C++ capture bridge started:",
)

_IMPORTANT_OVERLAY_STATUS_PREFIXES: tuple[str, ...] = (
    "Initializing STT backend...",
    "STT provider active:",
    "Capture started @",
    "Capture recovered @",
    "No audio chunks for 8s. Restarting capture backend...",
    "STT bootstrap failed:",
    "Audio capture init failed:",
    "Capture recovery failed:",
    "Run loop crashed:",
    "Transcription failed:",
    "WhisperX fallback active:",
    "WhisperX warmup started",
    "WhisperX warmup completed.",
    "WhisperX warmup failed:",
    "WhisperX CUDA runtime is unavailable on this environment.",
    "CUDA compatibility alias active.",
    "whisper model download started:",
    "whisper model download completed:",
    "whisper model re-download completed:",
    "WhisperX STT model download started:",
    "WhisperX STT model download completed:",
)


def should_surface_overlay_status(message: str) -> bool:
    text = (message or '').strip()
    if not text:
        return False
    if text.startswith("[download] "):
        return True
    for prefix in _SUPPRESSED_OVERLAY_STATUS_PREFIXES:
        if text.startswith(prefix):
            return False
    for prefix in _IMPORTANT_OVERLAY_STATUS_PREFIXES:
        if text.startswith(prefix):
            return True
    return False
