"""Rules that decide which status lines should be surfaced in the overlay."""
from __future__ import annotations
_SUPPRESSED_OVERLAY_STATUS_PREFIXES: tuple[str, ...] = ('App session match not found for current targets.',)

def should_surface_overlay_status(message: str) -> bool:
    text = (message or '').strip()
    if not text:
        return False
    for prefix in _SUPPRESSED_OVERLAY_STATUS_PREFIXES:
        if text.startswith(prefix):
            return False
    return True
