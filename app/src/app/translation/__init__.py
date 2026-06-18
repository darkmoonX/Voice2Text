"""Pluggable translation backends + off-thread engine (round 0026).

Public API:
- `TranslationBackend` / `TranslationState` — the backend protocol + status type.
- `ArgosTranslator` — the concrete Argos adapter (also re-exported from `app.translator` for back-compat).
- `build_backend` / `KNOWN_BACKENDS` — name -> backend registry.
- `TranslationEngine` / `build_translation_engine` — backend wrapper with queue/timeout/retry policy.
"""
from __future__ import annotations

from .base import TranslationBackend, TranslationState
from .argos_backend import ArgosTranslator
from .registry import KNOWN_BACKENDS, UnavailableBackend, build_backend
from .engine import TranslationEngine, build_translation_engine

__all__ = [
    "TranslationBackend",
    "TranslationState",
    "ArgosTranslator",
    "KNOWN_BACKENDS",
    "UnavailableBackend",
    "build_backend",
    "TranslationEngine",
    "build_translation_engine",
]
