"""Back-compat re-export shim.

The Argos adapter and translation-state type moved to the `voice2text.translation` package in round 0026
(pluggable backends + off-thread engine). Existing imports `from .translator import ArgosTranslator`
keep working through these re-exports — no caller churn.
"""
from __future__ import annotations

from .translation.argos_backend import ArgosTranslator
from .translation.base import TranslationState

__all__ = ["ArgosTranslator", "TranslationState"]
