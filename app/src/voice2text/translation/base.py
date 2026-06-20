"""Translation backend protocol + shared state type (round 0026).

A `TranslationBackend` is any object that can turn a source string into a target string (or `None`
when it can't / is disabled). The concrete Argos adapter lives in `argos_backend.py`; future LLM/cloud
backends slot in via `registry.py`. The runtime never talks to a backend directly — it goes through
`TranslationEngine` (`engine.py`), which adds the off-thread queue/timeout/retry policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable


@dataclass
class TranslationState:
    """Whether a backend is actively translating, plus a human-readable status line."""
    active: bool
    message: str


@runtime_checkable
class TranslationBackend(Protocol):
    """Structural interface the engine/registry depend on.

    `enabled` is the back-compat "this backend will attempt translation" flag the loop already checks;
    `state` carries the surfaced status message (with any credentials redacted by the backend itself).
    """

    @property
    def name(self) -> str: ...

    @property
    def enabled(self) -> bool: ...

    @property
    def state(self) -> TranslationState: ...

    def translate(self, text: str, source_code: str | None = None) -> Optional[str]: ...
