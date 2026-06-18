"""Translation backend registry (round 0026).

Maps a backend name to a `TranslationBackend`. Argos is the only real backend today; `llm` / `cloud`
are reserved slots that resolve to a disabled stub with a clear "not yet implemented" status (no
network code ships this round). An unknown name degrades to Argos with a warning so a corrupted config
can never disable translation outright.
"""
from __future__ import annotations

from typing import Callable, Optional

from .base import TranslationBackend, TranslationState
from .argos_backend import ArgosTranslator


# Backend names recognized by the registry. Only "argos" is implemented; the others are placeholders.
KNOWN_BACKENDS = ("argos", "llm", "cloud")
_NOT_IMPLEMENTED = ("llm", "cloud")


class UnavailableBackend:
    """A disabled backend for reserved-but-unimplemented names (llm/cloud).

    `translate` never raises (raising would stall the loop); it returns `None` and reports an inactive
    state with a clear message so the UI shows *why* translation is off.
    """

    def __init__(self, name: str, message: str) -> None:
        self._name = name
        self._state = TranslationState(False, message)

    @property
    def name(self) -> str:
        return self._name

    @property
    def enabled(self) -> bool:
        return False

    @property
    def state(self) -> TranslationState:
        return self._state

    def translate(self, text: str, source_code: str | None = None) -> Optional[str]:
        return None


def build_backend(
    name: str | None,
    config: object,
    *,
    on_status: Optional[Callable[[str], None]] = None,
) -> TranslationBackend:
    """Construct the named backend from a `RuntimeConfig`-like object.

    Reads `translation_enabled` / `translation_from` / `translation_to` off `config` for Argos.
    """
    token = (name or "argos").strip().lower()
    enabled = bool(getattr(config, "translation_enabled", False))
    source = getattr(config, "translation_from", "auto")
    target = getattr(config, "translation_to", "zh")

    if token in ("", "argos"):
        return ArgosTranslator(enabled=enabled, source_code=source, target_code=target)

    if token in _NOT_IMPLEMENTED:
        return UnavailableBackend(
            token,
            f"Translation backend '{token}' is reserved but not yet implemented; translation is off.",
        )

    if on_status is not None:
        on_status(f"Unknown translation backend '{token}', falling back to argos.")
    return ArgosTranslator(enabled=enabled, source_code=source, target_code=target)
