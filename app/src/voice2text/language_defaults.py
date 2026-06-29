"""Per-language default parameter bundles.

Some accuracy/speaker knobs are genuinely language-dependent because the underlying models
differ per language (alignment wav2vec2; the embedding similarity distribution the speaker
profile layer sees). This module seeds those knobs from the resolved `source_language` BEFORE
the transcriber is built, filling ONLY fields the user has left at their global default — any
explicit CLI/persisted value still wins.

First landed case (round 0045, 2026-06-30): `whisperx_speaker_profile_reconcile_threshold`.
The per-window auto-reconcile merges profiles with centroid cosine >= this value. zh speakers'
short-window embeddings sit at cross-speaker cosine ~0.78 (same-speaker ~0.95), so the global
0.52 default merges distinct zh speakers back into one every window (realtime collapse). 0.88
sits in the same/cross gap: Bn realtime speaker-attribution 55% -> 90%. English same-speaker
embeddings are less self-consistent (A-Vskw fragments into 7 at 0.88), so English keeps ~0.52 —
hence this is per-language, not a global change. See docs/reference/language-parameters.md.
"""
from __future__ import annotations

from .config import RuntimeConfig

# Resolved-language-key -> {RuntimeConfig field: language-appropriate default}.
# Only list fields that genuinely diverge from the global default for that language.
LANGUAGE_DEFAULTS: dict[str, dict[str, object]] = {
    "zh": {
        "whisperx_speaker_profile_reconcile_threshold": 0.88,
    },
}

# Language codes folded onto a single key (script/region suffixes stripped first).
_LANGUAGE_ALIASES: dict[str, str] = {
    "zh": "zh", "cmn": "zh", "yue": "zh", "wuu": "zh", "hak": "zh", "nan": "zh", "zho": "zh",
    "en": "en", "eng": "en",
}


def normalize_language_key(language: object) -> str | None:
    """Fold a language code (e.g. 'zh-Hant', 'cmn', 'en-US') to a defaults key, or None."""
    token = str(language or "").strip().lower()
    if not token or token == "auto":
        return None
    primary = token.replace("_", "-").split("-", 1)[0]
    return _LANGUAGE_ALIASES.get(primary, primary or None)


def language_default_overrides(language: object) -> dict[str, object]:
    """Return the per-language default bundle for a language code (empty if none)."""
    key = normalize_language_key(language)
    return dict(LANGUAGE_DEFAULTS.get(key, {})) if key else {}


def apply_language_defaults(config: object, *, base: RuntimeConfig | None = None) -> list[str]:
    """Seed per-language defaults onto a RuntimeConfig for fields still at the global default.

    Reads ``config.source_language``. A field is only overwritten when its current value equals
    the global default (i.e. the user has not explicitly chosen it) so explicit CLI/persisted
    settings always win. Returns the list of fields actually changed.
    """
    overrides = language_default_overrides(getattr(config, "source_language", None))
    if not overrides:
        return []
    reference = base if base is not None else RuntimeConfig()
    applied: list[str] = []
    for field, value in overrides.items():
        if not hasattr(config, field):
            continue
        if getattr(config, field) == getattr(reference, field, object()) and getattr(config, field) != value:
            setattr(config, field, value)
            applied.append(field)
    return applied
