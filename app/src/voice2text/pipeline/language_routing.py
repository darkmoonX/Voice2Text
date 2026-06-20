"""Per-window translation source-language routing.

The session language lock remains the stable display hint. This module only
decides which source code to request from the translation backend for one
window, allowing confident code-switch windows to route immediately.
"""
from __future__ import annotations

from collections.abc import Iterable


ROUTE_MIN_TOKEN_COUNT = 8
ROUTE_MIN_STABILITY_RATIO = 0.70
SCRIPT_MAJORITY_RATIO = 0.45


def route_source_language(
    *,
    explicit_source_language: object,
    locked_source_language: object,
    detected_language: object,
    stability_ratio: object,
    token_count: object,
    text: str,
    allowed_languages: Iterable[str],
) -> str | None:
    """Return the app-level source code to pass into translation.

    Explicit user source language wins. Auto mode uses the detected language
    only when it is allowed, corroborated by the text script, and the window is
    not too short/noisy; otherwise it falls back to the session lock.
    """
    explicit = _normalize_language_token(explicit_source_language)
    if explicit:
        return explicit

    locked = _normalize_language_token(locked_source_language)
    detected = _normalize_language_token(detected_language)
    allowed = {_normalize_language_token(item) for item in allowed_languages}
    allowed.discard("")

    if not detected or detected not in allowed:
        return locked or None
    if not _is_script_corroborated(detected, text):
        return locked or None

    tokens = _to_int(token_count, 0)
    stability = _to_float(stability_ratio, 0.0)
    if tokens >= ROUTE_MIN_TOKEN_COUNT or stability >= ROUTE_MIN_STABILITY_RATIO:
        return detected
    return locked or None


def _normalize_language_token(value: object) -> str:
    token = str(value or "").strip().lower()
    if not token or token == "auto":
        return ""
    if token in {"zh-hant", "zh-hans", "zh-tw", "zh-cn", "zh-hk", "zh-sg"}:
        return "zh"
    return token


def _is_script_corroborated(language: str, text: str) -> bool:
    signal = _script_signal(text)
    total = signal["total"]
    if total <= 0:
        return False
    han_ratio = signal["han"] / total
    kana_ratio = signal["kana"] / total
    hangul_ratio = signal["hangul"] / total
    latin_ratio = signal["latin"] / total

    if language == "zh":
        return han_ratio >= SCRIPT_MAJORITY_RATIO and kana_ratio < 0.20 and hangul_ratio < 0.20
    if language == "ja":
        return (kana_ratio + han_ratio) >= SCRIPT_MAJORITY_RATIO and kana_ratio > 0.0
    if language == "ko":
        return hangul_ratio >= SCRIPT_MAJORITY_RATIO
    if language in {"en", "de", "fr", "es", "it", "pt"}:
        return latin_ratio >= SCRIPT_MAJORITY_RATIO and (han_ratio + kana_ratio + hangul_ratio) < SCRIPT_MAJORITY_RATIO
    if language == "ru":
        return signal["cyrillic"] / total >= SCRIPT_MAJORITY_RATIO
    # Unknown-but-allowed languages are not script-validated by this helper.
    return True


def _script_signal(text: str) -> dict[str, int]:
    counts = {"total": 0, "han": 0, "kana": 0, "hangul": 0, "latin": 0, "cyrillic": 0}
    for ch in text or "":
        code = ord(ch)
        if ch.isdigit() or ch.isspace() or _is_punctuation(code):
            continue
        counts["total"] += 1
        if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF:
            counts["han"] += 1
        elif 0x3040 <= code <= 0x30FF:
            counts["kana"] += 1
        elif 0xAC00 <= code <= 0xD7AF or 0x1100 <= code <= 0x11FF:
            counts["hangul"] += 1
        elif ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
            counts["latin"] += 1
        elif 0x0400 <= code <= 0x04FF:
            counts["cyrillic"] += 1
    return counts


def _is_punctuation(code: int) -> bool:
    return (
        0x2000 <= code <= 0x206F
        or 0x3000 <= code <= 0x303F
        or 0xFF00 <= code <= 0xFFEF
        or code in {0x0021, 0x0022, 0x0027, 0x0028, 0x0029, 0x002C, 0x002D, 0x002E, 0x003A, 0x003B, 0x003F}
    )


def _to_int(value: object, fallback: int) -> int:
    try:
        return int(value)
    except Exception:
        return fallback


def _to_float(value: object, fallback: float) -> float:
    try:
        return float(value)
    except Exception:
        return fallback
