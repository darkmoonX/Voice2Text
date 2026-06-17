"""Quality gate for speaker-profile learning (round 0023).

Pure, dependency-free heuristics that decide whether a speaker-clip is trustworthy enough to
*learn* from (create/update a centroid). It deliberately does NOT decide which display label a
span receives — that stays a merge anchor (see memory `speaker-markers-are-merge-anchors`). A
clip that fails the gate is still allowed to *match* an existing mature profile for display via
the read-only `match_or_create(allow_update=False)` path; it just never writes a centroid.

All thresholds are conservative by default so the gate is CER-neutral; the harness A/B (Phase B)
tunes them. No torch/Qt imports — fully unit-testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Sequence


# Bracketed / parenthesized non-speech tags and musical-note glyphs that WhisperX emits for
# music / applause / sound effects. These are the classic centroid-polluting "music tail" spans.
_NON_SPEECH_TAG = re.compile(
    r"^[\s]*[\[\(（【]?\s*(music|applause|laughter|cheering|noise|sound|silence|背景音乐|音乐|掌声|笑声|♪|♫)"
    r".*?[\]\)）】]?[\s]*$",
    flags=re.IGNORECASE,
)
_MUSIC_GLYPH = re.compile(r"[♪♫♬♩]")
_REAL_CHAR = re.compile(r"[0-9A-Za-z㐀-鿿぀-ヿ가-힣]")
_TOKEN_SPLIT = re.compile(r"\s+")


@dataclass
class ClipQualityConfig:
    """Thresholds for the learn-path quality gate. Conservative defaults => CER-neutral."""

    enabled: bool = False
    min_confidence: float = 0.45          # mean word `score`; below => don't learn
    min_real_chars: int = 1               # minimum substantive (alnum/CJK/kana/hangul) chars
    min_unique_char_ratio: float = 0.22   # below (for non-trivial length) => degenerate/repetitive
    repetition_min_length: int = 6        # only apply the ratio rule at/above this many real chars


@dataclass
class ClipQuality:
    ok: bool
    score: float
    reasons: list[str] = field(default_factory=list)


def _real_chars(text: str) -> str:
    return "".join(_REAL_CHAR.findall(str(text or "")))


def _mean_score(word_scores: Sequence[float] | None) -> float | None:
    if not word_scores:
        return None
    vals = []
    for raw in word_scores:
        try:
            vals.append(float(raw))
        except (TypeError, ValueError):
            continue
    if not vals:
        return None
    return sum(vals) / float(len(vals))


def _is_non_speech(text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return False
    if _MUSIC_GLYPH.search(stripped):
        return True
    return bool(_NON_SPEECH_TAG.match(stripped))


def _is_repetitive(real: str, config: ClipQualityConfig) -> bool:
    if len(real) < int(max(1, config.repetition_min_length)):
        return False
    unique_ratio = len(set(real)) / float(len(real))
    return unique_ratio < float(config.min_unique_char_ratio)


def evaluate_clip_quality(
    *,
    text: str,
    word_scores: Sequence[float] | None,
    duration_seconds: float,
    config: ClipQualityConfig,
) -> ClipQuality:
    """Decide whether a speaker-clip is trustworthy enough to update/create a centroid.

    Returns `ok=False` with `reasons` when the clip looks like gibberish, a music/sound tag, an
    empty/degenerate span, or low-confidence ASR. When `config.enabled` is False this always
    returns `ok=True` (pure pass-through, so the gate is a no-op until explicitly turned on).
    """
    if not config.enabled:
        return ClipQuality(ok=True, score=1.0, reasons=[])

    reasons: list[str] = []
    body = str(text or "").strip()
    real = _real_chars(body)

    if not body:
        reasons.append("empty_text")
    elif _is_non_speech(body):
        reasons.append("non_speech_tag")
    if real and len(real) < int(max(0, config.min_real_chars)):
        reasons.append("too_short_text")
    if real and _is_repetitive(real, config):
        reasons.append("repetitive")

    mean_score = _mean_score(word_scores)
    if mean_score is not None and mean_score < float(config.min_confidence):
        reasons.append("low_confidence")

    # Trust score for telemetry: confidence when known, else 1.0; penalised by any hard reason.
    base = mean_score if mean_score is not None else 1.0
    score = 0.0 if reasons else float(max(0.0, min(1.0, base)))
    return ClipQuality(ok=not reasons, score=score, reasons=reasons)
