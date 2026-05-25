"""Whisper runtime config parsing kept independent from heavy STT imports."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class WhisperRuntimeParams:
    """Decoded optional runtime parameters loaded from whisper_config.json."""

    max_context: Optional[int] = None
    entropy_thold: Optional[float] = None
    logprob_thold: Optional[float] = None
    no_speech_thold: Optional[float] = None
    temperature: Optional[float] = None
    beam_size: Optional[int] = None
    best_of: Optional[int] = None


def _pick_value(raw: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in raw:
            return raw[key]
    return None


def _parse_optional_int(raw: dict[str, Any], keys: list[str]) -> Optional[int]:
    value = _pick_value(raw, keys)
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _parse_optional_float(raw: dict[str, Any], keys: list[str]) -> Optional[float]:
    value = _pick_value(raw, keys)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_whisper_runtime_params(config_path: Path) -> WhisperRuntimeParams:
    """Load optional whisper decode defaults from JSON; return safe defaults on errors."""
    if not config_path.is_file():
        return WhisperRuntimeParams()
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return WhisperRuntimeParams()
    if not isinstance(payload, dict):
        return WhisperRuntimeParams()
    section = payload.get("whisper")
    data = section if isinstance(section, dict) else payload
    return WhisperRuntimeParams(
        max_context=_parse_optional_int(data, ["max-context", "max_context", "mc", "-mc"]),
        entropy_thold=_parse_optional_float(data, ["entropy-thold", "entropy_thold"]),
        logprob_thold=_parse_optional_float(data, ["logprob-thold", "logprob_thold"]),
        no_speech_thold=_parse_optional_float(data, ["no-speech-thold", "no_speech_thold"]),
        temperature=_parse_optional_float(data, ["temperature"]),
        beam_size=_parse_optional_int(data, ["beam-size", "beam_size"]),
        best_of=_parse_optional_int(data, ["best-of", "best_of"]),
    )


__all__ = ["WhisperRuntimeParams", "load_whisper_runtime_params"]
