"""Device-aware resolution of the 'auto' WhisperX model default (round 0072).

`model_size='auto'` picks the best default for the *effective* ASR device, decided
after all CUDA-availability fallbacks:

- CUDA  -> large-v3. Round 0072 quality baselines: zh truth CER 19.1% -> 15.0%,
  en 15.5% -> 13.7% vs the old `small` default, completeness up on every case, and
  paced realtime keeps up on an RTX 3060-class GPU with diarization enabled.
- CPU   -> small. Large models do not sustain CPU realtime (round 0024/0015); the
  `cpu` preset pins `small` explicitly for the same reason.

Explicit model names are always honored unchanged. This module must stay free of
heavy imports (torch/ctranslate2) — it is consulted on the bootstrap path.
"""
from __future__ import annotations

AUTO_MODEL = "auto"
DEFAULT_CUDA_MODEL = "large-v3"
DEFAULT_CPU_MODEL = "small"


def is_auto_model(model_size: str | None) -> bool:
    return str(model_size or "").strip().lower() in ("", AUTO_MODEL)


def resolve_model_size(model_size: str | None, device: str | None) -> str:
    """Return the effective model name for the given (possibly 'auto') size + device."""
    if not is_auto_model(model_size):
        return str(model_size).strip()
    if str(device or "").strip().lower().startswith("cuda"):
        return DEFAULT_CUDA_MODEL
    return DEFAULT_CPU_MODEL
