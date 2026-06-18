"""Runtime presets: named bundles of latency/accuracy knobs.

A preset is a *base* the user can pick instead of hand-tuning model / compute /
beam / segment-hop / alignment / diarization / speaker-profile. Any explicit
per-knob flag (CLI) or persisted setting still wins over the preset.

`balanced` reproduces the shipped default (round 0014 measured operating point);
`high-accuracy` is the quality/offline point (round 0015 Phase B); `cpu` is the
non-CUDA realtime point (round 0024 — alignment off + synthesized word timestamps).
"""
from __future__ import annotations

import argparse

# round-0015 Phase B dropped the `low-latency` preset: on hard CJK every "faster"
# lever backfires through whisper's temperature-fallback retries -- a smaller model,
# int8_float16, and beam 1 all measured SLOWER (more retries), and shrinking seg/hop
# to the overlap-3 floor drops ~20% of words. The only real headroom lever is a
# larger hop, which trades completeness, so there is no faster-AND-acceptable point
# to ship. `balanced` is already near-optimal for live; `high-accuracy` is a
# quality/offline mode (rtf ~2.1x, does NOT sustain live realtime).
PRESET_NAMES: tuple[str, ...] = ("balanced", "high-accuracy", "cpu")

# Canonical bundles in RuntimeConfig field-name space.
PRESETS: dict[str, dict[str, object]] = {
    "cpu": {
        # Round 0024: realtime subtitles on a non-CUDA / CPU-only machine. Forced alignment is the CPU
        # cost killer (~8x), so it is OFF; the provider synthesizes word timestamps from segment spans so
        # the rolling merge still de-duplicates overlapping windows. int8 + beam 1 + small keep rtf < 1 on
        # CPU. Trade-off: no word-level timestamps (no precise SRT word timing / speaker word-attribution);
        # best for non-CJK speech. Forces the CPU execution variant so it does not try CUDA.
        "stt_variant": "cpu",
        "model_size": "small",
        "compute_type": "int8",
        "whisper_beam_size": 1,
        "segment_seconds": 10.0,
        "hop_seconds": 2.0,
        "whisperx_enable_forced_alignment": False,
        "whisperx_enable_diarization": False,
        "whisperx_speaker_profile_enabled": False,
    },
    "balanced": {
        "model_size": "medium",
        "compute_type": "float16",
        "whisper_beam_size": 5,
        "segment_seconds": 10.0,
        "hop_seconds": 2.0,  # overlap 5.0 (round-0014 default)
        "whisperx_enable_forced_alignment": True,
        "whisperx_enable_diarization": False,
        "whisperx_speaker_profile_enabled": True,
    },
    "high-accuracy": {
        # Best quality (large-v2, CER 0.136 vs balanced 0.147) but rtf ~2.1x: it
        # does NOT sustain live realtime -- intended for imported-file processing
        # or users who accept lag.
        "model_size": "large-v2",
        "compute_type": "float16",
        "whisper_beam_size": 5,
        "segment_seconds": 10.0,
        "hop_seconds": 2.0,
        "whisperx_enable_forced_alignment": True,
        "whisperx_enable_diarization": True,
        "whisperx_speaker_profile_enabled": True,
    },
}

# RuntimeConfig field -> argparse dest, so a preset can seed parser defaults
# (which explicit flags then override). Every bundled field is CLI-expressible.
_FIELD_TO_ARG_DEST: dict[str, str] = {
    "stt_variant": "stt_variant",
    "model_size": "model",
    "compute_type": "compute_type",
    "whisper_beam_size": "beam_size",
    "segment_seconds": "segment_seconds",
    "hop_seconds": "hop_seconds",
    "whisperx_enable_forced_alignment": "whisperx_forced_alignment",
    "whisperx_enable_diarization": "whisperx_diarization",
    "whisperx_speaker_profile_enabled": "whisperx_speaker_profile",
}


def normalize_preset(name: str | None) -> str:
    """Return a valid preset name, or '' for none/unknown (accepts a few aliases)."""
    token = str(name or "").strip().lower().replace("_", "-")
    if token in PRESETS:
        return token
    aliases = {"accurate": "high-accuracy", "accuracy": "high-accuracy", "quality": "high-accuracy",
               "default": "balanced", "balance": "balanced"}
    return aliases.get(token, "")


def apply_preset(config: object, name: str | None) -> list[str]:
    """Set a preset's bundled fields on a RuntimeConfig-like object.

    Only the bundled fields are touched; everything else is left as-is. Returns
    the list of fields applied (empty for an unknown/empty preset).
    """
    resolved = normalize_preset(name)
    if not resolved:
        return []
    applied: list[str] = []
    for field, value in PRESETS[resolved].items():
        setattr(config, field, value)
        applied.append(field)
    if hasattr(config, "runtime_preset"):
        config.runtime_preset = resolved
    return applied


def preset_arg_defaults(name: str | None) -> dict[str, object]:
    """Map a preset bundle to argparse dests for `parser.set_defaults(**...)`."""
    resolved = normalize_preset(name)
    if not resolved:
        return {}
    return {
        _FIELD_TO_ARG_DEST[field]: value
        for (field, value) in PRESETS[resolved].items()
        if field in _FIELD_TO_ARG_DEST
    }


def apply_preset_defaults(parser: argparse.ArgumentParser, argv: list[str] | None) -> str:
    """Pre-parse `--preset` from argv and seed the parser's defaults with its bundle.

    Run after the parser is built but before `parse_args`, so explicit per-knob
    flags still override the preset. Returns the resolved preset name ('' if none).
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--preset", default="")
    known, _ = pre.parse_known_args(argv)
    resolved = normalize_preset(known.preset)
    if resolved:
        parser.set_defaults(**preset_arg_defaults(resolved))
    return resolved
