"""Per-session timing aggregation for the transcription loop.

Accumulates the per-window stage timings the loop already builds (the
`[window-timing]` dict plus the whisperx `provider_timing` sub-stages) into
end-of-session percentile summaries and a steady-state realtime factor, so
latency is observable without re-instrumenting anything.

The realtime factor is `sum(window_total) / (window_count * hop)` =
`mean(window_total) / hop`: in steady state a new window of audio arrives every
`hop` seconds and must be processed within `hop` to keep up, so a value > 1.0
means the pipeline falls behind at that operating point. It is derived from the
stage timings, NOT any wall-clock replay duration (the file-replay harness is not
realtime-paced).
"""
from __future__ import annotations

import math

# Stages from the per-window `timing` dict (capture -> render).
_WINDOW_STAGES = (
    "raw_artifact_seconds",
    "preprocess_seconds",
    "stt_artifact_seconds",
    "transcribe_seconds",
    "language_route_seconds",
    "timestamp_enrich_seconds",
    "merge_seconds",
    "subtitle_payload_seconds",
)
# whisperx sub-stages from `transcription_meta["provider_timing"]` (when present).
_PROVIDER_STAGES = (
    "asr_seconds",
    "align_seconds",
    "diarization_seconds",
    "speaker_profile_seconds",
)


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile (q in [0, 100]); 0.0 for empty input."""
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    n = len(ordered)
    if n == 1:
        return ordered[0]
    rank = (max(0.0, min(100.0, float(q))) / 100.0) * (n - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[int(lo)]
    return ordered[int(lo)] * (hi - rank) + ordered[int(hi)] * (rank - lo)


def format_stage_breakdown(stages: dict) -> str:
    """Render a `summary()["stages"]` dict as `name=p50/p95/max s (n=N)` fields, sorted by p50 desc.

    Returns an empty string when there are no stages, so callers can skip the emit.
    """
    if not isinstance(stages, dict) or not stages:
        return ""
    ordered = sorted(
        stages.items(),
        key=lambda kv: float((kv[1] or {}).get("p50", 0.0) or 0.0),
        reverse=True,
    )
    fields: list[str] = []
    for name, stat in ordered:
        stat = stat or {}
        fields.append(
            f"{name}="
            f"{float(stat.get('p50', 0.0) or 0.0):.4f}/"
            f"{float(stat.get('p95', 0.0) or 0.0):.4f}/"
            f"{float(stat.get('max', 0.0) or 0.0):.4f}s "
            f"(n={int(stat.get('n', 0) or 0)})"
        )
    return "; ".join(fields)


class TimingAggregator:
    def __init__(self) -> None:
        self._stages: dict[str, list[float]] = {}
        self._window_total: list[float] = []
        self._audio_seconds = 0.0
        self._window_count = 0

    @property
    def window_count(self) -> int:
        return self._window_count

    def add_window(
        self,
        *,
        timing: dict | None,
        hop_seconds: float,
        transcription_meta: dict | None = None,
    ) -> None:
        """Record one window's stage timings. Called once per processed window."""
        self._window_count += 1
        self._audio_seconds += float(max(0.0, hop_seconds))
        timing = timing or {}
        self._window_total.append(float(timing.get("window_total_seconds", 0.0) or 0.0))
        for key in _WINDOW_STAGES:
            value = timing.get(key)
            if value is not None:
                self._stages.setdefault(key, []).append(float(value))
        provider = (transcription_meta or {}).get("provider_timing")
        if isinstance(provider, dict):
            for key in _PROVIDER_STAGES:
                value = provider.get(key)
                if value is not None:
                    self._stages.setdefault("wx_" + key, []).append(float(value))

    def summary(self) -> dict:
        processing_seconds = float(sum(self._window_total))
        realtime_factor = (
            processing_seconds / self._audio_seconds if self._audio_seconds > 0 else 0.0
        )
        stages: dict[str, dict] = {}
        for name, values in self._stages.items():
            stages[name] = {
                "p50": round(percentile(values, 50), 4),
                "p95": round(percentile(values, 95), 4),
                "max": round(max(values), 4) if values else 0.0,
                "mean": round(sum(values) / len(values), 4) if values else 0.0,
                "n": len(values),
            }
        dominant_stage = ""
        if stages:
            dominant_stage = max(stages.items(), key=lambda kv: kv[1]["p50"])[0]
        return {
            "window_count": self._window_count,
            "audio_seconds": round(self._audio_seconds, 3),
            "processing_seconds": round(processing_seconds, 3),
            "realtime_factor": round(realtime_factor, 4),
            "dominant_stage": dominant_stage,
            "window_total": {
                "p50": round(percentile(self._window_total, 50), 4),
                "p95": round(percentile(self._window_total, 95), 4),
                "max": round(max(self._window_total), 4) if self._window_total else 0.0,
            },
            "stages": stages,
        }
