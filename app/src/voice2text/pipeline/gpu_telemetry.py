"""Periodic GPU telemetry reporter for runtime diagnostics."""
from __future__ import annotations

import subprocess

from .timing_stats import percentile


class GpuTelemetryReporter:
    def __init__(self, *, interval_seconds: float = 10.0) -> None:
        self._last_at = 0.0
        self._interval_seconds = max(1.0, float(interval_seconds))
        self._gpu_index: int | None = None
        self._unavailable_logged = False
        # Session-end aggregation: samples collected on each successful periodic tick.
        self._vram_used_mb: list[float] = []
        self._gpu_util_pct: list[float] = []
        self._mem_util_pct: list[float] = []
        self._torch_alloc_mb: list[float] = []
        self._torch_max_alloc_peak_mb = 0.0
        self._mem_total_mb = 0.0

    def maybe_emit(
        self,
        *,
        now_monotonic: float,
        debug_mode: bool,
        model_device: str,
        emit_status,
    ) -> None:
        if not bool(debug_mode):
            return
        if "cuda" not in str(model_device or "").strip().lower():
            return
        if now_monotonic - self._last_at < self._interval_seconds:
            return
        self._last_at = now_monotonic
        message = self._build_message()
        if not message:
            if not self._unavailable_logged:
                self._unavailable_logged = True
                emit_status("GPU telemetry unavailable (no CUDA/nvidia-smi stats).")
            return
        self._unavailable_logged = False
        emit_status(message)

    def _build_message(self) -> str:
        torch_stats = self._collect_torch_cuda_stats()
        smi_stats = self._collect_nvidia_smi_stats()
        self._accumulate(torch_stats, smi_stats)
        if not torch_stats and not smi_stats:
            return ""
        parts: list[str] = ["[gpu-telemetry]"]
        if torch_stats:
            parts.append(
                "torch"
                f"(dev={int(torch_stats['device_index'])},"
                f"alloc={torch_stats['alloc_mb']:.0f}MB,"
                f"reserved={torch_stats['reserved_mb']:.0f}MB,"
                f"max_alloc={torch_stats['max_alloc_mb']:.0f}MB,"
                f"total={torch_stats['total_mb']:.0f}MB)"
            )
        if smi_stats:
            parts.append(
                "nvidia-smi"
                f"(gpu={int(smi_stats['index'])},"
                f"util={smi_stats['gpu_util_pct']:.0f}%,"
                f"mem_util={smi_stats['mem_util_pct']:.0f}%,"
                f"vram={smi_stats['mem_used_mb']:.0f}/{smi_stats['mem_total_mb']:.0f}MB)"
            )
        return " ".join(parts)

    def _accumulate(self, torch_stats: dict | None, smi_stats: dict | None) -> None:
        """Record one tick's stats for the session-end percentile summary."""
        if torch_stats:
            self._torch_alloc_mb.append(float(torch_stats["alloc_mb"]))
            self._torch_max_alloc_peak_mb = max(
                self._torch_max_alloc_peak_mb, float(torch_stats["max_alloc_mb"])
            )
            if not self._mem_total_mb:
                self._mem_total_mb = float(torch_stats["total_mb"])
        if smi_stats:
            self._vram_used_mb.append(float(smi_stats["mem_used_mb"]))
            self._gpu_util_pct.append(float(smi_stats["gpu_util_pct"]))
            self._mem_util_pct.append(float(smi_stats["mem_util_pct"]))
            self._mem_total_mb = float(smi_stats["mem_total_mb"])

    @staticmethod
    def _percentiles(values: list[float]) -> dict[str, float]:
        return {
            "p50": round(percentile(values, 50), 1),
            "p95": round(percentile(values, 95), 1),
            "max": round(max(values), 1) if values else 0.0,
            "n": len(values),
        }

    def summary(self) -> dict:
        """Session-end percentile aggregation of the collected telemetry ticks."""
        samples = max(len(self._vram_used_mb), len(self._torch_alloc_mb))
        return {
            "samples": samples,
            "mem_total_mb": round(self._mem_total_mb, 1),
            "torch_max_alloc_mb": round(self._torch_max_alloc_peak_mb, 1),
            "vram_used_mb": self._percentiles(self._vram_used_mb),
            "gpu_util_pct": self._percentiles(self._gpu_util_pct),
            "mem_util_pct": self._percentiles(self._mem_util_pct),
            "torch_alloc_mb": self._percentiles(self._torch_alloc_mb),
        }

    def emit_summary(self, emit_status) -> None:
        """Emit a `[gpu-telemetry-summary]` line at session end; no-op when no samples were collected.

        Samples are only accumulated by `maybe_emit` (already gated on debug-mode + cuda device), so a
        non-debug/non-cuda session never has samples and this is a silent no-op without re-checking gating.
        """
        summary = self.summary()
        if int(summary.get("samples", 0) or 0) <= 0:
            return
        vram = summary["vram_used_mb"]
        util = summary["gpu_util_pct"]
        emit_status(
            "[gpu-telemetry-summary] "
            f"samples={int(summary['samples'])}; "
            f"vram_used_p50/p95/max={vram['p50']:.0f}/{vram['p95']:.0f}/{vram['max']:.0f}MB "
            f"of {summary['mem_total_mb']:.0f}MB; "
            f"gpu_util_p50/p95/max={util['p50']:.0f}/{util['p95']:.0f}/{util['max']:.0f}%; "
            f"torch_max_alloc={summary['torch_max_alloc_mb']:.0f}MB"
        )

    def _collect_torch_cuda_stats(self) -> dict[str, float] | None:
        try:
            import torch  # type: ignore
        except Exception:
            return None
        try:
            if not torch.cuda.is_available():
                return None
            device_index = int(torch.cuda.current_device())
            if self._gpu_index is None:
                self._gpu_index = device_index
            alloc_mb = float(torch.cuda.memory_allocated(device_index)) / (1024.0 * 1024.0)
            reserved_mb = float(torch.cuda.memory_reserved(device_index)) / (1024.0 * 1024.0)
            max_alloc_mb = float(torch.cuda.max_memory_allocated(device_index)) / (1024.0 * 1024.0)
            total_mb = float(torch.cuda.get_device_properties(device_index).total_memory) / (1024.0 * 1024.0)
            return {
                "device_index": float(device_index),
                "alloc_mb": alloc_mb,
                "reserved_mb": reserved_mb,
                "max_alloc_mb": max_alloc_mb,
                "total_mb": total_mb,
            }
        except Exception:
            return None

    def _collect_nvidia_smi_stats(self) -> dict[str, float] | None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=1.5,
                check=False,
            )
        except Exception:
            return None
        if result.returncode != 0:
            return None
        rows: list[dict[str, float]] = []
        for raw in (result.stdout or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            cols = [c.strip() for c in line.split(",")]
            if len(cols) < 5:
                continue
            try:
                rows.append(
                    {
                        "index": float(cols[0]),
                        "gpu_util_pct": float(cols[1]),
                        "mem_util_pct": float(cols[2]),
                        "mem_used_mb": float(cols[3]),
                        "mem_total_mb": float(cols[4]),
                    }
                )
            except Exception:
                continue
        if not rows:
            return None
        target_index = self._gpu_index if self._gpu_index is not None else 0
        for row in rows:
            if int(row.get("index", -1)) == int(target_index):
                return row
        return rows[0]
