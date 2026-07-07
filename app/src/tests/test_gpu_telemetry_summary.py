"""Unit tests for GpuTelemetryReporter session-end aggregation (round 0029)."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice2text.pipeline.gpu_telemetry import GpuTelemetryReporter


def _torch(alloc: float, max_alloc: float, total: float = 8192.0) -> dict[str, float]:
    return {"device_index": 0.0, "alloc_mb": alloc, "reserved_mb": alloc, "max_alloc_mb": max_alloc, "total_mb": total}


def _smi(used: float, gpu_util: float, mem_util: float, total: float = 8192.0) -> dict[str, float]:
    return {"index": 0.0, "gpu_util_pct": gpu_util, "mem_util_pct": mem_util, "mem_used_mb": used, "mem_total_mb": total}


class GpuTelemetrySummaryTests(unittest.TestCase):
    def test_empty_summary_has_no_samples(self) -> None:
        summary = GpuTelemetryReporter().summary()
        self.assertEqual(summary["samples"], 0)
        self.assertEqual(summary["vram_used_mb"]["p50"], 0.0)
        self.assertEqual(summary["torch_max_alloc_mb"], 0.0)

    def test_emit_summary_noop_without_samples(self) -> None:
        messages: list[str] = []
        GpuTelemetryReporter().emit_summary(messages.append)
        self.assertEqual(messages, [])

    def test_accumulates_percentiles_and_peak(self) -> None:
        rep = GpuTelemetryReporter()
        rep._accumulate(_torch(1000.0, 1200.0), _smi(2000.0, 30.0, 40.0))
        rep._accumulate(_torch(1500.0, 1800.0), _smi(3000.0, 50.0, 60.0))
        rep._accumulate(_torch(1100.0, 1400.0), _smi(4000.0, 70.0, 80.0))
        summary = rep.summary()
        self.assertEqual(summary["samples"], 3)
        self.assertEqual(summary["vram_used_mb"]["p50"], 3000.0)
        self.assertEqual(summary["vram_used_mb"]["max"], 4000.0)
        self.assertEqual(summary["gpu_util_pct"]["p50"], 50.0)
        # peak is the running max of torch max_alloc across ticks
        self.assertEqual(summary["torch_max_alloc_mb"], 1800.0)
        self.assertEqual(summary["mem_total_mb"], 8192.0)

    def test_emit_summary_line_shape(self) -> None:
        rep = GpuTelemetryReporter()
        rep._accumulate(_torch(1000.0, 1200.0), _smi(2000.0, 30.0, 40.0))
        messages: list[str] = []
        rep.emit_summary(messages.append)
        self.assertEqual(len(messages), 1)
        line = messages[0]
        self.assertTrue(line.startswith("[gpu-telemetry-summary] "))
        self.assertIn("samples=1", line)
        self.assertIn("vram_used_p50/p95/max=", line)
        self.assertIn("torch_max_alloc=1200MB", line)

    def test_torch_only_and_smi_only_tolerated(self) -> None:
        torch_only = GpuTelemetryReporter()
        torch_only._accumulate(_torch(1000.0, 1200.0), None)
        s1 = torch_only.summary()
        self.assertEqual(s1["samples"], 1)
        self.assertEqual(s1["vram_used_mb"]["n"], 0)
        self.assertEqual(s1["torch_alloc_mb"]["p50"], 1000.0)

        smi_only = GpuTelemetryReporter()
        smi_only._accumulate(None, _smi(2000.0, 30.0, 40.0))
        s2 = smi_only.summary()
        self.assertEqual(s2["samples"], 1)
        self.assertEqual(s2["vram_used_mb"]["p50"], 2000.0)
        self.assertEqual(s2["torch_alloc_mb"]["n"], 0)


if __name__ == "__main__":
    unittest.main()
