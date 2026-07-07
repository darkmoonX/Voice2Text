"""Regression/stress checks for segment-arrival status routing behavior.

Validates two guardrails:
1) Overlay status curation stays selective under high-frequency segment events.
2) Debug/full-log stream keeps every status line for diagnostics.
"""
from __future__ import annotations

import logging
import sys
import unittest
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = APP_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice2text.debug_window import DebugWindowLogHandler
from voice2text.status_routing import should_surface_overlay_status


def _build_segment_arrival_stream(*, windows: int = 1200) -> list[str]:
    """Create a deterministic high-volume stream that mimics run-loop activity."""
    events: list[str] = []
    events.append("Initializing STT backend...")
    events.append("STT provider active: whisperx | model=large-v2")
    events.append("Capture started @ 48000 Hz, 2 ch")
    for idx in range(windows):
        events.append(
            f"[capture-status] segment-arrival idx={idx} bytes=192000 lag_ms={idx % 17}"
        )
        events.append(
            f"Audio preprocessing active: configured=spectral-gate,adaptive-gain; active=spectral-gate"
        )
        events.append(
            f"[gpu-telemetry] util={(idx * 7) % 100}% mem_util={(idx * 13) % 100}%"
        )
        if idx % 75 == 0:
            events.append(
                f"[download] whisper/model.bin {idx + 1}/2048 MB ({((idx + 1) / 2048.0) * 100:.1f}%)"
            )
        if idx % 200 == 0:
            events.append("No audio chunks for 8s. Restarting capture backend...")
            events.append("Capture recovered @ 48000 Hz, 2 ch")
    events.append("WhisperX warmup completed.")
    return events


class SegmentArrivalRegressionTests(unittest.TestCase):
    def test_overlay_curation_under_segment_arrival_stress(self) -> None:
        events = _build_segment_arrival_stream()
        overlay_events = [line for line in events if should_surface_overlay_status(line)]

        self.assertGreater(len(events), 3000)
        self.assertGreater(len(overlay_events), 10)

        # Overlay should not be flooded by per-segment/status-noise lines.
        self.assertLess(len(overlay_events) / float(len(events)), 0.2)

        # Important status lines should remain visible.
        self.assertIn("Initializing STT backend...", overlay_events)
        self.assertIn("STT provider active: whisperx | model=large-v2", overlay_events)
        self.assertIn("Capture started @ 48000 Hz, 2 ch", overlay_events)
        self.assertIn("No audio chunks for 8s. Restarting capture backend...", overlay_events)
        self.assertIn("Capture recovered @ 48000 Hz, 2 ch", overlay_events)
        self.assertIn("WhisperX warmup completed.", overlay_events)

        # Suppressed noise should not leak into overlay.
        leaked = [
            line
            for line in overlay_events
            if line.startswith("[capture-status] ")
            or line.startswith("Audio preprocessing active:")
            or line.startswith("[gpu-telemetry] ")
        ]
        self.assertEqual(leaked, [])

        # Download progress is intentionally surfaced.
        download_lines = [line for line in events if line.startswith("[download] ")]
        for line in download_lines:
            self.assertIn(line, overlay_events)

    def test_debug_full_log_stream_keeps_all_lines(self) -> None:
        events = _build_segment_arrival_stream(windows=600)
        captured: list[str] = []

        handler = DebugWindowLogHandler(captured.append)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger = logging.getLogger("voice2text.segment_arrival_regression")
        logger.handlers = []
        logger.propagate = False
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        try:
            for line in events:
                logger.info(line)
        finally:
            logger.removeHandler(handler)
            logger.handlers = []

        self.assertEqual(captured, events)


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SegmentArrivalRegressionTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())

