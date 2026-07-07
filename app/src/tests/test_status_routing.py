"""Unit tests for overlay status filtering rules."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice2text.status_routing import should_surface_overlay_status


class StatusRoutingTests(unittest.TestCase):
    def test_app_session_mismatch_status_is_suppressed_on_overlay(self) -> None:
        message = (
            "App session match not found for current targets. "
            "targets=msedge.exe; active_sessions=LineCall.exe, msedge.exe"
        )
        self.assertFalse(should_surface_overlay_status(message))

    def test_regular_status_is_visible_on_overlay(self) -> None:
        self.assertTrue(should_surface_overlay_status("Capture started @ 48000 Hz, 2 ch"))


if __name__ == "__main__":
    unittest.main()
