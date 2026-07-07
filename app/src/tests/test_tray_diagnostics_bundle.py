"""Round 0069: tray-triggered diagnostics bundle action (manual trigger for round 0025 Phase B)."""
from __future__ import annotations

import os
import time
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication, QWidget

    _APP = QApplication.instance() or QApplication([])
    _QT_OK = True
except Exception:  # pragma: no cover
    _QT_OK = False

from voice2text.config import RuntimeConfig


def _pump_until(predicate, *, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        _APP.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


@unittest.skipUnless(_QT_OK, "PySide6 / offscreen Qt unavailable")
class TrayDiagnosticsBundleTests(unittest.TestCase):
    def _controller(self):
        from voice2text.tray_controller import Voice2TextTrayController

        overlay = QWidget()
        return Voice2TextTrayController(
            app=_APP,
            overlay=overlay,
            config=RuntimeConfig(),
            on_settings_applied=lambda updates: None,
        )

    def test_create_diagnostics_bundle_emits_created_on_success(self):
        ctl = self._controller()
        results: list[str] = []
        ctl._bundle_created.connect(lambda path: results.append(path))

        with patch("voice2text.crash_bundle.create_crash_bundle", return_value=Path("C:/fake/crash_1.zip")):
            ctl.create_diagnostics_bundle()
            self.assertTrue(_pump_until(lambda: bool(results)))

        self.assertEqual(results[0], "C:\\fake\\crash_1.zip" if os.name == "nt" else "/fake/crash_1.zip")

    def test_create_diagnostics_bundle_emits_failed_on_error(self):
        ctl = self._controller()
        errors: list[str] = []
        ctl._bundle_failed.connect(lambda message: errors.append(message))

        with patch("voice2text.crash_bundle.create_crash_bundle", side_effect=RuntimeError("boom")):
            ctl.create_diagnostics_bundle()
            self.assertTrue(_pump_until(lambda: bool(errors)))

        self.assertIn("boom", errors[0])


if __name__ == "__main__":
    unittest.main()
