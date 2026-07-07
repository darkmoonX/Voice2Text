"""Round 0022 Phase B: health-check + model-cache Qt dialogs (rendering, offscreen)."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication

    _APP = QApplication.instance() or QApplication([])
    _QT_OK = True
except Exception:  # pragma: no cover
    _QT_OK = False

from voice2text.config import RuntimeConfig
from voice2text.stt.healthcheck import HealthCheck, ProviderHealthReport
from voice2text.stt.model_cache import ModelCacheEntry, ModelCacheScan


def _pump_until(predicate, *, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        _APP.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


@unittest.skipUnless(_QT_OK, "PySide6 / offscreen Qt unavailable")
class HealthDialogTests(unittest.TestCase):
    def _dialog(self):
        from voice2text.diagnostics_dialogs import HealthCheckDialog

        # Construct without auto-running the background check by stubbing _run.
        HealthCheckDialog._run = lambda self: None  # type: ignore
        return HealthCheckDialog(RuntimeConfig())

    def test_populate_health_rows_and_fix_hidden_when_ok(self):
        dlg = self._dialog()
        report = ProviderHealthReport(provider="whisperx", ok=False)
        report.checks = [
            HealthCheck("cuda", "CUDA", "ok", detail="fine", fix_hint="should-not-show"),
            HealthCheck("ffmpeg", "FFmpeg", "warn", detail="missing", fix_hint="install ffmpeg"),
        ]
        rows = dlg._populate_health([report])
        self.assertEqual(len(rows), 2)
        self.assertEqual(dlg._table.rowCount(), 2)
        # ok row hides the fix; warn row shows it.
        self.assertEqual(dlg._table.item(0, 3).text(), "")
        self.assertEqual(dlg._table.item(1, 3).text(), "install ffmpeg")
        self.assertEqual(dlg._table.item(1, 0).text(), "WARN")


@unittest.skipUnless(_QT_OK, "PySide6 / offscreen Qt unavailable")
class CacheDialogTests(unittest.TestCase):
    def _dialog(self):
        from voice2text.diagnostics_dialogs import ModelCacheDialog

        ModelCacheDialog._run = lambda self: None  # type: ignore
        return ModelCacheDialog(RuntimeConfig())

    def test_populate_cache_table_and_header(self):
        dlg = self._dialog()
        scan = ModelCacheScan(root="r")
        scan.entries = [
            ModelCacheEntry("stt", "medium", "", "/p/medium", 1048576, True),
            ModelCacheEntry("align", "zh-model", "zh", "/p/zh", 2097152, False),
        ]
        entries = dlg._populate_cache(scan)
        self.assertEqual(len(entries), 2)
        self.assertEqual(dlg._table.rowCount(), 2)
        self.assertEqual(dlg._table.item(0, 0).text(), "medium")
        self.assertEqual(dlg._table.item(1, 1).text(), "zh")
        # path stored on the name cell for delete targeting
        from PySide6.QtCore import Qt

        self.assertEqual(dlg._table.item(0, 0).data(Qt.ItemDataRole.UserRole), "/p/medium")
        self.assertIn("3.0 MB", dlg._header.text())  # 1MB + 2MB total


@unittest.skipUnless(_QT_OK, "PySide6 / offscreen Qt unavailable")
class PredownloadTests(unittest.TestCase):
    """Round 0069 (round 0022 Phase C): predownload-from-UI in ModelCacheDialog."""

    def _dialog(self):
        from voice2text.diagnostics_dialogs import ModelCacheDialog

        ModelCacheDialog._run = lambda self: None  # type: ignore
        return ModelCacheDialog(RuntimeConfig())

    def test_predownload_success_calls_prewarm_with_selected_language_and_reruns_scan(self):
        dlg = self._dialog()
        idx = dlg._predownload_lang_combo.findData("zh-hant")
        self.assertGreaterEqual(idx, 0)
        dlg._predownload_lang_combo.setCurrentIndex(idx)

        calls: dict[str, object] = {}

        class _FakeTranscriber:
            def prewarm(self, lang):
                calls["prewarm_lang"] = lang

        def fake_create_stt_transcriber(config, *, progress_callback=None, **kwargs):
            calls["config_language"] = config.source_language
            calls["config_provider"] = config.stt_provider
            calls["config_diarization"] = config.whisperx_enable_diarization
            if progress_callback is not None:
                progress_callback("[download] fake progress")
            return _FakeTranscriber()

        rerun_calls = {"n": 0}
        dlg._run = lambda: rerun_calls.__setitem__("n", rerun_calls["n"] + 1)  # type: ignore

        with patch("voice2text.stt.factory.create_stt_transcriber", side_effect=fake_create_stt_transcriber):
            dlg._start_predownload()
            self.assertTrue(_pump_until(lambda: rerun_calls["n"] > 0))

        self.assertEqual(calls["prewarm_lang"], "zh-hant")
        self.assertEqual(calls["config_language"], "zh-hant")
        self.assertEqual(calls["config_provider"], "whisperx")
        self.assertFalse(calls["config_diarization"])
        self.assertIn(dlg._t["predownload_done"], dlg._predownload_status.text())
        self.assertIn("zh-hant", dlg._predownload_status.text())
        self.assertTrue(dlg._predownload_btn.isEnabled())

    def test_predownload_failure_shows_message_and_reenables_button(self):
        dlg = self._dialog()

        def fake_create_stt_transcriber(config, *, progress_callback=None, **kwargs):
            raise RuntimeError("boom")

        with patch("voice2text.stt.factory.create_stt_transcriber", side_effect=fake_create_stt_transcriber):
            dlg._start_predownload()
            self.assertTrue(_pump_until(lambda: dlg._predownload_btn.isEnabled()))

        self.assertIn("boom", dlg._predownload_status.text())


if __name__ == "__main__":
    unittest.main()
