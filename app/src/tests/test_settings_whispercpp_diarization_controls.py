from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from voice2text.config import RuntimeConfig

try:
    from PySide6.QtWidgets import QApplication

    _APP = QApplication.instance() or QApplication([])
    _QT_OK = True
except Exception:  # pragma: no cover - headless without offscreen plugin
    _QT_OK = False


@unittest.skipUnless(_QT_OK, "PySide6 / offscreen Qt platform unavailable")
class WhispercppProviderControlGatingTests(unittest.TestCase):
    def _dialog(self):
        from voice2text.settings_dialog import SettingsDialog

        return SettingsDialog(RuntimeConfig(), devices=[], app_sessions=[])

    def _select_provider(self, dlg, provider: str) -> None:
        dlg._stt_provider_combo.setCurrentIndex(dlg._stt_provider_combo.findData(provider))

    def test_whispercpp_keeps_diarization_and_speaker_profile_controls_enabled(self) -> None:
        # Round 0065 gave whisper.cpp's live path its own diarization module honoring the
        # same whisperx_diarization_*/whisperx_speaker_* config keys, so a user must be able
        # to enable these from the Settings dialog too, not just by hand-editing JSON.
        dlg = self._dialog()
        self._select_provider(dlg, "whispercpp")

        self.assertTrue(dlg._whisperx_diarization_check.isEnabled())
        self.assertTrue(dlg._whisperx_diar_device_combo.isEnabled())
        self.assertTrue(dlg._whisperx_diar_model_edit.isEnabled())
        self.assertTrue(dlg._whisperx_hf_token_edit.isEnabled())
        self.assertTrue(dlg._whisperx_speaker_profile_check.isEnabled())
        self.assertTrue(dlg._whisperx_speaker_backend_combo.isEnabled())

    def test_whispercpp_still_disables_alignment_only_controls(self) -> None:
        # whisper.cpp genuinely has no forced-alignment pass (CLAUDE.md hard constraint) -
        # these controls must stay disabled, this round only re-enables diarization/speaker
        # controls, not alignment ones.
        dlg = self._dialog()
        self._select_provider(dlg, "whispercpp")

        self.assertFalse(dlg._whisperx_vad_check.isEnabled())
        self.assertFalse(dlg._whisperx_align_language_combo.isEnabled())
        self.assertFalse(dlg._whisperx_align_device_combo.isEnabled())
        self.assertFalse(dlg._whisperx_align_guard_combo.isEnabled())
        self.assertFalse(dlg._whisperx_align_guard_revert_btn.isEnabled())
        self.assertFalse(dlg._whisperx_align_model_edit.isEnabled())

    def test_whisperx_provider_keeps_all_controls_enabled(self) -> None:
        dlg = self._dialog()
        self._select_provider(dlg, "whispercpp")
        self._select_provider(dlg, "whisperx")

        for field in (
            dlg._whisperx_vad_check,
            dlg._whisperx_diarization_check,
            dlg._whisperx_align_language_combo,
            dlg._whisperx_align_device_combo,
            dlg._whisperx_align_guard_combo,
            dlg._whisperx_diar_device_combo,
            dlg._whisperx_align_model_edit,
            dlg._whisperx_diar_model_edit,
            dlg._whisperx_hf_token_edit,
            dlg._whisperx_speaker_profile_check,
            dlg._whisperx_speaker_backend_combo,
        ):
            self.assertTrue(field.isEnabled())

    def test_whispercpp_diarization_toggle_round_trips_through_collect_updates(self) -> None:
        dlg = self._dialog()
        self._select_provider(dlg, "whispercpp")
        dlg._whisperx_diarization_check.setChecked(True)
        dlg._whisperx_speaker_profile_check.setChecked(True)

        updates = dlg._collect_updates()

        self.assertEqual(updates["stt_provider"], "whispercpp")
        self.assertTrue(updates["whisperx_enable_diarization"])
        self.assertTrue(updates["whisperx_speaker_profile_enabled"])


if __name__ == "__main__":
    unittest.main()
