"""Round 0077: Settings dialog's generalized per-language alignment-model default
(right-click "set/clear as default" on the alignment-model combo)."""
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
class AlignmentModelDefaultsDialogTests(unittest.TestCase):
    def _dialog(self, config: RuntimeConfig | None = None):
        from voice2text.settings_dialog import SettingsDialog

        return SettingsDialog(config or RuntimeConfig(), devices=[], app_sessions=[])

    def test_sync_reads_map_from_config(self) -> None:
        cfg = RuntimeConfig(
            whisperx_alignment_model_defaults={"zh": "wbbbbb/wav2vec2-large-chinese-zh-cn"}
        )
        dlg = self._dialog(cfg)
        self.assertEqual(
            dlg._alignment_model_defaults, {"zh": "wbbbbb/wav2vec2-large-chinese-zh-cn"}
        )

    def test_set_default_clears_explicit_pin_and_updates_map(self) -> None:
        dlg = self._dialog()
        dlg._whisperx_align_model_edit.setEditText("jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn")

        dlg._set_alignment_default_for_language("zh", "wbbbbb/wav2vec2-large-chinese-zh-cn")

        self.assertEqual(
            dlg._alignment_model_defaults, {"zh": "wbbbbb/wav2vec2-large-chinese-zh-cn"}
        )
        self.assertEqual(dlg._whisperx_align_model_edit.currentText().strip(), "")

    def test_clear_default_removes_language_entry(self) -> None:
        dlg = self._dialog(
            RuntimeConfig(whisperx_alignment_model_defaults={"zh": "wbbbbb/wav2vec2-large-chinese-zh-cn"})
        )
        dlg._clear_alignment_default_for_language("zh")
        self.assertEqual(dlg._alignment_model_defaults, {})

    def test_collect_updates_carries_the_map(self) -> None:
        dlg = self._dialog()
        dlg._set_alignment_default_for_language("ja", "kresnik/wav2vec2-large-xlsr-korean")
        updates = dlg._collect_updates()
        self.assertEqual(
            updates["whisperx_alignment_model_defaults"], {"ja": "kresnik/wav2vec2-large-xlsr-korean"}
        )

    def test_no_leftover_wbbbbb_checkbox(self) -> None:
        # Round 0077 removed the single-language checkbox in favor of the generalized map.
        dlg = self._dialog()
        self.assertFalse(hasattr(dlg, "_whisperx_zh_align_wbbbbb_check"))


if __name__ == "__main__":
    unittest.main()
