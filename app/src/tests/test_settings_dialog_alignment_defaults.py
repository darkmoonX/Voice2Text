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
    from PySide6.QtCore import QPoint, Qt
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

    @staticmethod
    def _row_for_repo(dlg, repo: str) -> int:
        model = dlg._whisperx_align_model_edit.model()
        for row in range(dlg._whisperx_align_model_edit.count()):
            item = model.item(row)
            if item is not None and item.text().strip() == repo:
                return row
        raise AssertionError(f"{repo!r} not found in alignment-model suggestions")

    def _right_click_row(self, dlg, row: int) -> None:
        # Bypass pixel-geometry indexAt() (unreliable offscreen) by monkeypatching it to
        # return the target row's index directly, exactly what a real click would resolve to.
        model = dlg._whisperx_align_model_edit.model()
        view = dlg._whisperx_align_model_edit.view()
        index = model.index(row, 0)
        original = view.indexAt
        view.indexAt = lambda pos: index
        try:
            dlg._on_align_model_suggestion_context_menu(QPoint(0, 0))
        finally:
            view.indexAt = original

    def test_refresh_ticks_the_current_default(self) -> None:
        dlg = self._dialog(
            RuntimeConfig(
                source_language="zh-hant",
                whisperx_alignment_model_defaults={"zh": "wbbbbb/wav2vec2-large-chinese-zh-cn"},
            )
        )
        row = self._row_for_repo(dlg, "wbbbbb/wav2vec2-large-chinese-zh-cn")
        model = dlg._whisperx_align_model_edit.model()
        self.assertEqual(model.item(row).checkState(), Qt.CheckState.Checked)
        other_row = self._row_for_repo(dlg, "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn")
        self.assertEqual(model.item(other_row).checkState(), Qt.CheckState.Unchecked)

    def test_right_click_ticks_writes_map_and_clears_pin(self) -> None:
        dlg = self._dialog(RuntimeConfig(source_language="zh-hant"))
        dlg._whisperx_align_model_edit.setCurrentText(
            "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
        )
        row = self._row_for_repo(dlg, "wbbbbb/wav2vec2-large-chinese-zh-cn")

        self._right_click_row(dlg, row)

        self.assertEqual(
            dlg._alignment_model_defaults.get("zh"), "wbbbbb/wav2vec2-large-chinese-zh-cn"
        )
        self.assertEqual(dlg._whisperx_align_model_edit.model().item(row).checkState(), Qt.CheckState.Checked)
        self.assertEqual(dlg._whisperx_align_model_edit.currentText().strip(), "")

    def test_right_click_again_unticks(self) -> None:
        dlg = self._dialog(
            RuntimeConfig(
                source_language="zh-hant",
                whisperx_alignment_model_defaults={"zh": "wbbbbb/wav2vec2-large-chinese-zh-cn"},
            )
        )
        row = self._row_for_repo(dlg, "wbbbbb/wav2vec2-large-chinese-zh-cn")

        self._right_click_row(dlg, row)

        self.assertNotIn("zh", dlg._alignment_model_defaults)
        self.assertEqual(dlg._whisperx_align_model_edit.model().item(row).checkState(), Qt.CheckState.Unchecked)

    def test_at_most_one_tick_per_language(self) -> None:
        dlg = self._dialog(
            RuntimeConfig(
                source_language="zh-hant",
                whisperx_alignment_model_defaults={"zh": "wbbbbb/wav2vec2-large-chinese-zh-cn"},
            )
        )
        old_row = self._row_for_repo(dlg, "wbbbbb/wav2vec2-large-chinese-zh-cn")
        new_row = self._row_for_repo(dlg, "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn")

        self._right_click_row(dlg, new_row)

        model = dlg._whisperx_align_model_edit.model()
        self.assertEqual(model.item(new_row).checkState(), Qt.CheckState.Checked)
        self.assertEqual(model.item(old_row).checkState(), Qt.CheckState.Unchecked)
        self.assertEqual(
            dlg._alignment_model_defaults.get("zh"),
            "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
        )


if __name__ == "__main__":
    unittest.main()
