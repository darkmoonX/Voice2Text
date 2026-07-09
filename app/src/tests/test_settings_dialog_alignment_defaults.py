"""Rounds 0077+0079: Settings dialog's generalized per-language alignment-model default.

The alignment-model dropdown (round 0079) is one flat list grouped by a disabled header row
per language, covering every curated language at once; right-clicking a candidate ticks/unticks
it as that row's language default (whisperx_alignment_model_defaults) without ever closing the
dropdown, so several languages can be adjusted in one open/close cycle."""
from __future__ import annotations

import os
from pathlib import Path
import sys
from unittest import mock
import unittest

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from voice2text.config import RuntimeConfig
from voice2text.settings.presenter import alignment_repos_for_language

try:
    from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
    from PySide6.QtGui import QContextMenuEvent, QMouseEvent
    from PySide6.QtWidgets import QApplication

    _APP = QApplication.instance() or QApplication([])
    _QT_OK = True
except Exception:  # pragma: no cover - headless without offscreen plugin
    _QT_OK = False


@unittest.skipUnless(_QT_OK, "PySide6 / offscreen Qt platform unavailable")
class AlignmentModelDefaultsDialogTests(unittest.TestCase):
    def _dialog(self, config: RuntimeConfig | None = None, *, discovered: dict | None = None):
        from voice2text.settings_dialog import SettingsDialog

        # Round 0081: _refresh_alignment_model_suggestions() now also scans the real on-disk
        # alignment cache via discover_custom_alignment_candidates(). Patch it so these tests
        # stay hermetic regardless of what's actually been downloaded on the machine running
        # them; tests that care about the merge behavior pass an explicit `discovered` dict.
        patcher = mock.patch(
            "voice2text.settings_dialog.discover_custom_alignment_candidates",
            return_value=dict(discovered or {}),
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        return SettingsDialog(config or RuntimeConfig(), devices=[], app_sessions=[])

    @staticmethod
    def _row_for_repo(dlg, repo: str) -> int:
        model = dlg._whisperx_align_model_edit.model()
        for row in range(dlg._whisperx_align_model_edit.count()):
            item = model.item(row)
            if item is not None and item.text().strip() == repo:
                return row
        raise AssertionError(f"{repo!r} not found in suggestions")

    @staticmethod
    def _right_click_row(dlg, row: int) -> None:
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

    def test_dropdown_has_one_header_per_curated_language_with_its_candidates(self) -> None:
        # Round 0079: unlike the old single-language dropdown, every curated language's group
        # is present at once regardless of the currently-selected alignment/source language.
        dlg = self._dialog()
        model = dlg._whisperx_align_model_edit.model()
        for lang_key, label in dlg._ALIGNMENT_DEFAULT_LANGUAGES:
            header_row = self._row_for_repo(dlg, f"— {label} —")
            header_item = model.item(header_row)
            self.assertEqual(header_item.flags(), Qt.ItemFlag.NoItemFlags)
            repos = [
                model.item(row).text().strip()
                for row, row_lang in dlg._alignment_suggestion_row_language.items()
                if row_lang == lang_key
            ]
            self.assertEqual(sorted(repos), sorted(alignment_repos_for_language(lang_key)))

    def test_discovered_custom_candidate_appears_in_its_language_group(self) -> None:
        # Round 0081: a custom repo found via discover_custom_alignment_candidates() (i.e.
        # actually downloaded/used at some point) shows up as an extra, unchecked, tickable
        # row under its language's group, without needing to be on the static curated list.
        dlg = self._dialog(discovered={"zh": ["some-org/my-custom-zh-align-model"]})
        row = self._row_for_repo(dlg, "some-org/my-custom-zh-align-model")
        self.assertEqual(dlg._alignment_suggestion_row_language.get(row), "zh")
        self.assertEqual(
            dlg._whisperx_align_model_edit.model().item(row).checkState(), Qt.CheckState.Unchecked
        )

    def test_discovered_custom_candidate_already_curated_is_not_duplicated(self) -> None:
        dlg = self._dialog(
            discovered={"zh": ["wbbbbb/wav2vec2-large-chinese-zh-cn"]}  # already curated
        )
        rows = [
            row
            for row, lang in dlg._alignment_suggestion_row_language.items()
            if lang == "zh"
            and dlg._whisperx_align_model_edit.model().item(row).text().strip()
            == "wbbbbb/wav2vec2-large-chinese-zh-cn"
        ]
        self.assertEqual(len(rows), 1)

    def test_persisted_custom_default_not_in_curated_or_discovered_list_still_shows_and_ticks(
        self,
    ) -> None:
        # Closes the previously-deferred "orphaned display" gap: a persisted default that is
        # neither on the curated list nor (yet) discovered on disk must still show up, ticked,
        # since the setting is already in effect either way.
        dlg = self._dialog(
            RuntimeConfig(
                whisperx_alignment_model_defaults={"ja": "some-org/never-downloaded-repo"},
            )
        )
        row = self._row_for_repo(dlg, "some-org/never-downloaded-repo")
        self.assertEqual(dlg._alignment_suggestion_row_language.get(row), "ja")
        self.assertEqual(
            dlg._whisperx_align_model_edit.model().item(row).checkState(), Qt.CheckState.Checked
        )

    def test_refresh_ticks_each_languages_current_default_simultaneously(self) -> None:
        dlg = self._dialog(
            RuntimeConfig(
                whisperx_alignment_model_defaults={
                    "zh": "wbbbbb/wav2vec2-large-chinese-zh-cn",
                    "en": "WAV2VEC2_ASR_LARGE_LV60K_960H",
                },
            )
        )
        model = dlg._whisperx_align_model_edit.model()
        zh_row = self._row_for_repo(dlg, "wbbbbb/wav2vec2-large-chinese-zh-cn")
        self.assertEqual(model.item(zh_row).checkState(), Qt.CheckState.Checked)
        zh_other_row = self._row_for_repo(dlg, "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn")
        self.assertEqual(model.item(zh_other_row).checkState(), Qt.CheckState.Unchecked)
        en_row = self._row_for_repo(dlg, "WAV2VEC2_ASR_LARGE_LV60K_960H")
        self.assertEqual(model.item(en_row).checkState(), Qt.CheckState.Checked)

    def test_right_click_ticks_writes_map_and_clears_pin(self) -> None:
        dlg = self._dialog()
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

    def test_right_click_matching_current_text_does_not_clear_it(self) -> None:
        # Round 0084: right-clicking the very candidate that's already shown in the field must
        # not blank it out — that has zero functional effect (pin == repo == the new default) but
        # visually reads as "my selection just vanished". Only an actually-different stale pin
        # should be cleared (covered by test_right_click_ticks_writes_map_and_clears_pin above).
        dlg = self._dialog()
        repo = "wbbbbb/wav2vec2-large-chinese-zh-cn"
        dlg._whisperx_align_model_edit.setCurrentText(repo)
        row = self._row_for_repo(dlg, repo)

        self._right_click_row(dlg, row)

        self.assertEqual(dlg._alignment_model_defaults.get("zh"), repo)
        self.assertEqual(dlg._whisperx_align_model_edit.model().item(row).checkState(), Qt.CheckState.Checked)
        self.assertEqual(dlg._whisperx_align_model_edit.currentText().strip(), repo)

    def test_right_click_again_unticks(self) -> None:
        dlg = self._dialog(
            RuntimeConfig(
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

    def test_right_clicking_two_languages_in_the_same_open_dropdown(self) -> None:
        # The core round-0079 ask: adjust several languages' defaults without the dropdown ever
        # rebuilding/closing between clicks (rebuild=False path), then only rebuild once at the
        # end via _refresh_alignment_model_suggestions to confirm both stuck.
        dlg = self._dialog()
        zh_row = self._row_for_repo(dlg, "wbbbbb/wav2vec2-large-chinese-zh-cn")
        en_row = self._row_for_repo(dlg, "WAV2VEC2_ASR_LARGE_LV60K_960H")

        self._right_click_row(dlg, zh_row)
        self._right_click_row(dlg, en_row)

        self.assertEqual(
            dlg._alignment_model_defaults,
            {
                "zh": "wbbbbb/wav2vec2-large-chinese-zh-cn",
                "en": "WAV2VEC2_ASR_LARGE_LV60K_960H",
            },
        )

    def test_header_row_is_not_checkable_and_right_click_ignores_it(self) -> None:
        dlg = self._dialog()
        label = dict(dlg._ALIGNMENT_DEFAULT_LANGUAGES)["zh"]
        header_row = self._row_for_repo(dlg, f"— {label} —")

        self._right_click_row(dlg, header_row)

        self.assertEqual(dlg._alignment_model_defaults, {})

    def test_right_click_forces_dropdown_back_open_if_closed(self) -> None:
        # Round 0080: belt-and-suspenders — if the popup still somehow ended up hidden despite
        # the hidePopup() suppression below, the handler must force it back open, so a
        # right-click never ends the "keep adjusting" workflow.
        dlg = self._dialog()
        row = self._row_for_repo(dlg, "wbbbbb/wav2vec2-large-chinese-zh-cn")
        calls = []
        dlg._whisperx_align_model_edit.showPopup = lambda: calls.append(True)

        self._right_click_row(dlg, row)

        self.assertEqual(len(calls), 1)

    def test_combo_hide_popup_is_suppressible(self) -> None:
        # The _AlignmentModelCombo subclass: hidePopup() is a no-op while suppress_hide_popup is
        # set, and behaves normally otherwise.
        dlg = self._dialog()
        combo = dlg._whisperx_align_model_edit
        combo.showPopup()
        self.assertTrue(combo.view().isVisible())

        combo.suppress_hide_popup = True
        combo.hidePopup()
        self.assertTrue(combo.view().isVisible())

        combo.suppress_hide_popup = False
        combo.hidePopup()
        self.assertFalse(combo.view().isVisible())

    def test_show_popup_always_clears_suppression(self) -> None:
        # Safety net: a stuck suppress flag (e.g. a right-click gesture that never reached the
        # context-menu handler) must not survive into the next legitimate popup open.
        dlg = self._dialog()
        combo = dlg._whisperx_align_model_edit
        combo.arm_suppression()

        combo.showPopup()

        self.assertFalse(combo.suppress_hide_popup)
        self.assertFalse(combo._suppress_release_timer.isActive())

    def test_arm_suppression_starts_a_grace_timer_and_release_clears_it(self) -> None:
        # Round 0082: suppression is released via a short grace-period timer, not immediately,
        # since a live trace showed the actual close is a SECOND, DELAYED hidePopup() call
        # arriving after the immediate gesture is done — see _AlignmentModelCombo's docstring.
        dlg = self._dialog()
        combo = dlg._whisperx_align_model_edit

        combo.arm_suppression()

        self.assertTrue(combo.suppress_hide_popup)
        self.assertTrue(combo._suppress_release_timer.isActive())

        combo._release_suppression()  # simulate the timer firing, without waiting for real time

        self.assertFalse(combo.suppress_hide_popup)

    def test_right_button_press_on_viewport_arms_suppression(self) -> None:
        # Round 0080: the dialog's eventFilter, installed on the popup's viewport, must arm
        # suppress_hide_popup as soon as the right button goes down — before any context-menu
        # handling — since whatever closes the popup on right-click may act during the press/
        # release phase, ahead of the deferred context-menu event.
        dlg = self._dialog()
        combo = dlg._whisperx_align_model_edit
        viewport = combo.view().viewport()
        press = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            QPointF(1, 1),
            QPointF(1, 1),
            Qt.MouseButton.RightButton,
            Qt.MouseButton.RightButton,
            Qt.KeyboardModifier.NoModifier,
        )

        dlg.eventFilter(viewport, press)

        self.assertTrue(combo.suppress_hide_popup)
        self.assertTrue(combo._suppress_release_timer.isActive())

    def test_context_menu_event_is_swallowed_and_toggles_directly(self) -> None:
        # Round 0083: the eventFilter now intercepts the raw QEvent.Type.ContextMenu itself
        # (the event a live trace tied to the delayed native-driven close) before Qt's own
        # contextMenuEvent() handling ever sees it, does the tick-toggle right there, and
        # swallows it (returns True) instead of only reacting after the fact via a timer.
        dlg = self._dialog()
        combo = dlg._whisperx_align_model_edit
        combo.showPopup()
        row = self._row_for_repo(dlg, "wbbbbb/wav2vec2-large-chinese-zh-cn")
        model = combo.model()
        view = combo.view()
        index = model.index(row, 0)
        original = view.indexAt
        view.indexAt = lambda pos: index
        try:
            event = QContextMenuEvent(QContextMenuEvent.Reason.Mouse, QPoint(0, 0), QPoint(0, 0))
            handled = dlg.eventFilter(view.viewport(), event)
        finally:
            view.indexAt = original

        self.assertTrue(handled)
        self.assertEqual(model.item(row).checkState(), Qt.CheckState.Checked)
        self.assertTrue(combo.suppress_hide_popup)
        self.assertTrue(combo._suppress_release_timer.isActive())

    def test_right_click_mouse_press_and_release_are_swallowed(self) -> None:
        # Round 0083: the raw right-button press/release are also swallowed (belt-and-suspenders
        # against any other click-driven side effect), even though the ContextMenu event above is
        # the actual load-bearing fix.
        dlg = self._dialog()
        combo = dlg._whisperx_align_model_edit
        viewport = combo.view().viewport()
        press = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            QPointF(1, 1),
            QPointF(1, 1),
            Qt.MouseButton.RightButton,
            Qt.MouseButton.RightButton,
            Qt.KeyboardModifier.NoModifier,
        )
        release = QMouseEvent(
            QEvent.Type.MouseButtonRelease,
            QPointF(1, 1),
            QPointF(1, 1),
            Qt.MouseButton.RightButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )

        self.assertTrue(dlg.eventFilter(viewport, press))
        self.assertTrue(dlg.eventFilter(viewport, release))

    def test_context_menu_handler_re_arms_suppression_grace_period_when_done(self) -> None:
        # Round 0082: the handler must NOT clear suppression immediately on finishing — it
        # re-arms the grace timer instead, since the real close is a delayed second hidePopup()
        # call that can land after the handler returns. Only the timer firing (simulated here)
        # actually releases it.
        dlg = self._dialog()
        combo = dlg._whisperx_align_model_edit
        combo.showPopup()  # so the handler's "still hidden?" fallback path isn't what's exercised
        combo.arm_suppression()
        row = self._row_for_repo(dlg, "wbbbbb/wav2vec2-large-chinese-zh-cn")

        self._right_click_row(dlg, row)

        self.assertTrue(combo.suppress_hide_popup)
        self.assertTrue(combo._suppress_release_timer.isActive())

        combo._release_suppression()

        self.assertFalse(combo.suppress_hide_popup)

    def test_hide_popup_logs_to_the_voice2text_logger(self) -> None:
        # Round 0080 follow-up: every hidePopup() call (suppressed or not) is logged at INFO to
        # the 'voice2text' logger — the same one DebugWindowLogHandler (INFO-level) forwards
        # into the Debug Trace window/log file — so a real "still closes sometimes" occurrence
        # can be captured live and traced back to its call stack.
        dlg = self._dialog()
        combo = dlg._whisperx_align_model_edit
        combo.showPopup()

        with self.assertLogs("voice2text", level="INFO") as captured:
            combo.suppress_hide_popup = True
            combo.hidePopup()
        self.assertTrue(any("SUPPRESSED" in msg for msg in captured.output))

        with self.assertLogs("voice2text", level="INFO") as captured:
            combo.suppress_hide_popup = False
            combo.hidePopup()
        self.assertTrue(any("executing" in msg for msg in captured.output))


if __name__ == "__main__":
    unittest.main()
