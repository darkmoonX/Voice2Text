from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice2text.config import RuntimeConfig
from voice2text.settings.source_selection_dialog import SourceSelectionDialog
from voice2text.settings_dialog import SettingsDialog, TranscriptExportDialog


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _is_topmost(dialog) -> bool:
    return bool(dialog.windowFlags() & Qt.WindowType.WindowStaysOnTopHint)


def test_settings_dialog_stays_above_overlay() -> None:
    _app()
    dialog = SettingsDialog(
        config=RuntimeConfig(),
        devices=[],
        app_sessions=[],
        parent=None,
    )

    assert _is_topmost(dialog)


def test_transcript_export_dialog_stays_above_overlay() -> None:
    _app()
    dialog = TranscriptExportDialog(
        auto_export_enabled=False,
        include_timestamps=True,
        include_speaker=True,
        default_format="txt",
        lang="zh",
        parent=None,
    )

    assert _is_topmost(dialog)


def test_transcript_export_dialog_is_txt_display_export_only() -> None:
    _app()
    dialog = TranscriptExportDialog(
        auto_export_enabled=False,
        include_timestamps=True,
        include_speaker=True,
        default_format="json",
        lang="zh",
        parent=None,
    )

    assert dialog.export_format == "txt"
    assert dialog.include_timestamps is False
    assert dialog.include_speaker is True


def test_source_selection_dialog_stays_above_overlay() -> None:
    _app()
    dialog = SourceSelectionDialog(
        "Select",
        entries=[],
        selected_values=[],
        parent=None,
    )

    assert _is_topmost(dialog)
