"""Reusable source-selection dialog used by settings UI."""
from __future__ import annotations

from typing import Callable, Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .i18n import SETTINGS_I18N, normalize_ui_language


class SourceSelectionDialog(QDialog):
    def __init__(
        self,
        title: str,
        entries: Sequence[tuple[str, str]],
        selected_values: Sequence[str],
        parent: QWidget | None = None,
        refresh_entries_callback: Callable[[], Sequence[tuple[str, str]]] | None = None,
        ui_language: str = "zh",
    ) -> None:
        super().__init__(parent)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self._lang = normalize_ui_language(ui_language)
        self._entries = list(entries)
        self._refresh_entries_callback = refresh_entries_callback
        self._checks: list[QCheckBox] = []

        self.setWindowTitle(title)
        self.setMinimumSize(460, 420)

        root = QVBoxLayout(self)
        header = QHBoxLayout()
        self._select_all = QCheckBox(SETTINGS_I18N[self._lang]["select_all"])
        self._select_all.stateChanged.connect(self._on_select_all_changed)
        header.addWidget(self._select_all)
        header.addStretch(1)

        self._refresh_button = QToolButton()
        self._refresh_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self._refresh_button.setToolTip(SETTINGS_I18N[self._lang]["refresh"])
        self._refresh_button.setVisible(self._refresh_entries_callback is not None)
        self._refresh_button.clicked.connect(self._on_refresh_clicked)
        header.addWidget(self._refresh_button)
        root.addLayout(header)

        body = QWidget()
        self._body_layout = QVBoxLayout(body)
        self._body_layout.setContentsMargins(0, 0, 0, 0)
        self._body_layout.setSpacing(4)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)
        root.addWidget(scroll, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self._rebuild_entries(selected_values)

    @property
    def selected_values(self) -> list[str]:
        values: list[str] = []
        for cb in self._checks:
            if cb.isChecked():
                values.append(str(cb.property("value")))
        return values

    def _on_select_all_changed(self, state: int) -> None:
        checked = state == Qt.CheckState.Checked.value
        for cb in self._checks:
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)
        self._refresh_select_all_state()

    def _refresh_select_all_state(self) -> None:
        if not self._checks:
            self._select_all.setEnabled(False)
            self._select_all.setChecked(False)
            return
        checked_count = sum((1 for cb in self._checks if cb.isChecked()))
        all_count = len(self._checks)
        self._select_all.blockSignals(True)
        if checked_count == 0:
            self._select_all.setTristate(False)
            self._select_all.setCheckState(Qt.CheckState.Unchecked)
        elif checked_count == all_count:
            self._select_all.setTristate(False)
            self._select_all.setCheckState(Qt.CheckState.Checked)
        else:
            self._select_all.setTristate(True)
            self._select_all.setCheckState(Qt.CheckState.PartiallyChecked)
        self._select_all.blockSignals(False)

    def _on_refresh_clicked(self) -> None:
        if self._refresh_entries_callback is None:
            return
        selected = self.selected_values
        self._entries = list(self._refresh_entries_callback())
        self._rebuild_entries(selected)

    def _rebuild_entries(self, selected_values: Sequence[str]) -> None:
        selected = set(selected_values)
        for cb in self._checks:
            cb.deleteLater()
        self._checks.clear()
        while self._body_layout.count():
            item = self._body_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        if not self._entries:
            self._body_layout.addWidget(QLabel(SETTINGS_I18N[self._lang]["no_sources"]))
        else:
            for value, label in self._entries:
                cb = QCheckBox(label)
                cb.setProperty("value", value)
                cb.setChecked(value in selected)
                cb.stateChanged.connect(self._refresh_select_all_state)
                self._checks.append(cb)
                self._body_layout.addWidget(cb)
        self._body_layout.addStretch(1)
        self._refresh_select_all_state()
