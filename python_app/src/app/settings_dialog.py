from __future__ import annotations

from typing import Sequence

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .audio_capture import AudioDevice
from .config import RuntimeConfig


class SourceSelectionDialog(QDialog):
    def __init__(
        self,
        title: str,
        entries: Sequence[tuple[str, str]],
        selected_values: Sequence[str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(460, 420)

        self._entries = list(entries)
        selected = set(selected_values)
        self._checks: list[QCheckBox] = []

        root = QVBoxLayout(self)

        self._select_all = QCheckBox("全選")
        self._select_all.stateChanged.connect(self._on_select_all_changed)
        root.addWidget(self._select_all)

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(4)

        if not self._entries:
            body_layout.addWidget(QLabel("(沒有可選的來源)"))
        else:
            for value, label in self._entries:
                cb = QCheckBox(label)
                cb.setProperty("value", value)
                cb.setChecked(value in selected)
                cb.stateChanged.connect(self._refresh_select_all_state)
                self._checks.append(cb)
                body_layout.addWidget(cb)

        body_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)
        root.addWidget(scroll, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._refresh_select_all_state()

    @property
    def selected_values(self) -> list[str]:
        values: list[str] = []
        for cb in self._checks:
            if cb.isChecked():
                value = str(cb.property("value"))
                values.append(value)
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

        checked_count = sum(1 for cb in self._checks if cb.isChecked())
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


class SettingsDialog(QDialog):
    def __init__(
        self,
        config: RuntimeConfig,
        devices: Sequence[AudioDevice],
        app_sessions: Sequence[str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._devices = list(devices)
        self._app_sessions = list(app_sessions)
        self._updates: dict[str, object] = {}

        self._selected_loopback_indices = self._init_loopback_indices()
        self._selected_app_names = self._init_app_names()

        self.setWindowTitle("Voice2Text 設定")
        self.setMinimumWidth(580)

        self._mode_combo = QComboBox()
        self._mode_combo.addItem("輸出回放", "loopback")
        self._mode_combo.addItem("麥克風", "microphone")
        self._mode_combo.addItem("指定應用", "app")
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)

        self._select_source_btn = QPushButton("選擇來源")
        self._select_source_btn.clicked.connect(self._open_source_selection)
        self._source_summary = QLabel()
        self._source_summary.setWordWrap(True)

        self._segment_spin = QDoubleSpinBox()
        self._segment_spin.setDecimals(2)
        self._segment_spin.setRange(1.0, 12.0)
        self._segment_spin.setSingleStep(0.1)

        self._hop_spin = QDoubleSpinBox()
        self._hop_spin.setDecimals(2)
        self._hop_spin.setRange(0.1, 6.0)
        self._hop_spin.setSingleStep(0.1)

        self._merge_method_combo = QComboBox()
        self._merge_method_combo.addItem("覆蓋最近視窗（推薦）", "replace-window")
        self._merge_method_combo.addItem("字尾重疊合併", "suffix-overlap")
        self._merge_method_combo.addItem("模糊重疊合併", "fuzzy-overlap")
        self._merge_method_combo.addItem("僅追加（舊行為）", "append-only")

        self._source_language_combo = QComboBox()
        self._source_language_combo.addItem("自動", "auto")
        self._source_language_combo.addItem("英文", "en")
        self._source_language_combo.addItem("中文（繁體）", "zh-hant")
        self._source_language_combo.addItem("中文（簡體）", "zh-hans")
        self._source_language_combo.addItem("日文", "ja")
        self._source_language_combo.addItem("韓文", "ko")

        self._translation_enabled_check = QCheckBox()
        self._translation_enabled_check.setText("")
        self._translation_enabled_check.stateChanged.connect(self._on_translation_toggle)

        self._bilingual_combo = QComboBox()
        self._bilingual_combo.addItem("上下分行", "stacked")
        self._bilingual_combo.addItem("僅翻譯", "translation-only")

        self._translation_language_combo = QComboBox()
        self._translation_language_combo.addItem("英文", "en")
        self._translation_language_combo.addItem("中文", "zh")
        self._translation_language_combo.addItem("日文", "ja")
        self._translation_language_combo.addItem("韓文", "ko")

        self._font_size_spin = QSpinBox()
        self._font_size_spin.setRange(10, 60)

        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(20, 100)
        self._opacity_label = QLabel()

        self._source_color_btn = QPushButton()
        self._translated_color_btn = QPushButton()
        self._bg_color_btn = QPushButton()

        self._sync_from_config()
        self._build_ui()

    @property
    def updates(self) -> dict[str, object]:
        return self._updates

    def accept(self) -> None:
        try:
            self._updates = self._collect_updates()
        except ValueError as exc:
            QMessageBox.warning(self, "設定錯誤", str(exc))
            return
        super().accept()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        form = QFormLayout()

        form.addRow("聲音來源模式", self._mode_combo)

        source_row = QHBoxLayout()
        source_row.addWidget(self._select_source_btn)
        source_row.addWidget(self._source_summary, 1)
        form.addRow("來源選擇", source_row)

        form.addRow("分段秒數", self._segment_spin)
        form.addRow("滑動步長秒數", self._hop_spin)
        form.addRow("重疊整合方法", self._merge_method_combo)

        form.addRow("偵測語言", self._source_language_combo)

        translation_label = QWidget()
        translation_label_layout = QHBoxLayout(translation_label)
        translation_label_layout.setContentsMargins(0, 0, 0, 0)
        translation_label_layout.setSpacing(6)
        translation_label_layout.addWidget(QLabel("翻譯"))
        translation_label_layout.addWidget(self._translation_enabled_check)
        translation_label_layout.addStretch(1)
        form.addRow(translation_label, self._bilingual_combo)
        form.addRow("翻譯語言", self._translation_language_combo)

        form.addRow("字體大小", self._font_size_spin)

        opacity_row = QHBoxLayout()
        opacity_row.addWidget(self._opacity_slider, 1)
        opacity_row.addWidget(self._opacity_label)
        form.addRow("透明度", opacity_row)

        self._source_color_btn.clicked.connect(lambda: self._pick_color(self._source_color_btn))
        self._translated_color_btn.clicked.connect(
            lambda: self._pick_color(self._translated_color_btn)
        )
        self._bg_color_btn.clicked.connect(lambda: self._pick_color(self._bg_color_btn))

        form.addRow("來源文字顏色", self._source_color_btn)
        form.addRow("翻譯文字顏色", self._translated_color_btn)
        form.addRow("背景顏色", self._bg_color_btn)

        self._opacity_slider.valueChanged.connect(self._update_opacity_label)
        self._update_opacity_label(self._opacity_slider.value())

        root.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._on_mode_changed()
        self._on_translation_toggle()

    def _init_loopback_indices(self) -> list[int]:
        indices = list(self._config.source_device_indices)
        if not indices and self._config.device_index is not None:
            indices = [self._config.device_index]
        return indices

    def _init_app_names(self) -> list[str]:
        names = list(self._config.source_app_names)
        if not names and self._config.source_app_name:
            names = [self._config.source_app_name]
        return names

    def _sync_from_config(self) -> None:
        self._set_combo_data(self._mode_combo, self._config.source_mode)

        self._segment_spin.setValue(self._config.segment_seconds)
        self._hop_spin.setValue(self._config.hop_seconds)
        self._set_combo_data(self._merge_method_combo, self._config.overlap_merge_method)

        self._translation_enabled_check.setChecked(self._config.translation_enabled)
        style = self._config.bilingual_style
        if style == "inline":
            style = "stacked"
        self._set_combo_data(self._bilingual_combo, style)
        self._set_combo_data(self._translation_language_combo, self._config.translation_to)

        source_language = self._config.source_language or "auto"
        if source_language == "zh":
            source_language = "zh-hant"
        self._set_combo_data(self._source_language_combo, source_language)

        self._font_size_spin.setValue(self._config.font_size)
        self._opacity_slider.setValue(int(self._config.overlay_opacity * 100))

        source_color = self._config.source_text_color or self._config.text_color
        self._set_color_button(self._source_color_btn, source_color)
        self._set_color_button(self._translated_color_btn, self._config.translated_text_color)
        self._set_color_button(self._bg_color_btn, self._config.background_color)

    def _on_mode_changed(self) -> None:
        mode = self._mode_combo.currentData()
        selectable = mode in {"loopback", "app"}
        self._select_source_btn.setVisible(selectable)
        self._source_summary.setVisible(selectable)

        if mode == "microphone":
            self._source_summary.setText("使用預設麥克風來源")
            return

        if mode == "loopback":
            if not self._selected_loopback_indices:
                self._source_summary.setText("尚未選擇來源")
            else:
                values = ", ".join(str(v) for v in self._selected_loopback_indices)
                self._source_summary.setText(f"已選擇裝置索引: {values}")
            return

        if not self._selected_app_names:
            self._source_summary.setText("尚未選擇應用")
        else:
            self._source_summary.setText("已選擇應用: " + ", ".join(self._selected_app_names))

    def _open_source_selection(self) -> None:
        mode = self._mode_combo.currentData()

        if mode == "loopback":
            entries = [
                (str(dev.index), f"[{dev.index}] {dev.name}")
                for dev in self._devices
                if dev.kind == "loopback"
            ]
            selected = [str(v) for v in self._selected_loopback_indices]
            dlg = SourceSelectionDialog(
                "選擇輸出回放來源",
                entries,
                selected,
                parent=self,
            )
            if dlg.exec() == SourceSelectionDialog.DialogCode.Accepted:
                self._selected_loopback_indices = [int(v) for v in dlg.selected_values]
                self._on_mode_changed()
            return

        if mode == "app":
            entries = [(name, name) for name in self._app_sessions]
            dlg = SourceSelectionDialog(
                "選擇應用來源",
                entries,
                self._selected_app_names,
                parent=self,
            )
            if dlg.exec() == SourceSelectionDialog.DialogCode.Accepted:
                self._selected_app_names = dlg.selected_values
                self._on_mode_changed()

    def _on_translation_toggle(self) -> None:
        enabled = self._translation_enabled_check.isChecked()
        self._bilingual_combo.setEnabled(enabled)
        self._translation_language_combo.setEnabled(enabled)
        self._translated_color_btn.setEnabled(enabled)

    def _collect_updates(self) -> dict[str, object]:
        source_mode = self._mode_combo.currentData()
        source_lang_data = self._source_language_combo.currentData()
        if source_lang_data == "auto":
            translation_from = "auto"
        elif source_lang_data in {"zh-hant", "zh-hans"}:
            translation_from = "zh"
        else:
            translation_from = str(source_lang_data)
        translation_to = str(self._translation_language_combo.currentData())

        segment_seconds = float(self._segment_spin.value())
        hop_seconds = float(self._hop_spin.value())
        if hop_seconds > segment_seconds:
            raise ValueError("滑動步長秒數不能大於分段秒數。")

        source_device_indices: list[int] = []
        source_app_names: list[str] = []
        if source_mode == "loopback":
            source_device_indices = list(self._selected_loopback_indices)
        elif source_mode == "app":
            source_app_names = list(self._selected_app_names)

        return {
            "source_mode": source_mode,
            "source_device_indices": source_device_indices,
            "source_app_name": source_app_names[0] if source_app_names else "",
            "source_app_names": source_app_names,
            "source_language": None
            if self._source_language_combo.currentData() == "auto"
            else self._source_language_combo.currentData(),
            "segment_seconds": segment_seconds,
            "hop_seconds": hop_seconds,
            "overlap_merge_method": self._merge_method_combo.currentData(),
            "translation_enabled": self._translation_enabled_check.isChecked(),
            "translation_from": translation_from,
            "translation_to": translation_to,
            "bilingual_style": self._bilingual_combo.currentData(),
            "font_size": int(self._font_size_spin.value()),
            "overlay_opacity": float(self._opacity_slider.value()) / 100.0,
            "text_color": self._source_color_btn.text().strip(),
            "source_text_color": self._source_color_btn.text().strip(),
            "translated_text_color": self._translated_color_btn.text().strip(),
            "background_color": self._bg_color_btn.text().strip(),
        }

    def _pick_color(self, button: QPushButton) -> None:
        initial = QColor(button.text())
        selected = QColorDialog.getColor(initial, self, "選擇顏色")
        if selected.isValid():
            self._set_color_button(button, selected.name())

    def _set_color_button(self, button: QPushButton, color: str) -> None:
        button.setText(color)
        text_color = "#000000" if QColor(color).lightness() > 120 else "#FFFFFF"
        button.setStyleSheet(f"background:{color}; color:{text_color};")

    def _update_opacity_label(self, value: int) -> None:
        self._opacity_label.setText(f"{value}%")

    @staticmethod
    def _set_combo_data(combo: QComboBox, value: str) -> None:
        idx = combo.findData(value)
        combo.setCurrentIndex(max(0, idx))