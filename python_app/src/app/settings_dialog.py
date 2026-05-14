"""Settings dialog UI for capture/STT/runtime options and live config updates."""
from __future__ import annotations

from typing import Callable, Sequence

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
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .capture import AudioDevice
from .config import RuntimeConfig
from .settings.i18n import SETTINGS_I18N, normalize_ui_language
from .settings.mapping import SettingsPayloadInput, build_settings_updates
from .settings.schema import allowed_stt_variants, default_stt_model, is_path_like, provider_supports_source_language



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


class SettingsDialog(QDialog):
    def __init__(
        self,
        config: RuntimeConfig,
        devices: Sequence[AudioDevice],
        app_sessions: Sequence[str],
        device_provider: Callable[[], Sequence[AudioDevice]] | None = None,
        app_session_provider: Callable[[], Sequence[str]] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._devices = list(devices)
        self._app_sessions = list(app_sessions)
        self._device_provider = device_provider
        self._app_session_provider = app_session_provider
        self._updates: dict[str, object] = {}
        self._form_layout: QFormLayout | None = None
        self._last_stt_provider = str(config.stt_provider or "whisper").strip() or "whisper"
        self._selected_loopback_indices = self._init_loopback_indices()
        self._selected_app_names = self._init_app_names()
        self._lang = normalize_ui_language(config.ui_language)

        self.setWindowTitle(self._t("settings_title"))
        self.setMinimumWidth(600)

        self._mode_combo = QComboBox()
        self._mode_combo.addItem("Loopback", "loopback")
        self._mode_combo.addItem("Microphone", "microphone")
        self._mode_combo.addItem("App Session", "app")
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)

        self._ui_language_combo = QComboBox()
        self._ui_language_combo.addItem("繁體中文", "zh")
        self._ui_language_combo.addItem("English", "en")
        self._ui_language_note = QLabel(self._t("ui_next_open"))
        self._ui_language_note.setWordWrap(True)

        self._select_source_btn = QPushButton(self._t("source"))
        self._select_source_btn.clicked.connect(self._open_source_selection)
        self._source_summary = QLabel()
        self._source_summary.setWordWrap(True)

        self._stt_provider_combo = QComboBox()
        self._stt_provider_combo.addItem("Whisper (faster-whisper)", "whisper")
        self._stt_provider_combo.addItem("Vosk", "vosk")
        self._stt_provider_combo.addItem("Sherpa-ONNX", "sherpa-onnx")
        self._stt_provider_combo.addItem("NVIDIA Riva", "riva")
        self._stt_provider_combo.addItem("FunASR", "funasr")
        self._stt_provider_combo.currentIndexChanged.connect(self._on_stt_provider_changed)

        self._stt_variant_combo = QComboBox()
        self._stt_variant_combo.addItem("Auto", "auto")
        self._stt_variant_combo.addItem("CPU", "cpu")
        self._stt_variant_combo.addItem("GPU", "gpu")

        self._stt_model_path_edit = QLineEdit()
        self._stt_auto_download_check = QCheckBox(self._t("auto_download"))
        self._sherpa_provider_combo = QComboBox()
        self._sherpa_provider_combo.addItem("CPU", "cpu")
        self._sherpa_provider_combo.addItem("CUDA", "cuda")

        self._riva_uri_edit = QLineEdit()
        self._riva_uri_edit.setPlaceholderText("localhost:50051")
        self._riva_language_edit = QLineEdit()
        self._riva_language_edit.setPlaceholderText("en-US")
        self._riva_ssl_check = QCheckBox("TLS")
        self._riva_ssl_check.stateChanged.connect(self._on_stt_provider_changed)
        self._riva_ssl_cert_edit = QLineEdit()
        self._riva_api_key_edit = QLineEdit()

        self._funasr_device_edit = QLineEdit()
        self._funasr_vad_model_edit = QLineEdit()
        self._stt_hint = QLabel()
        self._stt_hint.setWordWrap(True)

        self._segment_spin = QDoubleSpinBox()
        self._segment_spin.setDecimals(2)
        self._segment_spin.setRange(1.0, 12.0)
        self._segment_spin.setSingleStep(0.1)
        self._hop_spin = QDoubleSpinBox()
        self._hop_spin.setDecimals(2)
        self._hop_spin.setRange(0.1, 6.0)
        self._hop_spin.setSingleStep(0.1)

        self._merge_method_combo = QComboBox()
        self._merge_method_combo.addItem("stable-tail", "stable-tail")
        self._merge_method_combo.addItem("commit-on-break", "commit-on-break")

        self._preprocess_enabled_check = QCheckBox()
        self._preprocess_modules_edit = QLineEdit()
        self._vad_adaptive_check = QCheckBox()
        self._vad_threshold_spin = QDoubleSpinBox()
        self._vad_threshold_spin.setDecimals(4)
        self._vad_threshold_spin.setRange(0.0, 0.2)
        self._vad_threshold_spin.setSingleStep(0.001)

        self._source_language_combo = QComboBox()
        self._source_language_combo.addItem("auto", "auto")
        self._source_language_combo.addItem("en", "en")
        self._source_language_combo.addItem("zh-hant", "zh-hant")
        self._source_language_combo.addItem("zh-hans", "zh-hans")
        self._source_language_combo.addItem("ja", "ja")
        self._source_language_combo.addItem("ko", "ko")

        self._translation_enabled_check = QCheckBox()
        self._translation_enabled_check.stateChanged.connect(self._on_translation_toggle)
        self._bilingual_combo = QComboBox()
        self._bilingual_combo.addItem("stacked", "stacked")
        self._bilingual_combo.addItem("translation-only", "translation-only")
        self._translation_language_combo = QComboBox()
        self._translation_language_combo.addItem("en", "en")
        self._translation_language_combo.addItem("zh", "zh")
        self._translation_language_combo.addItem("ja", "ja")
        self._translation_language_combo.addItem("ko", "ko")

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

    def _t(self, key: str) -> str:
        return SETTINGS_I18N[self._lang][key]

    def accept(self) -> None:
        try:
            self._updates = self._collect_updates()
        except ValueError as exc:
            QMessageBox.warning(self, self._t("invalid_settings"), str(exc))
            return
        super().accept()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        form = QFormLayout()
        self._form_layout = form

        form.addRow(self._t("ui_language"), self._ui_language_combo)
        form.addRow("", self._ui_language_note)
        form.addRow(self._t("source_mode"), self._mode_combo)

        source_row = QHBoxLayout()
        source_row.addWidget(self._select_source_btn)
        source_row.addWidget(self._source_summary, 1)
        form.addRow(self._t("source"), source_row)

        form.addRow(self._t("stt_provider"), self._stt_provider_combo)
        form.addRow(self._t("stt_variant"), self._stt_variant_combo)
        form.addRow(self._t("model_path"), self._stt_model_path_edit)
        form.addRow(self._t("auto_download"), self._stt_auto_download_check)
        form.addRow(self._t("sherpa_runtime"), self._sherpa_provider_combo)
        form.addRow(self._t("riva_uri"), self._riva_uri_edit)
        form.addRow(self._t("riva_lang"), self._riva_language_edit)
        form.addRow(self._t("riva_tls"), self._riva_ssl_check)
        form.addRow(self._t("riva_cert"), self._riva_ssl_cert_edit)
        form.addRow(self._t("riva_key"), self._riva_api_key_edit)
        form.addRow(self._t("funasr_device"), self._funasr_device_edit)
        form.addRow(self._t("funasr_vad"), self._funasr_vad_model_edit)
        form.addRow(self._t("preprocess"), self._preprocess_enabled_check)
        form.addRow(self._t("preprocess_modules"), self._preprocess_modules_edit)
        form.addRow(self._t("adaptive_vad"), self._vad_adaptive_check)
        form.addRow(self._t("vad_rms"), self._vad_threshold_spin)
        form.addRow(self._t("stt_notes"), self._stt_hint)
        form.addRow(self._t("segment_seconds"), self._segment_spin)
        form.addRow(self._t("hop_seconds"), self._hop_spin)
        form.addRow(self._t("merge_method"), self._merge_method_combo)
        form.addRow(self._t("source_language"), self._source_language_combo)

        translation_label = QWidget()
        translation_label_layout = QHBoxLayout(translation_label)
        translation_label_layout.setContentsMargins(0, 0, 0, 0)
        translation_label_layout.setSpacing(6)
        translation_label_layout.addWidget(QLabel(self._t("translation")))
        translation_label_layout.addWidget(self._translation_enabled_check)
        translation_label_layout.addStretch(1)
        form.addRow(translation_label, self._bilingual_combo)
        form.addRow(self._t("translation_target"), self._translation_language_combo)

        form.addRow(self._t("font_size"), self._font_size_spin)
        opacity_row = QHBoxLayout()
        opacity_row.addWidget(self._opacity_slider, 1)
        opacity_row.addWidget(self._opacity_label)
        form.addRow(self._t("opacity"), opacity_row)

        self._source_color_btn.clicked.connect(lambda: self._pick_color(self._source_color_btn))
        self._translated_color_btn.clicked.connect(lambda: self._pick_color(self._translated_color_btn))
        self._bg_color_btn.clicked.connect(lambda: self._pick_color(self._bg_color_btn))
        form.addRow(self._t("source_color"), self._source_color_btn)
        form.addRow(self._t("translated_color"), self._translated_color_btn)
        form.addRow(self._t("background_color"), self._bg_color_btn)

        self._opacity_slider.valueChanged.connect(self._update_opacity_label)
        self._update_opacity_label(self._opacity_slider.value())
        root.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._on_mode_changed()
        self._on_stt_provider_changed()
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
        self._set_combo_data(self._ui_language_combo, self._config.ui_language)
        self._set_combo_data(self._mode_combo, self._config.source_mode)
        self._set_combo_data(self._stt_provider_combo, self._config.stt_provider)
        self._set_combo_data(self._stt_variant_combo, self._config.stt_variant)
        self._stt_model_path_edit.setText(self._config.stt_model_path)
        self._stt_auto_download_check.setChecked(self._config.stt_auto_download)
        self._set_combo_data(self._sherpa_provider_combo, self._config.sherpa_onnx_provider)
        self._riva_uri_edit.setText(self._config.riva_uri)
        self._riva_language_edit.setText(self._config.riva_language_code)
        self._riva_ssl_check.setChecked(self._config.riva_use_ssl)
        self._riva_ssl_cert_edit.setText(self._config.riva_ssl_cert)
        self._riva_api_key_edit.setText(self._config.riva_api_key)
        self._funasr_device_edit.setText(self._config.funasr_device)
        self._funasr_vad_model_edit.setText(self._config.funasr_vad_model)
        self._segment_spin.setValue(self._config.segment_seconds)
        self._hop_spin.setValue(self._config.hop_seconds)
        self._set_combo_data(self._merge_method_combo, self._config.overlap_merge_method)
        self._preprocess_enabled_check.setChecked(self._config.preprocess_enabled)
        self._preprocess_modules_edit.setText(self._config.preprocess_modules)
        self._vad_adaptive_check.setChecked(self._config.vad_adaptive_enabled)
        self._vad_threshold_spin.setValue(self._config.vad_rms_threshold)
        self._translation_enabled_check.setChecked(self._config.translation_enabled)
        style = self._config.bilingual_style if self._config.bilingual_style != "inline" else "stacked"
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
            self._source_summary.setText(self._t("mic_default"))
            return
        if mode == "loopback":
            if not self._selected_loopback_indices:
                self._source_summary.setText(self._t("loopback_default"))
            else:
                values = ", ".join((str(v) for v in self._selected_loopback_indices))
                self._source_summary.setText(f'{self._t("loopback_selected")} {values}')
            return
        if not self._selected_app_names:
            self._source_summary.setText(self._t("apps_none"))
        else:
            self._source_summary.setText(f'{self._t("apps_selected")} {", ".join(self._selected_app_names)}')

    def _on_stt_provider_changed(self) -> None:
        """Sync provider-dependent fields (variant, model default, language lock, provider-specific options)."""
        provider = str(self._stt_provider_combo.currentData() or "whisper")
        previous_provider = self._last_stt_provider
        self._last_stt_provider = provider
        is_sherpa = provider == "sherpa-onnx"
        is_riva = provider == "riva"
        is_funasr = provider == "funasr"
        self._configure_stt_variant(provider)
        self._apply_stt_model_default(provider, previous_provider=previous_provider)
        self._configure_source_language_field(provider)

        self._stt_model_path_edit.setEnabled(not is_riva)
        self._stt_auto_download_check.setEnabled(not is_riva)

        self._sherpa_provider_combo.setEnabled(is_sherpa)
        self._set_form_row_visible(self._sherpa_provider_combo, is_sherpa)

        self._riva_uri_edit.setEnabled(is_riva)
        self._riva_language_edit.setEnabled(is_riva)
        self._riva_ssl_check.setEnabled(is_riva)
        self._riva_ssl_cert_edit.setEnabled(is_riva and self._riva_ssl_check.isChecked())
        self._riva_api_key_edit.setEnabled(is_riva)
        self._set_form_row_visible(self._riva_uri_edit, is_riva)
        self._set_form_row_visible(self._riva_language_edit, is_riva)
        self._set_form_row_visible(self._riva_ssl_check, is_riva)
        self._set_form_row_visible(self._riva_ssl_cert_edit, is_riva and self._riva_ssl_check.isChecked())
        self._set_form_row_visible(self._riva_api_key_edit, is_riva)

        self._funasr_device_edit.setEnabled(is_funasr)
        self._funasr_vad_model_edit.setEnabled(is_funasr)
        self._set_form_row_visible(self._funasr_device_edit, is_funasr)
        self._set_form_row_visible(self._funasr_vad_model_edit, is_funasr)

        if provider == "whisper":
            self._stt_hint.setText(self._t("provider_hint_whisper"))
        elif provider == "vosk":
            self._stt_hint.setText(self._t("provider_hint_vosk"))
        elif provider == "sherpa-onnx":
            self._stt_hint.setText(self._t("provider_hint_sherpa"))
        elif provider == "riva":
            self._stt_hint.setText(self._t("provider_hint_riva"))
        else:
            self._stt_hint.setText(self._t("provider_hint_funasr"))

    def _configure_stt_variant(self, provider: str) -> None:
        current = str(self._stt_variant_combo.currentData() or "auto")
        allowed = allowed_stt_variants(provider)
        label_map = {"auto": "Auto", "cpu": "CPU", "gpu": "GPU"}
        self._stt_variant_combo.blockSignals(True)
        self._stt_variant_combo.clear()
        for item in allowed:
            self._stt_variant_combo.addItem(label_map.get(item, item.upper()), item)
        next_value = current if current in allowed else ("auto" if "auto" in allowed else allowed[0])
        self._set_combo_data(self._stt_variant_combo, next_value)
        self._stt_variant_combo.setEnabled(len(allowed) > 1)
        self._stt_variant_combo.blockSignals(False)

    def _configure_source_language_field(self, provider: str) -> None:
        supported = provider_supports_source_language(provider)
        self._source_language_combo.setEnabled(supported)
        if not supported:
            self._set_combo_data(self._source_language_combo, "auto")

    def _apply_stt_model_default(self, provider: str, *, previous_provider: str) -> None:
        if provider == previous_provider:
            return
        current = self._stt_model_path_edit.text().strip()
        if is_path_like(current):
            return
        default_value = default_stt_model(provider)
        self._stt_model_path_edit.setText(default_value)

    def _open_source_selection(self) -> None:
        self._refresh_available_sources()
        mode = self._mode_combo.currentData()
        if mode == "loopback":
            dlg = SourceSelectionDialog(
                self._t("select_loopback"),
                self._build_loopback_entries(),
                [str(v) for v in self._selected_loopback_indices],
                parent=self,
                refresh_entries_callback=self._refresh_loopback_entries_for_dialog,
                ui_language=self._lang,
            )
            if dlg.exec() == SourceSelectionDialog.DialogCode.Accepted:
                self._selected_loopback_indices = [int(v) for v in dlg.selected_values]
                self._on_mode_changed()
            return
        if mode == "app":
            dlg = SourceSelectionDialog(
                self._t("select_apps"),
                self._build_app_entries(),
                self._selected_app_names,
                parent=self,
                refresh_entries_callback=self._refresh_app_entries_for_dialog,
                ui_language=self._lang,
            )
            if dlg.exec() == SourceSelectionDialog.DialogCode.Accepted:
                self._selected_app_names = dlg.selected_values
                self._on_mode_changed()

    def _refresh_available_sources(self) -> None:
        if self._device_provider is not None:
            try:
                self._devices = list(self._device_provider())
            except Exception:
                pass
        if self._app_session_provider is not None:
            try:
                self._app_sessions = list(self._app_session_provider())
            except Exception:
                pass

    def _build_loopback_entries(self) -> list[tuple[str, str]]:
        return [(str(dev.index), f"[{dev.index}] {dev.name}") for dev in self._devices if dev.kind == "loopback"]

    def _build_app_entries(self) -> list[tuple[str, str]]:
        return [(name, name) for name in self._app_sessions]

    def _refresh_loopback_entries_for_dialog(self) -> Sequence[tuple[str, str]]:
        self._refresh_available_sources()
        return self._build_loopback_entries()

    def _refresh_app_entries_for_dialog(self) -> Sequence[tuple[str, str]]:
        self._refresh_available_sources()
        return self._build_app_entries()

    def _on_translation_toggle(self) -> None:
        enabled = self._translation_enabled_check.isChecked()
        self._bilingual_combo.setEnabled(enabled)
        self._translation_language_combo.setEnabled(enabled)
        self._translated_color_btn.setEnabled(enabled)

    def _collect_updates(self) -> dict[str, object]:
        """Collect validated settings payload applied by bootstrap.apply_settings()."""
        payload = SettingsPayloadInput(
            ui_language=str(self._ui_language_combo.currentData() or self._lang),
            source_mode=str(self._mode_combo.currentData()),
            stt_provider=str(self._stt_provider_combo.currentData() or "whisper"),
            stt_variant=str(self._stt_variant_combo.currentData() or "auto"),
            stt_model_path=self._stt_model_path_edit.text(),
            stt_auto_download=self._stt_auto_download_check.isChecked(),
            sherpa_onnx_provider=str(self._sherpa_provider_combo.currentData() or "cpu"),
            riva_uri=self._riva_uri_edit.text(),
            riva_use_ssl=self._riva_ssl_check.isChecked(),
            riva_ssl_cert=self._riva_ssl_cert_edit.text(),
            riva_language_code=self._riva_language_edit.text(),
            riva_api_key=self._riva_api_key_edit.text(),
            funasr_device=self._funasr_device_edit.text(),
            funasr_vad_model=self._funasr_vad_model_edit.text(),
            source_language=str(self._source_language_combo.currentData()),
            translation_to=str(self._translation_language_combo.currentData()),
            segment_seconds=float(self._segment_spin.value()),
            hop_seconds=float(self._hop_spin.value()),
            selected_loopback_indices=list(self._selected_loopback_indices),
            selected_app_names=list(self._selected_app_names),
            overlap_merge_method=str(self._merge_method_combo.currentData()),
            preprocess_enabled=self._preprocess_enabled_check.isChecked(),
            preprocess_modules=self._preprocess_modules_edit.text(),
            vad_adaptive_enabled=self._vad_adaptive_check.isChecked(),
            vad_rms_threshold=float(self._vad_threshold_spin.value()),
            translation_enabled=self._translation_enabled_check.isChecked(),
            bilingual_style=str(self._bilingual_combo.currentData()),
            font_size=int(self._font_size_spin.value()),
            overlay_opacity=float(self._opacity_slider.value()) / 100.0,
            source_text_color=self._source_color_btn.text(),
            translated_text_color=self._translated_color_btn.text(),
            background_color=self._bg_color_btn.text(),
        )
        return build_settings_updates(
            payload,
            lang=str(self._ui_language_combo.currentData() or self._lang),
            need_riva_uri_message=self._t("need_riva_uri"),
            hop_gt_segment_message=self._t("hop_gt_segment"),
        )

    def _pick_color(self, button: QPushButton) -> None:
        initial = QColor(button.text())
        selected = QColorDialog.getColor(initial, self, self._t("pick_color"))
        if selected.isValid():
            self._set_color_button(button, selected.name())

    def _set_color_button(self, button: QPushButton, color: str) -> None:
        button.setText(color)
        text_color = "#000000" if QColor(color).lightness() > 120 else "#FFFFFF"
        button.setStyleSheet(f"background:{color}; color:{text_color};")

    def _update_opacity_label(self, value: int) -> None:
        self._opacity_label.setText(f"{value}%")

    def _set_form_row_visible(self, field: QWidget, visible: bool) -> None:
        if self._form_layout is None:
            return
        label = self._form_layout.labelForField(field)
        if label is not None:
            label.setVisible(visible)
        field.setVisible(visible)

    @staticmethod
    def _set_combo_data(combo: QComboBox, value: str) -> None:
        idx = combo.findData(value)
        combo.setCurrentIndex(max(0, idx))




