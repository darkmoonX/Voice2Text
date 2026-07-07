"""Settings dialog UI for capture/STT/runtime options and live config updates."""
from __future__ import annotations

from pathlib import Path
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
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .capture import AudioDevice
from .config import RuntimeConfig
from .pipeline.transcript_exporter import export_format_suffix
from .settings.i18n import SETTINGS_I18N, normalize_ui_language
from .settings.mapping import SettingsPayloadInput, build_settings_updates
from .settings.presets import PRESET_NAMES, apply_preset, normalize_preset
from .settings.presenter import (
    alignment_repos_for_language,
    app_names_from_config,
    loopback_indices_from_config,
    normalize_source_language,
)
from .settings.schema import allowed_stt_variants, default_stt_model
from .settings.source_selection_dialog import SourceSelectionDialog
from .settings.widgets import (
    create_compute_type_combo,
    create_mode_combo,
    create_source_language_combo,
    create_stt_provider_combo,
    create_stt_variant_combo,
    create_translation_backend_combo,
    create_whisperx_align_device_combo,
    create_whisperx_align_guard_combo,
    create_whisperx_diarization_device_combo,
    create_whisperx_speaker_backend_combo,
    create_translation_language_combo,
    create_whisperx_align_language_combo,
)


def _keep_above_overlay(dialog: QDialog) -> None:
    """Keep operator dialogs above the always-on-top subtitle overlay."""
    dialog.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)


class TranscriptExportDialog(QDialog):
    # (format key, English label, Chinese label) — `display` is the overlay text as shown.
    _FORMAT_ITEMS: tuple[tuple[str, str, str], ...] = (
        ("display", "Displayed text (.txt)", "畫面顯示文字 (.txt)"),
        ("txt", "Timestamped text (.txt)", "時間戳文字 (.txt)"),
        ("srt", "SubRip subtitle (.srt)", "SubRip 字幕 (.srt)"),
        ("json", "JSON transcript (.json)", "JSON 逐字稿 (.json)"),
    )

    def __init__(
        self,
        *,
        auto_export_enabled: bool,
        include_timestamps: bool,
        include_speaker: bool,
        default_format: str,
        lang: str = "en",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        _keep_above_overlay(self)
        zh = str(lang or "").strip().lower() == "zh"
        self.setWindowTitle("匯出字幕" if zh else "Export Subtitle")
        self.setMinimumWidth(420)

        root = QVBoxLayout(self)
        form = QFormLayout()

        self._auto_export_check = QCheckBox("停止時自動匯出" if zh else "Enable auto export on stop")
        self._auto_export_check.setChecked(bool(auto_export_enabled))
        form.addRow("自動匯出" if zh else "Auto export", self._auto_export_check)

        self._format_combo = QComboBox()
        for key, label_en, label_zh in self._FORMAT_ITEMS:
            self._format_combo.addItem(label_zh if zh else label_en, key)
        self._select_format(default_format)
        form.addRow("格式" if zh else "Format", self._format_combo)

        self._timestamps_check = QCheckBox("含時間戳" if zh else "Include timestamps")
        self._timestamps_check.setChecked(bool(include_timestamps))
        form.addRow("時間戳" if zh else "Timestamps", self._timestamps_check)

        self._speaker_check = QCheckBox("含說話人" if zh else "Include speaker")
        self._speaker_check.setChecked(bool(include_speaker))
        form.addRow("說話人" if zh else "Speaker", self._speaker_check)

        description = QLabel(
            "「畫面顯示文字」匯出主畫面目前看到的字幕；TXT/SRT/JSON 匯出含時間軸的逐字稿。"
            if zh
            else "“Displayed text” exports the overlay as shown; TXT/SRT/JSON export the timed transcript."
        )
        description.setWordWrap(True)
        form.addRow("", description)

        self._format_combo.currentIndexChanged.connect(self._sync_option_enabled)
        self._sync_option_enabled()

        root.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _select_format(self, fmt: str) -> None:
        target = str(fmt or "").strip().lower()
        index = self._format_combo.findData(target)
        self._format_combo.setCurrentIndex(index if index >= 0 else 0)

    def _sync_option_enabled(self) -> None:
        fmt = self.export_format
        # SRT always carries timecodes; the displayed-text dump ignores both options.
        self._timestamps_check.setEnabled(fmt in {"txt", "json"})
        self._speaker_check.setEnabled(fmt in {"txt", "srt", "json"})

    @property
    def auto_export_enabled(self) -> bool:
        return self._auto_export_check.isChecked()

    @property
    def include_timestamps(self) -> bool:
        return self._timestamps_check.isChecked()

    @property
    def include_speaker(self) -> bool:
        return self._speaker_check.isChecked()

    @property
    def export_format(self) -> str:
        data = self._format_combo.currentData()
        return str(data or "display")


class SettingsDialog(QDialog):
    # Sentinel data for the model-size combo's "Other…" entry (reveals the custom field).
    CUSTOM_MODEL_DATA = "__custom__"

    def __init__(
        self,
        config: RuntimeConfig,
        devices: Sequence[AudioDevice],
        app_sessions: Sequence[str],
        device_provider: Callable[[], Sequence[AudioDevice]] | None = None,
        app_session_provider: Callable[[], Sequence[str]] | None = None,
        export_transcript_callback: Callable[[str, str, bool, bool], str] | None = None,
        import_audio_callback: Callable[[str, str], str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        _keep_above_overlay(self)
        self._config = config
        self._default_config = RuntimeConfig()
        self._devices = list(devices)
        self._app_sessions = list(app_sessions)
        self._device_provider = device_provider
        self._app_session_provider = app_session_provider
        self._export_transcript_callback = export_transcript_callback
        self._import_audio_callback = import_audio_callback
        self._updates: dict[str, object] = {}
        self._form_layouts: list[QFormLayout] = []
        self._last_stt_provider = "whisperx"
        # Remember each provider's last model-size selection so switching providers back
        # and forth restores the previous choice instead of snapping to the default.
        self._stt_model_size_by_provider: dict[str, str] = {}
        self._ui_built = False
        self._applying_preset = False
        self._selected_loopback_indices = self._init_loopback_indices()
        self._selected_app_names = self._init_app_names()
        self._lang = normalize_ui_language(config.ui_language)

        self.setWindowTitle(self._t("settings_title"))
        self.setMinimumWidth(600)

        self._mode_combo = create_mode_combo()
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

        self._stt_provider_combo = create_stt_provider_combo()
        self._stt_provider_combo.currentIndexChanged.connect(self._on_stt_provider_changed)

        self._preset_combo = QComboBox()
        self._preset_combo.addItem("（未套用）" if self._lang == "zh" else "(none)", "")
        for _name in PRESET_NAMES:
            self._preset_combo.addItem(_name, _name)
        self._preset_combo.currentIndexChanged.connect(self._on_preset_changed)

        self._model_size_combo = QComboBox()
        # 'auto' resolves by effective device at build time: large-v3 on CUDA, small on
        # CPU (round 0072). whisper.cpp sessions ignore it (mapping pins medium).
        for _m in ("auto", "tiny", "base", "small", "medium", "large-v2", "large-v3"):
            self._model_size_combo.addItem(_m, _m)
        # "Other…" reveals a single custom field (model name OR local path) so size and
        # custom path are a mutually-exclusive choice — never two path-like fields at once.
        self._model_size_combo.addItem(self._t("model_size_custom"), self.CUSTOM_MODEL_DATA)
        self._model_size_combo.currentIndexChanged.connect(self._on_bundled_field_edited)
        self._model_size_combo.currentIndexChanged.connect(self._update_custom_model_visibility)

        self._beam_spin = QSpinBox()
        self._beam_spin.setRange(1, 10)
        self._beam_spin.valueChanged.connect(self._on_bundled_field_edited)

        self._whisperx_speaker_profile_check = QCheckBox()
        self._whisperx_speaker_profile_check.toggled.connect(self._on_bundled_field_edited)

        self._stt_variant_combo = create_stt_variant_combo()
        self._compute_type_combo = create_compute_type_combo()
        self._compute_type_combo.currentIndexChanged.connect(self._on_bundled_field_edited)

        self._stt_model_path_edit = QLineEdit()
        self._stt_model_path_edit.textChanged.connect(self._on_bundled_field_edited)
        # Row label kept as an attribute so the whole custom-model row can be hidden together.
        self._stt_model_custom_label = QLabel(self._t("model_path"))
        self._stt_auto_download_check = QCheckBox(self._t("auto_download"))
        self._whisperx_vad_check = QComboBox()
        self._whisperx_vad_check.addItem("silero-vad", "silero-vad")
        self._whisperx_vad_check.addItem("pyannote", "pyannote")
        self._whisperx_diarization_check = QCheckBox()
        self._whisperx_diarization_check.toggled.connect(self._on_bundled_field_edited)
        self._whisperx_align_model_edit = QComboBox()
        self._whisperx_align_language_combo = create_whisperx_align_language_combo()
        self._whisperx_align_device_combo = create_whisperx_align_device_combo()
        self._whisperx_align_guard_combo = create_whisperx_align_guard_combo()
        self._whisperx_align_guard_combo.currentIndexChanged.connect(self._on_align_guard_changed)
        self._whisperx_align_guard_warning = QLabel()
        self._whisperx_align_guard_warning.setWordWrap(True)
        self._whisperx_align_guard_warning.setStyleSheet("color: #D9534F;")
        self._whisperx_align_guard_revert_btn = QPushButton()
        self._whisperx_align_guard_revert_btn.clicked.connect(self._on_align_guard_revert)
        self._whisperx_diar_device_combo = create_whisperx_diarization_device_combo()
        self._whisperx_expected_speakers_spin = QSpinBox()
        self._whisperx_expected_speakers_spin.setRange(0, 20)
        self._whisperx_expected_speakers_spin.valueChanged.connect(self._on_bundled_field_edited)
        self._whisperx_align_model_edit.setEditable(True)
        self._whisperx_align_model_edit.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._whisperx_align_model_edit.lineEdit().setPlaceholderText("auto (leave empty) or HF repo id")
        self._whisperx_diar_model_edit = QLineEdit()
        self._whisperx_hf_token_edit = QLineEdit()
        self._whisperx_hf_token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._whisperx_speaker_backend_combo = create_whisperx_speaker_backend_combo()
        # Round 0051: recently-shipped live knobs, previously CLI/JSON-only.
        self._whisperx_zh_align_wbbbbb_check = QCheckBox()
        self._asr_temperatures_edit = QLineEdit()
        self._asr_temperatures_edit.setPlaceholderText("empty = full library schedule (0.0,0.2,0.4,0.6,0.8,1.0)")
        self._commit_hold_spin = QDoubleSpinBox()
        self._commit_hold_spin.setDecimals(1)
        self._commit_hold_spin.setRange(0.0, 120.0)
        self._commit_hold_spin.setSingleStep(1.0)

        self._segment_spin = QDoubleSpinBox()
        self._segment_spin.setDecimals(2)
        self._segment_spin.setRange(1.0, 12.0)
        self._segment_spin.setSingleStep(0.1)
        self._segment_spin.valueChanged.connect(self._on_bundled_field_edited)
        self._hop_spin = QDoubleSpinBox()
        self._hop_spin.setDecimals(2)
        self._hop_spin.setRange(0.1, 6.0)
        self._hop_spin.setSingleStep(0.1)
        self._hop_spin.valueChanged.connect(self._on_bundled_field_edited)

        self._merge_method_combo = QComboBox()
        self._merge_method_combo.addItem("stable-tail", "stable-tail")
        self._merge_method_combo.addItem("commit-on-break", "commit-on-break")

        self._preprocess_enabled_check = QCheckBox()

        self._source_language_combo = create_source_language_combo()
        self._source_language_combo.currentIndexChanged.connect(self._on_source_language_changed)

        self._translation_enabled_check = QCheckBox()
        self._translation_enabled_check.stateChanged.connect(self._on_translation_toggle)
        self._bilingual_combo = QComboBox()
        self._bilingual_combo.addItem("stacked", "stacked")
        self._bilingual_combo.addItem("translation-only", "translation-only")
        self._translation_backend_combo = create_translation_backend_combo()
        self._translation_backend_hint = QLabel(self._t("translation_backend_hint"))
        self._translation_backend_hint.setWordWrap(True)
        self._translation_language_combo = create_translation_language_combo()

        self._font_size_spin = QSpinBox()
        self._font_size_spin.setRange(10, 60)
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(20, 100)
        self._opacity_label = QLabel()

        self._source_color_btn = QPushButton()
        self._translated_color_btn = QPushButton()
        self._bg_color_btn = QPushButton()
        self._debug_mode_check = QCheckBox()
        self._transcript_export_enabled = bool(getattr(config, "transcript_export_enabled", False))
        self._transcript_export_formats = str(
            getattr(config, "transcript_export_formats", "txt,srt,json") or "txt,srt,json"
        )
        self._transcript_export_include_timestamps = bool(
            getattr(config, "transcript_export_include_timestamps", True)
        )
        self._transcript_export_include_speaker = bool(
            getattr(config, "transcript_export_include_speaker", True)
        )
        self._transcript_export_default_format = self._resolve_default_export_format(self._transcript_export_formats)
        self._transcript_export_now_btn = QPushButton("匯出字幕..." if self._lang == "zh" else "Export Subtitle...")
        self._transcript_export_now_btn.clicked.connect(self._on_export_transcript_now)
        self._import_audio_btn = QPushButton("匯入音檔..." if self._lang == "zh" else "Import Audio...")
        self._import_audio_btn.clicked.connect(self._on_import_audio)

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
        form_left = QFormLayout()
        form_right = QFormLayout()
        self._form_layouts = [form_left, form_right]

        form_left.addRow(self._t("ui_language"), self._ui_language_combo)
        form_left.addRow("", self._ui_language_note)
        form_left.addRow(self._t("source_mode"), self._mode_combo)

        source_row = QHBoxLayout()
        source_row.addWidget(self._select_source_btn)
        source_row.addWidget(self._source_summary, 1)
        form_left.addRow(self._t("source"), source_row)

        form_left.addRow(self._t("stt_provider"), self._stt_provider_combo)
        form_left.addRow(self._t("runtime_preset"), self._preset_combo)
        form_left.addRow(self._t("model_size"), self._model_size_combo)
        form_left.addRow(self._t("stt_variant"), self._stt_variant_combo)
        form_left.addRow(self._t("compute_type"), self._compute_type_combo)
        form_left.addRow(self._t("beam_size"), self._beam_spin)
        self._stt_model_form = form_left
        form_left.addRow(self._stt_model_custom_label, self._stt_model_path_edit)
        form_left.addRow(self._t("auto_download"), self._stt_auto_download_check)
        form_left.addRow(self._t("whisperx_vad"), self._whisperx_vad_check)
        # --- Alignment group (kept contiguous) ---
        form_left.addRow(self._t("whisperx_align_language"), self._whisperx_align_language_combo)
        form_left.addRow(self._t("whisperx_align_device"), self._whisperx_align_device_combo)
        align_guard_row = QHBoxLayout()
        align_guard_row.addWidget(self._whisperx_align_guard_combo, 1)
        align_guard_row.addWidget(self._whisperx_align_guard_revert_btn)
        form_left.addRow(self._t("whisperx_align_guard"), align_guard_row)
        form_left.addRow("", self._whisperx_align_guard_warning)
        form_left.addRow(self._t("whisperx_align_model"), self._whisperx_align_model_edit)
        form_left.addRow(self._t("whisperx_zh_align_wbbbbb"), self._whisperx_zh_align_wbbbbb_check)
        # --- Diarization group (kept contiguous) ---
        form_left.addRow(self._t("whisperx_diarization"), self._whisperx_diarization_check)
        form_left.addRow(self._t("whisperx_expected_speakers"), self._whisperx_expected_speakers_spin)
        form_left.addRow("WhisperX Diarization device", self._whisperx_diar_device_combo)
        form_left.addRow(self._t("whisperx_diar_model"), self._whisperx_diar_model_edit)
        form_left.addRow(self._t("whisperx_hf_token"), self._whisperx_hf_token_edit)
        form_left.addRow(self._t("whisperx_speaker_profile"), self._whisperx_speaker_profile_check)
        form_left.addRow(self._t("whisperx_speaker_backend"), self._whisperx_speaker_backend_combo)

        form_right.addRow(self._t("segment_seconds"), self._segment_spin)
        form_right.addRow(self._t("hop_seconds"), self._hop_spin)
        form_right.addRow(self._t("asr_temperatures"), self._asr_temperatures_edit)
        form_right.addRow(self._t("subtitle_commit_hold_seconds"), self._commit_hold_spin)
        form_right.addRow(self._t("merge_method"), self._merge_method_combo)
        form_right.addRow(self._t("preprocess"), self._preprocess_enabled_check)
        form_right.addRow(self._t("source_language"), self._source_language_combo)

        translation_label = QWidget()
        translation_label_layout = QHBoxLayout(translation_label)
        translation_label_layout.setContentsMargins(0, 0, 0, 0)
        translation_label_layout.setSpacing(6)
        translation_label_layout.addWidget(QLabel(self._t("translation")))
        translation_label_layout.addWidget(self._translation_enabled_check)
        translation_label_layout.addStretch(1)
        form_right.addRow(translation_label, self._bilingual_combo)
        form_right.addRow(self._t("translation_backend"), self._translation_backend_combo)
        form_right.addRow("", self._translation_backend_hint)
        form_right.addRow(self._t("translation_target"), self._translation_language_combo)

        form_right.addRow(self._t("font_size"), self._font_size_spin)
        opacity_row = QHBoxLayout()
        opacity_row.addWidget(self._opacity_slider, 1)
        opacity_row.addWidget(self._opacity_label)
        form_right.addRow(self._t("opacity"), opacity_row)

        self._source_color_btn.clicked.connect(lambda: self._pick_color(self._source_color_btn))
        self._translated_color_btn.clicked.connect(lambda: self._pick_color(self._translated_color_btn))
        self._bg_color_btn.clicked.connect(lambda: self._pick_color(self._bg_color_btn))
        form_right.addRow(self._t("source_color"), self._source_color_btn)
        form_right.addRow(self._t("translated_color"), self._translated_color_btn)
        form_right.addRow(self._t("background_color"), self._bg_color_btn)
        form_right.addRow(self._t("debug_mode"), self._debug_mode_check)

        self._opacity_slider.valueChanged.connect(self._update_opacity_label)
        self._update_opacity_label(self._opacity_slider.value())
        columns = QHBoxLayout()
        columns.addLayout(form_left, 1)
        columns.addLayout(form_right, 1)
        root.addLayout(columns)

        footer = QHBoxLayout()
        self._reset_defaults_btn = QPushButton(self._t("reset_defaults"))
        self._reset_defaults_btn.clicked.connect(self._reset_to_defaults)
        footer.addWidget(self._reset_defaults_btn)
        footer.addWidget(self._transcript_export_now_btn)
        footer.addWidget(self._import_audio_btn)
        footer.addStretch(1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        footer.addWidget(buttons)
        root.addLayout(footer)

        self._ui_built = True
        self._update_custom_model_visibility()
        self._on_mode_changed()
        self._on_stt_provider_changed()
        self._on_translation_toggle()
        self._apply_help_tooltips()

    @staticmethod
    def _loopback_indices_from_config(config: RuntimeConfig) -> list[int]:
        return loopback_indices_from_config(config)

    @staticmethod
    def _app_names_from_config(config: RuntimeConfig) -> list[str]:
        return app_names_from_config(config)

    def _init_loopback_indices(self) -> list[int]:
        return self._loopback_indices_from_config(self._config)

    def _init_app_names(self) -> list[str]:
        return self._app_names_from_config(self._config)

    def _sync_from_config(self, config: RuntimeConfig | None = None) -> None:
        cfg = config or self._config
        # Programmatic fill must not trip the "manual edit clears the preset" handler.
        self._applying_preset = True
        try:
            self._sync_from_config_inner(cfg)
        finally:
            self._applying_preset = False
        # Keep the "outgoing provider" marker aligned with the freshly-synced provider so the
        # next _on_stt_provider_changed treats this as a no-op switch (no spurious size save).
        self._last_stt_provider = str(
            self._stt_provider_combo.currentData() or "whisperx"
        ).strip() or "whisperx"

    def _sync_from_config_inner(self, cfg: RuntimeConfig) -> None:
        self._set_combo_data(self._ui_language_combo, cfg.ui_language)
        self._set_combo_data(self._mode_combo, cfg.source_mode)
        self._set_combo_data(self._stt_provider_combo, str(getattr(cfg, "stt_provider", "whisperx") or "whisperx"))
        self._set_combo_data(self._preset_combo, normalize_preset(str(getattr(cfg, "runtime_preset", "") or "")))
        self._set_combo_data(self._stt_variant_combo, cfg.stt_variant)
        self._set_model_size(str(getattr(cfg, "model_size", "auto") or "auto"))
        self._beam_spin.setValue(int(getattr(cfg, "whisper_beam_size", 5) or 5))
        self._whisperx_speaker_profile_check.setChecked(bool(getattr(cfg, "whisperx_speaker_profile_enabled", True)))
        self._set_combo_data(self._compute_type_combo, str(getattr(cfg, "compute_type", "float16") or "float16"))
        provider = str(getattr(cfg, "stt_provider", "whisperx") or "whisperx").strip().lower()
        # The effective model reference per provider is "explicit path/alias OR size" — exactly
        # what the runtime resolves (factory: `stt_model_path or model_size`). _set_model_size
        # then routes it to a preset item or the "Other…" custom field automatically, which
        # also migrates legacy settings that stashed a bare alias in stt_model_path.
        whisperx_effective = (
            str(getattr(cfg, "stt_model_path", "") or "").strip()
            or str(getattr(cfg, "model_size", "auto") or "auto").strip()
            or "auto"
        )
        whispercpp_effective = (
            str(getattr(cfg, "stt_whispercpp_model_path", "") or getattr(cfg, "stt_model_path", "") or "").strip()
            or str(getattr(cfg, "stt_whispercpp_model_size", "") or getattr(cfg, "model_size", "medium") or "medium").strip()
            or "medium"
        )
        if whispercpp_effective.lower() == "auto":
            whispercpp_effective = "medium"  # 'auto' is whisperx-only (round 0072)
        if provider == "whispercpp":
            self._set_model_size(whispercpp_effective)
        else:
            self._set_model_size(whisperx_effective)
        # Seed per-provider model memory so a later provider switch restores each provider's
        # own last selection (preset size or custom value) instead of its hardcoded default.
        self._stt_model_size_by_provider = {
            "whisperx": whisperx_effective,
            "whispercpp": whispercpp_effective,
        }
        self._stt_auto_download_check.setChecked(cfg.stt_auto_download)
        self._set_combo_data(self._whisperx_vad_check, str(getattr(cfg, "whisperx_vad_method", "silero-vad") or "silero-vad"))
        self._whisperx_diarization_check.setChecked(cfg.whisperx_enable_diarization)
        self._set_combo_data(self._whisperx_align_language_combo, str(getattr(cfg, 'whisperx_alignment_language', 'auto') or 'auto'))
        self._set_combo_data(self._whisperx_align_device_combo, str(getattr(cfg, 'whisperx_alignment_device', 'auto') or 'auto'))
        self._set_combo_data(self._whisperx_align_guard_combo, str(getattr(cfg, 'whisperx_align_guard', 'safe') or 'safe'))
        self._update_align_guard_state()
        self._set_combo_data(self._whisperx_diar_device_combo, str(getattr(cfg, 'whisperx_diarization_device', 'auto') or 'auto'))
        min_speakers = int(max(0, getattr(cfg, "whisperx_diarization_min_speakers", 0) or 0))
        max_speakers = int(max(0, getattr(cfg, "whisperx_diarization_max_speakers", 0) or 0))
        self._whisperx_expected_speakers_spin.setValue(min_speakers if min_speakers == max_speakers else 0)
        self._set_alignment_model_value(cfg.whisperx_alignment_model)
        self._whisperx_diar_model_edit.setText(cfg.whisperx_diarization_model)
        self._whisperx_hf_token_edit.setText(cfg.whisperx_hf_token)
        self._set_combo_data(
            self._whisperx_speaker_backend_combo,
            str(getattr(cfg, "whisperx_speaker_profile_backend", "pyannote") or "pyannote"),
        )
        self._whisperx_zh_align_wbbbbb_check.setChecked(bool(getattr(cfg, "whisperx_zh_align_wbbbbb", False)))
        self._asr_temperatures_edit.setText(str(getattr(cfg, "whisperx_asr_temperatures", "") or ""))
        self._commit_hold_spin.setValue(float(getattr(cfg, "subtitle_commit_hold_seconds", 0.0) or 0.0))
        self._segment_spin.setValue(cfg.segment_seconds)
        self._hop_spin.setValue(cfg.hop_seconds)
        self._set_combo_data(self._merge_method_combo, cfg.overlap_merge_method)
        self._preprocess_enabled_check.setChecked(bool(getattr(cfg, "preprocess_enabled", True)))
        self._translation_enabled_check.setChecked(cfg.translation_enabled)
        style = cfg.bilingual_style if cfg.bilingual_style != "inline" else "stacked"
        self._set_combo_data(self._bilingual_combo, style)
        self._set_combo_data(self._translation_backend_combo, str(getattr(cfg, "translation_backend", "argos") or "argos"))
        self._set_combo_data(self._translation_language_combo, cfg.translation_to)
        source_language = normalize_source_language(cfg.source_language)
        self._set_combo_data(self._source_language_combo, source_language)
        self._font_size_spin.setValue(cfg.font_size)
        self._opacity_slider.setValue(int(cfg.overlay_opacity * 100))
        source_color = cfg.source_text_color or cfg.text_color
        self._set_color_button(self._source_color_btn, source_color)
        self._set_color_button(self._translated_color_btn, cfg.translated_text_color)
        self._set_color_button(self._bg_color_btn, cfg.background_color)
        self._debug_mode_check.setChecked(bool(getattr(cfg, "debug_mode", False)))
        self._transcript_export_enabled = bool(getattr(cfg, "transcript_export_enabled", False))
        self._transcript_export_formats = str(
            getattr(cfg, "transcript_export_formats", "txt,srt,json") or "txt,srt,json"
        )
        self._transcript_export_include_timestamps = bool(
            getattr(cfg, "transcript_export_include_timestamps", True)
        )
        self._transcript_export_include_speaker = bool(
            getattr(cfg, "transcript_export_include_speaker", True)
        )
        self._transcript_export_default_format = self._resolve_default_export_format(self._transcript_export_formats)

    def _set_model_size(self, value: str) -> None:
        """Route a model reference into the combo: a known preset selects that item and
        clears the custom field; anything else (custom alias or local path) selects
        "Other…" and lands in the custom field."""
        v = str(value or "small").strip() or "small"
        idx = self._model_size_combo.findData(v)
        if idx < 0 and v != self.CUSTOM_MODEL_DATA:
            idx = self._model_size_combo.findText(v)
        if idx >= 0 and self._model_size_combo.itemData(idx) != self.CUSTOM_MODEL_DATA:
            self._model_size_combo.setCurrentIndex(idx)
            self._stt_model_path_edit.setText("")
        else:
            custom_idx = self._model_size_combo.findData(self.CUSTOM_MODEL_DATA)
            if custom_idx >= 0:
                self._model_size_combo.setCurrentIndex(custom_idx)
            self._stt_model_path_edit.setText(v)
        self._update_custom_model_visibility()

    def _update_custom_model_visibility(self, *args: object) -> None:
        """Show the custom model/path row only when the "Other…" entry is selected."""
        is_custom = self._model_size_combo.currentData() == self.CUSTOM_MODEL_DATA
        self._stt_model_path_edit.setVisible(is_custom)
        self._stt_model_custom_label.setVisible(is_custom)
        form = getattr(self, "_stt_model_form", None)
        if form is not None and hasattr(form, "setRowVisible"):
            try:
                form.setRowVisible(self._stt_model_path_edit, is_custom)
            except Exception:
                pass

    def _current_model_value(self) -> str:
        """The effective model reference: custom field text when "Other…" is active,
        otherwise the selected preset size."""
        if self._model_size_combo.currentData() == self.CUSTOM_MODEL_DATA:
            return self._stt_model_path_edit.text().strip()
        return str(self._model_size_combo.currentText() or "").strip()

    def _collect_model_size_and_path(self) -> tuple[str, str]:
        """Split the current selection into (model_size, custom_path) for the payload.
        model_size is always a real preset (it also feeds whispercpp's ggml-<size>.bin);
        a custom value rides in the path field where both providers' resolvers honor it."""
        provider = str(self._stt_provider_combo.currentData() or "whisperx").strip() or "whisperx"
        if self._model_size_combo.currentData() == self.CUSTOM_MODEL_DATA:
            custom = self._stt_model_path_edit.text().strip()
            fallback = self._stt_model_size_by_provider.get(provider, "").strip()
            if self._model_size_combo.findData(fallback) < 0:
                fallback = default_stt_model(provider)
            return (fallback, custom)
        size = str(self._model_size_combo.currentText() or "small").strip() or "small"
        return (size, "")

    def _on_preset_changed(self) -> None:
        if self._applying_preset:
            return
        name = str(self._preset_combo.currentData() or "")
        if not name:
            return
        preview = RuntimeConfig()
        apply_preset(preview, name)
        self._applying_preset = True
        try:
            self._set_combo_data(self._stt_variant_combo, preview.stt_variant)
            self._set_model_size(preview.model_size)
            self._set_combo_data(self._compute_type_combo, preview.compute_type)
            self._beam_spin.setValue(int(preview.whisper_beam_size or 5))
            self._segment_spin.setValue(float(preview.segment_seconds))
            self._hop_spin.setValue(float(preview.hop_seconds))
            self._whisperx_diarization_check.setChecked(bool(preview.whisperx_enable_diarization))
            self._whisperx_speaker_profile_check.setChecked(bool(preview.whisperx_speaker_profile_enabled))
        finally:
            self._applying_preset = False

    def _preset_forced_alignment(self) -> bool:
        """Forced-alignment value to persist: the active preset's setting, else on (no dedicated widget)."""
        name = str(self._preset_combo.currentData() or "")
        if not name:
            return True
        preview = RuntimeConfig()
        apply_preset(preview, name)
        return bool(getattr(preview, "whisperx_enable_forced_alignment", True))

    def _on_bundled_field_edited(self, *args: object) -> None:
        # A manual edit to any preset-bundled field means the config no longer
        # matches a named preset; drop the preset label (without re-applying).
        if self._applying_preset:
            return
        if not self._preset_combo.currentData():
            return
        self._preset_combo.blockSignals(True)
        self._preset_combo.setCurrentIndex(0)
        self._preset_combo.blockSignals(False)

    def _on_mode_changed(self) -> None:
        mode = self._mode_combo.currentData()
        selectable = mode in {"loopback", "app"}
        if self._ui_built:
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
        """Sync provider-specific controls and hints."""
        provider = str(self._stt_provider_combo.currentData() or "whisperx").strip() or "whisperx"
        previous_provider = self._last_stt_provider
        self._last_stt_provider = provider
        self._configure_stt_variant(provider)
        self._apply_stt_model_default(provider, previous_provider=previous_provider)
        self._source_language_combo.setEnabled(True)

        self._stt_model_path_edit.setEnabled(True)
        self._stt_auto_download_check.setEnabled(True)

        is_whispercpp = provider == "whispercpp"
        # whisper.cpp has no forced-alignment pass (CLAUDE.md hard constraint), so the
        # alignment-only controls stay disabled for it. Diarization/speaker-profile
        # controls are NOT alignment-specific: round 0065 added an independent
        # diarization module for whisper.cpp's live/server path that honors the same
        # whisperx_diarization_*/whisperx_speaker_* config keys, so those fields must
        # stay enabled regardless of provider.
        for field in (
            self._whisperx_vad_check,
            self._whisperx_align_language_combo,
            self._whisperx_align_device_combo,
            self._whisperx_align_guard_combo,
            self._whisperx_align_guard_revert_btn,
            self._whisperx_align_model_edit,
        ):
            field.setEnabled(not is_whispercpp)
        if is_whispercpp:
            self._whisperx_align_guard_warning.setVisible(False)
        else:
            self._refresh_alignment_model_suggestions()
            self._whisperx_align_model_edit.setToolTip("留空=自動依語言選擇；選清單或填 HF repo id=固定使用指定模型")
            self._update_align_guard_state()

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

    def _on_export_transcript_now(self) -> None:
        callback = self._export_transcript_callback
        if callback is None:
            QMessageBox.warning(self, self._t("invalid_settings"), "Export callback is unavailable in current runtime.")
            return
        export_settings = TranscriptExportDialog(
            auto_export_enabled=self._transcript_export_enabled,
            include_timestamps=self._transcript_export_include_timestamps,
            include_speaker=self._transcript_export_include_speaker,
            default_format=self._transcript_export_default_format,
            lang=self._lang,
            parent=self,
        )
        if export_settings.exec() != QDialog.DialogCode.Accepted:
            return
        self._transcript_export_enabled = export_settings.auto_export_enabled
        self._transcript_export_include_timestamps = export_settings.include_timestamps
        self._transcript_export_include_speaker = export_settings.include_speaker
        self._transcript_export_default_format = export_settings.export_format
        self._transcript_export_formats = self._compose_export_formats(
            self._transcript_export_formats,
            self._transcript_export_default_format,
        )
        default_dir = str(getattr(self._config, "transcript_export_dir", "") or "").strip()
        if not default_dir:
            default_dir = str((Path(self._config.log_dir).resolve().parent / "exports"))
        fmt = self._transcript_export_default_format
        suffix = export_format_suffix(fmt)
        filters = self._export_save_filter(fmt)
        default_name = f"transcript{suffix}"
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Transcript",
            str(Path(default_dir) / default_name),
            filters,
            filters,
        )
        if not path:
            return
        if Path(path).suffix.lower() != suffix:
            path = str(Path(path).with_suffix(suffix))
        try:
            written = callback(
                path,
                fmt,
                self._transcript_export_include_timestamps,
                self._transcript_export_include_speaker,
            )
            QMessageBox.information(self, "Export", f"Transcript exported:\n{written}")
        except Exception as exc:
            QMessageBox.warning(self, "Export", f"Transcript export failed:\n{exc}")

    def _on_import_audio(self) -> None:
        callback = self._import_audio_callback
        if callback is None:
            QMessageBox.warning(self, self._t("invalid_settings"), "Import callback is unavailable in current runtime.")
            return
        default_dir = str(getattr(self._config, "source_file_path", "") or "").strip()
        if default_dir:
            default_dir = str(Path(default_dir).resolve().parent)
        else:
            default_dir = str(Path(self._config.log_dir).resolve().parent)
        filters = "Audio/Video (*.wav *.mp3 *.m4a *.aac *.flac *.ogg *.opus *.mp4 *.mkv *.webm);;All files (*.*)"
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Import Audio" if self._lang != "zh" else "匯入音檔",
            default_dir,
            filters,
        )
        if not path:
            return
        mode_box = QMessageBox(self)
        mode_box.setWindowTitle("Import Mode" if self._lang != "zh" else "匯入模式")
        mode_box.setText(
            "Choose how to process this imported audio file."
            if self._lang != "zh"
            else "選擇此匯入音檔的處理方式。"
        )
        direct_button = mode_box.addButton(
            "Direct (best quality, offline)" if self._lang != "zh" else "Direct（最佳品質，離線）",
            QMessageBox.ButtonRole.AcceptRole,
        )
        replay_button = mode_box.addButton(
            "Replay (realtime preview)" if self._lang != "zh" else "Replay（即時預覽）",
            QMessageBox.ButtonRole.ActionRole,
        )
        mode_box.addButton(QMessageBox.StandardButton.Cancel)
        mode_box.exec()
        clicked = mode_box.clickedButton()
        if clicked is None or mode_box.standardButton(clicked) == QMessageBox.StandardButton.Cancel:
            return
        mode = "direct" if clicked is direct_button else "replay"
        try:
            imported = callback(path, mode)
            if mode == "direct":
                started = "Imported audio direct transcription started:\n" if self._lang != "zh" else "已開始匯入音檔 Direct 轉寫：\n"
            else:
                started = "Imported audio replay started:\n" if self._lang != "zh" else "已開始匯入音檔重播：\n"
            QMessageBox.information(
                self,
                "Import" if self._lang != "zh" else "匯入",
                started + imported,
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Import" if self._lang != "zh" else "匯入",
                ("Audio import failed:\n" if self._lang != "zh" else "匯入音檔失敗：\n") + str(exc),
            )

    def _on_source_language_changed(self) -> None:
        self._refresh_alignment_model_suggestions()

    def _set_alignment_model_value(self, value: str) -> None:
        raw = (value or "").strip()
        if not raw:
            self._whisperx_align_model_edit.setCurrentText("")
            return
        idx = self._whisperx_align_model_edit.findText(raw)
        if idx >= 0:
            self._whisperx_align_model_edit.setCurrentIndex(idx)
        else:
            self._whisperx_align_model_edit.setEditText(raw)

    def _refresh_alignment_model_suggestions(self) -> None:
        current = self._whisperx_align_model_edit.currentText().strip()
        lang = str(self._whisperx_align_language_combo.currentData() or 'auto')
        if lang in {'auto','follow-source'}:
            lang = str(self._source_language_combo.currentData() or 'auto')
        repos = alignment_repos_for_language(lang)
        self._whisperx_align_model_edit.blockSignals(True)
        self._whisperx_align_model_edit.clear()
        self._whisperx_align_model_edit.addItem("")
        for repo in repos:
            self._whisperx_align_model_edit.addItem(repo)
        self._whisperx_align_model_edit.blockSignals(False)
        self._set_alignment_model_value(current)

    def _apply_stt_model_default(self, provider: str, *, previous_provider: str) -> None:
        if provider == previous_provider:
            return
        # Remember the outgoing provider's effective selection (preset size or custom value)
        # so switching back restores it instead of snapping to its hardcoded default.
        if previous_provider:
            self._stt_model_size_by_provider[previous_provider] = self._current_model_value()
        # Restore the incoming provider's remembered selection; _set_model_size routes it to
        # a preset item or the "Other…" custom field and toggles the custom row accordingly.
        remembered = self._stt_model_size_by_provider.get(provider, "").strip()
        self._set_model_size(remembered or default_stt_model(provider))

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

    def _on_align_guard_changed(self) -> None:
        self._update_align_guard_state()

    def _on_align_guard_revert(self) -> None:
        self._set_combo_data(self._whisperx_align_guard_combo, "safe")
        self._update_align_guard_state()

    def _update_align_guard_state(self) -> None:
        is_unsafe = str(self._whisperx_align_guard_combo.currentData() or "safe") == "unsafe-cuda"
        self._whisperx_align_guard_warning.setText(self._t("whisperx_align_guard_warning") if is_unsafe else "")
        self._whisperx_align_guard_warning.setVisible(is_unsafe)
        self._whisperx_align_guard_revert_btn.setText(self._t("whisperx_align_guard_revert"))
        self._whisperx_align_guard_revert_btn.setEnabled(is_unsafe)

    def _on_translation_toggle(self) -> None:
        enabled = self._translation_enabled_check.isChecked()
        self._bilingual_combo.setEnabled(enabled)
        self._translation_backend_combo.setEnabled(enabled)
        self._translation_backend_hint.setEnabled(enabled)
        self._translation_language_combo.setEnabled(enabled)
        self._translated_color_btn.setEnabled(enabled)

    def _apply_help_tooltips(self) -> None:
        tips = {
            self._mode_combo: "Choose audio source mode: loopback, microphone, or selected app sessions.",
            self._select_source_btn: "Open source picker to choose capture devices or app sessions.",
            self._stt_provider_combo: "Speech-to-text backend. WhisperX is default and is the only backend with forced alignment; whisper.cpp uses resident whisper-server for live Vulkan ASR and supports live diarization/speaker labels via its own module (no forced alignment).",
            self._stt_variant_combo: "Execution preference (Auto/CPU/GPU).",
            self._stt_model_path_edit: "Custom model: WhisperX accepts a model name (downloaded if absent) or a local path; whisper.cpp expects a local ggml file/dir.",
            self._stt_auto_download_check: "Auto-download missing models.",
            self._whisperx_vad_check: "WhisperX VAD engine: silero-vad (lighter) or pyannote (heavier).",
            self._whisperx_diarization_check: "Enable diarization (speaker separation, heavier).",
            self._whisperx_align_language_combo: "Alignment language: auto (ASR detected), follow-source (STT source language), or explicit language.",
            self._whisperx_align_device_combo: "Alignment device: auto (smart choice), cpu (lower VRAM), cuda (faster but higher VRAM).",
            self._whisperx_align_guard_combo: "Alignment CUDA safety guard. safe=downgrade CUDA alignment to CPU on Windows (default, avoids torchaudio/wav2vec2 crashes). unsafe-cuda=keep CUDA alignment (diagnostics only; may crash).",
            self._whisperx_align_guard_revert_btn: "Revert the alignment guard back to the safe default.",
            self._whisperx_diar_device_combo: "Diarization device: auto follows ASR device, cpu lowers VRAM pressure, cuda improves throughput.",
            self._whisperx_expected_speakers_spin: "Expected speaker count. 0=unknown/auto; positive values feed diarization and cap live speaker profiles.",
            self._whisperx_align_model_edit: "Alignment model. Empty=auto; choose suggestion or type HF repo id.",
            self._whisperx_diar_model_edit: "Diarization model id.",
            self._whisperx_hf_token_edit: "Hugging Face token for restricted/private models.",
            self._whisperx_speaker_backend_combo: "Speaker identity backend for profile embeddings.",
            self._segment_spin: "Segment window length sent to STT (seconds).",
            self._hop_spin: "Sliding hop size in seconds; smaller is lower latency but higher load.",
            self._merge_method_combo: "Merge strategy for overlapped subtitle segments.",
            self._preprocess_enabled_check: "Enable or disable pre-STT audio preprocessing pipeline.",
            self._source_language_combo: "Source language hint for STT input.",
            self._translation_enabled_check: "Enable live translation.",
            self._bilingual_combo: "Subtitle style: stacked bilingual or translation-only.",
            self._translation_backend_combo: "Translation backend: Argos is lighter; NLLB is offline multilingual and CPU-first.",
            self._translation_language_combo: "Translation target language.",
            self._font_size_spin: "Subtitle font size.",
            self._opacity_slider: "Overlay opacity.",
            self._source_color_btn: "Source text color.",
            self._translated_color_btn: "Translated text color.",
            self._bg_color_btn: "Overlay background color.",
            self._debug_mode_check: "Enable or disable Debug trace window and debug log output.",
            self._transcript_export_now_btn: "Open export subtitle settings and then choose file path.",
            self._import_audio_btn: "Import an audio/video file and replay it through the realtime subtitle pipeline.",
            self._reset_defaults_btn: "Reset all settings in this dialog back to default values.",
            self._compute_type_combo: "ASR compute type. float16 preserves current default; int8_float16/int8 can reduce load with possible accuracy cost.",
            self._whisperx_zh_align_wbbbbb_check: "Chinese alignment on GPU via wbbbbb/wav2vec2-large-chinese (~10x faster alignment; slightly worse CER than the CPU default). Needs alignment device=cuda and an empty explicit alignment model.",
            self._asr_temperatures_edit: "Temperature-fallback schedule for hard windows. Default 0.0,0.2,0.4 halves worst-case window latency with identical output in A/B; clear this field to restore the full 6-step library schedule.",
            self._commit_hold_spin: "Delay speaker-label lock-in for committed subtitle text by this many seconds so labels can settle/back-date (text still appears immediately). 0=off (legacy).",
        }
        for (widget, tip) in tips.items():
            try:
                widget.setToolTip(tip)
            except Exception:
                pass

    def _reset_to_defaults(self) -> None:
        defaults = self._default_config
        self._selected_loopback_indices = self._loopback_indices_from_config(defaults)
        self._selected_app_names = self._app_names_from_config(defaults)
        self._sync_from_config(defaults)
        self._last_stt_provider = str(self._stt_provider_combo.currentData() or "whisperx").strip() or "whisperx"
        self._on_mode_changed()
        self._on_stt_provider_changed()
        self._on_translation_toggle()


    def _collect_updates(self) -> dict[str, object]:
        """Collect validated settings payload applied by bootstrap.apply_settings()."""
        model_size_value, model_path_value = self._collect_model_size_and_path()
        payload = SettingsPayloadInput(
            ui_language=str(self._ui_language_combo.currentData() or self._lang),
            source_mode=str(self._mode_combo.currentData()),
            stt_provider=str(self._stt_provider_combo.currentData() or "whisperx"),
            runtime_preset=str(self._preset_combo.currentData() or ""),
            stt_variant=str(self._stt_variant_combo.currentData() or "auto"),
            model_size=model_size_value,
            whisper_beam_size=int(self._beam_spin.value()),
            whisperx_speaker_profile_enabled=self._whisperx_speaker_profile_check.isChecked(),
            compute_type=str(self._compute_type_combo.currentData() or "float16"),
            stt_model_path=model_path_value,
            stt_auto_download=self._stt_auto_download_check.isChecked(),
            whisperx_enable_phoneme_asr=True,
            # Forced alignment has no dedicated widget; derive it from the active preset so the `cpu`
            # preset (alignment off for non-CUDA realtime) actually persists. Defaults to on otherwise.
            whisperx_enable_forced_alignment=self._preset_forced_alignment(),
            whisperx_enable_vad=True,
            whisperx_vad_method=str(self._whisperx_vad_check.currentData() or "silero-vad"),
            whisperx_enable_diarization=self._whisperx_diarization_check.isChecked(),
            whisperx_alignment_model=self._whisperx_align_model_edit.currentText(),
            whisperx_alignment_language=str(self._whisperx_align_language_combo.currentData() or 'auto'),
            whisperx_alignment_device=str(self._whisperx_align_device_combo.currentData() or 'auto'),
            whisperx_align_guard=str(self._whisperx_align_guard_combo.currentData() or 'safe'),
            whisperx_diarization_device=str(self._whisperx_diar_device_combo.currentData() or 'auto'),
            whisperx_diarization_model=self._whisperx_diar_model_edit.text(),
            whisperx_diarization_expected_speakers=int(self._whisperx_expected_speakers_spin.value()),
            whisperx_hf_token=self._whisperx_hf_token_edit.text(),
            whisperx_speaker_profile_backend=str(self._whisperx_speaker_backend_combo.currentData() or "pyannote"),
            source_language=str(self._source_language_combo.currentData()),
            translation_to=str(self._translation_language_combo.currentData()),
            segment_seconds=float(self._segment_spin.value()),
            hop_seconds=float(self._hop_spin.value()),
            selected_loopback_indices=list(self._selected_loopback_indices),
            selected_app_names=list(self._selected_app_names),
            overlap_merge_method=str(self._merge_method_combo.currentData()),
            preprocess_enabled=self._preprocess_enabled_check.isChecked(),
            preprocess_modules="auto",
            translation_enabled=self._translation_enabled_check.isChecked(),
            translation_backend=str(self._translation_backend_combo.currentData() or "argos"),
            bilingual_style=str(self._bilingual_combo.currentData()),
            font_size=int(self._font_size_spin.value()),
            overlay_opacity=float(self._opacity_slider.value()) / 100.0,
            source_text_color=self._source_color_btn.text(),
            translated_text_color=self._translated_color_btn.text(),
            background_color=self._bg_color_btn.text(),
            debug_mode=self._debug_mode_check.isChecked(),
            transcript_export_enabled=bool(self._transcript_export_enabled),
            transcript_export_formats=self._transcript_export_formats,
            transcript_export_include_timestamps=bool(self._transcript_export_include_timestamps),
            transcript_export_include_speaker=bool(self._transcript_export_include_speaker),
            whisperx_zh_align_wbbbbb=self._whisperx_zh_align_wbbbbb_check.isChecked(),
            whisperx_asr_temperatures=self._asr_temperatures_edit.text(),
            subtitle_commit_hold_seconds=float(self._commit_hold_spin.value()),
            asr_temperatures_invalid_message=self._t("asr_temperatures_invalid"),
        )
        return build_settings_updates(
            payload,
            lang=str(self._ui_language_combo.currentData() or self._lang),
            hop_gt_segment_message=self._t("hop_gt_segment"),
        )

    @staticmethod
    def _export_save_filter(export_format: str) -> str:
        fmt = str(export_format or "").strip().lower()
        if fmt == "srt":
            return "SubRip subtitle (*.srt)"
        if fmt == "json":
            return "JSON transcript (*.json)"
        return "Text (*.txt)"

    @staticmethod
    def _resolve_default_export_format(raw_formats: str) -> str:
        items: list[str] = []
        for token in str(raw_formats or "").split(","):
            fmt = token.strip().lower()
            if fmt in {"txt", "srt", "json"} and fmt not in items:
                items.append(fmt)
        return items[0] if items else "txt"

    @staticmethod
    def _compose_export_formats(raw_formats: str, preferred: str) -> str:
        items: list[str] = []
        pref = preferred.strip().lower()
        if pref in {"txt", "srt", "json"}:
            items.append(pref)
        for token in str(raw_formats or "").split(","):
            fmt = token.strip().lower()
            if fmt in {"txt", "srt", "json"} and fmt not in items:
                items.append(fmt)
        if not items:
            items = ["txt", "srt", "json"]
        return ",".join(items)

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
        for form_layout in self._form_layouts:
            label = form_layout.labelForField(field)
            if label is not None:
                label.setVisible(visible)
        field.setVisible(visible)

    @staticmethod
    def _set_combo_data(combo: QComboBox, value: str) -> None:
        idx = combo.findData(value)
        combo.setCurrentIndex(max(0, idx))









