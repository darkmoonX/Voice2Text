from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from voice2text.config import RuntimeConfig
from voice2text.settings_persistence import (
    _PERSIST_KEYS,
    apply_updates_to_config,
    load_persisted_updates,
    save_runtime_settings,
)


class PersistenceAllowlistTests(unittest.TestCase):
    def test_preset_keys_persisted(self) -> None:
        # A GUI-applied preset must survive restart: model_size + runtime_preset
        # are now persisted (beam / speaker-profile / forced-align already were).
        for key in ("model_size", "runtime_preset", "whisper_beam_size", "whisperx_speaker_profile_enabled"):
            self.assertIn(key, _PERSIST_KEYS, key)

    def test_diarization_speaker_hint_keys_persisted(self) -> None:
        self.assertIn("whisperx_diarization_min_speakers", _PERSIST_KEYS)
        self.assertIn("whisperx_diarization_max_speakers", _PERSIST_KEYS)
        self.assertIn("whisperx_speaker_count_hint_enabled", _PERSIST_KEYS)
        self.assertIn("whisperx_speaker_count_hint_seconds", _PERSIST_KEYS)
        self.assertIn("whisperx_speaker_count_hint_window_seconds", _PERSIST_KEYS)
        self.assertIn("whisperx_speaker_count_hint_sliver_floor_seconds", _PERSIST_KEYS)
        self.assertIn("whisperx_speaker_merge_grace_windows", _PERSIST_KEYS)
        self.assertIn("whisperx_speaker_merge_grace_relief", _PERSIST_KEYS)
        self.assertIn("whisperx_speaker_merge_preserve_centroid", _PERSIST_KEYS)
        self.assertIn("whisperx_speaker_profile_max_exemplars", _PERSIST_KEYS)
        self.assertIn("whisperx_speaker_profile_exemplar_diversity_threshold", _PERSIST_KEYS)

    def test_speaker_merge_grace_persistence_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runtime_settings.json"
            cfg = RuntimeConfig(
                whisperx_speaker_merge_grace_windows=30,
                whisperx_speaker_merge_grace_relief=0.15,
            )

            save_runtime_settings(cfg, path=path)
            updates = load_persisted_updates(path=path)
            restored = RuntimeConfig()
            changed = apply_updates_to_config(restored, updates)

            self.assertIn("whisperx_speaker_merge_grace_windows", changed)
            self.assertIn("whisperx_speaker_merge_grace_relief", changed)
            self.assertEqual(restored.whisperx_speaker_merge_grace_windows, 30)
            self.assertEqual(restored.whisperx_speaker_merge_grace_relief, 0.15)

    def test_speaker_merge_preserve_centroid_persistence_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runtime_settings.json"
            cfg = RuntimeConfig(whisperx_speaker_merge_preserve_centroid=True)

            save_runtime_settings(cfg, path=path)
            updates = load_persisted_updates(path=path)
            restored = RuntimeConfig()
            changed = apply_updates_to_config(restored, updates)

            self.assertIn("whisperx_speaker_merge_preserve_centroid", changed)
            self.assertTrue(restored.whisperx_speaker_merge_preserve_centroid)

    def test_speaker_profile_max_exemplars_persistence_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runtime_settings.json"
            cfg = RuntimeConfig(
                whisperx_speaker_profile_max_exemplars=4,
                whisperx_speaker_profile_exemplar_diversity_threshold=0.85,
            )

            save_runtime_settings(cfg, path=path)
            updates = load_persisted_updates(path=path)
            restored = RuntimeConfig()
            changed = apply_updates_to_config(restored, updates)

            self.assertIn("whisperx_speaker_profile_max_exemplars", changed)
            self.assertIn("whisperx_speaker_profile_exemplar_diversity_threshold", changed)
            self.assertEqual(restored.whisperx_speaker_profile_max_exemplars, 4)
            self.assertEqual(restored.whisperx_speaker_profile_exemplar_diversity_threshold, 0.85)


try:
    from PySide6.QtWidgets import QApplication

    _APP = QApplication.instance() or QApplication([])
    _QT_OK = True
except Exception:  # pragma: no cover - headless without offscreen plugin
    _QT_OK = False


@unittest.skipUnless(_QT_OK, "PySide6 / offscreen Qt platform unavailable")
class PresetDropdownDialogTests(unittest.TestCase):
    def _dialog(self):
        from voice2text.settings_dialog import SettingsDialog

        return SettingsDialog(RuntimeConfig(), devices=[], app_sessions=[])

    def test_high_accuracy_preset_fills_widgets(self) -> None:
        dlg = self._dialog()
        dlg._preset_combo.setCurrentIndex(dlg._preset_combo.findData("high-accuracy"))
        self.assertEqual(dlg._model_size_combo.currentText(), "large-v3")
        self.assertEqual(dlg._beam_spin.value(), 5)
        self.assertEqual(float(dlg._segment_spin.value()), 10.0)
        self.assertEqual(float(dlg._hop_spin.value()), 2.0)
        self.assertTrue(dlg._whisperx_diarization_check.isChecked())
        self.assertTrue(dlg._whisperx_speaker_profile_check.isChecked())

        updates = dlg._collect_updates()
        self.assertEqual(updates["model_size"], "large-v3")
        self.assertEqual(updates["runtime_preset"], "high-accuracy")
        self.assertEqual(updates["whisper_beam_size"], 5)
        self.assertTrue(updates["whisperx_enable_diarization"])
        self.assertTrue(updates["whisperx_speaker_profile_enabled"])
        # Safety: forced alignment is never user-disablable (rolling merge needs it).
        self.assertTrue(updates["whisperx_enable_forced_alignment"])

    def test_balanced_preset_fills_widgets(self) -> None:
        dlg = self._dialog()
        dlg._preset_combo.setCurrentIndex(dlg._preset_combo.findData("balanced"))
        self.assertEqual(dlg._model_size_combo.currentText(), "medium")
        self.assertFalse(dlg._whisperx_diarization_check.isChecked())
        updates = dlg._collect_updates()
        self.assertEqual(updates["model_size"], "medium")
        self.assertEqual(updates["runtime_preset"], "balanced")
        self.assertFalse(updates["whisperx_enable_diarization"])

    def test_manual_edit_clears_preset_label(self) -> None:
        dlg = self._dialog()
        dlg._preset_combo.setCurrentIndex(dlg._preset_combo.findData("high-accuracy"))
        self.assertEqual(dlg._preset_combo.currentData(), "high-accuracy")
        # a manual edit to a bundled field means it is no longer a named preset
        dlg._beam_spin.setValue(3)
        self.assertEqual(dlg._preset_combo.currentData(), "")
        updates = dlg._collect_updates()
        self.assertEqual(updates["whisper_beam_size"], 3)
        self.assertEqual(updates["runtime_preset"], "")
        # the explicit model_size from the preset still stands (override semantics)
        self.assertEqual(updates["model_size"], "large-v3")

    def test_sync_from_config_reflects_preset_and_model(self) -> None:
        from voice2text.settings_dialog import SettingsDialog

        cfg = RuntimeConfig()
        cfg.runtime_preset = "high-accuracy"
        cfg.model_size = "large-v2"
        cfg.whisper_beam_size = 5
        dlg = SettingsDialog(cfg, devices=[], app_sessions=[])
        self.assertEqual(dlg._preset_combo.currentData(), "high-accuracy")
        self.assertEqual(dlg._model_size_combo.currentText(), "large-v2")

    def test_custom_model_size_routes_to_other_field(self) -> None:
        # A non-preset alias now lands in the "Other…" custom field, not an editable combo.
        from voice2text.settings_dialog import SettingsDialog

        cfg = RuntimeConfig()
        cfg.model_size = "large-v3-turbo"  # not in the preset list -> "Other…" + custom field
        dlg = SettingsDialog(cfg, devices=[], app_sessions=[])
        self.assertEqual(dlg._model_size_combo.currentData(), SettingsDialog.CUSTOM_MODEL_DATA)
        self.assertEqual(dlg._stt_model_path_edit.text(), "large-v3-turbo")
        self.assertTrue(dlg._stt_model_path_edit.isVisibleTo(dlg))
        updates = dlg._collect_updates()
        # custom value rides in the path field; model_size stays a real preset fallback
        self.assertEqual(updates["stt_model_path"], "large-v3-turbo")
        self.assertNotEqual(updates["model_size"], "large-v3-turbo")

    def test_translation_backend_nllb_is_collected(self) -> None:
        from voice2text.settings_dialog import SettingsDialog

        cfg = RuntimeConfig()
        cfg.translation_enabled = True
        cfg.translation_backend = "nllb"
        dlg = SettingsDialog(cfg, devices=[], app_sessions=[])
        self.assertEqual(dlg._translation_backend_combo.currentData(), "nllb")
        updates = dlg._collect_updates()
        self.assertEqual(updates["translation_backend"], "nllb")

    def test_whispercpp_provider_collects_size_without_alias_as_path(self) -> None:
        from voice2text.settings_dialog import SettingsDialog

        dlg = SettingsDialog(RuntimeConfig(), devices=[], app_sessions=[])
        dlg._set_combo_data(dlg._stt_provider_combo, "whispercpp")
        dlg._on_stt_provider_changed()
        updates = dlg._collect_updates()
        self.assertEqual(updates["stt_provider"], "whispercpp")
        self.assertEqual(updates["stt_whispercpp_model_size"], "medium")
        self.assertEqual(updates["stt_whispercpp_model_path"], "")

    def test_whisperx_model_dropdown_drives_model_without_path_typing(self) -> None:
        # Regression: for WhisperX the size dropdown must be the effective model selector.
        # Previously the model name lived in the path field, so changing the dropdown was
        # silently overridden by `stt_model_path or model_size`.
        from voice2text.settings_dialog import SettingsDialog

        dlg = SettingsDialog(RuntimeConfig(), devices=[], app_sessions=[])
        dlg._set_model_size("large-v2")
        updates = dlg._collect_updates()
        self.assertEqual(updates["model_size"], "large-v2")
        # path field stays empty -> dropdown wins at runtime (factory: path or size)
        self.assertEqual(updates["stt_model_path"], "")

    def test_whisperx_legacy_alias_path_migrates_into_dropdown(self) -> None:
        # A pre-fix runtime_settings.json persisted a bare alias in stt_model_path.
        # On load it must move into the dropdown and clear the path, so the dropdown
        # is no longer shadowed by the stale alias.
        from voice2text.settings_dialog import SettingsDialog

        cfg = RuntimeConfig()
        cfg.stt_provider = "whisperx"
        cfg.model_size = "small"
        cfg.stt_model_path = "large-v2"  # legacy: bare alias, not a path
        dlg = SettingsDialog(cfg, devices=[], app_sessions=[])
        self.assertEqual(dlg._model_size_combo.currentText(), "large-v2")
        self.assertEqual(dlg._stt_model_path_edit.text(), "")
        updates = dlg._collect_updates()
        self.assertEqual(updates["model_size"], "large-v2")
        self.assertEqual(updates["stt_model_path"], "")

    def test_other_entry_reveals_custom_field_and_preset_hides_it(self) -> None:
        # Selecting "Other…" shows the single custom field; a preset hides it again, so
        # size and custom path are never both visible (no two-paths priority confusion).
        from voice2text.settings_dialog import SettingsDialog

        dlg = SettingsDialog(RuntimeConfig(), devices=[], app_sessions=[])
        # default preset (small) -> custom row hidden
        self.assertFalse(dlg._stt_model_path_edit.isVisibleTo(dlg))

        custom_idx = dlg._model_size_combo.findData(SettingsDialog.CUSTOM_MODEL_DATA)
        self.assertGreaterEqual(custom_idx, 0)
        dlg._model_size_combo.setCurrentIndex(custom_idx)
        self.assertTrue(dlg._stt_model_path_edit.isVisibleTo(dlg))
        dlg._stt_model_path_edit.setText("D:/models/custom")
        updates = dlg._collect_updates()
        self.assertEqual(updates["stt_model_path"], "D:/models/custom")

        # back to a preset -> custom row hidden and path cleared from the payload
        dlg._set_model_size("medium")
        self.assertFalse(dlg._stt_model_path_edit.isVisibleTo(dlg))
        updates = dlg._collect_updates()
        self.assertEqual(updates["model_size"], "medium")
        self.assertEqual(updates["stt_model_path"], "")

    def test_provider_switch_restores_previous_whisperx_size(self) -> None:
        # Regression: switching WhisperX -> whispercpp -> WhisperX must restore the
        # user's WhisperX size (large-v2), not snap back to the default 'small'.
        from voice2text.settings_dialog import SettingsDialog

        cfg = RuntimeConfig()
        cfg.stt_provider = "whisperx"
        cfg.model_size = "large-v2"
        dlg = SettingsDialog(cfg, devices=[], app_sessions=[])
        self.assertEqual(dlg._model_size_combo.currentText(), "large-v2")

        # switch to whispercpp -> its own default (medium)
        dlg._set_combo_data(dlg._stt_provider_combo, "whispercpp")
        dlg._on_stt_provider_changed()
        self.assertEqual(dlg._model_size_combo.currentText(), "medium")

        # switch back to whisperx -> remembered large-v2, NOT small
        dlg._set_combo_data(dlg._stt_provider_combo, "whisperx")
        dlg._on_stt_provider_changed()
        self.assertEqual(dlg._model_size_combo.currentText(), "large-v2")
        updates = dlg._collect_updates()
        self.assertEqual(updates["model_size"], "large-v2")

    def test_whisperx_local_path_override_is_preserved(self) -> None:
        # A genuine local-file override (path-like) must stay in the path field and win.
        from voice2text.settings_dialog import SettingsDialog

        cfg = RuntimeConfig()
        cfg.stt_provider = "whisperx"
        cfg.stt_model_path = "D:/models/my-whisper"  # path-like override
        dlg = SettingsDialog(cfg, devices=[], app_sessions=[])
        self.assertEqual(dlg._stt_model_path_edit.text(), "D:/models/my-whisper")
        updates = dlg._collect_updates()
        self.assertEqual(updates["stt_model_path"], "D:/models/my-whisper")


if __name__ == "__main__":
    unittest.main()
