"""Widget builders for settings dialog controls."""
from __future__ import annotations

from PySide6.QtWidgets import QComboBox


def create_mode_combo() -> QComboBox:
    combo = QComboBox()
    combo.addItem("Loopback", "loopback")
    combo.addItem("Microphone", "microphone")
    combo.addItem("App Session", "app")
    return combo


def create_stt_provider_combo() -> QComboBox:
    combo = QComboBox()
    combo.addItem("WhisperX", "whisperx")
    combo.addItem("whisper.cpp Vulkan", "whispercpp")
    return combo


def create_stt_variant_combo() -> QComboBox:
    combo = QComboBox()
    combo.addItem("Auto", "auto")
    combo.addItem("CPU", "cpu")
    combo.addItem("GPU", "gpu")
    return combo


def create_compute_type_combo() -> QComboBox:
    combo = QComboBox()
    combo.addItem("float16", "float16")
    combo.addItem("int8_float16", "int8_float16")
    combo.addItem("int8", "int8")
    return combo


def create_source_language_combo() -> QComboBox:
    combo = QComboBox()
    combo.addItem("auto", "auto")
    combo.addItem("en", "en")
    combo.addItem("zh-hant", "zh-hant")
    combo.addItem("zh-hans", "zh-hans")
    combo.addItem("ja", "ja")
    combo.addItem("ko", "ko")
    return combo


def create_translation_language_combo() -> QComboBox:
    combo = QComboBox()
    combo.addItem("en", "en")
    combo.addItem("zh", "zh")
    combo.addItem("ja", "ja")
    combo.addItem("ko", "ko")
    return combo


def create_translation_backend_combo() -> QComboBox:
    combo = QComboBox()
    combo.addItem("Argos", "argos")
    combo.addItem("NLLB (offline, CPU)", "nllb")
    return combo


def create_whisperx_align_language_combo() -> QComboBox:
    combo = QComboBox()
    combo.addItem("auto", "auto")
    combo.addItem("follow-source", "follow-source")
    combo.addItem("en", "en")
    combo.addItem("zh-hant", "zh-hant")
    combo.addItem("zh-hans", "zh-hans")
    combo.addItem("ja", "ja")
    combo.addItem("ko", "ko")
    combo.addItem("de", "de")
    combo.addItem("fr", "fr")
    combo.addItem("es", "es")
    combo.addItem("it", "it")
    combo.addItem("pt", "pt")
    combo.addItem("ru", "ru")
    return combo


def create_whisperx_align_device_combo() -> QComboBox:
    combo = QComboBox()
    combo.addItem("auto", "auto")
    combo.addItem("cpu", "cpu")
    combo.addItem("cuda", "cuda")
    return combo


def create_whisperx_diarization_device_combo() -> QComboBox:
    combo = QComboBox()
    combo.addItem("auto", "auto")
    combo.addItem("cpu", "cpu")
    combo.addItem("cuda", "cuda")
    return combo


def create_whisperx_align_guard_combo() -> QComboBox:
    combo = QComboBox()
    combo.addItem("safe", "safe")
    combo.addItem("unsafe-cuda", "unsafe-cuda")
    return combo


def create_whisperx_speaker_backend_combo() -> QComboBox:
    combo = QComboBox()
    combo.addItem("pyannote", "pyannote")
    combo.addItem("speechbrain-ecapa", "speechbrain_ecapa")
    combo.addItem("nemo-titanet", "nemo_titanet")
    return combo
