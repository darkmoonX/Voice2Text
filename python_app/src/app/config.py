from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RuntimeConfig:
    model_size: str = "small"
    model_device: str = "cuda"
    compute_type: str = "float16"
    cpu_fallback_on_cuda_error: bool = True
    cuda_compat_source_dll: str = r"D:\CUDA\bin\x64\cublas64_13.dll"

    segment_seconds: float = 6.0
    hop_seconds: float = 1.5

    source_language: Optional[str] = None
    source_mode: str = "loopback"  # loopback | microphone | app
    source_device_indices: list[int] = field(default_factory=list)
    source_mix_weights: list[float] = field(default_factory=list)
    source_app_name: str = ""
    source_app_names: list[str] = field(default_factory=list)
    source_channel_mode: str = "mono"  # mono | left | right
    overlap_merge_method: str = "replace-window"  # replace-window | suffix-overlap | fuzzy-overlap | append-only

    whisper_max_context: Optional[int] = None
    whisper_entropy_thold: Optional[float] = None
    whisper_logprob_thold: Optional[float] = None
    whisper_no_speech_thold: Optional[float] = None
    whisper_temperature: Optional[float] = None
    whisper_beam_size: Optional[int] = None
    whisper_best_of: Optional[int] = None

    max_lines: int = 10
    overlay_width: int = 1200
    overlay_height: int = 320
    overlay_x: int = 40
    overlay_y: int = 700
    overlay_opacity: float = 0.8
    font_size: int = 18
    text_color: str = "#F0F2F5"
    source_text_color: str = "#F0F2F5"
    translated_text_color: str = "#FFD98A"
    status_color: str = "#78D7FF"
    error_color: str = "#FF7878"
    background_color: str = "#0A101A"

    translation_enabled: bool = False
    translation_from: str = "auto"
    translation_to: str = "zh"
    bilingual_style: str = "stacked"  # stacked | inline | translation-only

    device_index: Optional[int] = None
    log_dir: str = "logs"