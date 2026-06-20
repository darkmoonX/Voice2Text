"""Lightweight resolver/downloader for the whisper.cpp subprocess backend."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from ..config import RuntimeConfig
from ..model_paths import library_model_dir
from .model_download import download_hf_files_with_progress, emit_progress
from .registry import normalize_stt_variant

WHISPERCPP_REPO_ID = "ggerganov/whisper.cpp"
WHISPERCPP_VAD_REPO_ID = "ggml-org/whisper-vad"
DEFAULT_VAD_MODEL_FILENAME = "ggml-silero-v5.1.2.bin"
_VALID_MODEL_SIZES = {
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v1",
    "large-v2",
    "large-v3",
    "large-v3-turbo",
}


def default_whispercpp_binary_path() -> Path:
    return Path(__file__).resolve().parents[2] / "runtime_bin" / "whispercpp" / "whisper-cli.exe"


def default_whispercpp_server_path() -> Path:
    return Path(__file__).resolve().parents[2] / "runtime_bin" / "whispercpp" / "whisper-server.exe"


def whispercpp_model_dir() -> Path:
    return library_model_dir("whispercpp")


def normalize_whispercpp_model_size(value: str) -> str:
    token = str(value or "").strip().lower()
    if not token:
        return "medium"
    if token.startswith("ggml-") and token.endswith(".bin"):
        token = token[5:-4]
    elif token.endswith(".bin"):
        token = token[:-4]
    return token if token in _VALID_MODEL_SIZES else token


def ggml_model_filename(model_size: str) -> str:
    return f"ggml-{normalize_whispercpp_model_size(model_size)}.bin"


def whispercpp_vad_model_filename(value: str) -> str:
    token = str(value or "").strip()
    return token if token else DEFAULT_VAD_MODEL_FILENAME


def resolve_whispercpp_binary(config: RuntimeConfig) -> Path:
    override = str(os.environ.get("VOICE2TEXT_WHISPERCPP_BIN", "") or "").strip()
    if not override:
        override = str(getattr(config, "stt_whispercpp_binary_path", "") or "").strip()
    candidate = Path(override).expanduser() if override else default_whispercpp_binary_path()
    if not candidate.exists():
        raise RuntimeError(
            "whisper.cpp backend binary not found: "
            f"{candidate}. Build it with app/build_whispercpp.ps1 or set VOICE2TEXT_WHISPERCPP_BIN."
        )
    return candidate


def resolve_whispercpp_server_binary(config: RuntimeConfig) -> Path:
    override = str(os.environ.get("VOICE2TEXT_WHISPERCPP_SERVER_BIN", "") or "").strip()
    if not override:
        override = str(getattr(config, "stt_whispercpp_server_path", "") or "").strip()
    candidate = Path(override).expanduser() if override else default_whispercpp_server_path()
    if not candidate.exists():
        raise RuntimeError(
            "whisper.cpp server binary not found: "
            f"{candidate}. Build it with app/build_whispercpp.ps1 or set VOICE2TEXT_WHISPERCPP_SERVER_BIN."
        )
    return candidate


def resolve_whispercpp_vad_model(
    config: RuntimeConfig,
    *,
    progress_callback: Callable[[str], None] | None = None,
    allow_download: bool = True,
) -> Path:
    override = str(os.environ.get("VOICE2TEXT_WHISPERCPP_VAD_MODEL", "") or "").strip()
    if not override:
        override = str(getattr(config, "stt_whispercpp_vad_model_path", "") or "").strip()
    filename = whispercpp_vad_model_filename(str(getattr(config, "stt_whispercpp_vad_model", "") or ""))
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_dir():
            candidate = candidate / filename
        if not candidate.exists():
            raise RuntimeError(f"whisper.cpp VAD model not found: {candidate}")
        return candidate

    model_dir = whispercpp_model_dir()
    candidate = model_dir / filename
    if candidate.exists():
        return candidate
    matches = sorted(model_dir.glob("ggml-silero-*.bin"))
    if matches:
        return matches[0]
    if (not allow_download) or (not bool(getattr(config, "stt_auto_download", True))):
        raise RuntimeError(
            "whisper.cpp VAD model is missing and auto-download is disabled: "
            f"{candidate}. Enable stt_auto_download, set stt_whispercpp_vad_model_path, "
            "or disable whisper.cpp server VAD."
        )
    emit_progress(progress_callback, f"[download] whispercpp preparing VAD model: {filename}")
    download_hf_files_with_progress(
        repo_id=WHISPERCPP_VAD_REPO_ID,
        output_dir=str(model_dir),
        allow_patterns=[filename],
        progress_callback=progress_callback,
        provider="whispercpp",
        model_name=f"vad-{filename}",
    )
    if not candidate.exists():
        raise RuntimeError(f"whisper.cpp VAD model download did not produce expected file: {candidate}")
    return candidate


def resolve_whispercpp_model(
    config: RuntimeConfig,
    *,
    progress_callback: Callable[[str], None] | None = None,
) -> Path:
    explicit = str(getattr(config, "stt_whispercpp_model_path", "") or "").strip()
    if not explicit:
        explicit = str(getattr(config, "stt_model_path", "") or "").strip()
    if explicit:
        candidate = Path(explicit).expanduser()
        if not candidate.exists():
            raise RuntimeError(f"whisper.cpp ggml model not found: {candidate}")
        if candidate.is_dir():
            size = str(getattr(config, "stt_whispercpp_model_size", "") or getattr(config, "model_size", "medium"))
            candidate = candidate / ggml_model_filename(size)
            if not candidate.exists():
                raise RuntimeError(f"whisper.cpp ggml model not found: {candidate}")
        return candidate

    size = str(getattr(config, "stt_whispercpp_model_size", "") or getattr(config, "model_size", "medium"))
    filename = ggml_model_filename(size)
    model_dir = whispercpp_model_dir()
    candidate = model_dir / filename
    if candidate.exists():
        return candidate
    if not bool(getattr(config, "stt_auto_download", True)):
        raise RuntimeError(
            "whisper.cpp ggml model is missing and auto-download is disabled: "
            f"{candidate}. Enable stt_auto_download or set stt_whispercpp_model_path."
        )
    emit_progress(progress_callback, f"[download] whispercpp preparing ggml model: {filename}")
    download_hf_files_with_progress(
        repo_id=WHISPERCPP_REPO_ID,
        output_dir=str(model_dir),
        allow_patterns=[filename],
        progress_callback=progress_callback,
        provider="whispercpp",
        model_name=filename,
    )
    if not candidate.exists():
        raise RuntimeError(f"whisper.cpp ggml model download did not produce expected file: {candidate}")
    return candidate


def resolve_whispercpp_device(config: RuntimeConfig, *, device_override: str | None = None) -> str:
    raw = str(device_override or "").strip().lower()
    if raw.startswith("cpu"):
        return "cpu"
    if raw in {"cuda", "gpu", "vulkan"}:
        return "vulkan"
    variant = normalize_stt_variant(str(getattr(config, "stt_variant", "auto") or "auto"))
    return "cpu" if variant == "cpu" else "vulkan"
