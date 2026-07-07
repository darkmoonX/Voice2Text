"""Foreground WhisperX diarization stability probe with visible progress output."""
from __future__ import annotations

import argparse
import os
import json
import sys
import time
import wave
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = APP_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice2text.audio_capture import AudioChunk
from voice2text.config import RuntimeConfig
from voice2text.stt.factory import create_stt_transcriber


def _load_chunk(path: Path) -> AudioChunk:
    with wave.open(str(path), "rb") as wav_file:
        sample_rate = int(wav_file.getframerate())
        channels = int(wav_file.getnchannels())
        pcm16 = wav_file.readframes(wav_file.getnframes())
    return AudioChunk(pcm16=pcm16, sample_rate=sample_rate, channels=channels)


def _build_cfg(model: str, language: str) -> RuntimeConfig:
    cfg = RuntimeConfig()
    cfg.stt_provider = "whisperx"
    cfg.stt_variant = "gpu"
    cfg.model_device = "cuda"
    cfg.compute_type = "float16"
    cfg.stt_auto_download = True
    cfg.stt_model_path = model
    cfg.source_language = language
    cfg.whisperx_enable_forced_alignment = True
    cfg.whisperx_enable_diarization = True
    cfg.whisperx_vad_method = "silero-vad"
    cfg.whisperx_alignment_language = "follow-source"
    cfg.whisperx_alignment_model = "WAV2VEC2_ASR_BASE_960H"
    cfg.whisperx_alignment_device = "cuda"
    cfg.debug_mode = True
    return cfg


def _resolve_token(arg_token: str) -> str:
    if arg_token.strip():
        return arg_token.strip()
    settings_file = SRC_ROOT / "runtime_settings.json"
    if settings_file.exists():
        try:
            payload = json.loads(settings_file.read_text(encoding="utf-8"))
            token = str(payload.get("whisperx_hf_token") or "").strip()
            if token:
                return token
        except Exception:
            pass
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        value = str(os.environ.get(key, "") or "").strip()
        if value:
            return value
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="WhisperX diarization stability test.")
    parser.add_argument(
        "--source-wav",
        default=str(SRC_ROOT / "segments" / "latest_segment_stt.wav"),
        help="Source wav used for repeated inference.",
    )
    parser.add_argument("--duration-seconds", type=int, default=70)
    parser.add_argument("--model", default="medium")
    parser.add_argument("--language", default="en")
    parser.add_argument("--hf-token", default="", help="HF token override; falls back to runtime_settings/env.")
    parser.add_argument("--clear-proxy", action="store_true", help="Clear proxy env vars for this process.")
    args = parser.parse_args()

    if args.clear_proxy:
        for key in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
            "GIT_HTTP_PROXY",
            "GIT_HTTPS_PROXY",
        ):
            os.environ[key] = ""

    source_wav = Path(args.source_wav).resolve()
    if not source_wav.exists():
        print(f"[error] source wav missing: {source_wav}", flush=True)
        return 2

    chunk = _load_chunk(source_wav)
    cfg = _build_cfg(args.model, args.language)
    cfg.whisperx_hf_token = _resolve_token(args.hf_token)
    print(
        f"[start] wav={source_wav} sr={chunk.sample_rate} ch={chunk.channels} bytes={len(chunk.pcm16)} "
        f"duration={args.duration_seconds}s hf_token={'set' if cfg.whisperx_hf_token else 'empty'}",
        flush=True,
    )

    def progress(msg: str) -> None:
        print(f"[progress] {msg}", flush=True)

    transcriber = create_stt_transcriber(cfg, progress_callback=progress)
    resolved_align_device = str(getattr(transcriber, "_alignment_device", "unknown"))
    print(f"[info] resolved_alignment_device={resolved_align_device}", flush=True)

    deadline = time.monotonic() + float(max(1, args.duration_seconds))
    iteration = 0
    has_arrow_once = False
    errors = 0
    while time.monotonic() < deadline:
        iteration += 1
        try:
            text = transcriber.transcribe(chunk, language=args.language, channel_mode="mono")
            compact = " ".join((text or "").split())
            has_arrow = ">>" in compact
            has_arrow_once = has_arrow_once or has_arrow
            print(
                f"[iter {iteration:03d}] ok chars={len(compact)} has_turn_marker={has_arrow} preview={compact[:140]}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            errors += 1
            print(f"[iter {iteration:03d}] error {type(exc).__name__}: {exc}", flush=True)
        time.sleep(0.8)

    print(
        f"[summary] iterations={iteration} errors={errors} has_turn_marker_once={has_arrow_once}",
        flush=True,
    )
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
