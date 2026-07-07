"""Exercise WhisperX alignment for ~1 minute to validate crash-resilience changes."""
from __future__ import annotations

import argparse
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
from voice2text.logging_utils import configure_app_logger
from voice2text.stt.factory import create_stt_transcriber


def _load_chunk(path: Path) -> AudioChunk:
    with wave.open(str(path), "rb") as wav_file:
        sample_rate = int(wav_file.getframerate())
        channels = int(wav_file.getnchannels())
        pcm16 = wav_file.readframes(wav_file.getnframes())
    return AudioChunk(pcm16=pcm16, sample_rate=sample_rate, channels=channels)


def _build_config(model: str, alignment_device: str) -> RuntimeConfig:
    cfg = RuntimeConfig()
    cfg.stt_provider = "whisperx"
    cfg.stt_variant = "gpu"
    cfg.model_device = "cuda"
    cfg.compute_type = "float16"
    cfg.stt_auto_download = True
    cfg.stt_model_path = model
    cfg.source_language = "en"
    cfg.whisperx_enable_forced_alignment = True
    cfg.whisperx_enable_phoneme_asr = True
    cfg.whisperx_enable_vad = True
    cfg.whisperx_vad_method = "silero-vad"
    cfg.whisperx_alignment_language = "follow-source"
    cfg.whisperx_alignment_model = "WAV2VEC2_ASR_BASE_960H"
    cfg.whisperx_alignment_device = alignment_device
    cfg.debug_mode = True
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="WhisperX alignment stability probe.")
    parser.add_argument(
        "--source-wav",
        default=str(SRC_ROOT / "segments" / "latest_segment_stt.wav"),
        help="WAV source chunk to replay.",
    )
    parser.add_argument("--duration-seconds", type=int, default=65, help="Probe duration.")
    parser.add_argument("--model", default="medium", help="WhisperX model ref/path.")
    parser.add_argument(
        "--alignment-device",
        default="cuda",
        choices=["auto", "cpu", "cuda"],
        help="Requested WhisperX alignment device.",
    )
    parser.add_argument("--language", default="en", help="Language hint.")
    parser.add_argument("--log-dir", default="logs", help="Log directory.")
    args = parser.parse_args()

    wav_path = Path(args.source_wav).resolve()
    if not wav_path.exists():
        print(f"[error] source wav not found: {wav_path}", flush=True)
        return 2

    logger = configure_app_logger(args.log_dir)

    def progress(message: str) -> None:
        line = f"[whisperx-stability] {message}"
        print(line, flush=True)
        logger.info(line)

    chunk = _load_chunk(wav_path)
    cfg = _build_config(args.model, args.alignment_device)
    cfg.source_language = (args.language or "en").strip().lower()

    progress(
        "start "
        + json.dumps(
            {
                "wav": str(wav_path),
                "sample_rate": chunk.sample_rate,
                "channels": chunk.channels,
                "pcm_bytes": len(chunk.pcm16),
                "duration_seconds": args.duration_seconds,
                "requested_alignment_device": args.alignment_device,
                "model": args.model,
            },
            ensure_ascii=False,
        )
    )

    transcriber = create_stt_transcriber(cfg, progress_callback=progress)
    resolved_align_device = str(getattr(transcriber, "_alignment_device", "unknown"))
    progress(f"resolved_alignment_device={resolved_align_device}")

    deadline = time.monotonic() + float(max(1, args.duration_seconds))
    iterations = 0
    success = 0
    errors = 0
    last_preview = ""
    while time.monotonic() < deadline:
        iterations += 1
        try:
            text = transcriber.transcribe(chunk, language=cfg.source_language, channel_mode="mono")
            success += 1
            cleaned = " ".join((text or "").split())
            if cleaned:
                last_preview = cleaned[:120]
            progress(
                f"iteration={iterations} ok=1 text_chars={len(cleaned)}"
            )
        except Exception as exc:  # noqa: BLE001
            errors += 1
            progress(f"iteration={iterations} ok=0 error={type(exc).__name__}: {exc}")
        time.sleep(0.8)

    progress(
        "summary "
        + json.dumps(
            {
                "iterations": iterations,
                "success": success,
                "errors": errors,
                "last_preview": last_preview,
            },
            ensure_ascii=False,
        )
    )
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

