"""Runtime matrix script for provider boot checks and timed transcription probing."""
from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

APP_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = APP_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from app.audio_capture import AudioChunk
from app.config import RuntimeConfig
from app.logging_utils import configure_app_logger
from app.stt.factory import create_stt_transcriber
from app.stt.preprocessing import create_audio_preprocessing_pipeline
from app.stt.vad import create_vad_pipeline


@dataclass
class ProviderRunResult:
    provider: str
    started_at: str
    ended_at: str
    init_ok: bool
    iterations: int
    skipped_by_signal_gate: int
    success_count: int
    error_count: int
    last_text_preview: str
    errors: list[str] = field(default_factory=list)
    terminal_events: list[str] = field(default_factory=list)


def _make_test_chunk(sample_rate: int = 16000, seconds: float = 2.0) -> AudioChunk:
    samples = max(1, int(sample_rate * seconds))
    t = np.arange(samples, dtype=np.float32) / float(sample_rate)
    sine = 0.20 * np.sin(2.0 * np.pi * 440.0 * t)
    noise = 0.02 * np.random.randn(samples).astype(np.float32)
    audio = np.clip(sine + noise, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype(np.int16).tobytes()
    return AudioChunk(pcm16=pcm16, sample_rate=sample_rate, channels=1)


def _load_reference_speech_chunk() -> AudioChunk | None:
    sample = SRC_ROOT / "models" / "sherpa-onnx" / "sherpa-onnx-paraformer-zh-2023-03-28" / "test_wavs" / "0.wav"
    if not sample.exists():
        return None

    try:
        with wave.open(str(sample), "rb") as wav_file:
            pcm16 = wav_file.readframes(wav_file.getnframes())
            sample_rate = int(wav_file.getframerate())
            channels = int(wav_file.getnchannels())
        return AudioChunk(pcm16=pcm16, sample_rate=sample_rate, channels=channels)
    except Exception:
        return None


def _default_config_for(provider: str) -> RuntimeConfig:
    cfg = RuntimeConfig()
    cfg.stt_provider = provider
    cfg.stt_variant = "cpu"
    cfg.model_size = "small"
    cfg.stt_auto_download = True
    cfg.model_device = "cpu"
    cfg.compute_type = "int8"
    cfg.sherpa_onnx_provider = "cpu"
    cfg.funasr_device = "cpu"
    return cfg


def _trim(text: str, limit: int = 240) -> str:
    cleaned = " ".join(text.strip().split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def run_provider_for_duration(
    provider: str,
    duration_seconds: int,
    logger: Any,
) -> ProviderRunResult:
    started = datetime.now().isoformat(timespec="seconds")
    events: list[str] = []
    errors: list[str] = []
    last_text = ""

    def progress(msg: str) -> None:
        line = f"[{provider}] {msg}"
        events.append(line)
        print(line, flush=True)
        logger.info(line)

    cfg = _default_config_for(provider)

    transcriber = None
    preprocessor = create_audio_preprocessing_pipeline(cfg)
    vad_pipeline = create_vad_pipeline(cfg, provider=provider)
    progress(
        "preprocess="
        + (",".join(preprocessor.stage_names) or "disabled")
        + "; vad="
        + (",".join(vad_pipeline.stage_names) or "disabled")
    )
    init_ok = False
    init_err = ""
    try:
        transcriber = create_stt_transcriber(cfg, progress_callback=progress)
        init_ok = True
    except Exception as exc:  # noqa: BLE001
        init_err = f"{type(exc).__name__}: {exc}"
        errors.append(init_err)
        line = f"[{provider}] init failed: {init_err}"
        events.append(line)
        print(line, flush=True)
        logger.error(line)

    deadline = time.monotonic() + float(max(1, duration_seconds))
    chunk = _load_reference_speech_chunk() or _make_test_chunk(sample_rate=16000, seconds=6.0)
    iterations = 0
    skipped_by_signal_gate = 0
    success_count = 0

    if transcriber is None:
        line = f"[{provider}] holding {duration_seconds}s validation window after init failure"
        events.append(line)
        print(line, flush=True)
        logger.warning(line)

    while time.monotonic() < deadline:
        if transcriber is None:
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
            continue

        iterations += 1
        try:
            if not transcriber.has_enough_signal(chunk, channel_mode="mono"):
                skipped_by_signal_gate += 1
                time.sleep(1.0)
                continue

            processed_chunk = preprocessor.process(chunk, channel_mode="mono")
            if not vad_pipeline.should_process(processed_chunk, channel_mode="mono"):
                skipped_by_signal_gate += 1
                time.sleep(1.0)
                continue

            text = transcriber.transcribe(processed_chunk, language=None)
            last_text = text or ""
            success_count += 1
        except Exception as exc:  # noqa: BLE001
            detail = f"{type(exc).__name__}: {exc}"
            errors.append(detail)
            line = f"[{provider}] transcribe error: {detail}"
            events.append(line)
            print(line, flush=True)
            logger.error(line)

        time.sleep(1.0)

    ended = datetime.now().isoformat(timespec="seconds")
    result = ProviderRunResult(
        provider=provider,
        started_at=started,
        ended_at=ended,
        init_ok=init_ok,
        iterations=iterations,
        skipped_by_signal_gate=skipped_by_signal_gate,
        success_count=success_count,
        error_count=len(errors),
        last_text_preview=_trim(last_text),
        errors=errors,
        terminal_events=events,
    )

    if not init_ok and init_err:
        logger.error("[%s] failed during init: %s", provider, init_err)
    else:
        logger.info(
            "[%s] completed: iterations=%s skipped=%s success=%s errors=%s",
            provider,
            iterations,
            skipped_by_signal_gate,
            success_count,
            len(errors),
        )

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run runtime STT provider matrix test.")
    parser.add_argument(
        "--providers",
        default="whisper,vosk,sherpa-onnx,riva,funasr",
        help="Comma-separated providers to test.",
    )
    parser.add_argument(
        "--duration-seconds",
        type=int,
        default=60,
        help="How long to exercise each provider.",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory for test artifacts and logs.",
    )
    args = parser.parse_args()

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = configure_app_logger(str(log_dir))
    logger.info("provider runtime matrix start: providers=%s duration=%s", providers, args.duration_seconds)

    results: list[ProviderRunResult] = []
    for provider in providers:
        header = f"\n=== Provider: {provider} ({args.duration_seconds}s) ==="
        print(header, flush=True)
        logger.info(header)
        result = run_provider_for_duration(
            provider=provider,
            duration_seconds=args.duration_seconds,
            logger=logger,
        )
        results.append(result)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = log_dir / f"provider_runtime_matrix_{stamp}.json"
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "duration_seconds": args.duration_seconds,
        "providers": providers,
        "results": [asdict(item) for item in results],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Summary ===", flush=True)
    for item in results:
        print(
            f"{item.provider}: init_ok={item.init_ok} iterations={item.iterations} "
            f"skipped={item.skipped_by_signal_gate} success={item.success_count} errors={item.error_count}",
            flush=True,
        )

    print(f"Report: {report_path}", flush=True)

    logger.info("provider runtime matrix end report=%s", report_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
