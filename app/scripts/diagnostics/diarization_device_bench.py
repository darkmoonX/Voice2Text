"""Diarization device option (auto/cpu/cuda) — latency + VRAM justification.

The option is already fully wired (CLI `--whisperx-diarization-device`, config, settings dialog,
provider `_resolve_diarization_device`). The backlog gated it on "latency/VRAM data shows it is
needed"; this bench provides that data: it builds the transcriber pinned to one diarization device,
confirms the runtime actually routed there, then loads + runs the pyannote pipeline on a clip and
measures the diarization wall-time + the GPU VRAM the pipeline holds (= what `cpu` frees).

Run once per device (separate processes keep the CUDA memory measurement clean):
  ..\.venv\Scripts\python.exe ..\scripts\diagnostics\diarization_device_bench.py \
      --input <audio> --device-diar cuda --seconds 60
  ..\.venv\Scripts\python.exe ..\scripts\diagnostics\diarization_device_bench.py \
      --input <audio> --device-diar cpu  --seconds 60
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

_THIS = Path(__file__).resolve()
_SRC = _THIS.parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from voice2text.config import RuntimeConfig  # noqa: E402
from voice2text.pipeline.direct_transcription import decode_to_wav_16k_mono, read_wav  # noqa: E402
from voice2text.stt.factory import create_stt_transcriber  # noqa: E402

SAMPLE_RATE = 16000


def _cuda_mem_mb() -> float:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024.0 * 1024.0)
    except Exception:
        pass
    return 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Diarization device latency/VRAM bench")
    ap.add_argument("--input", required=True)
    ap.add_argument("--device-diar", choices=["auto", "cpu", "cuda"], default="cuda")
    ap.add_argument("--asr-device", default="cuda")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = RuntimeConfig()
    cfg.model_device = args.asr_device
    cfg.stt_variant = "auto"
    cfg.whisperx_enable_diarization = True
    cfg.whisperx_enable_forced_alignment = False
    cfg.whisperx_speaker_profile_enabled = False  # isolate the diarization pipeline cost
    cfg.whisperx_diarization_device = args.device_diar

    print(f"[bench] requested diarization device = {args.device_diar}", flush=True)
    transcriber = create_stt_transcriber(cfg, progress_callback=lambda m: print(f"[bench] {m}", flush=True))
    routed = str(getattr(transcriber, "_diarization_device", "?"))
    print(f"[bench] runtime routed diarization device = {routed}", flush=True)

    in_path = Path(args.input)
    if in_path.is_dir():
        in_path = in_path / "voice.m4a"
    wav = decode_to_wav_16k_mono(in_path, ffmpeg_dir=cfg.ffmpeg_dll_dir)
    chunk = read_wav(Path(wav))
    audio = np.frombuffer(chunk.pcm16, dtype=np.int16).astype(np.float32) / 32768.0
    if chunk.channels > 1:
        audio = audio.reshape(-1, chunk.channels).mean(axis=1)
    audio = audio[: int(args.seconds * SAMPLE_RATE)]
    print(f"[bench] audio = {audio.size / SAMPLE_RATE:.1f}s", flush=True)

    mem_before_load = _cuda_mem_mb()
    t0 = time.perf_counter()
    transcriber._ensure_diarization_pipeline_loaded()
    load_seconds = time.perf_counter() - t0
    mem_after_load = _cuda_mem_mb()

    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    t1 = time.perf_counter()
    segments = transcriber._diarization_pipeline(audio)
    run_seconds = time.perf_counter() - t1

    peak_mb = 0.0
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    except Exception:
        pass

    try:
        import pandas as pd  # type: ignore

        n_seg = int(len(segments)) if hasattr(segments, "__len__") else int(segments.shape[0])
        n_spk = int(segments["speaker"].nunique()) if isinstance(segments, pd.DataFrame) else 0
    except Exception:
        n_seg, n_spk = -1, -1

    report = {
        "requested_device": args.device_diar,
        "routed_device": routed,
        "audio_seconds": round(audio.size / SAMPLE_RATE, 1),
        "pipeline_load_seconds": round(load_seconds, 3),
        "pipeline_run_seconds": round(run_seconds, 3),
        "cuda_mb_before_load": round(mem_before_load, 1),
        "cuda_mb_after_load": round(mem_after_load, 1),
        "cuda_mb_held_by_diar": round(mem_after_load - mem_before_load, 1),
        "cuda_peak_mb_during_run": round(peak_mb, 1),
        "segments": n_seg,
        "speakers": n_spk,
    }
    print("\n==== diarization device bench ====", flush=True)
    for k, v in report.items():
        print(f"  {k} = {v}", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
