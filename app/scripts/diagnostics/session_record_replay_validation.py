"""Round 0020 Phase B — headless record -> replay GPU validation.

Drives the REAL `TranscriptionLoopEngine` (the same loop main.py / the compare harness use),
with a `FileReplayAudioCapture` standing in for the live source (so the run is deterministic and
needs no audio hardware or Qt). The file capture is wrapped in the product's `RecordingAudioCapture`,
so this exercises the genuine record path: PCM -> WAV + redacted manifest. Then it replays the
recorded session via `apply_replay_session` (source_mode=file on the recorded WAV + restored STT
config) through the same engine and compares the two transcripts.

This closes the 0020 acceptance "a recorded session replays through the main pipeline via the file
path with the saved config; output matches the original within non-determinism" without real live
capture (which a headless background job can't drive).

Usage (from app/src, venv python):
  ..\.venv\Scripts\python.exe ..\scripts\diagnostics\session_record_replay_validation.py \
      --input ..\src\tests\compare_whisperx_test\input\YT_aXqBRYQSGp0_2\voice.m4a \
      --seconds 25 --device cuda --out <work_dir>
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import difflib
import json
from pathlib import Path
import subprocess
import sys
import threading
import wave

# Make the product package importable when run from app/scripts/diagnostics.
_THIS = Path(__file__).resolve()
_SRC = _THIS.parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from voice2text.config import RuntimeConfig  # noqa: E402
from voice2text.capture import build_capture_from_config  # noqa: E402
from voice2text.capture.session_recorder import (  # noqa: E402
    RecordingAudioCapture,
    apply_replay_session,
    load_session_manifest,
)
from voice2text.pipeline.direct_transcription import decode_to_wav_16k_mono, resolve_ffmpeg  # noqa: E402
from voice2text.pipeline.gpu_telemetry import GpuTelemetryReporter  # noqa: E402
from voice2text.pipeline.segment_artifacts import SegmentArtifacts  # noqa: E402
from voice2text.pipeline.subtitle_assembler import SubtitleAssembler  # noqa: E402
from voice2text.pipeline.text_delta_logger import TextDeltaLogger  # noqa: E402
from voice2text.pipeline.transcription_loop import TranscriptionLoopDeps, TranscriptionLoopEngine  # noqa: E402
from voice2text.stt.factory import create_stt_transcriber  # noqa: E402
from voice2text.stt.preprocessing import create_audio_preprocessing_pipeline  # noqa: E402


def _make_short_clip(src: Path, seconds: float, out: Path, ffmpeg_dir: str) -> Path:
    """Slice the first `seconds` of `src` to a 16k mono wav (so the test pass is quick)."""
    ffmpeg = resolve_ffmpeg(ffmpeg_dir)
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-y", "-t", str(seconds), "-i", str(src), "-ac", "1", "-ar", "16000", "-f", "wav", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(f"ffmpeg slice failed: {proc.stderr[-400:]}")
    return out


def _base_cfg(device: str, clip: Path, log_dir: Path) -> RuntimeConfig:
    cfg = RuntimeConfig()
    cfg.model_device = device
    cfg.stt_variant = "auto"
    cfg.source_mode = "file"
    cfg.source_file_path = str(clip)
    cfg.source_file_replay_speed = 0.0
    cfg.source_file_chunk_seconds = 0.25
    cfg.whisperx_enable_diarization = False  # keep the pass fast; replay fidelity is the target
    cfg.log_dir = str(log_dir)
    return cfg


def _run_loop(cfg: RuntimeConfig, *, wrap_recorder_dir: Path | None) -> tuple[str, object]:
    """Build capture+transcriber and drive the real loop engine once; return (final_text, capture)."""
    transcriber = create_stt_transcriber(cfg, progress_callback=lambda m: print(f"[stt] {m}", flush=True))
    try:
        transcriber.prewarm()
    except Exception:
        pass
    capture = build_capture_from_config(cfg, on_status=lambda m: print(f"[src] {m}", flush=True))
    if wrap_recorder_dir is not None:
        capture = RecordingAudioCapture(
            capture,
            out_dir=wrap_recorder_dir,
            config_snapshot=asdict(cfg),
            on_status=lambda m: print(f"[rec] {m}", flush=True),
        )
    capture.start()

    assembler = SubtitleAssembler()
    preprocess_pipeline = create_audio_preprocessing_pipeline(cfg)
    final_text = ""

    def _record_event(event: dict) -> None:
        nonlocal final_text
        source = str(event.get("source_text") or "")
        if source.strip():
            final_text = source

    deps = TranscriptionLoopDeps(
        config=cfg,
        subtitle_assembler=assembler,
        text_delta_logger=TextDeltaLogger(lambda _p, _t: None),
        segment_artifacts=SegmentArtifacts(log_dir=str(cfg.log_dir)),
        gpu_telemetry=GpuTelemetryReporter(interval_seconds=5.0),
        get_capture=lambda: capture,
        get_transcriber=lambda: transcriber,
        get_preprocess_pipeline=lambda: preprocess_pipeline,
        get_translator=lambda: None,
        recover_capture_backend=lambda: False,
        recover_from_runtime_transcription_error=lambda _m: False,
        emit_status=lambda m: print(f"[loop] {m}", flush=True),
        emit_debug_event=lambda _r: None,
        emit_subtitle_ready=lambda _s, _t: None,
        record_transcript_event=_record_event,
    )
    running = threading.Event()
    running.set()
    try:
        TranscriptionLoopEngine(deps).run(running)
    finally:
        try:
            capture.stop()
        except Exception:
            pass
    return final_text.strip(), capture


def _wav_pcm_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(max(1, wf.getframerate()))


def main() -> int:
    ap = argparse.ArgumentParser(description="Round 0020 record->replay GPU validation")
    ap.add_argument("--input", required=True, help="source audio (m4a/wav)")
    ap.add_argument("--seconds", type=float, default=25.0, help="slice length for a quick pass")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True, help="work dir")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_dir = out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_dir = RuntimeConfig().ffmpeg_dll_dir

    src = Path(args.input)
    clip = _make_short_clip(src, args.seconds, out / "clip_16k.wav", ffmpeg_dir)
    print(f"[val] clip = {clip} ({_wav_pcm_seconds(clip):.1f}s)", flush=True)

    # ---- Pass 1: record ----
    rec_dir = out / "recordings" / "session"
    print("\n==== PASS 1: record (file source wrapped in RecordingAudioCapture) ====", flush=True)
    cfg1 = _base_cfg(args.device, clip, log_dir)
    text1, _ = _run_loop(cfg1, wrap_recorder_dir=rec_dir)
    print(f"[val] transcript1 len={len(text1)}", flush=True)

    # ---- Validate manifest + recorded WAV ----
    manifest = load_session_manifest(rec_dir)
    rec_wav = Path(manifest["_wav_path"])
    checks: dict[str, object] = {}
    checks["manifest_exists"] = (rec_dir / "manifest.json").exists()
    checks["wav_exists"] = rec_wav.exists()
    checks["chunk_count"] = int(manifest.get("chunk_count") or 0)
    checks["total_pcm_bytes"] = int(manifest.get("total_pcm_bytes") or 0)
    rec_seconds = _wav_pcm_seconds(rec_wav) if rec_wav.exists() else 0.0
    clip_seconds = _wav_pcm_seconds(clip)
    # The file capture appends `segment_seconds` of trailing silence (EOF flush, round 0009),
    # so a faithful recording is clip + ~segment_seconds. Assert no audio was dropped, and that
    # the extra tail matches the product's configured flush.
    seg = float(cfg1.segment_seconds)
    checks["recorded_wav_seconds"] = round(rec_seconds, 2)
    checks["clip_seconds"] = round(clip_seconds, 2)
    checks["expected_with_flush_seconds"] = round(clip_seconds + seg, 2)
    checks["recorded_captures_full_audio"] = rec_seconds + 0.5 >= clip_seconds
    checks["recorded_matches_clip_plus_flush"] = abs(rec_seconds - (clip_seconds + seg)) < 1.0
    cfg_snap = manifest.get("config") or {}
    tok = str(cfg_snap.get("whisperx_hf_token", ""))
    checks["token_redacted"] = tok in ("", "<redacted>")
    checks["config_has_model_size"] = "model_size" in cfg_snap
    checks["config_has_seg_hop"] = ("segment_seconds" in cfg_snap and "hop_seconds" in cfg_snap)

    # ---- Pass 2: replay the recorded session ----
    print("\n==== PASS 2: replay recorded session (apply_replay_session) ====", flush=True)
    cfg2 = RuntimeConfig()
    cfg2.model_device = args.device
    cfg2.log_dir = str(log_dir)
    applied = apply_replay_session(cfg2, rec_dir)
    print(f"[val] replay source_mode={cfg2.source_mode} file={Path(cfg2.source_file_path).name} "
          f"seg={cfg2.segment_seconds} hop={cfg2.hop_seconds} model={cfg2.model_size}", flush=True)
    checks["replay_source_is_file"] = (cfg2.source_mode == "file")
    checks["replay_points_at_recorded_wav"] = (Path(cfg2.source_file_path).resolve() == rec_wav.resolve())
    text2, _ = _run_loop(cfg2, wrap_recorder_dir=None)
    print(f"[val] transcript2 len={len(text2)}", flush=True)

    sim = difflib.SequenceMatcher(None, text1, text2).ratio()
    checks["transcript1_len"] = len(text1)
    checks["transcript2_len"] = len(text2)
    checks["transcript_char_sim"] = round(sim, 4)
    checks["transcript_match"] = sim >= 0.97

    report = {"input": str(src), "seconds": args.seconds, "checks": checks,
              "transcript1": text1, "transcript2": text2}
    (out / "record_replay_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n==== Round 0020 record->replay validation ====", flush=True)
    passed = True
    for k, v in checks.items():
        ok = v not in (False, 0) if k.endswith(("exists", "redacted", "matches_clip", "is_file",
                                                 "recorded_wav", "match")) or k.startswith(("config_has", "replay_")) else True
        flag = ""
        if isinstance(v, bool):
            flag = " OK" if v else " <-- FAIL"
            if not v:
                passed = False
        print(f"  {k} = {v}{flag}", flush=True)
    print(f"\n[val] OVERALL = {'PASS' if passed and checks['transcript_match'] else 'CHECK'} "
          f"(char_sim={checks['transcript_char_sim']})", flush=True)
    print(f"[val] report -> {out / 'record_replay_report.json'}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
