"""Backlog P0 (roadmap line 317) — main/imported-audio replay as the realtime source of truth.

`compare_whisperx_test` stays the fast per-window CER/diff regression, but it drives the
realtime engine through its OWN parallel harness (hand-built `TranscriptionLoopDeps`,
direct `engine.run()`) and never goes through the product orchestration layer
(`TranscriptionController`: bootstrap, capture build, transcriber fallback/warmup, run
loop wiring, export, temporary-source restore). So a bug in that orchestration layer
(e.g. the round-0036 hardcoded ffmpeg path) is caught by neither the harness nor unit tests.

This script closes that gap: it drives the REAL `TranscriptionController` headlessly
through its `import_audio_file` replay path (the exact path `main.py` / the tray "import
audio" action use), captures the committed realtime overlay payload exactly as the
compare harness defines `realtime_project.txt` (the clean `record_transcript_event`
source, NOT the display-only overlay frame), optionally runs the controller's own direct
whole-file pass for a no-speaker completeness baseline, then runs the shared
`subtitle_correctness_check` structural assertions (no duplicate stacking, completeness
vs direct, CJK mid-phrase spacing, speaker-marker line-start anchoring) and reports
PASS/FAIL.

Headless: uses the Qt "offscreen" platform plugin and a file source, so no GUI, display,
or audio hardware is needed. The controller's bootstrap/run still happen on their real
background threads; a `QEventLoop` in the main thread delivers the queued bootstrap/state
signals just as the live Qt app would.

Usage (from app/src, venv python):
  ..\.venv\Scripts\python.exe ..\scripts\diagnostics\main_replay_regression.py \
      --input ..\src\tests\compare_whisperx_test\input\YT_aXqBRYQSGp0_2\voice.m4a \
      --seconds 60 --device cuda --out <work_dir> [--language zh] [--diarization] [--skip-direct]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import wave
from pathlib import Path

# Headless Qt: must be set before PySide6 is imported anywhere.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Make the product package + the sibling correctness check importable.
_THIS = Path(__file__).resolve()
_SRC = _THIS.parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_THIS.parent) not in sys.path:
    sys.path.insert(0, str(_THIS.parent))

from voice2text.config import RuntimeConfig  # noqa: E402
from voice2text.controller import TranscriptionController  # noqa: E402
from voice2text.pipeline.direct_transcription import resolve_ffmpeg  # noqa: E402

import subtitle_correctness_check as scc  # noqa: E402


# --- Canonical realtime text accumulation (mirrors compare_test_data_whisperx so the
#     realtime_project.txt this produces is byte-equivalent to the harness output, which
#     is what subtitle_correctness_check's thresholds are calibrated against). ---

def _normalize_incremental_text(text: str) -> str:
    if not text:
        return ""
    lines: list[str] = []
    for raw in str(text).splitlines():
        cleaned = re.sub(r"[ \t]+", " ", raw).strip()
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines).strip()


def _append_runtime_snapshot(base: str, incoming: str) -> str:
    left = _normalize_incremental_text(base)
    right = _normalize_incremental_text(incoming)
    if not left:
        return right
    if not right or right in left:
        return left
    max_len = min(len(left), len(right), 2000)
    for size in range(max_len, 0, -1):
        if left.endswith(right[:size]):
            return _normalize_incremental_text(left + right[size:])
    sep = "\n" if re.match(r"^\s*(?:>>|S\d+:)\s*", right) else " "
    return _normalize_incremental_text(left + sep + right)


def _format_srt_time(seconds: float) -> str:
    total_ms = int(round(max(0.0, seconds) * 1000.0))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _write_realtime_payload(case_dir: Path, payload: str, duration_seconds: float) -> None:
    """Write realtime_project.{txt,srt} from the exact main overlay payload (no exporter
    snapshot collapse / speaker coalesce), matching the compare harness artifact shape."""
    case_dir.mkdir(parents=True, exist_ok=True)
    text = str(payload or "").strip()
    (case_dir / "realtime_project.txt").write_text(text + ("\n" if text else ""), encoding="utf-8")
    if text:
        end = max(0.001, float(duration_seconds or 0.0))
        srt = "\n".join(["1", f"00:00:00,000 --> {_format_srt_time(end)}", text, ""])
    else:
        srt = ""
    (case_dir / "realtime_project.srt").write_text(srt, encoding="utf-8")


def _make_short_clip(src: Path, seconds: float, out: Path, ffmpeg_dir: str) -> Path:
    """Slice the first `seconds` of `src` to 16k mono wav for a quick, deterministic pass."""
    ffmpeg = resolve_ffmpeg(ffmpeg_dir)
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-y", "-t", str(seconds), "-i", str(src), "-ac", "1", "-ar", "16000", "-f", "wav", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(f"ffmpeg slice failed: {proc.stderr[-400:]}")
    return out


def _wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(max(1, wf.getframerate()))


def _base_cfg(args, clip: Path, log_dir: Path, *, diarization: bool) -> RuntimeConfig:
    cfg = RuntimeConfig()
    cfg.model_device = args.device
    cfg.stt_variant = "auto"
    cfg.source_mode = "loopback"  # import_audio_file flips this to file
    cfg.source_file_replay_speed = 0.0
    cfg.source_file_chunk_seconds = 0.25
    cfg.whisperx_enable_diarization = bool(diarization)
    cfg.stt_provider = str(getattr(args, "stt_provider", "whisperx") or "whisperx")
    if cfg.stt_provider == "whispercpp":
        # stt_model_path defaults to "" already, but keep this explicit: whisper.cpp's
        # model resolver falls back to stt_model_path as an "explicit ggml path" when
        # set, which would misresolve to a WhisperX model dir (round 0063 gotcha).
        cfg.stt_model_path = ""
        cfg.stt_whispercpp_model_size = str(getattr(args, "whispercpp_model_size", "medium") or "medium")
        cfg.stt_whispercpp_mode = str(getattr(args, "whispercpp_mode", "server") or "server")
    cfg.transcript_export_enabled = False  # we write realtime artifacts ourselves
    cfg.session_record_enabled = False
    cfg.source_language = (str(args.language).strip() or None) if args.language else None
    cfg.subtitle_commit_hold_seconds = max(0.0, float(getattr(args, "commit_hold", 0.0) or 0.0))
    if bool(getattr(args, "accurate_speakers", False)):
        # Mirror the compare harness `--profile accurate` speaker-profile tuning
        # (the old traces used these; bare defaults resolve fewer realtime speakers).
        cfg.whisperx_speaker_profile_match_threshold = 0.65
        cfg.whisperx_speaker_profile_min_seconds = 2.0
        cfg.whisperx_speaker_profile_reconcile_threshold = 0.52
    cfg.log_dir = str(log_dir)
    return cfg


def _run_controller_pass(controller, start_callable, *, timeout_s: float, tag: str):
    """Start the controller via `start_callable` and pump a Qt event loop until the run
    finishes (runtime_state_changed False after it went True) or times out. Returns
    (errors, timed_out)."""
    from PySide6.QtCore import QEventLoop, QTimer

    loop = QEventLoop()
    seen_running = {"v": False}
    errors: list[str] = []
    timed_out = {"v": False}

    def _on_state(running: bool) -> None:
        if running:
            seen_running["v"] = True
        elif seen_running["v"] or errors:
            loop.quit()

    def _on_error(msg: str) -> None:
        errors.append(msg)
        print(f"[{tag}][error] {msg}", flush=True)
        if not seen_running["v"]:
            QTimer.singleShot(0, loop.quit)  # bootstrap failed before ever running

    def _on_status(msg: str) -> None:
        print(f"[{tag}] {msg}", flush=True)

    controller.runtime_state_changed.connect(_on_state)
    controller.error_message.connect(_on_error)
    controller.status_message.connect(_on_status)

    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(lambda: (timed_out.__setitem__("v", True), loop.quit()))
    timer.start(int(timeout_s * 1000))

    start_callable()
    loop.exec()
    timer.stop()
    try:
        controller.stop()
    except Exception:
        pass
    return errors, timed_out["v"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Main-replay realtime regression (source of truth)")
    ap.add_argument("--input", required=True, help="source audio (m4a/wav)")
    ap.add_argument("--seconds", type=float, default=60.0, help="slice length for a quick pass (<=0 = whole file)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--language", default="", help="pin source language (e.g. zh / en); empty = auto-detect")
    ap.add_argument("--diarization", action="store_true", help="enable diarization in the realtime pass (speaker markers)")
    ap.add_argument("--stt-provider", default="whisperx", choices=["whisperx", "whispercpp"], help="STT provider for the realtime pass (round 0065: whispercpp now supports live diarization too)")
    ap.add_argument("--whispercpp-model-size", default="medium", help="whisper.cpp ggml model size, used only when --stt-provider whispercpp")
    ap.add_argument("--whispercpp-mode", default="server", choices=["server", "subprocess"], help="whisper.cpp execution mode, used only when --stt-provider whispercpp")
    ap.add_argument("--commit-hold", type=float, default=0.0, help="delayed-freeze speaker re-anchor hold seconds (0 = legacy immediate freeze)")
    ap.add_argument("--accurate-speakers", action="store_true", help="use the harness --profile accurate speaker-profile tuning (match 0.65 / min_seconds 2.0 / reconcile 0.52)")
    ap.add_argument("--skip-direct", action="store_true", help="skip the direct baseline (no completeness ratio)")
    ap.add_argument("--timeout", type=float, default=1200.0, help="per-pass wall-clock timeout (s)")
    ap.add_argument("--out", required=True, help="work dir")
    args = ap.parse_args()

    from PySide6.QtWidgets import QApplication

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_dir = out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    case_dir = out / "case"
    case_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_dir = RuntimeConfig().ffmpeg_dll_dir

    src = Path(args.input)
    if not src.exists():
        print(f"[fatal] input not found: {src}", flush=True)
        return 2
    if args.seconds and args.seconds > 0:
        clip = _make_short_clip(src, args.seconds, out / "clip_16k.wav", ffmpeg_dir)
    else:
        clip = src
    duration = _wav_seconds(clip) if clip.suffix.lower() == ".wav" else float(args.seconds or 0.0)
    print(f"[main-replay] clip = {clip} (~{duration:.1f}s); device={args.device}; "
          f"language={args.language or 'auto'}; diarization={args.diarization}", flush=True)

    app = QApplication.instance() or QApplication(sys.argv[:1])

    # ---- Pass 1: realtime replay through the REAL controller ----
    print("\n==== PASS 1: realtime import_audio_file (real TranscriptionController) ====", flush=True)
    cfg_rt = _base_cfg(args, clip, log_dir, diarization=args.diarization)
    controller_rt = TranscriptionController(cfg_rt)

    # Mirror compare_test_data_whisperx exactly: each record_transcript_event carries the
    # full committed history snapshot (monotonically growing), so the canonical realtime
    # text is the LAST non-empty event's source (`final_text`); the running accumulation is
    # only a fallback for the non-monotonic case (`output_text = final_text or accumulated`).
    rt = {"last": "", "acc": ""}
    orig_record = controller_rt._record_transcript_event

    def _wrapped_record(payload: dict) -> None:
        src_text = str(payload.get("source_text") or "") if isinstance(payload, dict) else ""
        norm = _normalize_incremental_text(src_text)
        if norm:
            rt["last"] = norm
            rt["acc"] = _append_runtime_snapshot(rt["acc"], norm)
        return orig_record(payload)

    controller_rt._record_transcript_event = _wrapped_record  # tapped before the run loop builds deps

    errors_rt, to_rt = _run_controller_pass(
        controller_rt,
        lambda: controller_rt.import_audio_file(str(clip)),
        timeout_s=args.timeout,
        tag="realtime",
    )
    realtime_text = _normalize_incremental_text(rt["last"] or rt["acc"])
    _write_realtime_payload(case_dir, realtime_text, duration)
    rt_chars = len(re.sub(r"\s+", "", realtime_text))
    print(f"[main-replay] realtime payload chars={rt_chars} "
          f"timed_out={to_rt} errors={len(errors_rt)}", flush=True)

    # ---- Pass 2: direct whole-file baseline (no speaker) for completeness ----
    direct_done = False
    if not args.skip_direct:
        print("\n==== PASS 2: import_audio_file_direct (no-speaker completeness baseline) ====", flush=True)
        cfg_dir = _base_cfg(args, clip, log_dir, diarization=False)
        controller_dir = TranscriptionController(cfg_dir)
        direct_text = {"v": ""}
        controller_dir.subtitle_ready.connect(
            lambda text, _tr: direct_text.__setitem__("v", text) if str(text).strip() else None
        )
        errors_dir, to_dir = _run_controller_pass(
            controller_dir,
            lambda: controller_dir.import_audio_file_direct(str(clip)),
            timeout_s=args.timeout,
            tag="direct",
        )
        nospk = scc._strip_markers(str(direct_text["v"]))
        (case_dir / "direct_whisperx_nospk.txt").write_text(nospk.strip() + "\n", encoding="utf-8")
        direct_done = bool(nospk.strip()) and not to_dir
        d_chars = len(re.sub(r"\s+", "", nospk))
        print(f"[main-replay] direct baseline chars={d_chars} "
              f"timed_out={to_dir} errors={len(errors_dir)}", flush=True)

    # ---- Structural correctness assertions over the main-replay outputs ----
    print("\n==== Correctness check (subtitle_correctness_check) ====", flush=True)
    result = scc.check(case_dir, label="main-replay")

    passed = bool(result.get("ok")) and not to_rt and not errors_rt
    if not args.skip_direct and not direct_done:
        # A missing/timed-out baseline means completeness was not actually validated.
        passed = False
    report = {
        "input": str(src),
        "clip_seconds": round(duration, 2),
        "device": args.device,
        "language": args.language or "auto",
        "diarization": bool(args.diarization),
        "realtime_timed_out": to_rt,
        "realtime_errors": errors_rt,
        "direct_validated": direct_done,
        "correctness": {k: v for k, v in result.items()},
    }
    (out / "main_replay_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\n[main-replay] OVERALL = {'PASS' if passed else 'FAIL'}", flush=True)
    print(f"[main-replay] case dir -> {case_dir}", flush=True)
    print(f"[main-replay] report  -> {out / 'main_replay_report.json'}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
