"""Fast, mechanism-only smoke check for the diarization/speaker-count-hint knobs
(rounds 0054/0055 and future rounds in the same line).

Problem this solves: verifying "does the min/max_speakers hint actually reach pyannote"
or "does the count-hint cap actually rise over a session" today means re-running the full
`compare_test_data_whisperx.py` GPU harness on the Bn/aXqBR reference clips (CER + forced
alignment + full-length diarization) — many minutes per run, only to check a yes/no
mechanism question that doesn't need an accuracy number at all.

This script drives the REAL `TranscriptionController` headlessly (same pattern as
`main_replay_regression.py`) over a SHORT slice (default 60s) of an existing reference
clip, with no CER/alignment comparison, and asserts only:
  - the pass completes without error/timeout,
  - the realtime speaker-marker count never exceeds an explicit `--diarization-max-speakers`
    cap (when set),
  - when `--speaker-count-hint` is enabled, the profile store's soft cap ends up > 0 (i.e.
    at least one estimation cycle actually fired and forwarded a cap).

This is NOT a replacement for the full harness — it proves the wiring fires and behaves,
not that the resulting transcript/speaker accuracy improved. Reserve the full Bn/aXqBR A/B
for a round's closing verdict; use this for the "did I wire it correctly" iteration loop.

Usage (from app/src, venv python):
  ..\\.venv\\Scripts\\python.exe ..\\scripts\\diagnostics\\speaker_mechanism_smoke.py \\
      --input ..\\src\\tests\\compare_whisperx_test\\input\\YT_Bn_7OcZYwrI\\voice.m4a \\
      --seconds 60 --device cuda --language zh \\
      --diarization-max-speakers 3
  # or, count-hint mechanism (short cadence so a 60s clip gets multiple cycles):
  ..\\.venv\\Scripts\\python.exe ..\\scripts\\diagnostics\\speaker_mechanism_smoke.py \\
      --input ..\\src\\tests\\compare_whisperx_test\\input\\YT_Bn_7OcZYwrI\\voice.m4a \\
      --seconds 60 --device cuda --language zh \\
      --speaker-count-hint --speaker-count-hint-seconds 15 --speaker-count-hint-window-seconds 60
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_THIS = Path(__file__).resolve()
_SRC = _THIS.parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_THIS.parent) not in sys.path:
    sys.path.insert(0, str(_THIS.parent))

from voice2text.config import RuntimeConfig  # noqa: E402
from voice2text.controller import TranscriptionController  # noqa: E402

import main_replay_regression as mrr  # noqa: E402


def _speaker_labels(text: str) -> set[str]:
    return set(re.findall(r"\[spk_\d+\]", str(text or "")))


def _build_cfg(args, clip: Path, log_dir: Path) -> RuntimeConfig:
    cfg = mrr._base_cfg(args, clip, log_dir, diarization=True)
    cfg.whisperx_diarization_min_speakers = max(0, int(args.diarization_min_speakers or 0))
    cfg.whisperx_diarization_max_speakers = max(0, int(args.diarization_max_speakers or 0))
    cfg.whisperx_speaker_count_hint_enabled = bool(args.speaker_count_hint)
    cfg.whisperx_speaker_count_hint_seconds = float(args.speaker_count_hint_seconds)
    cfg.whisperx_speaker_count_hint_window_seconds = float(args.speaker_count_hint_window_seconds)
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description="Speaker-hint mechanism smoke check (fast, no CER)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--language", default="")
    ap.add_argument("--commit-hold", type=float, default=0.0)
    ap.add_argument("--accurate-speakers", action="store_true")
    ap.add_argument("--diarization-min-speakers", type=int, default=0)
    ap.add_argument("--diarization-max-speakers", type=int, default=0)
    ap.add_argument("--speaker-count-hint", action="store_true")
    ap.add_argument("--speaker-count-hint-seconds", type=float, default=15.0)
    ap.add_argument("--speaker-count-hint-window-seconds", type=float, default=60.0)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from PySide6.QtWidgets import QApplication

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_dir = out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_dir = RuntimeConfig().ffmpeg_dll_dir

    src = Path(args.input)
    if not src.exists():
        print(f"[fatal] input not found: {src}", flush=True)
        return 2
    clip = mrr._make_short_clip(src, args.seconds, out / "clip_16k.wav", ffmpeg_dir) if args.seconds > 0 else src
    duration = mrr._wav_seconds(clip) if clip.suffix.lower() == ".wav" else float(args.seconds or 0.0)
    print(f"[smoke] clip={clip} (~{duration:.1f}s) device={args.device} language={args.language or 'auto'} "
          f"diar_min/max={args.diarization_min_speakers}/{args.diarization_max_speakers} "
          f"count_hint={args.speaker_count_hint}", flush=True)

    app = QApplication.instance() or QApplication(sys.argv[:1])

    cfg = _build_cfg(args, clip, log_dir)
    controller = TranscriptionController(cfg)

    seen_text = {"last": ""}
    orig_record = controller._record_transcript_event

    def _wrapped(payload: dict) -> None:
        src_text = str(payload.get("source_text") or "") if isinstance(payload, dict) else ""
        if src_text.strip():
            seen_text["last"] = src_text
        return orig_record(payload)

    controller._record_transcript_event = _wrapped

    errors, timed_out = mrr._run_controller_pass(
        controller,
        lambda: controller.import_audio_file(str(clip)),
        timeout_s=args.timeout,
        tag="smoke",
    )

    labels = _speaker_labels(seen_text["last"])
    soft_cap = 0
    try:
        engine = getattr(controller._transcriber, "_speaker_identity_engine", None)
        store = getattr(engine, "_profile_store", None)
        soft_cap = int(getattr(store, "_soft_speaker_cap", 0) or 0)
    except Exception:
        pass

    ok = not errors and not timed_out
    reasons: list[str] = []
    if errors:
        reasons.append(f"errors={errors}")
    if timed_out:
        reasons.append("timed_out")
    if args.diarization_max_speakers > 0 and len(labels) > args.diarization_max_speakers:
        ok = False
        reasons.append(f"speaker_labels {len(labels)} exceeds cap {args.diarization_max_speakers}")
    if args.speaker_count_hint and soft_cap <= 0:
        ok = False
        reasons.append("speaker_count_hint enabled but soft cap never rose above 0 (no estimation cycle fired -- clip/cadence too short?)")

    report = {
        "input": str(src),
        "clip_seconds": round(duration, 2),
        "diarization_min_speakers": args.diarization_min_speakers,
        "diarization_max_speakers": args.diarization_max_speakers,
        "speaker_count_hint": args.speaker_count_hint,
        "speaker_labels_seen": sorted(labels),
        "speaker_label_count": len(labels),
        "soft_speaker_cap_final": soft_cap,
        "errors": errors,
        "timed_out": timed_out,
        "ok": ok,
        "reasons": reasons,
    }
    (out / "smoke_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[smoke] speaker_labels={sorted(labels)} soft_cap_final={soft_cap}", flush=True)
    print(f"[smoke] OVERALL = {'PASS' if ok else 'FAIL'} {('(' + '; '.join(reasons) + ')') if reasons else ''}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
