"""One-shot quality-baseline runner (round 0070).

Runs a fixed matrix of main-replay regression passes (real TranscriptionController,
headless Qt) over the standard reference clips, aggregates the correctness metrics +
truth CER into a dated ``baseline.json``, and auto-compares against the latest previous
baseline of the same tier.

Tiers (never cross-compared — slice metrics and full-length metrics are different
populations):

  --tier quick   ~15 min: 90 s slices, 3 runs. "I changed something, is the pipeline sane?"
  --tier full    ~45-60 min: full clips (Bn capped at 600 s), 5 runs. Closing verdict.

Pacing is per-run: diarization runs are paced (--replay-speed 1.0; round 0041's GPU
pyannote crash was unpaced-replay-specific), non-diarization runs are unpaced for speed.

Usage (venv python, any cwd):
  ..\\.venv\\Scripts\\python.exe scripts\\diagnostics\\run_quality_baseline.py --tier quick
  ..\\.venv\\Scripts\\python.exe scripts\\diagnostics\\run_quality_baseline.py --tier full --device cuda

Exit codes: 0 = all pass, no regression; 1 = a run failed or regressed vs previous
baseline; 2 = setup error.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_THIS = Path(__file__).resolve()
APP_ROOT = _THIS.parents[2]
INPUT_ROOT = APP_ROOT / "src" / "tests" / "compare_whisperx_test" / "input"
DEFAULT_BASELINE_ROOT = APP_ROOT / "src" / "tests" / "claude_output" / "quality_baseline"
REGRESSION_DRIVER = _THIS.parent / "main_replay_regression.py"

# Regression thresholds vs the previous same-tier baseline.
COMPLETENESS_DROP_PP = 0.03      # completeness ratio drop > 3 percentage points
TRUTH_CER_RISE_PP = 0.02         # truth CER rise > 2 percentage points
CJK_SPACES_ABS = 5               # CJK mid-space regression needs BOTH: +5 absolute
CJK_SPACES_REL = 0.5             #   ... and +50 % relative


@dataclasses.dataclass(frozen=True)
class RunSpec:
    case_id: str
    clip_dir: str            # dir name under INPUT_ROOT
    language: str            # pinned (memory: compare-harness-pin-language)
    provider: str            # whisperx | whispercpp
    diarization: bool
    replay_speed: float      # 0 = unpaced, 1.0 = realtime pacing
    seconds: float           # slice length; 0 = whole file
    clip_seconds: float      # effective audio length actually replayed (slice or full)
    truth_srt: str = ""      # reference srt filename inside clip_dir ("" = no truth CER)
    whispercpp_model_size: str = "medium"
    whispercpp_mode: str = "server"
    # Absolute completeness sanity floor for the driver verdict. 0.85 is calibrated on the
    # standard clips; Bn (hard multi-speaker) sits at ~0.80 from the known per-window merge
    # ceiling (diffuse small merge-drops, NOT lost sections/speakers — verified round 0070).
    # Trend regression (drop vs previous baseline) is enforced separately regardless.
    min_completeness: float = 0.85
    # Per-case completeness trend tolerance (pp drop vs previous baseline before flagging).
    # vskw's whispercpp DIRECT pass is a coin flip on the music intro (~327 chars appear in
    # ~1 of 5 runs), swinging the ratio by ~3.7 pp with a byte-identical realtime side —
    # round 0071 determinism probes. Cases with known denominator noise get a wider band.
    completeness_trend_tolerance: float = COMPLETENESS_DROP_PP

    def timeout_seconds(self) -> float:
        """Per-pass driver timeout: paced replay needs at least the audio length; give
        2x + setup margin (round 0070: Bn 600 s paced blew the fixed 1200 s default)."""
        if self.replay_speed > 0:
            return max(1200.0, self.clip_seconds / self.replay_speed * 2.0 + 600.0)
        return 1800.0


QUICK_MATRIX: list[RunSpec] = [
    RunSpec("q1-axqbr2-zh-whisperx-diar", "YT_aXqBRYQSGp0_2", "zh", "whisperx", True, 1.0, 90.0, 90.0, "zh-CN.srt"),
    RunSpec("q2-vskw-en-whisperx", "YT_A-VskwEu8u4", "en", "whisperx", False, 0.0, 90.0, 90.0, "en.srt"),
    RunSpec("q3-axqbr2-zh-whispercpp", "YT_aXqBRYQSGp0_2", "zh", "whispercpp", False, 0.0, 90.0, 90.0, "zh-CN.srt"),
]

FULL_MATRIX: list[RunSpec] = [
    RunSpec("f1-axqbr2-zh-whisperx-diar", "YT_aXqBRYQSGp0_2", "zh", "whisperx", True, 1.0, 0.0, 181.0, "zh-CN.srt"),
    RunSpec("f2-vskw-en-whisperx-diar", "YT_A-VskwEu8u4", "en", "whisperx", True, 1.0, 0.0, 464.0, "en.srt"),
    RunSpec("f3-bn-zh-whisperx-diar", "YT_Bn_7OcZYwrI", "zh", "whisperx", True, 1.0, 600.0, 600.0,
            min_completeness=0.75),
    RunSpec("f4-axqbr2-zh-whispercpp-diar", "YT_aXqBRYQSGp0_2", "zh", "whispercpp", True, 1.0, 0.0, 181.0, "zh-CN.srt"),
    RunSpec("f5-vskw-en-whispercpp", "YT_A-VskwEu8u4", "en", "whispercpp", False, 0.0, 0.0, 464.0, "en.srt",
            completeness_trend_tolerance=0.05),
]

MATRICES = {"quick": QUICK_MATRIX, "full": FULL_MATRIX}


# --- truth CER -----------------------------------------------------------------

# Bump whenever normalize_for_cer changes: baselines with different versions must not
# have their truth-CER values compared (different measurement, not a regression).
CER_NORM_VERSION = "v2-opencc-t2s"

_SRT_TIME = re.compile(r"^\d{2}:\d{2}:\d{2}[,.]\d{3}\s+-->")
_PUNCT = re.compile(r"[　-〿＀-￯!-/:-@\[-`{-~]")

_T2S = None


def _to_simplified(text: str) -> str:
    """Unify Han script to Simplified so Traditional output vs zh-CN truth doesn't
    read as ~50 % CER (the pipeline emits Traditional; reference srt is Simplified)."""
    global _T2S
    if _T2S is None:
        try:
            import opencc
            _T2S = opencc.OpenCC("t2s")
        except Exception:
            _T2S = False
    return _T2S.convert(text) if _T2S else text


def srt_to_text(srt: str) -> str:
    """Strip srt indices/timestamps, keep subtitle text lines in order."""
    lines: list[str] = []
    for raw in srt.splitlines():
        s = raw.strip()
        if not s or s.isdigit() or _SRT_TIME.match(s):
            continue
        lines.append(s)
    return "\n".join(lines)


def normalize_for_cer(text: str, language: str) -> str:
    """Whitespace/marker/punctuation-free canonical form; lowercased for latin scripts.

    Absolute CER vs YouTube-caption truth is rough by nature; the invariant that matters
    is that this normalization stays IDENTICAL across baselines so the trend is real.
    """
    t = re.sub(r"\[spk_\d+\]", "", text)
    t = _PUNCT.sub("", t)
    t = re.sub(r"\s+", "", t)
    lang = language.lower()
    if lang.startswith(("en", "de", "fr", "es")):
        t = t.lower()
    elif lang.startswith("zh"):
        t = _to_simplified(t)
    return t


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ch_a in enumerate(a, start=1):
        curr = [i]
        for j, ch_b in enumerate(b, start=1):
            cost = 0 if ch_a == ch_b else 1
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def truth_cer(realtime_text: str, truth_srt_text: str, language: str) -> float | None:
    ref = normalize_for_cer(srt_to_text(truth_srt_text), language)
    hyp = normalize_for_cer(realtime_text, language)
    if not ref:
        return None
    return levenshtein(hyp, ref) / len(ref)


def _slice_truth_srt(srt: str, seconds: float) -> str:
    """Keep only cues that START before the slice end (rough but consistent)."""
    if seconds <= 0:
        return srt
    out: list[str] = []
    keep = True
    for raw in srt.splitlines():
        s = raw.strip()
        m = re.match(r"^(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s+-->", s)
        if m:
            start = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) + int(m.group(4)) / 1000.0
            keep = start < seconds
        if keep:
            out.append(raw)
        if not s:
            keep = True  # blank line ends a cue block
    return "\n".join(out)


# --- run + aggregate -----------------------------------------------------------


def run_one(spec: RunSpec, *, device: str, out_dir: Path, python_exe: str) -> dict:
    clip = INPUT_ROOT / spec.clip_dir / "voice.m4a"
    cmd = [
        python_exe, str(REGRESSION_DRIVER),
        "--input", str(clip),
        "--seconds", str(spec.seconds),
        "--device", device,
        "--language", spec.language,
        "--replay-speed", str(spec.replay_speed),
        "--stt-provider", spec.provider,
        "--timeout", str(spec.timeout_seconds()),
        "--completeness-floor", str(spec.min_completeness),
        "--out", str(out_dir),
    ]
    if spec.provider == "whispercpp":
        cmd += ["--whispercpp-model-size", spec.whispercpp_model_size,
                "--whispercpp-mode", spec.whispercpp_mode]
    if spec.diarization:
        cmd.append("--diarization")

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "runner_console.log"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(APP_ROOT / "src"))
    wall = time.monotonic() - started

    entry: dict = {
        "case_id": spec.case_id,
        "spec": dataclasses.asdict(spec),
        "exit_code": proc.returncode,
        "wall_seconds": round(wall, 1),
        "report": None,
        "truth_cer": None,
    }
    report_path = out_dir / "main_replay_report.json"
    if report_path.exists():
        entry["report"] = json.loads(report_path.read_text(encoding="utf-8"))
    if spec.truth_srt:
        truth_path = INPUT_ROOT / spec.clip_dir / spec.truth_srt
        rt_path = out_dir / "case" / "realtime_project.txt"
        if truth_path.exists() and rt_path.exists():
            srt = _slice_truth_srt(truth_path.read_text(encoding="utf-8", errors="replace"), spec.seconds)
            cer = truth_cer(rt_path.read_text(encoding="utf-8"), srt, spec.language)
            entry["truth_cer"] = round(cer, 4) if cer is not None else None
    return entry


def _run_passed(entry: dict) -> bool:
    return entry.get("exit_code") == 0


def compare_baselines(current: dict, previous: dict | None) -> list[str]:
    """Return human-readable regression findings (empty = clean)."""
    if not previous:
        return []
    cer_comparable = (
        (previous.get("meta") or {}).get("cer_norm_version")
        == (current.get("meta") or {}).get("cer_norm_version")
    )
    prev_by_id = {r["case_id"]: r for r in previous.get("runs", [])}
    findings: list[str] = []
    for run in current.get("runs", []):
        cid = run["case_id"]
        prev = prev_by_id.get(cid)
        if prev is None:
            continue
        if _run_passed(prev) and not _run_passed(run):
            findings.append(f"{cid}: PASS -> FAIL (exit {run.get('exit_code')})")
        cur_c = ((run.get("report") or {}).get("correctness") or {})
        prv_c = ((prev.get("report") or {}).get("correctness") or {})
        cur_comp, prv_comp = cur_c.get("completeness"), prv_c.get("completeness")
        if isinstance(cur_comp, (int, float)) and isinstance(prv_comp, (int, float)):
            tolerance = float(
                (run.get("spec") or {}).get("completeness_trend_tolerance", COMPLETENESS_DROP_PP))
            if prv_comp - cur_comp > tolerance:
                findings.append(f"{cid}: completeness {prv_comp:.1%} -> {cur_comp:.1%}")
        if cur_c.get("dup") and not prv_c.get("dup"):
            findings.append(f"{cid}: NEW dup-stacking: {cur_c['dup']!r}")
        cur_sp, prv_sp = cur_c.get("cjk_mid_spaces"), prv_c.get("cjk_mid_spaces")
        if isinstance(cur_sp, int) and isinstance(prv_sp, int):
            if cur_sp - prv_sp > CJK_SPACES_ABS and cur_sp > prv_sp * (1 + CJK_SPACES_REL):
                findings.append(f"{cid}: CJK mid-spaces {prv_sp} -> {cur_sp}")
        cur_cer, prv_cer = run.get("truth_cer"), prev.get("truth_cer")
        if cer_comparable and isinstance(cur_cer, (int, float)) and isinstance(prv_cer, (int, float)):
            if cur_cer - prv_cer > TRUTH_CER_RISE_PP:
                findings.append(f"{cid}: truth CER {prv_cer:.1%} -> {cur_cer:.1%}")
    return findings


def find_previous_baseline(root: Path, tier: str, current_dir: Path) -> dict | None:
    candidates = sorted(
        (p for p in root.glob(f"*_{tier}/baseline.json") if p.parent != current_dir),
        key=lambda p: p.parent.name,
        reverse=True,
    )
    for path in candidates:
        data = json.loads(path.read_text(encoding="utf-8"))
        if (data.get("meta") or {}).get("partial"):
            continue  # --only probe runs are not valid compare references
        return data
    return None


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, cwd=str(APP_ROOT), check=False)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def write_summary(path: Path, baseline: dict, findings: list[str], previous_label: str) -> None:
    lines = [
        f"# Quality baseline — {baseline['meta']['tier']} tier, {baseline['meta']['created_utc']}",
        "",
        f"- git commit: `{baseline['meta']['git_commit']}`; device: {baseline['meta']['device']}",
        f"- compared against: {previous_label}",
        "",
        "| case | pass | completeness | dup | CJK spaces | markers | truth CER | wall s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for run in baseline["runs"]:
        c = ((run.get("report") or {}).get("correctness") or {})
        comp = c.get("completeness")
        cer = run.get("truth_cer")
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            run["case_id"],
            "PASS" if _run_passed(run) else f"FAIL({run.get('exit_code')})",
            f"{comp:.1%}" if isinstance(comp, (int, float)) else "n/a",
            c.get("dup") or "-",
            c.get("cjk_mid_spaces", "n/a"),
            c.get("markers", "n/a"),
            f"{cer:.1%}" if isinstance(cer, (int, float)) else "n/a",
            run.get("wall_seconds", "n/a"),
        ))
    lines += ["", "## Regressions vs previous baseline", ""]
    lines += [f"- {f}" for f in findings] if findings else ["- none"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="One-shot quality baseline (round 0070)")
    ap.add_argument("--tier", choices=sorted(MATRICES), required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--baseline-root", default=str(DEFAULT_BASELINE_ROOT))
    ap.add_argument("--only", default="", help="substring filter on case_id (debugging)")
    ap.add_argument("--skip-compare", action="store_true")
    args = ap.parse_args()

    specs = [s for s in MATRICES[args.tier] if args.only in s.case_id]
    if not specs:
        print(f"[fatal] no cases match --only {args.only!r}")
        return 2
    missing = [s.clip_dir for s in specs if not (INPUT_ROOT / s.clip_dir / "voice.m4a").exists()]
    if missing:
        print(f"[fatal] missing input clips: {missing}")
        return 2

    root = Path(args.baseline_root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / f"{stamp}_{args.tier}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[baseline] tier={args.tier} cases={len(specs)} -> {run_dir}", flush=True)
    runs: list[dict] = []
    for i, spec in enumerate(specs, 1):
        print(f"\n[baseline] ({i}/{len(specs)}) {spec.case_id} "
              f"(provider={spec.provider} diar={spec.diarization} "
              f"pace={spec.replay_speed} seconds={spec.seconds or 'full'})", flush=True)
        entry = run_one(spec, device=args.device, out_dir=run_dir / spec.case_id,
                        python_exe=sys.executable)
        c = ((entry.get("report") or {}).get("correctness") or {})
        comp = c.get("completeness")
        print("[baseline]   -> {} wall={}s completeness={} dup={} cer={}".format(
            "PASS" if _run_passed(entry) else f"FAIL({entry.get('exit_code')})",
            entry.get("wall_seconds"),
            f"{comp:.1%}" if isinstance(comp, (int, float)) else "n/a",
            "yes" if c.get("dup") else "no",
            entry.get("truth_cer"),
        ), flush=True)
        runs.append(entry)

    baseline = {
        "meta": {
            "tier": args.tier,
            "created_utc": stamp,
            "git_commit": _git_commit(),
            "device": args.device,
            "driver": "main_replay_regression.py",
            "cer_norm_version": CER_NORM_VERSION,
            # A probe run (--only subset) must never become a future compare reference.
            "partial": len(specs) != len(MATRICES[args.tier]),
        },
        "runs": runs,
    }
    (run_dir / "baseline.json").write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    previous = None if args.skip_compare else find_previous_baseline(root, args.tier, run_dir)
    findings = compare_baselines(baseline, previous)
    prev_label = previous["meta"]["created_utc"] if previous else "none (first baseline of this tier)"
    write_summary(run_dir / "summary.md", baseline, findings, prev_label)

    all_pass = all(_run_passed(r) for r in runs)
    print(f"\n[baseline] summary -> {run_dir / 'summary.md'}", flush=True)
    print(f"[baseline] compared against: {prev_label}", flush=True)
    for f in findings:
        print(f"[baseline] REGRESSION: {f}", flush=True)
    print(f"[baseline] OVERALL = {'PASS' if all_pass and not findings else 'FAIL'}", flush=True)
    return 0 if all_pass and not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())
