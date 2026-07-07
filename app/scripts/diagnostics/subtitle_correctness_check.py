"""Repeatable main-subtitle correctness checklist (backlog P0 / line 304).

Runs structural correctness checks over a realtime output dir (or several), the
"source of truth" being the main/imported-audio replay outputs, NOT the harness
direct pass. Checks:

  1. duplicate stacking  -- no adjacent repeated >=8-gram in the committed text.
  2. history completeness -- realtime char count vs direct_whisperx_nospk (>=85%).
  3. CJK pause spacing    -- count mid-phrase (CJK-flanked) spaces that are NOT at a
                             genuine pause; these are mostly a symptom of the
                             at-ceiling per-window merge-drop (see memory
                             cjk-rolling-merge-at-ceiling), partly genuine pauses.
  4. speaker markers      -- present and line-start anchored in the SRT.

Usage:
  python subtitle_correctness_check.py <output_case_dir> [<output_case_dir> ...]
Each dir must contain realtime_project.txt (+ optionally .srt and
direct_whisperx_nospk.txt). With no args, runs a small built-in CJK+EN sample set
under tests/compare_whisperx_test/output.
"""
from __future__ import annotations
import sys
import re
from pathlib import Path

# Some Windows terminals default to a non-UTF-8 codepage (e.g. cp950), which raises
# UnicodeEncodeError on the visible-space marker printed below. Reconfigure stdout to
# tolerate it rather than crashing mid-report.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

CJK = r"㐀-䶿一-鿿豈-﫿぀-ヿ"


def _strip_markers(text: str) -> str:
    return re.sub(r"\[spk_\d+\]", "", text)


def _midphrase_cjk_spaces(text: str) -> list[str]:
    hits: list[str] = []
    for m in re.finditer(rf"(?<=[{CJK}])[ 　]+(?=[{CJK}])", text):
        i = m.start()
        hits.append(text[max(0, i - 6):i] + "␣" + text[m.end():m.end() + 6])
    return hits


def _max_adjacent_repeat(text: str, n: int = 8) -> str:
    t = re.sub(r"\s+", "", text)
    for i in range(len(t) - n + 1):
        g = t[i:i + n]
        if t[i + n:i + 2 * n] == g:
            return g
    return ""


def check(case_dir: Path, label: str | None = None) -> dict:
    label = label or case_dir.name
    proj = case_dir / "realtime_project.txt"
    if not proj.exists():
        print(f"[{label}] MISSING realtime_project.txt")
        return {"ok": False}
    raw = proj.read_text(encoding="utf-8")
    body = _strip_markers(raw)
    r_chars = len(re.sub(r"\s+", "", body))

    direct = case_dir / "direct_whisperx_nospk.txt"
    d_chars = len(re.sub(r"\s+", "", direct.read_text(encoding="utf-8"))) if direct.exists() else 0
    completeness = (r_chars / d_chars) if d_chars else None

    spaces = _midphrase_cjk_spaces(body)
    dup = _max_adjacent_repeat(body, 8)

    markers = re.findall(r"\[spk_\d+\]", raw)
    bad_anchor = 0
    srt = case_dir / "realtime_project.srt"
    if srt.exists():
        for line in srt.read_text(encoding="utf-8").splitlines():
            for m in re.finditer(r"\[spk_\d+\]|spk_\d+:", line):
                if m.start() != 0 and line[:m.start()].strip() != "":
                    bad_anchor += 1

    comp_str = f"{completeness:.1%}" if completeness is not None else "n/a"
    comp_flag = "OK" if (completeness is None or completeness >= 0.85) else "LOW"
    print(f"[{label}]")
    print(f"   completeness   : realtime {r_chars} / direct {d_chars} = {comp_str}  {comp_flag}")
    print(f"   dup-stacking   : {'NONE' if not dup else 'FOUND ' + dup}")
    print(f"   CJK mid-spaces : {len(spaces)}  {'OK' if not spaces else 'review'}")
    for h in spaces[:12]:
        print(f"        ...{h}...")
    print(f"   speaker markers: {len(markers)} present; bad-anchored: {bad_anchor}")
    print()
    return {
        "ok": (not dup) and (completeness is None or completeness >= 0.85) and bad_anchor == 0,
        "completeness": completeness,
        "dup": dup,
        "cjk_mid_spaces": len(spaces),
        "markers": len(markers),
        "bad_anchor": bad_anchor,
    }


def _default_cases() -> list[tuple[str, Path]]:
    root = Path(__file__).resolve().parents[2] / "src" / "tests" / "compare_whisperx_test" / "output"
    sample = {
        "CJK aXqBR2(181s)": "0036-verify-cjk-harness/YT_aXqBRYQSGp0_2",
        "CJK X2ymhL-zh(966s)": "0023-gate-off-X2ymhL-zh/YT_X2ymhL-4Dsg",
        "EN vskw(464s)": "27-lv60k-vskw/YT_A-VskwEu8u4",
        "EN mdqm(620s)": "27-lv60k-mdqm/YT_mdqmC3vyRJg",
    }
    return [(lab, root / rel) for lab, rel in sample.items() if (root / rel).exists()]


def main(argv: list[str]) -> int:
    if argv:
        cases = [(Path(p).name, Path(p)) for p in argv]
    else:
        cases = _default_cases()
        if not cases:
            print("No cases found; pass output case dirs as arguments.")
            return 2
    for lab, d in cases:
        check(d, lab)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
