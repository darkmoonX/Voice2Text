"""Per-speaker character-count / share tally across compare-harness output dirs.

Ingests one or more compare output directories (each holding
`direct_whisperx.json` and/or `realtime_project.json`) and reports, per clip and
per path (direct vs realtime), every speaker's character count, its share of the
clip's total non-marker characters, and a "minority" flag for speakers whose
share falls below a threshold.

Why two parsers:
- direct json is per-cue: each cue carries `speaker`/`visible_speaker`/
  `profile_speaker` fields (whole-file pyannote clustering + cross-window
  profiles). We attribute each cue's text to its chosen speaker field.
- realtime json is ONE big committed-history cue with inline `[spk_xxx]`
  markers; speaker is encoded in the text, not a field. We split on the markers
  and attribute each span to the marker that precedes it. Text before the first
  marker is unattributed ("<none>").

Char counting excludes whitespace/newlines and the `[spk_xxx]` markers
themselves, so CJK and Latin clips are compared on visible glyph count.

Usage:
    python spk_char_distribution.py <dir> [<dir> ...] [options]

A <dir> that itself contains the json files is treated as one clip; otherwise
its immediate subdirectories are scanned for clips (so you can point at a parent
like app/src/tests/compare_whisperx_test/output/).

Options:
    --speaker-field {profile,visible,raw}   direct speaker field (default profile)
    --minority-threshold FLOAT              share below this flags minority (default 0.10)
    --out PATH                              also write a UTF-8 json report
    --glob PATTERN                          only scan subdirs matching this (default *)

Console output is ASCII-only (speaker ids + numbers) so it is safe on cp950
consoles; the optional --out json is UTF-8.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Optional

_MARKER_RE = re.compile(r"\[(spk_[0-9A-Za-z]+|S[0-9]+|speaker[_0-9A-Za-z]*)\]", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")

_DIRECT_FIELD = {
    "profile": ("profile_speaker", "visible_speaker", "speaker"),
    "visible": ("visible_speaker", "speaker", "profile_speaker"),
    "raw": ("raw_speaker", "speaker", "visible_speaker"),
}


def _visible_len(text: str) -> int:
    """Glyph count excluding whitespace and inline speaker markers."""
    no_marker = _MARKER_RE.sub("", text or "")
    return len(_WS_RE.sub("", no_marker))


def _pick(cue: dict, fields: tuple[str, ...]) -> str:
    for f in fields:
        val = str(cue.get(f, "") or "").strip()
        if val:
            return val
    return "<none>"


def _tally_direct(path: Path, field_pref: tuple[str, ...]) -> Optional[dict[str, int]]:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    cues = data.get("cues", []) if isinstance(data, dict) else (data or [])
    counts: dict[str, int] = {}
    for cue in cues:
        if not isinstance(cue, dict):
            continue
        spk = _pick(cue, field_pref)
        counts[spk] = counts.get(spk, 0) + _visible_len(str(cue.get("text", "") or ""))
    return counts


def _tally_realtime(path: Path) -> Optional[dict[str, int]]:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    cues = data.get("cues", []) if isinstance(data, dict) else (data or [])
    text = "\n".join(str(c.get("text", "") or "") for c in cues if isinstance(c, dict))
    counts: dict[str, int] = {}
    pos = 0
    current = "<none>"
    for m in _MARKER_RE.finditer(text):
        span = text[pos:m.start()]
        counts[current] = counts.get(current, 0) + _visible_len(span)
        current = m.group(1).lower()
        pos = m.end()
    counts[current] = counts.get(current, 0) + _visible_len(text[pos:])
    return {k: v for k, v in counts.items() if v > 0} or {current: 0}


def _format_table(counts: dict[str, int], minority: float) -> tuple[list[str], int]:
    total = sum(counts.values())
    rows = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    lines = []
    minority_n = 0
    for spk, n in rows:
        share = (n / total) if total else 0.0
        flag = ""
        if spk != "<none>" and total and share < minority:
            flag = "  <-- minority"
            minority_n += 1
        lines.append(f"    {spk:<12} {n:>7d}  {share*100:6.2f}%{flag}")
    lines.append(f"    {'TOTAL':<12} {total:>7d}  {'100.00' if total else '0.00':>6}%")
    return lines, minority_n


def _find_clips(root: Path, pattern: str) -> list[Path]:
    if (root / "direct_whisperx.json").exists() or (root / "realtime_project.json").exists():
        return [root]
    clips = []
    for sub in sorted(root.glob(pattern)):
        if sub.is_dir() and (
            (sub / "direct_whisperx.json").exists() or (sub / "realtime_project.json").exists()
        ):
            clips.append(sub)
    return clips


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path, help="compare output dir(s) or parent dir(s)")
    ap.add_argument("--speaker-field", choices=("profile", "visible", "raw"), default="profile")
    ap.add_argument("--minority-threshold", type=float, default=0.10)
    ap.add_argument("--glob", default="*", help="subdir glob when scanning a parent dir")
    ap.add_argument("--out", type=Path, default=None, help="write UTF-8 json report")
    args = ap.parse_args(argv)

    field_pref = _DIRECT_FIELD[args.speaker_field]
    clips: list[Path] = []
    for d in args.dirs:
        if not d.exists():
            print(f"[skip] {d} (not found)")
            continue
        clips.extend(_find_clips(d, args.glob))
    if not clips:
        print("No clips with direct_whisperx.json / realtime_project.json found.")
        return 1

    report: dict[str, dict] = {}
    print(f"speaker-field={args.speaker_field}  minority<{args.minority_threshold*100:.0f}%  clips={len(clips)}")
    for clip in clips:
        name = clip.name
        print(f"\n=== {name}")
        report[name] = {"path": str(clip)}
        for label, counts in (
            ("direct", _tally_direct(clip / "direct_whisperx.json", field_pref)),
            ("realtime", _tally_realtime(clip / "realtime_project.json")),
        ):
            if counts is None:
                print(f"  {label}: (missing)")
                report[name][label] = None
                continue
            real_spk = sorted(k for k in counts if k != "<none>")
            lines, minority_n = _format_table(counts, args.minority_threshold)
            print(f"  {label}: speakers={len(real_spk)} minority={minority_n}")
            for ln in lines:
                print(ln)
            total = sum(counts.values())
            report[name][label] = {
                "speakers": len(real_spk),
                "minority_count": minority_n,
                "total_chars": total,
                "by_speaker": {
                    k: {"chars": v, "share": (v / total if total else 0.0)}
                    for k, v in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
                },
            }

    if args.out:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
