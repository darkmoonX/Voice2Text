"""Convert SRT subtitle file(s) into a single-line plain-text transcript.

Point it at a folder (or folders): for every `.srt` directly inside, it writes a
sibling `<stem>.txt` containing only the subtitle text — cue indices, timestamp
lines, formatting tags and blank lines stripped, and all cues joined into ONE
line (single-spaced). You can also pass an `.srt` file directly.

Usage:
    python srt_to_line_txt.py <folder-or-srt> [<folder-or-srt> ...]
        [--suffix .txt] [--recursive] [--no-overwrite]

Output is UTF-8 without BOM. Example:
    python srt_to_line_txt.py app/src/tests/compare_whisperx_test/input/YT_mdqmC3vyRJg
    -> writes en.txt next to en.srt
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

_TS_RE = re.compile(r"-->")
_INDEX_RE = re.compile(r"^\d+$")
_TAG_RE = re.compile(r"<[^>]+>")          # <i>, <b>, <font ...>
_ASS_RE = re.compile(r"\{[^}]*\}")         # {\an8} style override blocks
_WS_RE = re.compile(r"\s+")


def srt_to_single_line(srt_text: str) -> str:
    """Return all cue text from an SRT string joined into one whitespace-normalized line."""
    parts: list[str] = []
    for raw in srt_text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw.strip()
        if not line:
            continue
        if _TS_RE.search(line):           # timestamp line
            continue
        if _INDEX_RE.match(line):         # cue index
            continue
        line = _TAG_RE.sub("", line)
        line = _ASS_RE.sub("", line)
        line = line.strip()
        if line:
            parts.append(line)
    return _WS_RE.sub(" ", " ".join(parts)).strip()


def _convert_one(srt_path: Path, *, suffix: str, overwrite: bool) -> Optional[Path]:
    out_path = srt_path.with_suffix(suffix if suffix.startswith(".") else "." + suffix)
    if out_path.exists() and not overwrite:
        print(f"[skip] {out_path.name} exists (use default overwrite, or remove --no-overwrite)")
        return None
    text = srt_path.read_text(encoding="utf-8-sig")  # tolerate a BOM on input
    line = srt_to_single_line(text)
    out_path.write_text(line + "\n", encoding="utf-8")  # UTF-8 no BOM
    print(f"[ok] {srt_path.name} -> {out_path.name}  ({len(line)} chars)")
    return out_path


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path, help="folder(s) containing .srt, or .srt file(s)")
    ap.add_argument("--suffix", default=".txt", help="output extension (default .txt)")
    ap.add_argument("--recursive", action="store_true", help="scan folders recursively for .srt")
    ap.add_argument("--no-overwrite", action="store_true", help="skip when the .txt already exists")
    args = ap.parse_args(argv)

    srts: list[Path] = []
    for p in args.paths:
        if not p.exists():
            print(f"[skip] {p} (not found)")
            continue
        if p.is_file() and p.suffix.lower() == ".srt":
            srts.append(p)
        elif p.is_dir():
            found = sorted(p.rglob("*.srt") if args.recursive else p.glob("*.srt"))
            if not found:
                print(f"[skip] {p} (no .srt found)")
            srts.extend(found)
        else:
            print(f"[skip] {p} (not a folder or .srt)")

    if not srts:
        print("No .srt files to convert.")
        return 1

    written = 0
    for srt in srts:
        if _convert_one(srt, suffix=args.suffix, overwrite=not args.no_overwrite):
            written += 1
    print(f"\ndone: {written}/{len(srts)} written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
