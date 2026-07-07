"""Speaker-attribution accuracy of direct / realtime vs a ground-truth spk_subtitles.

The compare harness only reports speaker error *realtime-vs-direct* (both can be
wrong). This tool scores each path against a hand-corrected ground truth so the
diarization/profile quality has an objective number.

Ground truth = `<input>/<clip>/spk_subtitles`, lines `[spk_xxx] <text>` (one turn
per line). Per the way it was built (derived from direct text with speaker labels
corrected by hand), the *text* is ~direct and the *speaker labels are correct* —
so aligning direct to it is almost pure speaker scoring, and realtime adds its own
text drift on top.

Because predicted speaker ids are an arbitrary clustering (pred `spk_000` need not
equal truth `spk_000`), labels are matched label-agnostically: we align candidate
text tokens to reference tokens, build a pred x ref confusion matrix over the
matched tokens, then pick the optimal pred->ref mapping (Hungarian) and report the
token-level accuracy under that mapping, plus per-reference-speaker recall.

Only text-matched tokens are scored for speakers (substitutions/insertions are text
errors, not speaker errors), so the metric isolates "given the words we got right,
did we attribute them to the right person".

Usage:
    python spk_accuracy_vs_truth.py <case-dir|parent> [...] [--unit auto|char|word]
        [--input-root DIR] [--reference FILE] [--out report.json]

A case dir holds direct_whisperx.json / realtime_project.json; a parent dir is
scanned for such subdirs. Reference is auto-found at
<input-root>/<case-name>/spk_subtitles unless --reference is given (single clip).
Console output is ASCII-only (cp950-safe).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

from scipy.optimize import linear_sum_assignment  # type: ignore

_MARKER_RE = re.compile(r"\[(spk_[0-9A-Za-z]+|s[0-9]+|speaker[_0-9A-Za-z]*)\]", re.IGNORECASE)
_CJK_RE = re.compile(r"[㐀-鿿豈-﫿぀-ヿ]")
_WORD_PUNCT_RE = re.compile(r"^[^\w']+|[^\w']+$")

_DIRECT_FIELDS = ("profile_speaker", "visible_speaker", "speaker")


def _strip_markers(text: str) -> str:
    return _MARKER_RE.sub(" ", text or "")


def _detect_unit(text: str) -> str:
    sample = _strip_markers(text)
    if not sample:
        return "word"
    cjk = len(_CJK_RE.findall(sample))
    letters = sum(1 for c in sample if c.isalpha())
    return "char" if letters and (cjk / max(1, letters)) > 0.3 else "char" if cjk and not letters else "word"


def _tokenize(text: str, unit: str) -> list[str]:
    text = text or ""
    if unit == "char":
        return [c for c in text if not c.isspace() and not _MARKER_RE.match(c)]
    out = []
    for tok in text.split():
        t = _WORD_PUNCT_RE.sub("", tok).lower()
        if t:
            out.append(t)
    return out


def _ref_tokens(ref_path: Path, unit: str) -> list[tuple[str, str]]:
    """[(token, speaker)] from the ground-truth spk_subtitles."""
    pairs: list[tuple[str, str]] = []
    for raw in ref_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _MARKER_RE.match(line)
        spk = m.group(1).lower() if m else "<none>"
        body = line[m.end():] if m else line
        for tok in _tokenize(body, unit):
            pairs.append((tok, spk))
    return pairs


def _cand_tokens_direct(path: Path, unit: str) -> list[tuple[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cues = data.get("cues", []) if isinstance(data, dict) else (data or [])
    pairs: list[tuple[str, str]] = []
    for cue in cues:
        if not isinstance(cue, dict):
            continue
        spk = "<none>"
        for f in _DIRECT_FIELDS:
            v = str(cue.get(f, "") or "").strip()
            if v:
                spk = v.lower()
                break
        for tok in _tokenize(str(cue.get("text", "") or ""), unit):
            pairs.append((tok, spk))
    return pairs


def _cand_tokens_realtime(path: Path, unit: str) -> list[tuple[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cues = data.get("cues", []) if isinstance(data, dict) else (data or [])
    text = "\n".join(str(c.get("text", "") or "") for c in cues if isinstance(c, dict))
    pairs: list[tuple[str, str]] = []
    pos, cur = 0, "<none>"
    for m in _MARKER_RE.finditer(text):
        for tok in _tokenize(text[pos:m.start()], unit):
            pairs.append((tok, cur))
        cur = m.group(1).lower()
        pos = m.end()
    for tok in _tokenize(text[pos:], unit):
        pairs.append((tok, cur))
    return pairs


def _score(cand: list[tuple[str, str]], ref: list[tuple[str, str]]) -> dict:
    cand_toks = [t for t, _ in cand]
    ref_toks = [t for t, _ in ref]
    sm = SequenceMatcher(a=cand_toks, b=ref_toks, autojunk=False)
    aligned: list[tuple[str, str]] = []  # (pred_spk, ref_spk) over text-matched tokens
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for k in range(i2 - i1):
                aligned.append((cand[i1 + k][1], ref[j1 + k][1]))
    pred_labels = sorted({p for p, _ in aligned})
    ref_labels = sorted({r for _, r in aligned})
    # confusion[pred][ref]
    conf: dict[str, dict[str, int]] = {p: {r: 0 for r in ref_labels} for p in pred_labels}
    for p, r in aligned:
        conf[p][r] += 1
    # optimal pred->ref mapping (maximize matched tokens)
    mapping: dict[str, str] = {}
    if pred_labels and ref_labels:
        import numpy as np
        mat = np.array([[conf[p][r] for r in ref_labels] for p in pred_labels], dtype=float)
        rows, cols = linear_sum_assignment(-mat)
        for ri, ci in zip(rows, cols):
            mapping[pred_labels[ri]] = ref_labels[ci]
    correct = sum(1 for p, r in aligned if mapping.get(p) == r)
    total = len(aligned)
    # per-ref recall
    ref_total: dict[str, int] = {r: 0 for r in ref_labels}
    ref_correct: dict[str, int] = {r: 0 for r in ref_labels}
    for p, r in aligned:
        ref_total[r] += 1
        if mapping.get(p) == r:
            ref_correct[r] += 1
    return {
        "pred_speakers": len({s for _, s in cand if s != "<none>"}),
        "ref_speakers": len({r for r in ref_labels if r != "<none>"}),
        "aligned_tokens": total,
        "cand_tokens": len(cand),
        "ref_tokens": len(ref),
        "speaker_accuracy": (correct / total) if total else 0.0,
        "mapping": mapping,
        "ref_recall": {r: (ref_correct[r] / ref_total[r] if ref_total[r] else 0.0) for r in ref_labels},
        "ref_token_share": {r: (ref_total[r] / total if total else 0.0) for r in ref_labels},
    }


# Public aliases so the compare harness can reuse the exact same scoring (identical
# numbers in summary.txt and in this standalone tool).
detect_unit = _detect_unit
ref_tokens_from_file = _ref_tokens
cand_tokens_direct = _cand_tokens_direct
cand_tokens_realtime = _cand_tokens_realtime
score_attribution = _score


def _find_clips(root: Path) -> list[Path]:
    if (root / "direct_whisperx.json").exists() or (root / "realtime_project.json").exists():
        return [root]
    out = []
    for sub in sorted(root.rglob("*")):
        if sub.is_dir() and ((sub / "direct_whisperx.json").exists() or (sub / "realtime_project.json").exists()):
            out.append(sub)
    return out


def main(argv: Optional[list[str]] = None) -> int:
    default_input_root = Path(__file__).resolve().parents[2] / "src/tests/compare_whisperx_test/input"
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--unit", choices=("auto", "char", "word"), default="auto")
    ap.add_argument("--input-root", type=Path, default=default_input_root)
    ap.add_argument("--reference", type=Path, default=None, help="explicit spk_subtitles (single clip only)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    clips: list[Path] = []
    for d in args.dirs:
        if d.exists():
            clips.extend(_find_clips(d))
        else:
            print(f"[skip] {d} (not found)")
    if not clips:
        print("No case dirs with direct_whisperx.json / realtime_project.json found.")
        return 1

    report: dict[str, dict] = {}
    for clip in clips:
        name = clip.name
        ref_path = args.reference if (args.reference and len(clips) == 1) else (args.input_root / name / "spk_subtitles")
        print(f"\n=== {name}")
        if not ref_path.exists():
            print(f"  (no ground truth at {ref_path})")
            continue
        unit = args.unit if args.unit != "auto" else _detect_unit(ref_path.read_text(encoding="utf-8"))
        ref = _ref_tokens(ref_path, unit)
        report[name] = {"unit": unit, "reference": str(ref_path)}
        for label, loader, fname in (
            ("direct", _cand_tokens_direct, "direct_whisperx.json"),
            ("realtime", _cand_tokens_realtime, "realtime_project.json"),
        ):
            fpath = clip / fname
            if not fpath.exists():
                print(f"  {label}: (missing {fname})")
                continue
            res = _score(loader(fpath, unit), ref)
            report[name][label] = res
            cnt_flag = ""
            if res["pred_speakers"] < res["ref_speakers"]:
                cnt_flag = " [MERGED/under-segmented]"
            elif res["pred_speakers"] > res["ref_speakers"]:
                cnt_flag = " [OVER-SPLIT]"
            print(
                f"  {label:8} unit={unit} ref_spk={res['ref_speakers']} pred_spk={res['pred_speakers']}{cnt_flag}"
                f"  spk_acc={res['speaker_accuracy']*100:5.1f}%  (aligned {res['aligned_tokens']}/{res['ref_tokens']} ref toks)"
            )
            mp = ", ".join(f"{p}->{r}" for p, r in sorted(res["mapping"].items()))
            print(f"             map: {mp}")
            for r in sorted(res["ref_recall"], key=lambda x: res["ref_token_share"].get(x, 0), reverse=True):
                if r == "<none>":
                    continue
                print(f"             ref {r}: recall={res['ref_recall'][r]*100:5.1f}%  share={res['ref_token_share'][r]*100:4.1f}%")

    if args.out:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
