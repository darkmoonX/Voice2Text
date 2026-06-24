"""Subprocess entry for probing whether WhisperX CUDA alignment is stable."""
from __future__ import annotations

import argparse
import json
import sys


def _probe_audio():
    import numpy as np

    sample_rate = 16000
    duration = 6.0
    samples = int(sample_rate * duration)
    t = np.arange(samples, dtype=np.float32) / float(sample_rate)
    # Deterministic non-silent signal. The probe is for the CUDA align code path,
    # not ASR accuracy.
    audio = 0.08 * np.sin(2.0 * np.pi * 220.0 * t)
    audio += 0.04 * np.sin(2.0 * np.pi * 440.0 * t)
    return audio.astype("float32")


def _probe_segments() -> list[dict[str, object]]:
    return [
        {
            "start": 0.25,
            "end": 2.75,
            "text": "cuda alignment probe first segment with several words",
        },
        {
            "start": 3.15,
            "end": 5.75,
            "text": "second segment checks word level timestamps safely",
        },
    ]


def run_probe(*, model: str, language: str, model_dir: str) -> dict[str, object]:
    import whisperx  # type: ignore

    kwargs: dict[str, object] = {
        "language_code": language,
        "device": "cuda",
    }
    if model_dir:
        kwargs["model_dir"] = model_dir
    if model:
        kwargs["model_name"] = model
    align_model, metadata = whisperx.load_align_model(**kwargs)
    result = whisperx.align(
        _probe_segments(),
        align_model,
        metadata,
        _probe_audio(),
        "cuda",
        return_char_alignments=False,
    )
    word_count = 0
    if isinstance(result, dict):
        for segment in result.get("segments", []) or []:
            if isinstance(segment, dict):
                words = segment.get("words", []) or []
                if isinstance(words, list):
                    word_count += len(words)
    return {"ok": True, "word_count": word_count}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe WhisperX CUDA alignment in an isolated process.")
    parser.add_argument("--model", default="", help="Alignment model id/path/bundle to probe.")
    parser.add_argument("--language", default="en", help="Alignment language code.")
    parser.add_argument("--model-dir", default="", help="Optional local alignment model cache directory.")
    args = parser.parse_args(argv)
    try:
        verdict = run_probe(model=str(args.model or ""), language=str(args.language or "en"), model_dir=str(args.model_dir or ""))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": type(exc).__name__}, ensure_ascii=False), flush=True)
        return 2
    print(json.dumps(verdict, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
