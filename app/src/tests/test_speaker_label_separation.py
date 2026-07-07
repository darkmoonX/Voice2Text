"""Round 0027: visible/raw/profile speaker labels are exported separately (additive, metric-neutral)."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice2text.pipeline.transcript_exporter import TranscriptExportOptions, TranscriptExporterSession


def _options(tmp, *, include_speaker=True):
    return TranscriptExportOptions(
        enabled=True,
        formats=["json"],
        include_timestamps=True,
        include_speaker=include_speaker,
        output_dir=str(tmp),
        include_confidence=False,
    )


def _token(word, start, end, *, profile, local, score=0.9):
    # Mirrors the provider token rows: `speaker` == profile-preferred, plus the raw `local_speaker`.
    return {
        "word": word,
        "absolute_start": start,
        "absolute_end": end,
        "score": score,
        "speaker": profile,
        "profile_speaker": profile,
        "local_speaker": local,
    }


def _record(session, tokens):
    text = " ".join(str(t["word"]) for t in tokens)
    session.record(raw_text=text, source_text=text, translated_text="", meta={"elapsed_seconds": 0.0, "token_timestamps": tokens})


def _first_cue(session, tmp) -> dict:
    out = Path(tmp) / "labels.json"
    session.export_single_file(output_path=str(out), format_hint="json")
    data = json.loads(out.read_text(encoding="utf-8"))
    return data["cues"][0]


class LabelSeparationTests(unittest.TestCase):
    def test_three_labels_recorded_distinctly(self):
        # Diarization said spk_1 (raw), profile mapped it to spk_0; the rendered/visible marker is spk_0.
        with tempfile.TemporaryDirectory() as tmp:
            s = TranscriptExporterSession(_options(tmp))
            _record(
                s,
                [
                    _token("hello", 0.0, 0.4, profile="spk_0", local="spk_1"),
                    _token("world", 0.4, 0.8, profile="spk_0", local="spk_1"),
                ],
            )
            cue = _first_cue(s, tmp)
            self.assertEqual(cue["speaker"], "spk_000")          # effective (unchanged)
            self.assertEqual(cue["visible_speaker"], "spk_000")  # rendered marker
            self.assertEqual(cue["profile_speaker"], "spk_000")  # cross-window identity
            self.assertEqual(cue["raw_speaker"], "spk_001")      # local per-window diarization

    def test_raw_dominant_within_bucket(self):
        # The raw label is the per-bucket majority, independent of the effective speaker.
        with tempfile.TemporaryDirectory() as tmp:
            s = TranscriptExporterSession(_options(tmp))
            _record(
                s,
                [
                    _token("a", 0.0, 0.3, profile="spk_0", local="spk_2"),
                    _token("b", 0.3, 0.6, profile="spk_0", local="spk_2"),
                    _token("c", 0.6, 0.9, profile="spk_0", local="spk_5"),
                ],
            )
            cue = _first_cue(s, tmp)
            self.assertEqual(cue["profile_speaker"], "spk_000")
            self.assertEqual(cue["raw_speaker"], "spk_002")  # majority of the local labels

    def test_srt_txt_use_only_effective_speaker(self):
        # The detail fields are json-only; SRT/TXT must render the effective marker, nothing leaked.
        with tempfile.TemporaryDirectory() as tmp:
            s = TranscriptExporterSession(_options(tmp))
            _record(s, [_token("hi", 0.0, 0.5, profile="spk_0", local="spk_1")])
            txt = (Path(tmp) / "out.txt")
            s.export_single_file(output_path=str(txt), format_hint="txt")
            body = txt.read_text(encoding="utf-8")
            self.assertIn("[spk_000]", body)
            self.assertNotIn("spk_001", body)  # raw label never appears in txt

    def test_detail_gated_off_when_speaker_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = TranscriptExporterSession(_options(tmp, include_speaker=False))
            _record(s, [_token("hi", 0.0, 0.5, profile="spk_0", local="spk_1")])
            cue = _first_cue(s, tmp)
            self.assertNotIn("visible_speaker", cue)
            self.assertNotIn("raw_speaker", cue)
            self.assertNotIn("profile_speaker", cue)


if __name__ == "__main__":
    unittest.main()
