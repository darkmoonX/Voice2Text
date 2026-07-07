"""Round 0021: confidence/stability fields in the json transcript export."""
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


def _options(tmp, *, formats=("json",), include_confidence=True, txt_confidence_annotations=False):
    return TranscriptExportOptions(
        enabled=True,
        formats=list(formats),
        include_timestamps=True,
        include_speaker=True,
        output_dir=str(tmp),
        include_confidence=include_confidence,
        txt_confidence_annotations=txt_confidence_annotations,
    )


def _token(word, start, end, score, speaker="spk_000"):
    return {
        "word": word,
        "absolute_start": start,
        "absolute_end": end,
        "score": score,
        "profile_speaker": speaker,
    }


def _record_tokens(session, tokens, *, stability_ratio=None, stable_token_count=None):
    meta = {"elapsed_seconds": 0.0, "token_timestamps": tokens}
    if stability_ratio is not None:
        meta["stability_ratio"] = stability_ratio
    if stable_token_count is not None:
        meta["stable_token_count"] = stable_token_count
    text = " ".join(str(t["word"]) for t in tokens)
    session.record(raw_text=text, source_text=text, translated_text="", meta=meta)


def _finalize_json(session, tmp):
    paths = session.finalize()
    js = [p for p in paths if p.suffix == ".json"]
    assert js, f"no json written: {paths}"
    return json.loads(js[0].read_text(encoding="utf-8"))


class CueConfidenceTests(unittest.TestCase):
    def test_cue_and_summary_confidence_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = TranscriptExporterSession(_options(tmp))
            _record_tokens(
                s,
                [
                    _token("hello", 0.0, 0.4, 0.90),
                    _token("world", 0.4, 0.8, 0.80),
                ],
                stability_ratio=1.0,
                stable_token_count=2,
            )
            data = _finalize_json(s, tmp)
            cue = data["cues"][0]
            self.assertAlmostEqual(cue["confidence"], 0.85, places=4)
            self.assertAlmostEqual(cue["min_score"], 0.80, places=4)
            self.assertAlmostEqual(cue["stable_ratio"], 1.0, places=4)
            self.assertAlmostEqual(data["summary"]["mean_confidence"], 0.85, places=4)
            self.assertAlmostEqual(data["summary"]["stable_token_ratio"], 1.0, places=4)

    def test_stable_ratio_matches_score_and_duration_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = TranscriptExporterSession(_options(tmp))
            _record_tokens(
                s,
                [
                    _token("a", 0.0, 0.4, 0.90),   # stable
                    _token("b", 0.4, 0.8, 0.40),   # low score -> unstable
                    _token("c", 0.8, 3.0, 0.95),   # 2.2s duration -> unstable
                ],
            )
            data = _finalize_json(s, tmp)
            # single cue (same speaker, gaps < 2s except c which is 0 gap from b's end 0.8)
            stable_ratios = [c.get("stable_ratio") for c in data["cues"] if "stable_ratio" in c]
            # exactly one of three tokens is stable across the export
            self.assertAlmostEqual(data["summary"]["stable_token_ratio"], 1.0 / 3.0, places=4)
            self.assertTrue(stable_ratios)

    def test_event_carries_stability_ratio(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = TranscriptExporterSession(_options(tmp))
            _record_tokens(
                s,
                [_token("hi", 0.0, 0.4, 0.9)],
                stability_ratio=0.75,
                stable_token_count=3,
            )
            data = _finalize_json(s, tmp)
            ev = data["events"][0]
            self.assertAlmostEqual(ev["stability_ratio"], 0.75, places=4)
            self.assertEqual(ev["stable_token_count"], 3)


class OptOutTests(unittest.TestCase):
    def test_include_confidence_false_omits_all_new_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = TranscriptExporterSession(_options(tmp, include_confidence=False))
            _record_tokens(
                s,
                [_token("hello", 0.0, 0.4, 0.9), _token("world", 0.4, 0.8, 0.8)],
                stability_ratio=1.0,
                stable_token_count=2,
            )
            data = _finalize_json(s, tmp)
            for cue in data["cues"]:
                self.assertNotIn("confidence", cue)
                self.assertNotIn("min_score", cue)
                self.assertNotIn("stable_ratio", cue)
            self.assertNotIn("mean_confidence", data["summary"])
            self.assertNotIn("stable_token_ratio", data["summary"])
            for ev in data["events"]:
                self.assertNotIn("stability_ratio", ev)
                self.assertNotIn("stable_token_count", ev)

    def test_txt_and_srt_identical_regardless_of_confidence(self):
        tokens = [_token("hello", 0.0, 0.4, 0.9), _token("world", 0.4, 0.8, 0.8)]
        renders = {}
        for flag in (True, False):
            with tempfile.TemporaryDirectory() as tmp:
                s = TranscriptExporterSession(_options(tmp, formats=("txt", "srt"), include_confidence=flag))
                _record_tokens(s, tokens, stability_ratio=1.0, stable_token_count=2)
                paths = s.finalize()
                renders[flag] = {
                    p.suffix: p.read_text(encoding="utf-8") for p in paths
                }
        self.assertEqual(renders[True][".txt"], renders[False][".txt"])
        self.assertEqual(renders[True][".srt"], renders[False][".srt"])


class TxtConfidenceAnnotationTests(unittest.TestCase):
    """Round 0069: optional compact `(conf=0.87)` suffix per txt line."""

    def test_disabled_by_default_txt_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = TranscriptExporterSession(_options(tmp, formats=("txt",)))
            _record_tokens(s, [_token("hello", 0.0, 0.4, 0.90), _token("world", 0.4, 0.8, 0.80)])
            paths = s.finalize()
            txt = [p for p in paths if p.suffix == ".txt"][0].read_text(encoding="utf-8")
            self.assertNotIn("conf=", txt)

    def test_enabled_appends_confidence_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = TranscriptExporterSession(_options(tmp, formats=("txt",), txt_confidence_annotations=True))
            _record_tokens(s, [_token("hello", 0.0, 0.4, 0.90), _token("world", 0.4, 0.8, 0.80)])
            paths = s.finalize()
            txt = [p for p in paths if p.suffix == ".txt"][0].read_text(encoding="utf-8")
            self.assertIn("(conf=0.85)", txt)

    def test_enabled_with_include_confidence_false_is_a_no_op(self):
        # No "confidence" key is ever attached to the cue when include_confidence=False, so the
        # annotation flag has nothing to append -> txt stays byte-identical to the disabled case.
        with tempfile.TemporaryDirectory() as tmp:
            s = TranscriptExporterSession(
                _options(tmp, formats=("txt",), include_confidence=False, txt_confidence_annotations=True)
            )
            _record_tokens(s, [_token("hello", 0.0, 0.4, 0.90), _token("world", 0.4, 0.8, 0.80)])
            paths = s.finalize()
            txt = [p for p in paths if p.suffix == ".txt"][0].read_text(encoding="utf-8")
            self.assertNotIn("conf=", txt)

    def test_srt_and_json_unaffected_by_txt_annotation_flag(self):
        tokens = [_token("hello", 0.0, 0.4, 0.90), _token("world", 0.4, 0.8, 0.80)]
        renders = {}
        for flag in (True, False):
            with tempfile.TemporaryDirectory() as tmp:
                s = TranscriptExporterSession(
                    _options(tmp, formats=("srt", "json"), txt_confidence_annotations=flag)
                )
                _record_tokens(s, tokens)
                paths = s.finalize()
                renders[flag] = {p.suffix: p.read_text(encoding="utf-8") for p in paths}
        self.assertEqual(renders[True][".srt"], renders[False][".srt"])
        # Drop the wall-clock "timestamp" field (varies run-to-run) before comparing json.
        json_true = json.loads(renders[True][".json"])
        json_false = json.loads(renders[False][".json"])
        for data in (json_true, json_false):
            for ev in data.get("events", []):
                ev.pop("timestamp", None)
        self.assertEqual(json_true, json_false)


class EventOnlyCueTests(unittest.TestCase):
    def test_event_derived_cues_omit_confidence(self):
        # No token_timestamps -> cues are built from event source text, which carry no scores.
        with tempfile.TemporaryDirectory() as tmp:
            s = TranscriptExporterSession(_options(tmp))
            s.record(raw_text="alpha beta", source_text="alpha beta", translated_text="", meta={"elapsed_seconds": 0.0})
            data = _finalize_json(s, tmp)
            self.assertTrue(data["cues"])
            for cue in data["cues"]:
                self.assertNotIn("confidence", cue)
            # summary still reports session-level fields (zero tokens)
            self.assertIn("mean_confidence", data["summary"])
            self.assertEqual(data["summary"]["mean_confidence"], 0.0)


if __name__ == "__main__":
    unittest.main()
