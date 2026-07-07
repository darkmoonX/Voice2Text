"""Round 0024 Phase A: synthesized word timestamps for the alignment-off path.

With forced alignment off, WhisperX segments carry [start, end] but no words, so the assembler's
timestamp-driven dedup would be bypassed and overlapping windows would duplicate. The provider
synthesizes per-word pseudo-timestamps from the segment span; the same utterance in two overlapping
windows then lands at the same absolute time (once the assembler adds `elapsed`), so the existing
word-state dedup collapses it. These tests cover the pure synthesis helper + the dedup end-to-end.
"""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice2text.pipeline.subtitle_assembler import SubtitleAssembler
from voice2text.stt.whisperx_provider import WhisperXTranscriber


_synth = WhisperXTranscriber._synthesize_segment_word_timestamps


class SynthesizeWordTimestampsTests(unittest.TestCase):
    def test_distributes_words_across_segment_span(self):
        # Uniform split across the segment span (the char-weighted variant was tested and reverted —
        # it measured slightly worse CER on the CNN clip, see round 0024 follow-up).
        rows = _synth({"text": "hello there world", "start": 1.0, "end": 4.0})
        self.assertEqual([r["word"] for r in rows], ["hello", "there", "world"])
        self.assertAlmostEqual(rows[0]["start"], 1.0, places=6)
        self.assertAlmostEqual(rows[0]["end"], 2.0, places=6)
        self.assertAlmostEqual(rows[1]["start"], 2.0, places=6)
        self.assertAlmostEqual(rows[2]["end"], 4.0, places=6)
        self.assertTrue(all(r["score"] == 1.0 for r in rows))

    def test_same_segment_in_overlapping_windows_gets_same_absolute_time(self):
        # Window A (elapsed 0): the utterance sits at relative [3, 5].
        a = _synth({"text": "look for evos", "start": 3.0, "end": 5.0})
        # Window B (elapsed 2): the same audio is now relative [1, 3]; +elapsed=2 -> absolute [3, 5].
        b = _synth({"text": "look for evos", "start": 1.0, "end": 3.0})
        a_abs = [(r["start"] + 0.0, r["end"] + 0.0) for r in a]
        b_abs = [(r["start"] + 2.0, r["end"] + 2.0) for r in b]
        for (as_, ae), (bs, be) in zip(a_abs, b_abs):
            self.assertAlmostEqual(as_, bs, places=6)
            self.assertAlmostEqual(ae, be, places=6)

    def test_cjk_splits_per_character(self):
        rows = _synth({"text": "你好世界", "start": 0.0, "end": 4.0})
        self.assertEqual([r["word"] for r in rows], ["你", "好", "世", "界"])

    def test_empty_or_bad_timing_returns_empty(self):
        self.assertEqual(_synth({"text": "", "start": 0.0, "end": 1.0}), [])
        self.assertEqual(_synth({"text": "hi", "start": 1.0, "end": 1.0}), [])
        self.assertEqual(_synth({"text": "hi", "start": None, "end": 1.0}), [])


class DedupViaSyntheticTimestampsTests(unittest.TestCase):
    """End-to-end: synthetic timestamps feed the normal word-state dedup, collapsing recurrence."""

    @staticmethod
    def _meta(rows, elapsed):
        return {"elapsed_seconds": float(elapsed), "token_timestamps": rows}

    def _merge(self, asm, rows, elapsed):
        out = asm.merge_incremental_text(
            " ".join(str(r["word"]) for r in rows),
            overlap_merge_method="exact",
            segment_seconds=10.0,
            hop_seconds=2.0,
            transcription_meta=self._meta(rows, elapsed),
        )
        return out

    def test_overlapping_windows_collapse_not_duplicate(self):
        asm = SubtitleAssembler()
        # The same utterance recurs in overlapping windows at the same absolute audio time.
        seg = {"text": "look for evos insta", "start": 3.0, "end": 6.0}
        last = ""
        # window 0: utterance at relative [3,6] (absolute [3,6])
        last = self._merge(asm, _synth(seg), elapsed=0.0) or last
        # window 1 (elapsed 2): same audio now relative [1,4] -> absolute [3,6]
        last = self._merge(asm, _synth({**seg, "start": 1.0, "end": 4.0}), elapsed=2.0) or last
        # window 2 (elapsed 4): same audio now relative [0,2] -> absolute [4,6]
        last = self._merge(asm, _synth({**seg, "start": 0.0, "end": 2.0}), elapsed=4.0) or last
        final = asm.finalize_text() if hasattr(asm, "finalize_text") else last
        text = str(final or last)
        # "evos" must not stack once-per-window; the timestamp dedup keeps it to a single occurrence.
        self.assertLessEqual(text.lower().count("evos"), 1)


if __name__ == "__main__":
    unittest.main()
