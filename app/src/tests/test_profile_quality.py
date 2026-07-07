"""Round 0023 Phase A: learn-path clip quality gate (pure heuristics)."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice2text.stt.profile_quality import ClipQualityConfig, evaluate_clip_quality


def _evaluate(text, *, scores=None, duration=3.0, **overrides):
    config = ClipQualityConfig(enabled=True, **overrides)
    return evaluate_clip_quality(text=text, word_scores=scores, duration_seconds=duration, config=config)


class GateDisabledTests(unittest.TestCase):
    def test_disabled_is_passthrough(self):
        config = ClipQualityConfig(enabled=False)
        # Even obvious garbage passes when the gate is off (no-op until explicitly enabled).
        result = evaluate_clip_quality(text="[Music]", word_scores=[0.0], duration_seconds=3.0, config=config)
        self.assertTrue(result.ok)
        self.assertEqual(result.reasons, [])
        self.assertEqual(result.score, 1.0)


class AcceptCleanSpeechTests(unittest.TestCase):
    def test_clean_english_accepted(self):
        result = _evaluate("hello everyone welcome to the show", scores=[0.9, 0.85, 0.8, 0.92])
        self.assertTrue(result.ok, result.reasons)
        self.assertGreater(result.score, 0.0)

    def test_clean_cjk_accepted(self):
        result = _evaluate("大家好今天我們來聊聊這個主題", scores=[0.8, 0.75, 0.9])
        self.assertTrue(result.ok, result.reasons)

    def test_short_real_word_accepted(self):
        # A single real CJK word should not be rejected as too-short.
        result = _evaluate("對", scores=[0.8])
        self.assertTrue(result.ok, result.reasons)

    def test_missing_scores_does_not_reject(self):
        # No word scores available -> confidence check is skipped, not failed.
        result = _evaluate("a normal sentence here", scores=None)
        self.assertTrue(result.ok, result.reasons)


class RejectLowQualityTests(unittest.TestCase):
    def test_empty_text_rejected(self):
        result = _evaluate("   ", scores=[0.9])
        self.assertFalse(result.ok)
        self.assertIn("empty_text", result.reasons)
        self.assertEqual(result.score, 0.0)

    def test_music_tag_rejected(self):
        for tag in ("[Music]", "(applause)", "[背景音乐]", "音乐"):
            result = _evaluate(tag, scores=[0.9])
            self.assertFalse(result.ok, tag)
            self.assertIn("non_speech_tag", result.reasons)

    def test_music_glyph_rejected(self):
        result = _evaluate("♪ la la la ♪", scores=[0.9])
        self.assertFalse(result.ok)
        self.assertIn("non_speech_tag", result.reasons)

    def test_repetitive_rejected(self):
        result = _evaluate("啦啦啦啦啦啦啦啦", scores=[0.9])
        self.assertFalse(result.ok)
        self.assertIn("repetitive", result.reasons)

    def test_low_confidence_rejected(self):
        result = _evaluate("some words here", scores=[0.2, 0.1, 0.3], min_confidence=0.45)
        self.assertFalse(result.ok)
        self.assertIn("low_confidence", result.reasons)

    def test_confidence_threshold_edge(self):
        # Mean exactly at threshold passes (>=), just below fails.
        at = _evaluate("words go here", scores=[0.45, 0.45, 0.45], min_confidence=0.45)
        self.assertTrue(at.ok, at.reasons)
        below = _evaluate("words go here", scores=[0.44, 0.44, 0.44], min_confidence=0.45)
        self.assertFalse(below.ok)


if __name__ == "__main__":
    unittest.main()
