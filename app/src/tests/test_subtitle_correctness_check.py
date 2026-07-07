"""Round 0070 — dup-stacking detector rules in subtitle_correctness_check."""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "diagnostics" / "subtitle_correctness_check.py"
    spec = importlib.util.spec_from_file_location("subtitle_correctness_check", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["subtitle_correctness_check"] = module
    spec.loader.exec_module(module)
    return module


scc = _load_module()


class MaxAdjacentRepeatTests(unittest.TestCase):
    def test_real_phrase_stacking_detected(self) -> None:
        phrase = "今天天氣真的很好"  # 8 chars, mixed
        self.assertEqual(scc._max_adjacent_repeat(phrase + phrase), phrase)

    def test_monotone_laughter_run_not_flagged(self) -> None:
        # 64x 哈 is legitimate laughter transcription, not cross-window stacking
        # (round 0070 founding-baseline false positive on whispercpp+diar aXqBR_2).
        self.assertEqual(scc._max_adjacent_repeat("哈" * 64), "")

    def test_monotone_run_embedded_in_text_not_flagged(self) -> None:
        self.assertEqual(scc._max_adjacent_repeat("大家都在準備好了" + "哈" * 30 + "我感覺無所謂"), "")

    def test_stacking_next_to_monotone_run_still_detected(self) -> None:
        phrase = "重複的內容又出現"
        self.assertEqual(scc._max_adjacent_repeat("哈" * 20 + phrase + phrase), phrase)

    def test_whitespace_ignored_when_matching(self) -> None:
        self.assertEqual(scc._max_adjacent_repeat("abcd efgh abcdefgh"), "abcdefgh")

    def test_clean_text_returns_empty(self) -> None:
        self.assertEqual(scc._max_adjacent_repeat("平凡無奇的一段正常字幕文字內容"), "")


class CompletenessFloorTests(unittest.TestCase):
    def _case_dir(self, td: str, realtime_chars: int, direct_chars: int) -> Path:
        d = Path(td)
        (d / "realtime_project.txt").write_text("字" * realtime_chars, encoding="utf-8")
        (d / "direct_whisperx_nospk.txt").write_text("字" * direct_chars, encoding="utf-8")
        return d

    def test_default_floor_fails_at_80_percent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            result = scc.check(self._case_dir(td, 80, 100))
            self.assertFalse(result["ok"])
            self.assertEqual(result["min_completeness"], 0.85)

    def test_lowered_floor_passes_known_hard_clip_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            result = scc.check(self._case_dir(td, 80, 100), min_completeness=0.75)
            self.assertTrue(result["ok"])
            self.assertEqual(result["min_completeness"], 0.75)


if __name__ == "__main__":
    unittest.main()
