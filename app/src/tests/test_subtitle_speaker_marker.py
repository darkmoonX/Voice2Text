"""Unit tests for speaker marker rendering in subtitle assembly."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice2text.pipeline.subtitle_assembler import SubtitleAssembler, _WordState


class SubtitleSpeakerMarkerTests(unittest.TestCase):
    @staticmethod
    def _word(word: str, start: float, end: float, speaker: str) -> _WordState:
        return _WordState(
            word=word,
            start=start,
            end=end,
            score=0.95,
            count=3,
            last_seen=end,
            speaker=speaker,
        )

    def test_marker_uses_spk_label_on_speaker_change_by_default(self) -> None:
        assembler = SubtitleAssembler()
        words = [
            self._word("hello", 0.00, 0.10, "SPEAKER_00"),
            self._word("there", 0.10, 0.22, "SPEAKER_00"),
            self._word("general", 0.22, 0.34, "SPEAKER_01"),
            self._word("kenobi", 0.34, 0.47, "SPEAKER_01"),
        ]

        text = assembler._words_to_text(words)
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        self.assertEqual(len(lines), 2, text)
        self.assertTrue(lines[0].startswith("[spk_000] "), text)
        self.assertTrue(lines[1].startswith("[spk_001] "), text)

    def test_marker_can_use_double_arrow_style(self) -> None:
        assembler = SubtitleAssembler()
        assembler.set_speaker_marker_style("arrow")
        words = [
            self._word("hello", 0.00, 0.10, "SPEAKER_00"),
            self._word("there", 0.10, 0.22, "SPEAKER_00"),
            self._word("general", 0.22, 0.34, "SPEAKER_01"),
            self._word("kenobi", 0.34, 0.47, "SPEAKER_01"),
        ]

        text = assembler._words_to_text(words)
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        self.assertEqual(len(lines), 2, text)
        self.assertTrue(lines[0].startswith(">> "), text)
        self.assertTrue(lines[1].startswith(">> "), text)

    def test_tiny_aba_speaker_blip_is_smoothed_before_marker_render(self) -> None:
        assembler = SubtitleAssembler()
        words = [
            self._word("我", 0.00, 0.20, "SPEAKER_00"),
            self._word("們", 0.20, 0.40, "SPEAKER_00"),
            self._word("啊", 0.40, 0.56, "SPEAKER_01"),
            self._word("呃", 0.56, 0.72, "SPEAKER_01"),
            self._word("繼", 0.72, 0.92, "SPEAKER_00"),
            self._word("續", 0.92, 1.12, "SPEAKER_00"),
        ]

        text = assembler._words_to_text(words)
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        self.assertEqual(len(lines), 1, text)
        self.assertTrue(lines[0].startswith("[spk_000] "), text)
        self.assertNotIn("[spk_001]", text)

    def test_tiny_trailing_speaker_blip_is_smoothed_to_previous_speaker(self) -> None:
        assembler = SubtitleAssembler()
        words = [
            self._word("我", 0.00, 0.20, "SPEAKER_00"),
            self._word("們", 0.20, 0.40, "SPEAKER_00"),
            self._word("繼", 0.40, 0.60, "SPEAKER_00"),
            self._word("續", 0.60, 0.80, "SPEAKER_00"),
            self._word("啊", 0.80, 0.94, "SPEAKER_01"),
            self._word("呃", 0.94, 1.08, "SPEAKER_01"),
        ]

        text = assembler._words_to_text(words)
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        self.assertEqual(len(lines), 1, text)
        self.assertTrue(lines[0].startswith("[spk_000] "), text)
        self.assertNotIn("[spk_001]", text)

    def test_short_new_speaker_run_can_still_switch_when_long_enough(self) -> None:
        assembler = SubtitleAssembler()
        words = [
            self._word("hello", 0.00, 0.20, "SPEAKER_00"),
            self._word("there", 0.20, 0.40, "SPEAKER_00"),
            self._word("new", 0.40, 0.90, "SPEAKER_01"),
            self._word("speaker", 0.90, 1.35, "SPEAKER_01"),
        ]

        text = assembler._words_to_text(words)
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        self.assertEqual(len(lines), 2, text)
        self.assertTrue(lines[1].startswith("[spk_001] "), text)


if __name__ == "__main__":
    unittest.main()
