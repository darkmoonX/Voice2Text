"""Round 0017 — live committed|raw boundary frame (display-only)."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice2text.pipeline.subtitle_assembler import SubtitleAssembler

SEP = SubtitleAssembler._LIVE_RAW_SEPARATOR


def _merge(assembler, text, meta):
    return assembler.merge_incremental_text(
        text,
        overlap_merge_method="stable-tail",
        segment_seconds=10.0,
        hop_seconds=2.0,
        transcription_meta=meta,
    )


class LiveBoundaryTests(unittest.TestCase):
    def test_boundary_and_immediate_marker_on_raw_region(self) -> None:
        assembler = SubtitleAssembler()
        # Pre-seed committed history (what is already fully fixed).
        assembler._rolling_committed_text = "hello world"
        meta = {
            "elapsed_seconds": 5.0,
            "token_timestamps": [
                {"word": "foo", "start": 0.0, "end": 0.3, "score": 0.95, "speaker": "SPEAKER_01"},
                {"word": "bar", "start": 0.3, "end": 0.6, "score": 0.95, "speaker": "SPEAKER_01"},
            ],
        }
        clean = _merge(assembler, "foo bar", meta)
        frame = assembler.get_live_overlay_frame()

        # Overlay frame: committed, then the separator, then the raw region.
        self.assertIn(SEP, frame)
        self.assertTrue(frame.startswith("hello world"))
        self.assertIn("foo bar", frame)
        # Raw region carries the immediate (un-gated) window speaker marker.
        self.assertIn("[spk_001]", frame)

        # Export-facing clean text must carry neither the separator nor a marker.
        self.assertNotIn(SEP, clean)
        self.assertNotIn("[spk_001]", clean)

    def test_no_separator_when_no_committed_history(self) -> None:
        assembler = SubtitleAssembler()
        meta = {
            "elapsed_seconds": 0.0,
            "token_timestamps": [
                {"word": "alpha", "start": 0.0, "end": 0.3, "score": 0.95, "speaker": "SPEAKER_00"},
                {"word": "beta", "start": 0.3, "end": 0.6, "score": 0.95, "speaker": "SPEAKER_00"},
            ],
        }
        _merge(assembler, "alpha beta", meta)
        frame = assembler.get_live_overlay_frame()
        self.assertNotIn(SEP, frame)

    def test_no_separator_when_window_fully_contained(self) -> None:
        # When the merged frame adds nothing beyond committed (raw region empty)
        # there is no live edge -> no rule this frame.
        assembler = SubtitleAssembler()
        assembler._rolling_committed_text = "hello world"
        frame = assembler._project_display_script(
            assembler._compose_live_overlay_frame("hello world", [])
        )
        self.assertNotIn(SEP, frame)

    def test_clean_return_is_byte_identical_without_feature(self) -> None:
        # The decorated frame must not perturb the returned (export/translation) text:
        # build the same session twice and only read different accessors.
        def run() -> tuple[str, str]:
            a = SubtitleAssembler()
            a._rolling_committed_text = "the quick brown fox"
            meta = {
                "elapsed_seconds": 4.0,
                "token_timestamps": [
                    {"word": "jumps", "start": 0.0, "end": 0.3, "score": 0.95, "speaker": "SPEAKER_02"},
                    {"word": "over", "start": 0.3, "end": 0.6, "score": 0.95, "speaker": "SPEAKER_02"},
                ],
            }
            clean = _merge(a, "jumps over", meta)
            return clean, a.get_live_overlay_frame()

        clean, frame = run()
        self.assertNotIn(SEP, clean)
        self.assertIn(SEP, frame)
        # committed prefix preserved in both
        self.assertTrue(clean.startswith("the quick brown fox"))
        self.assertTrue(frame.startswith("the quick brown fox"))

    def test_cjk_boundary_keeps_committed_clean(self) -> None:
        assembler = SubtitleAssembler()
        assembler.set_language_context("zh")
        assembler._rolling_committed_text = "今天天氣很好"
        meta = {
            "elapsed_seconds": 6.0,
            "token_timestamps": [
                {"word": "我們", "start": 0.0, "end": 0.3, "score": 0.95, "speaker": "SPEAKER_03"},
                {"word": "出門", "start": 0.3, "end": 0.6, "score": 0.95, "speaker": "SPEAKER_03"},
            ],
        }
        clean = _merge(assembler, "我們出門", meta)
        frame = assembler.get_live_overlay_frame()
        self.assertIn(SEP, frame)
        self.assertTrue(frame.startswith("今天天氣很好"))
        self.assertIn("我們出門", frame)
        self.assertNotIn(SEP, clean)

    def test_immediate_marker_absent_without_diarization_labels(self) -> None:
        assembler = SubtitleAssembler()
        assembler._rolling_committed_text = "committed text"
        meta = {
            "elapsed_seconds": 3.0,
            "token_timestamps": [
                {"word": "raw", "start": 0.0, "end": 0.3, "score": 0.95},
                {"word": "tail", "start": 0.3, "end": 0.6, "score": 0.95},
            ],
        }
        _merge(assembler, "raw tail", meta)
        frame = assembler.get_live_overlay_frame()
        # Separator still helps even with diarization off; no marker is added.
        self.assertIn(SEP, frame)
        self.assertNotIn("[spk_", frame)


if __name__ == "__main__":
    unittest.main()
