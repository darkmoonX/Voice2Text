"""Unit tests for WhisperX speaker-turn switching behavior."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice2text.stt.whisperx_provider import WhisperXTranscriber


class WhisperXSpeakerTurnTests(unittest.TestCase):
    def _new_transcriber_stub(self) -> WhisperXTranscriber:
        inst = WhisperXTranscriber.__new__(WhisperXTranscriber)
        inst._last_speaker_label = None
        inst._speaker_display_map = {}
        inst._speaker_display_next_index = 0
        inst._speaker_marker_style = "spk"
        inst._speaker_switch_confirm_segments = 2
        inst._speaker_switch_min_duration_seconds = 0.18
        inst._speaker_switch_single_segment_min_duration_seconds = 0.25
        inst._speaker_switch_pending_label = ""
        inst._speaker_switch_pending_count = 0
        inst._speaker_switch_pending_duration_seconds = 0.0
        inst._speaker_pause_break_seconds = 1.8
        inst._last_speaker_segment_end = None
        inst._trace_enabled = False
        inst._emit = lambda _msg: None
        return inst

    @staticmethod
    def _seg(text: str, speaker: str, start: float, end: float) -> dict[str, object]:
        return {"text": text, "speaker": speaker, "start": start, "end": end}

    def test_switch_confirms_across_single_segment_windows(self) -> None:
        transcriber = self._new_transcriber_stub()

        out_a = transcriber._format_diarized_text([self._seg("hello", "SPEAKER_00", 0.00, 0.30)])
        out_b1 = transcriber._format_diarized_text([self._seg("world", "SPEAKER_01", 0.00, 0.30)])

        self.assertTrue(out_a.startswith("[spk_000] "), out_a)
        self.assertTrue(out_b1.startswith("[spk_001] "), out_b1)
        self.assertEqual(transcriber._last_speaker_label, "SPEAKER_01")
        self.assertEqual(transcriber._speaker_switch_pending_count, 0)
        self.assertEqual(transcriber._speaker_switch_pending_label, "")

    def test_very_short_spurious_segment_does_not_confirm_switch(self) -> None:
        transcriber = self._new_transcriber_stub()

        transcriber._format_diarized_text([self._seg("alpha", "SPEAKER_00", 0.00, 0.30)])
        out = transcriber._format_diarized_text([self._seg("click", "SPEAKER_01", 0.00, 0.08)])

        self.assertEqual(out, "click")
        self.assertEqual(transcriber._last_speaker_label, "SPEAKER_00")
        self.assertEqual(transcriber._speaker_switch_pending_label, "SPEAKER_01")

    def test_pending_candidate_clears_when_old_speaker_returns(self) -> None:
        transcriber = self._new_transcriber_stub()

        transcriber._format_diarized_text([self._seg("alpha", "SPEAKER_00", 0.00, 0.30)])
        transcriber._format_diarized_text([self._seg("beta", "SPEAKER_01", 0.00, 0.08)])
        out = transcriber._format_diarized_text([self._seg("gamma", "SPEAKER_00", 0.00, 0.30)])

        self.assertEqual(out, "gamma")
        self.assertEqual(transcriber._last_speaker_label, "SPEAKER_00")
        self.assertEqual(transcriber._speaker_switch_pending_label, "")
        self.assertEqual(transcriber._speaker_switch_pending_count, 0)

    def test_single_long_segment_can_confirm_switch(self) -> None:
        transcriber = self._new_transcriber_stub()

        transcriber._format_diarized_text([self._seg("alpha", "SPEAKER_00", 0.00, 0.30)])
        out = transcriber._format_diarized_text([self._seg("beta", "SPEAKER_01", 0.00, 0.55)])

        self.assertTrue(out.startswith("[spk_001] "), out)
        self.assertEqual(transcriber._last_speaker_label, "SPEAKER_01")
        self.assertEqual(transcriber._speaker_switch_pending_label, "")
        self.assertEqual(transcriber._speaker_switch_pending_count, 0)

    def test_turn_detection_prefers_local_speaker_over_profile_speaker(self) -> None:
        transcriber = self._new_transcriber_stub()

        transcriber._format_diarized_text(
            [{"text": "alpha", "speaker": "SPEAKER_00", "profile_speaker": "SPK_000", "start": 0.0, "end": 0.40}]
        )
        out = transcriber._format_diarized_text(
            [{"text": "beta", "speaker": "SPEAKER_01", "profile_speaker": "SPK_000", "start": 0.0, "end": 0.55}]
        )

        self.assertTrue(out.startswith("[spk_001] "), out)
        self.assertEqual(transcriber._last_speaker_label, "SPEAKER_01")

    def test_profile_identity_display_suppresses_unconfirmed_local_candidate(self) -> None:
        transcriber = self._new_transcriber_stub()
        transcriber._enable_diarization = True
        transcriber._speaker_profile_enabled = True
        transcriber._speaker_identity_engine = object()
        transcriber._last_speaker_profile_stats = {"status": "done_no_assignment"}

        out = transcriber._format_diarized_text(
            [{"text": "music tail", "speaker": "SPEAKER_01", "start": 0.0, "end": 0.40}]
        )

        self.assertEqual(out, "music tail")
        self.assertIsNone(transcriber._last_speaker_label)

    def test_token_metadata_can_prefer_profile_speaker_for_export_stability(self) -> None:
        speaker = WhisperXTranscriber._resolve_word_speaker(
            {"word": "beta", "speaker": "SPEAKER_01", "profile_speaker": "SPK_000"},
            {"speaker": "SPEAKER_01", "profile_speaker": "SPK_000"},
            "SPK_000",
            prefer_profile=True,
        )

        self.assertEqual(speaker, "SPK_000")

    def test_token_metadata_defaults_to_local_speaker_for_turn_compatibility(self) -> None:
        speaker = WhisperXTranscriber._resolve_word_speaker(
            {"word": "beta", "profile_speaker": "SPK_000"},
            {"speaker": "SPEAKER_01", "profile_speaker": "SPK_000"},
            "SPEAKER_01",
        )

        self.assertEqual(speaker, "SPEAKER_01")

    def test_arrow_marker_style_is_still_supported(self) -> None:
        transcriber = self._new_transcriber_stub()
        transcriber._speaker_marker_style = "arrow"

        out = transcriber._format_diarized_text([self._seg("hello", "SPEAKER_00", 0.00, 0.30)])

        self.assertTrue(out.startswith(">> "), out)

    def test_same_speaker_after_long_pause_reemits_marker(self) -> None:
        transcriber = self._new_transcriber_stub()

        out = transcriber._format_diarized_text(
            [
                self._seg("alpha", "SPEAKER_00", 0.00, 0.40),
                self._seg("beta", "SPEAKER_00", 2.50, 2.90),
            ]
        )

        self.assertEqual(out, "[spk_000] alpha\n\n[spk_000] beta")

    def test_same_speaker_small_gap_stays_inline(self) -> None:
        transcriber = self._new_transcriber_stub()

        out = transcriber._format_diarized_text(
            [
                self._seg("alpha", "SPEAKER_00", 0.00, 0.40),
                self._seg("beta", "SPEAKER_00", 0.80, 1.10),
            ]
        )

        self.assertEqual(out, "[spk_000] alpha beta")

    def test_diarization_model_ref_normalizes_legacy_duplicate_slug(self) -> None:
        self.assertEqual(
            WhisperXTranscriber._normalize_diarization_model_ref(
                "pyannote/speaker-diarization-diarization-3.1"
            ),
            "pyannote/speaker-diarization-3.1",
        )


if __name__ == "__main__":
    unittest.main()
