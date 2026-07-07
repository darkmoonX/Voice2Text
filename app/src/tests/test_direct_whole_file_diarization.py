"""Round 0045: direct whole-file diarization assigns globally-consistent speakers.

Validates run_direct_transcription's whole_file_diarization path with a fake transcriber
(no GPU / pyannote): per-chunk diarization is suppressed during the ASR loop and a single
whole-file pass stamps every token by absolute-time overlap, bypassing the profile
re-cluster. Also checks the legacy path still calls reconcile.
"""
from __future__ import annotations

import unittest

from voice2text.capture import AudioChunk
from voice2text.config import RuntimeConfig
from voice2text.pipeline import direct_transcription as dt


class _FakeTranscriber:
    """Emits one token per transcribe() call spanning the chunk, with no speaker."""

    def __init__(self, *, support_whole_file: bool = True) -> None:
        self._support = support_whole_file
        self.suppressed_history: list[bool] = []
        self._last_meta: dict = {}
        self.reconcile_called = False
        self.whole_file_called = False

    # capability + suppression seam
    def supports_whole_file_diarization(self) -> bool:
        return self._support

    def set_diarization_suppressed(self, flag: bool) -> None:
        self.suppressed_history.append(bool(flag))

    def diarize_whole_file_turns(self, chunk, channel_mode="mono"):
        self.whole_file_called = True
        # Two global speakers split at 5.0s over a 10s clip.
        return [
            {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
            {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_01"},
        ]

    # ASR seam
    def transcribe(self, chunk, language=None, channel_mode="mono") -> str:
        dur = dt.audio_duration_seconds(chunk)
        # one word token spanning the (chunk-relative) span, no speaker fields
        self._last_meta = {
            "token_timestamps": [
                {"start": 0.0, "end": dur, "score": 1.0, "word": "w"},
            ],
            "detected_language": "zh",
            "alignment_language": "zh",
        }
        return "w"

    def get_last_transcription_meta(self) -> dict:
        return self._last_meta

    def reconcile_speaker_profiles(self, **kwargs):
        self.reconcile_called = True
        return {"status": "ok", "merged_count": 0, "remap": {}}


def _silence_chunk(seconds: float, sample_rate: int = 16000) -> AudioChunk:
    return AudioChunk(pcm16=b"\x00\x00" * int(seconds * sample_rate), sample_rate=sample_rate, channels=1)


def _cfg() -> RuntimeConfig:
    cfg = RuntimeConfig()
    cfg.stt_provider = "whisperx"
    cfg.source_language = "zh"
    cfg.source_channel_mode = "mono"
    return cfg


class DirectWholeFileDiarizationTests(unittest.TestCase):
    def test_whole_file_assigns_global_speakers_by_time(self):
        fake = _FakeTranscriber()
        # 10s audio, chunked at 5s -> two 5s chunks -> two tokens at abs [0,5] and [5,10].
        result = dt.run_direct_transcription(
            _cfg(),
            _silence_chunk(10.0),
            transcriber=fake,
            chunk_seconds=5.0,
            whole_file_diarization=True,
        )
        meta = result["meta"]
        toks = meta["token_timestamps"]
        self.assertEqual(len(toks), 2)
        # first chunk token (abs 0-5) -> SPEAKER_00; second (abs 5-10) -> SPEAKER_01
        self.assertEqual(toks[0]["speaker"], "SPEAKER_00")
        self.assertEqual(toks[0]["profile_speaker"], "SPEAKER_00")
        self.assertEqual(toks[0]["local_speaker"], "SPEAKER_00")
        self.assertEqual(toks[1]["speaker"], "SPEAKER_01")
        # suppression toggled on then off; reconcile NOT used; whole-file pass used
        self.assertEqual(fake.suppressed_history, [True, False])
        self.assertTrue(fake.whole_file_called)
        self.assertFalse(fake.reconcile_called)
        recon = meta["speaker_profile_reconciliation"]
        self.assertEqual(recon["status"], "whole_file_diarization")
        self.assertEqual(recon["speaker_count"], 2)
        self.assertEqual(recon["tokens_assigned"], 2)

    def test_legacy_path_calls_reconcile_and_no_suppression(self):
        fake = _FakeTranscriber()
        result = dt.run_direct_transcription(
            _cfg(),
            _silence_chunk(6.0),
            transcriber=fake,
            chunk_seconds=0.0,
            whole_file_diarization=False,
        )
        self.assertTrue(fake.reconcile_called)
        self.assertFalse(fake.whole_file_called)
        self.assertEqual(fake.suppressed_history, [])
        self.assertEqual(
            result["meta"]["speaker_profile_reconciliation"]["status"], "ok"
        )

    def test_unsupported_transcriber_falls_back_to_legacy(self):
        fake = _FakeTranscriber(support_whole_file=False)
        dt.run_direct_transcription(
            _cfg(),
            _silence_chunk(4.0),
            transcriber=fake,
            chunk_seconds=0.0,
            whole_file_diarization=True,
        )
        # even though requested, capability gate routes to legacy reconcile
        self.assertTrue(fake.reconcile_called)
        self.assertFalse(fake.whole_file_called)
        self.assertEqual(fake.suppressed_history, [])


if __name__ == "__main__":
    unittest.main()
