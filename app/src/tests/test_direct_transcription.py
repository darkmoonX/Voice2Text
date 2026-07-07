from __future__ import annotations

from pathlib import Path
import sys
import unittest

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice2text.capture import AudioChunk
from voice2text.config import RuntimeConfig
from voice2text.pipeline.direct_transcription import (
    decode_to_wav_16k_mono,
    resolve_ffmpeg,
    run_direct_transcription,
)


def _audio(seconds: float) -> AudioChunk:
    sample_rate = 16000
    frames = int(sample_rate * seconds)
    return AudioChunk(pcm16=b"\0\0" * frames, sample_rate=sample_rate, channels=1)


class FakeDirectTranscriber:
    def __init__(self) -> None:
        self.calls: list[tuple[float, str | None]] = []
        self._last_meta: dict[str, object] = {}

    def transcribe(self, chunk: AudioChunk, language: str | None = None, channel_mode: str = "mono") -> str:
        duration = len(chunk.pcm16) / float(chunk.sample_rate * chunk.channels * 2)
        self.calls.append((duration, language))
        index = len(self.calls)
        self._last_meta = {
            "detected_language": "en",
            "alignment_language": "en",
            "token_timestamps": [
                {
                    "word": f"w{index}",
                    "start": 0.1,
                    "end": 0.4,
                    "score": 0.9,
                    "speaker": "SPEAKER_00",
                }
            ],
        }
        return f"text{index}"

    def get_last_transcription_meta(self) -> dict[str, object]:
        return dict(self._last_meta)

    def reconcile_speaker_profiles(self, *, threshold: float = 0.0) -> dict[str, object]:
        return {"status": "ok", "merged_count": 1, "remap": {"SPEAKER_00": "SPK_000"}, "threshold": threshold}


class DirectTranscriptionTests(unittest.TestCase):
    def test_single_pass_combines_text_meta_and_speaker_remap(self) -> None:
        cfg = RuntimeConfig(source_language="en")
        transcriber = FakeDirectTranscriber()

        result = run_direct_transcription(
            cfg,
            _audio(2.0),
            transcriber=transcriber,
            chunk_seconds=0.0,
            speaker_profile_reconcile_threshold=0.5,
        )

        self.assertEqual(result["text"], "text1")
        meta = result["meta"]
        self.assertIsInstance(meta, dict)
        self.assertEqual(meta["direct_chunk_count"], 1)
        self.assertEqual(meta["direct_chunk_seconds"], 0.0)
        rows = meta["token_timestamps"]
        self.assertEqual(rows[0]["absolute_start"], 0.1)
        self.assertEqual(rows[0]["absolute_end"], 0.4)
        self.assertEqual(rows[0]["speaker"], "SPK_000")
        self.assertEqual(meta["speaker_profile_reconciliation"]["threshold"], 0.5)

    def test_chunked_language_auto_uses_subchunks_and_reports_progress(self) -> None:
        cfg = RuntimeConfig(source_language=None)
        transcriber = FakeDirectTranscriber()
        progress: list[tuple[float, float]] = []

        result = run_direct_transcription(
            cfg,
            _audio(5.0),
            transcriber=transcriber,
            chunk_seconds=4.0,
            language_subchunk_seconds=2.0,
            on_progress=lambda completed, total: progress.append((completed, total)),
        )

        self.assertEqual(result["text"], "text1\ntext2\ntext3")
        self.assertEqual([round(duration, 1) for duration, _language in transcriber.calls], [2.0, 2.0, 1.0])
        self.assertEqual([language for _duration, language in transcriber.calls], [None, None, None])
        rows = result["meta"]["token_timestamps"]
        self.assertEqual([round(row["absolute_start"], 1) for row in rows], [0.1, 2.1, 4.1])
        self.assertEqual(progress[-1], (5.0, 5.0))
        self.assertEqual(result["meta"]["direct_language_subchunk_seconds"], 2.0)

    def test_transcriber_without_whole_file_hooks_falls_back_to_reconcile_regardless_of_provider(self) -> None:
        # Round 0066: whole-file diarization support is decided purely by whether the
        # TRANSCRIBER implements the duck-typed hooks (supports_whole_file_diarization /
        # diarize_whole_file_turns / set_diarization_suppressed), not by provider name -
        # whisper.cpp's WhisperCppServerTranscriber now implements them when a diarizer is
        # configured (see test_whispercpp_whole_file_diarization.py). FakeDirectTranscriber
        # here has none of those hooks, so it must take the non-whole-file reconcile path
        # (its own reconcile_speaker_profiles) even with stt_provider=whispercpp - there is
        # no more hardcoded whispercpp-specific degrade message.
        cfg = RuntimeConfig(stt_provider="whispercpp", source_language="en")
        statuses: list[str] = []

        result = run_direct_transcription(
            cfg,
            _audio(1.0),
            transcriber=FakeDirectTranscriber(),
            chunk_seconds=0.0,
            on_status=statuses.append,
        )

        self.assertFalse(any("whispercpp has no diarization" in item for item in statuses))
        self.assertEqual(result["meta"]["speaker_profile_reconciliation"]["status"], "ok")


class ResolveFfmpegTests(unittest.TestCase):
    def test_prefers_configured_dir_over_path(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "ffmpeg.exe"
            exe.write_bytes(b"")
            resolved = resolve_ffmpeg(str(tmp))
            self.assertEqual(resolved, str(exe))

    def test_falls_back_to_path_when_dir_empty_or_missing(self) -> None:
        # Empty or missing configured dir -> resolution comes from shutil.which, not a
        # baked-in constant. Mock which so the assertion is host-independent.
        import voice2text.pipeline.direct_transcription as dt

        original = dt.shutil.which
        try:
            dt.shutil.which = lambda name: "/opt/ffmpeg" if name == "ffmpeg" else None
            self.assertEqual(resolve_ffmpeg(""), "/opt/ffmpeg")
            self.assertEqual(
                resolve_ffmpeg(str(Path(__file__).parent / "no_such_dir")), "/opt/ffmpeg"
            )
        finally:
            dt.shutil.which = original

    def test_wav_input_is_passed_through_without_ffmpeg(self) -> None:
        # .wav short-circuits decode (no ffmpeg needed) and returns the path unchanged.
        wav = Path(__file__)  # any existing path with a non-.wav suffix would decode; use a .wav name
        wav_path = wav.with_suffix(".wav")
        self.assertEqual(decode_to_wav_16k_mono(wav_path), wav_path)


if __name__ == "__main__":
    unittest.main()
