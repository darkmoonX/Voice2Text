from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice2text.audio_capture import AudioChunk
from voice2text.config import RuntimeConfig
from voice2text.pipeline.direct_transcription import run_direct_transcription
from voice2text.stt.whispercpp_diarization import WhisperCppDiarizer
from voice2text.stt.whispercpp_server import (
    WhisperCppServerManager,
    WhisperCppServerTranscriber,
)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"stub")
    return path


def _pcm_chunk(seconds: float = 2.0) -> AudioChunk:
    samples = [12000, -12000] * int(16000 * seconds / 2)
    pcm = b"".join(int(v).to_bytes(2, "little", signed=True) for v in samples)
    return AudioChunk(pcm16=pcm, sample_rate=16000, channels=1)


class _WholeFileRow:
    def __init__(self, start: float, end: float, speaker: str) -> None:
        self.start = start
        self.end = end
        self.speaker = speaker


class _WholeFileTurnsResult:
    def __init__(self, rows: list[_WholeFileRow]) -> None:
        self._rows = rows

    def itertuples(self, index: bool = False):
        return iter(self._rows)


class _FakePipelineInstance:
    """A callable diarization pipeline stub: `pipeline(audio, **kwargs) -> turns-result`."""

    def __init__(self, rows: list[_WholeFileRow]) -> None:
        self._rows = rows

    def __call__(self, audio, **kwargs):
        return _WholeFileTurnsResult(list(self._rows))


class _RecordingPipelineFactory:
    """Records the `device` kwarg each time a pipeline is constructed, so tests can
    assert the live and whole-file pipelines are pinned to different devices."""

    def __init__(self, rows: list[_WholeFileRow] | None = None) -> None:
        self.devices: list[str] = []
        self._rows = rows or [
            _WholeFileRow(0.0, 1.0, "SPEAKER_00"),
            _WholeFileRow(1.0, 2.0, "SPEAKER_01"),
        ]

    def __call__(self, *, model_name: str, device: str, hf_token):
        self.devices.append(device)
        return _FakePipelineInstance(self._rows)


class _FakeSpeakerIdentityEngine:
    def __init__(self) -> None:
        self.last_stats = {"enabled": True, "backend": "fake", "status": "done_assigned"}

    def prewarm(self) -> None:
        pass

    def apply(self, *, audio, segments, resolve_local_speaker):
        # Mirror test_whispercpp_diarization.py's fake: must actually set profile_speaker,
        # otherwise _require_profile_identity_for_display()'s status="done_assigned" makes
        # _finalize_word_speaker_labels require a (never-populated) profile id and blank
        # out `speaker` entirely, defeating any test that checks for a rendered marker.
        for segment in segments:
            local = resolve_local_speaker(segment)
            if not local:
                continue
            segment["profile_speaker"] = "profile_000"
            for word in segment.get("words") or []:
                word["profile_speaker"] = "profile_000"
        return segments


class _FakeClient:
    def __init__(self, payloads: list[dict] | None = None, **kwargs) -> None:
        self.payloads = list(payloads or [])

    def ready(self) -> bool:
        return True

    def infer_wav(self, wav_path: Path, *, language: str) -> dict:
        if self.payloads:
            return self.payloads.pop(0)
        return {"segments": [], "detected_language": language}


class _FakeProcess:
    returncode = None

    def poll(self):
        return None

    def terminate(self) -> None:
        self.returncode = 0

    def wait(self, timeout=None):
        return 0

    def kill(self) -> None:
        self.returncode = -9


class _FakeFallback:
    def transcribe(self, chunk, language=None, channel_mode="mono") -> str:
        return ""


def _assign_one_speaker(_diarize_segments, payload: dict) -> dict:
    """Mocked `whisperx.diarize.assign_word_speakers` for the LIVE per-window path
    (only used by tests that exercise `_attach_speaker_labels`, not the whole-file pass)."""
    segments = list(payload["segments"])
    for segment in segments:
        segment["speaker"] = "SPEAKER_00"
        for word in segment.get("words") or []:
            word["speaker"] = "SPEAKER_00"
    return {"segments": segments}


class WholeFileDiarizationUnitTests(unittest.TestCase):
    def _diarizer(self, *, diarization_device: str = "auto", pipeline_factory=None) -> WhisperCppDiarizer:
        return WhisperCppDiarizer(
            device="cpu",
            diarization_device=diarization_device,
            auto_download=False,
            pipeline_factory=pipeline_factory or _RecordingPipelineFactory(),
            speaker_identity_engine=_FakeSpeakerIdentityEngine(),
        )

    def test_too_short_audio_returns_empty_turns(self) -> None:
        dz = self._diarizer()
        turns = dz.diarize_whole_file_turns_from_audio(np.zeros((100,), dtype=np.float32))
        self.assertEqual(turns, [])

    def test_returns_sorted_turns_from_mocked_pipeline(self) -> None:
        factory = _RecordingPipelineFactory(
            rows=[
                _WholeFileRow(1.0, 2.0, "SPEAKER_01"),
                _WholeFileRow(0.0, 1.0, "SPEAKER_00"),
            ]
        )
        dz = self._diarizer(pipeline_factory=factory)
        audio = np.ones((32000,), dtype=np.float32) * 0.05
        turns = dz.diarize_whole_file_turns_from_audio(audio)
        self.assertEqual(
            turns,
            [
                {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
                {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_01"},
            ],
        )

    def test_whole_file_pipeline_is_cpu_pinned_even_when_live_device_is_cuda(self) -> None:
        factory = _RecordingPipelineFactory()
        dz = self._diarizer(diarization_device="cpu", pipeline_factory=factory)
        # Force the resolved live device to "cuda" directly (bypassing the real
        # torch.cuda probe, which always resolves to "cpu" in this CPU-only test env) so
        # this test proves the whole-file pass stays CPU-pinned independent of it.
        dz._diarization_device = "cuda"

        dz._ensure_diarization_pipeline_loaded()
        dz._ensure_whole_file_diarization_pipeline()

        self.assertEqual(factory.devices, ["cuda", "cpu"])

    def test_failing_pipeline_returns_empty_turns_not_raise(self) -> None:
        def _broken_factory(*, model_name, device, hf_token):
            raise RuntimeError("boom")

        dz = self._diarizer(pipeline_factory=_broken_factory)
        turns = dz.diarize_whole_file_turns_from_audio(np.ones((32000,), dtype=np.float32) * 0.05)
        self.assertEqual(turns, [])


class WholeFileDiarizationServerTranscriberTests(unittest.TestCase):
    def _transcriber(self, *, with_diarizer: bool = True) -> tuple[WhisperCppServerTranscriber, Path]:
        tmp = Path(tempfile.mkdtemp())
        client = _FakeClient(payloads=[{}])
        manager = WhisperCppServerManager(
            server_path=_touch(tmp / "whisper-server.exe"),
            model_path=_touch(tmp / "ggml-medium.bin"),
            client_factory=lambda **kwargs: client,
            popen_factory=lambda *args, **kwargs: _FakeProcess(),
        )
        diarizer = (
            WhisperCppDiarizer(
                device="cpu",
                auto_download=False,
                pipeline_factory=_RecordingPipelineFactory(),
                assign_word_speakers=_assign_one_speaker,
                speaker_identity_engine=_FakeSpeakerIdentityEngine(),
            )
            if with_diarizer
            else None
        )
        transcriber = WhisperCppServerTranscriber(
            manager=manager,
            fallback_transcriber=_FakeFallback(),
            diarizer=diarizer,
        )
        return transcriber, tmp

    def test_supports_whole_file_diarization_reflects_diarizer_presence(self) -> None:
        with_dz, _ = self._transcriber(with_diarizer=True)
        without_dz, _ = self._transcriber(with_diarizer=False)
        self.assertTrue(with_dz.supports_whole_file_diarization())
        self.assertFalse(without_dz.supports_whole_file_diarization())

    def test_diarize_whole_file_turns_without_diarizer_returns_empty(self) -> None:
        transcriber, _ = self._transcriber(with_diarizer=False)
        turns = transcriber.diarize_whole_file_turns(_pcm_chunk(), "mono")
        self.assertEqual(turns, [])

    def test_diarize_whole_file_turns_delegates_to_diarizer(self) -> None:
        transcriber, _ = self._transcriber(with_diarizer=True)
        turns = transcriber.diarize_whole_file_turns(_pcm_chunk(), "mono")
        self.assertEqual([t["speaker"] for t in turns], ["SPEAKER_00", "SPEAKER_01"])

    def test_set_diarization_suppressed_skips_diarization_in_transcribe(self) -> None:
        payload = {
            "detected_language": "en",
            "segments": [
                {
                    "text": "hello",
                    "start": 0.0,
                    "end": 0.8,
                    "words": [{"word": "hello", "start": 0.0, "end": 0.8, "probability": 0.99}],
                }
            ],
        }
        transcriber, _ = self._transcriber(with_diarizer=True)
        transcriber.prewarm("en")
        transcriber._manager.client.payloads = [payload]

        transcriber.set_diarization_suppressed(True)
        text = transcriber.transcribe(_pcm_chunk(), language="en")
        meta = transcriber.get_last_transcription_meta()

        self.assertEqual(text, "hello")
        self.assertNotIn("[spk_", text)
        self.assertEqual(meta["speaker_turn_count"], 0)
        self.assertEqual(meta["token_timestamps"][0]["speaker"], "")

    def test_set_diarization_suppressed_false_restores_normal_behavior(self) -> None:
        payload = {
            "detected_language": "en",
            "segments": [
                {
                    "text": "hello",
                    "start": 0.0,
                    "end": 0.8,
                    "words": [{"word": "hello", "start": 0.0, "end": 0.8, "probability": 0.99}],
                }
            ],
        }
        transcriber, _ = self._transcriber(with_diarizer=True)
        transcriber.prewarm("en")
        transcriber._manager.client.payloads = [payload]

        transcriber.set_diarization_suppressed(True)
        transcriber.set_diarization_suppressed(False)
        text = transcriber.transcribe(_pcm_chunk(), language="en")

        self.assertIn("[spk_000] hello", text)


class WholeFileDiarizationEndToEndTests(unittest.TestCase):
    def test_run_direct_transcription_activates_whole_file_diarization(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        segment_payload = {
            "detected_language": "en",
            "segments": [
                {
                    "text": "hello",
                    "start": 0.0,
                    "end": 0.8,
                    "words": [{"word": "hello", "start": 0.0, "end": 0.8, "probability": 0.99}],
                },
                {
                    "text": "world",
                    "start": 1.0,
                    "end": 1.8,
                    "words": [{"word": "world", "start": 1.0, "end": 1.8, "probability": 0.98}],
                },
            ],
        }
        # First payload is consumed by prewarm()'s own warmup inference (mirroring the
        # real product path: controller._run_session_finalize_relabel_guarded and the
        # manual import-direct action both warm the transcriber up before calling
        # run_direct_transcription), the second is the real transcription.
        client = _FakeClient(payloads=[{}, segment_payload])
        manager = WhisperCppServerManager(
            server_path=_touch(tmp / "whisper-server.exe"),
            model_path=_touch(tmp / "ggml-medium.bin"),
            client_factory=lambda **kwargs: client,
            popen_factory=lambda *args, **kwargs: _FakeProcess(),
        )
        diarizer = WhisperCppDiarizer(
            device="cpu",
            auto_download=False,
            pipeline_factory=_RecordingPipelineFactory(),
            speaker_identity_engine=_FakeSpeakerIdentityEngine(),
        )
        transcriber = WhisperCppServerTranscriber(
            manager=manager,
            fallback_transcriber=_FakeFallback(),
            diarizer=diarizer,
        )
        transcriber.prewarm("en")

        cfg = RuntimeConfig(stt_provider="whispercpp", source_language="en")
        full_audio = _pcm_chunk(seconds=2.0)

        apply_calls = {"n": 0}
        original_apply = diarizer.apply

        def _counting_apply(*args, **kwargs):
            apply_calls["n"] += 1
            return original_apply(*args, **kwargs)

        diarizer.apply = _counting_apply  # type: ignore[method-assign]

        result = run_direct_transcription(
            cfg,
            full_audio,
            transcriber=transcriber,
            chunk_seconds=0.0,
            whole_file_diarization=True,
        )

        meta = result["meta"]
        self.assertEqual(meta["speaker_profile_reconciliation"]["status"], "whole_file_diarization")
        self.assertGreater(meta["speaker_profile_reconciliation"]["tokens_assigned"], 0)
        labeled = [row for row in meta["token_timestamps"] if row.get("speaker")]
        self.assertTrue(labeled)
        self.assertEqual({row["speaker"] for row in labeled}, {"SPEAKER_00", "SPEAKER_01"})
        # Per-chunk diarization must have been suppressed during the ASR loop: the
        # per-window diarizer.apply() (round 0065's live path) must never fire while the
        # whole-file pass owns speaker assignment, or per-chunk noise would pollute the
        # segments before the global time-overlap stamping runs.
        self.assertEqual(apply_calls["n"], 0)
        # And suppression must be released again afterward (not left stuck on), so any
        # later LIVE use of the same transcriber instance still gets per-window diarization.
        self.assertFalse(transcriber._diarization_suppressed)


if __name__ == "__main__":
    unittest.main()
