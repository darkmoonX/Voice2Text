from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice2text.audio_capture import AudioChunk
from voice2text.config import RuntimeConfig
from voice2text.pipeline.direct_transcription import run_direct_transcription
from voice2text.stt.factory import create_stt_transcriber
from voice2text.stt.whispercpp_diarization import WhisperCppDiarizer
from voice2text.stt.whispercpp_provider import WhisperCppTranscriber


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
    def __init__(self, rows: list[_WholeFileRow]) -> None:
        self._rows = rows

    def __call__(self, audio, **kwargs):
        return _WholeFileTurnsResult(list(self._rows))


class _RecordingPipelineFactory:
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
        for segment in segments:
            local = resolve_local_speaker(segment)
            if not local:
                continue
            segment["profile_speaker"] = "profile_000"
            for word in segment.get("words") or []:
                word["profile_speaker"] = "profile_000"
        return segments


def _assign_one_speaker(_diarize_segments, payload: dict) -> dict:
    segments = list(payload["segments"])
    for segment in segments:
        segment["speaker"] = "SPEAKER_00"
        for word in segment.get("words") or []:
            word["speaker"] = "SPEAKER_00"
    return {"segments": segments}


def _fake_cli_run(payload_segments: list[dict]):
    """Build a subprocess.run side_effect that writes whisper.cpp CLI-style JSON output."""
    import json

    def _run(cmd, **kwargs):
        prefix = Path(cmd[cmd.index("-of") + 1])
        prefix.with_suffix(".json").write_text(
            json.dumps({"transcription": payload_segments}, ensure_ascii=False),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return _run


class SubprocessDiarizationUnitTests(unittest.TestCase):
    def _diarizer(self, *, pipeline_factory=None, assign_word_speakers=None) -> WhisperCppDiarizer:
        return WhisperCppDiarizer(
            device="cpu",
            auto_download=False,
            pipeline_factory=pipeline_factory or _RecordingPipelineFactory(),
            assign_word_speakers=assign_word_speakers,
            speaker_identity_engine=_FakeSpeakerIdentityEngine(),
        )

    def _transcriber(self, tmp: Path, *, diarizer=None) -> WhisperCppTranscriber:
        return WhisperCppTranscriber(
            binary_path=_touch(tmp / "whisper-cli.exe"),
            model_path=_touch(tmp / "ggml-medium.bin"),
            device="vulkan",
            beam_size=5,
            diarizer=diarizer,
        )

    def test_no_diarizer_behavior_is_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            provider = self._transcriber(tmp)
            self.assertFalse(provider.supports_whole_file_diarization())
            with patch(
                "voice2text.stt.whispercpp_provider.subprocess.run",
                side_effect=_fake_cli_run([{"text": "hello world", "offsets": {"from": 0, "to": 2000}}]),
            ):
                text = provider.transcribe(_pcm_chunk(), language="en")
            self.assertEqual(text, "hello world")
            meta = provider.get_last_transcription_meta()
            self.assertEqual(meta["speaker_turn_count"], 0)

    def test_prewarm_no_diarizer_does_not_raise(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            provider = self._transcriber(Path(raw_tmp))
            provider.prewarm("en")  # must not raise

    def test_prewarm_with_diarizer_calls_diarizer_prewarm(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            dz = self._diarizer()
            calls = {"n": 0}
            original = dz.prewarm

            def _counting_prewarm():
                calls["n"] += 1
                return original()

            dz.prewarm = _counting_prewarm  # type: ignore[method-assign]
            provider = self._transcriber(Path(raw_tmp), diarizer=dz)
            provider.prewarm("en")
            self.assertEqual(calls["n"], 1)

    def test_transcribe_runs_diarization_when_configured(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            dz = self._diarizer(assign_word_speakers=_assign_one_speaker)
            provider = self._transcriber(tmp, diarizer=dz)
            with patch(
                "voice2text.stt.whispercpp_provider.subprocess.run",
                side_effect=_fake_cli_run([{"text": "hello", "offsets": {"from": 0, "to": 800}}]),
            ):
                text = provider.transcribe(_pcm_chunk(), language="en")
            self.assertIn("[spk_000] hello", text)
            meta = provider.get_last_transcription_meta()
            self.assertGreater(meta["speaker_turn_count"], 0)

    def test_set_diarization_suppressed_skips_diarization(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            dz = self._diarizer(assign_word_speakers=_assign_one_speaker)
            provider = self._transcriber(tmp, diarizer=dz)
            provider.set_diarization_suppressed(True)
            with patch(
                "voice2text.stt.whispercpp_provider.subprocess.run",
                side_effect=_fake_cli_run([{"text": "hello", "offsets": {"from": 0, "to": 800}}]),
            ):
                text = provider.transcribe(_pcm_chunk(), language="en")
            self.assertEqual(text, "hello")
            self.assertNotIn("[spk_", text)
            meta = provider.get_last_transcription_meta()
            self.assertEqual(meta["speaker_turn_count"], 0)

            provider.set_diarization_suppressed(False)
            with patch(
                "voice2text.stt.whispercpp_provider.subprocess.run",
                side_effect=_fake_cli_run([{"text": "hello", "offsets": {"from": 0, "to": 800}}]),
            ):
                text = provider.transcribe(_pcm_chunk(), language="en")
            self.assertIn("[spk_000] hello", text)

    def test_supports_and_diarize_whole_file_turns_without_diarizer(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            provider = self._transcriber(Path(raw_tmp))
            self.assertFalse(provider.supports_whole_file_diarization())
            self.assertEqual(provider.diarize_whole_file_turns(_pcm_chunk(), "mono"), [])

    def test_diarize_whole_file_turns_delegates_to_diarizer(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            dz = self._diarizer()
            provider = self._transcriber(Path(raw_tmp), diarizer=dz)
            self.assertTrue(provider.supports_whole_file_diarization())
            turns = provider.diarize_whole_file_turns(_pcm_chunk(), "mono")
            self.assertEqual([t["speaker"] for t in turns], ["SPEAKER_00", "SPEAKER_01"])


class SubprocessDiarizationFactoryTests(unittest.TestCase):
    def test_subprocess_mode_wires_diarizer_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            cfg = RuntimeConfig(
                stt_provider="whispercpp",
                stt_whispercpp_mode="subprocess",
                stt_whispercpp_binary_path=str(_touch(tmp / "whisper-cli.exe")),
                stt_whispercpp_model_path=str(_touch(tmp / "ggml-medium.bin")),
                whisperx_enable_diarization=True,
            )
            sentinel = object()
            with patch("voice2text.stt.factory._build_whispercpp_diarizer", return_value=sentinel):
                transcriber = create_stt_transcriber(cfg)
            self.assertIsInstance(transcriber, WhisperCppTranscriber)
            self.assertIs(transcriber._diarizer, sentinel)

    def test_subprocess_mode_no_diarizer_when_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            cfg = RuntimeConfig(
                stt_provider="whispercpp",
                stt_whispercpp_mode="subprocess",
                stt_whispercpp_binary_path=str(_touch(tmp / "whisper-cli.exe")),
                stt_whispercpp_model_path=str(_touch(tmp / "ggml-medium.bin")),
                whisperx_enable_diarization=False,
            )
            transcriber = create_stt_transcriber(cfg)
            self.assertIsInstance(transcriber, WhisperCppTranscriber)
            self.assertIsNone(transcriber._diarizer)


class SubprocessDiarizationEndToEndTests(unittest.TestCase):
    def test_run_direct_transcription_activates_whole_file_diarization(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            dz = WhisperCppDiarizer(
                device="cpu",
                auto_download=False,
                pipeline_factory=_RecordingPipelineFactory(),
                speaker_identity_engine=_FakeSpeakerIdentityEngine(),
            )
            transcriber = WhisperCppTranscriber(
                binary_path=_touch(tmp / "whisper-cli.exe"),
                model_path=_touch(tmp / "ggml-medium.bin"),
                device="vulkan",
                beam_size=5,
                diarizer=dz,
            )
            transcriber.prewarm("en")

            cfg = RuntimeConfig(stt_provider="whispercpp", source_language="en")
            full_audio = _pcm_chunk(seconds=2.0)

            apply_calls = {"n": 0}
            original_apply = dz.apply

            def _counting_apply(*args, **kwargs):
                apply_calls["n"] += 1
                return original_apply(*args, **kwargs)

            dz.apply = _counting_apply  # type: ignore[method-assign]

            segments_payload = [
                {"text": "hello", "offsets": {"from": 0, "to": 800}},
                {"text": "world", "offsets": {"from": 1000, "to": 1800}},
            ]
            with patch(
                "voice2text.stt.whispercpp_provider.subprocess.run",
                side_effect=_fake_cli_run(segments_payload),
            ):
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
            self.assertEqual(apply_calls["n"], 0)
            self.assertFalse(transcriber._diarization_suppressed)


if __name__ == "__main__":
    unittest.main()
