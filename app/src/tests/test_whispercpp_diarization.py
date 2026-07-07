from __future__ import annotations

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
from voice2text.stt.factory import create_stt_transcriber
from voice2text.stt.whispercpp_common import build_transcription_meta
from voice2text.stt.whispercpp_diarization import WhisperCppDiarizer
from voice2text.stt.whispercpp_server import (
    WhisperCppServerManager,
    WhisperCppServerTranscriber,
)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"stub")
    return path


def _pcm_chunk() -> AudioChunk:
    samples = [12000, -12000] * 16000
    pcm = b"".join(int(v).to_bytes(2, "little", signed=True) for v in samples)
    return AudioChunk(pcm16=pcm, sample_rate=16000, channels=1)


class _FakePipeline:
    def __call__(self, audio, **kwargs):
        return [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}, {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_01"}]


class _FakeSpeakerIdentityEngine:
    def __init__(self) -> None:
        self.prewarm_calls = 0
        self.last_stats = {"enabled": True, "backend": "fake", "status": "done_assigned"}

    def prewarm(self) -> None:
        self.prewarm_calls += 1

    def apply(self, *, audio, segments, resolve_local_speaker):
        for segment in segments:
            local = resolve_local_speaker(segment)
            if not local:
                continue
            profile = "profile_000" if local.endswith("00") else "profile_001"
            segment["profile_speaker"] = profile
            for word in segment.get("words") or []:
                word["profile_speaker"] = profile
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


def _assign_two_speakers(_diarize_segments, payload: dict) -> dict:
    # Mirrors the real whisperx.diarize.assign_word_speakers contract: only sets `speaker`
    # (the raw local diarization label) at segment and word level. It does NOT set
    # `local_speaker` — that field must be derived later by the diarizer's own
    # profile-preferring resolution pass, so this mock must not pre-populate it (doing so
    # would mask a bug where that derivation step is missing).
    segments = list(payload["segments"])
    for index, segment in enumerate(segments):
        local = "SPEAKER_00" if index == 0 else "SPEAKER_01"
        segment["speaker"] = local
        for word in segment.get("words") or []:
            word["speaker"] = local
    return {"segments": segments}


class WhisperCppDiarizationTests(unittest.TestCase):
    def _diarizer(self) -> WhisperCppDiarizer:
        return WhisperCppDiarizer(
            device="cpu",
            auto_download=False,
            pipeline_factory=lambda **kwargs: _FakePipeline(),
            assign_word_speakers=_assign_two_speakers,
            speaker_identity_engine=_FakeSpeakerIdentityEngine(),
        )

    def test_default_meta_without_diarizer_keeps_empty_speaker_fields(self) -> None:
        meta = build_transcription_meta(
            provider_timing={},
            segments=[
                {
                    "text": "hello",
                    "start": 0.0,
                    "end": 1.0,
                    "words": [{"word": "hello", "start": 0.0, "end": 1.0, "score": 0.9}],
                }
            ],
        )

        self.assertEqual(meta["speaker_turns"], [])
        self.assertEqual(meta["speaker_turn_count"], 0)
        row = meta["token_timestamps"][0]
        self.assertEqual(row["speaker"], "")
        self.assertEqual(row["profile_speaker"], "")
        self.assertEqual(row["local_speaker"], "")

    def test_diarizer_assigns_local_and_profile_speakers_and_formats_markers(self) -> None:
        diarizer = self._diarizer()
        audio = np.ones((32000,), dtype=np.float32) * 0.05
        segments = [
            {
                "text": "hello",
                "start": 0.0,
                "end": 0.8,
                "words": [{"word": "hello", "start": 0.0, "end": 0.8, "score": 0.99}],
            },
            {
                "text": "world",
                "start": 1.0,
                "end": 1.8,
                "words": [{"word": "world", "start": 1.0, "end": 1.8, "score": 0.98}],
            },
        ]

        aligned = diarizer.apply(audio, segments)
        text = diarizer.format_display_text(aligned)
        turns = diarizer.build_speaker_turns(aligned)
        meta = build_transcription_meta(
            provider_timing={},
            segments=aligned,
            speaker_turns=turns,
            speaker_profile_stats=diarizer.speaker_profile_stats,
        )

        self.assertIn("[spk_000] hello", text)
        self.assertIn("[spk_001] world", text)
        self.assertEqual([turn["speaker"] for turn in turns], ["SPEAKER_00", "SPEAKER_01"])
        rows = list(meta["token_timestamps"])
        # `speaker` must match WhisperX's contract: the profile-preferring resolved value
        # (identical to `profile_speaker`), NOT the raw per-window diarization label -
        # `subtitle_assembler._WordState` renders `raw.get('speaker') or
        # raw.get('profile_speaker')` as the display speaker, so an unstable raw label here
        # would defeat cross-window profile identity entirely.
        self.assertEqual(rows[0]["speaker"], "profile_000")
        self.assertEqual(rows[0]["profile_speaker"], "profile_000")
        self.assertEqual(rows[0]["local_speaker"], "SPEAKER_00")
        self.assertEqual(rows[1]["speaker"], "profile_001")
        self.assertEqual(rows[1]["profile_speaker"], "profile_001")
        self.assertEqual(rows[1]["local_speaker"], "SPEAKER_01")
        self.assertEqual(meta["speaker_turn_count"], 2)
        self.assertEqual(meta["speaker_profile_stats"]["status"], "done_assigned")

    def test_server_live_path_uses_diarizer_text_and_meta(self) -> None:
        payload = {
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
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            client = _FakeClient(payloads=[{}, payload])
            manager = WhisperCppServerManager(
                server_path=_touch(tmp / "whisper-server.exe"),
                model_path=_touch(tmp / "ggml-medium.bin"),
                client_factory=lambda **kwargs: client,
                popen_factory=lambda *args, **kwargs: _FakeProcess(),
            )
            transcriber = WhisperCppServerTranscriber(
                manager=manager,
                fallback_transcriber=_FakeFallback(),
                diarizer=self._diarizer(),
            )
            transcriber.prewarm("en")
            text = transcriber.transcribe(_pcm_chunk(), language="en")

        self.assertIn("[spk_000] hello", text)
        self.assertIn("[spk_001] world", text)
        meta = transcriber.get_last_transcription_meta()
        self.assertEqual(meta["speaker_turn_count"], 2)
        self.assertEqual(meta["token_timestamps"][0]["profile_speaker"], "profile_000")

    def test_module_import_does_not_load_torch_or_whisperx(self) -> None:
        heavy = {"torch", "whisperx"}
        saved = {name: module for (name, module) in sys.modules.items() if name.split(".")[0] in heavy}
        for name in list(sys.modules):
            if name == "voice2text.stt.whispercpp_diarization" or name.split(".")[0] in heavy:
                sys.modules.pop(name, None)
        try:
            __import__("voice2text.stt.whispercpp_diarization")
            self.assertFalse(heavy & {name.split(".")[0] for name in sys.modules})
        finally:
            sys.modules.update(saved)

    def test_factory_does_not_import_diarizer_when_diarization_disabled(self) -> None:
        sys.modules.pop("voice2text.stt.whispercpp_diarization", None)
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            cfg = RuntimeConfig(
                stt_provider="whispercpp",
                stt_whispercpp_mode="server",
                stt_whispercpp_binary_path=str(_touch(tmp / "whisper-cli.exe")),
                stt_whispercpp_server_path=str(_touch(tmp / "whisper-server.exe")),
                stt_whispercpp_model_path=str(_touch(tmp / "ggml-medium.bin")),
                whisperx_enable_diarization=False,
            )
            with patch("voice2text.stt.whispercpp_server.subprocess.Popen", return_value=_FakeProcess()):
                transcriber = create_stt_transcriber(cfg)

        self.assertIsInstance(transcriber, WhisperCppServerTranscriber)
        self.assertNotIn("voice2text.stt.whispercpp_diarization", sys.modules)

    def test_factory_constructs_diarizer_when_diarization_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            cfg = RuntimeConfig(
                stt_provider="whispercpp",
                stt_whispercpp_mode="server",
                stt_whispercpp_binary_path=str(_touch(tmp / "whisper-cli.exe")),
                stt_whispercpp_server_path=str(_touch(tmp / "whisper-server.exe")),
                stt_whispercpp_model_path=str(_touch(tmp / "ggml-medium.bin")),
                whisperx_enable_diarization=True,
                stt_auto_download=False,
                whisperx_speaker_profile_enabled=False,
            )
            with patch("voice2text.stt.whispercpp_server.subprocess.Popen", return_value=_FakeProcess()):
                transcriber = create_stt_transcriber(cfg)

        self.assertIsInstance(transcriber, WhisperCppServerTranscriber)
        self.assertIsInstance(getattr(transcriber, "_diarizer"), WhisperCppDiarizer)


if __name__ == "__main__":
    unittest.main()
