from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice2text.audio_capture import AudioChunk
from voice2text.config import RuntimeConfig
from voice2text.stt.factory import create_stt_transcriber
from voice2text.stt.whispercpp_server import (
    WhisperCppQualityGate,
    WhisperCppServerManager,
    WhisperCppServerTranscriber,
)
from voice2text.stt.whispercpp_runtime import resolve_whispercpp_vad_model


def _pcm_chunk() -> AudioChunk:
    samples = [12000, -12000] * 1600
    pcm = b"".join(int(v).to_bytes(2, "little", signed=True) for v in samples)
    return AudioChunk(pcm16=pcm, sample_rate=16000, channels=1)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"stub")
    return path


class _FakeClient:
    def __init__(self, payloads: list[dict] | None = None, fail: bool = False, **kwargs) -> None:
        self.payloads = list(payloads or [])
        self.fail = fail
        self.requests = 0

    def ready(self) -> bool:
        return True

    def infer_wav(self, wav_path: Path, *, language: str) -> dict:
        self.requests += 1
        if self.fail:
            raise RuntimeError("server down")
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
    def __init__(self) -> None:
        self.calls = 0
        self.meta = {
            "provider": "whispercpp",
            "alignment_enabled": False,
            "token_timestamps": [],
            "provider_timing": {"backend": "subprocess"},
        }

    def transcribe(self, chunk, language=None, channel_mode="mono") -> str:
        self.calls += 1
        return "fallback text"

    def get_last_transcription_meta(self) -> dict:
        return dict(self.meta)


class WhisperCppServerTests(unittest.TestCase):
    def _manager(self, tmp: Path, client: _FakeClient) -> WhisperCppServerManager:
        return WhisperCppServerManager(
            server_path=_touch(tmp / "whisper-server.exe"),
            model_path=_touch(tmp / "ggml-medium.bin"),
            vad_model_path=_touch(tmp / "ggml-silero-v5.1.2.bin"),
            client_factory=lambda **kwargs: client,
            popen_factory=lambda *args, **kwargs: _FakeProcess(),
        )

    def test_server_verbose_json_builds_monotonic_synth_meta(self) -> None:
        payload = {
            "detected_language": "zh",
            "language_probabilities": {"zh": 0.99},
            "segments": [
                {"text": "你好", "start": 0.0, "end": 1.0, "no_speech_prob": 0.0, "avg_logprob": -0.1},
                {"text": "hello world", "start": 1.0, "end": 3.0, "no_speech_prob": 0.0, "avg_logprob": -0.2},
            ],
        }
        with tempfile.TemporaryDirectory() as raw_tmp:
            client = _FakeClient(payloads=[{}, payload])
            transcriber = WhisperCppServerTranscriber(
                manager=self._manager(Path(raw_tmp), client),
                fallback_transcriber=_FakeFallback(),
            )
            transcriber.prewarm("zh")
            text = transcriber.transcribe(_pcm_chunk(), language="zh")

        self.assertEqual(text, "你好 hello world")
        meta = transcriber.get_last_transcription_meta()
        self.assertFalse(meta["alignment_enabled"])
        self.assertEqual(meta["detected_language"], "zh")
        self.assertEqual(meta["language_probabilities"], {"zh": 0.99})
        rows = list(meta["token_timestamps"])
        self.assertEqual([row["word"] for row in rows], ["你", "好", "hello", "world"])
        self.assertTrue(all(float(rows[i]["end"]) <= float(rows[i + 1]["start"]) for i in range(len(rows) - 1)))
        self.assertEqual(meta["stable_token_count"], len(rows))

    def test_server_verbose_json_uses_real_word_timestamps_when_present(self) -> None:
        payload = {
            "segments": [
                {
                    "text": "讓這裏",
                    "start": 1.0,
                    "end": 4.0,
                    "words": [
                        {"word": "讓", "start": 1.23, "end": 1.35, "probability": 0.91},
                        {"word": "這", "start": 1.36, "end": 1.51, "probability": 0.92},
                        {"word": "裏", "start": 1.52, "end": 1.70, "probability": 0.93},
                    ],
                }
            ],
            "detected_language": "zh",
        }
        with tempfile.TemporaryDirectory() as raw_tmp:
            client = _FakeClient(payloads=[{}, payload])
            transcriber = WhisperCppServerTranscriber(
                manager=self._manager(Path(raw_tmp), client),
                fallback_transcriber=_FakeFallback(),
            )
            transcriber.prewarm("zh")
            text = transcriber.transcribe(_pcm_chunk(), language="zh")

        self.assertEqual(text, "讓這裏")
        rows = list(transcriber.get_last_transcription_meta()["token_timestamps"])
        self.assertEqual([row["word"] for row in rows], ["讓", "這", "裏"])
        self.assertEqual([row["start"] for row in rows], [1.23, 1.36, 1.52])
        self.assertEqual([row["end"] for row in rows], [1.35, 1.51, 1.70])
        self.assertEqual([row["score"] for row in rows], [0.91, 0.92, 0.93])

    def test_quality_gate_drops_repetition_boilerplate_and_silence(self) -> None:
        gate = WhisperCppQualityGate(no_speech_threshold=0.8, repetition_similarity=0.9)
        segments = [
            {"text": "正常內容", "start": 0.0, "end": 1.0, "no_speech_prob": 0.0, "avg_logprob": -0.1},
            {"text": "正常 內容", "start": 1.0, "end": 2.0, "no_speech_prob": 0.0, "avg_logprob": -0.1},
            {"text": "请不吝点赞 订阅", "start": 2.0, "end": 3.0, "no_speech_prob": 0.0, "avg_logprob": -0.01},
            {"text": "雜訊", "start": 3.0, "end": 4.0, "no_speech_prob": 0.95, "avg_logprob": -0.1},
        ]

        kept = gate.filter_segments(segments)

        self.assertEqual([segment["text"] for segment in kept], ["正常內容"])
        self.assertIn("repetition", gate.dropped_reasons)
        self.assertIn("boilerplate", gate.dropped_reasons)
        self.assertIn("low-quality", gate.dropped_reasons)

    def test_request_failure_degrades_to_subprocess_without_raising(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            fallback = _FakeFallback()
            transcriber = WhisperCppServerTranscriber(
                manager=self._manager(Path(raw_tmp), _FakeClient(fail=True)),
                fallback_transcriber=fallback,
            )
            text = transcriber.transcribe(_pcm_chunk(), language="en")

        self.assertEqual(text, "fallback text")
        self.assertGreaterEqual(fallback.calls, 1)
        meta = transcriber.get_last_transcription_meta()
        self.assertEqual(meta["provider_timing"]["backend"], "subprocess-fallback")

    def test_server_vad_command_requires_and_passes_model_path(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            manager = WhisperCppServerManager(
                server_path=_touch(tmp / "whisper-server.exe"),
                model_path=_touch(tmp / "ggml-medium.bin"),
                vad_model_path=_touch(tmp / "ggml-silero-v5.1.2.bin"),
                use_vad=True,
                popen_factory=lambda *args, **kwargs: _FakeProcess(),
            )
            cmd = manager._build_command(49152)

            self.assertIn("--vad", cmd)
            self.assertIn("--vad-model", cmd)
            self.assertEqual(cmd[cmd.index("--vad-model") + 1], str(tmp / "ggml-silero-v5.1.2.bin"))

            with self.assertRaisesRegex(RuntimeError, "no VAD model path"):
                WhisperCppServerManager(
                    server_path=_touch(tmp / "server2.exe"),
                    model_path=_touch(tmp / "ggml-small.bin"),
                    use_vad=True,
                )

    def test_vad_prewarm_starts_server_without_silence_inference(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            client = _FakeClient(payloads=[])
            manager = WhisperCppServerManager(
                server_path=_touch(tmp / "whisper-server.exe"),
                model_path=_touch(tmp / "ggml-medium.bin"),
                vad_model_path=_touch(tmp / "ggml-silero-v5.1.2.bin"),
                use_vad=True,
                client_factory=lambda **kwargs: client,
                popen_factory=lambda *args, **kwargs: _FakeProcess(),
            )

            manager.prewarm("zh")

            self.assertTrue(manager.enabled)
            self.assertEqual(client.requests, 0)

    def test_vad_request_failure_uses_bounded_restart_then_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            client = _FakeClient(fail=True)
            fallback = _FakeFallback()
            transcriber = WhisperCppServerTranscriber(
                manager=WhisperCppServerManager(
                    server_path=_touch(tmp / "whisper-server.exe"),
                    model_path=_touch(tmp / "ggml-medium.bin"),
                    vad_model_path=_touch(tmp / "ggml-silero-v5.1.2.bin"),
                    use_vad=True,
                    client_factory=lambda **kwargs: client,
                    popen_factory=lambda *args, **kwargs: _FakeProcess(),
                ),
                fallback_transcriber=fallback,
            )
            transcriber.prewarm("zh")

            text = transcriber.transcribe(_pcm_chunk(), language="zh")

            self.assertEqual(text, "fallback text")
            self.assertEqual(fallback.calls, 1)
            self.assertEqual(client.requests, 2)

    def test_default_config_keeps_server_vad_off(self) -> None:
        from voice2text.bootstrap_args import build_arg_parser
        from voice2text.bootstrap_config import build_runtime_config
        from voice2text.whisper_config import WhisperRuntimeParams

        self.assertFalse(RuntimeConfig().stt_whispercpp_server_vad)
        self.assertEqual(RuntimeConfig().stt_whispercpp_server_max_len, 0)
        parser = build_arg_parser(WhisperRuntimeParams())
        args = parser.parse_args([])
        self.assertFalse(args.whispercpp_server_vad)
        cfg = build_runtime_config(args)
        self.assertFalse(cfg.stt_whispercpp_server_vad)
        self.assertEqual(cfg.stt_whispercpp_server_max_len, 0)

    def test_factory_server_route_does_not_import_heavy_stt_modules(self) -> None:
        heavy = {"torch", "ctranslate2", "whisperx", "faster_whisper"}
        saved = {name: module for (name, module) in sys.modules.items() if name.split(".")[0] in heavy}
        for name in list(sys.modules):
            if name.split(".")[0] in heavy:
                sys.modules.pop(name, None)
        try:
            with tempfile.TemporaryDirectory() as raw_tmp:
                tmp = Path(raw_tmp)
                cfg = RuntimeConfig(
                    stt_provider="whispercpp",
                    stt_whispercpp_mode="server",
                    stt_whispercpp_binary_path=str(_touch(tmp / "whisper-cli.exe")),
                    stt_whispercpp_server_path=str(_touch(tmp / "whisper-server.exe")),
                    stt_whispercpp_model_path=str(_touch(tmp / "ggml-medium.bin")),
                    stt_whispercpp_vad_model_path=str(_touch(tmp / "ggml-silero-v5.1.2.bin")),
                )
                with patch("voice2text.stt.whispercpp_server.subprocess.Popen", return_value=_FakeProcess()):
                    transcriber = create_stt_transcriber(cfg)
            self.assertIsInstance(transcriber, WhisperCppServerTranscriber)
            self.assertFalse(heavy & {name.split(".")[0] for name in sys.modules})
        finally:
            sys.modules.update(saved)

    def test_vad_model_resolver_uses_explicit_config_path(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            vad_path = _touch(tmp / "custom-vad.bin")
            cfg = RuntimeConfig(stt_whispercpp_vad_model_path=str(vad_path))

            self.assertEqual(resolve_whispercpp_vad_model(cfg), vad_path)

    def test_vad_model_resolver_downloads_expected_hf_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            cfg = RuntimeConfig(stt_whispercpp_vad_model_path="", stt_whispercpp_vad_model="ggml-silero-test.bin")
            with patch("voice2text.stt.whispercpp_runtime.whispercpp_model_dir", return_value=tmp):
                with patch("voice2text.stt.whispercpp_runtime.download_hf_files_with_progress") as download:
                    def fake_download(**kwargs):
                        Path(kwargs["output_dir"]).mkdir(parents=True, exist_ok=True)
                        (Path(kwargs["output_dir"]) / "ggml-silero-test.bin").write_bytes(b"vad")

                    download.side_effect = fake_download
                    resolved = resolve_whispercpp_vad_model(cfg, progress_callback=lambda msg: None)

            self.assertEqual(resolved, tmp / "ggml-silero-test.bin")
            kwargs = download.call_args.kwargs
            self.assertEqual(kwargs["repo_id"], "ggml-org/whisper-vad")
            self.assertEqual(kwargs["allow_patterns"], ["ggml-silero-test.bin"])


if __name__ == "__main__":
    unittest.main()
