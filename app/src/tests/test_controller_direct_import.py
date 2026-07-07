from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import time
import unittest

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice2text.capture import AudioChunk
from voice2text.config import RuntimeConfig
import voice2text.controller as controller_mod
from voice2text.controller import TranscriptionController


class ControllerDirectImportTests(unittest.TestCase):
    def test_direct_import_runs_shared_core_and_records_exporter(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-direct-controller-") as td:
            root = Path(td)
            audio_path = root / "voice.wav"
            audio_path.write_bytes(b"not a real wav; test stubs decode/read")
            cfg = RuntimeConfig(
                log_dir=str(root / "logs"),
                transcript_export_formats="txt",
                transcript_export_dir=str(root / "exports"),
                import_direct_chunk_seconds=12.0,
                import_direct_language_subchunk_seconds=4.0,
            )
            ctl = TranscriptionController(cfg)

            calls: dict[str, object] = {}

            class FakeTranscriber:
                pass

            def fake_decode(path: Path, *, ffmpeg_dir: str = "") -> Path:
                calls["decode_path"] = path
                calls["decode_ffmpeg_dir"] = ffmpeg_dir
                return path

            def fake_read(path: Path) -> AudioChunk:
                calls["read_path"] = path
                return AudioChunk(pcm16=b"\0\0" * 16000, sample_rate=16000, channels=1)

            def fake_run(cfg_arg, audio, *, transcriber, chunk_seconds, language_subchunk_seconds, **kwargs):
                calls["cfg"] = cfg_arg
                calls["transcriber"] = transcriber
                calls["chunk_seconds"] = chunk_seconds
                calls["language_subchunk_seconds"] = language_subchunk_seconds
                progress = kwargs.get("on_progress")
                if callable(progress):
                    progress(1.0, 1.0)
                return {
                    "text": "[spk_000] hello world",
                    "meta": {
                        "elapsed_seconds": 0.0,
                        "token_timestamps": [
                            {
                                "word": "hello",
                                "absolute_start": 0.0,
                                "absolute_end": 0.4,
                                "speaker": "spk_000",
                            },
                            {
                                "word": "world",
                                "absolute_start": 0.4,
                                "absolute_end": 0.8,
                                "speaker": "spk_000",
                            },
                        ],
                    },
                }

            originals = (
                controller_mod.decode_to_wav_16k_mono,
                controller_mod.read_wav,
                controller_mod.run_direct_transcription,
            )
            try:
                controller_mod.decode_to_wav_16k_mono = fake_decode
                controller_mod.read_wav = fake_read
                controller_mod.run_direct_transcription = fake_run
                fake_transcriber = FakeTranscriber()
                ctl._create_transcriber_with_fallback = lambda: fake_transcriber  # type: ignore[method-assign]
                ctl._warmup_transcriber_instance = lambda transcriber: None  # type: ignore[method-assign]
                ctl._shutdown_transcriber_object = lambda transcriber: None  # type: ignore[method-assign]

                returned = ctl.import_audio_file_direct(str(audio_path))
                deadline = time.monotonic() + 5.0
                while ctl.is_running() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertFalse(ctl.is_running())

                self.assertEqual(Path(returned), audio_path)
                self.assertIs(calls["cfg"], cfg)
                self.assertIs(calls["transcriber"], fake_transcriber)
                self.assertEqual(calls["chunk_seconds"], 12.0)
                self.assertEqual(calls["language_subchunk_seconds"], 4.0)
                self.assertEqual(calls["decode_ffmpeg_dir"], cfg.ffmpeg_dll_dir)
                exported = ctl.export_transcript_now(
                    output_path=str(root / "manual.txt"),
                    export_format="txt",
                )
                self.assertIn("hello world", Path(exported).read_text(encoding="utf-8"))
            finally:
                (
                    controller_mod.decode_to_wav_16k_mono,
                    controller_mod.read_wav,
                    controller_mod.run_direct_transcription,
                ) = originals


if __name__ == "__main__":
    unittest.main()
