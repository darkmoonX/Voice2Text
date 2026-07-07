from __future__ import annotations

import json
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
from voice2text.pipeline.subtitle_assembler import SubtitleAssembler
from voice2text.stt.factory import create_stt_transcriber
from voice2text.stt.registry import normalize_stt_provider
from voice2text.stt.whispercpp_provider import WhisperCppTranscriber


def _pcm_chunk() -> AudioChunk:
    # Non-silent 16 kHz mono PCM, short but sufficient for the provider WAV path.
    samples = [12000, -12000] * 1600
    pcm = b"".join(int(v).to_bytes(2, "little", signed=True) for v in samples)
    return AudioChunk(pcm16=pcm, sample_rate=16000, channels=1)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"stub")
    return path


class WhisperCppProviderTests(unittest.TestCase):
    def _provider(self, tmp: Path) -> WhisperCppTranscriber:
        return WhisperCppTranscriber(
            binary_path=_touch(tmp / "whisper-cli.exe"),
            model_path=_touch(tmp / "ggml-medium.bin"),
            device="vulkan",
            beam_size=5,
        )

    def test_parses_json_and_builds_synthesized_monotonic_meta(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            provider = self._provider(tmp)

            def fake_run(cmd, **kwargs):
                prefix = Path(cmd[cmd.index("-of") + 1])
                prefix.with_suffix(".json").write_text(
                    json.dumps(
                        {
                            "transcription": [
                                {"text": "你好", "offsets": {"from": 0, "to": 1000}},
                                {"text": "hello world", "offsets": {"from": 1000, "to": 3000}},
                            ]
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

            with patch("voice2text.stt.whispercpp_provider.subprocess.run", side_effect=fake_run):
                text = provider.transcribe(_pcm_chunk(), language="zh-hant")

            self.assertEqual(text, "你好 hello world")
            meta = provider.get_last_transcription_meta()
            self.assertFalse(meta["alignment_enabled"])
            rows = list(meta["token_timestamps"])
            self.assertEqual([row["word"] for row in rows], ["你", "好", "hello", "world"])
            self.assertTrue(all(float(rows[i]["end"]) <= float(rows[i + 1]["start"]) for i in range(len(rows) - 1)))
            self.assertEqual(meta["stable_token_count"], len(rows))
            self.assertEqual(meta["detected_language"], "zh")

    def test_nonzero_exit_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            provider = self._provider(Path(raw_tmp))
            with patch(
                "voice2text.stt.whispercpp_provider.subprocess.run",
                return_value=subprocess.CompletedProcess(["whisper-cli"], 2, stdout="", stderr="boom"),
            ):
                with self.assertRaisesRegex(RuntimeError, "whisper.cpp transcription failed"):
                    provider.transcribe(_pcm_chunk(), language="en")

    def test_bad_json_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            provider = self._provider(Path(raw_tmp))

            def fake_run(cmd, **kwargs):
                prefix = Path(cmd[cmd.index("-of") + 1])
                prefix.with_suffix(".json").write_text("{bad", encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

            with patch("voice2text.stt.whispercpp_provider.subprocess.run", side_effect=fake_run):
                with self.assertRaisesRegex(RuntimeError, "invalid JSON"):
                    provider.transcribe(_pcm_chunk(), language="en")

    def test_missing_binary_raises_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            with self.assertRaisesRegex(RuntimeError, "build_whispercpp.ps1|VOICE2TEXT_WHISPERCPP_BIN"):
                WhisperCppTranscriber(
                    binary_path=tmp / "missing.exe",
                    model_path=_touch(tmp / "ggml-medium.bin"),
                )

    def test_overlapping_windows_dedup_with_synthesized_timestamps(self) -> None:
        asm = SubtitleAssembler()
        rows_a = WhisperCppTranscriber._synthesize_segment_word_timestamps(
            {"text": "look for evos", "start": 3.0, "end": 5.0}
        )
        rows_b = WhisperCppTranscriber._synthesize_segment_word_timestamps(
            {"text": "look for evos", "start": 1.0, "end": 3.0}
        )
        asm.merge_incremental_text(
            "look for evos",
            overlap_merge_method="exact",
            segment_seconds=10.0,
            hop_seconds=2.0,
            transcription_meta={"elapsed_seconds": 0.0, "token_timestamps": rows_a},
        )
        asm.merge_incremental_text(
            "look for evos",
            overlap_merge_method="exact",
            segment_seconds=10.0,
            hop_seconds=2.0,
            transcription_meta={"elapsed_seconds": 2.0, "token_timestamps": rows_b},
        )
        self.assertLessEqual(asm.finalize().lower().count("evos"), 1)

    def test_factory_routes_whispercpp_without_heavy_imports(self) -> None:
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
                    stt_whispercpp_mode="subprocess",
                    stt_whispercpp_binary_path=str(_touch(tmp / "whisper-cli.exe")),
                    stt_whispercpp_model_path=str(_touch(tmp / "ggml-medium.bin")),
                )
                transcriber = create_stt_transcriber(cfg)
            self.assertIsInstance(transcriber, WhisperCppTranscriber)
            self.assertFalse(heavy & {name.split(".")[0] for name in sys.modules})
        finally:
            sys.modules.update(saved)

    def test_legacy_aliases_still_map_to_whisperx(self) -> None:
        self.assertEqual(normalize_stt_provider("whisper"), "whisperx")
        self.assertEqual(normalize_stt_provider("faster-whisper"), "whisperx")
        self.assertEqual(normalize_stt_provider("whispercpp"), "whispercpp")


if __name__ == "__main__":
    unittest.main()
