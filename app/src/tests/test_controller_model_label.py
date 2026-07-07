from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice2text.config import RuntimeConfig
from voice2text.controller import TranscriptionController


class EffectiveModelLabelTests(unittest.TestCase):
    def test_whisperx_provider_uses_model_size(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-label-") as td:
            cfg = RuntimeConfig(log_dir=str(Path(td) / "logs"), stt_provider="whisperx", model_size="large-v2")
            ctl = TranscriptionController(cfg)
            self.assertEqual(ctl._effective_model_label(), "large-v2")

    def test_whispercpp_provider_uses_whispercpp_model_size_not_whisperx_model_size(self) -> None:
        # Regression: _effective_model_label() used to read the WhisperX-only `model_size`
        # field regardless of provider, so the "STT provider active: whispercpp | model=..."
        # status line always showed the leftover WhisperX default ("small") instead of the
        # actual whisper.cpp model in use (e.g. "large-v2"), even though transcription itself
        # correctly used stt_whispercpp_model_size all along (a display-only bug).
        with tempfile.TemporaryDirectory(prefix="v2t-label-") as td:
            cfg = RuntimeConfig(
                log_dir=str(Path(td) / "logs"),
                stt_provider="whispercpp",
                model_size="small",
                stt_whispercpp_model_size="large-v2",
            )
            ctl = TranscriptionController(cfg)
            self.assertEqual(ctl._effective_model_label(), "large-v2")

    def test_whispercpp_explicit_model_path_wins(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-label-") as td:
            cfg = RuntimeConfig(
                log_dir=str(Path(td) / "logs"),
                stt_provider="whispercpp",
                stt_whispercpp_model_size="large-v2",
                stt_whispercpp_model_path=r"D:\models\ggml-custom.bin",
            )
            ctl = TranscriptionController(cfg)
            self.assertEqual(ctl._effective_model_label(), r"D:\models\ggml-custom.bin")


if __name__ == "__main__":
    unittest.main()
