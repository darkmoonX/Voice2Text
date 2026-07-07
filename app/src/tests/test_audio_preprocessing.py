"""Unit tests for pre-WhisperX audio preprocessing."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice2text.audio_capture import AudioChunk
from voice2text.config import RuntimeConfig
from voice2text.stt.preprocessing import create_audio_preprocessing_pipeline


def _chunk_from_float(audio: np.ndarray, sample_rate: int = 16000) -> AudioChunk:
    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16).tobytes()
    return AudioChunk(pcm16=pcm16, sample_rate=sample_rate, channels=1)


class AudioPreprocessingTests(unittest.TestCase):
    def test_preprocessing_pipeline_outputs_16k_mono_chunk(self) -> None:
        cfg = RuntimeConfig(
            preprocess_enabled=True,
            preprocess_modules="spectral-gate,adaptive-gain",
        )
        pipeline = create_audio_preprocessing_pipeline(cfg)
        chunk = _chunk_from_float(np.full((24000,), 0.02, dtype=np.float32), sample_rate=48000)

        processed = pipeline.process(chunk, channel_mode="mono")

        self.assertEqual(processed.sample_rate, 16000)
        self.assertEqual(processed.channels, 1)
        self.assertGreater(len(processed.pcm16), 0)
        self.assertEqual(pipeline.stage_names, ["spectral-gate", "adaptive-gain"])


if __name__ == "__main__":
    unittest.main()
