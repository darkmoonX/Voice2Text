"""Round 0024 Phase B: the `cpu` (non-CUDA realtime) preset bundle + cpu_threads lever."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice2text.config import RuntimeConfig
from voice2text.settings.presets import PRESET_NAMES, apply_preset, normalize_preset


class CpuPresetTests(unittest.TestCase):
    def test_cpu_in_preset_names(self):
        self.assertIn("cpu", PRESET_NAMES)
        self.assertEqual(normalize_preset("cpu"), "cpu")

    def test_cpu_preset_bundle(self):
        cfg = RuntimeConfig()
        applied = apply_preset(cfg, "cpu")
        # The defining levers for non-CUDA realtime.
        self.assertEqual(cfg.stt_variant, "cpu")
        self.assertEqual(cfg.compute_type, "int8")
        self.assertEqual(cfg.model_size, "small")
        self.assertEqual(cfg.whisper_beam_size, 3)
        self.assertFalse(cfg.whisperx_enable_forced_alignment)  # the key CPU-cost lever
        self.assertFalse(cfg.whisperx_enable_diarization)
        self.assertFalse(cfg.whisperx_speaker_profile_enabled)
        self.assertEqual(cfg.runtime_preset, "cpu")
        self.assertIn("whisperx_enable_forced_alignment", applied)
        self.assertIn("stt_variant", applied)

    def test_cpu_threads_defaults_zero(self):
        self.assertEqual(RuntimeConfig().cpu_threads, 0)


if __name__ == "__main__":
    unittest.main()
