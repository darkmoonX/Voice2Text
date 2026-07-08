"""Round 0077: migrating the legacy alignment-model booleans into the generalized
whisperx_alignment_model_defaults map (settings_persistence.seed_alignment_model_defaults)."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice2text.config import RuntimeConfig
from voice2text.settings_persistence import seed_alignment_model_defaults


class AlignmentModelDefaultsMigrationTests(unittest.TestCase):
    def test_zh_wbbbbb_flag_seeds_map(self) -> None:
        cfg = RuntimeConfig(whisperx_zh_align_wbbbbb=True)
        seeded = seed_alignment_model_defaults(cfg)
        self.assertEqual(seeded, ["zh"])
        self.assertEqual(
            cfg.whisperx_alignment_model_defaults,
            {"zh": "wbbbbb/wav2vec2-large-chinese-zh-cn"},
        )

    def test_zh_flag_off_does_not_seed(self) -> None:
        cfg = RuntimeConfig(whisperx_zh_align_wbbbbb=False)
        seeded = seed_alignment_model_defaults(cfg)
        self.assertEqual(seeded, [])
        self.assertEqual(cfg.whisperx_alignment_model_defaults, {})

    def test_english_opt_out_seeds_base_model(self) -> None:
        cfg = RuntimeConfig(whisperx_english_align_large=False)
        seeded = seed_alignment_model_defaults(cfg)
        self.assertEqual(seeded, ["en"])
        self.assertEqual(cfg.whisperx_alignment_model_defaults, {"en": "WAV2VEC2_ASR_BASE_960H"})

    def test_english_default_true_does_not_seed(self) -> None:
        # True is the field's own default (no explicit intent signal), and the legacy
        # fallback in the provider already reproduces this behavior without a map entry.
        cfg = RuntimeConfig()
        seeded = seed_alignment_model_defaults(cfg)
        self.assertEqual(seeded, [])

    def test_idempotent_does_not_overwrite_existing_map_entry(self) -> None:
        cfg = RuntimeConfig(
            whisperx_zh_align_wbbbbb=True,
            whisperx_alignment_model_defaults={"zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"},
        )
        seeded = seed_alignment_model_defaults(cfg)
        self.assertEqual(seeded, [])
        self.assertEqual(
            cfg.whisperx_alignment_model_defaults,
            {"zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"},
        )

    def test_both_legacy_flags_migrate_together(self) -> None:
        cfg = RuntimeConfig(whisperx_zh_align_wbbbbb=True, whisperx_english_align_large=False)
        seeded = seed_alignment_model_defaults(cfg)
        self.assertEqual(sorted(seeded), ["en", "zh"])
        self.assertEqual(
            cfg.whisperx_alignment_model_defaults,
            {
                "zh": "wbbbbb/wav2vec2-large-chinese-zh-cn",
                "en": "WAV2VEC2_ASR_BASE_960H",
            },
        )


if __name__ == "__main__":
    unittest.main()
