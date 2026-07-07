"""Round 0045: per-language default bundles (first case: zh reconcile_threshold 0.88)."""
from __future__ import annotations

import unittest

from voice2text.config import RuntimeConfig
from voice2text.language_defaults import (
    apply_language_defaults,
    language_default_overrides,
    normalize_language_key,
)

_RC = "whisperx_speaker_profile_reconcile_threshold"


class LanguageDefaultsTests(unittest.TestCase):
    def test_normalize_folds_zh_family_and_strips_region(self):
        for code in ("zh", "zh-Hant", "zh_hans", "cmn", "yue", "ZH-CN"):
            self.assertEqual(normalize_language_key(code), "zh", code)
        self.assertEqual(normalize_language_key("en-US"), "en")
        self.assertIsNone(normalize_language_key("auto"))
        self.assertIsNone(normalize_language_key(""))
        self.assertIsNone(normalize_language_key(None))

    def test_zh_fills_reconcile_when_at_global_default(self):
        cfg = RuntimeConfig()
        cfg.source_language = "zh"
        self.assertEqual(getattr(cfg, _RC), 0.52)  # global default
        applied = apply_language_defaults(cfg)
        self.assertEqual(applied, [_RC])
        self.assertEqual(getattr(cfg, _RC), 0.88)

    def test_zh_hant_also_applies(self):
        cfg = RuntimeConfig()
        cfg.source_language = "zh-Hant"
        apply_language_defaults(cfg)
        self.assertEqual(getattr(cfg, _RC), 0.88)

    def test_explicit_nondefault_value_is_preserved(self):
        cfg = RuntimeConfig()
        cfg.source_language = "zh"
        cfg.whisperx_speaker_profile_reconcile_threshold = 0.60  # user-chosen != default
        applied = apply_language_defaults(cfg)
        self.assertEqual(applied, [])
        self.assertEqual(getattr(cfg, _RC), 0.60)

    def test_english_has_no_overrides(self):
        cfg = RuntimeConfig()
        cfg.source_language = "en"
        applied = apply_language_defaults(cfg)
        self.assertEqual(applied, [])
        self.assertEqual(getattr(cfg, _RC), 0.52)

    def test_zh_min_seconds_guard_is_inert_at_current_default(self):
        # min_seconds global default (2.0) already equals the zh target -> no change, not applied.
        cfg = RuntimeConfig()
        cfg.source_language = "zh"
        applied = apply_language_defaults(cfg)
        self.assertNotIn("whisperx_speaker_profile_min_seconds", applied)
        self.assertEqual(cfg.whisperx_speaker_profile_min_seconds, 2.0)

    def test_zh_min_seconds_guard_activates_if_global_default_lowered(self):
        # Simulate a future global default of 1.0: a zh user sitting at that default gets
        # pulled back to 2.0 (guards against CJK speaker suppression).
        base = RuntimeConfig()
        base.whisperx_speaker_profile_min_seconds = 1.0  # simulated lowered global default
        cfg = RuntimeConfig()
        cfg.source_language = "zh"
        cfg.whisperx_speaker_profile_min_seconds = 1.0  # user left at the (lowered) default
        applied = apply_language_defaults(cfg, base=base)
        self.assertIn("whisperx_speaker_profile_min_seconds", applied)
        self.assertEqual(cfg.whisperx_speaker_profile_min_seconds, 2.0)

    def test_auto_language_no_overrides(self):
        cfg = RuntimeConfig()
        cfg.source_language = None
        self.assertEqual(apply_language_defaults(cfg), [])
        self.assertEqual(language_default_overrides(None), {})


if __name__ == "__main__":
    unittest.main()
