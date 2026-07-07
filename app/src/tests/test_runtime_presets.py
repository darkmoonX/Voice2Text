from __future__ import annotations

from pathlib import Path
import sys
import types
import unittest

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice2text.bootstrap_args import build_arg_parser
from voice2text.bootstrap_config import build_runtime_config
from voice2text.settings.presets import (
    PRESETS,
    apply_preset,
    apply_preset_defaults,
    normalize_preset,
    preset_arg_defaults,
)
from voice2text.whisper_config import WhisperRuntimeParams


def _parse(argv: list[str]):
    parser = build_arg_parser(WhisperRuntimeParams())
    apply_preset_defaults(parser, argv)
    return parser.parse_args(argv)


class PresetTableTests(unittest.TestCase):
    def test_normalize_preset_aliases_and_unknown(self) -> None:
        self.assertEqual(normalize_preset("high-accuracy"), "high-accuracy")
        self.assertEqual(normalize_preset("HIGH_ACCURACY"), "high-accuracy")
        self.assertEqual(normalize_preset("accurate"), "high-accuracy")
        self.assertEqual(normalize_preset("default"), "balanced")
        self.assertEqual(normalize_preset(""), "")
        self.assertEqual(normalize_preset("nonsense"), "")
        self.assertEqual(normalize_preset("low-latency"), "")  # dropped in Phase B

    def test_apply_preset_sets_bundle_and_leaves_others(self) -> None:
        cfg = types.SimpleNamespace(runtime_preset="", unrelated=123)
        applied = apply_preset(cfg, "high-accuracy")
        self.assertEqual(cfg.model_size, "large-v2")
        self.assertEqual(cfg.compute_type, "float16")
        self.assertEqual(cfg.whisper_beam_size, 5)
        self.assertEqual(cfg.segment_seconds, 10.0)
        self.assertTrue(cfg.whisperx_enable_diarization)
        self.assertTrue(cfg.whisperx_speaker_profile_enabled)
        self.assertEqual(cfg.runtime_preset, "high-accuracy")
        self.assertEqual(cfg.unrelated, 123)  # untouched
        self.assertIn("model_size", applied)

    def test_apply_unknown_preset_is_noop(self) -> None:
        cfg = types.SimpleNamespace(runtime_preset="")
        self.assertEqual(apply_preset(cfg, ""), [])
        self.assertEqual(cfg.runtime_preset, "")

    def test_preset_arg_defaults_maps_to_dests(self) -> None:
        d = preset_arg_defaults("high-accuracy")
        self.assertEqual(d["model"], "large-v2")
        self.assertEqual(d["beam_size"], 5)
        self.assertEqual(d["segment_seconds"], 10.0)
        self.assertTrue(d["whisperx_diarization"])
        self.assertTrue(d["whisperx_speaker_profile"])


class PresetCliTests(unittest.TestCase):
    def test_balanced_bundle_values(self) -> None:
        cfg = build_runtime_config(_parse(["--preset", "balanced"]))
        self.assertEqual(cfg.model_size, "medium")
        self.assertEqual(cfg.compute_type, "float16")
        self.assertEqual(cfg.whisper_beam_size, 5)
        self.assertEqual(cfg.segment_seconds, 10.0)
        self.assertEqual(cfg.hop_seconds, 2.0)
        self.assertTrue(cfg.whisperx_enable_forced_alignment)
        self.assertFalse(cfg.whisperx_enable_diarization)
        self.assertTrue(cfg.whisperx_speaker_profile_enabled)

    def test_balanced_matches_bare_default_except_model(self) -> None:
        # round 0014 set the bare seg/hop/beam/compute default to the balanced point;
        # only model_size differs (bare default stays the no-download 'small').
        base = build_runtime_config(_parse([]))
        balanced = build_runtime_config(_parse(["--preset", "balanced"]))
        for field in ("compute_type", "whisper_beam_size", "segment_seconds", "hop_seconds"):
            self.assertEqual(getattr(balanced, field), getattr(base, field), field)
        self.assertEqual(base.model_size, "small")
        self.assertEqual(balanced.model_size, "medium")

    def test_high_accuracy_bundle_applied(self) -> None:
        cfg = build_runtime_config(_parse(["--preset", "high-accuracy"]))
        self.assertEqual(cfg.model_size, "large-v2")
        self.assertEqual(cfg.segment_seconds, 10.0)
        self.assertEqual(cfg.hop_seconds, 2.0)
        self.assertEqual(cfg.whisper_beam_size, 5)
        self.assertTrue(cfg.whisperx_enable_forced_alignment)
        self.assertTrue(cfg.whisperx_enable_diarization)
        self.assertTrue(cfg.whisperx_speaker_profile_enabled)
        self.assertEqual(cfg.runtime_preset, "high-accuracy")

    def test_explicit_flag_overrides_preset(self) -> None:
        cfg = build_runtime_config(_parse(["--preset", "high-accuracy", "--beam-size", "1"]))
        self.assertEqual(cfg.whisper_beam_size, 1)  # explicit wins
        self.assertEqual(cfg.model_size, "large-v2")  # from preset

    def test_explicit_bool_flag_overrides_preset(self) -> None:
        cfg = build_runtime_config(
            _parse(["--preset", "high-accuracy", "--no-whisperx-diarization"])
        )
        self.assertFalse(cfg.whisperx_enable_diarization)  # explicit off wins over preset's on

    def test_diarization_speaker_hints_default_unset(self) -> None:
        cfg = build_runtime_config(_parse([]))
        self.assertEqual(cfg.whisperx_diarization_min_speakers, 0)
        self.assertEqual(cfg.whisperx_diarization_max_speakers, 0)

    def test_diarization_speaker_hints_pass_through_and_clamp(self) -> None:
        cfg = build_runtime_config(
            _parse(["--diarization-min-speakers", "2", "--diarization-max-speakers", "4"])
        )
        self.assertEqual(cfg.whisperx_diarization_min_speakers, 2)
        self.assertEqual(cfg.whisperx_diarization_max_speakers, 4)

        cfg = build_runtime_config(
            _parse(["--diarization-min-speakers", "-2", "--diarization-max-speakers", "-4"])
        )
        self.assertEqual(cfg.whisperx_diarization_min_speakers, 0)
        self.assertEqual(cfg.whisperx_diarization_max_speakers, 0)

    def test_speaker_count_hint_cli_round_trip(self) -> None:
        cfg = build_runtime_config(_parse([]))
        self.assertFalse(cfg.whisperx_speaker_count_hint_enabled)
        self.assertEqual(cfg.whisperx_speaker_count_hint_seconds, 60.0)
        self.assertEqual(cfg.whisperx_speaker_count_hint_window_seconds, 300.0)
        self.assertEqual(cfg.whisperx_speaker_count_hint_sliver_floor_seconds, 1.5)

        cfg = build_runtime_config(
            _parse(
                [
                    "--speaker-count-hint",
                    "--speaker-count-hint-seconds",
                    "12.5",
                    "--speaker-count-hint-window-seconds",
                    "45",
                    "--speaker-count-hint-sliver-floor-seconds",
                    "2.25",
                ]
            )
        )
        self.assertTrue(cfg.whisperx_speaker_count_hint_enabled)
        self.assertEqual(cfg.whisperx_speaker_count_hint_seconds, 12.5)
        self.assertEqual(cfg.whisperx_speaker_count_hint_window_seconds, 45.0)
        self.assertEqual(cfg.whisperx_speaker_count_hint_sliver_floor_seconds, 2.25)

        cfg = build_runtime_config(_parse(["--speaker-count-hint", "--no-speaker-count-hint"]))
        self.assertFalse(cfg.whisperx_speaker_count_hint_enabled)

    def test_speaker_merge_grace_cli_round_trip(self) -> None:
        cfg = build_runtime_config(_parse([]))
        self.assertEqual(cfg.whisperx_speaker_merge_grace_windows, 0)
        self.assertEqual(cfg.whisperx_speaker_merge_grace_relief, 0.10)

        cfg = build_runtime_config(
            _parse(["--speaker-merge-grace-windows", "30", "--speaker-merge-grace-relief", "0.15"])
        )
        self.assertEqual(cfg.whisperx_speaker_merge_grace_windows, 30)
        self.assertEqual(cfg.whisperx_speaker_merge_grace_relief, 0.15)

        cfg = build_runtime_config(
            _parse(["--speaker-merge-grace-windows", "-5", "--speaker-merge-grace-relief", "-0.25"])
        )
        self.assertEqual(cfg.whisperx_speaker_merge_grace_windows, 0)
        self.assertEqual(cfg.whisperx_speaker_merge_grace_relief, 0.0)

    def test_speaker_merge_preserve_centroid_cli_round_trip(self) -> None:
        cfg = build_runtime_config(_parse([]))
        self.assertFalse(cfg.whisperx_speaker_merge_preserve_centroid)

        cfg = build_runtime_config(_parse(["--speaker-merge-preserve-centroid"]))
        self.assertTrue(cfg.whisperx_speaker_merge_preserve_centroid)

        cfg = build_runtime_config(
            _parse(["--speaker-merge-preserve-centroid", "--no-speaker-merge-preserve-centroid"])
        )
        self.assertFalse(cfg.whisperx_speaker_merge_preserve_centroid)

    def test_speaker_profile_max_exemplars_cli_round_trip(self) -> None:
        cfg = build_runtime_config(_parse([]))
        self.assertEqual(cfg.whisperx_speaker_profile_max_exemplars, 1)
        self.assertEqual(cfg.whisperx_speaker_profile_exemplar_diversity_threshold, 0.90)

        cfg = build_runtime_config(
            _parse([
                "--speaker-profile-max-exemplars", "4",
                "--speaker-profile-exemplar-diversity-threshold", "0.85",
            ])
        )
        self.assertEqual(cfg.whisperx_speaker_profile_max_exemplars, 4)
        self.assertEqual(cfg.whisperx_speaker_profile_exemplar_diversity_threshold, 0.85)

        cfg = build_runtime_config(_parse(["--speaker-profile-max-exemplars", "-5"]))
        self.assertEqual(cfg.whisperx_speaker_profile_max_exemplars, 1)


if __name__ == "__main__":
    unittest.main()
