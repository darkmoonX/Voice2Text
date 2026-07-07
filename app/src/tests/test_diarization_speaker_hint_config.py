from __future__ import annotations

import unittest

from voice2text.bootstrap_runtime import build_restart_keys


class DiarizationSpeakerHintConfigTests(unittest.TestCase):
    def test_restart_keys_include_speaker_hints(self) -> None:
        keys = build_restart_keys()
        self.assertIn("whisperx_diarization_min_speakers", keys)
        self.assertIn("whisperx_diarization_max_speakers", keys)
        self.assertIn("whisperx_speaker_count_hint_enabled", keys)
        self.assertIn("whisperx_speaker_count_hint_seconds", keys)
        self.assertIn("whisperx_speaker_count_hint_window_seconds", keys)
        self.assertIn("whisperx_speaker_count_hint_sliver_floor_seconds", keys)
        self.assertIn("whisperx_speaker_merge_grace_windows", keys)
        self.assertIn("whisperx_speaker_merge_grace_relief", keys)
        self.assertIn("whisperx_speaker_merge_preserve_centroid", keys)
        self.assertIn("whisperx_speaker_profile_max_exemplars", keys)
        self.assertIn("whisperx_speaker_profile_exemplar_diversity_threshold", keys)


if __name__ == "__main__":
    unittest.main()
