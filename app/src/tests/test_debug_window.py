from __future__ import annotations

import unittest

from voice2text.debug_window import STTDebugWindow


class DebugWindowTests(unittest.TestCase):
    def test_compact_meta_replaces_full_token_timestamps_with_count_and_sample(self) -> None:
        meta = {
            "detected_language": "zh",
            "token_timestamps": [
                {"word": f"字{i}", "start": float(i), "end": float(i) + 0.1, "score": 0.9}
                for i in range(20)
            ],
            "speaker_profile_stats": {"large": "omitted"},
        }

        compact = STTDebugWindow._compact_meta(meta)

        self.assertNotIn("token_timestamps", compact)
        self.assertNotIn("speaker_profile_stats", compact)
        self.assertEqual(compact["token_timestamp_count"], 20)
        self.assertEqual(len(compact["token_timestamp_sample"]), 8)
        self.assertEqual(compact["detected_language"], "zh")


if __name__ == "__main__":
    unittest.main()
