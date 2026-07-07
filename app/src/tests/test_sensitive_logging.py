from __future__ import annotations

import unittest

from voice2text.bootstrap_runtime import sanitize_settings_for_log


class SensitiveLoggingTests(unittest.TestCase):
    def test_hf_token_is_redacted_in_settings_log_payload(self) -> None:
        safe = sanitize_settings_for_log(
            {
                "whisperx_hf_token": "hf_secret_value",
                "segment_seconds": 6.0,
            }
        )

        self.assertEqual(safe["whisperx_hf_token"], "<redacted>")
        self.assertEqual(safe["segment_seconds"], 6.0)
        self.assertNotIn("hf_secret_value", repr(safe))


if __name__ == "__main__":
    unittest.main()
