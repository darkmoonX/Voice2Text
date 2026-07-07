from __future__ import annotations

from pathlib import Path
import sys
import unittest

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice2text.pipeline.language_routing import route_source_language


class LanguageRoutingTests(unittest.TestCase):
    def test_explicit_language_always_wins(self) -> None:
        self.assertEqual(
            route_source_language(
                explicit_source_language="ja",
                locked_source_language="en",
                detected_language="zh",
                stability_ratio=1.0,
                token_count=100,
                text="這是一段中文",
                allowed_languages={"en", "zh", "ja"},
            ),
            "ja",
        )

    def test_corroborated_code_switch_routes_detected_language(self) -> None:
        self.assertEqual(
            route_source_language(
                explicit_source_language=None,
                locked_source_language="en",
                detected_language="zh",
                stability_ratio=0.92,
                token_count=12,
                text="這是一段中文內容",
                allowed_languages={"en", "zh", "ja"},
            ),
            "zh",
        )

    def test_cjk_text_misdetected_as_english_falls_back_to_lock(self) -> None:
        self.assertEqual(
            route_source_language(
                explicit_source_language=None,
                locked_source_language="zh",
                detected_language="en",
                stability_ratio=0.96,
                token_count=20,
                text="這是一段中文內容",
                allowed_languages={"en", "zh"},
            ),
            "zh",
        )

    def test_short_low_stability_window_falls_back_to_lock(self) -> None:
        self.assertEqual(
            route_source_language(
                explicit_source_language=None,
                locked_source_language="en",
                detected_language="ja",
                stability_ratio=0.30,
                token_count=2,
                text="これは",
                allowed_languages={"en", "ja"},
            ),
            "en",
        )

    def test_empty_no_lock_first_window_routes_none(self) -> None:
        self.assertIsNone(
            route_source_language(
                explicit_source_language=None,
                locked_source_language="",
                detected_language="",
                stability_ratio=0.0,
                token_count=0,
                text="",
                allowed_languages={"en", "zh"},
            )
        )

    def test_allow_list_rejection_falls_back_to_lock(self) -> None:
        self.assertEqual(
            route_source_language(
                explicit_source_language=None,
                locked_source_language="en",
                detected_language="ru",
                stability_ratio=1.0,
                token_count=20,
                text="privet",
                allowed_languages={"en", "zh"},
            ),
            "en",
        )


if __name__ == "__main__":
    unittest.main()
