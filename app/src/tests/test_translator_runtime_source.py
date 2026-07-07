from __future__ import annotations

import unittest

from voice2text.translator import ArgosTranslator


class _DummyTranslation:
    def __init__(self, prefix: str) -> None:
        self._prefix = prefix

    def translate(self, text: str) -> str:
        return f"{self._prefix}:{text}"


class TranslatorRuntimeSourceTests(unittest.TestCase):
    def _build_translator(self) -> ArgosTranslator:
        translator = ArgosTranslator.__new__(ArgosTranslator)
        translator._enabled = True
        translator._source_code = "auto"
        translator._target_code = "zh"
        translator._auto_install = False
        translator._argos_translate = None
        translator._translation = _DummyTranslation("default")
        translator._runtime_translation_cache = {}
        return translator

    def test_translate_accepts_source_code_argument(self) -> None:
        translator = self._build_translator()
        translator._source_code = "en"
        result = translator.translate("hello", source_code="en")
        self.assertEqual(result, "default:hello")

    def test_translate_prefers_runtime_source_translation_when_available(self) -> None:
        translator = self._build_translator()
        translator._runtime_translation_for_source = lambda source_code: _DummyTranslation(f"runtime-{source_code}")  # type: ignore[method-assign]
        result = translator.translate("hello", source_code="ja")
        self.assertEqual(result, "runtime-ja:hello")

    def test_translate_returns_none_when_disabled_or_empty(self) -> None:
        translator = self._build_translator()
        translator._enabled = False
        self.assertIsNone(translator.translate("hello", source_code="en"))
        translator._enabled = True
        self.assertIsNone(translator.translate("   ", source_code="en"))


if __name__ == "__main__":
    unittest.main()
