"""Unit tests for WhisperX speaker-profile language gating."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice2text.stt.whisperx_provider import WhisperXTranscriber


def _new_stub(max_exemplars: int, language_hint: str | None) -> WhisperXTranscriber:
    inst = WhisperXTranscriber.__new__(WhisperXTranscriber)
    inst._speaker_profile_max_exemplars = max_exemplars
    inst._source_language_hint = language_hint
    return inst


class SpeakerProfileLanguageGateTests(unittest.TestCase):
    def test_single_exemplar_default_returns_one_for_any_language_hint(self) -> None:
        for language_hint in (None, "", "auto", "en", "zh", "zh-TW"):
            with self.subTest(language_hint=language_hint):
                transcriber = _new_stub(1, language_hint)

                self.assertEqual(transcriber._effective_speaker_profile_max_exemplars(), 1)

    def test_multi_exemplar_returns_configured_value_for_zh_language_hints(self) -> None:
        for language_hint in ("zh", "zh-TW", "ZH-cn"):
            with self.subTest(language_hint=language_hint):
                transcriber = _new_stub(4, language_hint)

                self.assertEqual(transcriber._effective_speaker_profile_max_exemplars(), 4)

    def test_multi_exemplar_returns_one_for_non_zh_or_unresolved_language_hints(self) -> None:
        for language_hint in ("en", "auto", "", None):
            with self.subTest(language_hint=language_hint):
                transcriber = _new_stub(4, language_hint)

                self.assertEqual(transcriber._effective_speaker_profile_max_exemplars(), 1)

    def test_suppression_status_emits_only_when_gate_changes_value(self) -> None:
        emitted: list[str] = []
        transcriber = _new_stub(4, "en")
        transcriber._emit = emitted.append  # type: ignore[attr-defined]

        transcriber._emit_speaker_profile_max_exemplar_gate_status(1)

        self.assertEqual(len(emitted), 1)
        self.assertIn("max_exemplars=4", emitted[0])
        self.assertIn("session language 'en' is not zh", emitted[0])
        self.assertIn("using max_exemplars=1", emitted[0])

    def test_suppression_status_is_silent_for_default_or_zh_sessions(self) -> None:
        for max_exemplars, language_hint, effective in ((1, "en", 1), (4, "zh", 4)):
            with self.subTest(max_exemplars=max_exemplars, language_hint=language_hint):
                emitted: list[str] = []
                transcriber = _new_stub(max_exemplars, language_hint)
                transcriber._emit = emitted.append  # type: ignore[attr-defined]

                transcriber._emit_speaker_profile_max_exemplar_gate_status(effective)

                self.assertEqual(emitted, [])

    def test_suppression_status_reports_auto_for_empty_language_hint(self) -> None:
        emitted: list[str] = []
        transcriber = _new_stub(2, None)
        transcriber._emit = emitted.append  # type: ignore[attr-defined]

        transcriber._emit_speaker_profile_max_exemplar_gate_status(1)

        self.assertEqual(len(emitted), 1)
        self.assertIn("session language 'auto' is not zh", emitted[0])


if __name__ == "__main__":
    unittest.main()
