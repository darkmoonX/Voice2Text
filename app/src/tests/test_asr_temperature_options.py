from __future__ import annotations

import unittest

from voice2text.stt.whisperx_provider import WhisperXTranscriber, _parse_temperature_schedule


def _provider(
    *,
    temperatures_raw: str = "",
    log_prob: float | None = None,
    compression_ratio: float | None = None,
    no_speech: float | None = None,
) -> WhisperXTranscriber:
    provider = WhisperXTranscriber.__new__(WhisperXTranscriber)
    provider._beam_size = 5
    provider._asr_temperatures_raw = temperatures_raw
    provider._asr_temperatures = _parse_temperature_schedule(temperatures_raw)
    provider._asr_log_prob_threshold = log_prob
    provider._asr_compression_ratio_threshold = compression_ratio
    provider._asr_no_speech_threshold = no_speech
    provider._whisperx = object()  # no DEFAULT_ASR_OPTIONS attr -> defaults resolve to {}
    provider._emit = lambda message: None
    return provider


class ParseTemperatureScheduleTests(unittest.TestCase):
    def test_empty_and_none_return_none(self) -> None:
        self.assertIsNone(_parse_temperature_schedule(""))
        self.assertIsNone(_parse_temperature_schedule("   "))
        self.assertIsNone(_parse_temperature_schedule(None))

    def test_valid_schedule_parses_sorted(self) -> None:
        self.assertEqual(_parse_temperature_schedule("0.0,0.2,0.4"), [0.0, 0.2, 0.4])
        self.assertEqual(_parse_temperature_schedule("0.4, 0.0, 0.2"), [0.0, 0.2, 0.4])

    def test_garbage_returns_none(self) -> None:
        self.assertIsNone(_parse_temperature_schedule("0.0,abc"))
        self.assertIsNone(_parse_temperature_schedule("lots of nonsense"))
        self.assertIsNone(_parse_temperature_schedule(",,,"))

    def test_values_clamped_and_deduped(self) -> None:
        self.assertEqual(_parse_temperature_schedule("0.0,1.5,-0.3,1.5"), [0.0, 1.0])

    def test_missing_greedy_pass_prepends_zero(self) -> None:
        self.assertEqual(_parse_temperature_schedule("0.2,0.4"), [0.0, 0.2, 0.4])


class BuildAsrOptionsTests(unittest.TestCase):
    def test_no_overrides_is_byte_identical(self) -> None:
        provider = _provider()
        self.assertEqual(provider._build_asr_options(), {"beam_size": 5})

    def test_unparsable_schedule_is_ignored(self) -> None:
        provider = _provider(temperatures_raw="not-a-schedule")
        self.assertEqual(provider._build_asr_options(), {"beam_size": 5})

    def test_temperatures_override_lands(self) -> None:
        provider = _provider(temperatures_raw="0.0,0.2,0.4")
        options = provider._build_asr_options()
        self.assertEqual(options["temperatures"], [0.0, 0.2, 0.4])

    def test_threshold_overrides_land(self) -> None:
        provider = _provider(log_prob=-0.8, compression_ratio=2.0, no_speech=0.5)
        options = provider._build_asr_options()
        self.assertEqual(options["log_prob_threshold"], -0.8)
        self.assertEqual(options["compression_ratio_threshold"], 2.0)
        self.assertEqual(options["no_speech_threshold"], 0.5)
        self.assertNotIn("temperatures", options)


if __name__ == "__main__":
    unittest.main()
