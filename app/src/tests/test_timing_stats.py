from __future__ import annotations

from pathlib import Path
import sys
import unittest

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice2text.pipeline.timing_stats import TimingAggregator, format_stage_breakdown, percentile


class PercentileTests(unittest.TestCase):
    def test_empty_is_zero(self) -> None:
        self.assertEqual(percentile([], 50), 0.0)

    def test_single_value(self) -> None:
        self.assertEqual(percentile([2.5], 95), 2.5)

    def test_interpolated_percentiles(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(percentile(values, 0), 1.0)
        self.assertAlmostEqual(percentile(values, 50), 2.5)
        self.assertAlmostEqual(percentile(values, 100), 4.0)
        self.assertAlmostEqual(percentile(values, 95), 3.85)

    def test_unsorted_input_is_ordered(self) -> None:
        self.assertAlmostEqual(percentile([4.0, 1.0, 3.0, 2.0], 50), 2.5)


class TimingAggregatorTests(unittest.TestCase):
    def _two_windows(self) -> TimingAggregator:
        agg = TimingAggregator()
        agg.add_window(
            timing={"window_total_seconds": 1.0, "transcribe_seconds": 0.8, "merge_seconds": 0.1},
            hop_seconds=2.0,
            transcription_meta={"provider_timing": {"asr_seconds": 0.5, "align_seconds": 0.2}},
        )
        agg.add_window(
            timing={"window_total_seconds": 3.0, "transcribe_seconds": 2.5, "merge_seconds": 0.2},
            hop_seconds=2.0,
            transcription_meta={"provider_timing": {"asr_seconds": 2.0, "align_seconds": 0.3}},
        )
        return agg

    def test_realtime_factor_is_processing_over_audio(self) -> None:
        summary = self._two_windows().summary()
        self.assertEqual(summary["window_count"], 2)
        self.assertAlmostEqual(summary["audio_seconds"], 4.0)
        self.assertAlmostEqual(summary["processing_seconds"], 4.0)
        # sum(window_total)=4.0 over count*hop=4.0 -> 1.0x (right at the keep-up line)
        self.assertAlmostEqual(summary["realtime_factor"], 1.0)

    def test_window_total_and_stage_percentiles(self) -> None:
        summary = self._two_windows().summary()
        self.assertAlmostEqual(summary["window_total"]["p50"], 2.0)
        self.assertAlmostEqual(summary["window_total"]["max"], 3.0)
        self.assertAlmostEqual(summary["stages"]["transcribe_seconds"]["p50"], 1.65)
        self.assertAlmostEqual(summary["stages"]["wx_asr_seconds"]["p50"], 1.25)
        self.assertEqual(summary["stages"]["transcribe_seconds"]["n"], 2)

    def test_dominant_stage_is_largest_p50(self) -> None:
        summary = self._two_windows().summary()
        self.assertEqual(summary["dominant_stage"], "transcribe_seconds")

    def test_missing_provider_timing_is_tolerated(self) -> None:
        agg = TimingAggregator()
        agg.add_window(timing={"window_total_seconds": 0.5}, hop_seconds=1.5)
        summary = agg.summary()
        self.assertEqual(summary["window_count"], 1)
        self.assertAlmostEqual(summary["realtime_factor"], 0.5 / 1.5, places=4)
        self.assertNotIn("wx_asr_seconds", summary["stages"])

    def test_empty_aggregator_summary(self) -> None:
        summary = TimingAggregator().summary()
        self.assertEqual(summary["window_count"], 0)
        self.assertEqual(summary["realtime_factor"], 0.0)
        self.assertEqual(summary["dominant_stage"], "")


class FormatStageBreakdownTests(unittest.TestCase):
    def test_empty_or_invalid_is_blank(self) -> None:
        self.assertEqual(format_stage_breakdown({}), "")
        self.assertEqual(format_stage_breakdown(None), "")  # type: ignore[arg-type]

    def test_sorted_by_p50_desc_with_all_fields(self) -> None:
        stages = {
            "merge_seconds": {"p50": 0.1, "p95": 0.2, "max": 0.3, "n": 2},
            "transcribe_seconds": {"p50": 1.5, "p95": 2.4, "max": 2.5, "n": 2},
            "wx_align_seconds": {"p50": 0.25, "p95": 0.3, "max": 0.3, "n": 2},
        }
        out = format_stage_breakdown(stages)
        # transcribe (1.5) before align (0.25) before merge (0.1)
        self.assertLess(out.index("transcribe_seconds"), out.index("wx_align_seconds"))
        self.assertLess(out.index("wx_align_seconds"), out.index("merge_seconds"))
        self.assertIn("transcribe_seconds=1.5000/2.4000/2.5000s (n=2)", out)

    def test_breakdown_matches_aggregator_stages(self) -> None:
        agg = TimingAggregator()
        agg.add_window(
            timing={"window_total_seconds": 1.0, "transcribe_seconds": 0.8, "merge_seconds": 0.1},
            hop_seconds=2.0,
            transcription_meta={"provider_timing": {"asr_seconds": 0.5}},
        )
        out = format_stage_breakdown(agg.summary()["stages"])
        self.assertIn("transcribe_seconds=", out)
        self.assertIn("wx_asr_seconds=", out)


if __name__ == "__main__":
    unittest.main()
