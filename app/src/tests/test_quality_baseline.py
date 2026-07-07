"""Round 0070 — unit tests for run_quality_baseline's pure aggregation/compare logic."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "diagnostics" / "run_quality_baseline.py"
    spec = importlib.util.spec_from_file_location("run_quality_baseline", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # dataclasses resolves cls.__module__ through sys.modules at class-creation time;
    # a spec-loaded module must be registered first or @dataclass raises AttributeError.
    sys.modules["run_quality_baseline"] = module
    spec.loader.exec_module(module)
    return module


qb = _load_module()


SRT = """1
00:00:01,000 --> 00:00:03,000
Hello world.

2
00:01:40,500 --> 00:01:42,000
Past the slice
"""


class SrtAndCerTests(unittest.TestCase):
    def test_srt_to_text_strips_indices_and_timestamps(self) -> None:
        self.assertEqual(qb.srt_to_text(SRT), "Hello world.\nPast the slice")

    def test_slice_truth_srt_drops_cues_starting_after_cutoff(self) -> None:
        sliced = qb.srt_to_text(qb._slice_truth_srt(SRT, 90.0))
        self.assertEqual(sliced, "Hello world.")

    def test_slice_zero_keeps_everything(self) -> None:
        self.assertEqual(qb._slice_truth_srt(SRT, 0.0), SRT)

    def test_normalize_for_cer_en_lowercases_and_strips(self) -> None:
        self.assertEqual(qb.normalize_for_cer("[spk_000] Hello, World!", "en"), "helloworld")

    def test_normalize_for_cer_zh_strips_fullwidth_punct_and_spaces(self) -> None:
        self.assertEqual(qb.normalize_for_cer("你好，世界。 测试", "zh"), "你好世界测试")

    def test_normalize_for_cer_zh_unifies_traditional_and_simplified(self) -> None:
        # Pipeline emits Traditional, zh-CN truth is Simplified; both must normalize equal
        # or the truth CER is dominated by script mismatch (~50 %) instead of ASR quality.
        self.assertEqual(
            qb.normalize_for_cer("這是樣片日記第一次解鎖非洲地區", "zh"),
            qb.normalize_for_cer("这是样片日记第一次解锁非洲地区", "zh"),
        )

    def test_truth_cer_zero_for_identical(self) -> None:
        self.assertEqual(qb.truth_cer("Hello world.", SRT.split("\n\n")[0], "en"), 0.0)

    def test_truth_cer_none_for_empty_reference(self) -> None:
        self.assertIsNone(qb.truth_cer("anything", "", "en"))

    def test_levenshtein_basic(self) -> None:
        self.assertEqual(qb.levenshtein("kitten", "sitting"), 3)


def _mk_run(case_id: str, *, exit_code: int = 0, completeness: float | None = 0.95,
            dup: str = "", cjk: int = 0, cer: float | None = 0.10) -> dict:
    return {
        "case_id": case_id,
        "exit_code": exit_code,
        "truth_cer": cer,
        "report": {"correctness": {
            "completeness": completeness, "dup": dup, "cjk_mid_spaces": cjk,
            "markers": 0, "bad_anchor": 0,
        }},
    }


def _baseline(runs: list[dict], *, cer_norm: str = "v2-opencc-t2s") -> dict:
    return {"meta": {"tier": "quick", "cer_norm_version": cer_norm}, "runs": runs}


class CompareBaselinesTests(unittest.TestCase):
    def test_no_previous_means_no_findings(self) -> None:
        self.assertEqual(qb.compare_baselines(_baseline([_mk_run("a")]), None), [])

    def test_identical_baselines_are_clean(self) -> None:
        cur, prev = _baseline([_mk_run("a")]), _baseline([_mk_run("a")])
        self.assertEqual(qb.compare_baselines(cur, prev), [])

    def test_pass_to_fail_flagged(self) -> None:
        cur = _baseline([_mk_run("a", exit_code=1)])
        prev = _baseline([_mk_run("a")])
        findings = qb.compare_baselines(cur, prev)
        self.assertTrue(any("PASS -> FAIL" in f for f in findings))

    def test_completeness_drop_beyond_threshold_flagged(self) -> None:
        cur = _baseline([_mk_run("a", completeness=0.90)])
        prev = _baseline([_mk_run("a", completeness=0.95)])
        findings = qb.compare_baselines(cur, prev)
        self.assertTrue(any("completeness" in f for f in findings))

    def test_small_completeness_wiggle_not_flagged(self) -> None:
        cur = _baseline([_mk_run("a", completeness=0.93)])
        prev = _baseline([_mk_run("a", completeness=0.95)])
        self.assertEqual(qb.compare_baselines(cur, prev), [])

    def test_new_dup_stacking_flagged(self) -> None:
        cur = _baseline([_mk_run("a", dup="重複重複")])
        prev = _baseline([_mk_run("a")])
        findings = qb.compare_baselines(cur, prev)
        self.assertTrue(any("dup-stacking" in f for f in findings))

    def test_cjk_spaces_need_both_abs_and_rel_growth(self) -> None:
        prev = _baseline([_mk_run("a", cjk=20)])
        # +6 absolute but only +30 % relative -> NOT flagged
        self.assertEqual(qb.compare_baselines(_baseline([_mk_run("a", cjk=26)]), prev), [])
        # +16 absolute and +80 % relative -> flagged
        findings = qb.compare_baselines(_baseline([_mk_run("a", cjk=36)]), prev)
        self.assertTrue(any("mid-spaces" in f for f in findings))

    def test_truth_cer_rise_flagged_and_missing_cer_ignored(self) -> None:
        prev = _baseline([_mk_run("a", cer=0.10)])
        findings = qb.compare_baselines(_baseline([_mk_run("a", cer=0.15)]), prev)
        self.assertTrue(any("truth CER" in f for f in findings))
        self.assertEqual(qb.compare_baselines(_baseline([_mk_run("a", cer=None)]), prev), [])

    def test_truth_cer_not_compared_across_norm_versions(self) -> None:
        prev = _baseline([_mk_run("a", cer=0.10)], cer_norm="v1-raw")
        cur = _baseline([_mk_run("a", cer=0.15)])
        self.assertEqual(qb.compare_baselines(cur, prev), [])

    def test_unknown_case_in_current_ignored(self) -> None:
        cur = _baseline([_mk_run("new-case", exit_code=1)])
        prev = _baseline([_mk_run("a")])
        self.assertEqual(qb.compare_baselines(cur, prev), [])


class TimeoutSecondsTests(unittest.TestCase):
    def test_paced_long_clip_gets_headroom(self) -> None:
        spec = qb.RunSpec("x", "d", "zh", "whisperx", True, 1.0, 600.0, 600.0)
        # 600 s paced blew the fixed 1200 s default on the founding full run
        self.assertEqual(spec.timeout_seconds(), 600.0 * 2 + 600.0)

    def test_paced_short_clip_keeps_default_floor(self) -> None:
        spec = qb.RunSpec("x", "d", "zh", "whisperx", True, 1.0, 90.0, 90.0)
        self.assertEqual(spec.timeout_seconds(), 1200.0)

    def test_unpaced_uses_flat_timeout(self) -> None:
        spec = qb.RunSpec("x", "d", "en", "whispercpp", False, 0.0, 0.0, 464.0)
        self.assertEqual(spec.timeout_seconds(), 1800.0)


class MatrixFloorTests(unittest.TestCase):
    def test_only_bn_has_lowered_completeness_floor(self) -> None:
        for spec in qb.QUICK_MATRIX + qb.FULL_MATRIX:
            if spec.case_id == "f3-bn-zh-whisperx-diar":
                self.assertEqual(spec.min_completeness, 0.75)
            else:
                self.assertEqual(spec.min_completeness, 0.85)


class FindPreviousBaselineTests(unittest.TestCase):
    def test_picks_latest_same_tier_excluding_current(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name, tag in [("20260101T000000Z_quick", "old"),
                              ("20260301T000000Z_quick", "newer"),
                              ("20260401T000000Z_full", "wrong-tier")]:
                d = root / name
                d.mkdir()
                (d / "baseline.json").write_text(
                    json.dumps({"meta": {"tag": tag}, "runs": []}), encoding="utf-8")
            current = root / "20260501T000000Z_quick"
            current.mkdir()
            (current / "baseline.json").write_text("{}", encoding="utf-8")
            found = qb.find_previous_baseline(root, "quick", current)
            self.assertIsNotNone(found)
            self.assertEqual(found["meta"]["tag"], "newer")

    def test_none_when_no_previous(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(qb.find_previous_baseline(Path(td), "quick", Path(td) / "x"))


if __name__ == "__main__":
    unittest.main()
