"""Unit tests for incremental subtitle merge de-dup behavior."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice2text.pipeline.subtitle_assembler import SubtitleAssembler, _WordState


class SubtitleAssemblerTests(unittest.TestCase):
    def test_stable_tail_reduces_english_overlap_repeat(self) -> None:
        assembler = SubtitleAssembler()
        out1 = assembler.merge_incremental_text(
            "we are testing overlap merge behavior",
            overlap_merge_method="stable-tail",
            segment_seconds=6.0,
            hop_seconds=1.5,
        )
        out2 = assembler.merge_incremental_text(
            "overlap merge behavior for live subtitle output",
            overlap_merge_method="stable-tail",
            segment_seconds=6.0,
            hop_seconds=1.5,
        )
        self.assertIn("overlap merge behavior", out2)
        self.assertNotIn("behavior behavior", out2)
        self.assertTrue(len(out2) >= len(out1))

    def test_commit_on_break_reduces_cjk_repeat_suffix(self) -> None:
        assembler = SubtitleAssembler()
        assembler.merge_incremental_text(
            "這是中文測試字幕合併",
            overlap_merge_method="commit-on-break",
            segment_seconds=6.0,
            hop_seconds=1.5,
        )
        out = assembler.merge_incremental_text(
            "中文測試字幕合併與去重",
            overlap_merge_method="commit-on-break",
            segment_seconds=6.0,
            hop_seconds=1.5,
        )
        self.assertIn("中文測試字幕合併", out)
        self.assertNotIn("合併合併", out)

    def test_cjk_join_does_not_force_space(self) -> None:
        assembler = SubtitleAssembler()
        merged = assembler._merge_by_exact_overlap("這是一段中文", "字幕")
        self.assertEqual(merged, "這是一段中文字幕")

    def test_non_cjk_exact_overlap_keeps_word_boundary_space(self) -> None:
        assembler = SubtitleAssembler()

        merged = assembler._merge_by_exact_overlap(
            "this",
            "so we continue",
        )

        self.assertEqual(merged, "this so we continue")
        self.assertNotIn("thiso", merged)

    def test_non_cjk_fuzzy_overlap_does_not_slice_words_mid_token(self) -> None:
        assembler = SubtitleAssembler()

        merged = assembler._merge_by_exact_overlap(
            "Today we're hiking the exact same trail same distance",
            "the exact same trail same elevation gain",
        )
        summary = assembler.get_last_merge_diagnostics() or assembler._last_overlap_summary

        self.assertEqual(summary["method"], "word-fuzzy-replace")
        self.assertIn("same elevation gain", merged)
        self.assertNotIn("samelevation", merged)

    def test_non_cjk_overlap_keeps_cjk_path_byte_identical(self) -> None:
        assembler = SubtitleAssembler()

        merged = assembler._merge_by_exact_overlap("這是一段中文", "字幕")

        self.assertEqual(merged, "這是一段中文字幕")

    def test_non_cjk_word_merge_preserves_pause_break_blank_line(self) -> None:
        # Round 0008 pause break: a same-speaker blank-line boundary must survive
        # the word-overlap merge. The token joiner previously collapsed every
        # newline run to a single '\n', so the re-emitted [spk_000] marker got
        # merged inline and the whole pause-separated turn flattened to one line.
        assembler = SubtitleAssembler()
        assembler.set_language_context("en")

        merged = assembler._merge_by_exact_overlap(
            "[spk_000] hello there\n\n[spk_000] after a long pause",
            "after a long pause we continue",
        )

        self.assertIn("\n\n", merged)
        self.assertEqual(merged.count("[spk_000]"), 2)
        self.assertIn("after a long pause we continue", merged)

    def test_non_cjk_jittered_same_word_merges_into_single_state(self) -> None:
        # English wav2vec2 alignment places one spoken word at jittered timestamps
        # across windows (here abs 5.0 / 5.6 / 5.3, gaps > the 0.35s CJK gate). The
        # loose non-CJK match tolerance must merge them into one accumulating state
        # so it promotes exactly once -- not dropped (under-confirmed) and not
        # rendered as an adjacent duplicate (`completely completely`).
        assembler = SubtitleAssembler()
        assembler.set_language_context("en")
        for elapsed, abs_start in ((0.0, 5.0), (2.0, 5.6), (4.0, 5.3)):
            assembler.merge_incremental_text(
                "completely",
                overlap_merge_method="stable-tail",
                segment_seconds=10.0,
                hop_seconds=2.0,
                transcription_meta={
                    "elapsed_seconds": elapsed,
                    "token_timestamps": [
                        {
                            "word": "completely",
                            "start": abs_start - elapsed,
                            "end": abs_start - elapsed + 0.4,
                            "score": 0.9,
                        },
                    ],
                },
            )
        committed = (assembler.get_history_text() + " " + assembler.get_stable_text())
        self.assertEqual(committed.split().count("completely"), 1)

    def test_cjk_without_alignment_jittered_same_word_merges_into_single_state(self) -> None:
        assembler = SubtitleAssembler()
        assembler.set_language_context("zh")
        for elapsed, abs_start in ((0.0, 5.0), (2.0, 5.5), (4.0, 5.3)):
            assembler.merge_incremental_text(
                "团队",
                overlap_merge_method="stable-tail",
                segment_seconds=10.0,
                hop_seconds=2.0,
                transcription_meta={
                    "elapsed_seconds": elapsed,
                    "alignment_enabled": False,
                    "token_timestamps": [
                        {
                            "word": "团队",
                            "start": abs_start - elapsed,
                            "end": abs_start - elapsed + 0.4,
                            "score": 0.9,
                        },
                    ],
                },
            )
        committed = assembler.get_history_text() + assembler.get_stable_text()
        self.assertEqual(committed.count("团队"), 1)

    def test_cjk_with_alignment_jittered_same_word_stays_under_confirmed(self) -> None:
        assembler = SubtitleAssembler()
        assembler.set_language_context("zh")
        for elapsed, abs_start in ((0.0, 5.0), (2.0, 5.5), (4.0, 5.3)):
            assembler.merge_incremental_text(
                "团队",
                overlap_merge_method="stable-tail",
                segment_seconds=10.0,
                hop_seconds=2.0,
                transcription_meta={
                    "elapsed_seconds": elapsed,
                    "alignment_enabled": True,
                    "token_timestamps": [
                        {
                            "word": "团队",
                            "start": abs_start - elapsed,
                            "end": abs_start - elapsed + 0.4,
                            "score": 0.9,
                        },
                    ],
                },
            )
        committed = assembler.get_history_text() + assembler.get_stable_text()
        self.assertEqual(committed.count("团队"), 0)

    def test_cjk_fuzzy_overlap_replaces_revised_sliding_window(self) -> None:
        assembler = SubtitleAssembler()
        first = "這是漾天日記第一次解鎖非洲地區眼前這個國家叫做"
        second = "這是樣片日記第一次解鎖非洲地區眼前這個國家叫做馬達加斯加"

        merged = assembler._merge_by_exact_overlap(first, second)
        summary = assembler.get_last_merge_diagnostics()

        self.assertEqual(merged, second)
        self.assertNotIn("眼前這個國家叫做這是", merged)
        self.assertEqual(summary, {})

    def test_cjk_overlap_compares_simplified_and_traditional_as_same_phrase(self) -> None:
        assembler = SubtitleAssembler()
        first = "當你仔細觀察之後你會發現"
        second = "当你仔细观察之后你会发现新的线索"

        merged = assembler._merge_by_exact_overlap(first, second)

        self.assertEqual(merged, second)
        self.assertNotIn(f"{first}{second}", merged)
        self.assertEqual(merged.count("发现"), 1)

    def test_cjk_overlap_compact_falls_back_when_opencc_is_unavailable(self) -> None:
        from voice2text.stt import audio_utils

        with mock.patch.object(audio_utils, "OpenCC", None):
            compact = SubtitleAssembler._compact_for_overlap("當 你 仔 細")

        self.assertEqual(compact, "當你仔細")

    def test_cjk_short_revision_overlap_merges_high_similarity_tail(self) -> None:
        assembler = SubtitleAssembler()

        merged = assembler._merge_by_exact_overlap("你可以随意使", "你可以意使用")

        self.assertEqual(merged, "你可以随意使用")

    def test_cjk_short_revision_overlap_does_not_merge_distinct_low_similarity_phrases(self) -> None:
        assembler = SubtitleAssembler()

        merged = assembler._merge_by_exact_overlap("今天去吃飯", "明天去看海")

        self.assertEqual(merged, "今天去吃飯明天去看海")

    def test_cjk_short_revision_overlap_does_not_zip_simplified_traditional_replace(self) -> None:
        assembler = SubtitleAssembler()

        merged = assembler._merge_by_exact_overlap("这个国家随处", "這個國家隨處可見")

        self.assertEqual(merged, "這個國家隨處可見")
        self.assertNotIn("国國", merged)
        self.assertNotIn("随處", merged)

    def test_cjk_short_revision_overlap_does_not_zip_misaligned_simplified_traditional(self) -> None:
        assembler = SubtitleAssembler()

        # The traditional revision is misaligned against the simplified tail (an
        # extra measure word makes the replace block '国' vs '個國'); the folded
        # key of the shorter side is contained in the longer, so keep only the
        # superset instead of zipping (regression: '国家随处' -> '国個國家隨處').
        merged = assembler._merge_by_exact_overlap("还有这个国家随处", "這個國家隨處可")

        self.assertEqual(merged, "还有这个国家隨處可")
        self.assertNotIn("国個國", merged)
        self.assertEqual(merged.count("国家") + merged.count("國家"), 1)
        # The folded-key containment rule keeps the superset side, not a zip.
        self.assertEqual(
            SubtitleAssembler._short_common_supersequence("国家", "個國家"), "個國家"
        )

    def test_cjk_short_revision_overlap_keeps_long_divergent_remainder_appended(self) -> None:
        assembler = SubtitleAssembler()

        merged = assembler._merge_by_exact_overlap("你可以随意使", "你可以意使用全新的工具和路线")

        self.assertEqual(merged, "你可以随意使你可以意使用全新的工具和路线")

    def test_marker_boundary_overlap_dedupes_text_before_marker(self) -> None:
        assembler = SubtitleAssembler()

        merged = assembler._merge_by_exact_overlap("印度洋的丰沛 [spk_000]", "[spk_000] 丰沛水气")

        self.assertEqual(merged, "印度洋的\n[spk_000] 丰沛水气")
        self.assertEqual(merged.count("[spk_000]"), 1)
        self.assertNotIn("丰沛\n[spk_000] 丰沛", merged)
        for line in merged.splitlines():
            if "[spk_000]" in line:
                self.assertTrue(line.startswith("[spk_000]"))

    # --- Round 0005 skeletons: interior (non-boundary-aligned) cross-window dedup ---
    # The two `expectedFailure` tests below assert the TARGET behavior and fail on
    # today's code (the interior duplicate is appended). When 0005 is implemented,
    # remove the `@unittest.expectedFailure` decorator so they become real green
    # tests. The two guard tests already pass today and must KEEP passing after 0005
    # (they protect against over-merging distinct / intentional repeats).

    def test_interior_duplicate_head_within_line_is_deduped(self) -> None:
        assembler = SubtitleAssembler()

        # Faithful to output/03 `…可以看到这个马达加好您看到这个达加…`: `看到这个` is
        # interior to base (base suffix is `好您`), so exact/fuzzy suffix alignment
        # misses and incoming's `看到这个` head is duplicated. Note the divergent base
        # remnant after the interior match (`马达加好您`) is ~5 chars — calibrate the
        # short-remnant guard so this real case still fires.
        merged = assembler._merge_by_exact_overlap("可以看到这个马达加好您", "看到这个达加")

        self.assertLessEqual(merged.count("看到这个"), 1)
        self.assertNotIn("看到这个马达加好您看到这个", merged)

    def test_interior_dedup_keeps_distinct_repeat_with_real_content_between(self) -> None:
        assembler = SubtitleAssembler()

        # The repeated head `他们的饮食结构` (>= min match length) is followed by
        # genuinely distinct, long content on both sides — this is NOT an ASR
        # mis-revision, so the short-remnant guard must keep both occurrences.
        merged = assembler._merge_by_exact_overlap(
            "他们的饮食结构其实和我们很类似", "他们的饮食结构和欧洲完全不同"
        )

        self.assertIn("其实和我们很类似", merged)
        self.assertIn("和欧洲完全不同", merged)

    def test_interior_dedup_ignores_short_function_word_match(self) -> None:
        assembler = SubtitleAssembler()

        # The only interior match is the short function word `这个` (< min match
        # length); it must not trigger a merge.
        merged = assembler._merge_by_exact_overlap("我们现在在这个城市观光", "这个地方很漂亮")

        self.assertIn("城市观光", merged)
        self.assertIn("这个地方很漂亮", merged)

    # --- Round 0006: leading-marker strip dedup against an unmarked committed tail ---
    def test_leading_marker_content_dedupes_against_unmarked_base(self) -> None:
        assembler = SubtitleAssembler()

        # Faithful to output/07 window_index 9: a speaker-turn boundary inside the
        # window overlap re-transcribes the boundary audio. `base` is the previous
        # speaker's committed tail (NO marker); `incoming` carries a fresh leading
        # marker and re-states `印度洋的丰沛`. The marker-stripped content shares a
        # 6-char exact overlap with base, so the merge must dedup it (not append),
        # keeping the marker line-start anchored.
        base = "这是样片日记第一次解锁非洲地区眼前这个国家叫做马达加斯加印度洋的丰沛"
        merged = assembler._merge_by_exact_overlap(
            base, "[spk_000] 印度洋的丰沛水池和中高周低的地形"
        )

        self.assertEqual(merged.count("印度洋的"), 1)
        self.assertEqual(merged.count("[spk_000]"), 1)
        self.assertIn("水池", merged)
        for line in merged.splitlines():
            if "[spk_000]" in line:
                self.assertTrue(line.startswith("[spk_000]"))

    def test_leading_marker_content_keeps_unrelated_new_speaker_content(self) -> None:
        assembler = SubtitleAssembler()

        merged = assembler._merge_by_exact_overlap(
            "上一位講者正在介紹馬達加斯加",
            "[spk_000] 接下來我們切換到另一個完全不同的主題",
        )

        self.assertEqual(merged.count("[spk_000]"), 1)
        self.assertIn("接下來我們切換到另一個完全不同的主題", merged)
        for line in merged.splitlines():
            if "[spk_000]" in line:
                self.assertTrue(line.startswith("[spk_000]"))

    def test_cjk_word_match_folds_simplified_and_traditional_keys(self) -> None:
        assembler = SubtitleAssembler()
        simplified = _WordState("团队", 10.0, 10.4, 0.95, 1, 10.4)
        traditional = _WordState("團隊", 10.02, 10.42, 0.93, 1, 10.42)

        self.assertTrue(assembler._word_matches(simplified, traditional))

    def test_non_cjk_low_alignment_scores_reach_agreement(self) -> None:
        # Regression: English wav2vec2 alignment scores are systematically low
        # (median ~0.37). A CJK-calibrated 0.60 gate dropped ~85% of legitimate
        # English words before they could reach the agreement count, collapsing
        # the realtime transcript to a word-salad. The non-CJK gate (0.10) must
        # let mid-confidence words through so they promote across windows.
        assembler = SubtitleAssembler()
        assembler.set_language_context("en")
        meta = {
            "elapsed_seconds": 0.0,
            "token_timestamps": [
                {"word": "hello", "start": 1.0, "end": 1.4, "score": 0.42},
                {"word": "world", "start": 1.5, "end": 1.9, "score": 0.38},
            ],
        }
        for _ in range(3):  # three overlapping windows -> reach agreement count
            assembler.merge_incremental_text(
                "hello world",
                overlap_merge_method="stable-tail",
                segment_seconds=10.0,
                hop_seconds=2.0,
                transcription_meta=meta,
            )
        committed = (assembler.get_history_text() + " " + assembler.get_stable_text())
        self.assertIn("hello", committed)
        self.assertIn("world", committed)

    def test_cjk_keeps_strict_score_gate(self) -> None:
        # The strict 0.60 gate is unchanged for CJK: genuinely low-confidence CJK
        # alignments (which are rare — CJK alignment median ~0.96) stay dropped.
        assembler = SubtitleAssembler()
        assembler.set_language_context("zh")
        meta = {
            "elapsed_seconds": 0.0,
            "token_timestamps": [
                {"word": "你好", "start": 1.0, "end": 1.4, "score": 0.42},
            ],
        }
        for _ in range(3):
            assembler.merge_incremental_text(
                "你好",
                overlap_merge_method="stable-tail",
                segment_seconds=10.0,
                hop_seconds=2.0,
                transcription_meta=meta,
            )
        committed = (assembler.get_history_text() + " " + assembler.get_stable_text()).strip()
        self.assertEqual(committed, "")

    def test_cjk_word_state_merge_does_not_interleave_simplified_traditional_tokens(self) -> None:
        assembler = SubtitleAssembler()
        assembler.set_language_context("zh")
        simplified_meta = {
            "elapsed_seconds": 0.0,
            "token_timestamps": [
                {"word": "作为", "start": 0.0, "end": 0.2, "score": 0.95},
                {"word": "团队", "start": 0.2, "end": 0.5, "score": 0.95},
            ],
        }
        traditional_meta = {
            "elapsed_seconds": 0.0,
            "token_timestamps": [
                {"word": "作為", "start": 0.01, "end": 0.21, "score": 0.94},
                {"word": "團隊", "start": 0.21, "end": 0.51, "score": 0.94},
            ],
        }

        for text, meta in (
            ("作为团队", simplified_meta),
            ("作為團隊", traditional_meta),
            ("作为团队", simplified_meta),
        ):
            assembler.merge_incremental_text(
                text,
                overlap_merge_method="stable-tail",
                segment_seconds=9.6,
                hop_seconds=1.2,
                transcription_meta=meta,
            )

        stable = assembler.get_stable_text()

        self.assertEqual(stable, "作为团队")
        self.assertNotIn("团團", stable)
        self.assertNotIn("队隊", stable)
        self.assertEqual(len(assembler.get_stable_state()), 2)

    def test_cjk_word_match_falls_back_when_opencc_is_unavailable(self) -> None:
        from voice2text.stt import audio_utils

        with mock.patch.object(audio_utils, "OpenCC", None):
            simplified_key = SubtitleAssembler._normalize_word("团队")
            traditional_key = SubtitleAssembler._normalize_word("團隊")

        self.assertEqual(simplified_key, "团队")
        self.assertEqual(traditional_key, "團隊")
        self.assertNotEqual(simplified_key, traditional_key)

    def test_visible_rolling_fuzzy_overlap_prevents_raw_window_stacking(self) -> None:
        assembler = SubtitleAssembler()
        assembler.set_language_context("zh")
        out1 = assembler.merge_incremental_text(
            "這是漾天日記第一次解鎖非洲地區眼前這個國家叫做",
            overlap_merge_method="stable-tail",
            segment_seconds=9.6,
            hop_seconds=1.2,
        )
        out2 = assembler.merge_incremental_text(
            "這是樣片日記第一次解鎖非洲地區眼前這個國家叫做馬達加斯加",
            overlap_merge_method="stable-tail",
            segment_seconds=9.6,
            hop_seconds=1.2,
        )
        diagnostics = assembler.get_last_merge_diagnostics()

        self.assertIn("馬達加斯加", out2)
        self.assertLessEqual(out2.count("第一次解鎖非洲地區"), 1)
        self.assertGreaterEqual(len(out2), len(out1))
        self.assertEqual(diagnostics["rolling_overlap"]["method"], "fuzzy-replace")
        self.assertEqual(diagnostics["rolling_base_source"], "previous_rolling")

    def test_cjk_visible_merge_adds_space_after_pause_gap(self) -> None:
        assembler = SubtitleAssembler()
        assembler.set_language_context("zh")
        assembler.set_cjk_no_space_gap_seconds(0.2)

        merged = assembler.merge_incremental_text(
            "今天很好我們去吃飯",
            overlap_merge_method="stable-tail",
            segment_seconds=6.0,
            hop_seconds=1.5,
            transcription_meta={
                "elapsed_seconds": 0.0,
                "token_timestamps": [
                    {"word": "今天", "start": 0.0, "end": 0.4, "score": 0.9},
                    {"word": "很好", "start": 0.45, "end": 0.8, "score": 0.9},
                    {"word": "我們", "start": 1.15, "end": 1.5, "score": 0.9},
                    {"word": "去", "start": 1.55, "end": 1.7, "score": 0.9},
                    {"word": "吃飯", "start": 1.75, "end": 2.1, "score": 0.9},
                ],
            },
        )

        self.assertEqual(merged, "今天很好 我們去吃飯")

    def test_cjk_visible_merge_keeps_short_gap_compact(self) -> None:
        assembler = SubtitleAssembler()
        assembler.set_language_context("zh")
        assembler.set_cjk_no_space_gap_seconds(0.2)

        merged = assembler.merge_incremental_text(
            "今天很好我們去吃飯",
            overlap_merge_method="stable-tail",
            segment_seconds=6.0,
            hop_seconds=1.5,
            transcription_meta={
                "elapsed_seconds": 0.0,
                "token_timestamps": [
                    {"word": "今天", "start": 0.0, "end": 0.4, "score": 0.9},
                    {"word": "很好", "start": 0.45, "end": 0.8, "score": 0.9},
                    {"word": "我們", "start": 0.95, "end": 1.3, "score": 0.9},
                    {"word": "去", "start": 1.35, "end": 1.5, "score": 0.9},
                    {"word": "吃飯", "start": 1.55, "end": 1.9, "score": 0.9},
                ],
            },
        )

        self.assertEqual(merged, "今天很好我們去吃飯")

    def test_cjk_visible_merge_falls_back_when_tokens_are_unavailable(self) -> None:
        assembler = SubtitleAssembler()
        assembler.set_language_context("zh")

        merged = assembler.merge_incremental_text(
            "這是一段很長的中文字幕內容用來測試沒有時間戳的時候也能有基本分隔",
            overlap_merge_method="stable-tail",
            segment_seconds=6.0,
            hop_seconds=1.5,
            transcription_meta={"elapsed_seconds": 0.0, "token_timestamps": []},
        )
        summary = assembler.get_debug_summary()["cjk_spacing"]

        self.assertNotIn("\n", merged)
        self.assertIn(" ", merged)
        self.assertGreaterEqual(int(summary["fallback_spaces"]), 1)
        self.assertEqual(int(summary["fallback_line_breaks"]), 0)
        self.assertEqual(summary["reason"], "no_tokens")

    def test_cjk_visible_merge_spaces_long_compact_text_when_gaps_are_tiny(self) -> None:
        assembler = SubtitleAssembler()
        assembler.set_language_context("zh")
        assembler.set_cjk_no_space_gap_seconds(0.2)
        text = "這是一段很長的中文字幕內容用來測試時間戳全部連在一起時仍然需要可讀換行"
        tokens = []
        cursor = 0.0
        for char in text:
            tokens.append({"word": char, "start": cursor, "end": cursor + 0.02, "score": 0.9})
            cursor += 0.02

        merged = assembler.merge_incremental_text(
            text,
            overlap_merge_method="stable-tail",
            segment_seconds=6.0,
            hop_seconds=1.5,
            transcription_meta={"elapsed_seconds": 0.0, "token_timestamps": tokens},
        )
        summary = assembler.get_debug_summary()["cjk_spacing"]

        self.assertNotIn("\n", merged)
        self.assertIn(" ", merged)
        self.assertEqual(summary["reason"], "inserted")
        self.assertGreaterEqual(int(summary["fallback_spaces"]), 1)
        self.assertEqual(int(summary["fallback_line_breaks"]), 0)

    def test_cjk_spacing_summary_reports_pause_insertions(self) -> None:
        assembler = SubtitleAssembler()
        assembler.set_language_context("zh")
        assembler.set_cjk_no_space_gap_seconds(0.2)

        assembler.merge_incremental_text(
            "今天很好我們去吃飯",
            overlap_merge_method="stable-tail",
            segment_seconds=6.0,
            hop_seconds=1.5,
            transcription_meta={
                "elapsed_seconds": 0.0,
                "token_timestamps": [
                    {"word": "今天", "start": 0.0, "end": 0.4, "score": 0.9},
                    {"word": "很好", "start": 0.45, "end": 0.8, "score": 0.9},
                    {"word": "我們", "start": 1.15, "end": 1.5, "score": 0.9},
                ],
            },
        )
        summary = assembler.get_debug_summary()["cjk_spacing"]

        self.assertEqual(summary["reason"], "inserted")
        self.assertEqual(int(summary["pause_spaces"]), 1)
        self.assertGreater(float(summary["max_gap_seconds"]), 0.2)

    def test_visible_rolling_text_is_separate_from_full_history(self) -> None:
        assembler = SubtitleAssembler()
        meta = {
            "elapsed_seconds": 0.0,
            "token_timestamps": [
                {"word": "hello", "start": 0.0, "end": 0.4, "score": 0.95},
            ],
        }
        for _ in range(3):
            assembler.merge_incremental_text(
                "hello",
                overlap_merge_method="stable-tail",
                segment_seconds=6.0,
                hop_seconds=1.5,
                transcription_meta=meta,
            )

        merged = assembler.merge_incremental_text(
            "next",
            overlap_merge_method="stable-tail",
            segment_seconds=6.0,
            hop_seconds=1.5,
            transcription_meta={
                "elapsed_seconds": 10.0,
                "token_timestamps": [
                    {"word": "next", "start": 0.0, "end": 0.4, "score": 0.95},
                ],
            },
        )
        diagnostics = assembler.get_last_merge_diagnostics()

        self.assertIn("hello", assembler.get_history_text())
        self.assertIn("next", merged)
        self.assertEqual(float(diagnostics["history_render_seconds"]), 0.0)
        self.assertGreaterEqual(int(diagnostics["history_count_after"]), 1)
        self.assertGreaterEqual(int(diagnostics["rolling_base_chars"]), len("hello"))

    def test_history_flush_uses_bounded_tail_dedupe_for_long_history(self) -> None:
        assembler = SubtitleAssembler()
        assembler._history_words = [
            _WordState(f"h{i}", i * 0.2, i * 0.2 + 0.08, 0.95, 1, i * 0.2 + 0.08)
            for i in range(2500)
        ]
        assembler._stable_words = [
            _WordState("tail", 500.00, 500.20, 0.90, 3, 500.20),
            _WordState("tail", 500.03, 500.23, 0.92, 3, 500.23),
            _WordState("next", 500.30, 500.50, 0.95, 3, 500.50),
        ]

        moved = assembler._flush_stable_to_history(501.0)
        summary = assembler.get_debug_summary()["history_dedupe"]

        self.assertEqual(len(moved), 3)
        self.assertEqual(summary["mode"], "tail")
        self.assertLessEqual(int(summary["input_count"]), 170)
        self.assertEqual(int(summary["prefix_count"]), 2340)
        self.assertIn("tail", assembler.get_history_text())

    def test_history_flush_falls_back_to_full_dedupe_for_out_of_order_words(self) -> None:
        assembler = SubtitleAssembler()
        assembler._history_words = [
            _WordState(f"h{i}", i * 0.2, i * 0.2 + 0.08, 0.95, 1, i * 0.2 + 0.08)
            for i in range(2500)
        ]
        assembler._stable_words = [
            _WordState("old", 1.0, 1.2, 0.95, 3, 1.2),
        ]

        assembler._flush_stable_to_history(2.0)
        summary = assembler.get_debug_summary()["history_dedupe"]

        self.assertEqual(summary["mode"], "full")
        self.assertGreater(int(summary["input_count"]), 2500)

    def test_ui_committed_history_stays_when_raw_window_moves_forward(self) -> None:
        assembler = SubtitleAssembler()
        stable_meta = {
            "elapsed_seconds": 0.0,
            "token_timestamps": [
                {"word": "開", "start": 0.0, "end": 0.1, "score": 0.95},
                {"word": "頭", "start": 0.1, "end": 0.2, "score": 0.95},
            ],
        }
        for _ in range(3):
            assembler.merge_incremental_text(
                "開頭",
                overlap_merge_method="stable-tail",
                segment_seconds=6.0,
                hop_seconds=1.5,
                transcription_meta=stable_meta,
            )

        merged = assembler.merge_incremental_text(
            "後續",
            overlap_merge_method="stable-tail",
            segment_seconds=6.0,
            hop_seconds=1.5,
            transcription_meta={
                "elapsed_seconds": 1.0,
                "token_timestamps": [
                    {"word": "後", "start": 0.0, "end": 0.1, "score": 0.95},
                    {"word": "續", "start": 0.1, "end": 0.2, "score": 0.95},
                ],
            },
        )
        diagnostics = assembler.get_last_merge_diagnostics()

        self.assertIn("開頭", merged)
        self.assertIn("後續", merged)
        self.assertEqual(diagnostics["moved_to_history_count"], 2)
        self.assertEqual(diagnostics["rolling_base_source"], "committed_history")

    def test_committed_history_does_not_repeat_same_speaker_marker(self) -> None:
        assembler = SubtitleAssembler()
        assembler.set_language_context("zh")

        first_meta = {
            "elapsed_seconds": 0.0,
            "token_timestamps": [
                {"word": "開", "start": 0.0, "end": 0.1, "score": 0.95, "speaker": "SPEAKER_00"},
                {"word": "頭", "start": 0.1, "end": 0.2, "score": 0.95, "speaker": "SPEAKER_00"},
            ],
        }
        for _ in range(3):
            assembler.merge_incremental_text(
                "開頭",
                overlap_merge_method="stable-tail",
                segment_seconds=6.0,
                hop_seconds=1.5,
                transcription_meta=first_meta,
            )

        second_meta = {
            "elapsed_seconds": 1.0,
            "token_timestamps": [
                {"word": "後", "start": 0.0, "end": 0.1, "score": 0.95, "speaker": "SPEAKER_00"},
                {"word": "續", "start": 0.1, "end": 0.2, "score": 0.95, "speaker": "SPEAKER_00"},
            ],
        }
        for _ in range(3):
            assembler.merge_incremental_text(
                "後續",
                overlap_merge_method="stable-tail",
                segment_seconds=6.0,
                hop_seconds=1.5,
                transcription_meta=second_meta,
            )

        merged = assembler.merge_incremental_text(
            "結尾",
            overlap_merge_method="stable-tail",
            segment_seconds=6.0,
            hop_seconds=1.5,
            transcription_meta={
                "elapsed_seconds": 2.0,
                "token_timestamps": [
                    {"word": "結", "start": 0.0, "end": 0.1, "score": 0.95, "speaker": "SPEAKER_00"},
                    {"word": "尾", "start": 0.1, "end": 0.2, "score": 0.95, "speaker": "SPEAKER_00"},
                ],
            },
        )

        self.assertEqual(merged.count("[spk_000]"), 1)
        self.assertIn("開頭", merged)
        self.assertIn("後續", merged)
        self.assertIn("結尾", merged)

    def test_redundant_same_speaker_marker_lines_are_collapsed(self) -> None:
        assembler = SubtitleAssembler()
        assembler.set_language_context("zh")

        normalized = assembler._normalize_output_text(
            "[spk_000] 開頭\n[spk_000] 後續\n[spk_001] 換人\n[spk_001] 繼續"
        )

        self.assertEqual(normalized.count("[spk_000]"), 1)
        self.assertEqual(normalized.count("[spk_001]"), 1)
        self.assertIn("[spk_000] 開頭後續", normalized)
        self.assertIn("[spk_001] 換人繼續", normalized)
        self.assertIn("\n[spk_001]", normalized)

    def test_pause_separated_same_speaker_marker_lines_are_preserved(self) -> None:
        assembler = SubtitleAssembler()
        assembler.set_language_context("zh")

        normalized = assembler._normalize_output_text(
            "[spk_000] 開頭\n\n[spk_000] 長停頓後續"
        )

        self.assertEqual(normalized.count("[spk_000]"), 2)
        # The blank-line hard boundary is preserved (not collapsed to a single
        # newline) so the pause break survives the next normalize/merge pass.
        self.assertIn("[spk_000] 開頭\n\n[spk_000] 長停頓後續", normalized)

    def _spk_word(self, word: str, start: float, end: float, speaker: str) -> dict:
        return {"word": word, "start": start, "end": end, "score": 0.95, "speaker": speaker}

    def test_same_speaker_long_pause_breaks_to_new_marker_line(self) -> None:
        # Absolute word times: same speaker resumes after a > pause-threshold
        # gap -> a blank-line hard boundary + re-emitted marker (works across the
        # overlapping windows where provider window-relative detection cannot).
        assembler = SubtitleAssembler()
        assembler.set_language_context("zh")
        assembler.set_speaker_pause_break_seconds(1.8)
        words = assembler._extract_incoming_words(
            {
                "token_timestamps": [
                    self._spk_word("前句", 0.0, 0.5, "spk_000"),
                    self._spk_word("結束", 0.5, 1.0, "spk_000"),
                    # 3.0s silence (> 1.8) before the same speaker resumes.
                    self._spk_word("停頓", 4.0, 4.5, "spk_000"),
                    self._spk_word("之後", 4.5, 5.0, "spk_000"),
                ]
            },
            0.0,
        )
        rendered = assembler._words_to_text(words)
        # Initial marker + the re-emitted marker after the pause.
        self.assertEqual(rendered.count("[spk_000]"), 2)
        self.assertIn("\n\n[spk_000]", rendered)
        self.assertTrue(rendered.rsplit("\n\n", 1)[-1].startswith("[spk_000] 停頓"))

    def test_same_speaker_small_gap_does_not_break(self) -> None:
        assembler = SubtitleAssembler()
        assembler.set_language_context("zh")
        assembler.set_speaker_pause_break_seconds(1.8)
        words = assembler._extract_incoming_words(
            {
                "token_timestamps": [
                    self._spk_word("前句", 0.0, 0.5, "spk_000"),
                    self._spk_word("結束", 0.5, 1.0, "spk_000"),
                    # 0.3s gap (< 1.8): normal flow, no break.
                    self._spk_word("緊接", 1.3, 1.8, "spk_000"),
                ]
            },
            0.0,
        )
        rendered = assembler._words_to_text(words)
        self.assertNotIn("\n\n", rendered)

    def test_pause_break_across_committed_batch_boundary(self) -> None:
        # The pause straddles a commit boundary: the words after the silence are
        # appended in a later batch, yet the break is still emitted via the
        # tracked previous-committed end time.
        assembler = SubtitleAssembler()
        assembler.set_language_context("zh")
        assembler.set_speaker_pause_break_seconds(1.8)
        batch_a = assembler._extract_incoming_words(
            {"token_timestamps": [self._spk_word("前句", 0.0, 1.0, "spk_000")]},
            0.0,
        )
        batch_b = assembler._extract_incoming_words(
            {"token_timestamps": [self._spk_word("停頓後", 4.0, 4.5, "spk_000")]},
            0.0,
        )
        assembler._append_to_rolling_committed_text(batch_a)
        assembler._append_to_rolling_committed_text(batch_b)
        self.assertIn("\n\n[spk_000]", assembler._rolling_committed_text)

    def test_unconfirmed_profile_candidate_does_not_fallback_to_local_speaker(self) -> None:
        assembler = SubtitleAssembler()
        assembler.set_language_context("zh")

        words = assembler._extract_incoming_words(
            {
                "token_timestamps": [
                    {
                        "word": "開頭",
                        "start": 0.0,
                        "end": 0.4,
                        "score": 0.95,
                        "speaker": "SPK_000",
                        "profile_speaker": "SPK_000",
                        "local_speaker": "SPEAKER_00",
                    },
                    {
                        "word": "背景",
                        "start": 0.45,
                        "end": 0.75,
                        "score": 0.95,
                        "speaker": "",
                        "profile_speaker": "",
                        "local_speaker": "SPEAKER_01",
                    },
                    {
                        "word": "聲",
                        "start": 0.75,
                        "end": 0.9,
                        "score": 0.95,
                        "speaker": "",
                        "profile_speaker": "",
                        "local_speaker": "SPEAKER_01",
                    },
                ],
            },
            0.0,
        )
        merged = assembler._words_to_text(words)

        self.assertEqual(merged.count("[spk_000]"), 1)
        self.assertNotIn("[spk_001]", merged)
        self.assertIn("開頭背景聲", merged)

    def test_short_trailing_speaker_island_is_merged_to_previous_speaker(self) -> None:
        assembler = SubtitleAssembler()
        assembler.set_language_context("zh")

        merged = assembler._words_to_text(
            [
                _WordState("主要", 0.0, 0.4, 0.95, 1, 0.4, "SPK_000"),
                _WordState("說話", 0.4, 0.8, 0.95, 1, 0.8, "SPK_000"),
                _WordState("內容", 0.8, 1.2, 0.95, 1, 1.2, "SPK_000"),
                _WordState("尾", 1.25, 1.45, 0.95, 1, 1.45, "SPK_001"),
                _WordState("音", 1.45, 1.65, 0.95, 1, 1.65, "SPK_001"),
            ]
        )

        self.assertEqual(merged.count("[spk_000]"), 1)
        self.assertNotIn("[spk_001]", merged)
        self.assertIn("主要說話內容尾音", merged)

    def test_finalize_commits_tail_seen_fewer_than_required_windows_once(self) -> None:
        assembler = SubtitleAssembler()

        for _ in range(2):
            assembler.merge_incremental_text(
                "tail",
                overlap_merge_method="stable-tail",
                segment_seconds=6.0,
                hop_seconds=1.5,
                transcription_meta={
                    "elapsed_seconds": 0.0,
                    "token_timestamps": [
                        {"word": "tail", "start": 5.0, "end": 5.4, "score": 0.95},
                    ],
                },
            )

        self.assertNotIn("tail", assembler.get_history_text())

        finalized = assembler.finalize()
        finalized_again = assembler.finalize()

        self.assertIn("tail", finalized)
        self.assertEqual(finalized_again, "")
        self.assertEqual(assembler.get_history_text().count("tail"), 1)
        self.assertEqual(len(assembler.get_partial_state()), 0)
        self.assertEqual(len(assembler.get_stable_state()), 0)

    def test_tail_partial_survives_trailing_eof_windows_until_finalize(self) -> None:
        assembler = SubtitleAssembler()

        assembler.merge_incremental_text(
            "alpha tail",
            overlap_merge_method="stable-tail",
            segment_seconds=10.0,
            hop_seconds=2.0,
            transcription_meta={
                "elapsed_seconds": 0.0,
                "token_timestamps": [
                    {"word": "alpha", "start": 0.0, "end": 0.4, "score": 0.95},
                    {"word": "tail", "start": 8.5, "end": 9.0, "score": 0.95},
                ],
            },
        )
        assembler.merge_incremental_text(
            "tail",
            overlap_merge_method="stable-tail",
            segment_seconds=10.0,
            hop_seconds=2.0,
            transcription_meta={
                "elapsed_seconds": 2.0,
                "token_timestamps": [
                    {"word": "tail", "start": 6.5, "end": 7.0, "score": 0.95},
                ],
            },
        )
        assembler.merge_incremental_text(
            "end",
            overlap_merge_method="stable-tail",
            segment_seconds=10.0,
            hop_seconds=2.0,
            transcription_meta={
                "elapsed_seconds": 10.2,
                "token_timestamps": [
                    {"word": "end", "start": 0.0, "end": 0.3, "score": 0.95},
                ],
            },
        )

        self.assertNotIn("tail", assembler.get_history_text())

        finalized = assembler.finalize()

        self.assertIn("tail", finalized)
        self.assertIn("end", finalized)
        self.assertEqual(assembler.get_history_text().count("tail"), 1)

    def test_finalize_empty_or_already_committed_is_noop(self) -> None:
        assembler = SubtitleAssembler()

        self.assertEqual(assembler.finalize(), "")

        meta = {
            "elapsed_seconds": 0.0,
            "token_timestamps": [
                {"word": "done", "start": 0.0, "end": 0.3, "score": 0.95},
            ],
        }
        for _ in range(3):
            assembler.merge_incremental_text(
                "done",
                overlap_merge_method="stable-tail",
                segment_seconds=6.0,
                hop_seconds=1.5,
                transcription_meta=meta,
            )
        assembler.mark_sentence_break()
        committed = assembler.get_history_text()

        self.assertEqual(assembler.finalize(), "")
        self.assertEqual(assembler.get_history_text(), committed)


    def test_display_script_hant_folds_visible_output_to_traditional(self) -> None:
        assembler = SubtitleAssembler()
        assembler.set_display_script("hant")
        out = assembler.merge_incremental_text(
            "这个小渔村的现代化痕迹",  # all Simplified
            overlap_merge_method="stable-tail",
            segment_seconds=6.0,
            hop_seconds=1.5,
        )
        for ch in "这渔现迹":  # Simplified-only characters must not survive
            self.assertNotIn(ch, out)
        self.assertIn("這個小漁村", out)

    def test_display_script_off_keeps_original_script(self) -> None:
        assembler = SubtitleAssembler()
        assembler.set_display_script("off")
        out = assembler.merge_incremental_text(
            "这个小渔村",  # Simplified
            overlap_merge_method="stable-tail",
            segment_seconds=6.0,
            hop_seconds=1.5,
        )
        self.assertIn("这个", out)  # off must not convert the script
        self.assertNotIn("這個", out)

    def test_get_prompt_tail_strips_markers_and_bounds_length(self) -> None:
        assembler = SubtitleAssembler()
        assembler.reset()
        assembler._rolling_committed_text = "[spk_000] 你好世界\n\n[spk_001] 這是一段測試文字"
        tail = assembler.get_prompt_tail(8)
        self.assertNotIn("[spk_", tail)
        self.assertNotIn("\n", tail)
        self.assertLessEqual(len(tail), 8)
        self.assertTrue(tail)
        self.assertEqual(assembler.get_prompt_tail(0), "")

    def test_display_script_fold_is_cer_neutral_under_st_fold(self) -> None:
        # The char-level display fold must cancel under the comparison's own S/T
        # fold (no vocabulary localization), so CER cannot move.
        from voice2text.stt.audio_utils import normalize_chinese_script, unify_chinese_script

        sample = "软件信息这个小渔村"
        folded_display = unify_chinese_script(sample, "hant")
        self.assertEqual(
            normalize_chinese_script(folded_display, "hans"),
            normalize_chinese_script(sample, "hans"),
        )

    # ---- Round 0053: alias-stable display labels across a profile-store merge -------

    def test_alias_remap_propagates_dropped_ids_alias_to_kept_id(self) -> None:
        """The core round-0046 churn fix: words already displayed under a dropped id must not
        cause the surviving (kept) id to acquire a FRESH display number going forward."""
        assembler = SubtitleAssembler()
        assembler.set_language_context("en")
        # Establish SPK_003's display alias by rendering a word under it first (as if it had
        # already been on screen for a while before the merge).
        batch = [
            _WordState(word="hello", start=0.0, end=0.5, score=0.9, count=3, last_seen=0.5, speaker="SPK_003"),
        ]
        assembler._pending_commit_batches = [batch]
        assembler._drain_pending_commits(now=100.0, force_all=True)
        self.assertIn("[spk_003]", assembler._rolling_committed_text)

        # A profile-store merge drops SPK_003 into SPK_007 (the online path will label all
        # FUTURE words SPK_007 from now on).
        assembler.apply_speaker_alias_remap({"SPK_003": "SPK_007"})

        # A later word carries the surviving raw id -- it must render under the SAME alias the
        # viewer already saw (SPK_003), not a fresh SPK_007.
        later_batch = [
            _WordState(word="world", start=10.0, end=10.5, score=0.9, count=3, last_seen=10.5, speaker="SPK_007"),
        ]
        assembler._pending_commit_batches = [later_batch]
        assembler._drain_pending_commits(now=200.0, force_all=True)
        self.assertIn("[spk_003]", assembler._rolling_committed_text)
        self.assertNotIn("[spk_007]", assembler._rolling_committed_text)

    def test_alias_remap_is_forward_only_does_not_rewrite_frozen_text(self) -> None:
        assembler = SubtitleAssembler()
        assembler.set_language_context("en")
        batch = [
            _WordState(word="hello", start=0.0, end=0.5, score=0.9, count=3, last_seen=0.5, speaker="SPK_003"),
        ]
        assembler._pending_commit_batches = [batch]
        assembler._drain_pending_commits(now=100.0, force_all=True)
        frozen = assembler._rolling_committed_text

        assembler.apply_speaker_alias_remap({"SPK_003": "SPK_007"})

        self.assertEqual(assembler._rolling_committed_text, frozen)

    def test_alias_remap_noop_when_neither_side_has_a_display_alias_yet(self) -> None:
        """Both ids are brand new (never rendered) -- nothing to propagate; first-sight display
        assignment proceeds normally later, unaffected."""
        assembler = SubtitleAssembler()
        assembler.apply_speaker_alias_remap({"SPK_003": "SPK_007"})
        self.assertEqual(assembler._speaker_display_map, {})

    def test_alias_remap_empty_or_none_is_a_noop(self) -> None:
        assembler = SubtitleAssembler()
        assembler._speaker_display_map = {"SPK_003": "SPK_003"}
        assembler.apply_speaker_alias_remap({})
        assembler.apply_speaker_alias_remap(None)
        self.assertEqual(assembler._speaker_display_map, {"SPK_003": "SPK_003"})

    def test_alias_remap_self_map_and_blank_ids_are_ignored(self) -> None:
        assembler = SubtitleAssembler()
        assembler._speaker_display_map = {"SPK_003": "SPK_003"}
        assembler.apply_speaker_alias_remap({"SPK_003": "SPK_003", "": "SPK_009", "SPK_010": ""})
        self.assertEqual(assembler._speaker_display_map, {"SPK_003": "SPK_003"})

    # ---- Round 0048: pre-commit local-diarization relabel -------------------------

    def test_relabel_resolver_is_unset_by_default(self) -> None:
        assembler = SubtitleAssembler()
        self.assertIsNone(assembler._relabel_resolver)

    def test_relabel_resolver_overrides_speaker_before_freeze(self) -> None:
        assembler = SubtitleAssembler()
        assembler.set_language_context("en")
        calls: list[tuple[float, float]] = []

        def resolver(start: float, end: float) -> str | None:
            calls.append((start, end))
            return "SPEAKER_05"

        assembler.set_relabel_resolver(resolver)
        batch = [
            _WordState(word="hello", start=10.0, end=10.5, score=0.9, count=3, last_seen=10.5, speaker="SPEAKER_00"),
            _WordState(word="world", start=10.5, end=11.0, score=0.9, count=3, last_seen=11.0, speaker="SPEAKER_00"),
        ]
        assembler._pending_commit_batches = [batch]
        assembler._drain_pending_commits(now=100.0, force_all=True)

        self.assertEqual(calls, [(10.0, 11.0)])
        self.assertIn("[spk_005]", assembler._rolling_committed_text)
        self.assertNotIn("[spk_000]", assembler._rolling_committed_text)
        self.assertIn("hello world", assembler._rolling_committed_text)
        # The word states themselves were relabeled (not just the rendered text).
        self.assertTrue(all(w.speaker == "SPEAKER_05" for w in batch))

    def test_relabel_resolver_none_keeps_existing_label(self) -> None:
        assembler = SubtitleAssembler()
        assembler.set_language_context("en")
        assembler.set_relabel_resolver(lambda start, end: None)
        batch = [
            _WordState(word="hello", start=10.0, end=10.5, score=0.9, count=3, last_seen=10.5, speaker="SPEAKER_00"),
        ]
        assembler._pending_commit_batches = [batch]
        assembler._drain_pending_commits(now=100.0, force_all=True)

        self.assertIn("[spk_000]", assembler._rolling_committed_text)
        self.assertEqual(batch[0].speaker, "SPEAKER_00")

    def test_relabel_resolver_disabled_matches_legacy_output(self) -> None:
        """Byte-identical contract: with no resolver set, drain output must match the
        pre-0048 legacy path exactly."""
        def make_batch() -> list[_WordState]:
            return [
                _WordState(word="hello", start=10.0, end=10.5, score=0.9, count=3, last_seen=10.5, speaker="SPEAKER_00"),
            ]

        legacy = SubtitleAssembler()
        legacy.set_language_context("en")
        legacy._pending_commit_batches = [make_batch()]
        legacy._drain_pending_commits(now=100.0, force_all=True)

        with_none_resolver = SubtitleAssembler()
        with_none_resolver.set_language_context("en")
        with_none_resolver.set_relabel_resolver(None)
        with_none_resolver._pending_commit_batches = [make_batch()]
        with_none_resolver._drain_pending_commits(now=100.0, force_all=True)

        self.assertEqual(legacy._rolling_committed_text, with_none_resolver._rolling_committed_text)

    def test_relabel_resolver_exception_is_swallowed_and_keeps_existing_label(self) -> None:
        assembler = SubtitleAssembler()
        assembler.set_language_context("en")

        def boom(start: float, end: float) -> str | None:
            raise RuntimeError("synthetic resolver failure")

        assembler.set_relabel_resolver(boom)
        batch = [
            _WordState(word="hello", start=10.0, end=10.5, score=0.9, count=3, last_seen=10.5, speaker="SPEAKER_00"),
        ]
        assembler._pending_commit_batches = [batch]
        # Must not raise.
        assembler._drain_pending_commits(now=100.0, force_all=True)

        self.assertIn("[spk_000]", assembler._rolling_committed_text)
        self.assertEqual(batch[0].speaker, "SPEAKER_00")

    def test_relabel_resolver_never_rewrites_already_frozen_text(self) -> None:
        """A batch that already froze (drained) in an earlier call must never be touched by a
        later relabel resolution -- forward-only, no retroactive rewrite (the exact failure
        mode round 0046 proved must be avoided)."""
        assembler = SubtitleAssembler()
        assembler.set_language_context("en")
        assembler.set_relabel_resolver(lambda start, end: "SPEAKER_09")
        first_batch = [
            _WordState(word="first", start=0.0, end=0.5, score=0.9, count=3, last_seen=0.5, speaker="SPEAKER_00"),
        ]
        assembler._pending_commit_batches = [first_batch]
        assembler._drain_pending_commits(now=100.0, force_all=True)
        frozen_after_first = assembler._rolling_committed_text
        self.assertIn("[spk_009]", frozen_after_first)

        # A second, later batch drains -- the resolver call for it must not touch the
        # already-committed text from the first drain.
        assembler.set_relabel_resolver(lambda start, end: "SPEAKER_11")
        second_batch = [
            _WordState(word="second", start=1.0, end=1.5, score=0.9, count=3, last_seen=1.5, speaker="SPEAKER_00"),
        ]
        assembler._pending_commit_batches = [second_batch]
        assembler._drain_pending_commits(now=200.0, force_all=True)

        self.assertTrue(assembler._rolling_committed_text.startswith(frozen_after_first))
        self.assertIn("[spk_011]", assembler._rolling_committed_text)

    # ---- Round 0052: turn-aware relabel + margin gate ------------------------------

    @staticmethod
    def _turn_batch() -> list[_WordState]:
        return [
            _WordState(word="alpha", start=10.0, end=10.5, score=0.9, count=3, last_seen=10.5, speaker="SPEAKER_00"),
            _WordState(word="beta", start=10.5, end=11.0, score=0.9, count=3, last_seen=11.0, speaker="SPEAKER_00"),
            _WordState(word="gamma", start=12.0, end=12.5, score=0.9, count=3, last_seen=12.5, speaker="SPEAKER_00"),
        ]

    def test_turn_aware_relabel_splits_batch_at_span_boundaries(self) -> None:
        assembler = SubtitleAssembler()
        assembler.set_language_context("en")
        assembler.set_relabel_resolver(
            lambda start, end: [
                {"start": 10.0, "end": 11.0, "resolved": "SPEAKER_05",
                 "resolved_cosine": 0.9, "scores": {"SPEAKER_05": 0.9, "SPEAKER_00": 0.3}},
                {"start": 11.9, "end": 12.6, "resolved": "SPEAKER_07",
                 "resolved_cosine": 0.8, "scores": {"SPEAKER_07": 0.8, "SPEAKER_00": 0.2}},
            ],
            margin=0.05,
        )
        batch = self._turn_batch()
        assembler._pending_commit_batches = [batch]
        assembler._drain_pending_commits(now=100.0, force_all=True)

        self.assertEqual([w.speaker for w in batch], ["SPEAKER_05", "SPEAKER_05", "SPEAKER_07"])

    def test_margin_gate_keeps_incumbent_when_evidence_is_close(self) -> None:
        assembler = SubtitleAssembler()
        assembler.set_language_context("en")
        # Resolved beats the incumbent by only 0.02 < margin 0.05 -> incumbent survives.
        assembler.set_relabel_resolver(
            lambda start, end: [
                {"start": 10.0, "end": 12.6, "resolved": "SPEAKER_05",
                 "resolved_cosine": 0.72, "scores": {"SPEAKER_05": 0.72, "SPEAKER_00": 0.70}},
            ],
            margin=0.05,
        )
        batch = self._turn_batch()
        assembler._pending_commit_batches = [batch]
        assembler._drain_pending_commits(now=100.0, force_all=True)

        self.assertTrue(all(w.speaker == "SPEAKER_00" for w in batch))

    def test_margin_gate_backfills_empty_labels_without_margin(self) -> None:
        assembler = SubtitleAssembler()
        assembler.set_language_context("en")
        assembler.set_relabel_resolver(
            lambda start, end: [
                {"start": 10.0, "end": 12.6, "resolved": "SPEAKER_05",
                 "resolved_cosine": 0.66, "scores": {"SPEAKER_05": 0.66}},
            ],
            margin=0.05,
        )
        batch = self._turn_batch()
        for w in batch:
            w.speaker = ""
        assembler._pending_commit_batches = [batch]
        assembler._drain_pending_commits(now=100.0, force_all=True)

        self.assertTrue(all(w.speaker == "SPEAKER_05" for w in batch))

    def test_turn_aware_relabel_keeps_labels_in_uncovered_gaps(self) -> None:
        assembler = SubtitleAssembler()
        assembler.set_language_context("en")
        # Only the middle word's midpoint falls inside the resolved span.
        assembler.set_relabel_resolver(
            lambda start, end: [
                {"start": 10.4, "end": 11.1, "resolved": "SPEAKER_05",
                 "resolved_cosine": 0.9, "scores": {"SPEAKER_05": 0.9, "SPEAKER_00": 0.2}},
            ],
            margin=0.05,
        )
        batch = self._turn_batch()
        assembler._pending_commit_batches = [batch]
        assembler._drain_pending_commits(now=100.0, force_all=True)

        self.assertEqual([w.speaker for w in batch], ["SPEAKER_00", "SPEAKER_05", "SPEAKER_00"])

    # ---- Round 0052 Phase B: async resolver (RELABEL_PENDING) -----------------------

    def test_pending_batch_is_not_frozen_and_retries_on_next_drain(self) -> None:
        from voice2text.pipeline.subtitle_assembler import RELABEL_PENDING

        assembler = SubtitleAssembler()
        assembler.set_language_context("en")
        assembler.set_commit_hold(hold_seconds=5.0)
        calls: list[tuple[float, float]] = []

        def resolver(start: float, end: float):
            calls.append((start, end))
            return RELABEL_PENDING if len(calls) == 1 else "SPEAKER_05"

        assembler.set_relabel_resolver(resolver, defer=True)
        batch = [
            _WordState(word="hello", start=1.0, end=1.5, score=0.9, count=3, last_seen=1.5, speaker="SPEAKER_00"),
        ]
        assembler._pending_commit_batches = [batch]

        # now=100 -> batch is aged past the 5s hold, so it's a drain candidate; resolver's first
        # answer is PENDING -> must NOT freeze, must stay in the pending buffer.
        assembler._drain_pending_commits(now=100.0, force_all=False)
        self.assertEqual(len(calls), 1)
        self.assertEqual(assembler._pending_commit_batches, [batch])
        self.assertEqual(assembler._rolling_committed_text, '')

        # A later drain call re-tries the same (still-aged) batch; this time it resolves.
        assembler._drain_pending_commits(now=100.0, force_all=False)
        self.assertEqual(len(calls), 2)
        self.assertEqual(assembler._pending_commit_batches, [])
        self.assertIn("[spk_005]", assembler._rolling_committed_text)

    def test_pending_result_never_freezes_under_force_all(self) -> None:
        """force_all (finalize/stream-end) cannot wait on an async worker -- must freeze with
        the existing labels rather than blocking, even if the resolver would say PENDING."""
        from voice2text.pipeline.subtitle_assembler import RELABEL_PENDING

        assembler = SubtitleAssembler()
        assembler.set_language_context("en")
        assembler.set_relabel_resolver(lambda start, end: RELABEL_PENDING, defer=True)
        batch = [
            _WordState(word="hello", start=1.0, end=1.5, score=0.9, count=3, last_seen=1.5, speaker="SPEAKER_00"),
        ]
        assembler._pending_commit_batches = [batch]

        assembler._drain_pending_commits(now=100.0, force_all=True)

        self.assertEqual(assembler._pending_commit_batches, [])
        self.assertIn("[spk_000]", assembler._rolling_committed_text)

    def test_pending_batch_blocks_later_batches_from_draining(self) -> None:
        """FIFO: a still-pending earlier batch must hold back a later, already-resolved batch --
        commit order stays monotonic even under async resolution."""
        from voice2text.pipeline.subtitle_assembler import RELABEL_PENDING

        assembler = SubtitleAssembler()
        assembler.set_language_context("en")
        assembler.set_commit_hold(hold_seconds=5.0)

        def resolver(start: float, end: float):
            return RELABEL_PENDING if start < 5.0 else "SPEAKER_07"

        assembler.set_relabel_resolver(resolver, defer=True)
        first_batch = [
            _WordState(word="first", start=1.0, end=1.5, score=0.9, count=3, last_seen=1.5, speaker="SPEAKER_00"),
        ]
        second_batch = [
            _WordState(word="second", start=10.0, end=10.5, score=0.9, count=3, last_seen=10.5, speaker="SPEAKER_00"),
        ]
        assembler._pending_commit_batches = [first_batch, second_batch]

        assembler._drain_pending_commits(now=100.0, force_all=False)

        self.assertEqual(assembler._pending_commit_batches, [first_batch, second_batch])
        self.assertEqual(assembler._rolling_committed_text, '')

    def test_mark_sentence_break_routes_through_pending_when_deferred(self) -> None:
        """With an async (defer=True) resolver and commit-hold on, mark_sentence_break must NOT
        force-freeze stable words directly -- they go into the pending buffer like a normal flush,
        so a still-unresolved batch can wait for its worker result instead of committing blind."""
        from voice2text.pipeline.subtitle_assembler import RELABEL_PENDING

        assembler = SubtitleAssembler()
        assembler.set_language_context("en")
        assembler.set_commit_hold(hold_seconds=5.0)
        assembler.set_relabel_resolver(lambda start, end: RELABEL_PENDING, defer=True)
        assembler._stable_words = [
            _WordState(word="hello", start=1.0, end=1.5, score=0.9, count=3, last_seen=1.5, speaker="SPEAKER_00"),
        ]

        assembler.mark_sentence_break(now=2.0)

        self.assertEqual(assembler._stable_words, [])
        self.assertEqual(assembler._rolling_committed_text, '')
        self.assertEqual(len(assembler._pending_commit_batches), 1)

    def test_turn_aware_relabel_unknown_incumbent_is_overwritten(self) -> None:
        assembler = SubtitleAssembler()
        assembler.set_language_context("en")
        # Incumbent label absent from the score map (e.g. deleted profile) -> confident
        # resolution wins without a margin comparison.
        assembler.set_relabel_resolver(
            lambda start, end: [
                {"start": 10.0, "end": 12.6, "resolved": "SPEAKER_05",
                 "resolved_cosine": 0.7, "scores": {"SPEAKER_05": 0.7}},
            ],
            margin=0.05,
        )
        batch = self._turn_batch()
        assembler._pending_commit_batches = [batch]
        assembler._drain_pending_commits(now=100.0, force_all=True)

        self.assertTrue(all(w.speaker == "SPEAKER_05" for w in batch))


if __name__ == "__main__":
    unittest.main()
