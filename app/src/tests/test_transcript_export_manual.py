"""Unit tests for manual single-file transcript export."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import time
import unittest

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice2text.pipeline.transcript_exporter import TranscriptExportOptions, TranscriptExporterSession


class TranscriptExportManualTests(unittest.TestCase):
    def _new_session(self, out_dir: Path) -> TranscriptExporterSession:
        return TranscriptExporterSession(
            TranscriptExportOptions(
                enabled=True,
                formats=["txt", "srt", "json"],
                include_timestamps=True,
                include_speaker=True,
                output_dir=str(out_dir),
            )
        )

    def test_export_single_txt_appends_extension_and_writes_content(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-export-") as td:
            out_dir = Path(td)
            session = self._new_session(out_dir)
            session.record(
                raw_text="hello",
                source_text=">> hello",
                translated_text="",
                meta={"elapsed_seconds": 0.0, "token_timestamps": []},
            )
            path = session.export_single_file(
                output_path=str(out_dir / "manual_export"),
                format_hint="txt",
            )
            self.assertEqual(path.suffix, ".txt")
            text = path.read_text(encoding="utf-8")
            self.assertIn("hello", text.lower())

    def test_export_display_text_file_writes_latest_overlay_text_only(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-export-") as td:
            out_dir = Path(td)
            session = self._new_session(out_dir)
            session.record(
                raw_text="raw window",
                source_text="[spk_000] 第一行\n[spk_001] 第二行",
                translated_text="",
                meta={
                    "elapsed_seconds": 0.0,
                    "token_timestamps": [
                        {"word": "不", "absolute_start": 0.0, "absolute_end": 0.1},
                        {"word": "應", "absolute_start": 0.1, "absolute_end": 0.2},
                        {"word": "重", "absolute_start": 0.2, "absolute_end": 0.3},
                        {"word": "組", "absolute_start": 0.3, "absolute_end": 0.4},
                    ],
                },
            )

            path = session.export_display_text_file(output_path=str(out_dir / "main_display"))

            self.assertEqual(path.suffix, ".txt")
            self.assertEqual(
                path.read_text(encoding="utf-8"),
                "[spk_000] 第一行\n[spk_001] 第二行\n",
            )

    def test_display_text_only_finalize_writes_single_txt_snapshot(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-export-") as td:
            out_dir = Path(td)
            session = TranscriptExporterSession(
                TranscriptExportOptions(
                    enabled=True,
                    formats=["txt", "srt", "json"],
                    include_timestamps=True,
                    include_speaker=True,
                    output_dir=str(out_dir),
                    display_text_only=True,
                )
            )
            session.record(
                raw_text="raw",
                source_text="main overlay text",
                translated_text="",
                meta={"elapsed_seconds": 0.0, "token_timestamps": []},
            )

            written = session.finalize()

            self.assertEqual([path.suffix for path in written], [".txt"])
            self.assertEqual(written[0].read_text(encoding="utf-8"), "main overlay text\n")
            self.assertEqual(list(out_dir.glob("*.srt")), [])
            self.assertEqual(list(out_dir.glob("*.json")), [])

    def test_finalize_is_idempotent_and_manual_export_still_works_after_finalize(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-export-") as td:
            out_dir = Path(td)
            session = TranscriptExporterSession(
                TranscriptExportOptions(
                    enabled=True,
                    formats=["txt"],
                    include_timestamps=True,
                    include_speaker=True,
                    output_dir=str(out_dir),
                )
            )
            session.record(
                raw_text="first run",
                source_text="first run",
                translated_text="",
                meta={"elapsed_seconds": 0.0, "token_timestamps": []},
            )

            first = session.finalize()
            time.sleep(1.1)
            second = session.finalize()

            self.assertEqual(first, second)
            self.assertEqual(len(list(out_dir.glob("transcript_*.txt"))), 1)

            manual_path = session.export_single_file(
                output_path=str(out_dir / "manual_after_pause"),
                format_hint="txt",
            )
            self.assertIn("first run", manual_path.read_text(encoding="utf-8"))

    def test_export_single_json_respects_include_flags(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-export-") as td:
            out_dir = Path(td)
            session = self._new_session(out_dir)
            session.record(
                raw_text="foo",
                source_text="foo",
                translated_text="",
                meta={"elapsed_seconds": 0.0, "token_timestamps": []},
            )
            path = session.export_single_file(
                output_path=str(out_dir / "manual.json"),
                format_hint="json",
                include_timestamps=False,
                include_speaker=False,
            )
            payload = path.read_text(encoding="utf-8")
            self.assertIn('"include_timestamps": false', payload)
            self.assertIn('"include_speaker": false', payload)

    def test_export_single_requires_supported_format(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-export-") as td:
            out_dir = Path(td)
            session = self._new_session(out_dir)
            session.record(
                raw_text="bar",
                source_text="bar",
                translated_text="",
                meta={"elapsed_seconds": 0.0, "token_timestamps": []},
            )
            with self.assertRaises(ValueError):
                session.export_single_file(
                    output_path=str(out_dir / "bad.out"),
                    format_hint="csv",
                )

    def test_cjk_token_cues_do_not_split_at_four_seconds_only(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-export-") as td:
            out_dir = Path(td)
            session = self._new_session(out_dir)
            words = list("这是样片日记第一次解锁非洲地区眼前这个国家叫做马达加斯加")
            session.record(
                raw_text="".join(words),
                source_text="".join(words),
                translated_text="",
                meta={
                    "elapsed_seconds": 0.0,
                    "token_timestamps": [
                        {
                            "word": word,
                            "absolute_start": idx * 0.22,
                            "absolute_end": idx * 0.22 + 0.18,
                        }
                        for idx, word in enumerate(words)
                    ],
                },
            )
            path = session.export_single_file(
                output_path=str(out_dir / "cjk.txt"),
                format_hint="txt",
            )
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(lines), 1)
            self.assertIn("国家叫做马达加斯加", lines[0])

    def test_ascii_fragments_inside_cjk_are_joined(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-export-") as td:
            out_dir = Path(td)
            session = self._new_session(out_dir)
            words = ["台", "1", "1", "6", "寸", "比", "较", "l", "o", "c", "a", "l", "的", "景", "区"]
            session.record(
                raw_text="台116寸比较local的景区",
                source_text="台116寸比较local的景区",
                translated_text="",
                meta={
                    "elapsed_seconds": 0.0,
                    "token_timestamps": [
                        {
                            "word": word,
                            "absolute_start": idx * 0.2,
                            "absolute_end": idx * 0.2 + 0.15,
                        }
                        for idx, word in enumerate(words)
                    ],
                },
            )
            path = session.export_single_file(
                output_path=str(out_dir / "mixed.txt"),
                format_hint="txt",
            )
            text = path.read_text(encoding="utf-8")
            self.assertIn("台116寸", text)
            self.assertIn("比较local的景区", text)
            self.assertNotIn("1 1 6", text)
            self.assertNotIn("l o c a l", text)

    def test_token_cues_do_not_split_on_sentence_punctuation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-export-") as td:
            out_dir = Path(td)
            session = self._new_session(out_dir)
            words = ["7", ".", "1", ".", "4", "對", "不知道"]
            session.record(
                raw_text="7.1.4對不知道",
                source_text="7.1.4對不知道",
                translated_text="",
                meta={
                    "elapsed_seconds": 0.0,
                    "token_timestamps": [
                        {
                            "word": word,
                            "absolute_start": idx * 0.2,
                            "absolute_end": idx * 0.2 + 0.15,
                            "speaker": "spk_000",
                        }
                        for idx, word in enumerate(words)
                    ],
                },
            )
            path = session.export_single_file(
                output_path=str(out_dir / "punctuation.txt"),
                format_hint="txt",
            )
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(lines), 1)
            self.assertIn("7.1.4對不知道", lines[0])

    def test_token_cues_use_absolute_timestamps(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-export-") as td:
            out_dir = Path(td)
            session = self._new_session(out_dir)
            session.record(
                raw_text="後半段",
                source_text="後半段",
                translated_text="",
                meta={
                    "elapsed_seconds": 0.0,
                    "token_timestamps": [
                        {"word": "後", "absolute_start": 56.0, "absolute_end": 56.2, "speaker": "spk_000"},
                        {"word": "半", "absolute_start": 56.2, "absolute_end": 56.4, "speaker": "spk_000"},
                        {"word": "段", "absolute_start": 56.4, "absolute_end": 56.6, "speaker": "spk_000"},
                    ],
                },
            )
            path = session.export_single_file(
                output_path=str(out_dir / "absolute.txt"),
                format_hint="txt",
            )
            text = path.read_text(encoding="utf-8")
            self.assertIn("00:00:56.000", text)
            self.assertIn("[spk_000]", text)

    def test_token_cues_prefer_profile_speaker_over_local_speaker(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-export-") as td:
            out_dir = Path(td)
            session = self._new_session(out_dir)
            session.record(
                raw_text="hello",
                source_text="hello",
                translated_text="",
                meta={
                    "elapsed_seconds": 0.0,
                    "token_timestamps": [
                        {
                            "word": "hello",
                            "absolute_start": 0.0,
                            "absolute_end": 0.5,
                            "speaker": "SPEAKER_07",
                            "profile_speaker": "SPK_002",
                        }
                    ],
                },
            )
            path = session.export_single_file(
                output_path=str(out_dir / "profile_speaker.txt"),
                format_hint="txt",
            )
            text = path.read_text(encoding="utf-8")
            self.assertIn("[spk_002] hello", text)
            self.assertNotIn("[spk_007]", text)

    def test_event_fallback_parses_spk_prefixes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-export-") as td:
            out_dir = Path(td)
            session = self._new_session(out_dir)
            session.record(
                raw_text="[spk_001] hello",
                source_text="[spk_001] hello",
                translated_text="",
                meta={"elapsed_seconds": 12.5},
            )
            path = session.export_single_file(
                output_path=str(out_dir / "fallback.txt"),
                format_hint="txt",
            )
            text = path.read_text(encoding="utf-8")
            self.assertIn("00:00:12.500", text)
            self.assertIn("[spk_001] hello", text)

    def test_final_snapshot_merges_consecutive_same_speaker_lines(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-export-") as td:
            out_dir = Path(td)
            session = self._new_session(out_dir)
            session.record(
                raw_text="[spk_001] 甲\n[spk_001] 乙\n[spk_002] 丙\n[spk_002] 丁",
                source_text="[spk_001] 甲\n[spk_001] 乙\n[spk_002] 丙\n[spk_002] 丁",
                translated_text="",
                meta={
                    "elapsed_seconds": 0.0,
                    "snapshot_final": True,
                    "snapshot_total_duration_seconds": 12.0,
                },
            )
            path = session.export_single_file(
                output_path=str(out_dir / "snapshot.txt"),
                format_hint="txt",
            )
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

            self.assertEqual(len(lines), 2)
            self.assertIn("[spk_001] 甲 乙", lines[0])
            self.assertIn("[spk_002] 丙 丁", lines[1])

    def test_token_cues_split_on_two_second_gap_and_normalize_speaker(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-export-") as td:
            out_dir = Path(td)
            session = self._new_session(out_dir)
            session.record(
                raw_text="甲乙",
                source_text="甲乙",
                translated_text="",
                meta={
                    "elapsed_seconds": 0.0,
                    "token_timestamps": [
                        {"word": "甲", "absolute_start": 0.0, "absolute_end": 0.2, "speaker": "SPEAKER_00"},
                        {"word": "乙", "absolute_start": 2.1, "absolute_end": 2.3, "speaker": "SPEAKER_00"},
                    ],
                },
            )
            path = session.export_single_file(
                output_path=str(out_dir / "gap.txt"),
                format_hint="txt",
            )
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(lines), 1)
            self.assertIn("[spk_000]", lines[0])

            session.record(
                raw_text="丙",
                source_text="甲乙丙",
                translated_text="",
                meta={
                    "elapsed_seconds": 0.0,
                    "token_timestamps": [
                        {"word": "丙", "absolute_start": 4.5, "absolute_end": 4.7, "speaker": "SPEAKER_00"},
                    ],
                },
            )
            path = session.export_single_file(
                output_path=str(out_dir / "gap2.txt"),
                format_hint="txt",
            )
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(lines), 2)

    def test_long_cjk_token_cue_does_not_split_by_length_only(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-export-") as td:
            out_dir = Path(td)
            session = self._new_session(out_dir)
            words = list("这是一个很长但是没有明显停顿也没有换人的中文测试句子用来确认不会只因为长度被切开")
            session.record(
                raw_text="".join(words),
                source_text="".join(words),
                translated_text="",
                meta={
                    "elapsed_seconds": 0.0,
                    "token_timestamps": [
                        {
                            "word": word,
                            "absolute_start": idx * 0.08,
                            "absolute_end": idx * 0.08 + 0.05,
                            "speaker": "SPEAKER_00",
                        }
                        for idx, word in enumerate(words)
                    ],
                },
            )
            path = session.export_single_file(
                output_path=str(out_dir / "long.txt"),
                format_hint="txt",
            )
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(lines), 1)

    def test_token_cues_smooth_tiny_aba_speaker_blip(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-export-") as td:
            out_dir = Path(td)
            session = self._new_session(out_dir)
            session.record(
                raw_text="我們啊呃繼續",
                source_text="我們啊呃繼續",
                translated_text="",
                meta={
                    "elapsed_seconds": 0.0,
                    "token_timestamps": [
                        {"word": "我", "absolute_start": 0.00, "absolute_end": 0.20, "speaker": "SPEAKER_00"},
                        {"word": "們", "absolute_start": 0.20, "absolute_end": 0.40, "speaker": "SPEAKER_00"},
                        {"word": "啊", "absolute_start": 0.40, "absolute_end": 0.56, "speaker": "SPEAKER_01"},
                        {"word": "呃", "absolute_start": 0.56, "absolute_end": 0.72, "speaker": "SPEAKER_01"},
                        {"word": "繼", "absolute_start": 0.72, "absolute_end": 0.92, "speaker": "SPEAKER_00"},
                        {"word": "續", "absolute_start": 0.92, "absolute_end": 1.12, "speaker": "SPEAKER_00"},
                    ],
                },
            )
            path = session.export_single_file(
                output_path=str(out_dir / "speaker_smooth.txt"),
                format_hint="txt",
            )
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(lines), 1)
            self.assertIn("[spk_000]", lines[0])
            self.assertNotIn("[spk_001]", "\n".join(lines))

    def test_token_cues_smooth_tiny_trailing_speaker_blip(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-export-") as td:
            out_dir = Path(td)
            session = self._new_session(out_dir)
            session.record(
                raw_text="我們繼續啊呃",
                source_text="我們繼續啊呃",
                translated_text="",
                meta={
                    "elapsed_seconds": 0.0,
                    "token_timestamps": [
                        {"word": "我", "absolute_start": 0.00, "absolute_end": 0.20, "speaker": "SPEAKER_00"},
                        {"word": "們", "absolute_start": 0.20, "absolute_end": 0.40, "speaker": "SPEAKER_00"},
                        {"word": "繼", "absolute_start": 0.40, "absolute_end": 0.60, "speaker": "SPEAKER_00"},
                        {"word": "續", "absolute_start": 0.60, "absolute_end": 0.80, "speaker": "SPEAKER_00"},
                        {"word": "啊", "absolute_start": 0.80, "absolute_end": 0.94, "speaker": "SPEAKER_01"},
                        {"word": "呃", "absolute_start": 0.94, "absolute_end": 1.08, "speaker": "SPEAKER_01"},
                    ],
                },
            )
            path = session.export_single_file(
                output_path=str(out_dir / "speaker_tail_smooth.txt"),
                format_hint="txt",
            )
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(lines), 1)
            self.assertIn("[spk_000]", lines[0])
            self.assertNotIn("[spk_001]", "\n".join(lines))

    def test_final_snapshot_distributes_lines_across_total_duration(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-export-") as td:
            out_dir = Path(td)
            session = self._new_session(out_dir)
            session.record(
                raw_text="[spk_000] alpha\n[spk_001] beta",
                source_text="[spk_000] alpha\n[spk_001] beta",
                translated_text="",
                meta={
                    "elapsed_seconds": 0.0,
                    "snapshot_final": True,
                    "snapshot_total_duration_seconds": 10.0,
                },
            )
            path = session.export_single_file(
                output_path=str(out_dir / "snapshot.txt"),
                format_hint="txt",
            )
            text = path.read_text(encoding="utf-8")
            self.assertIn("00:00:00.000", text)
            self.assertIn("00:00:05.000", text)
            self.assertIn("[spk_000] alpha", text)
            self.assertIn("[spk_001] beta", text)


if __name__ == "__main__":
    unittest.main()
