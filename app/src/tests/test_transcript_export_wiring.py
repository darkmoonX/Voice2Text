from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice2text.pipeline.transcript_exporter import (
    TranscriptExportOptions,
    TranscriptExporterSession,
    export_format_suffix,
)


def _make_session(
    output_dir: Path,
    *,
    include_timestamps: bool = True,
    include_speaker: bool = True,
    display_text_only: bool = False,
) -> TranscriptExporterSession:
    options = TranscriptExportOptions(
        enabled=True,
        formats=["txt", "srt", "json"],
        include_timestamps=include_timestamps,
        include_speaker=include_speaker,
        output_dir=str(output_dir),
        display_text_only=display_text_only,
    )
    return TranscriptExporterSession(options)


def _record_two_speakers(session: TranscriptExporterSession) -> None:
    session.record(
        raw_text="",
        source_text="[spk_000] hello world",
        translated_text="",
        meta={
            "elapsed_seconds": 0.0,
            "token_timestamps": [
                {"word": "hello", "absolute_start": 0.0, "absolute_end": 0.6, "speaker": "spk_000", "score": 0.91},
                {"word": "world", "absolute_start": 0.6, "absolute_end": 1.2, "speaker": "spk_000", "score": 0.83},
                {"word": "foobar", "absolute_start": 4.0, "absolute_end": 5.2, "speaker": "spk_001", "score": 0.74},
            ],
        },
    )


class FormatSuffixTests(unittest.TestCase):
    def test_suffix_mapping(self) -> None:
        self.assertEqual(export_format_suffix("display"), ".txt")
        self.assertEqual(export_format_suffix("txt"), ".txt")
        self.assertEqual(export_format_suffix("srt"), ".srt")
        self.assertEqual(export_format_suffix("json"), ".json")
        self.assertEqual(export_format_suffix("SRT"), ".srt")
        self.assertEqual(export_format_suffix(""), ".txt")
        self.assertEqual(export_format_suffix("nonsense"), ".txt")


class ExportRoutingTests(unittest.TestCase):
    """The controller delegates verbatim to `export_to`; lock the per-format routing here."""

    def test_export_to_routes_each_format(self) -> None:
        calls: list[tuple[str, str]] = []

        class SpySession(TranscriptExporterSession):
            def export_display_text_file(self, *, output_path: str) -> Path:  # type: ignore[override]
                calls.append(("display", output_path))
                return Path(output_path)

            def export_single_file(self, *, output_path, format_hint, include_timestamps=None, include_speaker=None):  # type: ignore[override]
                calls.append((format_hint, output_path))
                return Path(output_path)

        with tempfile.TemporaryDirectory() as tmp:
            session = SpySession(
                TranscriptExportOptions(
                    enabled=True,
                    formats=["txt"],
                    include_timestamps=True,
                    include_speaker=True,
                    output_dir=tmp,
                )
            )
            session.export_to(output_path=str(Path(tmp) / "a"), export_format="display")
            session.export_to(output_path=str(Path(tmp) / "b"), export_format="")
            session.export_to(output_path=str(Path(tmp) / "c"), export_format="txt")
            session.export_to(output_path=str(Path(tmp) / "d"), export_format="srt")
            session.export_to(output_path=str(Path(tmp) / "e"), export_format="json")

        routed = [fmt for fmt, _ in calls]
        self.assertEqual(routed, ["display", "display", "txt", "srt", "json"])


class SrtRenderTests(unittest.TestCase):
    def test_srt_has_timecodes_and_speaker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session = _make_session(Path(tmp))
            _record_two_speakers(session)
            out = session.export_to(output_path=str(Path(tmp) / "x.srt"), export_format="srt")
            self.assertEqual(out.suffix, ".srt")
            body = out.read_text(encoding="utf-8")
        self.assertIn("00:00:00,000 --> 00:00:01,200", body)
        self.assertIn("00:00:04,000 --> 00:00:05,200", body)
        self.assertIn("spk_000: hello world", body)
        self.assertIn("spk_001: foobar", body)
        # cue indices present and monotonic
        self.assertIn("\n1\n", "\n" + body)
        self.assertIn("2", body.splitlines())

    def test_srt_speaker_suppressed_when_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session = _make_session(Path(tmp))
            _record_two_speakers(session)
            out = session.export_to(
                output_path=str(Path(tmp) / "x.srt"),
                export_format="srt",
                include_speaker=False,
            )
            body = out.read_text(encoding="utf-8")
        self.assertNotIn("spk_000:", body)
        self.assertIn("hello world", body)

    def test_suffix_forced_to_match_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session = _make_session(Path(tmp))
            _record_two_speakers(session)
            # caller passed a .txt path but asked for srt -> exporter rewrites suffix
            out = session.export_to(output_path=str(Path(tmp) / "wrong.txt"), export_format="srt")
            self.assertEqual(out.suffix, ".srt")


class JsonRenderTests(unittest.TestCase):
    def test_json_has_cues_events_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session = _make_session(Path(tmp))
            _record_two_speakers(session)
            out = session.export_to(output_path=str(Path(tmp) / "x.json"), export_format="json")
            self.assertEqual(out.suffix, ".json")
            payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(payload["summary"]["cue_count"], 2)
        self.assertEqual(payload["summary"]["token_count"], 3)
        self.assertEqual(len(payload["cues"]), 2)
        self.assertGreaterEqual(len(payload["events"]), 1)
        first = payload["cues"][0]
        for key in ("start", "end", "speaker", "text"):
            self.assertIn(key, first)
        self.assertEqual(first["speaker"], "spk_000")


class TxtRenderTests(unittest.TestCase):
    def test_timestamps_toggle_changes_txt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session = _make_session(Path(tmp))
            _record_two_speakers(session)
            with_ts = session.export_to(
                output_path=str(Path(tmp) / "ts.txt"), export_format="txt", include_timestamps=True
            ).read_text(encoding="utf-8")
            no_ts = session.export_to(
                output_path=str(Path(tmp) / "nots.txt"), export_format="txt", include_timestamps=False
            ).read_text(encoding="utf-8")
        self.assertIn("00:00:00.000", with_ts)
        self.assertNotIn("00:00:00.000", no_ts)
        self.assertIn("hello world", no_ts)

    def test_display_format_writes_overlay_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session = _make_session(Path(tmp))
            _record_two_speakers(session)
            out = session.export_to(output_path=str(Path(tmp) / "d.txt"), export_format="display")
            body = out.read_text(encoding="utf-8")
        self.assertEqual(out.suffix, ".txt")
        self.assertEqual(body.strip(), "[spk_000] hello world")


if __name__ == "__main__":
    unittest.main()
