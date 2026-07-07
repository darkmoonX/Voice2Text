from __future__ import annotations

import importlib.util
import re
from pathlib import Path

from voice2text.audio_capture import AudioChunk
from voice2text.config import RuntimeConfig
from voice2text.pipeline.transcript_exporter import TranscriptExportOptions, TranscriptExporterSession


def _load_compare_module():
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "diagnostics" / "compare_test_data_whisperx.py"
    spec = importlib.util.spec_from_file_location("compare_test_data_whisperx", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_direct_chunking_offsets_export_tokens(monkeypatch, tmp_path):
    compare = _load_compare_module()

    class FakeTranscriber:
        def __init__(self) -> None:
            self.calls = 0
            self._device = "cuda"
            self._compute_type = "float16"

        def transcribe(self, chunk, language=None, channel_mode="mono") -> str:
            self.calls += 1
            duration = len(chunk.pcm16) / float(chunk.sample_rate * chunk.channels * 2)
            return f"chunk{self.calls}:{duration:.1f}"

        def get_last_transcription_meta(self):
            return {
                "detected_language": "zh",
                "token_timestamps": [
                    {
                        "start": 0.0,
                        "end": 0.5,
                        "score": 0.9,
                        "word": f"w{self.calls}",
                        "speaker": "",
                    }
                ],
            }

    fake = FakeTranscriber()
    monkeypatch.setattr(compare, "create_stt_transcriber", lambda *args, **kwargs: fake)
    monkeypatch.setattr(compare, "_dispose_transcriber", lambda transcriber: None)
    monkeypatch.setattr(compare, "_release_runtime_memory", lambda label: None)

    audio = AudioChunk(pcm16=b"\x01\x00" * 16000 * 65, sample_rate=16000, channels=1)
    exporter = TranscriptExporterSession(
        TranscriptExportOptions(
            enabled=True,
            formats=["json"],
            include_timestamps=True,
            include_speaker=True,
            output_dir=str(tmp_path),
        )
    )

    result = compare._run_direct(
        RuntimeConfig(source_language="zh"),
        audio,
        exporter=exporter,
        require_gpu=True,
        chunk_seconds=30.0,
    )

    assert fake.calls == 3
    assert result["meta"]["direct_chunk_count"] == 3
    written = exporter.finalize()
    payload = written[0].read_text(encoding="utf-8")
    assert '"start": 30.0' in payload
    assert '"start": 60.0' in payload


def test_direct_chunk_zero_keeps_long_audio_single_pass(monkeypatch, tmp_path):
    compare = _load_compare_module()

    class FakeTranscriber:
        def __init__(self) -> None:
            self.calls = 0
            self._device = "cuda"
            self._compute_type = "float16"

        def transcribe(self, chunk, language=None, channel_mode="mono") -> str:
            del language, channel_mode
            self.calls += 1
            duration = len(chunk.pcm16) / float(chunk.sample_rate * chunk.channels * 2)
            return f"chunk{self.calls}:{duration:.1f}"

        def get_last_transcription_meta(self):
            return {
                "detected_language": "zh",
                "token_timestamps": [
                    {
                        "start": 0.0,
                        "end": 0.5,
                        "score": 0.9,
                        "word": f"w{self.calls}",
                        "speaker": "",
                    }
                ],
            }

    fake = FakeTranscriber()
    monkeypatch.setattr(compare, "create_stt_transcriber", lambda *args, **kwargs: fake)
    monkeypatch.setattr(compare, "_dispose_transcriber", lambda transcriber: None)
    monkeypatch.setattr(compare, "_release_runtime_memory", lambda label: None)

    audio = AudioChunk(pcm16=b"\x01\x00" * 16000 * 200, sample_rate=16000, channels=1)
    exporter = TranscriptExporterSession(
        TranscriptExportOptions(
            enabled=True,
            formats=["json"],
            include_timestamps=True,
            include_speaker=True,
            output_dir=str(tmp_path),
        )
    )

    result = compare._run_direct(
        RuntimeConfig(source_language="zh"),
        audio,
        exporter=exporter,
        require_gpu=True,
        chunk_seconds=0.0,
    )

    assert fake.calls == 1
    assert result["meta"]["direct_requested_chunk_seconds"] == 0.0
    assert result["meta"]["direct_chunk_seconds"] == 0.0
    assert result["meta"]["direct_chunk_mode"] == "single"
    assert result["meta"]["direct_auto_chunked"] is False


def test_direct_auto_language_uses_subchunk_routing_inside_large_chunks(monkeypatch, tmp_path):
    compare = _load_compare_module()

    class FakeTranscriber:
        def __init__(self) -> None:
            self.calls = 0
            self.languages: list[object] = []
            self.durations: list[float] = []
            self._device = "cuda"
            self._compute_type = "float16"

        def transcribe(self, chunk, language=None, channel_mode="mono") -> str:
            del channel_mode
            self.calls += 1
            self.languages.append(language)
            duration = len(chunk.pcm16) / float(chunk.sample_rate * chunk.channels * 2)
            self.durations.append(duration)
            return f"chunk{self.calls}:{duration:.1f}"

        def get_last_transcription_meta(self):
            return {
                "detected_language": "en" if self.calls >= 3 else "zh",
                "token_timestamps": [
                    {
                        "start": 0.0,
                        "end": 0.5,
                        "score": 0.9,
                        "word": f"w{self.calls}",
                        "speaker": "",
                    }
                ],
            }

    fake = FakeTranscriber()
    monkeypatch.setattr(compare, "create_stt_transcriber", lambda *args, **kwargs: fake)
    monkeypatch.setattr(compare, "_dispose_transcriber", lambda transcriber: None)
    monkeypatch.setattr(compare, "_release_runtime_memory", lambda label: None)

    audio = AudioChunk(pcm16=b"\x01\x00" * 16000 * 120, sample_rate=16000, channels=1)
    exporter = TranscriptExporterSession(
        TranscriptExportOptions(
            enabled=True,
            formats=["json"],
            include_timestamps=True,
            include_speaker=True,
            output_dir=str(tmp_path),
        )
    )

    result = compare._run_direct(
        RuntimeConfig(source_language=None),
        audio,
        exporter=exporter,
        require_gpu=True,
        chunk_seconds=120.0,
        language_subchunk_seconds=30.0,
    )

    assert fake.calls == 4
    assert fake.languages == [None, None, None, None]
    assert fake.durations == [30.0, 30.0, 30.0, 30.0]
    assert result["meta"]["direct_chunk_count"] == 4
    assert result["meta"]["direct_language_subchunk_seconds"] == 30.0


def test_direct_chunk_zero_keeps_short_audio_single_pass(monkeypatch, tmp_path):
    compare = _load_compare_module()

    class FakeTranscriber:
        def __init__(self) -> None:
            self.calls = 0
            self._device = "cuda"
            self._compute_type = "float16"

        def transcribe(self, chunk, language=None, channel_mode="mono") -> str:
            del language, channel_mode
            self.calls += 1
            duration = len(chunk.pcm16) / float(chunk.sample_rate * chunk.channels * 2)
            return f"chunk{self.calls}:{duration:.1f}"

        def get_last_transcription_meta(self):
            return {"detected_language": "zh", "token_timestamps": []}

    fake = FakeTranscriber()
    monkeypatch.setattr(compare, "create_stt_transcriber", lambda *args, **kwargs: fake)
    monkeypatch.setattr(compare, "_dispose_transcriber", lambda transcriber: None)
    monkeypatch.setattr(compare, "_release_runtime_memory", lambda label: None)

    audio = AudioChunk(pcm16=b"\x01\x00" * 16000 * 20, sample_rate=16000, channels=1)
    exporter = TranscriptExporterSession(
        TranscriptExportOptions(
            enabled=True,
            formats=["json"],
            include_timestamps=True,
            include_speaker=True,
            output_dir=str(tmp_path),
        )
    )

    result = compare._run_direct(
        RuntimeConfig(source_language="zh"),
        audio,
        exporter=exporter,
        require_gpu=True,
        chunk_seconds=0.0,
    )

    assert fake.calls == 1
    assert result["meta"]["direct_requested_chunk_seconds"] == 0.0
    assert result["meta"]["direct_chunk_seconds"] == 0.0
    assert result["meta"]["direct_chunk_mode"] == "single"
    assert result["meta"]["direct_auto_chunked"] is False


def test_direct_chunking_reconciles_profile_speakers_before_export(monkeypatch, tmp_path):
    compare = _load_compare_module()

    class FakeTranscriber:
        def __init__(self) -> None:
            self.calls = 0
            self._device = "cuda"
            self._compute_type = "float16"

        def transcribe(self, chunk, language=None, channel_mode="mono") -> str:
            del chunk, language, channel_mode
            self.calls += 1
            return f"chunk{self.calls}"

        def get_last_transcription_meta(self):
            speaker = "SPK_000" if self.calls == 1 else "SPK_001"
            return {
                "detected_language": "zh",
                "token_timestamps": [
                    {
                        "start": 0.0,
                        "end": 0.5,
                        "score": 0.9,
                        "word": f"w{self.calls}",
                        "speaker": speaker,
                    }
                ],
            }

        def reconcile_speaker_profiles(self, *, threshold=None):
            assert threshold == 0.52
            return {
                "status": "done",
                "threshold": threshold,
                "merged_count": 1,
                "remap": {"SPK_001": "SPK_000"},
            }

    fake = FakeTranscriber()
    monkeypatch.setattr(compare, "create_stt_transcriber", lambda *args, **kwargs: fake)
    monkeypatch.setattr(compare, "_dispose_transcriber", lambda transcriber: None)
    monkeypatch.setattr(compare, "_release_runtime_memory", lambda label: None)

    audio = AudioChunk(pcm16=b"\x01\x00" * 16000 * 35, sample_rate=16000, channels=1)
    exporter = TranscriptExporterSession(
        TranscriptExportOptions(
            enabled=True,
            formats=["json"],
            include_timestamps=True,
            include_speaker=True,
            output_dir=str(tmp_path),
        )
    )

    result = compare._run_direct(
        RuntimeConfig(source_language="zh"),
        audio,
        exporter=exporter,
        require_gpu=True,
        chunk_seconds=30.0,
        speaker_profile_reconcile_threshold=0.52,
    )

    assert result["meta"]["speaker_profile_reconciliation"]["merged_count"] == 1
    written = exporter.finalize()
    payload = written[0].read_text(encoding="utf-8")
    assert "spk_001" not in payload
    assert payload.count('"speaker": "spk_000"') == 2


def test_project_txt_to_single_line_collapses_repeated_speaker_markers():
    compare = _load_compare_module()
    text = "\n".join(
        [
            "[00:00:00.000 -> 00:00:01.000] [SPK_000] alpha",
            "[00:00:01.000 -> 00:00:02.000] [SPK_000] beta",
            "[00:00:02.000 -> 00:00:03.000] [SPEAKER_01] gamma",
            "[00:00:03.000 -> 00:00:04.000] [SPEAKER_01] delta",
        ]
    )

    assert compare._project_txt_to_single_line(text) == "[spk_000] alpha beta [spk_001] gamma delta"


def test_speaker_compare_summary_reports_extra_realtime_speakers():
    compare = _load_compare_module()

    summary = compare._speaker_compare_summary(
        "[spk_000] alpha [spk_001] beta",
        "[spk_000] alpha [spk_002] beta [spk_003] gamma",
    )

    assert summary["reference_speaker_count"] == 2
    assert summary["realtime_speaker_count"] == 3
    assert summary["reference_switch_count"] == 1
    assert summary["realtime_switch_count"] == 2
    assert summary["realtime_extra_speaker_labels"] == ["spk_002", "spk_003"]
    assert summary["realtime_missing_speaker_labels"] == ["spk_001"]
    assert summary["speaker_sequence_distance"] == 2


def test_speaker_compare_only_runs_for_accurate_profile():
    compare = _load_compare_module()

    assert compare._should_compare_speakers("accurate") is True
    assert compare._should_compare_speakers("fast") is False
    disabled = compare._speaker_compare_disabled_summary("fast")
    assert disabled["enabled"] is False
    assert disabled["disabled_reason"] == "profile=fast"


def test_realtime_speaker_label_cap_defaults_to_reference_count_for_accurate():
    compare = _load_compare_module()

    cap = compare._resolve_realtime_speaker_label_cap(
        requested_max_speakers=0,
        direct_text_for_compare="[spk_000] alpha [spk_001] beta [spk_000] gamma",
        profile="accurate",
    )

    assert cap == 2


def test_realtime_speaker_label_cap_uses_explicit_request_first():
    compare = _load_compare_module()

    cap = compare._resolve_realtime_speaker_label_cap(
        requested_max_speakers=3,
        direct_text_for_compare="[spk_000] alpha [spk_001] beta",
        profile="accurate",
    )

    assert cap == 3


def test_realtime_speaker_label_cap_disabled_for_fast_profile_without_explicit_request():
    compare = _load_compare_module()

    cap = compare._resolve_realtime_speaker_label_cap(
        requested_max_speakers=0,
        direct_text_for_compare="[spk_000] alpha [spk_001] beta",
        profile="fast",
    )

    assert cap == 0


def test_html_diff_wraps_same_text_as_normal_weight():
    compare = _load_compare_module()

    html = compare._render_html_diff(["a", "b", "c"], ["a", "x", "c"], unit="char")

    assert "<span class='same'>a</span>" in html
    assert "<span class='same'>c</span>" in html
    assert "<span class='extra'>x</span>" in html
    assert "<span class='missing'>b</span>" in html


def test_html_diff_groups_adjacent_runs_into_single_spans():
    compare = _load_compare_module()

    html = compare._render_html_diff(["a", "b", "c", "d"], ["a", "b", "x", "y", "d"], unit="char")

    assert "<span class='same'>ab</span>" in html
    assert "<span class='extra'>xy</span>" in html
    assert "<span class='missing'>c</span>" in html
    assert "<span class='same'>d</span>" in html


def test_html_reference_diff_preserves_compare_line_breaks():
    compare = _load_compare_module()

    diff = compare._build_reference_diff(
        "第一行\n第二行",
        "第一行\n第二行",
        language_hint="zh",
        preserve_newlines=True,
    )

    assert "\n" in diff["reference_aligned"]
    assert "<br>" in diff["realtime_annotated_html"]


def test_project_txt_to_compare_lines_keeps_cue_boundaries_without_speakers():
    compare = _load_compare_module()
    text = "\n".join(
        [
            "[00:00:49.643 -> 00:00:56.688] [SPK_000] alpha beta",
            "[00:01:25.910 -> 00:01:28.000] [SPK_000] gamma delta",
        ]
    )

    assert compare._project_txt_to_compare_lines(text) == "alpha beta\ngamma delta"


def test_project_txt_to_compare_lines_can_keep_speaker_headers():
    compare = _load_compare_module()
    text = "\n".join(
        [
            "[00:00:49.643 -> 00:00:56.688] [SPK_000] alpha beta",
            "[00:01:25.910 -> 00:01:28.000] [SPEAKER_01] gamma delta",
        ]
    )

    assert compare._project_txt_to_compare_lines(text, include_speaker=True) == (
        "[spk_000] alpha beta\n[spk_001] gamma delta"
    )


def test_html_compare_normalization_keeps_speaker_headers():
    compare = _load_compare_module()

    normalized = compare._normalize_for_html_compare("[SPEAKER_01] Hello\nS2: World")

    assert normalized == "[spk_001] hello\n[spk_002] world"


def test_speaker_sequence_html_renders_reference_and_realtime_rows():
    compare = _load_compare_module()
    summary = compare._speaker_compare_summary("[spk_000] a [spk_001] b", "[spk_000] a [spk_002] b")

    html = compare._render_speaker_sequence_html(summary)

    assert "WhisperX" in html
    assert "Realtime" in html
    assert "spk_001" in html
    assert "spk_002" in html


def test_rewrite_speaker_labels_restarts_at_zero_by_first_visible_occurrence():
    compare = _load_compare_module()

    rewritten = compare._rewrite_speaker_labels_text("[spk_001] alpha\n[spk_003] beta\n[spk_001] gamma")

    assert rewritten == "[spk_000] alpha\n[spk_001] beta\n[spk_000] gamma"


def test_rewrite_speaker_labels_can_collapse_profile_ids_before_renumbering():
    compare = _load_compare_module()

    rewritten = compare._rewrite_speaker_labels_text(
        "[spk_010] alpha\n[spk_099] beta\n[spk_011] gamma",
        profile_remap={"spk_099": "spk_010"},
    )

    assert rewritten == "[spk_000] alpha\n[spk_000] beta\n[spk_001] gamma"


def test_coalesce_txt_same_speaker_cues_after_profile_remap():
    compare = _load_compare_module()

    text = "\n".join(
        [
            "[00:00:00.000 -> 00:00:01.000] [spk_001] alpha",
            "[00:00:01.000 -> 00:00:02.000] [spk_001] beta",
            "[00:00:02.500 -> 00:00:03.000] [spk_002] gamma",
        ]
    )

    assert compare._coalesce_txt_same_speaker_cues(text) == "\n".join(
        [
            "[00:00:00.000 -> 00:00:02.000] [spk_001] alpha beta",
            "[00:00:02.500 -> 00:00:03.000] [spk_002] gamma",
        ]
    )


def test_coalesce_txt_same_speaker_cues_ignores_sentence_punctuation_boundaries():
    compare = _load_compare_module()

    text = "\n".join(
        [
            "[00:00:00.000 -> 00:00:01.000] [spk_001] alpha.",
            "[00:00:01.100 -> 00:00:02.000] [spk_001] beta",
            "[00:00:05.000 -> 00:00:06.000] [spk_001] gamma",
        ]
    )

    assert compare._coalesce_txt_same_speaker_cues(text) == "\n".join(
        [
            "[00:00:00.000 -> 00:00:02.000] [spk_001] alpha. beta",
            "[00:00:05.000 -> 00:00:06.000] [spk_001] gamma",
        ]
    )


def test_coalesce_srt_same_speaker_cues_after_profile_remap():
    compare = _load_compare_module()

    text = "\n\n".join(
        [
            "1\n00:00:00,000 --> 00:00:01,000\n[spk_001] alpha",
            "2\n00:00:01,000 --> 00:00:02,000\n[spk_001] beta",
            "3\n00:00:03,000 --> 00:00:04,000\n[spk_002] gamma",
        ]
    )

    assert compare._coalesce_srt_same_speaker_cues(text) == "\n\n".join(
        [
            "1\n00:00:00,000 --> 00:00:02,000\n[spk_001] alpha beta",
            "2\n00:00:03,000 --> 00:00:04,000\n[spk_002] gamma",
        ]
    )


def test_coalesce_json_same_speaker_cues_updates_cue_count():
    compare = _load_compare_module()
    payload = {
        "meta": {"cue_count": 3},
        "cues": [
            {"start": 0.0, "end": 1.0, "speaker": "spk_001", "text": "alpha"},
            {"start": 1.0, "end": 2.0, "speaker": "spk_001", "text": "beta"},
            {"start": 4.5, "end": 5.0, "speaker": "spk_001", "text": "gamma"},
        ],
    }

    out = compare._coalesce_json_same_speaker_cues(payload)

    assert out["meta"]["cue_count"] == 2
    assert out["cues"][0]["end"] == 2.0
    assert out["cues"][0]["text"] == "alpha beta"
    assert out["cues"][1]["text"] == "gamma"


def test_compare_html_page_does_not_prefix_reference_or_realtime_with_spaces():
    compare = _load_compare_module()

    page = compare._build_compare_html_page(
        input_path="input.wav",
        compare_unit="char",
        cer=0.0,
        distance=0,
        reference_text="[spk_000] alpha",
        realtime_annotated_html="<span class='same'>[spk_000] alpha</span>",
        marker_line="....",
    )

    assert "\n    [spk_000] alpha" not in page
    assert "\n[spk_000] alpha" in page
    assert "\n    <span class='same'>" not in page


def test_word_unit_html_diff_preserves_spaces_between_words():
    # Regression: the word tokenizer drops whitespace, so the HTML diff must
    # reinsert word separators (compare.html previously collapsed English to
    # "helloworld" while compare.txt / realtime_project.txt kept the spaces).
    compare = _load_compare_module()

    ref = "the quick brown fox jumps over the lazy dog"
    cand = "the quick brown cat jumps over a lazy dog"
    unit = compare._resolve_compare_unit(ref, cand, "en")
    assert unit == "word"
    ref_tokens = compare._tokenize_for_compare(ref, unit=unit)
    cand_tokens = compare._tokenize_for_compare(cand, unit=unit)
    html = compare._render_html_diff(ref_tokens, cand_tokens, unit=unit)
    flat = re.sub(r"<[^>]+>", "", html).replace("&nbsp;", " ")

    # words keep their separating spaces (the bug collapsed them to "thequickbrown")
    assert "the quick brown" in flat
    assert "jumps over" in flat and "lazy dog" in flat
    # no two adjacent words glued together (the bug signature)
    assert "browncat" not in flat and "quickbrown" not in flat

    # punctuation still glues correctly (no space before .,!?) and brackets
    ref2 = "Hello, world! (test) it works."
    cand2 = "Hello world (test) it works."
    unit2 = compare._resolve_compare_unit(ref2, cand2, "en")
    html2 = compare._render_html_diff(
        compare._tokenize_for_compare(ref2, unit=unit2),
        compare._tokenize_for_compare(cand2, unit=unit2),
        unit=unit2,
    )
    flat2 = re.sub(r"<[^>]+>", "", html2).replace("&nbsp;", " ")
    assert "it works." in flat2
    assert "works ." not in flat2  # period not detached from its word
    assert "(test)" in flat2  # brackets stay tight around the word
