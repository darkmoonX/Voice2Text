from __future__ import annotations

from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from voice2text.config import RuntimeConfig
import voice2text.pipeline.transcription_loop as transcription_loop_module
from voice2text.capture import AudioChunk
from voice2text.pipeline.transcription_loop import TranscriptionLoopDeps, TranscriptionLoopEngine
from voice2text.pipeline.subtitle_assembler import SubtitleAssembler


def _engine(config: RuntimeConfig) -> TranscriptionLoopEngine:
    deps = TranscriptionLoopDeps(
        config=config,
        subtitle_assembler=SubtitleAssembler(),
        text_delta_logger=SimpleNamespace(log=lambda *args, **kwargs: None),
        segment_artifacts=SimpleNamespace(),
        gpu_telemetry=SimpleNamespace(),
        get_capture=lambda: None,
        get_transcriber=lambda: None,
        get_preprocess_pipeline=lambda: None,
        get_translator=lambda: None,
        recover_capture_backend=lambda: False,
        recover_from_runtime_transcription_error=lambda error: False,
        emit_status=lambda message: None,
        emit_debug_event=lambda event: None,
        emit_subtitle_ready=lambda source, translated: None,
        record_transcript_event=lambda event: None,
    )
    return TranscriptionLoopEngine(deps)


def test_auto_language_keeps_asr_hint_auto_after_runtime_lock() -> None:
    engine = _engine(RuntimeConfig(source_language=None))

    engine._update_auto_source_language_hint(
        {"detected_language": "zh", "token_count": 10, "stability_ratio": 0.9}
    )

    assert engine._runtime_source_language_hint() is None
    assert engine._runtime_display_language_hint() == "zh"


def test_explicit_language_still_passes_hint_to_asr() -> None:
    engine = _engine(RuntimeConfig(source_language="zh"))

    assert engine._runtime_source_language_hint() == "zh"
    assert engine._runtime_display_language_hint() == "zh"


class TranscriptionLoopFinalizeTests(unittest.TestCase):
    def test_eof_finalizes_and_emits_tail_snapshot(self) -> None:
        events: list[dict[str, object]] = []
        subtitles: list[tuple[str, str]] = []

        class FakeCapture:
            sample_rate = 16000
            channels = 1

            def read_chunk(self, timeout: float = 0.25):
                return None

            def is_finished(self) -> bool:
                return True

        class FakeAssembler(SubtitleAssembler):
            def __init__(self) -> None:
                super().__init__()
                self.finalize_calls = 0

            def finalize(self) -> str:
                self.finalize_calls += 1
                return "final tail"

        assembler = FakeAssembler()
        deps = TranscriptionLoopDeps(
            config=RuntimeConfig(),
            subtitle_assembler=assembler,
            text_delta_logger=SimpleNamespace(log=lambda *args, **kwargs: None),
            segment_artifacts=SimpleNamespace(),
            gpu_telemetry=SimpleNamespace(
                maybe_emit=lambda *args, **kwargs: None,
                emit_summary=lambda *args, **kwargs: None,
            ),
            get_capture=lambda: FakeCapture(),
            get_transcriber=lambda: SimpleNamespace(),
            get_preprocess_pipeline=lambda: None,
            get_translator=lambda: None,
            recover_capture_backend=lambda: False,
            recover_from_runtime_transcription_error=lambda error: False,
            emit_status=lambda message: None,
            emit_debug_event=lambda event: None,
            emit_subtitle_ready=lambda source, translated: subtitles.append((source, translated)),
            record_transcript_event=events.append,
        )
        running = Event()
        running.set()

        TranscriptionLoopEngine(deps).run(running)

        self.assertEqual(assembler.finalize_calls, 1)
        self.assertEqual(subtitles, [("final tail", "")])
        self.assertEqual(events[-1]["source_text"], "final tail")
        self.assertTrue(events[-1]["meta"]["snapshot_final"])


class TranscriptionLoopRelabelWiringTests(unittest.TestCase):
    """Round 0048: the loop wires set_relabel_resolver(None) when disabled (default) and a
    callable resolver when enabled -- this is the observable proxy for "no rolling audio
    buffer allocated" (relabel_buffer/the resolver closure are local to run(), not exposed)."""

    @staticmethod
    def _run_with_config(config: RuntimeConfig) -> "list[object]":
        resolver_calls: list[object] = []

        class RecordingAssembler(SubtitleAssembler):
            def set_relabel_resolver(self, resolver, **kwargs) -> None:  # type: ignore[override]
                resolver_calls.append((resolver, kwargs))
                super().set_relabel_resolver(resolver, **kwargs)

        class FakeCapture:
            sample_rate = 16000
            channels = 1

            def read_chunk(self, timeout: float = 0.25):
                return None

            def is_finished(self) -> bool:
                return True

        deps = TranscriptionLoopDeps(
            config=config,
            subtitle_assembler=RecordingAssembler(),
            text_delta_logger=SimpleNamespace(log=lambda *args, **kwargs: None),
            segment_artifacts=SimpleNamespace(),
            gpu_telemetry=SimpleNamespace(
                maybe_emit=lambda *args, **kwargs: None,
                emit_summary=lambda *args, **kwargs: None,
            ),
            get_capture=lambda: FakeCapture(),
            get_transcriber=lambda: SimpleNamespace(),
            get_preprocess_pipeline=lambda: None,
            get_translator=lambda: None,
            recover_capture_backend=lambda: False,
            recover_from_runtime_transcription_error=lambda error: False,
            emit_status=lambda message: None,
            emit_debug_event=lambda event: None,
            emit_subtitle_ready=lambda source, translated: None,
            record_transcript_event=lambda event: None,
        )
        running = Event()
        running.set()
        TranscriptionLoopEngine(deps).run(running)
        return resolver_calls

    def test_relabel_resolver_not_wired_when_disabled(self) -> None:
        calls = self._run_with_config(RuntimeConfig(subtitle_relabel_enabled=False))
        self.assertEqual(len(calls), 1)
        self.assertIsNone(calls[0][0])

    def test_relabel_resolver_wired_when_enabled(self) -> None:
        calls = self._run_with_config(RuntimeConfig(subtitle_relabel_enabled=True))
        self.assertEqual(len(calls), 1)
        self.assertTrue(callable(calls[0][0]))
        self.assertFalse(calls[0][1].get("defer"))

    def test_relabel_resolver_defer_flag_reflects_async_config(self) -> None:
        calls = self._run_with_config(
            RuntimeConfig(subtitle_relabel_enabled=True, subtitle_relabel_async=True)
        )
        self.assertEqual(len(calls), 1)
        self.assertTrue(callable(calls[0][0]))
        self.assertTrue(calls[0][1].get("defer"))

    def test_relabel_resolver_defer_false_when_async_disabled(self) -> None:
        calls = self._run_with_config(
            RuntimeConfig(subtitle_relabel_enabled=True, subtitle_relabel_async=False)
        )
        self.assertFalse(calls[0][1].get("defer"))


class TranscriptionLoopAliasRemapWiringTests(unittest.TestCase):
    """Round 0053: an auto-reconcile remap surfaced in this window's transcription meta must
    reach the assembler's alias-continuity hook before the window's words are rendered."""

    def test_auto_reconcile_remap_is_applied_before_merge(self) -> None:
        from voice2text.capture import AudioChunk

        remap_calls: list[dict] = []

        class RecordingAssembler(SubtitleAssembler):
            def apply_speaker_alias_remap(self, remap):  # type: ignore[override]
                remap_calls.append(dict(remap or {}))
                super().apply_speaker_alias_remap(remap)

        class FakeCapture:
            sample_rate = 16000
            channels = 1

            def __init__(self) -> None:
                self._sent = False

            def read_chunk(self, timeout: float = 0.25):
                if self._sent:
                    return None
                self._sent = True
                return AudioChunk(pcm16=b"\x00\x00" * 160000, sample_rate=16000, channels=1)

            def is_finished(self) -> bool:
                return True

        class FakeTranscriber:
            def transcribe(self, chunk, language=None, channel_mode="mono"):
                return "hello"

            def get_last_transcription_meta(self):
                return {
                    "speaker_profile_stats": {
                        "auto_reconcile": {"remap": {"SPK_003": "SPK_007"}},
                    },
                }

        # Single shared instances: get_capture()/get_transcriber() are called MULTIPLE times
        # per window by the real loop -- a lambda that constructs a fresh FakeCapture() each
        # call would reset `_sent` every time, re-injecting the same chunk forever (the loop
        # never sees is_finished() take effect because it keeps getting a brand new object).
        capture = FakeCapture()
        transcriber = FakeTranscriber()
        deps = TranscriptionLoopDeps(
            config=RuntimeConfig(),
            subtitle_assembler=RecordingAssembler(),
            text_delta_logger=SimpleNamespace(log=lambda *args, **kwargs: None),
            segment_artifacts=SimpleNamespace(),
            gpu_telemetry=SimpleNamespace(
                maybe_emit=lambda *args, **kwargs: None,
                emit_summary=lambda *args, **kwargs: None,
            ),
            get_capture=lambda: capture,
            get_transcriber=lambda: transcriber,
            get_preprocess_pipeline=lambda: None,
            get_translator=lambda: None,
            recover_capture_backend=lambda: False,
            recover_from_runtime_transcription_error=lambda error: False,
            emit_status=lambda message: None,
            emit_debug_event=lambda event: None,
            emit_subtitle_ready=lambda source, translated: None,
            record_transcript_event=lambda event: None,
        )
        running = Event()
        running.set()
        TranscriptionLoopEngine(deps).run(running)

        # One shared chunk gets processed across several hop-sized windows (startup silence
        # padding + one chunk) -- assert engagement and correctness, not an exact window count.
        self.assertTrue(remap_calls)
        self.assertTrue(all(call == {"SPK_003": "SPK_007"} for call in remap_calls))

    def test_no_remap_when_reconcile_did_not_fire(self) -> None:
        from voice2text.capture import AudioChunk

        remap_calls: list[dict] = []

        class RecordingAssembler(SubtitleAssembler):
            def apply_speaker_alias_remap(self, remap):  # type: ignore[override]
                remap_calls.append(dict(remap or {}))
                super().apply_speaker_alias_remap(remap)

        class FakeCapture:
            sample_rate = 16000
            channels = 1

            def __init__(self) -> None:
                self._sent = False

            def read_chunk(self, timeout: float = 0.25):
                if self._sent:
                    return None
                self._sent = True
                return AudioChunk(pcm16=b"\x00\x00" * 160000, sample_rate=16000, channels=1)

            def is_finished(self) -> bool:
                return True

        class FakeTranscriber:
            def transcribe(self, chunk, language=None, channel_mode="mono"):
                return "hello"

            def get_last_transcription_meta(self):
                return {"speaker_profile_stats": {"auto_reconcile": {"remap": {}}}}

        capture = FakeCapture()
        transcriber = FakeTranscriber()
        deps = TranscriptionLoopDeps(
            config=RuntimeConfig(),
            subtitle_assembler=RecordingAssembler(),
            text_delta_logger=SimpleNamespace(log=lambda *args, **kwargs: None),
            segment_artifacts=SimpleNamespace(),
            gpu_telemetry=SimpleNamespace(
                maybe_emit=lambda *args, **kwargs: None,
                emit_summary=lambda *args, **kwargs: None,
            ),
            get_capture=lambda: capture,
            get_transcriber=lambda: transcriber,
            get_preprocess_pipeline=lambda: None,
            get_translator=lambda: None,
            recover_capture_backend=lambda: False,
            recover_from_runtime_transcription_error=lambda error: False,
            emit_status=lambda message: None,
            emit_debug_event=lambda event: None,
            emit_subtitle_ready=lambda source, translated: None,
            record_transcript_event=lambda event: None,
        )
        running = Event()
        running.set()
        TranscriptionLoopEngine(deps).run(running)

        self.assertEqual(remap_calls, [])


class TranscriptionLoopSpeakerCountHintTests(unittest.TestCase):
    class _SyncThread:
        def __init__(self, target, name=None, daemon=None):
            self._target = target
            self.name = name
            self.daemon = daemon

        def start(self):
            self._target()

    class _HeldThread:
        started: list[object] = []

        def __init__(self, target, name=None, daemon=None):
            self._target = target
            self.name = name
            self.daemon = daemon

        def start(self):
            type(self).started.append(self)

    class _Capture:
        sample_rate = 16000
        channels = 1

        def __init__(self, chunk_count: int) -> None:
            self._remaining = int(chunk_count)

        def read_chunk(self, timeout: float = 0.25):
            if self._remaining <= 0:
                return None
            self._remaining -= 1
            return AudioChunk(pcm16=b"\x00\x00" * 16000, sample_rate=16000, channels=1)

        def is_finished(self) -> bool:
            return self._remaining <= 0

    def _run_loop(self, config: RuntimeConfig, transcriber, *, chunk_count: int = 3) -> None:
        capture = self._Capture(chunk_count)
        deps = TranscriptionLoopDeps(
            config=config,
            subtitle_assembler=SubtitleAssembler(),
            text_delta_logger=SimpleNamespace(log=lambda *args, **kwargs: None),
            segment_artifacts=SimpleNamespace(),
            gpu_telemetry=SimpleNamespace(
                maybe_emit=lambda *args, **kwargs: None,
                emit_summary=lambda *args, **kwargs: None,
            ),
            get_capture=lambda: capture,
            get_transcriber=lambda: transcriber,
            get_preprocess_pipeline=lambda: None,
            get_translator=lambda: None,
            recover_capture_backend=lambda: False,
            recover_from_runtime_transcription_error=lambda error: False,
            emit_status=lambda message: None,
            emit_debug_event=lambda event: None,
            emit_subtitle_ready=lambda source, translated: None,
            record_transcript_event=lambda event: None,
        )
        running = Event()
        running.set()
        TranscriptionLoopEngine(deps).run(running)

    def test_default_off_makes_no_count_calls(self) -> None:
        class Transcriber:
            def __init__(self) -> None:
                self.estimate_calls = 0
                self.cap_calls: list[int] = []

            def estimate_speaker_count(self, *args, **kwargs):
                self.estimate_calls += 1
                return 2

            def set_speaker_count_cap(self, cap: int) -> None:
                self.cap_calls.append(cap)

            def transcribe(self, chunk, language=None, channel_mode="mono"):
                return ""

        transcriber = Transcriber()

        with patch.object(transcription_loop_module, "Thread", self._SyncThread):
            self._run_loop(RuntimeConfig(), transcriber, chunk_count=3)

        self.assertEqual(transcriber.estimate_calls, 0)
        self.assertEqual(transcriber.cap_calls, [])

    def test_count_hint_cap_is_monotonic_and_clamped_to_operator_hint(self) -> None:
        class Transcriber:
            def __init__(self) -> None:
                self._counts = [2, 1, 4]
                self.cap_calls: list[int] = []

            def estimate_speaker_count(self, *args, **kwargs):
                return self._counts.pop(0)

            def set_speaker_count_cap(self, cap: int) -> None:
                self.cap_calls.append(int(cap))

            def transcribe(self, chunk, language=None, channel_mode="mono"):
                return ""

        transcriber = Transcriber()
        cfg = RuntimeConfig(
            whisperx_speaker_count_hint_enabled=True,
            whisperx_speaker_count_hint_seconds=1.0,
            whisperx_speaker_count_hint_window_seconds=2.0,
            whisperx_speaker_count_hint_sliver_floor_seconds=1.5,
            whisperx_diarization_max_speakers=3,
        )

        with patch.object(transcription_loop_module, "Thread", self._SyncThread):
            self._run_loop(cfg, transcriber, chunk_count=3)

        self.assertEqual(transcriber.cap_calls, [3, 3, 4])

    def test_count_hint_cap_is_clamped_to_online_profile_count(self) -> None:
        class Transcriber:
            def __init__(self) -> None:
                self._counts = [2, 2, 2]
                self.cap_calls: list[int] = []

            def estimate_speaker_count(self, *args, **kwargs):
                return self._counts.pop(0)

            def get_speaker_profile_count(self) -> int:
                return 5

            def set_speaker_count_cap(self, cap: int) -> None:
                self.cap_calls.append(int(cap))

            def transcribe(self, chunk, language=None, channel_mode="mono"):
                return ""

        transcriber = Transcriber()
        cfg = RuntimeConfig(
            whisperx_speaker_count_hint_enabled=True,
            whisperx_speaker_count_hint_seconds=1.0,
            whisperx_speaker_count_hint_window_seconds=2.0,
            whisperx_speaker_count_hint_sliver_floor_seconds=1.5,
            whisperx_diarization_max_speakers=3,
        )

        with patch.object(transcription_loop_module, "Thread", self._SyncThread):
            self._run_loop(cfg, transcriber, chunk_count=3)

        self.assertEqual(transcriber.cap_calls, [5, 5, 5])

    def test_count_hint_operator_zero_matches_previous_cap_sequence_when_online_count_zero(self) -> None:
        def previous_round_cap_sequence(estimates: list[int], operator_max_speaker_count: int) -> list[int]:
            observed_max_speaker_count = 0
            cap_calls: list[int] = []
            for count in estimates:
                observed_max_speaker_count = max(observed_max_speaker_count, int(max(0, count)))
                if observed_max_speaker_count > 0:
                    effective_cap = (
                        max(operator_max_speaker_count, observed_max_speaker_count)
                        if operator_max_speaker_count > 0
                        else observed_max_speaker_count
                    )
                    cap_calls.append(effective_cap)
            return cap_calls

        class Transcriber:
            def __init__(self) -> None:
                self._counts = [2, 1, 4]
                self.cap_calls: list[int] = []

            def estimate_speaker_count(self, *args, **kwargs):
                return self._counts.pop(0)

            def get_speaker_profile_count(self) -> int:
                return 0

            def set_speaker_count_cap(self, cap: int) -> None:
                self.cap_calls.append(int(cap))

            def transcribe(self, chunk, language=None, channel_mode="mono"):
                return ""

        estimates = [2, 1, 4]
        transcriber = Transcriber()
        cfg = RuntimeConfig(
            whisperx_speaker_count_hint_enabled=True,
            whisperx_speaker_count_hint_seconds=1.0,
            whisperx_speaker_count_hint_window_seconds=2.0,
            whisperx_speaker_count_hint_sliver_floor_seconds=1.5,
            whisperx_diarization_max_speakers=0,
        )

        with patch.object(transcription_loop_module, "Thread", self._SyncThread):
            self._run_loop(cfg, transcriber, chunk_count=3)

        self.assertEqual(transcriber.cap_calls, previous_round_cap_sequence(estimates, 0))

    def test_count_hint_single_flight_skips_second_trigger_while_worker_is_running(self) -> None:
        class Transcriber:
            def __init__(self) -> None:
                self.estimate_calls = 0

            def estimate_speaker_count(self, *args, **kwargs):
                self.estimate_calls += 1
                return 2

            def set_speaker_count_cap(self, cap: int) -> None:
                pass

            def transcribe(self, chunk, language=None, channel_mode="mono"):
                return ""

        transcriber = Transcriber()
        cfg = RuntimeConfig(
            whisperx_speaker_count_hint_enabled=True,
            whisperx_speaker_count_hint_seconds=1.0,
            whisperx_speaker_count_hint_window_seconds=2.0,
        )
        self._HeldThread.started = []

        with patch.object(transcription_loop_module, "Thread", self._HeldThread):
            self._run_loop(cfg, transcriber, chunk_count=3)

        self.assertEqual(len(self._HeldThread.started), 1)
        self.assertEqual(transcriber.estimate_calls, 0)


class TranscriptionLoopRollingPromptTests(unittest.TestCase):
    class _Capture:
        sample_rate = 16000
        channels = 1

        def __init__(self, chunk_count: int) -> None:
            self._remaining = int(chunk_count)

        def read_chunk(self, timeout: float = 0.25):
            if self._remaining <= 0:
                return None
            self._remaining -= 1
            return AudioChunk(pcm16=b"\x00\x00" * 16000, sample_rate=16000, channels=1)

        def is_finished(self) -> bool:
            return self._remaining <= 0

    def _run_loop(
        self,
        config: RuntimeConfig,
        transcriber,
        assembler: SubtitleAssembler | None = None,
        *,
        chunk_count: int = 2,
    ) -> None:
        capture = self._Capture(chunk_count)
        deps = TranscriptionLoopDeps(
            config=config,
            subtitle_assembler=assembler or SubtitleAssembler(),
            text_delta_logger=SimpleNamespace(log=lambda *args, **kwargs: None),
            segment_artifacts=SimpleNamespace(),
            gpu_telemetry=SimpleNamespace(
                maybe_emit=lambda *args, **kwargs: None,
                emit_summary=lambda *args, **kwargs: None,
            ),
            get_capture=lambda: capture,
            get_transcriber=lambda: transcriber,
            get_preprocess_pipeline=lambda: None,
            get_translator=lambda: None,
            recover_capture_backend=lambda: False,
            recover_from_runtime_transcription_error=lambda error: False,
            emit_status=lambda message: None,
            emit_debug_event=lambda event: None,
            emit_subtitle_ready=lambda source, translated: None,
            record_transcript_event=lambda event: None,
        )
        running = Event()
        running.set()
        TranscriptionLoopEngine(deps).run(running)

    def test_enabled_sets_prompt_tail_once_per_window_before_transcribe(self) -> None:
        events: list[tuple[str, object]] = []

        class RecordingAssembler(SubtitleAssembler):
            def __init__(self) -> None:
                super().__init__()
                self.prompt_tail_calls: list[int] = []

            def get_prompt_tail(self, max_chars: int) -> str:
                self.prompt_tail_calls.append(int(max_chars))
                prompt = f"tail-{len(self.prompt_tail_calls)}-{max_chars}"
                events.append(("get_prompt_tail", prompt))
                return prompt

        class Transcriber:
            def __init__(self) -> None:
                self.prompt_calls: list[str] = []
                self.transcribe_calls = 0

            def set_initial_prompt(self, prompt: str) -> None:
                self.prompt_calls.append(prompt)
                events.append(("set_initial_prompt", prompt))

            def transcribe(self, chunk, language=None, channel_mode="mono"):
                self.transcribe_calls += 1
                events.append(("transcribe", self.transcribe_calls))
                return ""

        assembler = RecordingAssembler()
        transcriber = Transcriber()

        self._run_loop(
            RuntimeConfig(segment_seconds=1.0, hop_seconds=0.5, whisperx_rolling_prompt_chars=160),
            transcriber,
            assembler,
        )

        self.assertGreater(transcriber.transcribe_calls, 0)
        self.assertEqual(assembler.prompt_tail_calls, [160] * transcriber.transcribe_calls)
        self.assertEqual(
            transcriber.prompt_calls,
            [f"tail-{index}-160" for index in range(1, transcriber.transcribe_calls + 1)],
        )
        self.assertEqual(len(events), transcriber.transcribe_calls * 3)
        for index in range(0, len(events), 3):
            self.assertEqual(events[index][0], "get_prompt_tail")
            self.assertEqual(events[index + 1], ("set_initial_prompt", events[index][1]))
            self.assertEqual(events[index + 2][0], "transcribe")

    def test_default_zero_never_sets_initial_prompt(self) -> None:
        class Transcriber:
            def __init__(self) -> None:
                self.prompt_calls: list[str] = []
                self.transcribe_calls = 0

            def set_initial_prompt(self, prompt: str) -> None:
                self.prompt_calls.append(prompt)

            def transcribe(self, chunk, language=None, channel_mode="mono"):
                self.transcribe_calls += 1
                return ""

        transcriber = Transcriber()

        self._run_loop(RuntimeConfig(segment_seconds=1.0, hop_seconds=0.5), transcriber)

        self.assertGreater(transcriber.transcribe_calls, 0)
        self.assertEqual(transcriber.prompt_calls, [])

    def test_enabled_prompt_is_ignored_when_transcriber_has_no_setter(self) -> None:
        class TranscriberWithoutPromptSetter:
            def __init__(self) -> None:
                self.transcribe_calls = 0

            def transcribe(self, chunk, language=None, channel_mode="mono"):
                self.transcribe_calls += 1
                return ""

        transcriber = TranscriberWithoutPromptSetter()

        self._run_loop(
            RuntimeConfig(segment_seconds=1.0, hop_seconds=0.5, whisperx_rolling_prompt_chars=160),
            transcriber,
        )

        self.assertGreater(transcriber.transcribe_calls, 0)
