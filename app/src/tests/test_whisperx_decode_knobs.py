from __future__ import annotations

import sys
import types
import unittest
import importlib.util
from pathlib import Path
from unittest import mock

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice2text.config import RuntimeConfig
from voice2text.bootstrap_args import build_arg_parser
from voice2text.bootstrap_config import build_runtime_config
from voice2text.settings.schema import allowed_compute_types
from voice2text.audio_capture import AudioChunk
from voice2text.stt import factory
from voice2text.stt.whisperx_provider import WhisperXTranscriber
from voice2text.whisper_config import WhisperRuntimeParams


class WhisperXDecodeKnobTests(unittest.TestCase):
    def test_factory_wires_beam_and_batch_independently(self) -> None:
        cfg = RuntimeConfig(
            model_device="cpu",
            compute_type="int8",
            whisper_beam_size=1,
            whisper_batch_size=8,
        )

        with mock.patch("voice2text.stt.whisperx_provider.WhisperXTranscriber") as transcriber_cls:
            factory._build_whisperx(
                cfg,
                device_override="cpu",
                compute_type_override="int8",
                progress_callback=None,
            )

        kwargs = transcriber_cls.call_args.kwargs
        self.assertEqual(kwargs["beam_size"], 1)
        self.assertEqual(kwargs["batch_size"], 8)

    def test_whisperx_load_model_receives_beam_as_asr_option(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_load_model(*_args, **kwargs):
            calls.append(dict(kwargs))
            return object()

        fake_whisperx = types.SimpleNamespace(load_model=fake_load_model)

        with self._patched_provider_init(fake_whisperx):
            WhisperXTranscriber(
                model_ref="small",
                device="cpu",
                compute_type="int8",
                beam_size=2,
                batch_size=6,
                enable_forced_alignment=False,
                enable_diarization=False,
                auto_download=False,
            )

        self.assertEqual(calls[0]["asr_options"]["beam_size"], 2)
        self.assertEqual(calls[0]["compute_type"], "int8")

    def test_whisperx_load_model_falls_back_when_asr_options_unsupported(self) -> None:
        calls: list[dict[str, object]] = []
        messages: list[str] = []

        def fake_load_model(*_args, **kwargs):
            calls.append(dict(kwargs))
            if "asr_options" in kwargs:
                raise TypeError("load_model() got an unexpected keyword argument 'asr_options'")
            return object()

        fake_whisperx = types.SimpleNamespace(load_model=fake_load_model)

        with self._patched_provider_init(fake_whisperx):
            WhisperXTranscriber(
                model_ref="small",
                device="cpu",
                compute_type="int8",
                beam_size=3,
                enable_forced_alignment=False,
                enable_diarization=False,
                auto_download=False,
                progress_callback=messages.append,
            )

        self.assertIn("asr_options", calls[0])
        self.assertNotIn("asr_options", calls[1])
        self.assertIn("asr_options", "\n".join(messages))

    def test_compute_type_schema_exposes_int8_float16(self) -> None:
        self.assertEqual(allowed_compute_types(), ["float16", "int8_float16", "int8"])

    def test_cli_wires_compute_beam_and_batch_to_runtime_config(self) -> None:
        parser = build_arg_parser(WhisperRuntimeParams())

        args = parser.parse_args(
            [
                "--compute-type",
                "int8_float16",
                "--beam-size",
                "1",
                "--batch-size",
                "8",
            ]
        )
        cfg = build_runtime_config(args)

        self.assertEqual(cfg.compute_type, "int8_float16")
        self.assertEqual(cfg.whisper_beam_size, 1)
        self.assertEqual(cfg.whisper_batch_size, 8)
        self.assertEqual(cfg.whisperx_rolling_prompt_chars, 0)

        cfg_prompt = build_runtime_config(
            parser.parse_args(["--whisperx-rolling-prompt-chars", "160"])
        )
        self.assertEqual(cfg_prompt.whisperx_rolling_prompt_chars, 160)

    def test_rolling_initial_prompt_applies_and_degrades_gracefully(self) -> None:
        from dataclasses import dataclass

        @dataclass
        class _Opts:
            initial_prompt: object = None

        class _Model:
            def __init__(self) -> None:
                self.options = _Opts()

        provider = WhisperXTranscriber.__new__(WhisperXTranscriber)
        provider._rolling_initial_prompt = ""
        provider._initial_prompt_compat_ok = True
        provider._model = _Model()

        provider.set_initial_prompt("context tail")
        provider._apply_initial_prompt_to_model()
        self.assertEqual(provider._model.options.initial_prompt, "context tail")

        provider.set_initial_prompt("")
        provider._apply_initial_prompt_to_model()
        self.assertIsNone(provider._model.options.initial_prompt)

        # A build whose options lack initial_prompt disables the feature once.
        class _OptsNoPrompt:
            pass

        provider._model.options = _OptsNoPrompt()
        provider.set_initial_prompt("x")
        provider._apply_initial_prompt_to_model()
        self.assertFalse(provider._initial_prompt_compat_ok)

    def test_window_boundary_trace_emits_when_trace_enabled(self) -> None:
        messages: list[str] = []
        provider = self._transcribe_stub(trace_enabled=True, messages=messages)
        chunk = AudioChunk(pcm16=b"\x00\x00" * 1600, sample_rate=16000, channels=1)

        with mock.patch(
            "voice2text.stt.whisperx_provider.time.perf_counter",
            side_effect=[0.0, 0.01, 0.02, 0.03, 0.04, 0.10, 0.523, 0.60, 0.61, 0.70, 0.80, 0.90],
        ):
            provider.transcribe(chunk, language="en")

        boundary_lines = [line for line in messages if line.startswith("[window-boundary]")]
        self.assertEqual(
            boundary_lines,
            [
                "[window-boundary] trace=1; window_s=0.10; "
                "first_seg_start=0.04; last_seg_end=0.09; asr_s=0.423"
            ],
        )

    def test_window_boundary_trace_is_silent_when_trace_disabled(self) -> None:
        messages: list[str] = []
        provider = self._transcribe_stub(trace_enabled=False, messages=messages)
        chunk = AudioChunk(pcm16=b"\x00\x00" * 1600, sample_rate=16000, channels=1)

        provider.transcribe(chunk, language="en")

        self.assertFalse(any(line.startswith("[window-boundary]") for line in messages))

    def test_compare_summary_records_decode_knobs(self) -> None:
        compare = self._load_compare_script()
        cfg = RuntimeConfig(
            compute_type="int8_float16",
            whisper_beam_size=1,
            whisper_batch_size=8,
        )

        self.assertEqual(
            compare._decode_knob_summary(cfg),
            {
                "compute_type": "int8_float16",
                "beam_size": 1,
                "batch_size": 8,
            },
        )

    def _load_compare_script(self):
        script_path = SRC_ROOT.parent / "scripts" / "diagnostics" / "compare_test_data_whisperx.py"
        spec = importlib.util.spec_from_file_location("compare_test_data_whisperx_for_test", script_path)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def _transcribe_stub(self, *, trace_enabled: bool, messages: list[str]) -> WhisperXTranscriber:
        class _Model:
            def transcribe(self, _audio, **_kwargs):
                return {
                    "language": "en",
                    "segments": [
                        {
                            "start": 0.04,
                            "end": 0.09,
                            "text": "hello",
                            "words": [{"start": 0.04, "end": 0.09, "score": 0.9, "word": "hello"}],
                        }
                    ],
                    "text": "hello",
                }

        provider = WhisperXTranscriber.__new__(WhisperXTranscriber)
        provider._trace_enabled = trace_enabled
        provider._trace_counter = 0
        provider._enable_forced_alignment = False
        provider._enable_diarization = False
        provider._diarization_suppressed = False
        provider._speaker_profile_enabled = False
        provider._enable_phoneme_asr = False
        provider._batch_size = 4
        provider._model = _Model()
        provider._rolling_initial_prompt = ""
        provider._initial_prompt_compat_ok = True
        provider._alignment_language = "auto"
        provider._source_language_hint = None
        provider._language_route_logged = set()
        provider._alignment_device = "cpu"
        provider._last_alignment_timing = {}
        provider._last_diarization_timing = {}
        provider._last_speaker_profile_stats = {}
        provider._emit = messages.append
        return provider

    def _patched_provider_init(self, fake_whisperx):
        patches = [
            mock.patch.dict(sys.modules, {"whisperx": fake_whisperx}),
            mock.patch.object(WhisperXTranscriber, "_configure_hf_cache_env", lambda *_args, **_kwargs: None),
            mock.patch.object(WhisperXTranscriber, "_normalize_alignment_layout", lambda *_args, **_kwargs: None),
            mock.patch.object(WhisperXTranscriber, "_cleanup_alignment_partial_cache", lambda *_args, **_kwargs: None),
            mock.patch.object(WhisperXTranscriber, "_build_download_probe_roots", lambda *_args, **_kwargs: []),
            mock.patch.object(WhisperXTranscriber, "_resolve_alignment_device", lambda *_args, **_kwargs: "cpu"),
            mock.patch.object(WhisperXTranscriber, "_resolve_diarization_device", lambda *_args, **_kwargs: "cpu"),
        ]
        return _PatchStack(patches)


class _PatchStack:
    def __init__(self, patches) -> None:
        self._patches = list(patches)

    def __enter__(self):
        for patch in self._patches:
            patch.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        for patch in reversed(self._patches):
            patch.__exit__(exc_type, exc, tb)
        return False


if __name__ == "__main__":
    unittest.main()
