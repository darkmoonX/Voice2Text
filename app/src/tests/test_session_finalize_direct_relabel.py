from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import time
import unittest

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice2text.capture import AudioChunk
from voice2text.config import RuntimeConfig
import voice2text.controller as controller_mod
from voice2text.controller import TranscriptionController


class _FakeRecordingCapture:
    """Duck-types the subset of RecordingAudioCapture the gating logic reads."""

    def __init__(self, *, wav_path: Path, out_dir: Path, duration_seconds: float) -> None:
        self.wav_path = wav_path
        self.out_dir = out_dir
        self.duration_seconds = duration_seconds

    def stop(self) -> None:
        pass


def _base_config(root: Path, **overrides: object) -> RuntimeConfig:
    kwargs = dict(
        log_dir=str(root / "logs"),
        session_record_enabled=True,
        session_finalize_direct_relabel_enabled=True,
    )
    kwargs.update(overrides)
    return RuntimeConfig(**kwargs)


def _wait_for_call(calls: dict[str, object], key: str, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while key not in calls and time.monotonic() < deadline:
        time.sleep(0.01)


class SessionFinalizeGatingTests(unittest.TestCase):
    """Gating logic only — spies on `_run_session_finalize_relabel_guarded`, no real transcription."""

    def _spy_guarded(self, ctl: TranscriptionController) -> dict[str, object]:
        calls: dict[str, object] = {}

        def fake_guarded(wav_path: Path, out_dir: Path) -> None:
            calls["wav_path"] = wav_path
            calls["out_dir"] = out_dir

        ctl._run_session_finalize_relabel_guarded = fake_guarded  # type: ignore[method-assign]
        return calls

    def test_disabled_flag_is_a_no_op(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-finalize-relabel-") as td:
            root = Path(td)
            cfg = _base_config(root, session_finalize_direct_relabel_enabled=False)
            ctl = TranscriptionController(cfg)
            calls = self._spy_guarded(ctl)
            capture = _FakeRecordingCapture(
                wav_path=root / "session.wav", out_dir=root / "rec", duration_seconds=30.0
            )
            with ctl._capture_lock:
                ctl._capture = capture
            ctl.stop()
            time.sleep(0.05)
            self.assertNotIn("wav_path", calls)

    def test_session_record_disabled_is_a_no_op(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-finalize-relabel-") as td:
            root = Path(td)
            cfg = _base_config(root, session_record_enabled=False)
            ctl = TranscriptionController(cfg)
            calls = self._spy_guarded(ctl)
            capture = _FakeRecordingCapture(
                wav_path=root / "session.wav", out_dir=root / "rec", duration_seconds=30.0
            )
            with ctl._capture_lock:
                ctl._capture = capture
            ctl.stop()
            time.sleep(0.05)
            self.assertNotIn("wav_path", calls)

    def test_non_recording_capture_is_a_no_op(self) -> None:
        """A plain capture (no wav_path/out_dir) never triggers the finalize job."""
        with tempfile.TemporaryDirectory(prefix="v2t-finalize-relabel-") as td:
            root = Path(td)
            cfg = _base_config(root)
            ctl = TranscriptionController(cfg)
            calls = self._spy_guarded(ctl)

            class PlainCapture:
                def stop(self) -> None:
                    pass

            with ctl._capture_lock:
                ctl._capture = PlainCapture()
            ctl.stop()
            time.sleep(0.05)
            self.assertNotIn("wav_path", calls)

    def test_below_duration_floor_is_a_no_op(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-finalize-relabel-") as td:
            root = Path(td)
            cfg = _base_config(root)
            ctl = TranscriptionController(cfg)
            calls = self._spy_guarded(ctl)
            capture = _FakeRecordingCapture(
                wav_path=root / "session.wav", out_dir=root / "rec", duration_seconds=2.0
            )
            with ctl._capture_lock:
                ctl._capture = capture
            ctl.stop()
            time.sleep(0.05)
            self.assertNotIn("wav_path", calls)

    def test_enabled_recording_above_floor_invokes_background_job(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-finalize-relabel-") as td:
            root = Path(td)
            cfg = _base_config(root)
            ctl = TranscriptionController(cfg)
            calls = self._spy_guarded(ctl)
            wav_path = root / "rec" / "session.wav"
            out_dir = root / "rec"
            capture = _FakeRecordingCapture(wav_path=wav_path, out_dir=out_dir, duration_seconds=30.0)
            with ctl._capture_lock:
                ctl._capture = capture
            ctl.stop()
            _wait_for_call(calls, "wav_path")
            self.assertEqual(calls.get("wav_path"), wav_path)
            self.assertEqual(calls.get("out_dir"), out_dir)

    def test_restart_suppresses_finalize(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-finalize-relabel-") as td:
            root = Path(td)
            cfg = _base_config(root)
            ctl = TranscriptionController(cfg)
            calls = self._spy_guarded(ctl)
            ctl._create_transcriber_with_fallback = lambda: None  # type: ignore[method-assign]
            capture = _FakeRecordingCapture(
                wav_path=root / "session.wav", out_dir=root / "rec", duration_seconds=30.0
            )
            with ctl._capture_lock:
                ctl._capture = capture
            # restart() = stop(finalize_session_export=False) + start(); start() will spin up a
            # bootstrap thread that fails fast (no real STT stack here) -- only the suppression of
            # the finalize job on the stop() half is under test.
            ctl.stop(finalize_session_export=False)
            time.sleep(0.05)
            self.assertNotIn("wav_path", calls)

    def test_import_audio_file_direct_suppresses_finalize(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-finalize-relabel-") as td:
            root = Path(td)
            audio_path = root / "voice.wav"
            audio_path.write_bytes(b"not a real wav; stubbed decode/read")
            cfg = _base_config(root)
            ctl = TranscriptionController(cfg)
            calls = self._spy_guarded(ctl)
            capture = _FakeRecordingCapture(
                wav_path=root / "session.wav", out_dir=root / "rec", duration_seconds=30.0
            )
            with ctl._capture_lock:
                ctl._capture = capture

            originals = (
                controller_mod.decode_to_wav_16k_mono,
                controller_mod.read_wav,
                controller_mod.run_direct_transcription,
            )
            try:
                controller_mod.decode_to_wav_16k_mono = lambda path, **kw: path
                controller_mod.read_wav = lambda path: AudioChunk(
                    pcm16=b"\0\0" * 16000, sample_rate=16000, channels=1
                )
                controller_mod.run_direct_transcription = lambda *a, **kw: {"text": "", "meta": {}}
                ctl._create_transcriber_with_fallback = lambda: object()  # type: ignore[method-assign]
                ctl._warmup_transcriber_instance = lambda transcriber: None  # type: ignore[method-assign]
                ctl._shutdown_transcriber_object = lambda transcriber: None  # type: ignore[method-assign]

                ctl.import_audio_file_direct(str(audio_path))
                deadline = time.monotonic() + 5.0
                while ctl.is_running() and time.monotonic() < deadline:
                    time.sleep(0.01)
            finally:
                (
                    controller_mod.decode_to_wav_16k_mono,
                    controller_mod.read_wav,
                    controller_mod.run_direct_transcription,
                ) = originals
            self.assertNotIn("wav_path", calls)


class SessionFinalizeRelabelExecutionTests(unittest.TestCase):
    """Exercises `_run_session_finalize_relabel_guarded` itself with a stubbed transcription core."""

    def test_writes_direct_relabel_export_under_out_dir(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-finalize-relabel-exec-") as td:
            root = Path(td)
            wav_path = root / "rec" / "session.wav"
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            wav_path.write_bytes(b"\0\0" * 16000)  # existence is all that's checked before read_wav
            out_dir = root / "rec"
            cfg = _base_config(root, transcript_export_formats="txt")
            ctl = TranscriptionController(cfg)

            calls: dict[str, object] = {}

            def fake_read(path: Path) -> AudioChunk:
                calls["read_path"] = path
                return AudioChunk(pcm16=b"\0\0" * 16000, sample_rate=16000, channels=1)

            def fake_run(cfg_arg, audio, *, transcriber, chunk_seconds, language_subchunk_seconds, **kwargs):
                calls["transcriber"] = transcriber
                progress = kwargs.get("on_progress")
                if callable(progress):
                    progress(1.0, 1.0)
                return {
                    "text": "[spk_000] hello direct world",
                    "meta": {"elapsed_seconds": 0.0, "token_timestamps": []},
                }

            originals = (controller_mod.read_wav, controller_mod.run_direct_transcription)
            fake_transcriber = object()
            try:
                controller_mod.read_wav = fake_read
                controller_mod.run_direct_transcription = fake_run
                ctl._create_transcriber_with_fallback = lambda: fake_transcriber  # type: ignore[method-assign]
                ctl._warmup_transcriber_instance = lambda transcriber: None  # type: ignore[method-assign]
                ctl._shutdown_transcriber_object = lambda transcriber: None  # type: ignore[method-assign]

                ctl._run_session_finalize_relabel_guarded(wav_path, out_dir)

                self.assertEqual(calls.get("read_path"), wav_path)
                self.assertIs(calls.get("transcriber"), fake_transcriber)
                relabel_dir = out_dir / "direct_relabel"
                written = list(relabel_dir.glob("*.txt"))
                self.assertTrue(written, f"expected a .txt export under {relabel_dir}")
                self.assertIn("hello direct world", written[0].read_text(encoding="utf-8"))
            finally:
                controller_mod.read_wav, controller_mod.run_direct_transcription = originals

    def test_missing_wav_reports_error_and_never_raises(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-finalize-relabel-exec-") as td:
            root = Path(td)
            cfg = _base_config(root)
            ctl = TranscriptionController(cfg)
            errors: list[str] = []
            ctl._emit_error = lambda message: errors.append(message)  # type: ignore[method-assign]

            # Should not raise even though the wav does not exist.
            ctl._run_session_finalize_relabel_guarded(root / "missing.wav", root / "rec")

            self.assertTrue(any("not found" in e for e in errors))

    def test_transcription_failure_is_caught_not_raised(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-finalize-relabel-exec-") as td:
            root = Path(td)
            wav_path = root / "session.wav"
            wav_path.write_bytes(b"\0\0" * 16000)
            cfg = _base_config(root)
            ctl = TranscriptionController(cfg)
            errors: list[str] = []
            ctl._emit_error = lambda message: errors.append(message)  # type: ignore[method-assign]

            originals = (controller_mod.read_wav, controller_mod.run_direct_transcription)
            try:
                controller_mod.read_wav = lambda path: AudioChunk(
                    pcm16=b"\0\0" * 16000, sample_rate=16000, channels=1
                )

                def boom(*a, **kw):
                    raise RuntimeError("synthetic failure")

                controller_mod.run_direct_transcription = boom
                ctl._create_transcriber_with_fallback = lambda: object()  # type: ignore[method-assign]
                ctl._warmup_transcriber_instance = lambda transcriber: None  # type: ignore[method-assign]
                ctl._shutdown_transcriber_object = lambda transcriber: None  # type: ignore[method-assign]

                # Must not raise.
                ctl._run_session_finalize_relabel_guarded(wav_path, root / "rec")
            finally:
                controller_mod.read_wav, controller_mod.run_direct_transcription = originals
            self.assertTrue(any("synthetic failure" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
