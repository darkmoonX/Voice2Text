"""Core orchestration loop: capture audio, preprocess, call WhisperX STT, and emit UI-ready text/status signals."""
from __future__ import annotations
import gc
import logging
from pathlib import Path
import threading
import time
from PySide6.QtCore import QObject, Signal
from .capture import AudioChunk, AudioCaptureBase, build_capture_from_config
from .config import RuntimeConfig
from .cuda_compat import ensure_cublas12_from_source
from .pipeline.gpu_telemetry import GpuTelemetryReporter
from .pipeline.runtime_recovery import RuntimeRecoveryState, WhisperRuntimeRecovery
from .pipeline.direct_transcription import (
    decode_to_wav_16k_mono,
    read_wav,
    run_direct_transcription,
)
from .pipeline.segment_artifacts import SegmentArtifacts
from .pipeline.subtitle_assembler import SubtitleAssembler
from .pipeline.transcript_exporter import TranscriptExportOptions, TranscriptExporterSession
from .pipeline.text_delta_logger import TextDeltaLogger
from .pipeline.transcription_loop import TranscriptionLoopDeps, TranscriptionLoopEngine
from .status_routing import should_surface_overlay_status
from .stt import STTTranscriber, create_stt_transcriber, normalize_stt_provider
from .stt.preprocessing import AudioPreprocessingPipeline, create_audio_preprocessing_pipeline
from .translation import TranslationEngine, build_translation_engine

class TranscriptionController(QObject):
    """Runtime worker that receives audio chunks, invokes STT, and emits UI-facing signals."""
    subtitle_ready = Signal(str, str)
    status_message = Signal(str)
    error_message = Signal(str)
    debug_event = Signal(object)
    runtime_state_changed = Signal(bool)
    _bootstrap_ready = Signal(int, object, object)
    _bootstrap_failed = Signal(int, str)

    def __init__(self, config: RuntimeConfig, logger: logging.Logger | None=None, parent: QObject | None=None) -> None:
        super().__init__(parent)
        self._config = config
        self._logger = logger or logging.getLogger('voice2text')
        self._capture: AudioCaptureBase | None = None
        self._capture_lock = threading.Lock()
        self._transcriber: STTTranscriber | None = None
        self._preprocess_pipeline: AudioPreprocessingPipeline | None = None
        self._translator: TranslationEngine | None = None
        self._bootstrap_thread: threading.Thread | None = None
        self._worker: threading.Thread | None = None
        self._running = threading.Event()
        self._subtitle_assembler = SubtitleAssembler()
        self._text_delta_logger = TextDeltaLogger(lambda prefix, part: self._logger.info('%s: %s', prefix, part), max_entry_chars=180)
        self._runtime_recovery_state = RuntimeRecoveryState()
        self._runtime_epoch = 0
        self._segment_artifacts = SegmentArtifacts(log_dir=self._config.log_dir)
        self._gpu_telemetry = GpuTelemetryReporter(interval_seconds=10.0)
        self._transcript_exporter: TranscriptExporterSession | None = None
        self._temporary_source_restore: dict[str, object] | None = None
        self._bootstrap_ready.connect(self._on_bootstrap_ready)
        self._bootstrap_failed.connect(self._on_bootstrap_failed)

    def start(self) -> None:
        """Start asynchronous STT bootstrap then begin capture/transcription loop."""
        if self._running.is_set():
            return
        if self._bootstrap_thread is not None and self._bootstrap_thread.is_alive():
            return
        if self._worker is not None and self._worker.is_alive():
            return
        self._runtime_epoch += 1
        current_epoch = self._runtime_epoch
        self._running.set()
        # Keep runtime toggle in "paused" state until full bootstrap/warmup/capture init is done.
        self.runtime_state_changed.emit(False)
        self._runtime_recovery_state = RuntimeRecoveryState()
        self._subtitle_assembler.reset()
        self._text_delta_logger.reset()
        self._transcript_exporter = self._build_transcript_exporter()
        self._emit_status('Initializing STT backend...')
        self._bootstrap_thread = threading.Thread(target=lambda: self._bootstrap_stt_stack(current_epoch), daemon=True)
        self._bootstrap_thread.start()

    def stop(self, *, finalize_session_export: bool = True) -> None:
        """Stop capture/worker threads and reset transient runtime state.

        `finalize_session_export=False` suppresses the round-0047 session-finalize direct-relabel
        background job — used by callers where this isn't a genuine session end (a settings
        restart, or switching into file-replay/import mode), so it never fires on every settings
        tweak, only on a real stop.
        """
        self._runtime_epoch += 1
        self._running.clear()
        self.runtime_state_changed.emit(False)
        bootstrap = self._bootstrap_thread
        if bootstrap and bootstrap.is_alive() and (threading.current_thread() is not bootstrap):
            bootstrap.join(timeout=2.0)
        if bootstrap is None or not bootstrap.is_alive():
            self._bootstrap_thread = None
        worker = self._worker
        if worker and worker.is_alive() and (threading.current_thread() is not worker):
            worker.join(timeout=2.0)
        if worker is None or not worker.is_alive():
            self._worker = None
        with self._capture_lock:
            capture_before_stop = self._capture
        self._stop_capture_once()
        self._shutdown_transcriber_once()
        self._preprocess_pipeline = None
        self._translator = None
        self._subtitle_assembler.reset()
        self._text_delta_logger.reset()
        self._finalize_transcript_export()
        if finalize_session_export:
            self._maybe_start_session_finalize_relabel(capture_before_stop)
        self._restore_temporary_source_if_needed()
        self._release_runtime_memory("controller.stop")

    def restart(self) -> None:
        """Convenience API used by settings updates to rebuild runtime stack."""
        self.stop(finalize_session_export=False)
        self.start()

    def is_temporary_file_replay_active(self) -> bool:
        return self._temporary_source_restore is not None

    def temporary_source_restore_values(self) -> dict[str, object] | None:
        restore = self._temporary_source_restore
        return dict(restore) if restore is not None else None

    def import_audio_file(self, file_path: str) -> str:
        """Replay an imported media file through the normal realtime pipeline."""
        source = str(file_path or "").strip()
        if not source:
            raise RuntimeError("Audio import path is empty.")
        path = Path(source).expanduser()
        if not path.exists():
            raise RuntimeError(f"Audio import file does not exist: {path}")
        self.stop(finalize_session_export=False)
        self._temporary_source_restore = {
            "source_mode": self._config.source_mode,
            "source_file_path": getattr(self._config, "source_file_path", ""),
            "source_file_replay_speed": getattr(self._config, "source_file_replay_speed", 0.0),
            "source_file_chunk_seconds": getattr(self._config, "source_file_chunk_seconds", 0.25),
        }
        self._config.source_mode = "file"
        self._config.source_file_path = str(path)
        self._config.source_file_replay_speed = 0.0
        self._config.source_file_chunk_seconds = max(0.02, float(getattr(self._config, "source_file_chunk_seconds", 0.25) or 0.25))
        self._emit_status(f"Import audio replay started: {path}")
        self.start()
        return str(path)

    def import_audio_file_direct(self, file_path: str) -> str:
        """Transcribe an imported media file with one whole-file direct pass."""
        source = str(file_path or "").strip()
        if not source:
            raise RuntimeError("Audio import path is empty.")
        path = Path(source).expanduser()
        if not path.exists():
            raise RuntimeError(f"Audio import file does not exist: {path}")
        self.stop(finalize_session_export=False)
        self._runtime_epoch += 1
        epoch = self._runtime_epoch
        self._running.set()
        self.runtime_state_changed.emit(True)
        self._subtitle_assembler.reset()
        self._text_delta_logger.reset()
        self._transcript_exporter = self._build_transcript_exporter()
        self._emit_status(f"Direct imported-audio transcription started: {path}")
        self._worker = threading.Thread(
            target=lambda: self._run_direct_import_guarded(epoch, path),
            daemon=True,
        )
        self._worker.start()
        return str(path)


    def is_running(self) -> bool:
        return self._running.is_set()
    def _bootstrap_stt_stack(self, epoch: int) -> None:
        try:
            transcriber = self._create_transcriber_with_fallback()
            if transcriber is None or not self._running.is_set():
                self._shutdown_transcriber_object(transcriber)
                self._bootstrap_failed.emit(epoch, '')
                return
            self._warmup_transcriber_instance(transcriber)
            if not self._running.is_set():
                self._shutdown_transcriber_object(transcriber)
                self._bootstrap_failed.emit(epoch, '')
                return
            translator = build_translation_engine(self._config, on_status=self._emit_status)
            if not self._running.is_set():
                return
            self._bootstrap_ready.emit(epoch, transcriber, translator)
        except Exception as exc:
            self._bootstrap_failed.emit(epoch, str(exc))
        finally:
            self._bootstrap_thread = None

    def _on_bootstrap_ready(self, epoch: int, transcriber: object, translator: object) -> None:
        if epoch != self._runtime_epoch or not self._running.is_set():
            return
        self._transcriber = transcriber
        self._preprocess_pipeline = create_audio_preprocessing_pipeline(self._config)
        self._translator = translator if isinstance(translator, TranslationEngine) else None
        if self._preprocess_pipeline.stage_names:
            configured = ', '.join(self._preprocess_pipeline.stage_names)
            active = ', '.join(self._preprocess_pipeline.active_stage_names) or 'none'
            self._emit_status(f'Audio preprocessing active: configured={configured}; active={active}')
        else:
            self._emit_status('Audio preprocessing disabled.')
        if self._translator is not None:
            if self._config.translation_enabled and (not self._translator.state.active):
                self._emit_error(self._translator.state.message)
            else:
                self._emit_status(self._translator.state.message)
        try:
            self._capture = self._build_capture()
            self._capture.start()
        except Exception as exc:
            self._emit_error(f'Audio capture init failed: {exc}')
            self._capture = None
            self._shutdown_transcriber_once()
            self._preprocess_pipeline = None
            self._translator = None
            self._running.clear()
            self.runtime_state_changed.emit(False)
            return
        if not self._running.is_set():
            self._stop_capture_once()
            self._shutdown_transcriber_once()
            self._preprocess_pipeline = None
            self._translator = None
            return
        self._emit_status(f'Capture started @ {self._capture.sample_rate} Hz, {self._capture.channels} ch')
        self.runtime_state_changed.emit(True)
        self._worker = threading.Thread(target=lambda: self._run_loop_guarded(epoch), daemon=True)
        self._worker.start()

    def _on_bootstrap_failed(self, epoch: int, message: str) -> None:
        if epoch != self._runtime_epoch:
            return
        if message.strip():
            self._emit_error(f'STT bootstrap failed: {message}')
        self._running.clear()
        self.runtime_state_changed.emit(False)

    def _run_loop_guarded(self, epoch: int) -> None:
        try:
            self._run_loop()
        except Exception as exc:
            self._emit_error(f'Run loop crashed: {exc}')
        finally:
            self._stop_capture_once()
            self._shutdown_transcriber_once()
            self._preprocess_pipeline = None
            self._translator = None
            if epoch == self._runtime_epoch:
                self._running.clear()
                self.runtime_state_changed.emit(False)
            self._finalize_transcript_export()
            self._restore_temporary_source_if_needed()
            self._worker = None
            self._release_runtime_memory("run-loop-finally")

    def _run_direct_import_guarded(self, epoch: int, path: Path) -> None:
        transcriber: STTTranscriber | None = None
        try:
            transcriber = self._create_transcriber_with_fallback()
            self._transcriber = transcriber
            if transcriber is None or not self._running.is_set() or epoch != self._runtime_epoch:
                return
            self._warmup_transcriber_instance(transcriber)
            if not self._running.is_set() or epoch != self._runtime_epoch:
                return
            self._emit_status(f"Direct import decoding audio: {path}")
            decoded = decode_to_wav_16k_mono(
                path,
                ffmpeg_dir=str(getattr(self._config, "ffmpeg_dll_dir", "") or ""),
            )
            if not self._running.is_set() or epoch != self._runtime_epoch:
                return
            full_audio = read_wav(decoded)

            def _progress(completed: float, total: float) -> None:
                self._emit_status(f"Direct import progress: {completed:.1f}/{total:.1f}s audio")

            result = run_direct_transcription(
                self._config,
                full_audio,
                transcriber=transcriber,
                chunk_seconds=float(getattr(self._config, "import_direct_chunk_seconds", 0.0) or 0.0),
                language_subchunk_seconds=float(
                    getattr(self._config, "import_direct_language_subchunk_seconds", 30.0) or 30.0
                ),
                speaker_profile_reconcile_threshold=float(
                    getattr(self._config, "whisperx_speaker_profile_reconcile_threshold", 0.0) or 0.0
                ),
                whole_file_diarization=bool(
                    getattr(self._config, "import_direct_whole_file_diarization", True)
                ),
                on_progress=_progress,
                on_status=self._emit_status,
            )
            text = str(result.get("text") or "")
            meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
            exporter = self._transcript_exporter
            if exporter is not None and (text or meta.get("token_timestamps")):
                exporter.record(raw_text=text, source_text=text, translated_text="", meta=meta)
            if epoch == self._runtime_epoch:
                self.subtitle_ready.emit(text, "")
                self._emit_status("Direct imported-audio transcription finished.")
        except Exception as exc:
            self._emit_error(f"Direct imported-audio transcription failed: {exc}")
        finally:
            if self._transcriber is transcriber:
                self._transcriber = None
            self._shutdown_transcriber_object(transcriber)
            self._preprocess_pipeline = None
            self._translator = None
            if epoch == self._runtime_epoch:
                self._running.clear()
                self.runtime_state_changed.emit(False)
            self._finalize_transcript_export()
            self._worker = None
            self._release_runtime_memory("direct-import-finally")

    def _stop_capture_once(self) -> None:
        capture: AudioCaptureBase | None
        with self._capture_lock:
            capture = self._capture
            self._capture = None
        if capture is None:
            return
        try:
            capture.stop()
        except Exception:
            return

    def _shutdown_transcriber_once(self) -> None:
        transcriber = self._transcriber
        self._transcriber = None
        self._shutdown_transcriber_object(transcriber)

    def _shutdown_transcriber_object(self, transcriber: object | None) -> None:
        if transcriber is None:
            return
        shutdown = getattr(transcriber, "shutdown", None)
        if not callable(shutdown):
            shutdown = getattr(transcriber, "close", None)
        if not callable(shutdown):
            return
        try:
            shutdown()
        except Exception as exc:
            self._emit_error(f"STT transcriber shutdown failed: {exc}")


    def _build_capture(self) -> AudioCaptureBase:
        capture = build_capture_from_config(self._config, on_error=self._emit_error, on_status=self._emit_status)
        if not bool(getattr(self._config, "session_record_enabled", False)):
            return capture
        if str(getattr(self._config, "source_mode", "")).strip().lower() == "file":
            return capture  # a file replay is already reproducible; don't re-record it
        try:
            from dataclasses import asdict, is_dataclass
            from .capture.session_recorder import RecordingAudioCapture, default_recording_dir

            base = Path(self._config.log_dir).resolve().parent
            snapshot = asdict(self._config) if is_dataclass(self._config) else dict(vars(self._config))
            recorder = RecordingAudioCapture(
                capture,
                out_dir=default_recording_dir(base),
                config_snapshot=snapshot,
                on_status=self._emit_status,
            )
            self._emit_status(f"Session recording -> {recorder.out_dir}")
            return recorder
        except Exception as exc:
            self._emit_error(f"Session recording disabled (setup failed): {exc}")
            return capture

    def _recover_capture_backend(self) -> bool:
        try:
            self._stop_capture_once()
            self._capture = self._build_capture()
            self._capture.start()
            self._emit_status(f'Capture recovered @ {self._capture.sample_rate} Hz, {self._capture.channels} ch')
            return True
        except Exception as exc:
            self._emit_error(f'Capture recovery failed: {exc}')
            return False

    def _run_loop(self) -> None:
        deps = TranscriptionLoopDeps(
            config=self._config,
            subtitle_assembler=self._subtitle_assembler,
            text_delta_logger=self._text_delta_logger,
            segment_artifacts=self._segment_artifacts,
            gpu_telemetry=self._gpu_telemetry,
            get_capture=lambda: self._capture,
            get_transcriber=lambda: self._transcriber,
            get_preprocess_pipeline=lambda: self._preprocess_pipeline,
            get_translator=lambda: self._translator,
            recover_capture_backend=self._recover_capture_backend,
            recover_from_runtime_transcription_error=self._recover_from_runtime_transcription_error,
            emit_status=self._emit_status,
            emit_debug_event=lambda payload: self.debug_event.emit(payload),
            emit_subtitle_ready=lambda source, translated: self.subtitle_ready.emit(source, translated),
            record_transcript_event=self._record_transcript_event,
        )
        TranscriptionLoopEngine(deps).run(self._running)

    def _build_transcript_exporter(self) -> TranscriptExporterSession | None:
        raw_formats = str(getattr(self._config, "transcript_export_formats", "") or "txt")
        formats: list[str] = []
        for token in raw_formats.split(","):
            item = token.strip().lower()
            if item in {"txt", "srt", "json"} and item not in formats:
                formats.append(item)
        if not formats:
            formats = ["txt"]
        output_dir = str(getattr(self._config, "transcript_export_dir", "") or "").strip()
        if not output_dir:
            output_dir = str((Path(self._config.log_dir).resolve().parent / "exports"))
        options = TranscriptExportOptions(
            enabled=True,
            formats=formats,
            include_timestamps=bool(getattr(self._config, "transcript_export_include_timestamps", True)),
            include_speaker=bool(getattr(self._config, "transcript_export_include_speaker", True)),
            output_dir=output_dir,
            display_text_only=bool(getattr(self._config, "transcript_export_display_text_only", False)),
            include_confidence=bool(getattr(self._config, "transcript_export_include_confidence", True)),
            txt_confidence_annotations=bool(getattr(self._config, "transcript_export_txt_confidence_annotations", False)),
        )
        return TranscriptExporterSession(options, on_status=self._emit_status)

    def _record_transcript_event(self, payload: dict[str, object]) -> None:
        exporter = self._transcript_exporter
        if exporter is None:
            return
        exporter.record(
            raw_text=str(payload.get("raw_text") or ""),
            source_text=str(payload.get("source_text") or ""),
            translated_text=str(payload.get("translated_text") or ""),
            meta=payload.get("meta") if isinstance(payload.get("meta"), dict) else {},
        )

    def _finalize_transcript_export(self) -> None:
        exporter = self._transcript_exporter
        if exporter is None:
            return
        if not bool(getattr(self._config, "transcript_export_enabled", False)):
            return
        try:
            exporter.finalize()
        except Exception as exc:
            self._emit_error(f"Transcript export failed: {exc}")

    def _maybe_start_session_finalize_relabel(self, capture: AudioCaptureBase | None) -> None:
        """Round 0047: after a genuine session end, run one whole-file direct-quality
        transcription+diarization pass over the just-recorded session WAV on a background
        thread and write it as an ADDITIONAL export. Never touches the live overlay or the
        incremental export. No-ops unless both `session_record_enabled` and
        `session_finalize_direct_relabel_enabled` are set and the capture was actually a
        session recording of non-trivial duration.
        """
        if not bool(getattr(self._config, "session_finalize_direct_relabel_enabled", False)):
            return
        if not bool(getattr(self._config, "session_record_enabled", False)):
            return
        if capture is None:
            return
        wav_path = getattr(capture, "wav_path", None)
        out_dir = getattr(capture, "out_dir", None)
        if wav_path is None or out_dir is None:
            return  # not a RecordingAudioCapture (e.g. source_mode=file never wraps one)
        try:
            duration_seconds = float(getattr(capture, "duration_seconds", 0.0) or 0.0)
        except Exception:
            duration_seconds = 0.0
        floor_seconds = 5.0
        if duration_seconds < floor_seconds:
            return
        wav_path = Path(wav_path)
        out_dir = Path(out_dir)
        self._emit_status(
            f"Session finalize direct-relabel queued ({duration_seconds:.1f}s recorded) -> "
            f"{out_dir / 'direct_relabel'}"
        )
        thread = threading.Thread(
            target=lambda: self._run_session_finalize_relabel_guarded(wav_path, out_dir),
            daemon=True,
        )
        thread.start()

    def _run_session_finalize_relabel_guarded(self, wav_path: Path, out_dir: Path) -> None:
        transcriber: STTTranscriber | None = None
        try:
            if not wav_path.exists():
                self._emit_error(f"Session finalize direct-relabel: recorded WAV not found: {wav_path}")
                return
            transcriber = self._create_transcriber_with_fallback()
            if transcriber is None:
                self._emit_error("Session finalize direct-relabel: transcriber unavailable.")
                return
            self._warmup_transcriber_instance(transcriber)
            full_audio = read_wav(wav_path)

            def _progress(completed: float, total: float) -> None:
                self._emit_status(
                    f"Session finalize direct-relabel progress: {completed:.1f}/{total:.1f}s audio"
                )

            result = run_direct_transcription(
                self._config,
                full_audio,
                transcriber=transcriber,
                chunk_seconds=float(getattr(self._config, "import_direct_chunk_seconds", 0.0) or 0.0),
                language_subchunk_seconds=float(
                    getattr(self._config, "import_direct_language_subchunk_seconds", 30.0) or 30.0
                ),
                speaker_profile_reconcile_threshold=float(
                    getattr(self._config, "whisperx_speaker_profile_reconcile_threshold", 0.0) or 0.0
                ),
                whole_file_diarization=bool(
                    getattr(self._config, "import_direct_whole_file_diarization", True)
                ),
                on_progress=_progress,
                on_status=self._emit_status,
            )
            text = str(result.get("text") or "")
            meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
            if text or meta.get("token_timestamps"):
                exporter = self._build_session_finalize_relabel_exporter(out_dir)
                exporter.record(raw_text=text, source_text=text, translated_text="", meta=meta)
                exporter.finalize()
            self._emit_status(f"Session finalize direct-relabel finished -> {out_dir / 'direct_relabel'}")
        except Exception as exc:
            self._emit_error(f"Session finalize direct-relabel failed: {exc}")
        finally:
            self._shutdown_transcriber_object(transcriber)

    def _build_session_finalize_relabel_exporter(self, out_dir: Path) -> TranscriptExporterSession:
        raw_formats = str(getattr(self._config, "transcript_export_formats", "") or "txt,srt,json")
        formats: list[str] = []
        for token in raw_formats.split(","):
            item = token.strip().lower()
            if item in {"txt", "srt", "json"} and item not in formats:
                formats.append(item)
        if not formats:
            formats = ["txt", "srt", "json"]
        options = TranscriptExportOptions(
            enabled=True,
            formats=formats,
            include_timestamps=bool(getattr(self._config, "transcript_export_include_timestamps", True)),
            include_speaker=bool(getattr(self._config, "transcript_export_include_speaker", True)),
            output_dir=str(out_dir / "direct_relabel"),
            include_confidence=bool(getattr(self._config, "transcript_export_include_confidence", True)),
            txt_confidence_annotations=bool(getattr(self._config, "transcript_export_txt_confidence_annotations", False)),
        )
        return TranscriptExporterSession(options, on_status=self._emit_status)

    def export_transcript_now(
        self,
        *,
        output_path: str,
        export_format: str,
        include_timestamps: bool | None = None,
        include_speaker: bool | None = None,
    ) -> str:
        exporter = self._transcript_exporter
        if exporter is None:
            raise RuntimeError("Transcript exporter is unavailable; start capture first.")
        written = exporter.export_to(
            output_path=output_path,
            export_format=export_format,
            include_timestamps=include_timestamps,
            include_speaker=include_speaker,
        )
        return str(written)

    def _restore_temporary_source_if_needed(self) -> None:
        restore = self._temporary_source_restore
        if restore is None:
            return
        self._temporary_source_restore = None
        self._config.source_mode = str(restore.get("source_mode") or "loopback")
        self._config.source_file_path = str(restore.get("source_file_path") or "")
        self._config.source_file_replay_speed = float(restore.get("source_file_replay_speed") or 0.0)
        self._config.source_file_chunk_seconds = float(restore.get("source_file_chunk_seconds") or 0.25)
        self._emit_status("Imported audio replay finished. Restored previous capture source.")

    def _release_runtime_memory(self, reason: str) -> None:
        try:
            gc.collect()
        except Exception:
            pass
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
        self._logger.info("Runtime memory cleanup requested: %s", reason)

    def _warmup_transcriber_instance(self, transcriber: STTTranscriber | None) -> None:
        if transcriber is None:
            return
        provider = normalize_stt_provider(self._config.stt_provider)
        if not self._running.is_set():
            return
        warmup_scope = 'VAD/cache pre-init'
        if bool(getattr(self._config, 'whisperx_enable_diarization', False)):
            warmup_scope = 'VAD/cache/diarization pre-init'
        self._emit_status(f'STT warmup started ({provider}; {warmup_scope}).')
        try:
            prewarm_fn = getattr(transcriber, 'prewarm', None)
            if callable(prewarm_fn):
                prewarm_fn(self._config.source_language)
                if provider == "whispercpp":
                    self._emit_status('STT warmup completed.')
                    return
            if not self._running.is_set():
                return
            sample_rate = 16000
            duration_sec = 1.0
            channels = 1
            pcm = b'\x00\x00' * int(sample_rate * duration_sec * channels)
            warmup_chunk = AudioChunk(pcm16=pcm, sample_rate=sample_rate, channels=channels)
            transcriber.transcribe(
                warmup_chunk,
                language=self._config.source_language,
                channel_mode=self._config.source_channel_mode,
            )
            self._emit_status('STT warmup completed.')
        except Exception as exc:
            self._emit_error(f'STT warmup failed: {exc}')

    def _build_stt_transcriber(self, *, device_override: str | None=None, compute_type_override: str | None=None) -> STTTranscriber:
        """Construct provider-specific STT transcriber using current RuntimeConfig."""
        return create_stt_transcriber(self._config, device_override=device_override, compute_type_override=compute_type_override, progress_callback=self._emit_status)

    def _create_transcriber_with_fallback(self) -> STTTranscriber | None:
        model_label = self._effective_model_label()
        provider = normalize_stt_provider(self._config.stt_provider)
        try:
            transcriber = self._build_stt_transcriber()
            self._emit_status(f'STT provider active: {provider} | model={model_label}')
            return transcriber
        except Exception as exc:
            raw_message = str(exc)
            if self._config.cpu_fallback_on_cuda_error and self._config.model_device.lower().startswith('cuda'):
                if 'cublas64_12.dll' in raw_message or 'cannot be loaded' in raw_message:
                    self._emit_error('CUDA runtime missing (cublas64_12.dll).')
                    if self._try_prepare_cuda_compat_alias():
                        try:
                            transcriber = self._build_stt_transcriber()
                            self._emit_status('CUDA compatibility alias active. Keep using CUDA.')
                            return transcriber
                        except Exception as retry_exc:
                            self._emit_error(f'CUDA retry failed: {retry_exc}')
                else:
                    self._emit_error(f'CUDA init failed: {raw_message}')
                try:
                    transcriber = self._build_stt_transcriber(device_override='cpu', compute_type_override='int8')
                except Exception as cpu_exc:
                    self._emit_error(f'CPU fallback init failed: {cpu_exc}')
                    return None
                self._emit_status('WhisperX fallback active: device=cpu, compute_type=int8')
                return transcriber
            self._emit_error(f'WhisperX init failed: {raw_message}')
            return None

    def _effective_model_label(self) -> str:
        if normalize_stt_provider(self._config.stt_provider) == 'whispercpp':
            explicit = str(getattr(self._config, 'stt_whispercpp_model_path', '') or '').strip()
            if explicit:
                return explicit
            if self._config.stt_model_path.strip():
                return self._config.stt_model_path.strip()
            return (self._config.stt_whispercpp_model_size or '').strip() or 'unknown'
        if self._config.stt_model_path.strip():
            return self._config.stt_model_path.strip()
        return (self._config.model_size or '').strip() or 'unknown'

    def _recover_from_runtime_transcription_error(self, raw_message: str) -> bool:
        recovery = WhisperRuntimeRecovery(state=self._runtime_recovery_state, provider_name=normalize_stt_provider(self._config.stt_provider), model_device=self._config.model_device, cpu_fallback_on_cuda_error=self._config.cpu_fallback_on_cuda_error, try_prepare_cuda_compat_alias=self._try_prepare_cuda_compat_alias, rebuild_transcriber_cuda=lambda: self._build_stt_transcriber(), rebuild_transcriber_cpu=lambda: self._build_stt_transcriber(device_override='cpu', compute_type_override='int8'), emit_status=self._emit_status, emit_error=self._emit_error)
        recovered, transcriber = recovery.try_recover(raw_message)
        if recovered and transcriber is not None:
            self._transcriber = transcriber
        return recovered

    def _try_prepare_cuda_compat_alias(self) -> bool:
        return ensure_cublas12_from_source(source_dll=self._config.cuda_compat_source_dll, on_status=lambda msg: self._emit_status(msg))

    def _emit_status(self, message: str) -> None:
        self._logger.info(message)
        if should_surface_overlay_status(message):
            self.status_message.emit(message)

    def _emit_error(self, message: str) -> None:
        self._logger.error(message)
        self.error_message.emit(message)




