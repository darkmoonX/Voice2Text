"""Core orchestration loop: capture audio, preprocess/VAD, call STT, and emit UI-ready text/status signals."""
from __future__ import annotations
import logging
import threading
import time
from PySide6.QtCore import QObject, Signal
from .capture import AudioChunk, AudioCaptureBase, build_capture_from_config
from .config import RuntimeConfig
from .cuda_compat import ensure_cublas12_from_source
from .pipeline.gpu_telemetry import GpuTelemetryReporter
from .pipeline.runtime_recovery import RuntimeRecoveryState, WhisperRuntimeRecovery
from .pipeline.segment_artifacts import SegmentArtifacts
from .pipeline.subtitle_assembler import SubtitleAssembler
from .pipeline.text_delta_logger import TextDeltaLogger
from .pipeline.transcription_loop import TranscriptionLoopDeps, TranscriptionLoopEngine
from .status_routing import should_surface_overlay_status
from .stt import STTTranscriber, create_stt_transcriber, normalize_stt_provider
from .stt.preprocessing import AudioPreprocessingPipeline, create_audio_preprocessing_pipeline
from .stt.vad import VADPipeline, create_vad_pipeline
from .translator import ArgosTranslator

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
        self._vad_pipeline: VADPipeline | None = None
        self._translator: ArgosTranslator | None = None
        self._bootstrap_thread: threading.Thread | None = None
        self._worker: threading.Thread | None = None
        self._running = threading.Event()
        self._subtitle_assembler = SubtitleAssembler()
        self._text_delta_logger = TextDeltaLogger(lambda prefix, part: self._logger.info('%s: %s', prefix, part), max_entry_chars=180)
        self._runtime_recovery_state = RuntimeRecoveryState()
        self._runtime_epoch = 0
        self._segment_artifacts = SegmentArtifacts(log_dir=self._config.log_dir)
        self._gpu_telemetry = GpuTelemetryReporter(interval_seconds=10.0)
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
        self._emit_status('Initializing STT backend...')
        self._bootstrap_thread = threading.Thread(target=lambda: self._bootstrap_stt_stack(current_epoch), daemon=True)
        self._bootstrap_thread.start()

    def stop(self) -> None:
        """Stop capture/worker threads and reset transient runtime state."""
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
        self._stop_capture_once()
        self._transcriber = None
        self._preprocess_pipeline = None
        self._vad_pipeline = None
        self._translator = None
        self._subtitle_assembler.reset()
        self._text_delta_logger.reset()

    def restart(self) -> None:
        """Convenience API used by settings updates to rebuild runtime stack."""
        self.stop()
        self.start()


    def is_running(self) -> bool:
        return self._running.is_set()
    def _bootstrap_stt_stack(self, epoch: int) -> None:
        try:
            transcriber = self._create_transcriber_with_fallback()
            if transcriber is None or not self._running.is_set():
                self._bootstrap_failed.emit(epoch, '')
                return
            translator = ArgosTranslator(enabled=self._config.translation_enabled, source_code=self._config.translation_from, target_code=self._config.translation_to)
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
        provider = normalize_stt_provider(self._config.stt_provider)
        self._preprocess_pipeline = create_audio_preprocessing_pipeline(self._config)
        self._vad_pipeline = create_vad_pipeline(self._config, provider)
        self._translator = translator if isinstance(translator, ArgosTranslator) else None
        if self._preprocess_pipeline.stage_names:
            configured = ', '.join(self._preprocess_pipeline.stage_names)
            active = ', '.join(self._preprocess_pipeline.active_stage_names) or 'none'
            self._emit_status(f'Audio preprocessing active: configured={configured}; active={active}')
        else:
            self._emit_status('Audio preprocessing disabled.')
        if self._vad_pipeline.stage_names:
            self._emit_status('VAD pipeline active: ' + ', '.join(self._vad_pipeline.stage_names))
        else:
            self._emit_status('VAD pipeline disabled.')
        if self._translator is not None:
            if self._config.translation_enabled and (not self._translator.state.active):
                self._emit_error(self._translator.state.message)
            else:
                self._emit_status(self._translator.state.message)
        self._warmup_transcriber()
        try:
            self._capture = build_capture_from_config(self._config, on_error=self._emit_error, on_status=self._emit_status)
            self._capture.start()
        except Exception as exc:
            self._emit_error(f'Audio capture init failed: {exc}')
            self._capture = None
            self._transcriber = None
            self._preprocess_pipeline = None
            self._vad_pipeline = None
            self._translator = None
            self._running.clear()
            self.runtime_state_changed.emit(False)
            return
        if not self._running.is_set():
            self._stop_capture_once()
            self._transcriber = None
            self._preprocess_pipeline = None
            self._vad_pipeline = None
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
            self._transcriber = None
            self._preprocess_pipeline = None
            self._vad_pipeline = None
            self._translator = None
            if epoch == self._runtime_epoch:
                self._running.clear()
                self.runtime_state_changed.emit(False)
            self._worker = None

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


    def _recover_capture_backend(self) -> bool:
        try:
            self._stop_capture_once()
            self._capture = build_capture_from_config(self._config, on_error=self._emit_error, on_status=self._emit_status)
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
            get_vad_pipeline=lambda: self._vad_pipeline,
            get_translator=lambda: self._translator,
            recover_capture_backend=self._recover_capture_backend,
            recover_from_runtime_transcription_error=self._recover_from_runtime_transcription_error,
            emit_status=self._emit_status,
            emit_debug_event=lambda payload: self.debug_event.emit(payload),
            emit_subtitle_ready=lambda source, translated: self.subtitle_ready.emit(source, translated),
        )
        TranscriptionLoopEngine(deps).run(self._running)

    def _warmup_transcriber(self) -> None:
        transcriber = self._transcriber
        if transcriber is None:
            return
        provider = normalize_stt_provider(self._config.stt_provider)
        if provider != 'whisperx':
            return
        self._emit_status('WhisperX warmup started (VAD/cache pre-init).')
        try:
            prewarm_fn = getattr(transcriber, 'prewarm', None)
            if callable(prewarm_fn):
                prewarm_fn(self._config.source_language)
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
            self._emit_status('WhisperX warmup completed.')
        except Exception as exc:
            self._emit_error(f'WhisperX warmup failed: {exc}')

    def _build_stt_transcriber(self, *, device_override: str | None=None, compute_type_override: str | None=None) -> STTTranscriber:
        """Construct provider-specific STT transcriber using current RuntimeConfig."""
        return create_stt_transcriber(self._config, device_override=device_override, compute_type_override=compute_type_override, progress_callback=self._emit_status)

    def _create_transcriber_with_fallback(self) -> STTTranscriber | None:
        model_label = self._effective_model_label()
        provider = normalize_stt_provider(self._config.stt_provider)
        if provider != 'whisper':
            try:
                transcriber = self._build_stt_transcriber()
                self._emit_status(f'STT provider active: {provider} | model={model_label}')
                return transcriber
            except Exception as exc:
                self._emit_error(f'{provider} init failed: {exc}')
                return None
        try:
            transcriber = self._build_stt_transcriber()
            self._emit_status(f'STT provider active: whisper | model={model_label}')
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
                self._emit_status('Whisper fallback active: device=cpu, compute_type=int8')
                return transcriber
            self._emit_error(f'Whisper init failed: {raw_message}')
            return None

    def _effective_model_label(self) -> str:
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








