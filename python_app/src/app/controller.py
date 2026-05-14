"""Core orchestration loop: capture audio, preprocess/VAD, call STT, and emit UI-ready text/status signals."""
from __future__ import annotations
import logging
import threading
from PySide6.QtCore import QObject, Signal
from .capture import AudioChunk, AudioCaptureBase, build_capture_from_config
from .config import RuntimeConfig
from .cuda_compat import ensure_cublas12_from_source
from .pipeline.runtime_recovery import RuntimeRecoveryState, WhisperRuntimeRecovery
from .pipeline.subtitle_assembler import SubtitleAssembler
from .pipeline.text_delta_logger import TextDeltaLogger
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
    _bootstrap_ready = Signal(object, object)
    _bootstrap_failed = Signal(str)

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
        self._silence_hops = 0
        self._speech_hops = 0
        self._runtime_recovery_state = RuntimeRecoveryState()
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
        self._running.set()
        self._runtime_recovery_state = RuntimeRecoveryState()
        self._silence_hops = 0
        self._speech_hops = 0
        self._subtitle_assembler.reset()
        self._text_delta_logger.reset()
        self._emit_status('Initializing STT backend...')
        self._bootstrap_thread = threading.Thread(target=self._bootstrap_stt_stack, daemon=True)
        self._bootstrap_thread.start()

    def stop(self) -> None:
        """Stop capture/worker threads and reset transient runtime state."""
        self._running.clear()
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
        self._silence_hops = 0
        self._speech_hops = 0

    def restart(self) -> None:
        """Convenience API used by settings updates to rebuild runtime stack."""
        self.stop()
        self.start()

    def _bootstrap_stt_stack(self) -> None:
        try:
            transcriber = self._create_transcriber_with_fallback()
            if transcriber is None or not self._running.is_set():
                self._bootstrap_failed.emit('')
                return
            translator = ArgosTranslator(enabled=self._config.translation_enabled, source_code=self._config.translation_from, target_code=self._config.translation_to)
            if not self._running.is_set():
                return
            self._bootstrap_ready.emit(transcriber, translator)
        except Exception as exc:
            self._bootstrap_failed.emit(str(exc))
        finally:
            self._bootstrap_thread = None

    def _on_bootstrap_ready(self, transcriber: object, translator: object) -> None:
        if not self._running.is_set():
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
            return
        if not self._running.is_set():
            self._stop_capture_once()
            self._transcriber = None
            self._preprocess_pipeline = None
            self._vad_pipeline = None
            self._translator = None
            return
        self._emit_status(f'Capture started @ {self._capture.sample_rate} Hz, {self._capture.channels} ch')
        self._worker = threading.Thread(target=self._run_loop_guarded, daemon=True)
        self._worker.start()

    def _on_bootstrap_failed(self, message: str) -> None:
        if message.strip():
            self._emit_error(f'STT bootstrap failed: {message}')
        self._running.clear()

    def _run_loop_guarded(self) -> None:
        try:
            self._run_loop()
        finally:
            self._stop_capture_once()
            self._transcriber = None
            self._preprocess_pipeline = None
            self._vad_pipeline = None
            self._translator = None
            self._running.clear()
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

    def _run_loop(self) -> None:
        assert self._capture is not None
        assert self._transcriber is not None
        buffer = bytearray()
        stream_rate = self._capture.sample_rate
        stream_channels = self._capture.channels
        segment_seconds = min(max(1.0, float(self._config.segment_seconds)), 12.0)
        hop_seconds = min(max(0.1, float(self._config.hop_seconds)), max(0.1, segment_seconds - 0.1))

        def aligned_window_sizes(rate: int, channels: int) -> tuple[int, int, int, int]:
            bytes_per_second_local = max(1, rate * channels * 2)
            frame_bytes_local = max(2, channels * 2)
            segment_bytes_local = max(int(bytes_per_second_local * segment_seconds), bytes_per_second_local // 2)
            hop_bytes_local = max(1, int(bytes_per_second_local * hop_seconds))
            hop_bytes_local = min(hop_bytes_local, segment_bytes_local)
            segment_bytes_local = max(frame_bytes_local, segment_bytes_local // frame_bytes_local * frame_bytes_local)
            hop_bytes_local = max(frame_bytes_local, hop_bytes_local // frame_bytes_local * frame_bytes_local)
            hop_bytes_local = min(hop_bytes_local, segment_bytes_local)
            return (bytes_per_second_local, frame_bytes_local, segment_bytes_local, hop_bytes_local)
        (bytes_per_second, frame_bytes, segment_bytes, hop_bytes) = aligned_window_sizes(stream_rate, stream_channels)
        while self._running.is_set():
            chunk = self._capture.read_chunk(timeout=0.25)
            if chunk is None:
                continue
            if chunk.sample_rate != stream_rate or chunk.channels != stream_channels or segment_bytes <= 0:
                stream_rate = chunk.sample_rate
                stream_channels = chunk.channels
                (bytes_per_second, frame_bytes, segment_bytes, hop_bytes) = aligned_window_sizes(stream_rate, stream_channels)
                buffer.clear()
                self._emit_status(f'Stream format changed: {stream_rate} Hz, {stream_channels} ch')
            chunk_pcm = chunk.pcm16
            if len(chunk_pcm) < 2:
                continue
            if len(chunk_pcm) % 2 != 0:
                chunk_pcm = chunk_pcm[:-1]
            if not chunk_pcm:
                continue
            if chunk_pcm is not chunk.pcm16:
                chunk = AudioChunk(pcm16=chunk_pcm, sample_rate=chunk.sample_rate, channels=chunk.channels)
            buffer.extend(chunk.pcm16)
            max_buffer_bytes = max(segment_bytes * 6, bytes_per_second)
            max_buffer_bytes = max(frame_bytes, max_buffer_bytes // frame_bytes * frame_bytes)
            if len(buffer) > max_buffer_bytes:
                del buffer[:len(buffer) - max_buffer_bytes]
            while len(buffer) >= segment_bytes and self._running.is_set():
                window = bytes(buffer[:segment_bytes])
                del buffer[:hop_bytes]
                window_chunk = AudioChunk(pcm16=window, sample_rate=stream_rate, channels=stream_channels)
                stt_chunk = self._preprocess_pipeline.process(window_chunk, channel_mode=self._config.source_channel_mode) if self._preprocess_pipeline is not None and self._preprocess_pipeline.stage_names else window_chunk
                has_signal = self._vad_pipeline.should_process(stt_chunk, channel_mode=self._config.source_channel_mode) if self._vad_pipeline is not None else self._transcriber.has_enough_signal(stt_chunk, channel_mode=self._config.source_channel_mode)
                if not has_signal:
                    self._silence_hops += 1
                    silence_seconds = self._silence_hops * hop_seconds
                    if self._speech_hops > 0 and silence_seconds >= max(0.8, min(2.4, segment_seconds)):
                        self._mark_sentence_break()
                    continue
                self._silence_hops = 0
                try:
                    source_text = self._transcriber.transcribe(stt_chunk, language=self._config.source_language, channel_mode=self._config.source_channel_mode)
                except Exception as exc:
                    if self._recover_from_runtime_transcription_error(str(exc)):
                        continue
                    self._emit_error(f'Transcription failed: {exc}')
                    continue
                source_rolling = self._subtitle_assembler.merge_incremental_text(source_text, overlap_merge_method=self._config.overlap_merge_method, segment_seconds=float(self._config.segment_seconds), hop_seconds=float(self._config.hop_seconds))
                if not source_rolling:
                    continue
                self._speech_hops += 1
                (source_out, translated_out) = self._build_subtitle_payload(source_rolling)
                if not source_out and (not translated_out):
                    continue
                if source_out:
                    self._text_delta_logger.log('STT', source_out, translated=False)
                if translated_out:
                    self._text_delta_logger.log('TRANSLATE', translated_out, translated=True)
                self.subtitle_ready.emit(source_out, translated_out)
                if self._speech_hops * hop_seconds >= max(segment_seconds, hop_seconds * 2.0):
                    self._mark_sentence_break()

    def _build_subtitle_payload(self, source_text: str) -> tuple[str, str]:
        if self._translator is None or not self._translator.enabled:
            return (source_text, '')
        translated = self._translator.translate(source_text)
        if not translated:
            return (source_text, '')
        return (source_text, translated)

    def _mark_sentence_break(self) -> None:
        self._subtitle_assembler.mark_sentence_break()
        self._silence_hops = 0
        self._speech_hops = 0

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

