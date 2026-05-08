from __future__ import annotations

from difflib import SequenceMatcher
import logging
import re
import threading

from PySide6.QtCore import QObject, Signal

from .audio_capture import AudioChunk, AudioCaptureBase, build_capture_from_config
from .config import RuntimeConfig
from .cuda_compat import ensure_cublas12_from_source
from .transcriber import FasterWhisperTranscriber
from .translator import ArgosTranslator


class TranscriptionController(QObject):
    subtitle_ready = Signal(str, str)
    status_message = Signal(str)
    error_message = Signal(str)

    def __init__(
        self,
        config: RuntimeConfig,
        logger: logging.Logger | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._logger = logger or logging.getLogger("voice2text")

        self._capture: AudioCaptureBase | None = None
        self._transcriber: FasterWhisperTranscriber | None = None
        self._translator: ArgosTranslator | None = None

        self._worker: threading.Thread | None = None
        self._running = threading.Event()
        self._frozen_source_text = ""
        self._active_source_text = ""
        self._last_emitted_source_text = ""
        self._runtime_recovery_attempted = False
        self._cpu_runtime_fallback_done = False

    def start(self) -> None:
        if self._running.is_set():
            return

        self._transcriber = self._create_transcriber_with_fallback()
        if self._transcriber is None:
            return

        self._translator = ArgosTranslator(
            enabled=self._config.translation_enabled,
            source_code=self._config.translation_from,
            target_code=self._config.translation_to,
        )
        if self._config.translation_enabled and not self._translator.state.active:
            self._emit_error(self._translator.state.message)
        else:
            self._emit_status(self._translator.state.message)

        try:
            self._capture = build_capture_from_config(
                self._config,
                on_error=self._emit_error,
                on_status=self._emit_status,
            )
            self._capture.start()
        except Exception as exc:
            self._emit_error(f"Audio capture init failed: {exc}")
            if self._capture is not None:
                self._capture.stop()
            self._capture = None
            return

        self._running.set()
        self._runtime_recovery_attempted = False
        self._cpu_runtime_fallback_done = False
        self._worker = threading.Thread(target=self._run_loop, daemon=True)
        self._worker.start()

        self._emit_status(
            f"Capture started @ {self._capture.sample_rate} Hz, {self._capture.channels} ch"
        )

    def stop(self) -> None:
        self._running.clear()

        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=1.0)
        self._worker = None

        if self._capture is not None:
            self._capture.stop()
            self._capture = None

        self._frozen_source_text = ""
        self._active_source_text = ""
        self._last_emitted_source_text = ""

    def restart(self) -> None:
        self.stop()
        self.start()

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

            segment_bytes_local = max(
                int(bytes_per_second_local * segment_seconds),
                bytes_per_second_local // 2,
            )
            hop_bytes_local = max(1, int(bytes_per_second_local * hop_seconds))
            hop_bytes_local = min(hop_bytes_local, segment_bytes_local)

            # Keep slicing aligned with whole frames and int16 boundaries.
            segment_bytes_local = max(
                frame_bytes_local,
                (segment_bytes_local // frame_bytes_local) * frame_bytes_local,
            )
            hop_bytes_local = max(
                frame_bytes_local,
                (hop_bytes_local // frame_bytes_local) * frame_bytes_local,
            )
            hop_bytes_local = min(hop_bytes_local, segment_bytes_local)

            return bytes_per_second_local, frame_bytes_local, segment_bytes_local, hop_bytes_local

        bytes_per_second, frame_bytes, segment_bytes, hop_bytes = aligned_window_sizes(
            stream_rate,
            stream_channels,
        )

        while self._running.is_set():
            chunk = self._capture.read_chunk(timeout=0.25)
            if chunk is None:
                continue

            if (
                chunk.sample_rate != stream_rate
                or chunk.channels != stream_channels
                or segment_bytes <= 0
            ):
                stream_rate = chunk.sample_rate
                stream_channels = chunk.channels
                (
                    bytes_per_second,
                    frame_bytes,
                    segment_bytes,
                    hop_bytes,
                ) = aligned_window_sizes(stream_rate, stream_channels)
                buffer.clear()
                self._emit_status(
                    f"Stream format changed: {stream_rate} Hz, {stream_channels} ch"
                )

            chunk_pcm = chunk.pcm16
            if len(chunk_pcm) < 2:
                continue

            if len(chunk_pcm) % 2 != 0:
                chunk_pcm = chunk_pcm[:-1]
            if not chunk_pcm:
                continue

            if chunk_pcm is not chunk.pcm16:
                chunk = AudioChunk(
                    pcm16=chunk_pcm,
                    sample_rate=chunk.sample_rate,
                    channels=chunk.channels,
                )

            buffer.extend(chunk.pcm16)
            max_buffer_bytes = max(segment_bytes * 6, bytes_per_second)
            max_buffer_bytes = max(
                frame_bytes,
                (max_buffer_bytes // frame_bytes) * frame_bytes,
            )
            if len(buffer) > max_buffer_bytes:
                del buffer[: len(buffer) - max_buffer_bytes]

            while len(buffer) >= segment_bytes and self._running.is_set():
                window = bytes(buffer[:segment_bytes])
                del buffer[:hop_bytes]

                window_chunk = AudioChunk(
                    pcm16=window,
                    sample_rate=stream_rate,
                    channels=stream_channels,
                )

                if not self._transcriber.has_enough_signal(
                    window_chunk,
                    channel_mode=self._config.source_channel_mode,
                ):
                    continue

                try:
                    source_text = self._transcriber.transcribe(
                        window_chunk,
                        language=self._config.source_language,
                        channel_mode=self._config.source_channel_mode,
                    )
                except Exception as exc:
                    if self._recover_from_runtime_transcription_error(str(exc)):
                        continue
                    self._emit_error(f"Transcription failed: {exc}")
                    continue

                source_rolling = self._merge_incremental_text(source_text)
                if not source_rolling:
                    continue

                source_out, translated_out = self._build_subtitle_payload(source_rolling)
                if not source_out and not translated_out:
                    continue

                if source_out:
                    self._logger.info("STT: %s", source_out)
                if translated_out:
                    self._logger.info("TRANSLATE: %s", translated_out)

                self.subtitle_ready.emit(source_out, translated_out)

    def _build_subtitle_payload(self, source_text: str) -> tuple[str, str]:
        if self._translator is None or not self._translator.enabled:
            return source_text, ""

        translated = self._translator.translate(source_text)
        if not translated:
            return source_text, ""

        return source_text, translated

    def _merge_incremental_text(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return ""

        if not self._active_source_text:
            self._active_source_text = cleaned
            combined = self._compose_rolling_text(self._frozen_source_text, self._active_source_text)
            self._last_emitted_source_text = combined
            return combined

        method = (self._config.overlap_merge_method or "replace-window").strip().lower()

        if method == "append-only":
            combined_prev = self._compose_rolling_text(
                self._frozen_source_text,
                self._active_source_text,
            )
            combined = self._merge_by_exact_overlap(combined_prev, cleaned)
            combined = combined[-1800:]

            if combined == self._last_emitted_source_text:
                return ""

            self._frozen_source_text = ""
            self._active_source_text = combined
            self._last_emitted_source_text = combined
            return combined

        lock_ratio = self._segment_lock_ratio()

        lock_chars = int(round(len(self._active_source_text) * lock_ratio))
        lock_chars = max(0, min(lock_chars, len(self._active_source_text)))

        lock_chunk = self._active_source_text[:lock_chars].strip()
        overlap_tail = self._active_source_text[lock_chars:].strip()

        if lock_chunk:
            self._frozen_source_text = self._merge_by_exact_overlap(
                self._frozen_source_text,
                lock_chunk,
            )

        if method == "suffix-overlap":
            self._active_source_text = self._merge_by_exact_overlap(overlap_tail, cleaned)
        elif method == "fuzzy-overlap":
            self._active_source_text = self._merge_by_fuzzy_overlap(overlap_tail, cleaned)
        else:
            self._active_source_text = self._merge_replace_window(overlap_tail, cleaned, lock_ratio)

        combined = self._compose_rolling_text(self._frozen_source_text, self._active_source_text)
        combined = combined[-1800:]

        if combined == self._last_emitted_source_text:
            return ""

        self._last_emitted_source_text = combined
        return combined

    def _segment_lock_ratio(self) -> float:
        segment = max(0.1, float(self._config.segment_seconds))
        hop = max(0.01, float(self._config.hop_seconds))
        ratio = hop / segment
        return max(0.05, min(0.95, ratio))

    def _merge_replace_window(self, overlap_tail: str, incoming: str, lock_ratio: float) -> str:
        previous = re.sub(r"\s+", " ", overlap_tail).strip()
        latest = re.sub(r"\s+", " ", incoming).strip()

        if not previous:
            return latest
        if not latest:
            return previous

        preserve_ratio = max(0.22, min(0.55, lock_ratio * 2.0))
        keep_chars = int(round(len(previous) * preserve_ratio))
        keep_chars = max(10, min(keep_chars, len(previous)))

        stable_head = previous[:keep_chars].strip()
        mutable_tail = previous[keep_chars:].strip()

        reconciled_tail = self._merge_by_fuzzy_overlap(mutable_tail, latest)
        if reconciled_tail == latest and mutable_tail:
            # When overlap is weak, avoid trusting the unstable leading words of latest window.
            skip_chars = max(4, int(round(len(latest) * 0.18)))
            conservative = latest[skip_chars:].strip()
            if conservative:
                reconciled_tail = self._merge_by_exact_overlap(mutable_tail, conservative)
            if not reconciled_tail:
                reconciled_tail = mutable_tail

        merged = self._merge_by_exact_overlap(stable_head, reconciled_tail)
        return merged or previous

    def _compose_rolling_text(self, frozen: str, active: str) -> str:
        return self._merge_by_exact_overlap(frozen, active)

    def _merge_by_exact_overlap(self, base: str, incoming: str) -> str:
        base = re.sub(r"\s+", " ", base).strip()
        incoming = re.sub(r"\s+", " ", incoming).strip()

        if not base:
            return incoming
        if not incoming:
            return base

        overlap = self._max_prefix_suffix_overlap(base, incoming)
        if overlap >= len(incoming):
            return base

        if overlap > 0:
            return f"{base}{incoming[overlap:]}".strip()

        if incoming in base[-max(16, len(incoming) * 2) :]:
            return base

        sep = "" if base.endswith(("。", "！", "？", "，", ",", ".", " ")) else " "
        return f"{base}{sep}{incoming}".strip()

    def _merge_by_fuzzy_overlap(self, base: str, incoming: str) -> str:
        base = re.sub(r"\s+", " ", base).strip()
        incoming = re.sub(r"\s+", " ", incoming).strip()

        if not base:
            return incoming
        if not incoming:
            return base

        exact = self._max_prefix_suffix_overlap(base, incoming)
        if exact > 0:
            return self._merge_by_exact_overlap(base, incoming)

        max_len = min(len(base), len(incoming), 120)
        min_len = min(6, max_len)
        best_size = 0
        best_score = 0.0

        for size in range(max_len, min_len - 1, -1):
            tail = base[-size:]
            head = incoming[:size]
            score = SequenceMatcher(None, tail, head).ratio()
            if score > best_score:
                best_score = score
                best_size = size
            if score >= 0.76:
                return f"{base}{incoming[size:]}".strip()

        if best_size >= 8 and best_score >= 0.62:
            return f"{base}{incoming[best_size:]}".strip()

        return incoming

    @staticmethod
    def _max_prefix_suffix_overlap(base: str, incoming: str) -> int:
        max_len = min(len(base), len(incoming))
        for size in range(max_len, 0, -1):
            if base.endswith(incoming[:size]):
                return size
        return 0

    def _create_transcriber_with_fallback(self) -> FasterWhisperTranscriber | None:
        try:
            return FasterWhisperTranscriber(
                model_size=self._config.model_size,
                device=self._config.model_device,
                compute_type=self._config.compute_type,
                max_context=self._config.whisper_max_context,
                entropy_thold=self._config.whisper_entropy_thold,
                logprob_thold=self._config.whisper_logprob_thold,
                no_speech_thold=self._config.whisper_no_speech_thold,
                temperature=self._config.whisper_temperature,
                beam_size=self._config.whisper_beam_size,
                best_of=self._config.whisper_best_of,
            )
        except Exception as exc:
            raw_message = str(exc)
            if (
                self._config.cpu_fallback_on_cuda_error
                and self._config.model_device.lower().startswith("cuda")
            ):
                if "cublas64_12.dll" in raw_message or "cannot be loaded" in raw_message:
                    self._emit_error("CUDA runtime missing (cublas64_12.dll).")
                    if self._try_prepare_cuda_compat_alias():
                        try:
                            transcriber = FasterWhisperTranscriber(
                                model_size=self._config.model_size,
                                device=self._config.model_device,
                                compute_type=self._config.compute_type,
                            )
                            self._emit_status("CUDA compatibility alias active. Keep using CUDA.")
                            return transcriber
                        except Exception as retry_exc:
                            self._emit_error(f"CUDA retry failed: {retry_exc}")
                else:
                    self._emit_error(f"CUDA init failed: {raw_message}")

                try:
                    transcriber = FasterWhisperTranscriber(
                        model_size=self._config.model_size,
                        device="cpu",
                        compute_type="int8",
                        max_context=self._config.whisper_max_context,
                        entropy_thold=self._config.whisper_entropy_thold,
                        logprob_thold=self._config.whisper_logprob_thold,
                        no_speech_thold=self._config.whisper_no_speech_thold,
                        temperature=self._config.whisper_temperature,
                        beam_size=self._config.whisper_beam_size,
                        best_of=self._config.whisper_best_of,
                    )
                except Exception as cpu_exc:
                    self._emit_error(f"CPU fallback init failed: {cpu_exc}")
                    return None

                self._emit_status("Whisper fallback active: device=cpu, compute_type=int8")
                return transcriber

            self._emit_error(f"Whisper init failed: {raw_message}")
            return None

    def _recover_from_runtime_transcription_error(self, raw_message: str) -> bool:
        if not self._config.model_device.lower().startswith("cuda"):
            return False

        if "cublas64_12.dll" not in raw_message and "cannot be loaded" not in raw_message:
            return False

        if not self._runtime_recovery_attempted:
            self._runtime_recovery_attempted = True
            self._emit_error(
                "Runtime CUDA DLL error detected. Trying cublas64_13 -> cublas64_12 compatibility alias."
            )

            if self._try_prepare_cuda_compat_alias():
                try:
                    self._transcriber = FasterWhisperTranscriber(
                        model_size=self._config.model_size,
                        device=self._config.model_device,
                        compute_type=self._config.compute_type,
                    )
                    self._emit_status("CUDA transcriber reloaded after DLL compatibility patch.")
                    return True
                except Exception as retry_exc:
                    self._emit_error(f"Runtime CUDA retry failed: {retry_exc}")

        if self._config.cpu_fallback_on_cuda_error and not self._cpu_runtime_fallback_done:
            self._cpu_runtime_fallback_done = True
            try:
                self._transcriber = FasterWhisperTranscriber(
                    model_size=self._config.model_size,
                    device="cpu",
                    compute_type="int8",
                    max_context=self._config.whisper_max_context,
                    entropy_thold=self._config.whisper_entropy_thold,
                    logprob_thold=self._config.whisper_logprob_thold,
                    no_speech_thold=self._config.whisper_no_speech_thold,
                    temperature=self._config.whisper_temperature,
                    beam_size=self._config.whisper_beam_size,
                    best_of=self._config.whisper_best_of,
                )
                self._emit_status(
                    "Runtime fallback active: switched to CPU because CUDA DLL could not be loaded."
                )
                return True
            except Exception as cpu_exc:
                self._emit_error(f"Runtime CPU fallback failed: {cpu_exc}")

        return False

    def _try_prepare_cuda_compat_alias(self) -> bool:
        return ensure_cublas12_from_source(
            source_dll=self._config.cuda_compat_source_dll,
            on_status=lambda msg: self._emit_status(msg),
        )

    def _emit_status(self, message: str) -> None:
        self._logger.info(message)
        self.status_message.emit(message)

    def _emit_error(self, message: str) -> None:
        self._logger.error(message)
        self.error_message.emit(message)