"""Transcription capture-loop state machine extracted from controller orchestration."""
from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Event
from typing import Callable

from ..capture import AudioChunk
from ..config import RuntimeConfig
from ..stt import STTTranscriber
from ..stt.registry import normalize_stt_provider
from ..stt.preprocessing import AudioPreprocessingPipeline
from ..stt.vad import VADPipeline
from ..translator import ArgosTranslator
from .audio_windowing import aligned_window_sizes
from .gpu_telemetry import GpuTelemetryReporter
from .segment_artifacts import SegmentArtifacts
from .subtitle_assembler import SubtitleAssembler
from .text_delta_logger import TextDeltaLogger


@dataclass
class TranscriptionLoopDeps:
    config: RuntimeConfig
    subtitle_assembler: SubtitleAssembler
    text_delta_logger: TextDeltaLogger
    segment_artifacts: SegmentArtifacts
    gpu_telemetry: GpuTelemetryReporter
    get_capture: Callable[[], object | None]
    get_transcriber: Callable[[], STTTranscriber | None]
    get_preprocess_pipeline: Callable[[], AudioPreprocessingPipeline | None]
    get_vad_pipeline: Callable[[], VADPipeline | None]
    get_translator: Callable[[], ArgosTranslator | None]
    recover_capture_backend: Callable[[], bool]
    recover_from_runtime_transcription_error: Callable[[str], bool]
    emit_status: Callable[[str], None]
    emit_debug_event: Callable[[dict[str, object]], None]
    emit_subtitle_ready: Callable[[str, str], None]


class TranscriptionLoopEngine:
    def __init__(self, deps: TranscriptionLoopDeps) -> None:
        self._deps = deps
        self._silence_hops = 0
        self._speech_hops = 0
        self._window_elapsed_seconds = 0.0
        self._last_segment_artifact_log_at = 0.0

    def run(self, running: Event) -> None:
        capture = self._deps.get_capture()
        transcriber = self._deps.get_transcriber()
        if capture is None or transcriber is None:
            raise RuntimeError("Capture/transcriber is not initialized before run loop.")

        buffer = bytearray()
        stream_rate = int(getattr(capture, "sample_rate", 16000))
        stream_channels = int(getattr(capture, "channels", 1))
        segment_seconds = min(max(1.0, float(self._deps.config.segment_seconds)), 12.0)
        hop_seconds = min(max(0.1, float(self._deps.config.hop_seconds)), max(0.1, segment_seconds - 0.1))

        self._deps.subtitle_assembler.set_language_context(self._deps.config.source_language)
        self._deps.subtitle_assembler.set_cjk_no_space_gap_seconds(
            float(getattr(self._deps.config, "cjk_no_space_gap_seconds", 0.2) or 0.2)
        )

        (bytes_per_second, frame_bytes, segment_bytes, hop_bytes) = aligned_window_sizes(
            sample_rate=stream_rate,
            channels=stream_channels,
            segment_seconds=segment_seconds,
            hop_seconds=hop_seconds,
        )
        last_chunk_at = time.monotonic()
        last_recover_at = 0.0

        while running.is_set():
            capture = self._deps.get_capture()
            transcriber = self._deps.get_transcriber()
            if capture is None or transcriber is None:
                break
            try:
                chunk = capture.read_chunk(timeout=0.25)
            except Exception as exc:
                self._deps.emit_status(f"Capture read failed: {exc}")
                chunk = None

            now = time.monotonic()
            self._deps.gpu_telemetry.maybe_emit(
                now_monotonic=now,
                debug_mode=bool(getattr(self._deps.config, "debug_mode", False)),
                model_device=str(getattr(self._deps.config, "model_device", "") or ""),
                emit_status=self._deps.emit_status,
            )
            if chunk is None:
                if now - last_chunk_at >= 8.0 and now - last_recover_at >= 8.0:
                    last_recover_at = now
                    self._deps.emit_status("No audio chunks for 8s. Restarting capture backend...")
                    if self._deps.recover_capture_backend():
                        last_chunk_at = time.monotonic()
                continue

            last_chunk_at = now
            if chunk.sample_rate != stream_rate or chunk.channels != stream_channels or segment_bytes <= 0:
                stream_rate = chunk.sample_rate
                stream_channels = chunk.channels
                (bytes_per_second, frame_bytes, segment_bytes, hop_bytes) = aligned_window_sizes(
                    sample_rate=stream_rate,
                    channels=stream_channels,
                    segment_seconds=segment_seconds,
                    hop_seconds=hop_seconds,
                )
                buffer.clear()
                self._deps.emit_status(f"Stream format changed: {stream_rate} Hz, {stream_channels} ch")

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
                del buffer[: len(buffer) - max_buffer_bytes]

            while len(buffer) >= segment_bytes and running.is_set():
                current_window_elapsed = float(self._window_elapsed_seconds)
                self._window_elapsed_seconds += float(hop_seconds)
                window = bytes(buffer[:segment_bytes])
                del buffer[:hop_bytes]
                window_chunk = AudioChunk(pcm16=window, sample_rate=stream_rate, channels=stream_channels)
                self._deps.segment_artifacts.write_chunk(window_chunk, self._deps.segment_artifacts.latest_raw_segment_wav)

                preprocess_pipeline = self._deps.get_preprocess_pipeline()
                if preprocess_pipeline is not None and preprocess_pipeline.stage_names:
                    stt_chunk = preprocess_pipeline.process(window_chunk, channel_mode=self._deps.config.source_channel_mode)
                else:
                    stt_chunk = window_chunk
                self._deps.segment_artifacts.write_chunk(stt_chunk, self._deps.segment_artifacts.latest_stt_segment_wav)
                self._emit_segment_artifact_log(stt_chunk)

                vad_pipeline = self._deps.get_vad_pipeline()
                if vad_pipeline is not None:
                    has_signal = vad_pipeline.should_process(stt_chunk, channel_mode=self._deps.config.source_channel_mode)
                else:
                    transcriber = self._deps.get_transcriber()
                    if transcriber is None:
                        break
                    has_signal = transcriber.has_enough_signal(stt_chunk, channel_mode=self._deps.config.source_channel_mode)

                if not has_signal:
                    self._silence_hops += 1
                    silence_seconds = self._silence_hops * hop_seconds
                    if self._speech_hops > 0 and silence_seconds >= max(0.8, min(2.4, segment_seconds)):
                        self._mark_sentence_break()
                    continue

                self._silence_hops = 0
                transcriber = self._deps.get_transcriber()
                if transcriber is None:
                    break
                try:
                    source_text = transcriber.transcribe(
                        stt_chunk,
                        language=self._deps.config.source_language,
                        channel_mode=self._deps.config.source_channel_mode,
                    )
                except Exception as exc:
                    if self._deps.recover_from_runtime_transcription_error(str(exc)):
                        continue
                    self._deps.emit_status(f"Transcription failed: {exc}")
                    continue

                transcription_meta = getattr(transcriber, "get_last_transcription_meta", lambda: {})()
                if not isinstance(transcription_meta, dict):
                    transcription_meta = {}
                transcription_meta = dict(transcription_meta)
                transcription_meta["elapsed_seconds"] = float(current_window_elapsed)

                token_rows = transcription_meta.get("token_timestamps")
                if isinstance(token_rows, list):
                    enriched_rows: list[dict[str, object]] = []
                    for row in token_rows:
                        if not isinstance(row, dict):
                            continue
                        item = dict(row)
                        try:
                            start_rel = float(item.get("start"))
                            end_rel = float(item.get("end"))
                            item["absolute_start"] = float(current_window_elapsed + start_rel)
                            item["absolute_end"] = float(current_window_elapsed + end_rel)
                        except Exception:
                            pass
                        enriched_rows.append(item)
                    transcription_meta["token_timestamps"] = enriched_rows

                source_rolling = self._deps.subtitle_assembler.merge_incremental_text(
                    source_text,
                    overlap_merge_method=self._deps.config.overlap_merge_method,
                    segment_seconds=float(self._deps.config.segment_seconds),
                    hop_seconds=float(self._deps.config.hop_seconds),
                    transcription_meta=transcription_meta,
                )
                if bool(getattr(self._deps.config, "debug_mode", False)):
                    self._deps.emit_debug_event(
                        {
                            "provider": getattr(self._deps.config, "stt_provider", "unknown"),
                            "provider_normalized": normalize_stt_provider(
                                str(getattr(self._deps.config, "stt_provider", "whisper") or "whisper")
                            ),
                            "raw_text": source_text,
                            "merged_text": source_rolling,
                            "history_text": self._deps.subtitle_assembler.get_history_text(),
                            "history_state": self._deps.subtitle_assembler.get_history_state(),
                            "stable_text": self._deps.subtitle_assembler.get_stable_text(),
                            "stable_state": self._deps.subtitle_assembler.get_stable_state(),
                            "partial_state": self._deps.subtitle_assembler.get_partial_state(),
                            "meta": transcription_meta,
                        }
                    )
                if not source_rolling:
                    continue

                self._speech_hops += 1
                (source_out, translated_out) = self._build_subtitle_payload(source_rolling)
                if not source_out and (not translated_out):
                    continue
                if source_out:
                    self._deps.text_delta_logger.log("STT", source_out, translated=False)
                if translated_out:
                    self._deps.text_delta_logger.log("TRANSLATE", translated_out, translated=True)
                self._deps.emit_subtitle_ready(source_out, translated_out)

                if self._speech_hops * hop_seconds >= max(segment_seconds, hop_seconds * 2.0):
                    self._mark_sentence_break()

    def _build_subtitle_payload(self, source_text: str) -> tuple[str, str]:
        translator = self._deps.get_translator()
        if translator is None or not translator.enabled:
            return (source_text, "")
        translated = translator.translate(source_text)
        if not translated:
            return (source_text, "")
        return (source_text, translated)

    def _mark_sentence_break(self) -> None:
        self._deps.subtitle_assembler.mark_sentence_break()
        self._silence_hops = 0
        self._speech_hops = 0

    def _emit_segment_artifact_log(self, stt_chunk: AudioChunk) -> None:
        if not bool(getattr(self._deps.config, "debug_mode", False)):
            return
        now = time.monotonic()
        if now - self._last_segment_artifact_log_at < 1.0:
            return
        self._last_segment_artifact_log_at = now
        path = self._deps.segment_artifacts.latest_stt_segment_wav
        try:
            file_bytes = int(path.stat().st_size) if path.exists() else 0
        except Exception:
            file_bytes = 0
        sample_rate = max(1, int(stt_chunk.sample_rate))
        channels = max(1, int(stt_chunk.channels))
        sample_count = int(len(stt_chunk.pcm16) // 2 // channels)
        duration_sec = float(sample_count) / float(sample_rate)
        self._deps.emit_status(
            "[segment-artifact] latest_stt_segment.wav updated: "
            f"path={path}; bytes={file_bytes}; sample_rate={sample_rate}; channels={channels}; "
            f"samples={sample_count}; duration_sec={duration_sec:.3f}"
        )
