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
    get_translator: Callable[[], ArgosTranslator | None]
    recover_capture_backend: Callable[[], bool]
    recover_from_runtime_transcription_error: Callable[[str], bool]
    emit_status: Callable[[str], None]
    emit_debug_event: Callable[[dict[str, object]], None]
    emit_subtitle_ready: Callable[[str, str], None]
    record_transcript_event: Callable[[dict[str, object]], None]


class TranscriptionLoopEngine:
    def __init__(self, deps: TranscriptionLoopDeps) -> None:
        self._deps = deps
        self._silence_hops = 0
        self._speech_hops = 0
        self._window_elapsed_seconds = 0.0
        self._window_index = 0
        self._last_segment_artifact_log_at = 0.0
        self._auto_lang_locked = ""
        self._auto_lang_candidate = ""
        self._auto_lang_candidate_count = 0
        self._auto_lang_allowed = {"en", "zh", "ja", "ko", "de", "fr", "es", "it", "pt", "ru"}

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
        self._deps.subtitle_assembler.set_speaker_marker_style(
            str(getattr(self._deps.config, "speaker_marker_style", "spk") or "spk")
        )
        self._deps.subtitle_assembler.set_speaker_pause_break_seconds(
            float(getattr(self._deps.config, "speaker_pause_break_seconds", 1.8) or 0.0)
        )
        self._deps.subtitle_assembler.set_display_script(
            str(getattr(self._deps.config, "subtitle_display_script", "") or "")
        )

        (bytes_per_second, frame_bytes, segment_bytes, hop_bytes) = aligned_window_sizes(
            sample_rate=stream_rate,
            channels=stream_channels,
            segment_seconds=segment_seconds,
            hop_seconds=hop_seconds,
        )
        startup_silence_seconds = self._prefill_startup_silence(
            buffer=buffer,
            segment_bytes=segment_bytes,
            hop_bytes=hop_bytes,
            frame_bytes=frame_bytes,
            bytes_per_second=bytes_per_second,
            reason="startup",
        )
        if startup_silence_seconds > 0.0:
            # Keep token absolute timestamps aligned with real audio timeline
            # after injecting synthetic silence at startup.
            self._window_elapsed_seconds = -startup_silence_seconds
        last_chunk_at = time.monotonic()
        last_recover_at = 0.0

        try:
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
                    is_finished = getattr(capture, "is_finished", None)
                    if callable(is_finished) and bool(is_finished()):
                        self._deps.emit_status("Capture source finished.")
                        break
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
                    format_change_silence_seconds = self._prefill_startup_silence(
                        buffer=buffer,
                        segment_bytes=segment_bytes,
                        hop_bytes=hop_bytes,
                        frame_bytes=frame_bytes,
                        bytes_per_second=bytes_per_second,
                        reason="format-change",
                    )
                    if format_change_silence_seconds > 0.0:
                        self._window_elapsed_seconds -= format_change_silence_seconds
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
                    window_started_at = time.monotonic()
                    timing: dict[str, float] = {}
                    self._window_index += 1
                    current_window_index = int(self._window_index)
                    current_window_elapsed = float(self._window_elapsed_seconds)
                    self._window_elapsed_seconds += float(hop_seconds)
                    window = bytes(buffer[:segment_bytes])
                    del buffer[:hop_bytes]
                    window_chunk = AudioChunk(pcm16=window, sample_rate=stream_rate, channels=stream_channels)
                    stage_started_at = time.monotonic()
                    self._deps.segment_artifacts.write_chunk(window_chunk, self._deps.segment_artifacts.latest_raw_segment_wav)
                    timing["raw_artifact_seconds"] = time.monotonic() - stage_started_at

                    preprocess_pipeline = self._deps.get_preprocess_pipeline()
                    stage_started_at = time.monotonic()
                    if preprocess_pipeline is not None and preprocess_pipeline.stage_names:
                        stt_chunk = preprocess_pipeline.process(window_chunk, channel_mode=self._deps.config.source_channel_mode)
                    else:
                        stt_chunk = window_chunk
                    timing["preprocess_seconds"] = time.monotonic() - stage_started_at
                    stage_started_at = time.monotonic()
                    self._deps.segment_artifacts.write_chunk(stt_chunk, self._deps.segment_artifacts.latest_stt_segment_wav)
                    self._emit_segment_artifact_log(stt_chunk)
                    timing["stt_artifact_seconds"] = time.monotonic() - stage_started_at

                    transcriber = self._deps.get_transcriber()
                    if transcriber is None:
                        break
                    source_language_hint = self._runtime_source_language_hint()
                    prompt_chars = self._to_int(
                        getattr(self._deps.config, "whisperx_rolling_prompt_chars", 0), 0
                    )
                    if prompt_chars > 0:
                        set_prompt = getattr(transcriber, "set_initial_prompt", None)
                        if callable(set_prompt):
                            set_prompt(self._deps.subtitle_assembler.get_prompt_tail(prompt_chars))
                    try:
                        stage_started_at = time.monotonic()
                        source_text = transcriber.transcribe(
                            stt_chunk,
                            language=source_language_hint,
                            channel_mode=self._deps.config.source_channel_mode,
                        )
                        timing["transcribe_seconds"] = time.monotonic() - stage_started_at
                    except Exception as exc:
                        timing["transcribe_seconds"] = time.monotonic() - stage_started_at
                        if self._deps.recover_from_runtime_transcription_error(str(exc)):
                            continue
                        self._deps.emit_status(f"Transcription failed: {exc}")
                        continue

                    if not source_text.strip():
                        timing["window_total_seconds"] = time.monotonic() - window_started_at
                        self._emit_window_timing(
                            window_index=current_window_index,
                            elapsed_seconds=current_window_elapsed,
                            timing=timing,
                            has_text=False,
                        )
                        self._silence_hops += 1
                        silence_seconds = self._silence_hops * hop_seconds
                        if self._speech_hops > 0 and silence_seconds >= max(0.8, min(2.4, segment_seconds)):
                            self._mark_sentence_break()
                        continue
                    self._silence_hops = 0

                    transcription_meta = getattr(transcriber, "get_last_transcription_meta", lambda: {})()
                    if not isinstance(transcription_meta, dict):
                        transcription_meta = {}
                    transcription_meta = dict(transcription_meta)
                    self._emit_provider_timing(
                        window_index=current_window_index,
                        elapsed_seconds=current_window_elapsed,
                        transcription_meta=transcription_meta,
                    )
                    transcription_meta["runtime_window_index"] = int(current_window_index)
                    transcription_meta["elapsed_seconds"] = float(current_window_elapsed)
                    transcription_meta["runtime_source_language_hint"] = str(source_language_hint or "")
                    stage_started_at = time.monotonic()
                    self._update_auto_source_language_hint(transcription_meta)
                    transcription_meta["runtime_auto_source_language"] = self._runtime_display_language_hint()
                    timing["language_route_seconds"] = time.monotonic() - stage_started_at

                    stage_started_at = time.monotonic()
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
                    timing["timestamp_enrich_seconds"] = time.monotonic() - stage_started_at

                    stage_started_at = time.monotonic()
                    source_rolling = self._deps.subtitle_assembler.merge_incremental_text(
                        source_text,
                        overlap_merge_method=self._deps.config.overlap_merge_method,
                        segment_seconds=float(self._deps.config.segment_seconds),
                        hop_seconds=float(self._deps.config.hop_seconds),
                        transcription_meta=transcription_meta,
                    )
                    timing["merge_seconds"] = time.monotonic() - stage_started_at
                    merge_diagnostics = self._deps.subtitle_assembler.get_last_merge_diagnostics()
                    transcription_meta["merge_diagnostics"] = merge_diagnostics
                    self._emit_merge_timing(
                        window_index=current_window_index,
                        elapsed_seconds=current_window_elapsed,
                        merge_diagnostics=merge_diagnostics,
                    )
                    timing["window_total_seconds"] = time.monotonic() - window_started_at
                    transcription_meta["runtime_timing"] = {
                        key: round(float(value), 4) for (key, value) in timing.items()
                    }
                    if bool(getattr(self._deps.config, "debug_mode", False)):
                        history_tail_max_words = 160
                        history_count = int((merge_diagnostics or {}).get("history_count_after", 0) or 0)
                        self._deps.emit_debug_event(
                            {
                                "provider": getattr(self._deps.config, "stt_provider", "unknown"),
                                "provider_normalized": normalize_stt_provider(
                                    str(getattr(self._deps.config, "stt_provider", "whisperx") or "whisperx")
                                ),
                                "raw_text": source_text,
                                "merged_text": source_rolling,
                                "history_text": self._deps.subtitle_assembler.get_history_tail_text(history_tail_max_words),
                                "history_text_truncated": history_count > history_tail_max_words,
                                "history_text_max_words": history_tail_max_words,
                                "stable_text": self._deps.subtitle_assembler.get_stable_text(),
                                "partial_text": self._deps.subtitle_assembler.get_partial_text(),
                                "assembler_summary": self._deps.subtitle_assembler.get_debug_summary(),
                                "meta": transcription_meta,
                            }
                        )
                    if not source_rolling:
                        self._emit_window_timing(
                            window_index=current_window_index,
                            elapsed_seconds=current_window_elapsed,
                            timing=timing,
                            has_text=True,
                        )
                        continue

                    self._speech_hops += 1
                    stage_started_at = time.monotonic()
                    (source_out, translated_out) = self._build_subtitle_payload(
                        source_rolling,
                        runtime_source_language_hint=str(transcription_meta.get("runtime_auto_source_language") or ""),
                    )
                    timing["subtitle_payload_seconds"] = time.monotonic() - stage_started_at
                    timing["window_total_seconds"] = time.monotonic() - window_started_at
                    transcription_meta["runtime_timing"] = {
                        key: round(float(value), 4) for (key, value) in timing.items()
                    }
                    self._emit_window_timing(
                        window_index=current_window_index,
                        elapsed_seconds=current_window_elapsed,
                        timing=timing,
                        has_text=True,
                    )
                    if not source_out and (not translated_out):
                        continue
                    if source_out:
                        self._deps.text_delta_logger.log("STT", source_out, translated=False)
                    if translated_out:
                        self._deps.text_delta_logger.log("TRANSLATE", translated_out, translated=True)
                    self._deps.record_transcript_event(
                        {
                            "raw_text": source_text,
                            "source_text": source_out,
                            "translated_text": translated_out,
                            "meta": transcription_meta,
                        }
                    )
                    self._deps.emit_subtitle_ready(source_out, translated_out)

                    if self._speech_hops * hop_seconds >= max(segment_seconds, hop_seconds * 2.0):
                        self._mark_sentence_break()
        finally:
            self._finalize_stream()

    def _build_subtitle_payload(self, source_text: str, *, runtime_source_language_hint: str = "") -> tuple[str, str]:
        translator = self._deps.get_translator()
        if translator is None or not translator.enabled:
            return (source_text, "")
        translated = translator.translate(source_text, source_code=runtime_source_language_hint)
        if not translated:
            return (source_text, "")
        return (source_text, translated)

    def _finalize_stream(self) -> None:
        source_rolling = self._deps.subtitle_assembler.finalize()
        if not source_rolling:
            return
        (source_out, translated_out) = self._build_subtitle_payload(
            source_rolling,
            runtime_source_language_hint=self._runtime_display_language_hint(),
        )
        if not source_out and not translated_out:
            return
        if source_out:
            self._deps.text_delta_logger.log("STT", source_out, translated=False)
        if translated_out:
            self._deps.text_delta_logger.log("TRANSLATE", translated_out, translated=True)
        self._deps.record_transcript_event(
            {
                "raw_text": "",
                "source_text": source_out,
                "translated_text": translated_out,
                "meta": {
                    "snapshot_final": True,
                    "elapsed_seconds": float(self._window_elapsed_seconds),
                    "runtime_auto_source_language": self._runtime_display_language_hint(),
                    "merge_diagnostics": self._deps.subtitle_assembler.get_last_merge_diagnostics(),
                },
            }
        )
        self._deps.emit_subtitle_ready(source_out, translated_out)

    def _emit_window_timing(
        self,
        *,
        window_index: int,
        elapsed_seconds: float,
        timing: dict[str, float],
        has_text: bool,
    ) -> None:
        if not bool(getattr(self._deps.config, "debug_mode", False)):
            return
        self._deps.emit_status(
            "[window-timing] "
            f"window={int(window_index)}; "
            f"audio={float(elapsed_seconds):.3f}s; "
            f"text={str(bool(has_text)).lower()}; "
            f"raw={float(timing.get('raw_artifact_seconds', 0.0) or 0.0):.4f}s; "
            f"preprocess={float(timing.get('preprocess_seconds', 0.0) or 0.0):.4f}s; "
            f"stt_artifact={float(timing.get('stt_artifact_seconds', 0.0) or 0.0):.4f}s; "
            f"transcribe={float(timing.get('transcribe_seconds', 0.0) or 0.0):.4f}s; "
            f"language={float(timing.get('language_route_seconds', 0.0) or 0.0):.4f}s; "
            f"timestamp={float(timing.get('timestamp_enrich_seconds', 0.0) or 0.0):.4f}s; "
            f"merge={float(timing.get('merge_seconds', 0.0) or 0.0):.4f}s; "
            f"payload={float(timing.get('subtitle_payload_seconds', 0.0) or 0.0):.4f}s; "
            f"total={float(timing.get('window_total_seconds', 0.0) or 0.0):.4f}s"
        )

    def _emit_provider_timing(
        self,
        *,
        window_index: int,
        elapsed_seconds: float,
        transcription_meta: dict[str, object],
    ) -> None:
        if not bool(getattr(self._deps.config, "debug_mode", False)):
            return
        raw = transcription_meta.get("provider_timing")
        if not isinstance(raw, dict):
            return
        align_detail = raw.get("align_detail") if isinstance(raw.get("align_detail"), dict) else {}
        diar_detail = raw.get("diarization_detail") if isinstance(raw.get("diarization_detail"), dict) else {}
        self._deps.emit_status(
            "[whisperx-timing] "
            f"window={int(window_index)}; "
            f"audio={float(elapsed_seconds):.3f}s; "
            f"trace={int(raw.get('trace_id', 0) or 0)}; "
            f"pcm={float(raw.get('pcm_convert_seconds', 0.0) or 0.0):.4f}s; "
            f"resample={float(raw.get('resample_seconds', 0.0) or 0.0):.4f}s; "
            f"prepare={float(raw.get('prepare_seconds', 0.0) or 0.0):.4f}s; "
            f"asr={float(raw.get('asr_seconds', 0.0) or 0.0):.4f}s; "
            f"align={float(raw.get('align_seconds', 0.0) or 0.0):.4f}s; "
            f"align_model={float(align_detail.get('model_load_seconds', 0.0) or 0.0):.4f}s; "
            f"align_run={float(align_detail.get('run_seconds', 0.0) or 0.0):.4f}s; "
            f"diar={float(raw.get('diarization_seconds', 0.0) or 0.0):.4f}s; "
            f"diar_load={float(diar_detail.get('pipeline_load_seconds', 0.0) or 0.0):.4f}s; "
            f"diar_run={float(diar_detail.get('pipeline_run_seconds', 0.0) or 0.0):.4f}s; "
            f"diar_assign={float(diar_detail.get('assign_seconds', 0.0) or 0.0):.4f}s; "
            f"profile={float(raw.get('speaker_profile_seconds', 0.0) or 0.0):.4f}s; "
            f"meta={float(raw.get('meta_build_seconds', 0.0) or 0.0):.4f}s; "
            f"total={float(raw.get('total_seconds', 0.0) or 0.0):.4f}s; "
            f"tokens={int(raw.get('token_count', 0) or 0)}; "
            f"segments={int(raw.get('final_segment_count', 0) or 0)}; "
            f"align_status={str(align_detail.get('status') or '')}; "
            f"diar_status={str(diar_detail.get('status') or '')}"
        )

    def _emit_merge_timing(
        self,
        *,
        window_index: int,
        elapsed_seconds: float,
        merge_diagnostics: dict[str, object],
    ) -> None:
        if not bool(getattr(self._deps.config, "debug_mode", False)):
            return
        if not isinstance(merge_diagnostics, dict) or not merge_diagnostics:
            return
        history_dedupe = merge_diagnostics.get("history_dedupe")
        if not isinstance(history_dedupe, dict):
            history_dedupe = {}
        self._deps.emit_status(
            "[merge-timing] "
            f"window={int(window_index)}; "
            f"audio={float(elapsed_seconds):.3f}s; "
            f"total={float(merge_diagnostics.get('total_seconds', 0.0) or 0.0):.4f}s; "
            f"normalize={float(merge_diagnostics.get('normalize_seconds', 0.0) or 0.0):.4f}s; "
            f"extract={float(merge_diagnostics.get('extract_seconds', 0.0) or 0.0):.4f}s; "
            f"state={float(merge_diagnostics.get('state_update_seconds', 0.0) or 0.0):.4f}s; "
            f"partial_render={float(merge_diagnostics.get('partial_render_seconds', 0.0) or 0.0):.4f}s; "
            f"history_render={float(merge_diagnostics.get('history_render_seconds', 0.0) or 0.0):.4f}s; "
            f"spacing={float(merge_diagnostics.get('spacing_seconds', 0.0) or 0.0):.4f}s; "
            f"overlap={float(merge_diagnostics.get('overlap_seconds', 0.0) or 0.0):.4f}s; "
            f"final_norm={float(merge_diagnostics.get('final_normalize_seconds', 0.0) or 0.0):.4f}s; "
            f"incoming={int(merge_diagnostics.get('incoming_count', 0) or 0)}; "
            f"history_before={int(merge_diagnostics.get('history_count_before', 0) or 0)}; "
            f"history_after={int(merge_diagnostics.get('history_count_after', 0) or 0)}; "
            f"stable_after={int(merge_diagnostics.get('stable_count_after', 0) or 0)}; "
            f"partial_after={int(merge_diagnostics.get('partial_count_after', 0) or 0)}; "
            f"moved_to_history={int(merge_diagnostics.get('moved_to_history_count', 0) or 0)}; "
            f"rolling_source={str(merge_diagnostics.get('rolling_base_source', ''))}; "
            f"rolling_base_chars={int(merge_diagnostics.get('rolling_base_chars', 0) or 0)}; "
            f"rolling_committed_chars={int(merge_diagnostics.get('rolling_committed_chars', 0) or 0)}; "
            f"history_chars={int(merge_diagnostics.get('history_chars', 0) or 0)}; "
            f"merged_chars={int(merge_diagnostics.get('merged_chars', 0) or 0)}; "
            f"dedupe_mode={str(history_dedupe.get('mode', ''))}; "
            f"dedupe_input={int(history_dedupe.get('input_count', 0) or 0)}; "
            f"dedupe_tail={int(history_dedupe.get('tail_count', 0) or 0)}; "
            f"dedupe_moved={int(history_dedupe.get('moved_count', 0) or 0)}; "
            f"empty={str(bool(merge_diagnostics.get('returned_empty', False))).lower()}"
        )

    def _runtime_source_language_hint(self) -> str | None:
        raw = self._deps.config.source_language
        token = self._normalize_language_token(raw)
        if token:
            return token
        # Keep ASR in auto mode so WhisperX can continue detecting durable
        # language switches inside long sessions. The rolling lock is only a
        # downstream display/translation hint.
        return None

    def _runtime_display_language_hint(self) -> str:
        raw = self._deps.config.source_language
        token = self._normalize_language_token(raw)
        if token:
            return token
        return str(self._auto_lang_locked or "")

    @staticmethod
    def _normalize_language_token(value: object) -> str:
        token = str(value or "").strip().lower()
        if not token or token == "auto":
            return ""
        if token in {"zh-hant", "zh-hans", "zh-tw", "zh-cn", "zh-hk", "zh-sg"}:
            return "zh"
        return token

    def _update_auto_source_language_hint(self, transcription_meta: dict[str, object]) -> None:
        raw_source = self._deps.config.source_language
        if self._normalize_language_token(raw_source):
            # Explicit source language set by user; do not auto-lock.
            self._auto_lang_locked = self._normalize_language_token(raw_source)
            self._auto_lang_candidate = ""
            self._auto_lang_candidate_count = 0
            return

        detected = self._normalize_language_token(transcription_meta.get("detected_language"))
        if not detected:
            return
        if detected not in self._auto_lang_allowed:
            return
        token_count = int(max(0, self._to_int(transcription_meta.get("token_count"), 0)))
        stability_ratio = self._to_float(transcription_meta.get("stability_ratio"), 0.0)
        # Ignore very short/noisy windows; they are the primary cause of language jitter.
        if token_count < 8 and stability_ratio < 0.50:
            return

        if not self._auto_lang_locked:
            self._auto_lang_locked = detected
            self._deps.emit_status(
                f"Auto source language locked: {self._auto_lang_locked} "
                f"(tokens={token_count}, stability={stability_ratio:.2f})"
            )
            return
        if detected == self._auto_lang_locked:
            self._auto_lang_candidate = ""
            self._auto_lang_candidate_count = 0
            return
        if detected == self._auto_lang_candidate:
            self._auto_lang_candidate_count += 1
        else:
            self._auto_lang_candidate = detected
            self._auto_lang_candidate_count = 1

        if self._auto_lang_candidate_count < 2:
            return
        previous = self._auto_lang_locked
        self._auto_lang_locked = self._auto_lang_candidate
        self._auto_lang_candidate = ""
        self._auto_lang_candidate_count = 0
        self._deps.emit_status(
            f"Auto source language switched: {previous} -> {self._auto_lang_locked} "
            f"(tokens={token_count}, stability={stability_ratio:.2f})"
        )

    @staticmethod
    def _to_int(value: object, fallback: int) -> int:
        try:
            return int(value)
        except Exception:
            return fallback

    @staticmethod
    def _to_float(value: object, fallback: float) -> float:
        try:
            return float(value)
        except Exception:
            return fallback

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

    def _prefill_startup_silence(
        self,
        *,
        buffer: bytearray,
        segment_bytes: int,
        hop_bytes: int,
        frame_bytes: int,
        bytes_per_second: int,
        reason: str,
    ) -> float:
        padding_bytes = max(0, int(segment_bytes) - int(hop_bytes))
        if padding_bytes <= 0:
            return 0.0
        aligned_padding = max(int(frame_bytes), (padding_bytes // max(1, int(frame_bytes))) * int(frame_bytes))
        if aligned_padding <= 0:
            return 0.0
        if buffer:
            return 0.0
        buffer.extend(b"\x00" * aligned_padding)
        padding_seconds = float(aligned_padding) / float(max(1, int(bytes_per_second)))
        if bool(getattr(self._deps.config, "debug_mode", False)):
            self._deps.emit_status(
                "[startup-padding] injected leading silence: "
                f"reason={reason}; bytes={aligned_padding}; seconds={padding_seconds:.3f}; "
                f"segment_bytes={segment_bytes}; hop_bytes={hop_bytes}"
            )
        return padding_seconds
