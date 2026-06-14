"""WhisperX provider adapter with optional alignment for subtitle quality."""
from __future__ import annotations

import os
import numpy as np
from pathlib import Path
from typing import Callable, Optional
import threading
import time
import warnings
import shutil
import urllib.request

from ..model_paths import library_model_dir
from .audio_utils import has_enough_signal, normalize_chinese_script, normalize_language_hint, pcm16_to_mono_float, resample
from .model_download import download_hf_files_with_progress, download_hf_snapshot_with_progress, emit_progress, estimate_hf_files_total, format_download_progress
from .speaker_identity import SpeakerIdentityConfig, SpeakerIdentityEngine
import re


class WhisperXTranscriber:
    def __init__(
        self,
        model_ref: str = "small",
        *,
        device: str = "cuda",
        compute_type: str = "float16",
        batch_size: int = 4,
        enable_phoneme_asr: bool = True,
        enable_forced_alignment: bool = True,
        enable_vad: bool = True,
        vad_method: str = "silero-vad",
        enable_diarization: bool = False,
        alignment_model: str = "",
        alignment_language: str = "auto",
        alignment_device: str = "auto",
        diarization_device: str = "auto",
        source_language_hint: str | None = None,
        diarization_model: str = "pyannote/speaker-diarization-3.1",
        hf_token: str = "",
        speaker_profile_enabled: bool = True,
        speaker_profile_backend: str = "pyannote",
        speaker_profile_model: str = "pyannote/embedding",
        speaker_speechbrain_model: str = "speechbrain/spkrec-ecapa-voxceleb",
        speaker_nemo_model: str = "nvidia/speakerverification_en_titanet_large",
        speaker_profile_match_threshold: float = 0.72,
        speaker_profile_min_seconds: float = 0.8,
        speaker_profile_reconcile_threshold: float = 0.52,
        speaker_profile_store_path: str = "",
        speaker_marker_style: str = "spk",
        auto_download: bool = True,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        warnings.filterwarnings(
            'ignore',
            message=r'.*TensorFloat-32 \(TF32\) has been disabled.*',
            module=r'pyannote\.audio\.utils\.reproducibility',
        )
        warnings.filterwarnings(
            'ignore',
            message=r'.*torchcodec is not installed correctly.*',
            module=r'pyannote\.audio\.core\.io',
        )
        warnings.filterwarnings(
            'ignore',
            message=r'Mean of empty slice\.',
            category=RuntimeWarning,
            module=r'numpy\._core\.fromnumeric',
        )
        warnings.filterwarnings(
            'ignore',
            message=r'invalid value encountered in divide',
            category=RuntimeWarning,
            module=r'numpy\._core\._methods',
        )
        warnings.filterwarnings(
            'ignore',
            message=r'.*CUBLAS_STATUS_NOT_INITIALIZED.*Will attempt to recover by calling unfused cublas path.*',
            category=UserWarning,
            module=r'torch\.nn\.modules\.linear',
        )
        self._model_root = library_model_dir("whisperx")
        self._configure_hf_cache_env(self._model_root)
        try:
            import whisperx  # type: ignore
        except Exception as exc:  # pragma: no cover - import path depends on env
            raise RuntimeError(f"whisperx is not available: {exc}") from exc

        self._whisperx = whisperx
        self._model_ref = (model_ref or "small").strip() or "small"
        self._device = "cpu" if device.strip().lower().startswith("cpu") else "cuda"
        self._compute_type = (compute_type or "float16").strip()
        self._batch_size = max(1, int(batch_size))
        self._enable_phoneme_asr = bool(enable_phoneme_asr)
        self._enable_forced_alignment = bool(enable_forced_alignment)
        self._enable_vad = bool(enable_vad)
        self._vad_method = (vad_method or "silero-vad").strip().lower()
        self._enable_diarization = bool(enable_diarization)
        self._alignment_model = (alignment_model or "").strip()
        self._alignment_language = (alignment_language or 'auto').strip().lower()
        self._alignment_device_setting = (alignment_device or "auto").strip().lower()
        self._diarization_device_setting = (diarization_device or "auto").strip().lower()
        (normalized_source_lang, _) = normalize_language_hint(source_language_hint)
        self._source_language_hint = normalized_source_lang
        self._diarization_model = self._normalize_diarization_model_ref(
            diarization_model or "pyannote/speaker-diarization-3.1"
        )
        self._hf_token = (hf_token or "").strip()
        self._speaker_profile_enabled = bool(speaker_profile_enabled)
        self._speaker_profile_backend = (speaker_profile_backend or "pyannote").strip()
        self._speaker_profile_model = (speaker_profile_model or "pyannote/embedding").strip()
        self._speaker_speechbrain_model = (
            speaker_speechbrain_model or "speechbrain/spkrec-ecapa-voxceleb"
        ).strip()
        self._speaker_nemo_model = (
            speaker_nemo_model or "nvidia/speakerverification_en_titanet_large"
        ).strip()
        self._speaker_profile_match_threshold = float(max(0.0, min(0.999, speaker_profile_match_threshold)))
        self._speaker_profile_min_seconds = float(max(0.2, speaker_profile_min_seconds))
        self._speaker_profile_reconcile_threshold = float(
            max(0.0, min(0.999, speaker_profile_reconcile_threshold))
        )
        self._speaker_profile_store_path = self._resolve_speaker_profile_store_path(speaker_profile_store_path)
        marker_style = str(speaker_marker_style or "").strip().lower()
        self._speaker_marker_style = "arrow" if marker_style in {"arrow", "arrows", ">>"} else "spk"
        self._auto_download = bool(auto_download)
        self._progress_callback = progress_callback
        self._trace_enabled = (os.environ.get("VOICE2TEXT_TRACE_WHISPERX", "0").strip().lower() not in {"", "0", "false", "no", "off"})
        self._trace_counter = 0
        self._align_bench_count = 0
        self._align_bench_total_ms = 0.0
        self._align_bench_max_ms = 0.0
        self._alignment_device = self._resolve_alignment_device()
        self._diarization_device = self._resolve_diarization_device()
        self._external_download_monitor_lock = threading.Lock()
        self._external_download_monitor_suppress_count = 0
        self._external_download_monitor_suppress_epoch = 0

        self._download_probe_roots = self._build_download_probe_roots()
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self._normalize_alignment_layout()
        self._cleanup_alignment_partial_cache()

        model_arg = self._resolve_stt_model_arg(self._model_ref)
        self._model = whisperx.load_model(
            model_arg,
            self._device,
            compute_type=self._compute_type,
            language=self._source_language_hint,
            vad_method=self._resolve_whisperx_vad_method(self._vad_method),
            download_root=str(self._model_root / "stt"),
        )

        self._align_cache: dict[str, tuple[object, object]] = {}
        self._diarization_pipeline = None
        self._diarization_disabled_reason: str | None = None
        self._diarization_cuda_warmed = False
        self._speaker_embedding_inference = None
        self._speaker_embedding_disabled_reason: str | None = None
        self._speaker_identity_engine: SpeakerIdentityEngine | None = None
        self._last_speaker_profile_stats: dict[str, object] = {}
        self._last_transcription_meta: dict[str, object] = {}
        self._last_alignment_timing: dict[str, object] = {}
        self._last_diarization_timing: dict[str, object] = {}
        self._language_route_logged: set[tuple[str, str, str, str]] = set()
        self._last_speaker_label: str | None = None
        self._speaker_display_map: dict[str, str] = {}
        self._speaker_display_next_index = 0
        self._speaker_switch_confirm_segments = 2
        self._speaker_switch_min_duration_seconds = 0.18
        # Allow immediate switch if one new-speaker segment is clearly long.
        self._speaker_switch_single_segment_min_duration_seconds = max(
            0.35,
            float(self._speaker_switch_min_duration_seconds) * 2.0,
        )
        self._speaker_switch_pending_label = ""
        self._speaker_switch_pending_count = 0
        self._speaker_switch_pending_duration_seconds = 0.0
        if self._enable_diarization:
            if self._speaker_profile_enabled:
                self._reset_speaker_profile_store_on_startup()
            self._speaker_identity_engine = SpeakerIdentityEngine(
                SpeakerIdentityConfig(
                    enabled=self._speaker_profile_enabled,
                    backend=self._speaker_profile_backend,
                    store_path=self._speaker_profile_store_path,
                    match_threshold=self._speaker_profile_match_threshold,
                    min_seconds=self._speaker_profile_min_seconds,
                    reconcile_threshold=self._speaker_profile_reconcile_threshold,
                    model_root=str(self._model_root),
                    device=self._diarization_device,
                    hf_token=self._resolve_hf_token() or "",
                    pyannote_model=self._speaker_profile_model,
                    speechbrain_model=self._speaker_speechbrain_model,
                    nemo_model=self._speaker_nemo_model,
                    on_status=self._emit,
                )
            )
        self._emit(f"WhisperX initialized: model={model_arg}, device={self._device}")
        self._emit(f"WhisperX internal VAD method: {self._resolve_whisperx_vad_method(self._vad_method)}")
        if self._alignment_device != self._device:
            self._emit(
                "WhisperX alignment device override active: "
                f"asr_device={self._device}, align_device={self._alignment_device}"
            )
        if self._enable_diarization and self._diarization_device != self._device:
            self._emit(
                "WhisperX diarization device override active: "
                f"asr_device={self._device}, diarization_device={self._diarization_device}"
            )

    def has_enough_signal(self, chunk, threshold: float = 0.008, channel_mode: str = "mono") -> bool:
        return has_enough_signal(chunk, threshold=threshold, channel_mode=channel_mode)

    def transcribe(self, chunk, language: Optional[str] = None, channel_mode: str = "mono") -> str:
        provider_started_at = time.perf_counter()
        provider_timing: dict[str, object] = {
            "trace_id": int(self._trace_counter + 1),
            "input_sample_rate": int(getattr(chunk, "sample_rate", 0) or 0),
            "input_channels": int(getattr(chunk, "channels", 0) or 0),
            "alignment_enabled": bool(self._enable_forced_alignment),
            "diarization_enabled": bool(self._enable_diarization),
            "speaker_profile_enabled": bool(self._speaker_profile_enabled),
        }
        self._trace_counter += 1
        trace_id = self._trace_counter
        provider_timing["trace_id"] = int(trace_id)
        stage_started_at = time.perf_counter()
        audio = pcm16_to_mono_float(chunk.pcm16, chunk.channels, channel_mode=channel_mode)
        provider_timing["pcm_convert_seconds"] = time.perf_counter() - stage_started_at
        if audio.size == 0:
            provider_timing["total_seconds"] = time.perf_counter() - provider_started_at
            self._last_transcription_meta = {
                "stability_ratio": 1.0,
                "token_count": 0,
                "stable_token_count": 0,
                "token_timestamps": [],
                "provider_timing": provider_timing,
            }
            return ""
        if chunk.sample_rate != 16000:
            stage_started_at = time.perf_counter()
            audio = resample(audio, chunk.sample_rate, 16000)
            provider_timing["resample_seconds"] = time.perf_counter() - stage_started_at
        else:
            provider_timing["resample_seconds"] = 0.0
        if audio.size == 0:
            provider_timing["total_seconds"] = time.perf_counter() - provider_started_at
            self._last_transcription_meta = {
                "stability_ratio": 1.0,
                "token_count": 0,
                "stable_token_count": 0,
                "token_timestamps": [],
                "provider_timing": provider_timing,
            }
            return ""
        provider_timing["audio_samples"] = int(audio.size)
        provider_timing["audio_seconds"] = float(audio.size) / 16000.0
        if self._trace_enabled:
            self._emit(
                f"[whisperx-trace] #{trace_id} start: in={chunk.sample_rate}Hz/{chunk.channels}ch, "
                f"resampled=16000Hz/1ch, samples={int(audio.size)}"
            )

        stage_started_at = time.perf_counter()
        (lang_hint, zh_script) = normalize_language_hint(language)
        kwargs: dict[str, object] = {"batch_size": self._batch_size}
        if lang_hint is not None:
            kwargs["language"] = lang_hint
        if not self._enable_phoneme_asr:
            kwargs["task"] = "transcribe"
        provider_timing["prepare_seconds"] = time.perf_counter() - stage_started_at

        stage_started_at = time.perf_counter()
        result = self._transcribe_with_compat(audio, kwargs)
        provider_timing["asr_seconds"] = time.perf_counter() - stage_started_at
        segments = list(result.get("segments", []) or [])
        provider_timing["asr_segment_count"] = int(len(segments))
        lang_detected = str(result.get("language") or "").strip().lower()
        stage_started_at = time.perf_counter()
        align_lang = self._resolve_alignment_language(lang_hint, lang_detected)
        self._emit_language_route(language, lang_hint, lang_detected, align_lang)
        provider_timing["language_route_seconds"] = time.perf_counter() - stage_started_at
        if self._trace_enabled:
            self._emit(
                f"[whisperx-trace] #{trace_id} asr-done: segments={len(segments)}, "
                f"detected_lang={lang_detected or 'n/a'}"
            )
        self._last_alignment_timing = {}
        if self._enable_forced_alignment:
            stage_started_at = time.perf_counter()
            aligned = self._align_segments(audio, segments, align_lang, trace_id=trace_id)
            provider_timing["align_seconds"] = time.perf_counter() - stage_started_at
            provider_timing["align_detail"] = dict(self._last_alignment_timing)
        else:
            aligned = segments
            provider_timing["align_seconds"] = 0.0
            provider_timing["align_detail"] = {"status": "disabled"}
        if self._trace_enabled:
            self._emit(
                f"[whisperx-trace] #{trace_id} align-done: segments={len(aligned)}, "
                f"align_lang={align_lang or 'n/a'}, align_device={self._alignment_device}"
            )
        self._last_diarization_timing = {}
        if self._enable_diarization:
            stage_started_at = time.perf_counter()
            aligned = self._attach_speaker_labels(audio, aligned)
            provider_timing["diarization_seconds"] = time.perf_counter() - stage_started_at
            provider_timing["diarization_detail"] = dict(self._last_diarization_timing)
            stage_started_at = time.perf_counter()
            aligned = self._apply_speaker_profiles(audio, aligned)
            provider_timing["speaker_profile_seconds"] = time.perf_counter() - stage_started_at
        else:
            provider_timing["diarization_seconds"] = 0.0
            provider_timing["diarization_detail"] = {"status": "disabled"}
            provider_timing["speaker_profile_seconds"] = 0.0
        provider_timing["final_segment_count"] = int(len(aligned))

        stage_started_at = time.perf_counter()
        token_meta: list[dict[str, object]] = []
        words_fallback: list[str] = []
        diarized_turns: list[dict[str, object]] = []
        if self._enable_diarization:
            for seg in aligned:
                if not isinstance(seg, dict):
                    continue
                # Use local diarization label first for turn detection. Profile
                # identity can merge speakers across windows.
                speaker = self._resolve_segment_speaker(seg, prefer_profile=False)
                if not speaker:
                    continue
                try:
                    start = float(seg.get("start"))
                    end = float(seg.get("end"))
                except Exception:
                    continue
                diarized_turns.append(
                    {
                        "start": start,
                        "end": end,
                        "speaker": speaker,
                    }
                )
        text = self._format_diarized_text(aligned) if self._enable_diarization else ""
        if not text:
            text = " ".join((str(seg.get("text") or "").strip() for seg in aligned if str(seg.get("text") or "").strip())).strip()
        require_profile_identity = self._require_profile_identity_for_display()
        for seg in aligned:
            seg_speaker = (
                self._resolve_display_segment_speaker(seg)
                if isinstance(seg, dict)
                else None
            )
            local_seg_speaker = (
                self._resolve_segment_speaker(seg, prefer_profile=False)
                if isinstance(seg, dict)
                else None
            )
            for wd in (seg.get("words") or []):
                try:
                    start = float(wd.get("start"))
                    end = float(wd.get("end"))
                    score = float(wd.get("score")) if wd.get("score") is not None else 0.0
                    word_txt = str(wd.get("word") or "").strip()
                    local_word_speaker = self._resolve_word_speaker(
                        wd,
                        seg if isinstance(seg, dict) else {},
                        local_seg_speaker,
                        prefer_profile=False,
                    )
                    profile_word_speaker = self._resolve_word_speaker(
                        wd,
                        seg if isinstance(seg, dict) else {},
                        seg_speaker,
                        prefer_profile=True,
                        require_profile=require_profile_identity,
                    )
                    token_meta.append(
                        {
                            "start": start,
                            "end": end,
                            "score": score,
                            "word": word_txt,
                            "speaker": profile_word_speaker,
                            "profile_speaker": profile_word_speaker,
                            "local_speaker": local_word_speaker,
                        }
                    )
                    if word_txt:
                        words_fallback.append(word_txt)
                except Exception:
                    continue
        if not text and words_fallback:
            text = " ".join(words_fallback).strip()
        if not text:
            text = str(result.get("text") or "").strip()
        stable = [tok for tok in token_meta if tok.get("score", 0.0) >= 0.60 and 0.02 <= (tok.get("end", 0.0) - tok.get("start", 0.0)) <= 1.2]
        marker_count = self._count_speaker_markers(text)
        provider_timing["meta_build_seconds"] = time.perf_counter() - stage_started_at
        provider_timing["token_count"] = int(len(token_meta))
        provider_timing["stable_token_count"] = int(len(stable))
        provider_timing["speaker_turn_count"] = int(max(0, marker_count))
        provider_timing["total_seconds"] = time.perf_counter() - provider_started_at
        self._last_transcription_meta = {
            "stability_ratio": float(len(stable) / max(1, len(token_meta))) if token_meta else 1.0,
            "token_count": int(len(token_meta)),
            "stable_token_count": int(len(stable)),
            "token_timestamps": token_meta[:512],
            "detected_language": str(lang_detected or ""),
            "alignment_language": str(align_lang or ""),
            "speaker_turns": diarized_turns[:128],
            "speaker_turn_count": int(max(0, marker_count)),
            "speaker_profile_stats": dict(self._last_speaker_profile_stats),
            "provider_timing": provider_timing,
        }
        if self._enable_diarization and (self._trace_enabled or marker_count > 0 or len(diarized_turns) > 0):
            unique_speakers = sorted(
                {
                    str(item.get("speaker") or "").strip()
                    for item in diarized_turns
                    if str(item.get("speaker") or "").strip()
                }
            )
            pending_label = str(self._speaker_switch_pending_label or "")
            self._emit(
                "[speaker-turn] diarization summary: "
                f"segment_turns={len(diarized_turns)}; text_markers={marker_count}; "
                f"token_count={len(token_meta)}; speakers={','.join(unique_speakers) or 'n/a'}; "
                f"pending={pending_label or 'none'};"
                f"{int(self._speaker_switch_pending_count)}/"
                f"{float(self._speaker_switch_pending_duration_seconds):.2f}s"
            )
        if zh_script is not None:
            stage_started_at = time.perf_counter()
            text = normalize_chinese_script(text, zh_script)
            provider_timing["script_normalize_seconds"] = time.perf_counter() - stage_started_at
            provider_timing["total_seconds"] = time.perf_counter() - provider_started_at
            self._last_transcription_meta["provider_timing"] = provider_timing
        else:
            provider_timing["script_normalize_seconds"] = 0.0
        if not text:
            self._emit("WhisperX produced empty text after alignment/postprocess.")
        elif self._trace_enabled:
            self._emit(f"[whisperx-trace] #{trace_id} text-done: chars={len(text)}")
        return text

    def prewarm(self, language: Optional[str] = None) -> None:
        """Preload alignment/diarization assets without waiting for first spoken segment."""
        if self._enable_forced_alignment:
            (lang_hint, _) = normalize_language_hint(language)
            align_lang = self._resolve_alignment_language(lang_hint, "")
            if align_lang:
                self._ensure_alignment_model_loaded(align_lang)
        if self._enable_diarization:
            self._ensure_diarization_pipeline_loaded()
            if self._speaker_identity_engine is not None:
                self._speaker_identity_engine.prewarm()

    def reconcile_speaker_profiles(self, *, threshold: float | None = None) -> dict[str, object]:
        if self._speaker_identity_engine is None:
            return {
                "enabled": bool(self._speaker_profile_enabled),
                "status": "skip_engine_unavailable",
                "merged_count": 0,
                "remap": {},
            }
        return self._speaker_identity_engine.reconcile_profiles(threshold=threshold)

    def _transcribe_with_compat(self, audio, kwargs: dict[str, object]):
        active = dict(kwargs)
        for _ in range(6):
            try:
                return self._model.transcribe(audio, **active)
            except TypeError as exc:
                msg = str(exc)
                m = re.search(r"unexpected keyword argument '([^']+)'", msg)
                if not m:
                    raise
                bad = m.group(1)
                if bad in active:
                    self._emit(f"WhisperX compatibility: dropping unsupported kwarg '{bad}'")
                    active.pop(bad, None)
                    continue
                raise
        return self._model.transcribe(audio)

    def _resolve_alignment_language(self, lang_hint: Optional[str], lang_detected: str) -> str:
        mode = (self._alignment_language or 'auto').strip().lower()

        def _norm(token: str) -> str:
            value = (token or '').strip().lower()
            if value in {'zh-hant', 'zh-hans'}:
                return 'zh'
            return value

        if mode == 'follow-source':
            # When source language is set to auto/empty, follow-source should
            # still fall back to detected language so alignment/timestamps stay available.
            preferred = _norm(lang_hint or self._source_language_hint or '')
            if preferred:
                return preferred
            return _norm(lang_detected or '')
        if mode not in {'', 'auto'}:
            return _norm(mode)
        return _norm(lang_hint or self._source_language_hint or lang_detected or '')


    @staticmethod
    def _resolve_whisperx_vad_method(method: str) -> str:
        token = (method or 'silero-vad').strip().lower()
        if token in {'silero', 'silero-vad'}:
            return 'silero'
        return 'pyannote'

    def _emit_language_route(self, source_language: Optional[str], lang_hint: Optional[str], lang_detected: str, align_lang: str) -> None:
        mode = (self._alignment_language or 'auto').strip().lower()
        key = (
            (source_language or 'auto').strip().lower(),
            (lang_hint or '').strip().lower(),
            (lang_detected or '').strip().lower(),
            (align_lang or '').strip().lower(),
        )
        if key in self._language_route_logged:
            return
        self._language_route_logged.add(key)
        self._emit(
            "WhisperX language routing: "
            f"source={key[0] or 'auto'}; asr_hint={key[1] or 'auto'}; "
            f"detected={key[2] or 'n/a'}; align_mode={mode}; align={key[3] or 'n/a'}"
        )
    def _align_segments(self, audio, segments: list[dict], language_code: str, *, trace_id: int | None = None) -> list[dict]:
        detail_started_at = time.perf_counter()
        self._last_alignment_timing = {
            "status": "start",
            "input_segment_count": int(len(segments or [])),
            "language": str(language_code or ""),
            "device": str(self._alignment_device),
        }
        if not segments or not language_code:
            self._last_alignment_timing.update(
                {
                    "status": "skipped_no_segments_or_language",
                    "total_seconds": time.perf_counter() - detail_started_at,
                }
            )
            return segments
        try:
            stage_started_at = time.perf_counter()
            cached = self._ensure_alignment_model_loaded(language_code)
            self._last_alignment_timing["model_load_seconds"] = time.perf_counter() - stage_started_at
            if cached is None:
                self._last_alignment_timing.update(
                    {
                        "status": "skipped_model_unavailable",
                        "total_seconds": time.perf_counter() - detail_started_at,
                    }
                )
                return segments
            (align_model, metadata) = cached
            stage_started_at = time.perf_counter()
            audio_f32 = np.asarray(audio, dtype=np.float32)
            if audio_f32.ndim != 1:
                audio_f32 = audio_f32.reshape(-1)
            if not audio_f32.flags["C_CONTIGUOUS"]:
                audio_f32 = np.ascontiguousarray(audio_f32)
            if not np.isfinite(audio_f32).all():
                audio_f32 = np.nan_to_num(audio_f32, nan=0.0, posinf=1.0, neginf=-1.0)

            audio_duration = float(audio_f32.size) / 16000.0 if audio_f32.size > 0 else 0.0
            segments_for_align = self._sanitize_alignment_segments(segments, audio_duration)
            self._last_alignment_timing["prepare_seconds"] = time.perf_counter() - stage_started_at
            self._last_alignment_timing["clean_segment_count"] = int(len(segments_for_align))
            self._last_alignment_timing["audio_seconds"] = float(audio_duration)
            if not segments_for_align:
                self._last_alignment_timing.update(
                    {
                        "status": "skipped_no_clean_segments",
                        "total_seconds": time.perf_counter() - detail_started_at,
                    }
                )
                return segments

            max_end = max(float(seg.get("end", 0.0) or 0.0) for seg in segments_for_align)
            crop_samples = int(min(audio_duration, max(0.5, max_end + 0.2)) * 16000.0)
            crop_samples = min(max(1, crop_samples), int(audio_f32.size))
            audio_for_align = audio_f32[:crop_samples]
            self._last_alignment_timing["crop_seconds"] = float(crop_samples) / 16000.0
            if self._trace_enabled:
                tid = str(trace_id) if trace_id is not None else "?"
                self._emit(
                    f"[whisperx-trace] #{tid} align-input: lang={language_code}, "
                    f"segments_raw={len(segments)}, segments_clean={len(segments_for_align)}, "
                    f"audio_samples={int(audio_f32.size)}, crop_samples={int(crop_samples)}, "
                    f"audio_sec={audio_duration:.3f}, max_end={max_end:.3f}, "
                    f"align_device={self._alignment_device}"
                )

            if self._alignment_device == "cuda":
                self._sync_cuda_if_available()

            align_started_at = time.perf_counter()
            aligned_result = self._run_whisperx_align(
                segments_for_align,
                align_model,
                metadata,
                audio_for_align,
                self._alignment_device,
            )
            align_run_seconds = time.perf_counter() - align_started_at
            align_elapsed_ms = align_run_seconds * 1000.0
            self._last_alignment_timing["run_seconds"] = float(align_run_seconds)

            if self._alignment_device == "cuda":
                self._sync_cuda_if_available()
            aligned_count = len(list(aligned_result.get("segments", segments) or [])) if isinstance(aligned_result, dict) else len(segments)
            self._last_alignment_timing.update(
                {
                    "status": "ok",
                    "aligned_segment_count": int(aligned_count),
                    "total_seconds": time.perf_counter() - detail_started_at,
                }
            )
            if self._trace_enabled:
                tid = str(trace_id) if trace_id is not None else "?"
                self._align_bench_count += 1
                self._align_bench_total_ms += align_elapsed_ms
                self._align_bench_max_ms = max(self._align_bench_max_ms, align_elapsed_ms)
                avg_ms = self._align_bench_total_ms / max(1, self._align_bench_count)
                self._emit(
                    f"[align-bench] #{tid} device={self._alignment_device} lang={language_code} "
                    f"segments={len(segments_for_align)} audio_sec={audio_duration:.3f} "
                    f"elapsed_ms={align_elapsed_ms:.1f} avg_ms={avg_ms:.1f} max_ms={self._align_bench_max_ms:.1f} "
                    f"count={self._align_bench_count}"
                )
                self._emit(f"[whisperx-trace] #{tid} align-success: aligned_segments={aligned_count}")
            return list(aligned_result.get("segments", segments))
        except Exception as exc:
            self._last_alignment_timing.update(
                {
                    "status": "error",
                    "error": str(exc),
                    "total_seconds": time.perf_counter() - detail_started_at,
                }
            )
            self._emit(f"WhisperX alignment skipped: {exc}")
            return segments

    @staticmethod
    def _sanitize_alignment_segments(segments: list[dict], audio_duration: float) -> list[dict]:
        if audio_duration <= 0.0:
            return []
        cleaned: list[dict] = []
        max_end_allowed = max(0.02, audio_duration - 0.01)
        for raw in segments:
            if not isinstance(raw, dict):
                continue
            text = str(raw.get("text") or "").strip()
            if not text:
                continue
            try:
                start = float(raw.get("start"))
                end = float(raw.get("end"))
            except Exception:
                continue
            start = max(0.0, min(start, max_end_allowed))
            end = max(start + 0.02, min(end, audio_duration))
            if end - start < 0.02:
                continue
            item = dict(raw)
            item["start"] = float(start)
            item["end"] = float(end)
            cleaned.append(item)
        return cleaned

    def _run_whisperx_align(self, segments: list[dict], align_model, metadata, audio_f32: np.ndarray, device: str):
        try:
            import torch  # type: ignore
        except Exception:
            torch = None
        if torch is not None:
            with torch.inference_mode():
                try:
                    with torch.autocast(device_type="cuda", enabled=False):
                        return self._whisperx.align(
                            segments,
                            align_model,
                            metadata,
                            audio_f32,
                            device,
                            interpolate_method="nearest",
                            return_char_alignments=False,
                        )
                except TypeError:
                    return self._whisperx.align(
                        segments,
                        align_model,
                        metadata,
                        audio_f32,
                        device,
                        return_char_alignments=False,
                    )
        try:
            return self._whisperx.align(
                segments,
                align_model,
                metadata,
                audio_f32,
                device,
                interpolate_method="nearest",
                return_char_alignments=False,
            )
        except TypeError:
            return self._whisperx.align(
                segments,
                align_model,
                metadata,
                audio_f32,
                device,
                return_char_alignments=False,
            )

    @staticmethod
    def _sync_cuda_if_available() -> None:
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            return

    def _ensure_alignment_model_loaded(self, language_code: str) -> tuple[object, object] | None:
        if not language_code:
            return None
        self._cleanup_alignment_partial_cache()
        align_repo_id = self._resolve_alignment_repo_id(language_code)
        align_local_dir = self._resolve_alignment_local_dir(language_code, align_repo_id)
        explicit_model = self._alignment_model.strip()
        model_selection = explicit_model if explicit_model else f"auto:{align_repo_id or language_code}"
        cache_key = f"{language_code}|{model_selection.lower()}"
        cached = self._align_cache.get(cache_key)
        if cached is not None:
            return cached

        self._emit(
            f"WhisperX alignment model loading: language={language_code}; "
            f"model={model_selection}; repo={(align_repo_id or 'auto-map')}; cache_key={cache_key}"
        )
        self._prepare_optional_hf_repo_download(
            repo_id=align_repo_id,
            local_dir=align_local_dir,
            provider="whisperx-align",
            model_name=(explicit_model or language_code),
            token=self._resolve_hf_token() or None,
        )
        kwargs: dict[str, object] = {
            "language_code": language_code,
            "device": self._alignment_device,
            "model_dir": str(self._resolve_alignment_model_cache_dir(explicit_model=explicit_model, resolved_model_name="")),
        }
        resolved_model_name = self._resolve_alignment_model_name_for_load(language_code)
        kwargs["model_dir"] = str(
            self._resolve_alignment_model_cache_dir(
                explicit_model=explicit_model,
                resolved_model_name=resolved_model_name,
            )
        )
        if resolved_model_name:
            kwargs["model_name"] = resolved_model_name
            model_selection = resolved_model_name
        expected_external_total = self._estimate_alignment_external_download_total(
            repo_id=align_repo_id,
            local_dir=align_local_dir,
            token=self._resolve_hf_token() or None,
        )
        model_a, metadata = self._run_with_torch_hub_download_progress(
            f"align-{language_code}",
            lambda: self._run_with_external_download_progress(
                f"align-{language_code}",
                lambda: self._whisperx.load_align_model(**kwargs),
                expected_total_bytes=expected_external_total,
            ),
        )
        self._align_cache[cache_key] = (model_a, metadata)
        self._emit(
            f"WhisperX alignment model ready: language={language_code}; "
            f"model={model_selection}; cache_key={cache_key}"
        )
        return self._align_cache.get(cache_key)

    def _resolve_alignment_model_name_for_load(self, language_code: str) -> str:
        align_repo_id = self._resolve_alignment_repo_id(language_code)
        align_local_dir = self._resolve_alignment_local_dir(language_code, align_repo_id)
        explicit_model = self._alignment_model.strip()
        if self._is_local_hf_model_dir_ready(align_local_dir):
            return str(align_local_dir)
        # Legacy fallback: older runs may have cached by repo-id slug.
        legacy_repo_dir = self._alignment_hf_root_dir() / self._slugify_repo_id(align_repo_id)
        if self._is_local_hf_model_dir_ready(legacy_repo_dir):
            return str(legacy_repo_dir)
        # Auto mode fallback: older runs may already cache an alignment model
        # in language-scoped folders (for example `hf/zh`) instead of repo-id
        # slug folders. Reuse that cache before triggering new downloads.
        fallback_lang_dir = self._resolve_alignment_local_dir(language_code, "")
        if self._is_local_hf_model_dir_ready(fallback_lang_dir):
            return str(fallback_lang_dir)
        if explicit_model:
            return explicit_model
        return ""

    def _estimate_alignment_external_download_total(self, *, repo_id: str, local_dir: Path, token: str | None = None) -> int | None:
        if not repo_id or (not self._auto_download):
            return None
        if self._is_local_hf_model_dir_ready(local_dir):
            return None
        total, existing = estimate_hf_files_total(
            repo_id=repo_id,
            local_dir=local_dir,
            allow_patterns=self._hf_alignment_allow_patterns(),
            token=token,
            timeout_seconds=20,
        )
        if total is None or total <= 0:
            return None
        remaining = max(0, int(total) - max(0, int(existing)))
        return remaining or None

    @staticmethod
    def _hf_alignment_allow_patterns() -> list[str]:
        return [
            "*.bin",
            "*.json",
            "*.safetensors",
            "*.model",
            "*.txt",
            "*.yaml",
            "*.yml",
            "*.pt",
            "*.ckpt",
            "*.index",
            "*.onnx",
        ]

    def _alignment_root_dir(self) -> Path:
        root = self._model_root / "align"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _alignment_hf_root_dir(self) -> Path:
        root = self._alignment_root_dir() / "hf"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _alignment_torch_root_dir(self) -> Path:
        root = self._alignment_root_dir() / "torch"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _alignment_cache_root_dir(self) -> Path:
        root = self._alignment_root_dir() / "cache"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _alignment_custom_root_dir(self) -> Path:
        root = self._alignment_root_dir() / "custom"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _normalize_alignment_layout(self) -> None:
        """
        Normalize legacy align cache layout into stable subfolders:
        - align/hf
        - align/torch
        - align/cache
        - align/custom
        """
        align_root = self._alignment_root_dir()
        hf_root = self._alignment_hf_root_dir()
        torch_root = self._alignment_torch_root_dir()
        cache_root = self._alignment_cache_root_dir()
        custom_root = self._alignment_custom_root_dir()

        protected = {
            hf_root.resolve(),
            torch_root.resolve(),
            cache_root.resolve(),
            custom_root.resolve(),
        }
        moved = 0
        skipped = 0

        for child in list(align_root.iterdir()):
            try:
                child_resolved = child.resolve()
            except Exception:
                continue
            if child_resolved in protected:
                continue

            name = child.name
            lower = name.lower()
            target: Path | None = None

            if child.is_file():
                # Torchaudio/fairseq alignment checkpoints.
                if (
                    (lower.endswith(".pth") or lower.endswith(".pt"))
                    and (not lower.endswith(".partial"))
                ):
                    target = torch_root / name
            elif child.is_dir():
                # HF cache layout (huggingface_hub/transformers).
                if lower == ".locks" or lower.startswith("models--"):
                    target = cache_root / name
                # Legacy language-scoped align layout (en/zh/ja/...).
                elif re.fullmatch(r"[a-z]{2,3}(-[a-z0-9]+)?", lower):
                    target = hf_root / name
                # Legacy custom model folder at align root.
                elif self._is_local_hf_model_dir_ready(child):
                    target = custom_root / name

            if target is None:
                continue
            try:
                if target.exists():
                    if child.is_dir() and target.is_dir():
                        merged_moved, merged_skipped = self._merge_alignment_directory(child, target)
                        moved += merged_moved
                        skipped += merged_skipped
                    elif child.is_file() and target.is_file():
                        try:
                            same_size = child.stat().st_size == target.stat().st_size
                        except Exception:
                            same_size = False
                        if same_size:
                            child.unlink(missing_ok=True)
                            moved += 1
                        else:
                            skipped += 1
                    else:
                        skipped += 1
                    continue

                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(child), str(target))
                moved += 1
            except Exception:
                skipped += 1
                continue

        if moved > 0 or skipped > 0:
            self._emit(
                "WhisperX align layout normalized: "
                f"moved={moved}, skipped={skipped}, root={align_root}"
            )

    def _merge_alignment_directory(self, source: Path, target: Path) -> tuple[int, int]:
        moved = 0
        skipped = 0
        for item in list(source.iterdir()):
            dest = target / item.name
            try:
                if item.is_dir():
                    if dest.exists():
                        if dest.is_dir():
                            nested_moved, nested_skipped = self._merge_alignment_directory(item, dest)
                            moved += nested_moved
                            skipped += nested_skipped
                            try:
                                item.rmdir()
                            except Exception:
                                pass
                        else:
                            skipped += 1
                        continue
                    shutil.move(str(item), str(dest))
                    moved += 1
                    continue

                if dest.exists():
                    if dest.is_file():
                        try:
                            same_size = item.stat().st_size == dest.stat().st_size
                        except Exception:
                            same_size = False
                        if same_size:
                            item.unlink(missing_ok=True)
                            moved += 1
                        else:
                            skipped += 1
                    else:
                        skipped += 1
                    continue

                shutil.move(str(item), str(dest))
                moved += 1
            except Exception:
                skipped += 1
                continue

        try:
            source.rmdir()
        except Exception:
            pass
        return moved, skipped

    def _resolve_alignment_model_cache_dir(self, *, explicit_model: str, resolved_model_name: str) -> Path:
        model_name = (resolved_model_name or explicit_model or "").strip()
        if model_name:
            maybe_path = Path(model_name)
            if maybe_path.exists():
                # Local path model: keep downstream cache/temp isolated.
                return self._alignment_cache_root_dir()
            if "/" in model_name:
                # HF repo id (or equivalent path-like id).
                return self._alignment_cache_root_dir()
            upper = model_name.upper()
            if upper.startswith("WAV2VEC2_"):
                return self._alignment_torch_root_dir()
        return self._alignment_torch_root_dir()

    @staticmethod
    def _is_local_hf_model_dir_ready(path: Path) -> bool:
        if not path.exists() or (not path.is_dir()):
            return False
        has_config = any((path / name).exists() for name in ("config.json", "config.yaml", "preprocessor_config.json"))
        has_weights = any((path / name).exists() for name in ("model.safetensors", "pytorch_model.bin", "model.bin"))
        return bool(has_config and has_weights)

    def _attach_speaker_labels(self, audio, segments: list[dict]) -> list[dict]:
        detail_started_at = time.perf_counter()
        self._last_diarization_timing = {
            "status": "start",
            "input_segment_count": int(len(segments or [])),
            "device": str(self._diarization_device),
        }
        if not self._enable_diarization:
            self._last_diarization_timing.update(
                {
                    "status": "disabled",
                    "total_seconds": time.perf_counter() - detail_started_at,
                }
            )
            return segments
        if self._diarization_disabled_reason:
            self._last_diarization_timing.update(
                {
                    "status": "disabled_after_error",
                    "reason": str(self._diarization_disabled_reason),
                    "total_seconds": time.perf_counter() - detail_started_at,
                }
            )
            return segments
        if not segments:
            self._last_diarization_timing.update(
                {
                    "status": "skipped_no_segments",
                    "total_seconds": time.perf_counter() - detail_started_at,
                }
            )
            return segments
        try:
            stage_started_at = time.perf_counter()
            audio_f32 = np.asarray(audio, dtype=np.float32).reshape(-1)
        except Exception:
            audio_f32 = np.zeros((0,), dtype=np.float32)
        self._last_diarization_timing["prepare_seconds"] = time.perf_counter() - stage_started_at
        self._last_diarization_timing["audio_seconds"] = float(audio_f32.size) / 16000.0 if audio_f32.size > 0 else 0.0
        if audio_f32.size < 1600:
            # Skip ultra-short windows to avoid unstable diarization stats on empty/near-empty frames.
            self._last_diarization_timing.update(
                {
                    "status": "skipped_too_short",
                    "total_seconds": time.perf_counter() - detail_started_at,
                }
            )
            return segments
        if (not np.isfinite(audio_f32).all()):
            audio_f32 = np.nan_to_num(audio_f32, nan=0.0, posinf=1.0, neginf=-1.0)
        rms = float(np.sqrt(np.mean(np.square(audio_f32)))) if audio_f32.size > 0 else 0.0
        self._last_diarization_timing["rms"] = float(rms)
        if rms < 1e-4:
            self._last_diarization_timing.update(
                {
                    "status": "skipped_low_rms",
                    "total_seconds": time.perf_counter() - detail_started_at,
                }
            )
            return segments
        try:
            stage_started_at = time.perf_counter()
            self._ensure_diarization_pipeline_loaded()
            self._last_diarization_timing["pipeline_load_seconds"] = time.perf_counter() - stage_started_at

            stage_started_at = time.perf_counter()
            diarize_segments = self._diarization_pipeline(audio_f32)
            self._last_diarization_timing["pipeline_run_seconds"] = time.perf_counter() - stage_started_at
            stage_started_at = time.perf_counter()
            aligned = self._whisperx.assign_word_speakers(diarize_segments, {"segments": segments})
            self._last_diarization_timing["assign_seconds"] = time.perf_counter() - stage_started_at
            output_segments = list((aligned.get("segments", segments) if isinstance(aligned, dict) else segments))
            self._last_diarization_timing.update(
                {
                    "status": "ok",
                    "output_segment_count": int(len(output_segments)),
                    "total_seconds": time.perf_counter() - detail_started_at,
                }
            )
            return output_segments
        except Exception as exc:
            self._diarization_disabled_reason = str(exc)
            self._last_diarization_timing.update(
                {
                    "status": "error",
                    "error": str(exc),
                    "total_seconds": time.perf_counter() - detail_started_at,
                }
            )
            self._emit(f"[download] whisperx-diarization failed: {exc}")
            self._emit(f"WhisperX diarization skipped: {exc}")
            self._emit("WhisperX diarization disabled for this runtime session after initialization failure.")
            return segments

    def _apply_speaker_profiles(self, audio, segments: list[dict]) -> list[dict]:
        if not self._enable_diarization:
            self._last_speaker_profile_stats = {
                "enabled": False,
                "backend": str(self._speaker_profile_backend or "pyannote"),
                "status": "skip_diarization_disabled",
            }
            return segments
        if self._speaker_identity_engine is None:
            self._last_speaker_profile_stats = {
                "enabled": bool(self._speaker_profile_enabled),
                "backend": str(self._speaker_profile_backend or "pyannote"),
                "status": "skip_engine_unavailable",
            }
            return segments
        aligned = self._speaker_identity_engine.apply(
            audio=np.asarray(audio, dtype=np.float32).reshape(-1),
            segments=segments,
            resolve_local_speaker=self._resolve_segment_speaker,
        )
        self._last_speaker_profile_stats = dict(self._speaker_identity_engine.last_stats)
        return aligned

    def _ensure_speaker_embedding_inference_loaded(self) -> None:
        if not self._speaker_profile_enabled:
            return
        if not self._enable_diarization:
            return
        if self._speaker_embedding_disabled_reason:
            return
        if self._speaker_embedding_inference is not None:
            return
        try:
            from pyannote.audio import Inference, Model  # type: ignore
            import torch  # type: ignore
        except Exception as exc:
            self._speaker_embedding_disabled_reason = str(exc)
            self._emit(f"WhisperX speaker-profile disabled: {exc}")
            return

        token = self._resolve_hf_token()
        model_ref = self._speaker_profile_model or "pyannote/embedding"
        model = None
        errors: list[str] = []
        for kwargs in (
            {"token": token},
            {"use_auth_token": token},
            {},
        ):
            try:
                effective_kwargs = {k: v for (k, v) in kwargs.items() if v}
                model = Model.from_pretrained(model_ref, **effective_kwargs)
                break
            except Exception as exc:
                errors.append(str(exc))
                continue
        if model is None:
            self._speaker_embedding_disabled_reason = "; ".join(errors[:3]) or "model load failed"
            self._emit(f"WhisperX speaker-profile disabled: {self._speaker_embedding_disabled_reason}")
            return

        device_name = "cuda" if (self._diarization_device == "cuda" and torch.cuda.is_available()) else "cpu"
        try:
            self._speaker_embedding_inference = Inference(
                model,
                window="whole",
                device=torch.device(device_name),
            )
            self._emit(f"WhisperX speaker-profile embedding ready: model={model_ref}; device={device_name}")
        except Exception as exc:
            self._speaker_embedding_disabled_reason = str(exc)
            self._emit(f"WhisperX speaker-profile disabled: {exc}")

    def _collect_speaker_clip(self, audio_f32: np.ndarray, spans: list[tuple[float, float]]) -> np.ndarray:
        if audio_f32.size == 0 or not spans:
            return np.zeros((0,), dtype=np.float32)
        pieces: list[np.ndarray] = []
        kept_seconds = 0.0
        for (start, end) in sorted(spans):
            s = max(0, int(round(float(start) * 16000.0)))
            e = min(int(audio_f32.size), int(round(float(end) * 16000.0)))
            if e <= s:
                continue
            segment = np.ascontiguousarray(audio_f32[s:e], dtype=np.float32)
            if segment.size <= 0:
                continue
            pieces.append(segment)
            kept_seconds += float(segment.size) / 16000.0
            if kept_seconds >= 8.0:
                break
        if not pieces:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(pieces, axis=0)

    def _extract_speaker_embedding(self, clip: np.ndarray) -> np.ndarray | None:
        inference = self._speaker_embedding_inference
        if inference is None:
            return None
        try:
            import torch  # type: ignore
        except Exception:
            return None
        waveform = torch.from_numpy(np.ascontiguousarray(clip, dtype=np.float32)).unsqueeze(0)
        payloads = (
            {"waveform": waveform, "sample_rate": 16000},
            {"waveform": waveform, "sampling_rate": 16000},
            {"waveform": waveform, "sample_rate": 16000, "sampling_rate": 16000},
        )
        last_error: Exception | None = None
        for payload in payloads:
            try:
                value = inference(payload)
                vector = np.asarray(value, dtype=np.float32).reshape(-1)
                if vector.size > 0 and np.isfinite(vector).all():
                    return vector
            except Exception as exc:
                last_error = exc
                continue
        if last_error is not None and self._trace_enabled:
            self._emit(f"WhisperX speaker embedding extract skipped: {last_error}")
        return None

    def _ensure_diarization_pipeline_loaded(self) -> None:
        if not self._enable_diarization:
            return
        if self._diarization_disabled_reason:
            return
        if self._diarization_pipeline is not None:
            self._warmup_diarization_cuda_context()
            return
        self._emit(f"[download] whisperx-diarization preparing: {self._diarization_model}")
        self._emit("WhisperX diarization model loading...")
        self._sanitize_broken_proxy_env()
        resolved_hf_token = self._resolve_hf_token()
        self._apply_hf_token_env(resolved_hf_token)
        self._cleanup_hf_partial_cache()
        model_name_for_pipeline = self._resolve_diarization_model_for_pipeline(
            model_name=self._diarization_model,
            hf_token=resolved_hf_token,
        )
        self._diarization_pipeline = self._run_with_external_download_progress(
            "diarization",
            lambda: self._create_diarization_pipeline(
                model_name=model_name_for_pipeline,
                device=self._diarization_device,
                hf_token=resolved_hf_token,
            ),
        )
        self._emit("WhisperX diarization pipeline initialized.")
        self._emit("[download] whisperx-diarization ready.")
        self._warmup_diarization_cuda_context()

    def _resolve_diarization_model_for_pipeline(self, *, model_name: str, hf_token: str | None) -> str:
        token = self._normalize_diarization_model_ref(model_name)
        if not token:
            token = "pyannote/speaker-diarization-3.1"
        path_like = ("/" in token or "\\" in token) and Path(token).exists()
        if path_like:
            return token
        if "/" not in token:
            return token

        repo_id = token
        local_repo_dir = self._model_root / "diarization" / self._slugify_repo_id(repo_id)
        self._prepare_optional_hf_repo_download(
            repo_id=repo_id,
            local_dir=local_repo_dir,
            provider="whisperx-diarization",
            model_name=token,
            token=hf_token,
        )
        for dep_repo in self._diarization_dependency_repo_ids(repo_id):
            dep_local_dir = self._model_root / "diarization_deps" / self._slugify_repo_id(dep_repo)
            self._prepare_optional_hf_repo_download(
                repo_id=dep_repo,
                local_dir=dep_local_dir,
                provider="whisperx-diarization",
                model_name=f"{token} -> {dep_repo}",
                token=hf_token,
            )

        if (local_repo_dir / "config.yaml").exists():
            self._emit(f"[download] whisperx-diarization cache ready: {local_repo_dir}")
            return str(local_repo_dir)
        return token

    @staticmethod
    def _normalize_diarization_model_ref(model_name: str) -> str:
        token = str(model_name or "").strip()
        # Older settings may have persisted this typo; normalize before any
        # download or pyannote pipeline construction to avoid a guaranteed 404.
        if token.lower() == "pyannote/speaker-diarization-diarization-3.1":
            return "pyannote/speaker-diarization-3.1"
        return token

    @staticmethod
    def _diarization_dependency_repo_ids(repo_id: str) -> list[str]:
        # Known pyannote diarization pipeline dependencies.
        if repo_id.strip().lower() == "pyannote/speaker-diarization-3.1":
            return ["pyannote/segmentation-3.0", "pyannote/wespeaker-voxceleb-resnet34-lm"]
        return []

    def _resolve_hf_token(self) -> str | None:
        if self._hf_token.strip():
            return self._hf_token.strip()
        for env_key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
            value = str(os.environ.get(env_key, "") or "").strip()
            if value:
                return value
        return None

    def _resolve_speaker_profile_store_path(self, raw_path: str) -> str:
        token = str(raw_path or "").strip()
        if token:
            try:
                path = Path(token)
                if not path.is_absolute():
                    path = (Path(__file__).resolve().parents[2] / path).resolve()
                return str(path)
            except Exception:
                pass
        default_path = Path(__file__).resolve().parents[2] / "speaker_profiles" / "profiles.json"
        return str(default_path)

    def _reset_speaker_profile_store_on_startup(self) -> None:
        path = Path(self._speaker_profile_store_path)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        removed: list[str] = []
        errors: list[str] = []
        for target in (path, temp_path):
            try:
                if target.exists():
                    target.unlink()
                    removed.append(str(target))
            except Exception as exc:
                errors.append(f"{target}: {exc}")
        if removed:
            self._emit(f"WhisperX speaker profile reset on startup: removed={len(removed)}")
        if errors:
            self._emit(f"WhisperX speaker profile reset warning: {' | '.join(errors[:2])}")

    @staticmethod
    def _configure_hf_cache_env(model_root: Path) -> None:
        hf_home = model_root / "hf-home"
        hf_hub_cache = hf_home / "hub"
        hf_home.mkdir(parents=True, exist_ok=True)
        hf_hub_cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(hf_home)
        os.environ["HF_HUB_CACHE"] = str(hf_hub_cache)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_hub_cache)

    @staticmethod
    def _apply_hf_token_env(hf_token: str | None) -> None:
        value = str(hf_token or "").strip()
        if not value:
            return
        os.environ["HF_TOKEN"] = value
        os.environ["HUGGING_FACE_HUB_TOKEN"] = value
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = value

    def _cleanup_hf_partial_cache(self) -> None:
        cache_root_raw = os.environ.get("HF_HUB_CACHE", "") or str(self._model_root / "hf-home" / "hub")
        try:
            cache_root = Path(cache_root_raw).expanduser().resolve()
            model_root = self._model_root.resolve()
        except Exception:
            return
        try:
            cache_root.relative_to(model_root)
        except Exception:
            # Never clean arbitrary external paths.
            return
        if not cache_root.exists():
            return

        removed = 0
        for path in cache_root.rglob("*"):
            name = path.name.lower()
            is_tmp_dir = path.is_dir() and name.startswith("tmp_")
            is_partial_file = path.is_file() and (name.endswith(".incomplete") or name.endswith(".lock"))
            if not (is_tmp_dir or is_partial_file):
                continue
            try:
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)
                removed += 1
            except Exception:
                continue
        if removed > 0:
            self._emit(f"[download] whisperx-diarization cache cleanup: removed {removed} partial temp files")

    def _cleanup_alignment_partial_cache(self) -> None:
        align_root = self._alignment_root_dir()
        removed = 0
        failed = 0
        for path in align_root.rglob("*"):
            name = path.name.lower()
            is_tmp_dir = path.is_dir() and name.startswith("tmp_")
            in_locks_dir = ".locks" in (part.lower() for part in path.parts)
            is_partial_file = path.is_file() and (
                name.endswith(".partial")
                or name.endswith(".incomplete")
                or name.endswith(".lock")
            )
            if not (is_tmp_dir or is_partial_file or in_locks_dir):
                continue
            try:
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)
                removed += 1
            except Exception:
                failed += 1
                continue
        if removed > 0 or failed > 0:
            self._emit(
                "[download] whisperx-align cache cleanup: "
                f"removed={removed}, failed={failed}"
            )

    def _sanitize_broken_proxy_env(self) -> None:
        proxy_keys = (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
            "GIT_HTTP_PROXY",
            "GIT_HTTPS_PROXY",
        )
        cleared: list[str] = []
        for key in proxy_keys:
            value = str(os.environ.get(key, "") or "").strip().lower()
            if not value:
                continue
            if "127.0.0.1:9" in value or "localhost:9" in value:
                os.environ.pop(key, None)
                cleared.append(key)
        if cleared:
            self._emit(
                "[download] whisperx-diarization proxy bypass: "
                f"cleared invalid proxy env keys={','.join(cleared)}"
            )

    def _warmup_diarization_cuda_context(self) -> None:
        if self._diarization_cuda_warmed:
            return
        if self._diarization_device != "cuda":
            self._diarization_cuda_warmed = True
            return
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                _ = torch.empty((1,), device="cuda")
                torch.cuda.synchronize()
        except Exception as exc:
            self._emit(f"WhisperX diarization CUDA warmup skipped: {exc}")
        finally:
            self._diarization_cuda_warmed = True

    def _create_diarization_pipeline(self, *, model_name: str, device: str, hf_token: str | None):
        candidates: list[object] = []
        direct_cls = getattr(self._whisperx, "DiarizationPipeline", None)
        if callable(direct_cls):
            candidates.append(direct_cls)
        try:
            from whisperx.diarize import DiarizationPipeline as module_cls  # type: ignore
            if callable(module_cls):
                candidates.append(module_cls)
        except Exception:
            pass
        if not candidates:
            raise RuntimeError("DiarizationPipeline class is unavailable in current whisperx installation.")

        errors: list[str] = []
        variant_templates = [
            {"model_name": model_name, "device": device, "token": hf_token},
            {"model_name": model_name, "device": device, "use_auth_token": hf_token},
            {"model_name": model_name, "device": device},
            {"device": device, "token": hf_token},
            {"device": device, "use_auth_token": hf_token},
            {"device": device},
        ]
        for cls in candidates:
            for kwargs in variant_templates:
                filtered = {key: value for (key, value) in kwargs.items() if value not in {None, ""}}
                try:
                    return self._construct_with_compat(cls, filtered)
                except TypeError as exc:
                    errors.append(f"{cls.__name__} kwargs={sorted(filtered.keys())}: {exc}")
                    continue
                except Exception as exc:
                    errors.append(f"{cls.__name__} runtime={type(exc).__name__}: {exc}")
                    continue
        raise RuntimeError("Unable to initialize WhisperX diarization pipeline. " + " | ".join(errors[:4]))

    @staticmethod
    def _construct_with_compat(cls, kwargs: dict[str, object]):
        active = dict(kwargs)
        for _ in range(6):
            try:
                return cls(**active)
            except TypeError as exc:
                message = str(exc)
                match = re.search(r"unexpected keyword argument '([^']+)'", message)
                if not match:
                    raise
                bad_key = match.group(1)
                if bad_key not in active:
                    raise
                active.pop(bad_key, None)
                continue
        return cls(**active)

    def _format_diarized_text(self, segments: list[dict]) -> str:
        def _to_float(value: object, fallback: float = 0.0) -> float:
            try:
                return float(value)
            except Exception:
                return fallback

        turns: list[str] = []
        prev_speaker = self._last_speaker_label
        pending_speaker = str(self._speaker_switch_pending_label or "")
        pending_count = int(max(0, self._speaker_switch_pending_count))
        pending_duration = float(max(0.0, self._speaker_switch_pending_duration_seconds))
        switch_events: list[str] = []
        pending_trace: list[str] = []
        saw_any_speaker = False
        for raw in segments:
            if not isinstance(raw, dict):
                continue
            text = str(raw.get("text") or "").strip()
            if not text:
                continue
            speaker = self._resolve_display_segment_speaker(raw)
            display_speaker = prev_speaker
            if speaker:
                saw_any_speaker = True
                segment_duration = max(0.0, _to_float(raw.get("end"), 0.0) - _to_float(raw.get("start"), 0.0))
                if not prev_speaker:
                    prev_speaker = speaker
                    pending_speaker = ""
                    pending_count = 0
                    pending_duration = 0.0
                elif speaker == prev_speaker:
                    pending_speaker = ""
                    pending_count = 0
                    pending_duration = 0.0
                else:
                    if speaker == pending_speaker:
                        pending_count += 1
                        pending_duration += segment_duration
                    else:
                        pending_speaker = speaker
                        pending_count = 1
                        pending_duration = segment_duration
                    single_segment_ready = (
                        pending_count >= 1
                        and segment_duration >= float(self._speaker_switch_single_segment_min_duration_seconds)
                    )
                    if (
                        (
                            pending_count >= int(self._speaker_switch_confirm_segments)
                            and pending_duration >= float(self._speaker_switch_min_duration_seconds)
                        )
                        or single_segment_ready
                    ):
                        previous = str(prev_speaker or "")
                        prev_speaker = pending_speaker
                        switch_events.append(
                            "[speaker-turn] speaker switch confirmed: "
                            f"{previous or 'n/a'} -> {prev_speaker}; "
                            f"confirm_segments={pending_count}; duration={pending_duration:.2f}s; "
                            f"single_segment_ready={str(single_segment_ready).lower()}"
                        )
                        pending_speaker = ""
                        pending_count = 0
                        pending_duration = 0.0
                    else:
                        pending_trace.append(
                            "[speaker-turn] switch pending: "
                            f"{prev_speaker} -> {pending_speaker}; "
                            f"pending={pending_count}/{pending_duration:.2f}s; "
                            f"need={int(self._speaker_switch_confirm_segments)}/"
                            f"{float(self._speaker_switch_min_duration_seconds):.2f}s; "
                            f"single_segment_need={float(self._speaker_switch_single_segment_min_duration_seconds):.2f}s; "
                            f"last_segment={segment_duration:.2f}s"
                        )
                display_speaker = prev_speaker
            if display_speaker:
                marker = self._speaker_label_to_marker(display_speaker)
                is_first_marker = (not turns) and (self._last_speaker_label is None)
                if is_first_marker or (display_speaker != self._last_speaker_label):
                    if marker == ">>":
                        turns.append(f"\n{marker} {text}" if turns else f"{marker} {text}")
                    else:
                        turns.append(f"\n{marker} {text}" if turns else f"{marker} {text}")
                    self._last_speaker_label = display_speaker
                else:
                    turns.append(text)
            else:
                turns.append(text)
        if prev_speaker:
            self._last_speaker_label = prev_speaker
        if saw_any_speaker:
            self._speaker_switch_pending_label = str(pending_speaker or "")
            self._speaker_switch_pending_count = int(max(0, pending_count))
            self._speaker_switch_pending_duration_seconds = float(max(0.0, pending_duration))
        else:
            self._speaker_switch_pending_label = ""
            self._speaker_switch_pending_count = 0
            self._speaker_switch_pending_duration_seconds = 0.0
        if switch_events:
            for row in switch_events[:4]:
                self._emit(row)
        elif pending_trace:
            self._emit(pending_trace[-1])
        merged = " ".join(turns).strip()
        # Keep speaker markers line-start anchored even when segment stitching
        # introduces inline spacing around the marker.
        marker_pattern = r"(>>|S\d+:|\[spk_\d+\])"
        merged = re.sub(rf"([^\n])\s*{marker_pattern}\s*", r"\1\n\2 ", merged, flags=re.IGNORECASE)
        merged = re.sub(rf"^\s*{marker_pattern}\s*", r"\1 ", merged, flags=re.IGNORECASE)
        merged = re.sub(rf"\n\s*{marker_pattern}\s*", r"\n\1 ", merged, flags=re.IGNORECASE)
        lines = []
        for line in merged.splitlines():
            cleaned = re.sub(r"[ \t]+", " ", line).strip()
            if cleaned:
                lines.append(cleaned)
        return "\n".join(lines).strip()

    def _speaker_label_to_marker(self, speaker_label: str | None) -> str:
        label = str(speaker_label or "").strip()
        if not label:
            return ""
        if self._speaker_marker_style == "arrow":
            return ">>"
        return f"[{self._speaker_to_display_label(label).lower()}]"

    def _speaker_to_display_label(self, speaker_label: str) -> str:
        label = str(speaker_label or "").strip()
        if not label:
            return ""
        existing = self._speaker_display_map.get(label)
        if existing:
            return existing
        match = re.search(r"(\d+)$", label)
        if match is not None:
            display = f"SPK_{int(match.group(1)):03d}"
        else:
            display = f"SPK_{self._speaker_display_next_index:03d}"
        self._speaker_display_map[label] = display
        self._speaker_display_next_index = max(
            self._speaker_display_next_index + 1,
            int(display.rsplit("_", 1)[-1]) + 1,
        )
        return display

    @staticmethod
    def _count_speaker_markers(text: str) -> int:
        if not text:
            return 0
        return len(re.findall(r"(?m)^\s*(?:>>|S\d+:|\[spk_\d+\])\s+", text, flags=re.IGNORECASE))

    def _require_profile_identity_for_display(self) -> bool:
        if not bool(getattr(self, "_enable_diarization", False)):
            return False
        if not bool(getattr(self, "_speaker_profile_enabled", False)):
            return False
        if getattr(self, "_speaker_identity_engine", None) is None:
            return False
        stats = getattr(self, "_last_speaker_profile_stats", {})
        status = str((stats if isinstance(stats, dict) else {}).get("status") or "").strip().lower()
        if status in {
            "skip_engine_unavailable",
            "skip_backend_unavailable",
            "skip_disabled",
            "skip_store_unavailable",
        }:
            return False
        if status.startswith("skip_backend") or status.startswith("skip_engine"):
            return False
        return True

    def _resolve_display_segment_speaker(self, segment: dict) -> str | None:
        if self._require_profile_identity_for_display():
            profile_speaker = str(segment.get("profile_speaker") or "").strip()
            if profile_speaker:
                return profile_speaker
            words = segment.get("words")
            if isinstance(words, list):
                counts: dict[str, int] = {}
                for item in words:
                    if not isinstance(item, dict):
                        continue
                    label = str(item.get("profile_speaker") or "").strip()
                    if label:
                        counts[label] = counts.get(label, 0) + 1
                if counts:
                    return max(counts.items(), key=lambda pair: pair[1])[0]
            return None
        return self._resolve_segment_speaker(segment, prefer_profile=False)

    @staticmethod
    def _resolve_segment_speaker(segment: dict, *, prefer_profile: bool = True) -> str | None:
        profile_speaker = str(segment.get("profile_speaker") or "").strip()
        local_speaker = str(segment.get("speaker") or "").strip()
        if prefer_profile:
            if profile_speaker:
                return profile_speaker
            if local_speaker:
                return local_speaker
        else:
            if local_speaker:
                return local_speaker
            if profile_speaker:
                return profile_speaker
        words = segment.get("words")
        if not isinstance(words, list):
            return None
        counts_profile: dict[str, int] = {}
        counts_local: dict[str, int] = {}
        for item in words:
            if not isinstance(item, dict):
                continue
            profile_token = str(item.get("profile_speaker") or "").strip()
            if profile_token:
                counts_profile[profile_token] = counts_profile.get(profile_token, 0) + 1
            token = str(item.get("speaker") or "").strip()
            if token:
                counts_local[token] = counts_local.get(token, 0) + 1
        if prefer_profile:
            if counts_profile:
                return max(counts_profile.items(), key=lambda pair: pair[1])[0]
            if counts_local:
                return max(counts_local.items(), key=lambda pair: pair[1])[0]
            return None
        if counts_local:
            return max(counts_local.items(), key=lambda pair: pair[1])[0]
        if counts_profile:
            return max(counts_profile.items(), key=lambda pair: pair[1])[0]
        return None

    @staticmethod
    def _resolve_word_speaker(
        word: dict,
        segment: dict,
        segment_speaker: str | None = None,
        *,
        prefer_profile: bool = False,
        require_profile: bool = False,
    ) -> str:
        if prefer_profile:
            profile_word = str(word.get("profile_speaker") or "").strip()
            if profile_word:
                return profile_word
            if segment_speaker:
                return str(segment_speaker).strip()
            profile_segment = str(segment.get("profile_speaker") or "").strip()
            if profile_segment:
                return profile_segment
            if require_profile:
                return ""
        local_word = str(word.get("speaker") or "").strip()
        if local_word:
            return local_word
        if segment_speaker:
            return str(segment_speaker).strip()
        local_segment = str(segment.get("speaker") or "").strip()
        if local_segment:
            return local_segment
        profile_word = str(word.get("profile_speaker") or "").strip()
        if profile_word:
            return profile_word
        return str(segment.get("profile_speaker") or "").strip()

    def _resolve_stt_model_arg(self, model_ref: str) -> str:
        if not model_ref:
            return "small"
        if "/" in model_ref or "\\" in model_ref:
            return model_ref

        target_dir = self._model_root / "stt" / model_ref
        if self._is_stt_model_dir_ready(target_dir):
            return str(target_dir)
        if target_dir.exists() and (not self._is_stt_model_dir_ready(target_dir)):
            self._emit(f"WhisperX STT model folder exists but incomplete: {target_dir.name}; attempting repair download")

        repo_id = self._resolve_whisper_repo_id(model_ref)
        if repo_id and self._auto_download:
            self._emit(f"WhisperX STT model download started: {model_ref}")
            self._download_stt_model_with_progress(repo_id=repo_id, target_dir=target_dir, model_ref=model_ref)
            if not self._is_stt_model_dir_ready(target_dir):
                raise RuntimeError(
                    f"WhisperX STT model directory is incomplete after download: {target_dir}"
                )
            self._emit(f"WhisperX STT model download completed: {model_ref}")
            return str(target_dir)

        return model_ref

    @staticmethod
    def _is_stt_model_dir_ready(path: Path) -> bool:
        if not path.is_dir():
            return False
        model_bin = path / "model.bin"
        config_json = path / "config.json"
        if not model_bin.exists() or not config_json.exists():
            return False
        min_expected_bytes = 32 * 1024 * 1024
        try:
            return int(model_bin.stat().st_size) >= min_expected_bytes
        except Exception:
            return False

    def _download_stt_model_with_progress(self, *, repo_id: str, target_dir: Path, model_ref: str) -> None:
        allow_patterns = ["config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.*"]
        try:
            download_hf_files_with_progress(
                repo_id=repo_id,
                output_dir=str(target_dir),
                allow_patterns=allow_patterns,
                progress_callback=self._progress_callback,
                provider="whisperx",
                model_name=model_ref,
                timeout_seconds=60,
            )
            return
        except Exception as exc:
            self._emit(f"WhisperX direct download failed: {exc}")
            use_snapshot = str(os.environ.get("VOICE2TEXT_WHISPERX_USE_SNAPSHOT", "")).strip().lower() in {"1", "true", "yes"}
            if not use_snapshot:
                raise
            self._emit("WhisperX snapshot fallback enabled by VOICE2TEXT_WHISPERX_USE_SNAPSHOT=1")
            download_hf_snapshot_with_progress(
                repo_id=repo_id,
                output_dir=str(target_dir),
                allow_patterns=allow_patterns,
                progress_callback=self._progress_callback,
                provider="whisperx",
                model_name=model_ref,
            )

    def _prepare_optional_hf_repo_download(self, *, repo_id: str, local_dir: Path, provider: str, model_name: str, token: str | None = None) -> None:
        if not repo_id or (not self._auto_download):
            return
        if self._is_local_repo_ready_for_predownload(repo_id=repo_id, local_dir=local_dir):
            self._emit(f"[download] {provider} cache hit: {model_name}")
            return
        try:
            download_hf_files_with_progress(
                repo_id=repo_id,
                output_dir=str(local_dir),
                allow_patterns=self._hf_alignment_allow_patterns(),
                progress_callback=self._progress_callback,
                provider=provider,
                model_name=model_name,
                token=token,
                timeout_seconds=60,
            )
        except Exception as exc:
            try:
                download_hf_snapshot_with_progress(
                    repo_id=repo_id,
                    output_dir=str(local_dir),
                    allow_patterns=self._hf_alignment_allow_patterns(),
                    progress_callback=self._progress_callback,
                    provider=provider,
                    model_name=model_name,
                    token=token,
                )
            except Exception as exc2:
                self._emit(f"[download] {provider} skipped: {model_name} ({exc2})")

    @staticmethod
    def _is_local_repo_ready_for_predownload(*, repo_id: str, local_dir: Path) -> bool:
        if not local_dir.exists() or (not local_dir.is_dir()):
            return False

        rid = (repo_id or "").strip().lower()
        has_yaml = any((local_dir / name).exists() for name in ("config.yaml", "config.yml"))
        has_json = any((local_dir / name).exists() for name in ("config.json", "preprocessor_config.json"))

        # pyannote diarization pipeline repos are config-driven and can delegate
        # model artifacts to dependency repos.
        if rid in {
            "pyannote/speaker-diarization-3.1",
            "pyannote/speaker-diarization-community-1",
        }:
            return has_yaml

        has_weight = any(
            (local_dir / name).exists()
            for name in (
                "pytorch_model.bin",
                "model.bin",
                "model.safetensors",
                "weights.ckpt",
                "model.ckpt",
                "model.pt",
                "model.onnx",
            )
        )
        return bool((has_yaml or has_json) and has_weight)

    def _resolve_alignment_repo_id(self, language_code: str) -> str:
        # If user explicitly provided an HF repo id, use it directly.
        if self._alignment_model and ('/' in self._alignment_model):
            return self._alignment_model
        # If user provided a local/custom model name without repo id, skip predownload.
        if self._alignment_model:
            return ''
        try:
            alignment_mod = getattr(self._whisperx, 'alignment', None)
            if alignment_mod is None:
                import whisperx.alignment as alignment_mod  # type: ignore
            if alignment_mod is None:
                return ''
            mapping_hf = getattr(alignment_mod, 'DEFAULT_ALIGN_MODELS_HF', None)
            if isinstance(mapping_hf, dict):
                repo = mapping_hf.get(language_code)
                if isinstance(repo, str) and repo.strip():
                    return repo.strip()
        except Exception:
            return ''
        return ''

    @staticmethod
    def _slugify_repo_id(value: str) -> str:
        token = (value or '').strip().lower()
        if not token:
            return 'auto'
        slug = re.sub(r'[^0-9a-z._-]+', '-', token)
        slug = slug.strip('-')
        return slug or 'auto'

    def _resolve_alignment_local_dir(self, language_code: str, repo_id: str) -> Path:
        base = self._alignment_hf_root_dir()
        if self._alignment_model and ("/" not in self._alignment_model):
            return self._alignment_custom_root_dir() / self._slugify_repo_id(self._alignment_model)

        # For auto/follow-source/explicit-language flows, keep alignment cache
        # in language-scoped folders under `align/hf/{lang}` even when model
        # source is an explicit HF repo id.
        mode = (self._alignment_language or "auto").strip().lower()
        if mode == "follow-source":
            folder_lang = self._normalize_alignment_folder_language(self._source_language_hint)
            if not folder_lang:
                folder_lang = self._normalize_alignment_folder_language(language_code)
        elif mode in {"", "auto"}:
            folder_lang = self._normalize_alignment_folder_language(language_code)
        else:
            folder_lang = self._normalize_alignment_folder_language(mode)
        return base / self._slugify_repo_id(folder_lang or "auto")

    @staticmethod
    def _normalize_alignment_folder_language(value: str | None) -> str:
        token = (value or "").strip().lower()
        if not token:
            return ""
        if token in {"zh-hant", "zh-hans", "zh-tw", "zh-cn", "zh-hk", "zh-sg"}:
            return "zh"
        return token
    @staticmethod
    def _resolve_whisper_repo_id(model_ref: str) -> str | None:
        value = model_ref.strip()
        if not value:
            return None
        if "/" in value:
            return value
        try:
            from faster_whisper import utils as fw_utils  # type: ignore
            mapping = getattr(fw_utils, "_MODELS", None)
            if isinstance(mapping, dict):
                repo = mapping.get(value)
                if isinstance(repo, str) and repo.strip():
                    return repo
        except Exception:
            return None
        return None

    def _build_download_probe_roots(self) -> list[Path]:
        roots: list[Path] = []
        candidates = [
            os.environ.get("HF_HOME", ""),
            os.environ.get("HUGGINGFACE_HUB_CACHE", ""),
            os.environ.get("TRANSFORMERS_CACHE", ""),
            str(Path.home() / ".cache" / "huggingface"),
            str(Path.home() / ".cache" / "torch"),
            str(self._model_root),
        ]
        for raw in candidates:
            if not raw:
                continue
            try:
                path = Path(raw).expanduser().resolve()
            except Exception:
                continue
            if path not in roots:
                roots.append(path)
        return roots

    @staticmethod
    def _safe_dir_size(root: Path) -> int:
        total = 0
        try:
            if not root.exists():
                return 0
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                try:
                    total += int(path.stat().st_size)
                except Exception:
                    continue
        except Exception:
            return total
        return total

    def _current_probe_size(self) -> int:
        return sum(self._safe_dir_size(root) for root in self._download_probe_roots)

    def _run_with_external_download_progress(self, label: str, fn, expected_total_bytes: int | None = None):
        if self._progress_callback is None:
            return fn()

        done = threading.Event()
        before_size = self._current_probe_size()
        with self._external_download_monitor_lock:
            suppress_epoch_start = self._external_download_monitor_suppress_epoch
        state = {"peak_delta": 0, "last_delta": -1, "saw_suppressed": False}

        def monitor() -> None:
            while not done.is_set():
                try:
                    with self._external_download_monitor_lock:
                        suppressed = self._external_download_monitor_suppress_count > 0
                        if self._external_download_monitor_suppress_epoch > suppress_epoch_start:
                            state["saw_suppressed"] = True
                    if suppressed:
                        state["saw_suppressed"] = True
                        done.wait(0.6)
                        continue
                    if state["saw_suppressed"]:
                        # This transfer is already covered by a more specific progress source
                        # (for example torch-hub per-file download); avoid duplicate generic lines.
                        done.wait(0.6)
                        continue
                    now_size = self._current_probe_size()
                    delta = max(0, now_size - before_size)
                    if delta > state["peak_delta"]:
                        state["peak_delta"] = delta
                    if delta <= 0 and state["peak_delta"] <= 0:
                        done.wait(0.6)
                        continue
                    if delta == state["last_delta"]:
                        done.wait(0.6)
                        continue
                    state["last_delta"] = delta
                    display_delta = min(delta, expected_total_bytes) if expected_total_bytes else delta
                    emit_progress(
                        self._progress_callback,
                        format_download_progress("download", f"whisperx:{label}", display_delta, expected_total_bytes),
                    )
                except Exception as exc:
                    emit_progress(self._progress_callback, f"download monitor skipped: {exc}")
                    break
                done.wait(0.6)

        thread = threading.Thread(target=monitor, name=f"whisperx-dl-{label}", daemon=True)
        thread.start()
        try:
            return fn()
        finally:
            done.set()
            thread.join(timeout=1.0)
            with self._external_download_monitor_lock:
                if self._external_download_monitor_suppress_epoch > suppress_epoch_start:
                    state["saw_suppressed"] = True
            if state["saw_suppressed"]:
                return
            after_size = self._current_probe_size()
            final_delta = max(0, after_size - before_size)
            observed_delta = max(state["peak_delta"], final_delta, 0)
            if observed_delta > 0 or final_delta > 0:
                try:
                    display_delta = min(observed_delta, expected_total_bytes) if expected_total_bytes else observed_delta
                    emit_progress(
                        self._progress_callback,
                        format_download_progress("download", f"whisperx:{label}", display_delta, expected_total_bytes),
                    )
                except Exception as exc:
                    emit_progress(self._progress_callback, f"download final-progress skipped: {exc}")

    def _run_with_torch_hub_download_progress(self, label: str, fn):
        if self._progress_callback is None:
            return fn()
        try:
            import torch.hub as torch_hub  # type: ignore
        except Exception:
            return fn()
        original = getattr(torch_hub, "download_url_to_file", None)
        if not callable(original):
            return fn()

        def wrapped(url, dst, *args, **kwargs):
            target = Path(str(dst))
            display_name = f"whisperx:{label}:{target.name or 'file'}"
            expected_total = self._probe_remote_size_with_range(str(url), timeout_seconds=30)

            done = threading.Event()
            state = {"last_bytes": -1}

            with self._external_download_monitor_lock:
                self._external_download_monitor_suppress_count += 1
                self._external_download_monitor_suppress_epoch += 1

            def _current_downloaded_bytes() -> int:
                try:
                    if target.exists():
                        return max(0, int(target.stat().st_size))
                except Exception:
                    pass
                pattern = target.name + ".*.partial"
                best_size = 0
                newest_mtime = -1.0
                try:
                    for part in target.parent.glob(pattern):
                        if not part.is_file():
                            continue
                        stat = part.stat()
                        if stat.st_mtime >= newest_mtime:
                            newest_mtime = float(stat.st_mtime)
                            best_size = max(0, int(stat.st_size))
                except Exception:
                    return 0
                return best_size

            def monitor() -> None:
                while not done.is_set():
                    downloaded = _current_downloaded_bytes()
                    if expected_total is not None:
                        downloaded = min(downloaded, expected_total)
                    if downloaded != state["last_bytes"]:
                        state["last_bytes"] = downloaded
                        emit_progress(
                            self._progress_callback,
                            format_download_progress("download", display_name, downloaded, expected_total),
                        )
                    done.wait(0.5)

            monitor_thread = threading.Thread(target=monitor, name=f"torch-hub-dl-{label}", daemon=True)
            monitor_thread.start()
            try:
                kwargs["progress"] = False
                return original(url, dst, *args, **kwargs)
            finally:
                done.set()
                monitor_thread.join(timeout=1.0)
                with self._external_download_monitor_lock:
                    self._external_download_monitor_suppress_count = max(0, self._external_download_monitor_suppress_count - 1)
                final_size = _current_downloaded_bytes()
                final_downloaded = max(0, final_size)
                if expected_total is not None:
                    final_downloaded = min(final_downloaded, expected_total)
                emit_progress(
                    self._progress_callback,
                    format_download_progress("download", display_name, final_downloaded, expected_total),
                )

        try:
            torch_hub.download_url_to_file = wrapped
            return fn()
        finally:
            torch_hub.download_url_to_file = original

    @staticmethod
    def _probe_remote_size_with_range(url: str, timeout_seconds: int) -> int | None:
        headers = {"User-Agent": "Voice2Text/1.0", "Range": "bytes=0-0"}
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                content_range = response.headers.get("Content-Range")
                if content_range and "/" in content_range:
                    try:
                        value = int(content_range.rsplit("/", 1)[1].strip())
                        if value > 0:
                            return value
                    except Exception:
                        pass
                content_length = response.headers.get("Content-Length")
                if content_length:
                    try:
                        value = int(content_length.strip())
                        if value > 0:
                            return value
                    except Exception:
                        pass
        except Exception:
            return None
        return None
    def _emit(self, message: str) -> None:
        emit_progress(self._progress_callback, message)

    def get_last_transcription_meta(self) -> dict[str, object]:
        return dict(self._last_transcription_meta)

    def _resolve_alignment_device(self) -> str:
        if self._alignment_device_setting in {"cpu", "cuda"}:
            resolved = self._alignment_device_setting
            return self._apply_alignment_device_safety_policy(resolved)
        raw = os.environ.get("VOICE2TEXT_WHISPERX_ALIGN_DEVICE", "auto").strip().lower()
        if raw in {"cpu", "cuda"}:
            return self._apply_alignment_device_safety_policy(raw)
        if self._device != "cuda":
            return self._apply_alignment_device_safety_policy(self._device)
        model_token = (self._model_ref or "").strip().lower()
        if model_token.startswith("large"):
            return self._apply_alignment_device_safety_policy("cpu")
        return self._apply_alignment_device_safety_policy(self._device)

    def _resolve_diarization_device(self) -> str:
        setting = (self._diarization_device_setting or "auto").strip().lower()
        if setting in {"cpu", "cuda"}:
            resolved = setting
        else:
            # Safe default: keep historical behavior and follow ASR device.
            resolved = self._device
        if resolved != "cuda":
            return "cpu"
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        self._emit(
            "WhisperX diarization device downgraded to CPU because torch CUDA is unavailable. "
            f"requested={setting or 'auto'}; asr_device={self._device}"
        )
        return "cpu"

    def _apply_alignment_device_safety_policy(self, resolved: str) -> str:
        """Apply platform-specific guardrails to avoid known WhisperX alignment crashes."""
        token = (resolved or "").strip().lower()
        if token != "cuda":
            return token or "cpu"
        if os.name != "nt":
            return token
        allow_unsafe = str(os.environ.get("VOICE2TEXT_WHISPERX_ALLOW_UNSAFE_CUDA_ALIGN", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if allow_unsafe:
            self._emit("WhisperX alignment CUDA safety guard bypassed by VOICE2TEXT_WHISPERX_ALLOW_UNSAFE_CUDA_ALIGN=1.")
            return token
        self._emit(
            "WhisperX alignment device downgraded to CPU on Windows due known CUDA align access-violation risk. "
            "Set VOICE2TEXT_WHISPERX_ALLOW_UNSAFE_CUDA_ALIGN=1 to force CUDA alignment."
        )
        return "cpu"







