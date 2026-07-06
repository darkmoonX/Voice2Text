"""Live diarization support for the whisper.cpp server backend."""
from __future__ import annotations

from pathlib import Path
import os
import re
import shutil
import time
from typing import Callable

import numpy as np

from ..model_paths import library_model_dir
from .model_download import download_hf_files_with_progress, download_hf_snapshot_with_progress
from .profile_quality import ClipQualityConfig
from .speaker_identity import SpeakerIdentityConfig, SpeakerIdentityEngine


class WhisperCppDiarizer:
    def __init__(
        self,
        *,
        device: str,
        diarization_device: str = "auto",
        source_language_hint: str = "",
        diarization_model: str = "pyannote/speaker-diarization-3.1",
        diarization_min_speakers: int = 0,
        diarization_max_speakers: int = 0,
        hf_token: str = "",
        speaker_profile_enabled: bool = True,
        speaker_profile_backend: str = "pyannote",
        speaker_profile_model: str = "pyannote/embedding",
        speaker_speechbrain_model: str = "speechbrain/spkrec-ecapa-voxceleb",
        speaker_nemo_model: str = "nvidia/speakerverification_en_titanet_large",
        speaker_wespeaker_model: str = "pyannote/wespeaker-voxceleb-resnet34-lm",
        speaker_profile_match_threshold: float = 0.72,
        speaker_profile_min_seconds: float = 2.0,
        speaker_realtime_candidate_seconds: float = 6.0,
        speaker_realtime_candidate_samples: int = 8,
        speaker_realtime_candidate_match_threshold: float = 0.0,
        speaker_realtime_update_match_threshold: float = 0.0,
        speaker_realtime_visible_seconds: float = 24.0,
        speaker_realtime_visible_samples: int = 16,
        speaker_realtime_refresh_alpha: float = 0.5,
        speaker_realtime_refresh_assign_threshold: float = 0.55,
        speaker_realtime_refresh_min_cluster_seconds: float = 4.0,
        speaker_realtime_refresh_merge: bool = True,
        speaker_realtime_refresh_match_mode: str = "argmax",
        speaker_merge_grace_windows: int = 0,
        speaker_merge_grace_relief: float = 0.10,
        speaker_merge_preserve_centroid: bool = False,
        speaker_profile_max_exemplars: int = 1,
        speaker_profile_exemplar_diversity_threshold: float = 0.90,
        speaker_profile_reconcile_threshold: float = 0.52,
        speaker_profile_store_path: str = "",
        speaker_profile_quality_gate_enabled: bool = False,
        speaker_profile_quality_min_confidence: float = 0.45,
        speaker_marker_style: str = "spk",
        speaker_pause_break_seconds: float = 1.8,
        auto_download: bool = True,
        progress_callback: Callable[[str], None] | None = None,
        pipeline_factory: Callable[..., object] | None = None,
        assign_word_speakers: Callable[[object, dict], dict] | None = None,
        speaker_identity_engine: SpeakerIdentityEngine | None = None,
    ) -> None:
        self._device = str(device or "cpu").strip().lower()
        self._diarization_device_setting = str(diarization_device or "auto")
        self._source_language_hint = str(source_language_hint or "")
        self._diarization_model = str(diarization_model or "pyannote/speaker-diarization-3.1").strip()
        self._diar_min_speakers = int(max(0, diarization_min_speakers or 0))
        self._diar_max_speakers = int(max(0, diarization_max_speakers or 0))
        self._hf_token = str(hf_token or "")
        self._speaker_profile_enabled = bool(speaker_profile_enabled)
        self._speaker_profile_backend = str(speaker_profile_backend or "pyannote")
        self._speaker_profile_max_exemplars = int(max(1, speaker_profile_max_exemplars or 1))
        marker_style = str(speaker_marker_style or "").strip().lower()
        self._speaker_marker_style = "arrow" if marker_style in {"arrow", "arrows", ">>"} else "spk"
        self._speaker_pause_break_seconds = float(max(0.0, speaker_pause_break_seconds))
        self._auto_download = bool(auto_download)
        self._progress_callback = progress_callback
        self._model_root = library_model_dir("whisperx")
        self._configure_hf_cache_env(self._model_root)
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self._diarization_device = self._resolve_diarization_device()
        self._diarization_pipeline = None
        self._whole_file_diarization_pipeline = None
        self._pipeline_factory = pipeline_factory
        self._assign_word_speakers = assign_word_speakers
        self._diarization_disabled_reason: str | None = None
        self._diarization_cuda_warmed = False
        self._last_diarization_timing: dict[str, object] = {}
        self._last_speaker_profile_stats: dict[str, object] = {}
        self._speaker_identity_engine = speaker_identity_engine or SpeakerIdentityEngine(
            SpeakerIdentityConfig(
                enabled=bool(speaker_profile_enabled),
                backend=self._speaker_profile_backend,
                store_path=self._resolve_speaker_profile_store_path(speaker_profile_store_path),
                match_threshold=float(speaker_profile_match_threshold),
                min_seconds=float(speaker_profile_min_seconds),
                reconcile_threshold=float(speaker_profile_reconcile_threshold),
                model_root=str(self._model_root),
                device=self._diarization_device,
                hf_token=self._hf_token,
                pyannote_model=str(speaker_profile_model or "pyannote/embedding"),
                speechbrain_model=str(speaker_speechbrain_model or "speechbrain/spkrec-ecapa-voxceleb"),
                nemo_model=str(speaker_nemo_model or "nvidia/speakerverification_en_titanet_large"),
                wespeaker_model=str(speaker_wespeaker_model or "pyannote/wespeaker-voxceleb-resnet34-lm"),
                on_status=progress_callback,
                quality_gate=ClipQualityConfig(
                    enabled=bool(speaker_profile_quality_gate_enabled),
                    min_confidence=float(max(0.0, min(1.0, speaker_profile_quality_min_confidence))),
                ),
                realtime_candidate_seconds=float(speaker_realtime_candidate_seconds),
                realtime_candidate_samples=int(speaker_realtime_candidate_samples),
                realtime_candidate_match_threshold=float(speaker_realtime_candidate_match_threshold),
                realtime_update_match_threshold=float(speaker_realtime_update_match_threshold),
                realtime_visible_seconds=float(speaker_realtime_visible_seconds),
                realtime_visible_samples=int(speaker_realtime_visible_samples),
                realtime_refresh_alpha=float(speaker_realtime_refresh_alpha),
                realtime_refresh_assign_threshold=float(speaker_realtime_refresh_assign_threshold),
                realtime_refresh_min_cluster_seconds=float(speaker_realtime_refresh_min_cluster_seconds),
                realtime_refresh_merge=bool(speaker_realtime_refresh_merge),
                realtime_refresh_match_mode=str(speaker_realtime_refresh_match_mode or "argmax"),
                max_speakers_hint=int(self._diar_max_speakers),
                merge_grace_windows=int(speaker_merge_grace_windows),
                merge_grace_relief=float(speaker_merge_grace_relief),
                merge_preserve_centroid=bool(speaker_merge_preserve_centroid),
                max_exemplars=self._effective_speaker_profile_max_exemplars(),
                exemplar_diversity_threshold=float(speaker_profile_exemplar_diversity_threshold),
            )
        )
        self._last_speaker_label: str | None = None
        self._speaker_display_map: dict[str, str] = {}
        self._speaker_display_next_index = 0
        self._speaker_switch_confirm_segments = 2
        self._speaker_switch_min_duration_seconds = 0.18
        self._speaker_switch_single_segment_min_duration_seconds = 0.25
        self._speaker_switch_pending_label = ""
        self._speaker_switch_pending_count = 0
        self._speaker_switch_pending_duration_seconds = 0.0
        self._last_speaker_segment_end: float | None = None

    @property
    def speaker_profile_stats(self) -> dict[str, object]:
        return dict(self._last_speaker_profile_stats)

    def prewarm(self) -> None:
        self._ensure_diarization_pipeline_loaded()
        if self._speaker_identity_engine is not None:
            self._speaker_identity_engine.prewarm()

    def apply(self, audio: np.ndarray, segments: list[dict]) -> list[dict]:
        labeled = self._attach_speaker_labels(audio, segments)
        profiled = self._apply_speaker_profiles(audio, labeled)
        return self._finalize_word_speaker_labels(profiled)

    def _finalize_word_speaker_labels(self, segments: list[dict]) -> list[dict]:
        """Resolve each word's final speaker/profile_speaker/local_speaker fields.

        `_attach_speaker_labels` only sets the raw per-window diarization label
        (`speaker`) and `_apply_speaker_profiles` only adds the cross-window profile id
        (`profile_speaker`), mirroring `assign_word_speakers`/`SpeakerIdentityEngine.apply`'s
        own field ownership. Downstream (`subtitle_assembler._WordState`) renders
        `raw.get('speaker') or raw.get('profile_speaker')` as the DISPLAY speaker, so it must
        receive the profile-preferring resolved value here (matching
        `whisperx_provider.py`'s token_meta construction) rather than the raw unstable
        per-window label, and `local_speaker` must be populated separately for the
        assembler's onset-split heuristic (`_dominant_local_after`/`_dominant_local_before`).
        """
        require_profile_identity = self._require_profile_identity_for_display()
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            seg_speaker = self._resolve_display_segment_speaker(seg)
            local_seg_speaker = self._resolve_segment_speaker(seg, prefer_profile=False)
            for wd in seg.get("words") or []:
                if not isinstance(wd, dict):
                    continue
                local_word_speaker = self._resolve_word_speaker(
                    wd, seg, local_seg_speaker, prefer_profile=False
                )
                profile_word_speaker = self._resolve_word_speaker(
                    wd,
                    seg,
                    seg_speaker,
                    prefer_profile=True,
                    require_profile=require_profile_identity,
                )
                wd["speaker"] = profile_word_speaker
                wd["profile_speaker"] = profile_word_speaker
                wd["local_speaker"] = local_word_speaker
        return segments

    def format_display_text(self, segments: list[dict]) -> str:
        return self._format_diarized_text(segments)

    def build_speaker_turns(self, segments: list[dict]) -> list[dict[str, object]]:
        turns: list[dict[str, object]] = []
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            speaker = self._resolve_segment_speaker(seg, prefer_profile=False)
            if not speaker:
                continue
            try:
                start = float(seg.get("start"))
                end = float(seg.get("end"))
            except Exception:
                continue
            turns.append({"start": start, "end": end, "speaker": speaker})
        return turns[:128]

    def _attach_speaker_labels(self, audio, segments: list[dict]) -> list[dict]:
        detail_started_at = time.perf_counter()
        self._last_diarization_timing = {
            "status": "start",
            "input_segment_count": int(len(segments or [])),
            "device": str(self._diarization_device),
        }
        if self._diarization_disabled_reason:
            self._last_diarization_timing.update(
                {"status": "disabled_after_error", "reason": self._diarization_disabled_reason}
            )
            return segments
        if not segments:
            self._last_diarization_timing.update({"status": "skipped_no_segments"})
            return segments
        try:
            audio_f32 = np.asarray(audio, dtype=np.float32).reshape(-1)
        except Exception:
            audio_f32 = np.zeros((0,), dtype=np.float32)
        self._last_diarization_timing["audio_seconds"] = float(audio_f32.size) / 16000.0 if audio_f32.size > 0 else 0.0
        if audio_f32.size < 1600:
            self._last_diarization_timing.update({"status": "skipped_too_short"})
            return segments
        if not np.isfinite(audio_f32).all():
            audio_f32 = np.nan_to_num(audio_f32, nan=0.0, posinf=1.0, neginf=-1.0)
        rms = float(np.sqrt(np.mean(np.square(audio_f32)))) if audio_f32.size > 0 else 0.0
        self._last_diarization_timing["rms"] = float(rms)
        if rms < 1e-4:
            self._last_diarization_timing.update({"status": "skipped_low_rms"})
            return segments
        try:
            self._ensure_diarization_pipeline_loaded()
            diarize_segments = self._diarization_pipeline(audio_f32, **self._diar_speaker_count_kwargs())
            assign = self._resolve_assign_word_speakers()
            aligned = assign(diarize_segments, {"segments": segments})
            output_segments = list((aligned.get("segments", segments) if isinstance(aligned, dict) else segments))
            self._last_diarization_timing.update(
                {"status": "ok", "output_segment_count": int(len(output_segments))}
            )
            return output_segments
        except Exception as exc:
            self._diarization_disabled_reason = str(exc)
            self._last_diarization_timing.update({"status": "error", "error": str(exc)})
            self._emit(f"[download] whispercpp-diarization failed: {exc}")
            self._emit(f"whisper.cpp diarization skipped: {exc}")
            self._emit("whisper.cpp diarization disabled for this runtime session after initialization failure.")
            return segments
        finally:
            self._last_diarization_timing["total_seconds"] = time.perf_counter() - detail_started_at

    def _apply_speaker_profiles(self, audio, segments: list[dict]) -> list[dict]:
        if self._speaker_identity_engine is None:
            self._last_speaker_profile_stats = {
                "enabled": bool(self._speaker_profile_enabled),
                "backend": self._speaker_profile_backend,
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

    def _ensure_diarization_pipeline_loaded(self) -> None:
        if self._diarization_disabled_reason:
            return
        if self._diarization_pipeline is not None:
            self._warmup_diarization_cuda_context()
            return
        self._emit(f"[download] whispercpp-diarization preparing: {self._diarization_model}")
        self._emit("whisper.cpp diarization model loading...")
        self._sanitize_broken_proxy_env()
        resolved_hf_token = self._resolve_hf_token()
        self._apply_hf_token_env(resolved_hf_token)
        self._cleanup_hf_partial_cache()
        model_name = self._resolve_diarization_model_for_pipeline(
            model_name=self._diarization_model,
            hf_token=resolved_hf_token,
        )
        self._diarization_pipeline = self._create_diarization_pipeline(
            model_name=model_name,
            device=self._diarization_device,
            hf_token=resolved_hf_token,
        )
        self._emit("whisper.cpp diarization pipeline initialized.")
        self._emit("[download] whispercpp-diarization ready.")
        self._warmup_diarization_cuda_context()

    def _ensure_whole_file_diarization_pipeline(self):
        """Return a CPU-pinned diarization pipeline for the whole-file direct/import pass.

        Round 0066: a single continuous whole-file diarization pass is a large sustained
        GPU compute burst with known crash risk under sustained load (round 0041), so this
        is deliberately pinned to CPU regardless of the live `diarization_device` setting,
        mirroring `whisperx_provider.py`'s `_ensure_whole_file_diarization_pipeline`. Reuses
        the live pipeline when it already runs on CPU; otherwise builds and caches a
        separate CPU pipeline (the live/GPU pipeline, if any, is left untouched).
        """
        if str(self._diarization_device) == "cpu":
            self._ensure_diarization_pipeline_loaded()
            return self._diarization_pipeline
        if self._whole_file_diarization_pipeline is not None:
            return self._whole_file_diarization_pipeline
        self._emit("[download] whole-file whispercpp-diarization preparing (CPU)")
        self._sanitize_broken_proxy_env()
        resolved_hf_token = self._resolve_hf_token()
        self._apply_hf_token_env(resolved_hf_token)
        model_name_for_pipeline = self._resolve_diarization_model_for_pipeline(
            model_name=self._diarization_model,
            hf_token=resolved_hf_token,
        )
        self._whole_file_diarization_pipeline = self._create_diarization_pipeline(
            model_name=model_name_for_pipeline,
            device="cpu",
            hf_token=resolved_hf_token,
        )
        self._emit("whole-file whispercpp-diarization pipeline initialized (CPU).")
        return self._whole_file_diarization_pipeline

    def diarize_whole_file_turns_from_audio(self, audio_f32: np.ndarray) -> list[dict[str, object]]:
        """Run ONE diarization pass over the whole audio and return global turns.

        Returns ``[{"start", "end", "speaker"}, ...]`` (absolute seconds within the given
        audio, sorted by start time). Unlike the per-window `_attach_speaker_labels` path,
        these labels are globally consistent across the whole file, so the direct/import
        path can assign them to tokens by time overlap
        (`direct_transcription._assign_global_speakers`) and skip the cross-window profile
        re-cluster entirely. Returns ``[]`` on any failure or unusable input (caller
        degrades to no speaker labels rather than crashing the whole direct pass).
        """
        try:
            audio = np.asarray(audio_f32, dtype=np.float32).reshape(-1)
        except Exception:
            return []
        if audio.size < 1600:
            return []
        if not np.isfinite(audio).all():
            audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
        try:
            pipeline = self._ensure_whole_file_diarization_pipeline()
            if pipeline is None:
                return []
            diarize_segments = pipeline(audio, **self._diar_speaker_count_kwargs())
            turns: list[dict[str, object]] = []
            for row in diarize_segments.itertuples(index=False):
                try:
                    start = float(getattr(row, "start"))
                    end = float(getattr(row, "end"))
                    speaker = str(getattr(row, "speaker") or "").strip()
                except Exception:
                    continue
                if speaker and end > start:
                    turns.append({"start": start, "end": end, "speaker": speaker})
            turns.sort(key=lambda t: (float(t["start"]), float(t["end"])))
            return turns
        except Exception as exc:
            self._emit(f"whisper.cpp whole-file diarization failed: {exc}")
            return []

    def _create_diarization_pipeline(self, *, model_name: str, device: str, hf_token: str | None):
        if self._pipeline_factory is not None:
            return self._pipeline_factory(model_name=model_name, device=device, hf_token=hf_token)
        candidates: list[object] = []
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
                except Exception as exc:
                    errors.append(f"{cls.__name__} runtime={type(exc).__name__}: {exc}")
        raise RuntimeError("Unable to initialize whisper.cpp diarization pipeline. " + " | ".join(errors[:4]))

    def _resolve_assign_word_speakers(self):
        if self._assign_word_speakers is not None:
            return self._assign_word_speakers
        try:
            from whisperx.diarize import assign_word_speakers  # type: ignore
        except Exception as exc:
            raise RuntimeError("assign_word_speakers is unavailable in current whisperx installation.") from exc
        return assign_word_speakers

    def _resolve_diarization_device(self) -> str:
        setting = (self._diarization_device_setting or "auto").strip().lower()
        resolved = setting if setting in {"cpu", "cuda"} else self._device
        if resolved != "cuda":
            return "cpu"
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        self._emit(
            "whisper.cpp diarization device downgraded to CPU because torch CUDA is unavailable. "
            f"requested={setting or 'auto'}; asr_device={self._device}"
        )
        return "cpu"

    def _resolve_diarization_model_for_pipeline(self, *, model_name: str, hf_token: str | None) -> str:
        token = self._normalize_diarization_model_ref(model_name) or "pyannote/speaker-diarization-3.1"
        path_like = ("/" in token or "\\" in token) and Path(token).exists()
        if path_like or "/" not in token:
            return token
        local_repo_dir = self._model_root / "diarization" / self._slugify_repo_id(token)
        self._prepare_optional_hf_repo_download(
            repo_id=token,
            local_dir=local_repo_dir,
            provider="whispercpp-diarization",
            model_name=token,
            token=hf_token,
        )
        for dep_repo in self._diarization_dependency_repo_ids(token):
            dep_local_dir = self._model_root / "diarization_deps" / self._slugify_repo_id(dep_repo)
            self._prepare_optional_hf_repo_download(
                repo_id=dep_repo,
                local_dir=dep_local_dir,
                provider="whispercpp-diarization",
                model_name=f"{token} -> {dep_repo}",
                token=hf_token,
            )
        if (local_repo_dir / "config.yaml").exists():
            self._emit(f"[download] whispercpp-diarization cache ready: {local_repo_dir}")
            return str(local_repo_dir)
        return token

    def _prepare_optional_hf_repo_download(
        self,
        *,
        repo_id: str,
        local_dir: Path,
        provider: str,
        model_name: str,
        token: str | None = None,
    ) -> None:
        if not repo_id or not self._auto_download:
            return
        if self._is_local_repo_ready_for_predownload(repo_id=repo_id, local_dir=local_dir):
            self._emit(f"[download] {provider} cache hit: {model_name}")
            return
        allow_patterns = ["*.yaml", "*.yml", "*.json", "*.bin", "*.safetensors", "*.ckpt", "*.pt", "*.onnx"]
        try:
            download_hf_files_with_progress(
                repo_id=repo_id,
                output_dir=str(local_dir),
                allow_patterns=allow_patterns,
                progress_callback=self._progress_callback,
                provider=provider,
                model_name=model_name,
                token=token,
                timeout_seconds=60,
            )
        except Exception:
            try:
                download_hf_snapshot_with_progress(
                    repo_id=repo_id,
                    output_dir=str(local_dir),
                    allow_patterns=allow_patterns,
                    progress_callback=self._progress_callback,
                    provider=provider,
                    model_name=model_name,
                    token=token,
                )
            except Exception as exc2:
                self._emit(f"[download] {provider} skipped: {model_name} ({exc2})")

    @staticmethod
    def _normalize_diarization_model_ref(model_name: str) -> str:
        token = str(model_name or "").strip()
        if token.lower() == "pyannote/speaker-diarization-diarization-3.1":
            return "pyannote/speaker-diarization-3.1"
        return token

    @staticmethod
    def _diarization_dependency_repo_ids(repo_id: str) -> list[str]:
        if repo_id.strip().lower() == "pyannote/speaker-diarization-3.1":
            return ["pyannote/segmentation-3.0", "pyannote/wespeaker-voxceleb-resnet34-lm"]
        return []

    def _diar_speaker_count_kwargs(self) -> dict[str, int]:
        kwargs: dict[str, int] = {}
        if self._diar_min_speakers > 0:
            kwargs["min_speakers"] = self._diar_min_speakers
        if self._diar_max_speakers > 0:
            kwargs["max_speakers"] = self._diar_max_speakers
        return kwargs

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
            self._emit(f"whisper.cpp diarization CUDA warmup skipped: {exc}")
        finally:
            self._diarization_cuda_warmed = True

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
        saw_any_speaker = False
        last_segment_end = self._last_speaker_segment_end
        for raw in segments:
            if not isinstance(raw, dict):
                continue
            text = str(raw.get("text") or "").strip()
            if not text:
                continue
            segment_start = _to_float(raw.get("start"), 0.0)
            segment_end = _to_float(raw.get("end"), segment_start)
            if last_segment_end is not None and segment_start + 0.05 < float(last_segment_end):
                last_segment_end = None
            speaker = self._resolve_display_segment_speaker(raw)
            display_speaker = prev_speaker
            if speaker:
                saw_any_speaker = True
                segment_duration = max(0.0, segment_end - segment_start)
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
                        pending_count >= int(self._speaker_switch_confirm_segments)
                        and pending_duration >= float(self._speaker_switch_min_duration_seconds)
                    ) or single_segment_ready:
                        prev_speaker = pending_speaker
                        pending_speaker = ""
                        pending_count = 0
                        pending_duration = 0.0
                display_speaker = prev_speaker
            if display_speaker:
                marker = self._speaker_label_to_marker(display_speaker)
                is_first_marker = (not turns) and (self._last_speaker_label is None)
                gap = segment_start - float(last_segment_end) if last_segment_end is not None else 0.0
                pause_break = (
                    bool(turns)
                    and display_speaker == self._last_speaker_label
                    and self._speaker_pause_break_seconds > 0.0
                    and gap > self._speaker_pause_break_seconds
                )
                if is_first_marker or display_speaker != self._last_speaker_label or pause_break:
                    prefix = "\n\n" if pause_break else "\n"
                    turns.append(f"{prefix}{marker} {text}" if turns else f"{marker} {text}")
                    self._last_speaker_label = display_speaker
                else:
                    turns.append(text)
            else:
                turns.append(text)
            last_segment_end = segment_end
        self._last_speaker_segment_end = last_segment_end
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
        merged = " ".join(turns).strip()
        marker_pattern = r"(>>|S\d+:|\[spk_\d+\])"
        merged = re.sub(rf"([^\n])[ \t]*{marker_pattern}[ \t]*", r"\1\n\2 ", merged, flags=re.IGNORECASE)
        merged = re.sub(rf"^[ \t]*{marker_pattern}[ \t]*", r"\1 ", merged, flags=re.IGNORECASE)
        merged = re.sub(rf"\n[ \t]*{marker_pattern}[ \t]*", r"\n\1 ", merged, flags=re.IGNORECASE)
        lines = []
        for line in merged.splitlines():
            cleaned = re.sub(r"[ \t]+", " ", line).strip()
            if cleaned:
                lines.append(cleaned)
            elif lines and lines[-1] != "":
                lines.append("")
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
        if not self._speaker_profile_enabled or self._speaker_identity_engine is None:
            return False
        status = str(self._last_speaker_profile_stats.get("status") or "").strip().lower()
        if status in {"skip_engine_unavailable", "skip_backend_unavailable", "skip_disabled", "skip_store_unavailable"}:
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
        return str(Path(__file__).resolve().parents[2] / "speaker_profiles" / "profiles.json")

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
            cache_root.relative_to(model_root)
        except Exception:
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
            self._emit(f"[download] whispercpp-diarization cache cleanup: removed {removed} partial temp files")

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
            if value and ("127.0.0.1:9" in value or "localhost:9" in value):
                os.environ.pop(key, None)
                cleared.append(key)
        if cleared:
            self._emit(
                "[download] whispercpp-diarization proxy bypass: "
                f"cleared invalid proxy env keys={','.join(cleared)}"
            )

    @staticmethod
    def _is_local_repo_ready_for_predownload(*, repo_id: str, local_dir: Path) -> bool:
        if not local_dir.exists() or not local_dir.is_dir():
            return False
        rid = (repo_id or "").strip().lower()
        has_yaml = any((local_dir / name).exists() for name in ("config.yaml", "config.yml"))
        has_json = any((local_dir / name).exists() for name in ("config.json", "preprocessor_config.json"))
        if rid in {"pyannote/speaker-diarization-3.1", "pyannote/speaker-diarization-community-1"}:
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

    def _effective_speaker_profile_max_exemplars(self) -> int:
        configured = int(max(1, self._speaker_profile_max_exemplars))
        if configured <= 1:
            return 1
        folder_lang = self._normalize_alignment_folder_language(self._source_language_hint)
        if folder_lang == "zh":
            return configured
        self._emit(
            "[speaker-profile] multi-exemplar requested "
            f"(max_exemplars={configured}) but session language '{self._source_language_hint or 'auto'}' is not zh - "
            "using max_exemplars=1 this session (round 0061 evidence: zh-only benefit)."
        )
        return 1

    @staticmethod
    def _normalize_alignment_folder_language(value: str | None) -> str:
        token = (value or "").strip().lower()
        if not token:
            return ""
        if token in {"zh-hant", "zh-hans", "zh-tw", "zh-cn", "zh-hk", "zh-sg"}:
            return "zh"
        return token

    @staticmethod
    def _slugify_repo_id(value: str) -> str:
        token = (value or "").strip().lower()
        if not token:
            return "auto"
        slug = re.sub(r"[^0-9a-z._-]+", "-", token).strip("-")
        return slug or "auto"

    @staticmethod
    def _construct_with_compat(cls, kwargs: dict[str, object]):
        active = dict(kwargs)
        for _ in range(6):
            try:
                return cls(**active)
            except TypeError as exc:
                match = re.search(r"unexpected keyword argument '([^']+)'", str(exc))
                if not match:
                    raise
                active.pop(match.group(1), None)
        return cls(**active)

    def _emit(self, message: str) -> None:
        if self._progress_callback is None:
            return
        try:
            self._progress_callback(message)
        except Exception:
            pass
