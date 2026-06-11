"""Modular speaker identity engine with pluggable embedding backends."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .speaker_profiles import SpeakerProfileStore


def _normalize_backend(token: str) -> str:
    key = str(token or "").strip().lower().replace("-", "_")
    if key in {"pyannote", "pyannote_embedding", "pyannote_audio"}:
        return "pyannote"
    if key in {"speechbrain", "speechbrain_ecapa", "ecapa", "ecapa_tdnn"}:
        return "speechbrain_ecapa"
    if key in {"nemo", "nemo_titanet", "titanet", "nvidia_nemo"}:
        return "nemo_titanet"
    return "pyannote"


def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if vec.size == 0:
        return vec
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        return np.zeros_like(vec, dtype=np.float32)
    return (vec / norm).astype(np.float32, copy=False)


@dataclass
class SpeakerIdentityConfig:
    enabled: bool
    backend: str
    store_path: str
    match_threshold: float
    min_seconds: float
    model_root: str
    device: str
    hf_token: str
    pyannote_model: str
    speechbrain_model: str
    nemo_model: str
    on_status: Callable[[str], None] | None = None


class _BaseEmbeddingBackend:
    backend_name = "base"

    def __init__(self, *, device: str, model_root: Path, on_status: Callable[[str], None] | None) -> None:
        self._device = str(device or "cpu").strip().lower()
        self._model_root = model_root
        self._on_status = on_status
        self._disabled_reason: str | None = None

    @property
    def disabled_reason(self) -> str:
        return str(self._disabled_reason or "")

    def ensure_loaded(self) -> bool:
        raise NotImplementedError

    def extract_embedding(self, clip: np.ndarray) -> np.ndarray | None:
        raise NotImplementedError

    def _emit(self, message: str) -> None:
        if self._on_status is not None:
            self._on_status(message)


class _PyannoteEmbeddingBackend(_BaseEmbeddingBackend):
    backend_name = "pyannote"

    def __init__(
        self,
        *,
        model_ref: str,
        hf_token: str,
        device: str,
        model_root: Path,
        on_status: Callable[[str], None] | None,
    ) -> None:
        super().__init__(device=device, model_root=model_root, on_status=on_status)
        self._model_ref = str(model_ref or "pyannote/embedding").strip() or "pyannote/embedding"
        self._hf_token = str(hf_token or "").strip()
        self._inference = None

    def ensure_loaded(self) -> bool:
        if self._disabled_reason:
            return False
        if self._inference is not None:
            return True
        try:
            from pyannote.audio import Inference, Model  # type: ignore
            import torch  # type: ignore
        except Exception as exc:
            self._disabled_reason = str(exc)
            self._emit(f"Speaker backend disabled ({self.backend_name}): {exc}")
            return False

        token = self._hf_token or None
        model = None
        errors: list[str] = []
        for kwargs in ({"token": token}, {"use_auth_token": token}, {}):
            try:
                active = {k: v for (k, v) in kwargs.items() if v}
                model = Model.from_pretrained(self._model_ref, **active)
                break
            except Exception as exc:
                errors.append(str(exc))
        if model is None:
            self._disabled_reason = "; ".join(errors[:3]) or "model load failed"
            self._emit(f"Speaker backend disabled ({self.backend_name}): {self._disabled_reason}")
            return False

        device_name = "cuda" if (self._device == "cuda" and torch.cuda.is_available()) else "cpu"
        try:
            self._inference = Inference(model, window="whole", device=torch.device(device_name))
        except Exception as exc:
            self._disabled_reason = str(exc)
            self._emit(f"Speaker backend disabled ({self.backend_name}): {exc}")
            return False
        self._emit(f"Speaker backend ready: {self.backend_name}; model={self._model_ref}; device={device_name}")
        return True

    def extract_embedding(self, clip: np.ndarray) -> np.ndarray | None:
        if self._inference is None and (not self.ensure_loaded()):
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
        for payload in payloads:
            try:
                value = self._inference(payload)
                vec = np.asarray(value, dtype=np.float32).reshape(-1)
                if vec.size > 0 and np.isfinite(vec).all():
                    return _normalize_embedding(vec)
            except Exception:
                continue
        return None


class _FallbackEmbeddingBackend(_BaseEmbeddingBackend):
    backend_name = "fallback"

    def __init__(
        self,
        *,
        primary: _BaseEmbeddingBackend,
        fallback: _BaseEmbeddingBackend,
        on_status: Callable[[str], None] | None,
    ) -> None:
        super().__init__(device=primary._device, model_root=primary._model_root, on_status=on_status)
        self._primary = primary
        self._fallback = fallback
        self._active: _BaseEmbeddingBackend | None = None

    @property
    def active_backend_name(self) -> str:
        active = self._active or self._primary
        return str(getattr(active, "backend_name", "unknown"))

    def ensure_loaded(self) -> bool:
        if self._active is not None:
            return self._active.ensure_loaded()
        if self._primary.ensure_loaded():
            self._active = self._primary
            return True
        primary_reason = self._short_reason(self._primary.disabled_reason)
        self._emit(
            "Speaker backend fallback: "
            f"{self._primary.backend_name} unavailable ({primary_reason}); trying {self._fallback.backend_name}"
        )
        if self._fallback.ensure_loaded():
            self._active = self._fallback
            self._disabled_reason = None
            return True
        fallback_reason = self._short_reason(self._fallback.disabled_reason)
        self._disabled_reason = (
            f"{self._primary.backend_name}: {primary_reason}; "
            f"{self._fallback.backend_name}: {fallback_reason}"
        )
        return False

    def extract_embedding(self, clip: np.ndarray) -> np.ndarray | None:
        if not self.ensure_loaded():
            return None
        active = self._active or self._primary
        return active.extract_embedding(clip)

    @staticmethod
    def _short_reason(reason: str) -> str:
        text = " ".join(str(reason or "unavailable").split())
        if "pyannote/embedding" in text and ("403" in text or "gated repo" in text.lower()):
            return "pyannote/embedding gated access denied"
        if len(text) > 220:
            return text[:217] + "..."
        return text


class _SpeechBrainEcapaBackend(_BaseEmbeddingBackend):
    backend_name = "speechbrain_ecapa"

    def __init__(
        self,
        *,
        model_ref: str,
        device: str,
        model_root: Path,
        on_status: Callable[[str], None] | None,
    ) -> None:
        super().__init__(device=device, model_root=model_root, on_status=on_status)
        self._model_ref = str(model_ref or "speechbrain/spkrec-ecapa-voxceleb").strip() or "speechbrain/spkrec-ecapa-voxceleb"
        self._classifier = None

    def ensure_loaded(self) -> bool:
        if self._disabled_reason:
            return False
        if self._classifier is not None:
            return True
        try:
            import torch  # type: ignore
            try:
                from speechbrain.inference.speaker import EncoderClassifier  # type: ignore
            except Exception:
                from speechbrain.pretrained import EncoderClassifier  # type: ignore
        except Exception as exc:
            self._disabled_reason = str(exc)
            self._emit(f"Speaker backend disabled ({self.backend_name}): {exc}")
            return False

        cache_dir = self._model_root / "speaker_embeddings" / "speechbrain"
        cache_dir.mkdir(parents=True, exist_ok=True)
        run_device = "cuda" if (self._device == "cuda" and torch.cuda.is_available()) else "cpu"
        try:
            self._classifier = EncoderClassifier.from_hparams(
                source=self._model_ref,
                savedir=str(cache_dir / self._model_ref.replace("/", "__")),
                run_opts={"device": run_device},
            )
        except Exception as exc:
            self._disabled_reason = str(exc)
            self._emit(f"Speaker backend disabled ({self.backend_name}): {exc}")
            return False
        self._emit(f"Speaker backend ready: {self.backend_name}; model={self._model_ref}; device={run_device}")
        return True

    def extract_embedding(self, clip: np.ndarray) -> np.ndarray | None:
        if self._classifier is None and (not self.ensure_loaded()):
            return None
        try:
            import torch  # type: ignore
            wav = torch.from_numpy(np.ascontiguousarray(clip, dtype=np.float32)).unsqueeze(0)
            wav_lens = torch.tensor([1.0], dtype=torch.float32)
            emb = self._classifier.encode_batch(wav, wav_lens=wav_lens, normalize=True)
            if hasattr(emb, "detach"):
                emb = emb.detach()
            if hasattr(emb, "cpu"):
                emb = emb.cpu()
            vec = np.asarray(emb, dtype=np.float32).reshape(-1)
            if vec.size > 0 and np.isfinite(vec).all():
                return _normalize_embedding(vec)
        except Exception:
            return None
        return None


class _NemoTitanetBackend(_BaseEmbeddingBackend):
    backend_name = "nemo_titanet"

    def __init__(
        self,
        *,
        model_ref: str,
        device: str,
        model_root: Path,
        on_status: Callable[[str], None] | None,
    ) -> None:
        super().__init__(device=device, model_root=model_root, on_status=on_status)
        self._model_ref = str(model_ref or "nvidia/speakerverification_en_titanet_large").strip() or "nvidia/speakerverification_en_titanet_large"
        self._model = None

    def ensure_loaded(self) -> bool:
        if self._disabled_reason:
            return False
        if self._model is not None:
            return True
        try:
            import torch  # type: ignore
            from nemo.collections.asr.models import EncDecSpeakerLabelModel  # type: ignore
        except Exception as exc:
            self._disabled_reason = str(exc)
            self._emit(f"Speaker backend disabled ({self.backend_name}): {exc}")
            return False

        try:
            self._model = EncDecSpeakerLabelModel.from_pretrained(model_name=self._model_ref)
            device_name = "cuda" if (self._device == "cuda" and torch.cuda.is_available()) else "cpu"
            self._model = self._model.to(device_name)
        except Exception as exc:
            self._disabled_reason = str(exc)
            self._emit(f"Speaker backend disabled ({self.backend_name}): {exc}")
            return False
        self._emit(f"Speaker backend ready: {self.backend_name}; model={self._model_ref}")
        return True

    def extract_embedding(self, clip: np.ndarray) -> np.ndarray | None:
        if self._model is None and (not self.ensure_loaded()):
            return None
        try:
            import torch  # type: ignore
            waveform = torch.from_numpy(np.ascontiguousarray(clip, dtype=np.float32)).unsqueeze(0)
            lengths = torch.tensor([waveform.shape[-1]], dtype=torch.int64)
            if hasattr(self._model, "device"):
                model_device = self._model.device
                waveform = waveform.to(model_device)
                lengths = lengths.to(model_device)
            logits, emb = self._model.forward(input_signal=waveform, input_signal_length=lengths)
            _ = logits
            if hasattr(emb, "detach"):
                emb = emb.detach()
            if hasattr(emb, "cpu"):
                emb = emb.cpu()
            vec = np.asarray(emb, dtype=np.float32).reshape(-1)
            if vec.size > 0 and np.isfinite(vec).all():
                return _normalize_embedding(vec)
        except Exception:
            return None
        return None


class SpeakerIdentityEngine:
    def __init__(self, config: SpeakerIdentityConfig) -> None:
        self._enabled = bool(config.enabled)
        self._backend_name = _normalize_backend(config.backend)
        self._match_threshold = float(max(0.0, min(0.999, config.match_threshold)))
        self._min_seconds = float(max(0.2, config.min_seconds))
        self._on_status = config.on_status
        self._last_stats: dict[str, object] = {}
        self._profile_store: SpeakerProfileStore | None = None
        if self._enabled:
            self._profile_store = SpeakerProfileStore(path=str(config.store_path), on_status=self._on_status)
        self._backend = self._build_backend(config)

    @property
    def last_stats(self) -> dict[str, object]:
        return dict(self._last_stats)

    def prewarm(self) -> None:
        if not self._enabled:
            return
        self._backend.ensure_loaded()

    def reconcile_profiles(self, *, threshold: float | None = None) -> dict[str, object]:
        stats: dict[str, object] = {
            "enabled": bool(self._enabled),
            "backend": self._backend_name,
            "profile_store_ready": bool(self._profile_store is not None),
            "status": "init",
            "threshold": float(self._match_threshold if threshold is None else threshold),
            "merged_count": 0,
            "remap": {},
            "profile_count": int(self._profile_store.profile_count()) if self._profile_store is not None else 0,
        }
        if not self._enabled:
            stats["status"] = "skip_disabled"
            return stats
        if self._profile_store is None:
            stats["status"] = "skip_store_unavailable"
            return stats
        result = self._profile_store.reconcile_similar_profiles(
            threshold=float(self._match_threshold if threshold is None else threshold)
        )
        stats.update(result)
        stats["status"] = "done"
        if self._on_status is not None and int(stats.get("merged_count", 0) or 0) > 0:
            self._on_status(
                "[speaker-profile] reconciliation: "
                f"merged={stats.get('merged_count', 0)}; profile_total={stats.get('profile_count', 0)}"
            )
        return stats

    def apply(
        self,
        *,
        audio: np.ndarray,
        segments: list[dict],
        resolve_local_speaker: Callable[[dict], str | None],
    ) -> list[dict]:
        stats: dict[str, object] = {
            "enabled": bool(self._enabled),
            "backend": self._backend_name,
            "profile_store_ready": bool(self._profile_store is not None),
            "backend_ready": False,
            "backend_disabled_reason": "",
            "match_threshold": float(self._match_threshold),
            "min_seconds": float(self._min_seconds),
            "segment_count": int(len(segments)),
            "local_speaker_count": 0,
            "assigned_local_speaker_count": 0,
            "matched_count": 0,
            "created_count": 0,
            "skipped_short_count": 0,
            "skipped_no_embedding_count": 0,
            "skipped_no_profile_count": 0,
            "assignments": [],
            "profile_count": int(self._profile_store.profile_count()) if self._profile_store is not None else 0,
            "status": "init",
        }
        if not self._enabled:
            stats["status"] = "skip_disabled"
            self._last_stats = stats
            return segments
        if not segments:
            stats["status"] = "skip_no_segments"
            self._last_stats = stats
            return segments
        if self._profile_store is None:
            stats["status"] = "skip_store_unavailable"
            self._last_stats = stats
            return segments
        try:
            audio_f32 = np.asarray(audio, dtype=np.float32).reshape(-1)
        except Exception:
            stats["status"] = "skip_audio_convert_failed"
            self._last_stats = stats
            return segments
        if audio_f32.size < 1600:
            stats["status"] = "skip_audio_too_short"
            self._last_stats = stats
            return segments
        backend_ready = self._backend.ensure_loaded()
        stats["backend_ready"] = bool(backend_ready)
        if isinstance(self._backend, _FallbackEmbeddingBackend):
            stats["backend"] = self._backend.active_backend_name
        stats["backend_disabled_reason"] = str(self._backend.disabled_reason or "")
        if not backend_ready:
            stats["status"] = "skip_backend_unavailable"
            self._last_stats = stats
            return segments

        spans_by_speaker: dict[str, list[tuple[float, float]]] = {}
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            speaker = resolve_local_speaker(seg)
            if not speaker:
                continue
            try:
                start = float(seg.get("start"))
                end = float(seg.get("end"))
            except Exception:
                continue
            if end <= start:
                continue
            spans_by_speaker.setdefault(speaker, []).append((start, end))
        stats["local_speaker_count"] = int(len(spans_by_speaker))
        if not spans_by_speaker:
            stats["status"] = "skip_no_local_speaker"
            self._last_stats = stats
            return segments

        profile_by_local_speaker: dict[str, str] = {}
        assignment_rows: list[dict[str, object]] = []
        matched_count = 0
        created_count = 0
        skipped_short_count = 0
        skipped_no_embedding_count = 0
        skipped_no_profile_count = 0
        staged_count = 0
        promoted_count = 0
        candidate_min_seconds = float(max(self._min_seconds * 2.0, 4.0))
        candidate_threshold = float(max(0.0, self._match_threshold - 0.05))

        for (speaker, spans) in spans_by_speaker.items():
            clip = self._collect_speaker_clip(audio_f32, spans)
            if clip.size == 0:
                skipped_short_count += 1
                continue
            duration_seconds = float(clip.size) / 16000.0
            if duration_seconds < self._min_seconds:
                skipped_short_count += 1
                continue
            embedding = self._backend.extract_embedding(clip)
            if embedding is None:
                skipped_no_embedding_count += 1
                continue
            matched = self._profile_store.match_or_create(
                embedding=embedding,
                threshold=self._match_threshold,
                observed_label=speaker,
                duration_seconds=duration_seconds,
                candidate_min_seconds=candidate_min_seconds,
                candidate_threshold=candidate_threshold,
            )
            if not matched.profile_id:
                if matched.staged:
                    staged_count += 1
                    assignment_rows.append(
                        {
                            "local_speaker": str(speaker),
                            "candidate_speaker": str(matched.candidate_id),
                            "similarity": float(matched.similarity),
                            "staged": True,
                            "candidate_total_seconds": float(matched.candidate_total_seconds),
                            "duration_seconds": float(duration_seconds),
                            "span_count": int(len(spans)),
                        }
                    )
                skipped_no_profile_count += 1
                continue
            profile_by_local_speaker[speaker] = matched.profile_id
            if matched.created:
                created_count += 1
                if matched.promoted:
                    promoted_count += 1
            else:
                matched_count += 1
            assignment_rows.append(
                {
                    "local_speaker": str(speaker),
                    "profile_speaker": str(matched.profile_id),
                    "similarity": float(matched.similarity),
                    "created": bool(matched.created),
                    "promoted": bool(matched.promoted),
                    "duration_seconds": float(duration_seconds),
                    "span_count": int(len(spans)),
                }
            )

        stats["assignments"] = assignment_rows
        stats["assigned_local_speaker_count"] = int(len(profile_by_local_speaker))
        stats["matched_count"] = int(matched_count)
        stats["created_count"] = int(created_count)
        stats["skipped_short_count"] = int(skipped_short_count)
        stats["skipped_no_embedding_count"] = int(skipped_no_embedding_count)
        stats["skipped_no_profile_count"] = int(skipped_no_profile_count)
        stats["staged_candidate_count"] = int(staged_count)
        stats["promoted_candidate_count"] = int(promoted_count)
        stats["candidate_count"] = int(self._profile_store.candidate_count())
        stats["candidate_min_seconds"] = float(candidate_min_seconds)
        stats["candidate_threshold"] = float(candidate_threshold)

        if not profile_by_local_speaker:
            stats["profile_count"] = int(self._profile_store.profile_count())
            stats["status"] = "done_no_assignment"
            self._last_stats = stats
            return segments

        for seg in segments:
            if not isinstance(seg, dict):
                continue
            local_speaker = resolve_local_speaker(seg)
            if not local_speaker:
                continue
            profile_id = profile_by_local_speaker.get(local_speaker)
            if not profile_id:
                continue
            seg["profile_speaker"] = profile_id
            words = seg.get("words")
            if not isinstance(words, list):
                continue
            for wd in words:
                if not isinstance(wd, dict):
                    continue
                wd["profile_speaker"] = profile_id

        stats["profile_count"] = int(self._profile_store.profile_count())
        stats["status"] = "done_assigned"
        self._last_stats = stats
        if self._on_status is not None:
            self._on_status(
                "[speaker-profile] window summary: "
                f"backend={stats.get('backend', self._backend_name)}; local={stats.get('local_speaker_count', 0)}; "
                f"assigned={stats.get('assigned_local_speaker_count', 0)}; "
                f"matched={stats.get('matched_count', 0)}; created={stats.get('created_count', 0)}; "
                f"staged={stats.get('staged_candidate_count', 0)}; promoted={stats.get('promoted_candidate_count', 0)}; "
                f"candidates={stats.get('candidate_count', 0)}; "
                f"skip_short={stats.get('skipped_short_count', 0)}; profile_total={stats.get('profile_count', 0)}"
            )
        return segments

    def _build_backend(self, cfg: SpeakerIdentityConfig) -> _BaseEmbeddingBackend:
        model_root = Path(str(cfg.model_root or ".")).resolve()
        backend = _normalize_backend(cfg.backend)
        if backend == "speechbrain_ecapa":
            return _SpeechBrainEcapaBackend(
                model_ref=cfg.speechbrain_model,
                device=cfg.device,
                model_root=model_root,
                on_status=cfg.on_status,
            )
        if backend == "nemo_titanet":
            return _NemoTitanetBackend(
                model_ref=cfg.nemo_model,
                device=cfg.device,
                model_root=model_root,
                on_status=cfg.on_status,
            )
        pyannote_backend = _PyannoteEmbeddingBackend(
            model_ref=cfg.pyannote_model,
            hf_token=cfg.hf_token,
            device=cfg.device,
            model_root=model_root,
            on_status=cfg.on_status,
        )
        return _FallbackEmbeddingBackend(
            primary=pyannote_backend,
            fallback=_SpeechBrainEcapaBackend(
                model_ref=cfg.speechbrain_model,
                device=cfg.device,
                model_root=model_root,
                on_status=cfg.on_status,
            ),
            on_status=cfg.on_status,
        )

    @staticmethod
    def _collect_speaker_clip(audio_f32: np.ndarray, spans: list[tuple[float, float]]) -> np.ndarray:
        if audio_f32.size == 0 or not spans:
            return np.zeros((0,), dtype=np.float32)
        pieces: list[np.ndarray] = []
        kept_seconds = 0.0
        for (start, end) in sorted(spans):
            s = max(0, int(round(float(start) * 16000.0)))
            e = min(int(audio_f32.size), int(round(float(end) * 16000.0)))
            if e <= s:
                continue
            piece = np.ascontiguousarray(audio_f32[s:e], dtype=np.float32)
            if piece.size <= 0:
                continue
            pieces.append(piece)
            kept_seconds += float(piece.size) / 16000.0
            if kept_seconds >= 8.0:
                break
        if not pieces:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(pieces, axis=0)
