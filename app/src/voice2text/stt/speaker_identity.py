"""Modular speaker identity engine with pluggable embedding backends."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Callable

import numpy as np

from .profile_quality import ClipQualityConfig, evaluate_clip_quality
from .speaker_profiles import SpeakerProfileStore


def _normalize_backend(token: str) -> str:
    key = str(token or "").strip().lower().replace("-", "_")
    if key in {"pyannote", "pyannote_embedding", "pyannote_audio"}:
        return "pyannote"
    if key in {"wespeaker", "wespeaker_resnet34", "pyannote_wespeaker"}:
        return "wespeaker"
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
    reconcile_threshold: float
    model_root: str
    device: str
    hf_token: str
    pyannote_model: str
    speechbrain_model: str
    nemo_model: str
    # Round 0045 Fix 2: wespeaker-grade embedding for the cross-window profile layer
    # (the same model pyannote diarization uses internally; separates zh where the default
    # pyannote/embedding collapses). Cached as part of diar-3.1, no HF gating.
    wespeaker_model: str = "pyannote/wespeaker-voxceleb-resnet34-lm"
    on_status: Callable[[str], None] | None = None
    quality_gate: ClipQualityConfig = field(default_factory=ClipQualityConfig)
    # Realtime (rolling-window) maturity floors; defaults = shipped behavior. Direct
    # chunks ignore these and use a fixed conservative policy (see _profile_evidence_policy).
    realtime_candidate_seconds: float = 6.0
    realtime_candidate_samples: int = 8
    # Round 0045 step 1: decouple the realtime candidate-match gate from match_threshold.
    # 0.0 = keep the legacy derived value (match_threshold - 0.05). A lower value raises the
    # probability that a speaker's noisy short-clip windows match their OWN prior candidate
    # (less fragmentation -> minority candidates reach the samples floor -> get promoted).
    realtime_candidate_match_threshold: float = 0.0
    # Round 0045 step 1b: assign-vs-update decoupling (realtime). 0.0 = update at the assign
    # gate (legacy). Higher (e.g. 0.85) blends a clip into a profile centroid only on a strong
    # match, keeping centroids pure so the dominant profile cannot drift and absorb everyone.
    realtime_update_match_threshold: float = 0.0
    realtime_visible_seconds: float = 24.0
    realtime_visible_samples: int = 16
    realtime_refresh_alpha: float = 0.5
    realtime_refresh_assign_threshold: float = 0.55
    realtime_refresh_min_cluster_seconds: float = 4.0
    realtime_refresh_merge: bool = True
    realtime_refresh_match_mode: str = "argmax"


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
        self._reconcile_threshold = float(max(0.0, min(0.999, config.reconcile_threshold)))
        self._rt_candidate_seconds = float(max(0.0, getattr(config, "realtime_candidate_seconds", 6.0)))
        self._rt_candidate_samples = int(max(1, getattr(config, "realtime_candidate_samples", 8)))
        self._rt_candidate_match_threshold = float(max(0.0, getattr(config, "realtime_candidate_match_threshold", 0.0)))
        self._rt_update_match_threshold = float(max(0.0, getattr(config, "realtime_update_match_threshold", 0.0)))
        self._rt_visible_seconds = float(max(0.0, getattr(config, "realtime_visible_seconds", 24.0)))
        self._rt_visible_samples = int(max(1, getattr(config, "realtime_visible_samples", 16)))
        self._rt_refresh_alpha = float(max(0.0, min(1.0, getattr(config, "realtime_refresh_alpha", 0.5))))
        self._rt_refresh_assign_threshold = float(
            max(0.0, min(0.999, getattr(config, "realtime_refresh_assign_threshold", 0.55)))
        )
        self._rt_refresh_min_cluster_seconds = float(
            max(0.0, getattr(config, "realtime_refresh_min_cluster_seconds", 4.0))
        )
        self._rt_refresh_merge = bool(getattr(config, "realtime_refresh_merge", True))
        refresh_match_mode = str(getattr(config, "realtime_refresh_match_mode", "argmax") or "argmax").strip().lower()
        self._rt_refresh_match_mode = "mutual" if refresh_match_mode == "mutual" else "argmax"
        self._quality_gate = config.quality_gate or ClipQualityConfig()
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

    def refresh_inventory(self, cluster_centroids: dict[str, object]) -> dict[str, object]:
        started_at = time.perf_counter()
        stats: dict[str, object] = {
            "enabled": bool(self._enabled),
            "backend": self._backend_name,
            "profile_store_ready": bool(self._profile_store is not None),
            "status": "init",
            "alpha": float(self._rt_refresh_alpha),
            "assign_threshold": float(self._rt_refresh_assign_threshold),
            "min_cluster_seconds": float(self._rt_refresh_min_cluster_seconds),
            "merge_enabled": bool(self._rt_refresh_merge),
            "match_mode": str(self._rt_refresh_match_mode),
            "clusters": 0,
            "profile_count_before": int(self._profile_store.profile_count()) if self._profile_store is not None else 0,
            "profile_count": int(self._profile_store.profile_count()) if self._profile_store is not None else 0,
            "refreshed_count": 0,
            "merged_count": 0,
            "remap": {},
            "assignments": [],
        }
        if not self._enabled:
            stats["status"] = "skip_disabled"
            self._last_stats = stats
            return stats
        if self._profile_store is None:
            stats["status"] = "skip_store_unavailable"
            self._last_stats = stats
            return stats
        clusters: dict[str, np.ndarray] = {}
        cluster_seconds: dict[str, float] = {}
        for cluster_id, payload in dict(cluster_centroids or {}).items():
            cid = str(cluster_id or "").strip()
            if not cid:
                continue
            duration_seconds = self._rt_refresh_min_cluster_seconds
            centroid_payload = payload
            if isinstance(payload, dict):
                centroid_payload = payload.get("centroid")
                try:
                    duration_seconds = float(payload.get("duration_seconds", 0.0) or 0.0)
                except Exception:
                    duration_seconds = 0.0
            if duration_seconds < self._rt_refresh_min_cluster_seconds:
                continue
            centroid = _normalize_embedding(np.asarray(centroid_payload if centroid_payload is not None else [], dtype=np.float32))
            if centroid.size == 0:
                continue
            clusters[cid] = centroid
            cluster_seconds[cid] = float(duration_seconds)
        stats["clusters"] = int(len(clusters))
        if not clusters:
            stats["status"] = "skip_no_clusters"
            stats["elapsed_seconds"] = float(time.perf_counter() - started_at)
            self._last_stats = stats
            return stats

        profiles = self._profile_store.profile_summaries()
        if not profiles:
            stats["status"] = "skip_no_profiles"
            stats["elapsed_seconds"] = float(time.perf_counter() - started_at)
            self._last_stats = stats
            return stats
        cluster_ids = list(clusters.keys())
        rows: list[dict[str, object]] = []
        profile_rows: list[tuple[dict[str, object], np.ndarray]] = []
        best_profile_for_cluster: dict[str, tuple[str, float]] = {}
        for profile in profiles:
            profile_id = str(profile.get("id") or "")
            centroid = _normalize_embedding(np.asarray(profile.get("centroid") or [], dtype=np.float32))
            if not profile_id or centroid.size == 0:
                continue
            profile_rows.append((profile, centroid))
            for cid in cluster_ids:
                cluster = clusters[cid]
                similarity = float(np.dot(centroid, cluster)) if centroid.size == cluster.size else -1.0
                current = best_profile_for_cluster.get(cid)
                if current is None or similarity > current[1]:
                    best_profile_for_cluster[cid] = (profile_id, similarity)
        for profile, centroid in profile_rows:
            profile_id = str(profile.get("id") or "")
            similarities: list[tuple[str, float]] = []
            for cid in cluster_ids:
                cluster = clusters[cid]
                similarity = float(np.dot(centroid, cluster)) if centroid.size == cluster.size else -1.0
                similarities.append((cid, similarity))
            if not similarities:
                continue
            best_cluster_id, best_similarity = max(similarities, key=lambda item: item[1])
            if best_similarity < self._rt_refresh_assign_threshold:
                continue
            if self._rt_refresh_match_mode == "mutual":
                reciprocal = best_profile_for_cluster.get(best_cluster_id)
                if reciprocal is None or reciprocal[0] != profile_id:
                    continue
            rows.append(
                {
                    "profile_id": profile_id,
                    "cluster_id": best_cluster_id,
                    "similarity": float(best_similarity),
                    "weight": float(profile.get("weight", 0.0) or 0.0),
                    "samples": int(profile.get("samples", 0) or 0),
                    "total_seconds": float(profile.get("total_seconds", 0.0) or 0.0),
                }
            )

        if not rows:
            stats["status"] = "done_no_assignment"
            stats["elapsed_seconds"] = float(time.perf_counter() - started_at)
            self._last_stats = stats
            return stats

        remap: dict[str, str] = {}
        refreshed_count = 0
        merged_count = 0
        rows_by_cluster: dict[str, list[dict[str, object]]] = {}
        for row in rows:
            rows_by_cluster.setdefault(str(row.get("cluster_id") or ""), []).append(row)
        for cluster_id, cluster_rows in rows_by_cluster.items():
            keep_row = max(cluster_rows, key=self._profile_maturity_key)
            keep_id = str(keep_row.get("profile_id") or "")
            drop_ids = [
                str(row.get("profile_id") or "")
                for row in cluster_rows
                if str(row.get("profile_id") or "") and str(row.get("profile_id") or "") != keep_id
            ]
            if self._rt_refresh_merge and drop_ids:
                merge_stats = self._profile_store.merge_profiles(keep_id, drop_ids)
                merged_count += int(merge_stats.get("merged_count", 0) or 0)
                remap.update({str(old): str(new) for old, new in dict(merge_stats.get("remap") or {}).items()})
            if keep_id and self._profile_store.blend_centroid(keep_id, clusters[cluster_id], self._rt_refresh_alpha):
                refreshed_count += 1

        stats["assignments"] = rows
        stats["refreshed_count"] = int(refreshed_count)
        stats["merged_count"] = int(merged_count)
        stats["remap"] = dict(remap)
        stats["profile_count"] = int(self._profile_store.profile_count())
        stats["status"] = "done"
        stats["elapsed_seconds"] = float(time.perf_counter() - started_at)
        self._last_stats = stats
        return stats

    def match_profile_readonly(self, embedding: np.ndarray, *, threshold: float) -> str | None:
        """Round 0048: argmax-match a single embedding against persisted profile centroids,
        READ-ONLY -- never mutates the store (no EMA, no merge, no new-profile creation). Used by
        the pre-commit local-diarization relabel path, which needs a same-space comparison
        (embedding must already be extracted with THIS engine's backend, same principle as the
        round-0046 refresh) but must never write back, since the correction target there is a
        still-mutable subtitle batch, not the inventory itself."""
        if not self._enabled or self._profile_store is None:
            return None
        query = _normalize_embedding(np.asarray(embedding if embedding is not None else [], dtype=np.float32))
        if query.size == 0:
            return None
        profiles = self._profile_store.profile_summaries()
        if not profiles:
            return None
        best_id = ""
        best_similarity = -1.0
        for profile in profiles:
            profile_id = str(profile.get("id") or "")
            centroid = _normalize_embedding(np.asarray(profile.get("centroid") or [], dtype=np.float32))
            if not profile_id or centroid.size == 0 or centroid.size != query.size:
                continue
            similarity = float(np.dot(centroid, query))
            if similarity > best_similarity:
                best_similarity = similarity
                best_id = profile_id
        if best_id and best_similarity >= float(max(0.0, min(0.999, threshold))):
            return best_id
        return None

    def score_profiles_readonly(self, embedding: np.ndarray) -> dict[str, float]:
        """Round 0052: cosine of one embedding against EVERY persisted profile centroid,
        READ-ONLY (same non-mutation contract as `match_profile_readonly`). The caller uses the
        full score map for the relabel margin gate: a resolved profile may only overwrite a word's
        existing label when its cosine beats the incumbent label's own cosine by a margin -- which
        requires knowing the incumbent's score, not just the argmax."""
        if not self._enabled or self._profile_store is None:
            return {}
        query = _normalize_embedding(np.asarray(embedding if embedding is not None else [], dtype=np.float32))
        if query.size == 0:
            return {}
        scores: dict[str, float] = {}
        for profile in self._profile_store.profile_summaries():
            profile_id = str(profile.get("id") or "")
            centroid = _normalize_embedding(np.asarray(profile.get("centroid") or [], dtype=np.float32))
            if not profile_id or centroid.size == 0 or centroid.size != query.size:
                continue
            scores[profile_id] = float(np.dot(centroid, query))
        return scores

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
        # Per-speaker text + word scores, gathered from the same segments, feed the learn-path
        # quality gate (round 0023). Only consumed when the gate is enabled.
        text_by_speaker: dict[str, list[str]] = {}
        scores_by_speaker: dict[str, list[float]] = {}
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
            if self._quality_gate.enabled:
                seg_text = str(seg.get("text") or "").strip()
                if seg_text:
                    text_by_speaker.setdefault(speaker, []).append(seg_text)
                for wd in (seg.get("words") or []):
                    if not isinstance(wd, dict):
                        continue
                    raw_score = wd.get("score")
                    if raw_score is None:
                        continue
                    try:
                        scores_by_speaker.setdefault(speaker, []).append(float(raw_score))
                    except (TypeError, ValueError):
                        continue
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
        skipped_low_quality_count = 0
        audio_seconds = float(audio_f32.size) / 16000.0
        candidate_min_seconds, candidate_min_samples, evidence_cap_seconds = self._profile_evidence_policy(
            audio_seconds=audio_seconds,
            min_seconds=self._min_seconds,
        )
        visible_min_seconds, visible_min_samples = self._visible_profile_policy(
            audio_seconds=audio_seconds,
            candidate_min_seconds=candidate_min_seconds,
            candidate_min_samples=candidate_min_samples,
        )
        visible_alias_threshold = float(max(self._match_threshold, min(0.82, self._match_threshold + 0.04)))
        candidate_threshold = self._resolve_candidate_threshold(audio_seconds)
        update_threshold = self._resolve_update_threshold(audio_seconds)

        for (speaker, spans) in spans_by_speaker.items():
            clip = self._collect_speaker_clip(audio_f32, spans)
            if clip.size == 0:
                skipped_short_count += 1
                continue
            duration_seconds = float(clip.size) / 16000.0
            if duration_seconds < self._min_seconds:
                skipped_short_count += 1
                continue
            evidence_seconds = float(min(duration_seconds, evidence_cap_seconds))
            embedding = self._backend.extract_embedding(clip)
            if embedding is None:
                skipped_no_embedding_count += 1
                continue
            # Learn-path quality gate (round 0023): a low-quality clip (gibberish / music tail /
            # degenerate / low-confidence) may still *match* a mature profile for display, but must
            # not create or average into a centroid. This stays strictly off the display-label
            # path so the merge anchors — and thus the transcript/CER — are unchanged.
            learn_allowed = True
            if self._quality_gate.enabled:
                quality = evaluate_clip_quality(
                    text=" ".join(text_by_speaker.get(speaker, [])),
                    word_scores=scores_by_speaker.get(speaker),
                    duration_seconds=duration_seconds,
                    config=self._quality_gate,
                )
                if not quality.ok:
                    learn_allowed = False
                    skipped_low_quality_count += 1
            matched = self._profile_store.match_or_create(
                embedding=embedding,
                threshold=self._match_threshold,
                observed_label=speaker,
                duration_seconds=evidence_seconds,
                candidate_min_seconds=candidate_min_seconds,
                candidate_min_samples=candidate_min_samples,
                candidate_threshold=candidate_threshold,
                update_threshold=update_threshold,
                allow_update=learn_allowed,
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
                            "evidence_seconds": float(evidence_seconds),
                            "span_count": int(len(spans)),
                        }
                    )
                skipped_no_profile_count += 1
                continue
            visible = self._profile_store.visible_profile_alias(
                profile_id=matched.profile_id,
                embedding=embedding,
                min_total_seconds=visible_min_seconds,
                min_samples=visible_min_samples,
                similarity_threshold=visible_alias_threshold,
            )
            visible_profile_id = str(visible.get("alias") or matched.profile_id)
            profile_by_local_speaker[speaker] = visible_profile_id
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
                    "visible_profile_speaker": str(visible_profile_id),
                    "visible_profile_aliased": bool(visible.get("aliased", False)),
                    "visible_profile_alias_similarity": float(visible.get("similarity", -1.0) or -1.0),
                    "visible_profile_mature": bool(visible.get("mature", False)),
                    "similarity": float(matched.similarity),
                    "created": bool(matched.created),
                    "promoted": bool(matched.promoted),
                    "duration_seconds": float(duration_seconds),
                    "evidence_seconds": float(evidence_seconds),
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
        stats["skipped_low_quality_count"] = int(skipped_low_quality_count)
        stats["staged_candidate_count"] = int(staged_count)
        stats["promoted_candidate_count"] = int(promoted_count)
        stats["candidate_count"] = int(self._profile_store.candidate_count())
        stats["candidate_min_seconds"] = float(candidate_min_seconds)
        stats["candidate_min_samples"] = int(candidate_min_samples)
        stats["candidate_threshold"] = float(candidate_threshold)
        stats["evidence_cap_seconds"] = float(evidence_cap_seconds)
        stats["visible_min_seconds"] = float(visible_min_seconds)
        stats["visible_min_samples"] = int(visible_min_samples)
        stats["visible_alias_threshold"] = float(visible_alias_threshold)
        stats["visible_alias_count"] = int(
            sum(1 for row in assignment_rows if isinstance(row, dict) and bool(row.get("visible_profile_aliased", False)))
        )
        stats["auto_reconcile_threshold"] = float(self._reconcile_threshold)
        stats["auto_reconcile"] = {
            "enabled": bool(self._reconcile_threshold > 0.0),
            "merged_count": 0,
            "remap": {},
        }

        if self._reconcile_threshold > 0.0 and self._profile_store.profile_count() > 1:
            reconcile_stats = self._profile_store.reconcile_similar_profiles(
                threshold=float(self._reconcile_threshold)
            )
            remap = {
                str(old): str(new)
                for (old, new) in dict(reconcile_stats.get("remap") or {}).items()
                if str(old) and str(new)
            }
            if remap:
                profile_by_local_speaker = {
                    local: remap.get(profile_id, profile_id)
                    for (local, profile_id) in profile_by_local_speaker.items()
                }
                for row in assignment_rows:
                    if not isinstance(row, dict):
                        continue
                    profile_id = str(row.get("profile_speaker") or "")
                    if profile_id in remap:
                        row["profile_reconciled_from"] = profile_id
                        row["profile_speaker"] = remap[profile_id]
                    visible_id = str(row.get("visible_profile_speaker") or "")
                    if visible_id in remap:
                        row["visible_profile_reconciled_from"] = visible_id
                        row["visible_profile_speaker"] = remap[visible_id]
            stats["auto_reconcile"] = reconcile_stats

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
                f"candidate_min={float(stats.get('candidate_min_seconds', 0.0) or 0.0):.2f}s/"
                f"{int(stats.get('candidate_min_samples', 0) or 0)}x; "
                f"visible_min={float(stats.get('visible_min_seconds', 0.0) or 0.0):.2f}s/"
                f"{int(stats.get('visible_min_samples', 0) or 0)}x; "
                f"visible_alias={stats.get('visible_alias_count', 0)}; "
                f"evidence_cap={float(stats.get('evidence_cap_seconds', 0.0) or 0.0):.2f}s; "
                f"skip_short={stats.get('skipped_short_count', 0)}; "
                f"skip_lowq={stats.get('skipped_low_quality_count', 0)}; "
                f"profile_total={stats.get('profile_count', 0)}"
            )
        return segments

    def _resolve_candidate_threshold(self, audio_seconds: float) -> float:
        """Similarity gate for matching a clip to an existing candidate (not a profile).

        Realtime windows can override it via realtime_candidate_match_threshold (0.0 keeps
        the legacy match_threshold-0.05). Lowering it concentrates a speaker's fragmented
        short-clip windows onto one candidate so it can reach the promotion samples floor.
        Direct chunks (window > 15s) always keep the legacy derived value.
        """
        window_seconds = float(max(0.0, audio_seconds))
        if window_seconds <= 15.0 and self._rt_candidate_match_threshold > 0.0:
            return float(max(0.0, min(0.999, self._rt_candidate_match_threshold)))
        return float(max(0.0, self._match_threshold - 0.05))

    def _resolve_update_threshold(self, audio_seconds: float) -> float | None:
        """Centroid-update gate (realtime only). None => update at the assign gate (legacy)."""
        window_seconds = float(max(0.0, audio_seconds))
        if window_seconds <= 15.0 and self._rt_update_match_threshold > 0.0:
            return float(max(0.0, min(0.999, self._rt_update_match_threshold)))
        return None

    @staticmethod
    def _profile_maturity_key(profile: dict[str, object]) -> tuple[float, int, float, str]:
        return (
            float(profile.get("total_seconds", 0.0) or 0.0),
            int(profile.get("samples", 0) or 0),
            float(profile.get("weight", 0.0) or 0.0),
            str(profile.get("profile_id") or profile.get("id") or ""),
        )

    def _profile_evidence_policy(self, *, audio_seconds: float, min_seconds: float) -> tuple[float, int, float]:
        """Return candidate evidence thresholds for long direct chunks vs rolling windows.

        Rolling windows are heavily overlapped, so raw local-speaker duration is not
        unique evidence — their maturity floors are configurable (realtime_candidate_*).
        Direct chunks are much longer and keep a FIXED conservative quick-promotion
        policy (max(min_seconds*3, 6.0)) so reference/export speaker counts stay clean;
        they are deliberately decoupled from the realtime floors (lowering the realtime
        knobs must not over-split the direct reference).
        """
        window_seconds = float(max(0.0, audio_seconds))
        if window_seconds <= 15.0:
            base_min_seconds = float(max(0.0, self._rt_candidate_seconds))
            min_samples = int(max(1, self._rt_candidate_samples))
            evidence_cap = float(max(0.5, min(1.5, float(min_seconds))))
            return base_min_seconds, min_samples, evidence_cap
        direct_base_seconds = float(max(float(min_seconds) * 3.0, 6.0))
        return direct_base_seconds, 1, float(max(direct_base_seconds, window_seconds))

    def _visible_profile_policy(
        self,
        *,
        audio_seconds: float,
        candidate_min_seconds: float,
        candidate_min_samples: int,
    ) -> tuple[float, int]:
        """Return maturity thresholds before a profile becomes a stable visible identity.

        Long direct chunks are reference-style processing and label early (trivially
        visible). Short rolling windows need more repeated evidence because the same
        audio appears in many overlapping windows — controlled by realtime_visible_*.
        `candidate_*` args are unused for realtime now (visible floors are independent
        config rather than a multiple of the candidate floor); kept for call-site parity.
        """
        del candidate_min_seconds, candidate_min_samples
        window_seconds = float(max(0.0, audio_seconds))
        if window_seconds <= 15.0:
            return (
                float(max(0.0, self._rt_visible_seconds)),
                int(max(1, self._rt_visible_samples)),
            )
        return (0.0, 1)

    def _build_backend(self, cfg: SpeakerIdentityConfig) -> _BaseEmbeddingBackend:
        model_root = Path(str(cfg.model_root or ".")).resolve()
        backend = _normalize_backend(cfg.backend)
        if backend == "wespeaker":
            # wespeaker is a pyannote-wrapped Model -> reuse the pyannote Inference path.
            # No HF gating (local diar-3.1 cache), so fall back to pyannote/embedding only
            # if the wespeaker model is somehow missing.
            wespeaker_backend = _PyannoteEmbeddingBackend(
                model_ref=str(cfg.wespeaker_model or "pyannote/wespeaker-voxceleb-resnet34-lm"),
                hf_token=cfg.hf_token,
                device=cfg.device,
                model_root=model_root,
                on_status=cfg.on_status,
            )
            wespeaker_backend.backend_name = "wespeaker"
            return _FallbackEmbeddingBackend(
                primary=wespeaker_backend,
                fallback=_PyannoteEmbeddingBackend(
                    model_ref=str(cfg.pyannote_model or "pyannote/embedding"),
                    hf_token=cfg.hf_token,
                    device=cfg.device,
                    model_root=model_root,
                    on_status=cfg.on_status,
                ),
                on_status=cfg.on_status,
            )
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
