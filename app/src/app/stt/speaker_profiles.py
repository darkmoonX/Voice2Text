"""Persistent speaker profile matching based on embedding centroids."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import threading
from typing import Callable

import numpy as np


def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if vec.size == 0:
        return vec
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        return np.zeros_like(vec, dtype=np.float32)
    return (vec / norm).astype(np.float32, copy=False)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return -1.0
    return float(np.dot(a, b))


@dataclass
class SpeakerMatchResult:
    profile_id: str
    similarity: float
    created: bool


class SpeakerProfileStore:
    def __init__(
        self,
        *,
        path: str | Path,
        on_status: Callable[[str], None] | None = None,
        max_profiles: int = 512,
    ) -> None:
        self._path = Path(path)
        self._on_status = on_status
        self._max_profiles = max(1, int(max_profiles))
        self._lock = threading.Lock()
        self._profiles: list[dict[str, object]] = []
        self._next_index = 0
        self._load()

    def match_or_create(
        self,
        *,
        embedding: np.ndarray,
        threshold: float,
        observed_label: str,
        duration_seconds: float,
    ) -> SpeakerMatchResult:
        normalized = _normalize_embedding(embedding)
        if normalized.size == 0:
            return SpeakerMatchResult(profile_id="", similarity=-1.0, created=False)

        min_threshold = float(max(0.0, min(0.999, threshold)))
        update_weight = float(max(0.25, min(12.0, duration_seconds)))
        label = str(observed_label or "").strip()

        with self._lock:
            best_index = -1
            best_similarity = -1.0
            for idx, profile in enumerate(self._profiles):
                centroid_raw = profile.get("centroid")
                centroid = _normalize_embedding(np.asarray(centroid_raw or [], dtype=np.float32))
                similarity = _cosine_similarity(centroid, normalized)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_index = idx

            if best_index >= 0 and best_similarity >= min_threshold:
                profile = self._profiles[best_index]
                old_centroid = _normalize_embedding(np.asarray(profile.get("centroid") or [], dtype=np.float32))
                old_weight = float(profile.get("weight", 0.0) or 0.0)
                merged = old_centroid * old_weight + normalized * update_weight
                merged = _normalize_embedding(merged)
                profile["centroid"] = merged.tolist()
                profile["weight"] = float(old_weight + update_weight)
                profile["samples"] = int(profile.get("samples", 0) or 0) + 1
                profile["total_seconds"] = float(profile.get("total_seconds", 0.0) or 0.0) + float(max(0.0, duration_seconds))
                if label:
                    labels = list(profile.get("observed_labels") or [])
                    if label not in labels:
                        labels.append(label)
                    profile["observed_labels"] = labels[-12:]
                self._save_locked()
                return SpeakerMatchResult(
                    profile_id=str(profile.get("id") or ""),
                    similarity=float(best_similarity),
                    created=False,
                )

            if len(self._profiles) >= self._max_profiles:
                return SpeakerMatchResult(profile_id="", similarity=best_similarity, created=False)

            profile_id = f"SPK_{int(self._next_index):03d}"
            self._next_index += 1
            created_profile = {
                "id": profile_id,
                "centroid": normalized.tolist(),
                "weight": float(update_weight),
                "samples": 1,
                "total_seconds": float(max(0.0, duration_seconds)),
                "observed_labels": [label] if label else [],
            }
            self._profiles.append(created_profile)
            self._save_locked()
            return SpeakerMatchResult(
                profile_id=profile_id,
                similarity=best_similarity,
                created=True,
            )

    def profile_count(self) -> int:
        with self._lock:
            return int(len(self._profiles))

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception as exc:
            self._emit(f"speaker-profile load skipped: {exc}")
            return
        if not isinstance(payload, dict):
            return
        raw_profiles = payload.get("profiles")
        if not isinstance(raw_profiles, list):
            return
        loaded_profiles: list[dict[str, object]] = []
        next_index = 0
        for item in raw_profiles:
            if not isinstance(item, dict):
                continue
            profile_id = str(item.get("id") or "").strip()
            centroid = _normalize_embedding(np.asarray(item.get("centroid") or [], dtype=np.float32))
            if not profile_id or centroid.size == 0:
                continue
            loaded_profiles.append(
                {
                    "id": profile_id,
                    "centroid": centroid.tolist(),
                    "weight": float(item.get("weight", 0.0) or 0.0),
                    "samples": int(item.get("samples", 0) or 0),
                    "total_seconds": float(item.get("total_seconds", 0.0) or 0.0),
                    "observed_labels": list(item.get("observed_labels") or []),
                }
            )
            try:
                suffix = int(profile_id.split("_")[-1])
                next_index = max(next_index, suffix + 1)
            except Exception:
                continue
        self._profiles = loaded_profiles
        self._next_index = int(next_index)
        self._emit(f"speaker-profile loaded: profiles={len(self._profiles)}")

    def _save_locked(self) -> None:
        payload = {
            "version": 1,
            "profiles": self._profiles,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        try:
            temp_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temp_path.replace(self._path)
        except Exception as exc:
            try:
                self._path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            except Exception:
                self._emit(f"speaker-profile save skipped: {exc}")

    def _emit(self, message: str) -> None:
        if self._on_status is not None:
            self._on_status(message)
