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
    staged: bool = False
    candidate_id: str = ""
    candidate_total_seconds: float = 0.0
    promoted: bool = False


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
        self._soft_speaker_cap = 0
        self._merge_grace_windows = 0
        self._merge_grace_relief = 0.10
        self._merge_preserve_centroid = False
        self._lock = threading.Lock()
        self._profiles: list[dict[str, object]] = []
        self._candidates: list[dict[str, object]] = []
        self._next_index = 0
        self._next_candidate_index = 0
        self._load()

    def match_or_create(
        self,
        *,
        embedding: np.ndarray,
        threshold: float,
        observed_label: str,
        duration_seconds: float,
        candidate_min_seconds: float = 0.0,
        candidate_min_samples: int = 1,
        candidate_threshold: float | None = None,
        update_threshold: float | None = None,
        allow_update: bool = True,
    ) -> SpeakerMatchResult:
        normalized = _normalize_embedding(embedding)
        if normalized.size == 0:
            return SpeakerMatchResult(profile_id="", similarity=-1.0, created=False)

        min_threshold = float(max(0.0, min(0.999, threshold)))
        # Round 0045 step 1b: decouple ASSIGN from UPDATE. A window is assigned to (displayed
        # as) the nearest profile at min_threshold, but its embedding is only blended INTO the
        # centroid when similarity >= update_threshold. This keeps a profile's centroid "pure"
        # so it does not drift/blend toward other speakers and snowball into absorbing everyone.
        # update_threshold=None preserves legacy behavior (update == assign).
        effective_update_threshold = (
            float(max(0.0, min(0.999, update_threshold))) if update_threshold is not None else min_threshold
        )
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

            effective_min_threshold = min_threshold
            if best_index >= 0:
                winner = self._profiles[best_index]
                if int(winner.get("merge_grace_remaining", 0) or 0) > 0:
                    effective_min_threshold = max(0.0, min_threshold - float(self._merge_grace_relief))

            if not allow_update:
                # Read-only quality-gate path: identify against an existing centroid for
                # *display* only. Never average into a centroid, create a profile, or stage a
                # candidate — so a low-quality clip cannot pollute the learned identities.
                self._decrement_merge_grace_locked()
                if best_index >= 0 and best_similarity >= effective_min_threshold:
                    profile = self._profiles[best_index]
                    return SpeakerMatchResult(
                        profile_id=str(profile.get("id") or ""),
                        similarity=float(best_similarity),
                        created=False,
                    )
                return SpeakerMatchResult(profile_id="", similarity=float(best_similarity), created=False)

            if best_index >= 0 and best_similarity >= effective_min_threshold:
                profile = self._profiles[best_index]
                self._decrement_merge_grace_locked()
                # Always count the observation (maturity/display), but only blend the embedding
                # into the centroid when the match is strong enough (>= update threshold).
                if best_similarity >= effective_update_threshold:
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

            effective_cap = self._effective_profile_cap_locked()
            if len(self._profiles) >= effective_cap:
                self._decrement_merge_grace_locked()
                return SpeakerMatchResult(profile_id="", similarity=best_similarity, created=False)

            min_candidate_seconds = float(max(0.0, candidate_min_seconds))
            if min_candidate_seconds > 0.0:
                self._decrement_merge_grace_locked()
                return self._stage_or_promote_candidate_locked(
                    normalized=normalized,
                    update_weight=update_weight,
                    label=label,
                    duration_seconds=duration_seconds,
                    candidate_min_seconds=min_candidate_seconds,
                    candidate_min_samples=max(1, int(candidate_min_samples)),
                    candidate_threshold=(
                        float(max(0.0, min(0.999, candidate_threshold)))
                        if candidate_threshold is not None
                        else min_threshold
                    ),
                    best_profile_similarity=best_similarity,
                )

            self._decrement_merge_grace_locked()
            return self._create_profile_locked(
                normalized=normalized,
                update_weight=update_weight,
                label=label,
                duration_seconds=duration_seconds,
                similarity=best_similarity,
                promoted=False,
            )

    def profile_count(self) -> int:
        with self._lock:
            return int(len(self._profiles))

    def set_soft_speaker_cap(self, cap: int) -> None:
        with self._lock:
            self._soft_speaker_cap = max(0, int(cap))

    def set_merge_grace(self, windows: int, relief: float) -> None:
        with self._lock:
            self._merge_grace_windows = max(0, int(windows))
            self._merge_grace_relief = max(0.0, float(relief))

    def set_merge_preserve_centroid(self, enabled: bool) -> None:
        with self._lock:
            self._merge_preserve_centroid = bool(enabled)

    def _effective_profile_cap_locked(self) -> int:
        if self._soft_speaker_cap <= 0:
            return int(self._max_profiles)
        return int(min(self._max_profiles, self._soft_speaker_cap))

    def _decrement_merge_grace_locked(self) -> None:
        for profile in self._profiles:
            remaining = int(profile.get("merge_grace_remaining", 0) or 0)
            if remaining > 0:
                profile["merge_grace_remaining"] = remaining - 1

    def candidate_count(self) -> int:
        with self._lock:
            return int(len(self._candidates))

    def candidate_summaries(self) -> list[dict[str, object]]:
        with self._lock:
            return [
                {
                    "id": str(item.get("id") or ""),
                    "samples": int(item.get("samples", 0) or 0),
                    "total_seconds": float(item.get("total_seconds", 0.0) or 0.0),
                    "observed_labels": list(item.get("observed_labels") or []),
                }
                for item in self._candidates
            ]

    def profile_summaries(self) -> list[dict[str, object]]:
        with self._lock:
            return [dict(profile) for profile in self._profiles]

    def blend_centroid(self, profile_id: str, centroid: np.ndarray, alpha: float) -> bool:
        normalized_id = str(profile_id or "").strip()
        normalized = _normalize_embedding(centroid)
        if not normalized_id or normalized.size == 0:
            return False
        trust = float(max(0.0, min(1.0, alpha)))
        with self._lock:
            for profile in self._profiles:
                if str(profile.get("id") or "") != normalized_id:
                    continue
                old_centroid = _normalize_embedding(np.asarray(profile.get("centroid") or [], dtype=np.float32))
                if old_centroid.size == 0 or old_centroid.size != normalized.size:
                    return False
                blended = _normalize_embedding(normalized * trust + old_centroid * (1.0 - trust))
                profile["centroid"] = blended.tolist()
                self._save_locked()
                return True
        return False

    def merge_profiles(self, keep_id: str, drop_ids: list[str]) -> dict[str, object]:
        normalized_keep = str(keep_id or "").strip()
        normalized_drops = [
            str(item or "").strip()
            for item in drop_ids
            if str(item or "").strip() and str(item or "").strip() != normalized_keep
        ]
        with self._lock:
            remap: dict[str, str] = {}
            merged_rows: list[dict[str, object]] = []
            if not normalized_keep or not normalized_drops:
                return {
                    "merged_count": 0,
                    "remap": {},
                    "merged": [],
                    "profile_count": int(len(self._profiles)),
                }
            for drop_id in normalized_drops:
                merged = self._merge_profile_locked(normalized_keep, drop_id, similarity=None)
                if not merged:
                    continue
                remap[drop_id] = normalized_keep
                for old, new in list(remap.items()):
                    if new == drop_id:
                        remap[old] = normalized_keep
                merged_rows.append(merged)
            if remap:
                self._save_locked()
            return {
                "merged_count": int(len(merged_rows)),
                "remap": dict(remap),
                "merged": merged_rows,
                "profile_count": int(len(self._profiles)),
            }

    def visible_profile_alias(
        self,
        *,
        profile_id: str,
        embedding: np.ndarray,
        min_total_seconds: float,
        min_samples: int,
        similarity_threshold: float,
    ) -> dict[str, object]:
        """Return a mature display profile for a low-evidence profile when safe.

        Raw profiles are intentionally more granular than visible speaker labels:
        a new profile can exist for diagnostics while still being displayed as a
        mature neighbouring identity if its embedding is close enough.
        """
        normalized_id = str(profile_id or "").strip()
        if not normalized_id:
            return {"profile_id": "", "alias": "", "aliased": False, "similarity": -1.0, "mature": False}
        query = _normalize_embedding(embedding)
        if query.size == 0:
            return {
                "profile_id": normalized_id,
                "alias": normalized_id,
                "aliased": False,
                "similarity": -1.0,
                "mature": False,
            }
        min_seconds = float(max(0.0, min_total_seconds))
        min_sample_count = int(max(1, min_samples))
        threshold = float(max(0.0, min(0.999, similarity_threshold)))
        with self._lock:
            target: dict[str, object] | None = None
            mature: list[dict[str, object]] = []
            for profile in self._profiles:
                pid = str(profile.get("id") or "")
                is_mature = (
                    float(profile.get("total_seconds", 0.0) or 0.0) >= min_seconds
                    and int(profile.get("samples", 0) or 0) >= min_sample_count
                )
                if pid == normalized_id:
                    target = profile
                elif is_mature:
                    mature.append(profile)
            if target is None:
                return {
                    "profile_id": normalized_id,
                    "alias": normalized_id,
                    "aliased": False,
                    "similarity": -1.0,
                    "mature": False,
                }
            target_mature = (
                float(target.get("total_seconds", 0.0) or 0.0) >= min_seconds
                and int(target.get("samples", 0) or 0) >= min_sample_count
            )
            if target_mature or not mature:
                return {
                    "profile_id": normalized_id,
                    "alias": normalized_id,
                    "aliased": False,
                    "similarity": 1.0 if target_mature else -1.0,
                    "mature": bool(target_mature),
                }
            best_id = ""
            best_similarity = -1.0
            for profile in mature:
                centroid = _normalize_embedding(np.asarray(profile.get("centroid") or [], dtype=np.float32))
                similarity = _cosine_similarity(centroid, query)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_id = str(profile.get("id") or "")
            if best_id and best_similarity >= threshold:
                return {
                    "profile_id": normalized_id,
                    "alias": best_id,
                    "aliased": True,
                    "similarity": float(best_similarity),
                    "mature": False,
                }
            return {
                "profile_id": normalized_id,
                "alias": normalized_id,
                "aliased": False,
                "similarity": float(best_similarity),
                "mature": False,
            }

    def reconcile_similar_profiles(self, *, threshold: float) -> dict[str, object]:
        """Merge highly similar profiles and return a remap from removed IDs to kept IDs."""
        min_threshold = float(max(0.0, min(0.999, threshold)))
        with self._lock:
            remap: dict[str, str] = {}
            merged_rows: list[dict[str, object]] = []
            changed = True
            while changed:
                changed = False
                best_pair: tuple[int, int, float] | None = None
                for i in range(len(self._profiles)):
                    left = self._profiles[i]
                    left_centroid = _normalize_embedding(np.asarray(left.get("centroid") or [], dtype=np.float32))
                    for j in range(i + 1, len(self._profiles)):
                        right = self._profiles[j]
                        right_centroid = _normalize_embedding(np.asarray(right.get("centroid") or [], dtype=np.float32))
                        similarity = _cosine_similarity(left_centroid, right_centroid)
                        if similarity < min_threshold:
                            continue
                        if best_pair is None or similarity > best_pair[2]:
                            best_pair = (i, j, similarity)
                if best_pair is None:
                    break
                keep_index, drop_index, similarity = best_pair
                keep = self._profiles[keep_index]
                drop = self._profiles[drop_index]
                keep_id = str(keep.get("id") or "")
                drop_id = str(drop.get("id") or "")
                if not keep_id or not drop_id or keep_id == drop_id:
                    break

                merged = self._merge_profile_locked(keep_id, drop_id, similarity=similarity)
                if not merged:
                    break
                remap[drop_id] = keep_id
                for old, new in list(remap.items()):
                    if new == drop_id:
                        remap[old] = keep_id
                merged_rows.append(merged)
                changed = True

            if remap:
                self._save_locked()
            return {
                "threshold": float(min_threshold),
                "merged_count": int(len(merged_rows)),
                "remap": dict(remap),
                "merged": merged_rows,
                "profile_count": int(len(self._profiles)),
            }

    def _merge_profile_locked(self, keep_id: str, drop_id: str, similarity: float | None) -> dict[str, object] | None:
        keep_index = -1
        drop_index = -1
        for idx, profile in enumerate(self._profiles):
            pid = str(profile.get("id") or "")
            if pid == keep_id:
                keep_index = idx
            elif pid == drop_id:
                drop_index = idx
        if keep_index < 0 or drop_index < 0 or keep_index == drop_index:
            return None
        keep = self._profiles[keep_index]
        drop = self._profiles[drop_index]
        keep_weight = float(keep.get("weight", 0.0) or 0.0)
        drop_weight = float(drop.get("weight", 0.0) or 0.0)
        keep_centroid = _normalize_embedding(np.asarray(keep.get("centroid") or [], dtype=np.float32))
        drop_centroid = _normalize_embedding(np.asarray(drop.get("centroid") or [], dtype=np.float32))
        if keep_centroid.size == 0 or drop_centroid.size == 0 or keep_centroid.size != drop_centroid.size:
            return None
        if not self._merge_preserve_centroid:
            merged = _normalize_embedding(keep_centroid * keep_weight + drop_centroid * drop_weight)
            keep["centroid"] = merged.tolist()
        keep["weight"] = float(keep_weight + drop_weight)
        keep["samples"] = int(keep.get("samples", 0) or 0) + int(drop.get("samples", 0) or 0)
        keep["total_seconds"] = (
            float(keep.get("total_seconds", 0.0) or 0.0)
            + float(drop.get("total_seconds", 0.0) or 0.0)
        )
        labels = list(keep.get("observed_labels") or [])
        for label in list(drop.get("observed_labels") or []):
            if label not in labels:
                labels.append(label)
        keep["observed_labels"] = labels[-12:]
        if self._merge_grace_windows > 0:
            keep["merge_grace_remaining"] = int(self._merge_grace_windows)
        del self._profiles[drop_index]
        row: dict[str, object] = {
            "from": str(drop_id),
            "to": str(keep_id),
        }
        if similarity is not None:
            row["similarity"] = float(similarity)
        return row

    def _stage_or_promote_candidate_locked(
        self,
        *,
        normalized: np.ndarray,
        update_weight: float,
        label: str,
        duration_seconds: float,
        candidate_min_seconds: float,
        candidate_min_samples: int,
        candidate_threshold: float,
        best_profile_similarity: float,
    ) -> SpeakerMatchResult:
        best_candidate_index = -1
        best_candidate_similarity = -1.0
        for idx, candidate in enumerate(self._candidates):
            centroid = _normalize_embedding(np.asarray(candidate.get("centroid") or [], dtype=np.float32))
            similarity = _cosine_similarity(centroid, normalized)
            if similarity > best_candidate_similarity:
                best_candidate_similarity = similarity
                best_candidate_index = idx

        if best_candidate_index >= 0 and best_candidate_similarity >= candidate_threshold:
            candidate = self._candidates[best_candidate_index]
            old_centroid = _normalize_embedding(np.asarray(candidate.get("centroid") or [], dtype=np.float32))
            old_weight = float(candidate.get("weight", 0.0) or 0.0)
            merged = _normalize_embedding(old_centroid * old_weight + normalized * update_weight)
            candidate["centroid"] = merged.tolist()
            candidate["weight"] = float(old_weight + update_weight)
            candidate["samples"] = int(candidate.get("samples", 0) or 0) + 1
            candidate["total_seconds"] = (
                float(candidate.get("total_seconds", 0.0) or 0.0)
                + float(max(0.0, duration_seconds))
            )
            if label:
                labels = list(candidate.get("observed_labels") or [])
                if label not in labels:
                    labels.append(label)
                candidate["observed_labels"] = labels[-12:]
            total_seconds = float(candidate.get("total_seconds", 0.0) or 0.0)
            samples = int(candidate.get("samples", 0) or 0)
            if total_seconds >= candidate_min_seconds and samples >= max(1, int(candidate_min_samples)):
                del self._candidates[best_candidate_index]
                return self._create_profile_locked(
                    normalized=_normalize_embedding(np.asarray(candidate.get("centroid") or [], dtype=np.float32)),
                    update_weight=float(candidate.get("weight", 0.0) or 0.0),
                    label=label,
                    duration_seconds=total_seconds,
                    similarity=best_candidate_similarity,
                    promoted=True,
                    observed_labels=list(candidate.get("observed_labels") or []),
                    samples=samples,
                )
            return SpeakerMatchResult(
                profile_id="",
                similarity=float(max(best_profile_similarity, best_candidate_similarity)),
                created=False,
                staged=True,
                candidate_id=str(candidate.get("id") or ""),
                candidate_total_seconds=total_seconds,
            )

        candidate_id = f"CAND_{int(self._next_candidate_index):03d}"
        self._next_candidate_index += 1
        total_seconds = float(max(0.0, duration_seconds))
        if total_seconds >= candidate_min_seconds and max(1, int(candidate_min_samples)) <= 1:
            return self._create_profile_locked(
                normalized=normalized,
                update_weight=update_weight,
                label=label,
                duration_seconds=duration_seconds,
                similarity=best_profile_similarity,
                promoted=True,
            )
        self._candidates.append(
            {
                "id": candidate_id,
                "centroid": normalized.tolist(),
                "weight": float(update_weight),
                "samples": 1,
                "total_seconds": total_seconds,
                "observed_labels": [label] if label else [],
            }
        )
        return SpeakerMatchResult(
            profile_id="",
            similarity=float(max(best_profile_similarity, best_candidate_similarity)),
            created=False,
            staged=True,
            candidate_id=candidate_id,
            candidate_total_seconds=total_seconds,
        )

    def _create_profile_locked(
        self,
        *,
        normalized: np.ndarray,
        update_weight: float,
        label: str,
        duration_seconds: float,
        similarity: float,
        promoted: bool,
        observed_labels: list[object] | None = None,
        samples: int = 1,
    ) -> SpeakerMatchResult:
        profile_id = f"SPK_{int(self._next_index):03d}"
        self._next_index += 1
        labels: list[object] = list(observed_labels or [])
        if label and label not in labels:
            labels.append(label)
        created_profile = {
            "id": profile_id,
            "centroid": _normalize_embedding(normalized).tolist(),
            "weight": float(update_weight),
            "samples": int(max(1, samples)),
            "total_seconds": float(max(0.0, duration_seconds)),
            "observed_labels": labels[-12:],
        }
        self._profiles.append(created_profile)
        self._save_locked()
        return SpeakerMatchResult(
            profile_id=profile_id,
            similarity=float(similarity),
            created=True,
            promoted=bool(promoted),
        )

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
