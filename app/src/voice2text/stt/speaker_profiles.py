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
        self._max_exemplars = 1
        self._exemplar_diversity_threshold = 0.90
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
            best_exemplar_index = -1
            if self._max_exemplars > 1:
                best_index, best_similarity, best_exemplar_index = self._best_profile_match_locked(normalized)
            else:
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
                    if self._max_exemplars > 1:
                        self._update_exemplar_locked(
                            profile,
                            normalized=normalized,
                            update_weight=update_weight,
                            exemplar_index=best_exemplar_index,
                            similarity=best_similarity,
                        )
                    else:
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

    def set_max_exemplars(self, max_exemplars: int, diversity_threshold: float) -> None:
        with self._lock:
            self._max_exemplars = max(1, int(max_exemplars))
            self._exemplar_diversity_threshold = max(0.0, min(0.999, float(diversity_threshold)))

    def _profile_exemplars_locked(self, profile: dict[str, object]) -> list[dict[str, object]]:
        """Round 0061: a profile's representative embeddings. Falls back to a synthesized
        single-entry list from the legacy centroid/weight fields for any profile that predates
        multi-exemplar mode (or was created while it was off) — read-only, does not mutate."""
        exemplars = profile.get("exemplars")
        if isinstance(exemplars, list) and exemplars:
            return exemplars
        return [
            {
                "centroid": profile.get("centroid") or [],
                "weight": float(profile.get("weight", 0.0) or 0.0),
            }
        ]

    def _best_profile_match_locked(self, normalized: np.ndarray) -> tuple[int, float, int]:
        best_index = -1
        best_similarity = -1.0
        best_exemplar_index = -1
        for idx, profile in enumerate(self._profiles):
            for ex_idx, exemplar in enumerate(self._profile_exemplars_locked(profile)):
                centroid = _normalize_embedding(np.asarray(exemplar.get("centroid") or [], dtype=np.float32))
                similarity = _cosine_similarity(centroid, normalized)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_index = idx
                    best_exemplar_index = ex_idx
        return best_index, best_similarity, best_exemplar_index

    def _best_exemplar_similarity_locked(self, profile: dict[str, object], query: np.ndarray) -> float:
        best = -1.0
        for exemplar in self._profile_exemplars_locked(profile):
            centroid = _normalize_embedding(np.asarray(exemplar.get("centroid") or [], dtype=np.float32))
            similarity = _cosine_similarity(centroid, query)
            if similarity > best:
                best = similarity
        return best

    def _profile_pair_similarity_locked(self, left: dict[str, object], right: dict[str, object]) -> float:
        best = -1.0
        for le in self._profile_exemplars_locked(left):
            left_centroid = _normalize_embedding(np.asarray(le.get("centroid") or [], dtype=np.float32))
            for re in self._profile_exemplars_locked(right):
                right_centroid = _normalize_embedding(np.asarray(re.get("centroid") or [], dtype=np.float32))
                similarity = _cosine_similarity(left_centroid, right_centroid)
                if similarity > best:
                    best = similarity
        return best

    def _sync_profile_aggregate_from_exemplars(self, profile: dict[str, object]) -> None:
        """Recompute the legacy top-level centroid/weight fields as a weighted-average
        aggregate of the profile's exemplars, for backward-compatible display/persistence
        readers. Matching/update/merge math never reads these fields once exemplars exist —
        they are derived, not authoritative."""
        exemplars = profile.get("exemplars")
        if not isinstance(exemplars, list) or not exemplars:
            return
        total_weight = 0.0
        acc: np.ndarray | None = None
        for exemplar in exemplars:
            weight = float(exemplar.get("weight", 0.0) or 0.0)
            centroid = _normalize_embedding(np.asarray(exemplar.get("centroid") or [], dtype=np.float32))
            if centroid.size == 0:
                continue
            if acc is None:
                acc = np.zeros_like(centroid)
            acc = acc + centroid * weight
            total_weight += weight
        if acc is None:
            return
        profile["centroid"] = _normalize_embedding(acc).tolist()
        profile["weight"] = float(total_weight)

    def _update_exemplar_locked(
        self,
        profile: dict[str, object],
        *,
        normalized: np.ndarray,
        update_weight: float,
        exemplar_index: int,
        similarity: float,
    ) -> None:
        exemplars = profile.get("exemplars")
        if not isinstance(exemplars, list) or not exemplars:
            exemplars = [
                {
                    "centroid": list(profile.get("centroid") or []),
                    "weight": float(profile.get("weight", 0.0) or 0.0),
                }
            ]
            profile["exemplars"] = exemplars
            exemplar_index = 0
        exemplar_index = max(0, min(int(exemplar_index), len(exemplars) - 1))
        if similarity >= self._exemplar_diversity_threshold or len(exemplars) >= self._max_exemplars:
            target = exemplars[exemplar_index]
            old_centroid = _normalize_embedding(np.asarray(target.get("centroid") or [], dtype=np.float32))
            old_weight = float(target.get("weight", 0.0) or 0.0)
            merged = _normalize_embedding(old_centroid * old_weight + normalized * update_weight)
            target["centroid"] = merged.tolist()
            target["weight"] = float(old_weight + update_weight)
        else:
            exemplars.append({"centroid": normalized.tolist(), "weight": float(update_weight)})
        self._sync_profile_aggregate_from_exemplars(profile)

    def _merge_exemplars_locked(self, keep: dict[str, object], drop: dict[str, object]) -> None:
        """Union both profiles' exemplar sets; if over cap, repeatedly coalesce the most
        similar PAIR (weighted-average blend, same math as the legacy single-centroid merge)
        until back within max_exemplars. Ties in "most similar pair" resolve to the first pair
        found in union order (keep's exemplars first, then drop's, both in their existing
        order) — a deterministic but otherwise arbitrary tie-break."""
        union: list[dict[str, object]] = [
            {"centroid": list(ex.get("centroid") or []), "weight": float(ex.get("weight", 0.0) or 0.0)}
            for ex in self._profile_exemplars_locked(keep) + self._profile_exemplars_locked(drop)
        ]
        while len(union) > self._max_exemplars and len(union) > 1:
            best_pair: tuple[int, int] | None = None
            best_similarity = -2.0
            for i in range(len(union)):
                centroid_i = _normalize_embedding(np.asarray(union[i].get("centroid") or [], dtype=np.float32))
                for j in range(i + 1, len(union)):
                    centroid_j = _normalize_embedding(np.asarray(union[j].get("centroid") or [], dtype=np.float32))
                    similarity = _cosine_similarity(centroid_i, centroid_j)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_pair = (i, j)
            if best_pair is None:
                break
            i, j = best_pair
            a_weight = float(union[i].get("weight", 0.0) or 0.0)
            b_weight = float(union[j].get("weight", 0.0) or 0.0)
            a_centroid = _normalize_embedding(np.asarray(union[i].get("centroid") or [], dtype=np.float32))
            b_centroid = _normalize_embedding(np.asarray(union[j].get("centroid") or [], dtype=np.float32))
            merged_centroid = _normalize_embedding(a_centroid * a_weight + b_centroid * b_weight)
            merged_entry = {"centroid": merged_centroid.tolist(), "weight": float(a_weight + b_weight)}
            del union[j]
            del union[i]
            union.append(merged_entry)
        keep["exemplars"] = union
        self._sync_profile_aggregate_from_exemplars(keep)

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
                if self._max_exemplars > 1:
                    similarity = self._best_exemplar_similarity_locked(profile, query)
                else:
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
                    if self._max_exemplars <= 1:
                        left_centroid = _normalize_embedding(np.asarray(left.get("centroid") or [], dtype=np.float32))
                    for j in range(i + 1, len(self._profiles)):
                        right = self._profiles[j]
                        if self._max_exemplars > 1:
                            similarity = self._profile_pair_similarity_locked(left, right)
                        else:
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
        # Round 0061: multi-exemplar merge takes precedence over round 0057's
        # merge-preserve-centroid flag when both are enabled — union-then-coalesce is a
        # strict superset of "preserve the survivor centroid" (it preserves BOTH inputs as
        # separate exemplars whenever there's room, only coalescing what doesn't fit).
        if self._max_exemplars > 1:
            self._merge_exemplars_locked(keep, drop)
        elif not self._merge_preserve_centroid:
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
        if self._max_exemplars > 1:
            created_profile["exemplars"] = [
                {"centroid": created_profile["centroid"], "weight": created_profile["weight"]}
            ]
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
            loaded_profile: dict[str, object] = {
                "id": profile_id,
                "centroid": centroid.tolist(),
                "weight": float(item.get("weight", 0.0) or 0.0),
                "samples": int(item.get("samples", 0) or 0),
                "total_seconds": float(item.get("total_seconds", 0.0) or 0.0),
                "observed_labels": list(item.get("observed_labels") or []),
            }
            raw_exemplars = item.get("exemplars")
            if isinstance(raw_exemplars, list) and raw_exemplars:
                loaded_exemplars: list[dict[str, object]] = []
                for raw_exemplar in raw_exemplars:
                    if not isinstance(raw_exemplar, dict):
                        continue
                    exemplar_centroid = _normalize_embedding(
                        np.asarray(raw_exemplar.get("centroid") or [], dtype=np.float32)
                    )
                    if exemplar_centroid.size == 0:
                        continue
                    loaded_exemplars.append(
                        {
                            "centroid": exemplar_centroid.tolist(),
                            "weight": float(raw_exemplar.get("weight", 0.0) or 0.0),
                        }
                    )
                if loaded_exemplars:
                    loaded_profile["exemplars"] = loaded_exemplars
            loaded_profiles.append(loaded_profile)
            try:
                suffix = int(profile_id.split("_")[-1])
                next_index = max(next_index, suffix + 1)
            except Exception:
                continue
        self._profiles = loaded_profiles
        self._next_index = int(next_index)
        self._emit(f"speaker-profile loaded: profiles={len(self._profiles)}")

    def _save_locked(self) -> None:
        profiles_payload: list[dict[str, object]] = []
        for profile in self._profiles:
            item = dict(profile)
            if isinstance(item.get("exemplars"), list) and item["exemplars"]:
                self._sync_profile_aggregate_from_exemplars(item)
            profiles_payload.append(item)
        payload = {
            "version": 1,
            "profiles": profiles_payload,
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
