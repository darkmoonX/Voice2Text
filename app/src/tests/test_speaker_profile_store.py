from __future__ import annotations

import json
from pathlib import Path
import threading
import tempfile
import unittest

import numpy as np

from voice2text.stt.speaker_profiles import SpeakerProfileStore


def _vec(*values: float) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _bare_store(*profiles: dict[str, object]) -> SpeakerProfileStore:
    store = SpeakerProfileStore.__new__(SpeakerProfileStore)
    store._path = Path("unused.json")
    store._on_status = None
    store._max_profiles = 512
    store._soft_speaker_cap = 0
    store._merge_grace_windows = 0
    store._merge_grace_relief = 0.10
    store._merge_preserve_centroid = False
    store._max_exemplars = 1
    store._exemplar_diversity_threshold = 0.90
    store._lock = threading.Lock()
    store._profiles = [dict(profile) for profile in profiles]
    store._candidates = []
    store._next_index = len(profiles)
    store._next_candidate_index = 0
    store._save_locked = lambda: None
    return store


def _profile(profile_id: str, embedding: np.ndarray, *, weight: float = 1.0) -> dict[str, object]:
    return {
        "id": profile_id,
        "centroid": embedding.tolist(),
        "weight": float(weight),
        "samples": 1,
        "total_seconds": 1.0,
        "observed_labels": [profile_id],
    }


def _profile_with_exemplars(
    profile_id: str, exemplars: list[tuple[np.ndarray, float]], *, samples: int = 1, total_seconds: float = 1.0
) -> dict[str, object]:
    exemplar_rows = [{"centroid": vec.tolist(), "weight": float(weight)} for vec, weight in exemplars]
    return {
        "id": profile_id,
        "centroid": exemplar_rows[0]["centroid"],
        "weight": float(sum(weight for _, weight in exemplars)),
        "samples": samples,
        "total_seconds": total_seconds,
        "observed_labels": [profile_id],
        "exemplars": exemplar_rows,
    }


class SpeakerProfileStoreRefreshTests(unittest.TestCase):
    def test_soft_cap_blocks_direct_profile_creation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SpeakerProfileStore(path=Path(tmp) / "profiles.json")
            store.set_soft_speaker_cap(1)

            first = store.match_or_create(
                embedding=_vec(1.0, 0.0),
                threshold=0.99,
                observed_label="A",
                duration_seconds=2.0,
            )
            second = store.match_or_create(
                embedding=_vec(0.0, 1.0),
                threshold=0.99,
                observed_label="B",
                duration_seconds=2.0,
            )

            self.assertTrue(first.created)
            self.assertFalse(second.created)
            self.assertEqual(second.profile_id, "")
            self.assertEqual(store.profile_count(), 1)

    def test_soft_cap_blocks_candidate_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SpeakerProfileStore(path=Path(tmp) / "profiles.json")
            store.set_soft_speaker_cap(1)
            store.match_or_create(
                embedding=_vec(1.0, 0.0),
                threshold=0.99,
                observed_label="A",
                duration_seconds=2.0,
            )

            result = store.match_or_create(
                embedding=_vec(0.0, 1.0),
                threshold=0.99,
                observed_label="B",
                duration_seconds=2.0,
                candidate_min_seconds=1.0,
                candidate_min_samples=1,
            )

            self.assertFalse(result.created)
            self.assertFalse(result.staged)
            self.assertEqual(store.profile_count(), 1)
            self.assertEqual(store.candidate_count(), 0)

    def test_unset_soft_cap_is_unbounded_by_hint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SpeakerProfileStore(path=Path(tmp) / "profiles.json")
            store.set_soft_speaker_cap(0)

            first = store.match_or_create(
                embedding=_vec(1.0, 0.0),
                threshold=0.99,
                observed_label="A",
                duration_seconds=2.0,
            )
            second = store.match_or_create(
                embedding=_vec(0.0, 1.0),
                threshold=0.99,
                observed_label="B",
                duration_seconds=2.0,
            )

            self.assertTrue(first.created)
            self.assertTrue(second.created)
            self.assertEqual(store.profile_count(), 2)

    def test_repeated_soft_cap_call_with_higher_value_raises_effective_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SpeakerProfileStore(path=Path(tmp) / "profiles.json")
            store.set_soft_speaker_cap(1)
            first = store.match_or_create(
                embedding=_vec(1.0, 0.0),
                threshold=0.99,
                observed_label="A",
                duration_seconds=2.0,
            )
            blocked = store.match_or_create(
                embedding=_vec(0.0, 1.0),
                threshold=0.99,
                observed_label="B",
                duration_seconds=2.0,
            )

            store.set_soft_speaker_cap(2)
            second = store.match_or_create(
                embedding=_vec(0.0, 1.0),
                threshold=0.99,
                observed_label="B",
                duration_seconds=2.0,
            )

            self.assertTrue(first.created)
            self.assertFalse(blocked.created)
            self.assertTrue(second.created)
            self.assertEqual(store.profile_count(), 2)

    def test_soft_cap_never_exceeds_hard_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SpeakerProfileStore(path=Path(tmp) / "profiles.json", max_profiles=1)
            store.set_soft_speaker_cap(5)

            first = store.match_or_create(
                embedding=_vec(1.0, 0.0),
                threshold=0.99,
                observed_label="A",
                duration_seconds=2.0,
            )
            second = store.match_or_create(
                embedding=_vec(0.0, 1.0),
                threshold=0.99,
                observed_label="B",
                duration_seconds=2.0,
            )

            self.assertTrue(first.created)
            self.assertFalse(second.created)
            self.assertEqual(store.profile_count(), 1)

    def test_blend_centroid_ema_renormalizes_and_persists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profiles.json"
            store = SpeakerProfileStore(path=path)
            created = store.match_or_create(
                embedding=_vec(1.0, 0.0),
                threshold=0.99,
                observed_label="SPEAKER_00",
                duration_seconds=4.0,
            )

            changed = store.blend_centroid(created.profile_id, _vec(0.0, 1.0), alpha=0.5)

            self.assertTrue(changed)
            profile = store.profile_summaries()[0]
            np.testing.assert_allclose(profile["centroid"], [0.70710677, 0.70710677], rtol=1e-5)
            self.assertEqual(profile["weight"], 4.0)
            self.assertEqual(profile["samples"], 1)
            self.assertEqual(profile["total_seconds"], 4.0)
            payload = json.loads(path.read_text(encoding="utf-8"))
            np.testing.assert_allclose(payload["profiles"][0]["centroid"], [0.70710677, 0.70710677], rtol=1e-5)

    def test_merge_profiles_weight_blends_and_remaps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profiles.json"
            store = SpeakerProfileStore(path=path)
            keep = store.match_or_create(
                embedding=_vec(1.0, 0.0, 0.0),
                threshold=0.99,
                observed_label="A",
                duration_seconds=5.0,
            )
            drop = store.match_or_create(
                embedding=_vec(0.0, 1.0, 0.0),
                threshold=0.99,
                observed_label="B",
                duration_seconds=1.0,
            )

            stats = store.merge_profiles(keep.profile_id, [drop.profile_id])

            self.assertEqual(stats["merged_count"], 1)
            self.assertEqual(stats["remap"], {drop.profile_id: keep.profile_id})
            self.assertEqual(store.profile_count(), 1)
            profile = store.profile_summaries()[0]
            np.testing.assert_allclose(profile["centroid"], [0.9805807, 0.19611613, 0.0], rtol=1e-5)
            self.assertEqual(profile["weight"], 6.0)
            self.assertEqual(profile["samples"], 2)
            self.assertEqual(profile["total_seconds"], 6.0)
            self.assertEqual(profile["observed_labels"], ["A", "B"])
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual([item["id"] for item in payload["profiles"]], [keep.profile_id])

    def test_merge_profiles_preserve_centroid_keeps_survivor_embedding_and_merges_bookkeeping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profiles.json"
            store = SpeakerProfileStore(path=path)
            store.set_merge_preserve_centroid(True)
            keep = store.match_or_create(
                embedding=_vec(1.0, 0.0, 0.0),
                threshold=0.99,
                observed_label="A",
                duration_seconds=5.0,
            )
            drop = store.match_or_create(
                embedding=_vec(0.0, 1.0, 0.0),
                threshold=0.99,
                observed_label="B",
                duration_seconds=1.0,
            )
            before = store.profile_summaries()[0]["centroid"]

            stats = store.merge_profiles(keep.profile_id, [drop.profile_id])

            self.assertEqual(stats["merged_count"], 1)
            profile = store.profile_summaries()[0]
            self.assertEqual(profile["centroid"], before)
            self.assertEqual(profile["centroid"], [1.0, 0.0, 0.0])
            self.assertEqual(profile["weight"], 6.0)
            self.assertEqual(profile["samples"], 2)
            self.assertEqual(profile["total_seconds"], 6.0)
            self.assertEqual(profile["observed_labels"], ["A", "B"])
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["profiles"][0]["centroid"], [1.0, 0.0, 0.0])

    def test_merge_profiles_default_leaves_grace_key_absent(self) -> None:
        store = _bare_store(
            _profile("SPK_000", _vec(1.0, 0.0)),
            _profile("SPK_001", _vec(0.99, 0.1)),
        )
        store.set_merge_grace(0, 0.10)

        stats = store.merge_profiles("SPK_000", ["SPK_001"])

        self.assertEqual(stats["merged_count"], 1)
        profile = store.profile_summaries()[0]
        self.assertNotIn("merge_grace_remaining", profile)

    def test_merge_profiles_sets_grace_on_survivor_when_enabled(self) -> None:
        store = _bare_store(
            _profile("SPK_000", _vec(1.0, 0.0)),
            _profile("SPK_001", _vec(0.99, 0.1)),
        )
        store.set_merge_grace(3, 0.10)

        stats = store.merge_profiles("SPK_000", ["SPK_001"])

        self.assertEqual(stats["merged_count"], 1)
        profiles = store.profile_summaries()
        self.assertEqual(len(profiles), 1)
        self.assertEqual(profiles[0]["id"], "SPK_000")
        self.assertEqual(profiles[0]["merge_grace_remaining"], 3)

    def test_graced_winner_matches_inside_relaxed_threshold_and_decrements(self) -> None:
        store = _bare_store(_profile("SPK_000", _vec(1.0, 0.0)))
        store.set_merge_grace(2, 0.10)
        store._profiles[0]["merge_grace_remaining"] = 2
        query = _vec(0.75, 0.6614378)

        result = store.match_or_create(
            embedding=query,
            threshold=0.80,
            observed_label="SPEAKER_00",
            duration_seconds=1.0,
        )

        self.assertFalse(result.created)
        self.assertEqual(result.profile_id, "SPK_000")
        self.assertAlmostEqual(result.similarity, 0.75, places=5)
        self.assertEqual(store.profile_summaries()[0]["merge_grace_remaining"], 1)

    def test_default_ungraced_match_uses_normal_threshold(self) -> None:
        store = _bare_store(_profile("SPK_000", _vec(1.0, 0.0)))
        query = _vec(0.75, 0.6614378)

        result = store.match_or_create(
            embedding=query,
            threshold=0.80,
            observed_label="SPEAKER_00",
            duration_seconds=1.0,
        )

        self.assertTrue(result.created)
        self.assertNotEqual(result.profile_id, "SPK_000")
        self.assertNotIn("merge_grace_remaining", store.profile_summaries()[0])

    def test_grace_decrements_when_other_profile_wins(self) -> None:
        store = _bare_store(
            _profile("SPK_000", _vec(1.0, 0.0)),
            _profile("SPK_001", _vec(0.0, 1.0)),
        )
        store.set_merge_grace(2, 0.10)
        store._profiles[0]["merge_grace_remaining"] = 2

        result = store.match_or_create(
            embedding=_vec(0.0, 1.0),
            threshold=0.80,
            observed_label="SPEAKER_01",
            duration_seconds=1.0,
        )

        self.assertFalse(result.created)
        self.assertEqual(result.profile_id, "SPK_001")
        self.assertEqual(store.profile_summaries()[0]["merge_grace_remaining"], 1)

    def test_grace_expires_after_configured_match_calls(self) -> None:
        store = _bare_store(_profile("SPK_000", _vec(1.0, 0.0)))
        store.set_merge_grace(1, 0.10)
        store._profiles[0]["merge_grace_remaining"] = 1
        query = _vec(0.75, 0.6614378)

        first = store.match_or_create(
            embedding=query,
            threshold=0.80,
            observed_label="SPEAKER_00",
            duration_seconds=1.0,
        )
        second = store.match_or_create(
            embedding=query,
            threshold=0.80,
            observed_label="SPEAKER_00",
            duration_seconds=1.0,
        )

        self.assertFalse(first.created)
        self.assertEqual(first.profile_id, "SPK_000")
        self.assertTrue(second.created)
        self.assertNotEqual(second.profile_id, "SPK_000")
        self.assertEqual(store.profile_summaries()[0]["merge_grace_remaining"], 0)

    def test_below_relaxed_threshold_still_creates_profile_while_graced(self) -> None:
        store = _bare_store(_profile("SPK_000", _vec(1.0, 0.0)))
        store.set_merge_grace(2, 0.10)
        store._profiles[0]["merge_grace_remaining"] = 2
        query = _vec(0.69, 0.723809)

        result = store.match_or_create(
            embedding=query,
            threshold=0.80,
            observed_label="SPEAKER_00",
            duration_seconds=1.0,
        )

        self.assertTrue(result.created)
        self.assertNotEqual(result.profile_id, "SPK_000")
        self.assertEqual(store.profile_count(), 2)
        self.assertEqual(store.profile_summaries()[0]["merge_grace_remaining"], 1)


class MultiExemplarProfileTests(unittest.TestCase):
    """Round 0061 Phase A: bounded multi-exemplar profile representation."""

    def test_default_max_exemplars_is_byte_identical_to_single_centroid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profiles.json"
            store = SpeakerProfileStore(path=path)
            v1 = _vec(1.0, 0.0)
            v2 = _vec(0.5, 0.8660254)

            created = store.match_or_create(
                embedding=v1, threshold=0.3, observed_label="A", duration_seconds=5.0
            )
            updated = store.match_or_create(
                embedding=v2, threshold=0.3, observed_label="A", duration_seconds=3.0
            )

            self.assertTrue(created.created)
            self.assertFalse(updated.created)
            profile = store.profile_summaries()[0]
            self.assertNotIn("exemplars", profile)
            # Exact legacy weighted-average blend formula: normalize(v1*5 + v2*3).
            expected = v1 * 5.0 + v2 * 3.0
            expected = expected / np.linalg.norm(expected)
            np.testing.assert_allclose(profile["centroid"], expected, rtol=1e-5)
            self.assertEqual(profile["weight"], 8.0)

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertNotIn("exemplars", payload["profiles"][0])

    def test_distinct_observations_create_separate_exemplars_not_one_blend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SpeakerProfileStore(path=Path(tmp) / "profiles.json")
            store.set_max_exemplars(3, 0.90)
            v1 = _vec(1.0, 0.0)
            v2 = _vec(0.5, 0.8660254)  # cos(v1, v2) = 0.5: assign-worthy, below the 0.90 diversity bar

            created = store.match_or_create(
                embedding=v1, threshold=0.3, observed_label="A", duration_seconds=5.0
            )
            second = store.match_or_create(
                embedding=v2, threshold=0.3, observed_label="A", duration_seconds=3.0
            )

            self.assertTrue(created.created)
            self.assertFalse(second.created)
            self.assertEqual(second.profile_id, created.profile_id)
            self.assertAlmostEqual(second.similarity, 0.5, places=5)
            profile = store.profile_summaries()[0]
            self.assertEqual(len(profile["exemplars"]), 2)
            np.testing.assert_allclose(profile["exemplars"][0]["centroid"], v1, rtol=1e-5)
            self.assertEqual(profile["exemplars"][0]["weight"], 5.0)
            np.testing.assert_allclose(profile["exemplars"][1]["centroid"], v2, rtol=1e-5)
            self.assertEqual(profile["exemplars"][1]["weight"], 3.0)

            # A follow-up near v1 matches the v1 exemplar at near-1.0 similarity — proof this
            # is NOT a single blended centroid (which would sit roughly between v1 and v2 and
            # score well below 1.0 against a pure-v1 query).
            near_v1 = store.match_or_create(
                embedding=v1, threshold=0.3, observed_label="A", duration_seconds=2.0
            )
            self.assertAlmostEqual(near_v1.similarity, 1.0, places=5)
            profile = store.profile_summaries()[0]
            self.assertEqual(len(profile["exemplars"]), 2)  # blended into exemplar 0, no new append
            self.assertEqual(profile["exemplars"][0]["weight"], 7.0)
            self.assertEqual(profile["exemplars"][1]["weight"], 3.0)

            # A follow-up near v2 matches the v2 exemplar specifically.
            near_v2 = store.match_or_create(
                embedding=v2, threshold=0.3, observed_label="A", duration_seconds=1.0
            )
            self.assertAlmostEqual(near_v2.similarity, 1.0, places=5)
            profile = store.profile_summaries()[0]
            self.assertEqual(len(profile["exemplars"]), 2)
            self.assertEqual(profile["exemplars"][1]["weight"], 4.0)

    def test_cap_enforced_in_update_path_falls_back_to_blend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SpeakerProfileStore(path=Path(tmp) / "profiles.json")
            store.set_max_exemplars(2, 0.90)
            v1 = _vec(1.0, 0.0)
            v2 = _vec(0.5, 0.8660254)  # 60 degrees from v1, cos=0.5
            v3 = _vec(-0.5, 0.8660254)  # 120 degrees from v1 (cos=-0.5), 60 from v2 (cos=0.5)

            store.match_or_create(embedding=v1, threshold=0.1, observed_label="A", duration_seconds=2.0)
            store.match_or_create(embedding=v2, threshold=0.1, observed_label="A", duration_seconds=2.0)
            profile = store.profile_summaries()[0]
            self.assertEqual(len(profile["exemplars"]), 2)

            third = store.match_or_create(
                embedding=v3, threshold=0.1, observed_label="A", duration_seconds=1.0
            )

            self.assertFalse(third.created)
            profile = store.profile_summaries()[0]
            # At cap: v3 (best match = v2 exemplar, similarity 0.5) blends into that exemplar
            # instead of appending a 3rd — exemplar count never exceeds max_exemplars.
            self.assertEqual(len(profile["exemplars"]), 2)
            np.testing.assert_allclose(profile["exemplars"][0]["centroid"], v1, rtol=1e-5)
            self.assertEqual(profile["exemplars"][0]["weight"], 2.0)
            self.assertEqual(profile["exemplars"][1]["weight"], 3.0)  # 2.0 + 1.0 blended in

    def test_merge_unions_exemplars_and_coalesces_only_the_closest_pair(self) -> None:
        a1, a1_w = _vec(1.0, 0.0), 2.0
        a2, a2_w = _vec(0.0, 1.0), 3.0
        b1, b1_w = _vec(0.99619469, 0.08715574), 4.0  # 5 degrees from a1: the closest pair overall
        b2, b2_w = _vec(-1.0, 0.0), 1.0

        store = _bare_store(
            _profile_with_exemplars("A", [(a1, a1_w), (a2, a2_w)]),
            _profile_with_exemplars("B", [(b1, b1_w), (b2, b2_w)]),
        )
        store.set_max_exemplars(3, 0.90)

        stats = store.merge_profiles("A", ["B"])

        self.assertEqual(stats["merged_count"], 1)
        profile = store.profile_summaries()[0]
        exemplars = profile["exemplars"]
        self.assertEqual(len(exemplars), 3)

        # a1 and b1 (the closest pair, cos≈0.9962) coalesce; a2 and b2 survive untouched.
        # Deletion order (higher index first) + append means the result is [a2, b2, merged].
        np.testing.assert_allclose(exemplars[0]["centroid"], a2, rtol=1e-5)
        self.assertEqual(exemplars[0]["weight"], a2_w)
        np.testing.assert_allclose(exemplars[1]["centroid"], b2, rtol=1e-5)
        self.assertEqual(exemplars[1]["weight"], b2_w)

        merged_centroid = a1 * a1_w + b1 * b1_w
        merged_centroid = merged_centroid / np.linalg.norm(merged_centroid)
        np.testing.assert_allclose(exemplars[2]["centroid"], merged_centroid, rtol=1e-4)
        self.assertEqual(exemplars[2]["weight"], a1_w + b1_w)
        self.assertEqual(profile["weight"], a1_w + a2_w + b1_w + b2_w)

    def test_multi_exemplar_profile_persists_and_reloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profiles.json"
            store = SpeakerProfileStore(path=path)
            store.set_max_exemplars(3, 0.90)
            v1 = _vec(1.0, 0.0)
            v2 = _vec(0.5, 0.8660254)
            store.match_or_create(embedding=v1, threshold=0.3, observed_label="A", duration_seconds=5.0)
            store.match_or_create(embedding=v2, threshold=0.3, observed_label="A", duration_seconds=3.0)

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["profiles"][0]["exemplars"]), 2)

            reloaded = SpeakerProfileStore(path=path)
            reloaded.set_max_exemplars(3, 0.90)
            reloaded_profile = reloaded.profile_summaries()[0]
            self.assertEqual(len(reloaded_profile["exemplars"]), 2)
            np.testing.assert_allclose(reloaded_profile["exemplars"][0]["centroid"], v1, rtol=1e-5)
            np.testing.assert_allclose(reloaded_profile["exemplars"][1]["centroid"], v2, rtol=1e-5)

    def test_visible_profile_alias_uses_best_exemplar_similarity(self) -> None:
        mature = _profile_with_exemplars(
            "MATURE",
            [(_vec(0.0, 1.0), 20.0), (_vec(1.0, 0.0), 20.0)],
            samples=20,
            total_seconds=24.0,
        )
        low_evidence = _profile("LOW", _vec(0.99, 0.14106736), weight=1.0)
        low_evidence["samples"] = 1
        low_evidence["total_seconds"] = 2.0
        store = _bare_store(mature, low_evidence)
        store.set_max_exemplars(3, 0.90)

        alias = store.visible_profile_alias(
            profile_id="LOW",
            embedding=_vec(0.99, 0.14106736),
            min_total_seconds=24.0,
            min_samples=16,
            similarity_threshold=0.80,
        )

        # The query is near-identical to MATURE's SECOND exemplar (1,0), not its first (0,1) —
        # this only aliases correctly if the comparison checks every exemplar, not just the
        # profile's first/legacy centroid.
        self.assertTrue(alias["aliased"])
        self.assertEqual(alias["alias"], "MATURE")

    def test_reconcile_uses_best_cross_exemplar_similarity(self) -> None:
        left = _profile_with_exemplars(
            "LEFT", [(_vec(0.0, 1.0), 2.0), (_vec(1.0, 0.0), 2.0)]
        )
        right = _profile_with_exemplars("RIGHT", [(_vec(0.99619469, 0.08715574), 2.0)])
        store = _bare_store(left, right)
        store.set_max_exemplars(3, 0.90)

        stats = store.reconcile_similar_profiles(threshold=0.98)

        # RIGHT's only exemplar is ~5 degrees from LEFT's SECOND exemplar (1,0), not its
        # first (0,1) — this only merges if reconcile checks every exemplar pair.
        self.assertEqual(stats["merged_count"], 1)
        self.assertEqual(stats["remap"], {"RIGHT": "LEFT"})


if __name__ == "__main__":
    unittest.main()
