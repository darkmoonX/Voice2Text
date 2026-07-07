from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from voice2text.stt.speaker_profiles import SpeakerProfileStore
from voice2text.stt.speaker_identity import SpeakerIdentityEngine


def _policy_engine(
    *,
    rt_candidate_seconds: float = 6.0,
    rt_candidate_samples: int = 8,
    rt_visible_seconds: float = 24.0,
    rt_visible_samples: int = 16,
) -> SpeakerIdentityEngine:
    """Build a bare engine carrying only the realtime maturity-floor knobs.

    The policy methods are instance methods reading these config-driven floors;
    __new__ bypasses __init__ so no embedding backend is loaded.
    """
    eng = SpeakerIdentityEngine.__new__(SpeakerIdentityEngine)
    eng._rt_candidate_seconds = float(rt_candidate_seconds)
    eng._rt_candidate_samples = int(rt_candidate_samples)
    eng._rt_visible_seconds = float(rt_visible_seconds)
    eng._rt_visible_samples = int(rt_visible_samples)
    return eng


class SpeakerProfileReconciliationTests(unittest.TestCase):
    def test_low_duration_new_speaker_is_staged_before_profile_creation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profiles.json"
            store = SpeakerProfileStore(path=path)
            voice = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

            first = store.match_or_create(
                embedding=voice,
                threshold=0.80,
                observed_label="SPEAKER_00",
                duration_seconds=1.0,
                candidate_min_seconds=3.0,
                candidate_threshold=0.75,
            )

            self.assertFalse(first.created)
            self.assertTrue(first.staged)
            self.assertEqual(first.profile_id, "")
            self.assertEqual(store.profile_count(), 0)
            self.assertEqual(store.candidate_count(), 1)

            second = store.match_or_create(
                embedding=voice,
                threshold=0.80,
                observed_label="SPEAKER_00",
                duration_seconds=2.0,
                candidate_min_seconds=3.0,
                candidate_threshold=0.75,
            )

            self.assertTrue(second.created)
            self.assertTrue(second.promoted)
            self.assertEqual(second.profile_id, "SPK_000")
            self.assertEqual(store.profile_count(), 1)
            self.assertEqual(store.candidate_count(), 0)

    def test_long_single_observation_does_not_immediately_promote_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profiles.json"
            store = SpeakerProfileStore(path=path)
            voice = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

            first = store.match_or_create(
                embedding=voice,
                threshold=0.80,
                observed_label="SPEAKER_00",
                duration_seconds=6.0,
                candidate_min_seconds=4.0,
                candidate_min_samples=2,
                candidate_threshold=0.75,
            )

            self.assertFalse(first.created)
            self.assertTrue(first.staged)
            self.assertEqual(first.profile_id, "")
            self.assertEqual(first.candidate_total_seconds, 6.0)
            self.assertEqual(store.profile_count(), 0)
            self.assertEqual(store.candidate_count(), 1)

            second = store.match_or_create(
                embedding=voice,
                threshold=0.80,
                observed_label="SPEAKER_00",
                duration_seconds=1.0,
                candidate_min_seconds=4.0,
                candidate_min_samples=2,
                candidate_threshold=0.75,
            )

            self.assertTrue(second.created)
            self.assertTrue(second.promoted)
            self.assertEqual(second.profile_id, "SPK_000")
            self.assertEqual(store.profile_count(), 1)
            self.assertEqual(store.candidate_count(), 0)

    def test_realtime_evidence_policy_caps_overlapped_window_duration(self) -> None:
        candidate_min_seconds, candidate_min_samples, evidence_cap = _policy_engine()._profile_evidence_policy(
            audio_seconds=9.6,
            min_seconds=2.0,
        )

        self.assertEqual(candidate_min_seconds, 6.0)
        self.assertEqual(candidate_min_samples, 8)
        self.assertEqual(evidence_cap, 1.5)

    def test_direct_evidence_policy_keeps_single_long_chunk_promotion(self) -> None:
        candidate_min_seconds, candidate_min_samples, evidence_cap = _policy_engine()._profile_evidence_policy(
            audio_seconds=60.0,
            min_seconds=2.0,
        )

        self.assertEqual(candidate_min_seconds, 6.0)
        self.assertEqual(candidate_min_samples, 1)
        self.assertEqual(evidence_cap, 60.0)

    def test_realtime_visible_profile_policy_requires_more_evidence_than_promotion(self) -> None:
        engine = _policy_engine()
        candidate_min_seconds, candidate_min_samples, _ = engine._profile_evidence_policy(
            audio_seconds=9.6,
            min_seconds=2.0,
        )

        visible_min_seconds, visible_min_samples = engine._visible_profile_policy(
            audio_seconds=9.6,
            candidate_min_seconds=candidate_min_seconds,
            candidate_min_samples=candidate_min_samples,
        )

        self.assertEqual(visible_min_seconds, 24.0)
        self.assertEqual(visible_min_samples, 16)

    def test_realtime_floors_are_configurable_and_decoupled_from_direct(self) -> None:
        # Lowering the realtime floors must change only the rolling-window policy;
        # the long direct-chunk policy stays at the fixed conservative 6.0s floor.
        engine = _policy_engine(
            rt_candidate_seconds=4.0,
            rt_candidate_samples=5,
            rt_visible_seconds=6.0,
            rt_visible_samples=10,
        )

        rt_cand_s, rt_cand_n, _ = engine._profile_evidence_policy(audio_seconds=9.6, min_seconds=2.0)
        rt_vis_s, rt_vis_n = engine._visible_profile_policy(
            audio_seconds=9.6, candidate_min_seconds=rt_cand_s, candidate_min_samples=rt_cand_n
        )
        self.assertEqual((rt_cand_s, rt_cand_n), (4.0, 5))
        self.assertEqual((rt_vis_s, rt_vis_n), (6.0, 10))

        direct_s, direct_n, _ = engine._profile_evidence_policy(audio_seconds=60.0, min_seconds=2.0)
        self.assertEqual((direct_s, direct_n), (6.0, 1))

    def test_low_evidence_profile_aliases_to_similar_mature_profile_for_display(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profiles.json"
            store = SpeakerProfileStore(path=path)

            mature = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
            noisy_same_voice = np.asarray([0.85, 0.52, 0.0], dtype=np.float32)
            mature_match = store.match_or_create(
                embedding=mature,
                threshold=0.80,
                observed_label="SPEAKER_00",
                duration_seconds=24.0,
            )
            for _ in range(15):
                store.match_or_create(
                    embedding=mature,
                    threshold=0.80,
                    observed_label="SPEAKER_00",
                    duration_seconds=1.0,
                )
            noisy_match = store.match_or_create(
                embedding=noisy_same_voice,
                threshold=0.999,
                observed_label="SPEAKER_01",
                duration_seconds=12.0,
            )

            alias = store.visible_profile_alias(
                profile_id=noisy_match.profile_id,
                embedding=noisy_same_voice,
                min_total_seconds=24.0,
                min_samples=16,
                similarity_threshold=0.80,
            )

            self.assertEqual(alias["alias"], mature_match.profile_id)
            self.assertTrue(alias["aliased"])

    def test_low_evidence_profile_without_mature_match_stays_visible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profiles.json"
            store = SpeakerProfileStore(path=path)

            mature = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
            different_voice = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
            store.match_or_create(
                embedding=mature,
                threshold=0.80,
                observed_label="SPEAKER_00",
                duration_seconds=24.0,
            )
            for _ in range(15):
                store.match_or_create(
                    embedding=mature,
                    threshold=0.80,
                    observed_label="SPEAKER_00",
                    duration_seconds=1.0,
                )
            different_match = store.match_or_create(
                embedding=different_voice,
                threshold=0.80,
                observed_label="SPEAKER_01",
                duration_seconds=12.0,
            )

            alias = store.visible_profile_alias(
                profile_id=different_match.profile_id,
                embedding=different_voice,
                min_total_seconds=24.0,
                min_samples=16,
                similarity_threshold=0.80,
            )

            self.assertEqual(alias["alias"], different_match.profile_id)
            self.assertFalse(alias["aliased"])

    def test_reconcile_merges_similar_profiles_and_returns_remap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profiles.json"
            store = SpeakerProfileStore(path=path)

            first = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
            second = np.asarray([0.95, 0.05, 0.0], dtype=np.float32)

            a = store.match_or_create(
                embedding=first,
                threshold=0.999,
                observed_label="SPEAKER_00",
                duration_seconds=2.0,
            )
            b = store.match_or_create(
                embedding=second,
                threshold=0.999,
                observed_label="SPEAKER_00",
                duration_seconds=2.0,
            )
            self.assertTrue(a.created)
            self.assertTrue(b.created)
            self.assertEqual(store.profile_count(), 2)

            stats = store.reconcile_similar_profiles(threshold=0.98)

            self.assertEqual(stats["merged_count"], 1)
            self.assertEqual(stats["profile_count"], 1)
            self.assertEqual(stats["remap"], {b.profile_id: a.profile_id})
            self.assertEqual(store.profile_count(), 1)

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual([item["id"] for item in payload["profiles"]], [a.profile_id])


if __name__ == "__main__":
    unittest.main()
