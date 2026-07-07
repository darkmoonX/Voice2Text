"""Round 0023 Phase A: read-only (allow_update=False) profile match must not mutate state."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice2text.stt.speaker_profiles import SpeakerProfileStore


def _vec(*values, dim=16):
    arr = np.zeros(dim, dtype=np.float32)
    for i, v in enumerate(values):
        arr[i] = float(v)
    return arr


class ReadOnlyMatchTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = SpeakerProfileStore(path=Path(self._tmp.name) / "profiles.json")
        # Seed one mature profile directly (candidate_min_seconds=0 -> immediate create).
        res = self.store.match_or_create(
            embedding=_vec(1.0, 0.0, 0.0),
            threshold=0.5,
            observed_label="spk_000",
            duration_seconds=4.0,
            candidate_min_seconds=0.0,
        )
        self.assertTrue(res.profile_id)
        self.pid = res.profile_id

    def tearDown(self):
        self._tmp.cleanup()

    def _snapshot(self):
        return {
            "profiles": copy.deepcopy(self.store._profiles),
            "candidates": copy.deepcopy(self.store._candidates),
            "profile_count": self.store.profile_count(),
            "candidate_count": self.store.candidate_count(),
        }

    def test_readonly_match_returns_id_without_mutation(self):
        before = self._snapshot()
        # A clearly-similar embedding read-only matches the seeded profile.
        res = self.store.match_or_create(
            embedding=_vec(0.98, 0.05, 0.0),
            threshold=0.5,
            observed_label="spk_000",
            duration_seconds=4.0,
            candidate_min_seconds=0.0,
            allow_update=False,
        )
        self.assertEqual(res.profile_id, self.pid)
        self.assertFalse(res.created)
        after = self._snapshot()
        # Centroid / weight / samples / total_seconds untouched.
        self.assertEqual(before["profiles"], after["profiles"])
        self.assertEqual(before["candidates"], after["candidates"])
        self.assertEqual(before["profile_count"], after["profile_count"])
        self.assertEqual(before["candidate_count"], after["candidate_count"])

    def test_readonly_no_match_does_not_create_or_stage(self):
        before = self._snapshot()
        # An orthogonal embedding does not match; read-only must NOT create a profile or stage a candidate.
        res = self.store.match_or_create(
            embedding=_vec(0.0, 0.0, 1.0),
            threshold=0.5,
            observed_label="spk_999",
            duration_seconds=4.0,
            candidate_min_seconds=6.0,
            candidate_min_samples=4,
            allow_update=False,
        )
        self.assertEqual(res.profile_id, "")
        self.assertFalse(res.created)
        after = self._snapshot()
        self.assertEqual(before["profiles"], after["profiles"])
        self.assertEqual(before["candidates"], after["candidates"])
        self.assertEqual(before["profile_count"], after["profile_count"])
        self.assertEqual(before["candidate_count"], after["candidate_count"])

    def test_update_path_still_mutates(self):
        # Sanity: the default (allow_update=True) path DOES learn, so the read-only test is meaningful.
        before_weight = float(self.store._profiles[0]["weight"])
        self.store.match_or_create(
            embedding=_vec(0.97, 0.06, 0.0),
            threshold=0.5,
            observed_label="spk_000",
            duration_seconds=4.0,
            candidate_min_seconds=0.0,
            allow_update=True,
        )
        after_weight = float(self.store._profiles[0]["weight"])
        self.assertGreater(after_weight, before_weight)


if __name__ == "__main__":
    unittest.main()
