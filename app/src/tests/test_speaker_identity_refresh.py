from __future__ import annotations

from pathlib import Path
import threading
import tempfile
import unittest

import numpy as np

from voice2text.capture import AudioChunk
from voice2text.stt.speaker_identity import SpeakerIdentityEngine
from voice2text.stt.speaker_profiles import SpeakerProfileStore
from voice2text.stt.whisperx_provider import WhisperXTranscriber


def _vec(*values: float) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _seed(store: SpeakerProfileStore, vector: np.ndarray, *, seconds: float, label: str) -> str:
    result = store.match_or_create(
        embedding=vector,
        threshold=0.999,
        observed_label=label,
        duration_seconds=seconds,
    )
    return result.profile_id


def _engine(store: SpeakerProfileStore, *, merge: bool = True, match_mode: str = "argmax") -> SpeakerIdentityEngine:
    engine = SpeakerIdentityEngine.__new__(SpeakerIdentityEngine)
    engine._enabled = True
    engine._backend_name = "test"
    engine._profile_store = store
    engine._last_stats = {}
    engine._rt_refresh_alpha = 0.5
    engine._rt_refresh_assign_threshold = 0.55
    engine._rt_refresh_min_cluster_seconds = 4.0
    engine._rt_refresh_merge = bool(merge)
    engine._rt_refresh_match_mode = match_mode
    return engine


class SpeakerIdentityRefreshTests(unittest.TestCase):
    def test_assigns_profiles_to_argmax_cluster_at_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SpeakerProfileStore(path=Path(tmp) / "profiles.json")
            pid = _seed(store, _vec(1.0, 0.0), seconds=5.0, label="A")

            stats = _engine(store).refresh_inventory(
                {
                    "c0": {"centroid": _vec(0.56, 0.44), "duration_seconds": 4.0},
                    "c1": {"centroid": _vec(0.0, 1.0), "duration_seconds": 4.0},
                }
            )

            self.assertEqual(stats["status"], "done")
            self.assertEqual(stats["refreshed_count"], 1)
            self.assertEqual(stats["merged_count"], 0)
            self.assertEqual(stats["assignments"][0]["profile_id"], pid)
            self.assertEqual(stats["assignments"][0]["cluster_id"], "c0")

    def test_min_cluster_seconds_filter_drops_slivers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SpeakerProfileStore(path=Path(tmp) / "profiles.json")
            _seed(store, _vec(1.0, 0.0), seconds=5.0, label="A")

            stats = _engine(store).refresh_inventory(
                {"sliver": {"centroid": _vec(1.0, 0.0), "duration_seconds": 3.9}}
            )

            self.assertEqual(stats["status"], "skip_no_clusters")
            self.assertEqual(stats["clusters"], 0)
            self.assertEqual(store.profile_count(), 1)

    def test_same_cluster_profiles_merge_and_keep_most_mature_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SpeakerProfileStore(path=Path(tmp) / "profiles.json")
            low_evidence = _seed(store, _vec(1.0, 0.0), seconds=1.0, label="A")
            mature = _seed(store, _vec(0.98, 0.2), seconds=10.0, label="B")

            stats = _engine(store).refresh_inventory(
                {"teacher": {"centroid": _vec(1.0, 0.0), "duration_seconds": 12.0}}
            )

            self.assertEqual(stats["merged_count"], 1)
            self.assertEqual(stats["remap"], {low_evidence: mature})
            self.assertEqual(store.profile_count(), 1)
            self.assertEqual(store.profile_summaries()[0]["id"], mature)

    def test_unmatched_cluster_creates_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SpeakerProfileStore(path=Path(tmp) / "profiles.json")
            _seed(store, _vec(1.0, 0.0), seconds=5.0, label="A")

            stats = _engine(store).refresh_inventory(
                {"new": {"centroid": _vec(0.0, 1.0), "duration_seconds": 10.0}}
            )

            self.assertEqual(stats["status"], "done_no_assignment")
            self.assertEqual(stats["refreshed_count"], 0)
            self.assertEqual(stats["merged_count"], 0)
            self.assertEqual(store.profile_count(), 1)

    def test_merge_can_be_disabled_while_ema_still_refreshes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SpeakerProfileStore(path=Path(tmp) / "profiles.json")
            _seed(store, _vec(1.0, 0.0), seconds=1.0, label="A")
            _seed(store, _vec(0.98, 0.2), seconds=10.0, label="B")

            stats = _engine(store, merge=False).refresh_inventory(
                {"teacher": {"centroid": _vec(1.0, 0.0), "duration_seconds": 12.0}}
            )

            self.assertEqual(stats["merged_count"], 0)
            self.assertEqual(stats["refreshed_count"], 1)
            self.assertEqual(store.profile_count(), 2)

    def test_provider_refresh_single_flight_skips_second_call(self) -> None:
        provider = WhisperXTranscriber.__new__(WhisperXTranscriber)
        provider._speaker_inventory_refresh_lock = threading.Lock()
        provider._speaker_inventory_refresh_lock.acquire()
        try:
            stats = provider.refresh_speaker_inventory(
                AudioChunk(pcm16=b"\x00" * 3200, sample_rate=16000, channels=1)
            )
        finally:
            provider._speaker_inventory_refresh_lock.release()

        self.assertEqual(stats["status"], "skip_in_flight")


if __name__ == "__main__":
    unittest.main()
