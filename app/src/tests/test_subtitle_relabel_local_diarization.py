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


class _FixedBackend:
    """Returns the same embedding regardless of the input clip, so the match result is
    fully controlled by the test (no real embedding model involved). Records the size of
    each clip it saw so tests can tell WHICH local cluster's audio was embedded."""

    def __init__(self, embedding: np.ndarray) -> None:
        self._embedding = embedding
        self.seen_clip_sizes: list[int] = []

    def extract_embedding(self, clip: np.ndarray) -> np.ndarray | None:
        self.seen_clip_sizes.append(int(clip.size))
        return self._embedding


def _engine(store: SpeakerProfileStore, *, backend_embedding: np.ndarray) -> SpeakerIdentityEngine:
    engine = SpeakerIdentityEngine.__new__(SpeakerIdentityEngine)
    engine._enabled = True
    engine._backend_name = "test"
    engine._profile_store = store
    engine._backend = _FixedBackend(backend_embedding)
    engine._last_stats = {}
    return engine


def _provider(engine: SpeakerIdentityEngine, *, enable_diarization: bool = True) -> WhisperXTranscriber:
    provider = WhisperXTranscriber.__new__(WhisperXTranscriber)
    provider._speaker_inventory_refresh_lock = threading.Lock()
    provider._enable_diarization = enable_diarization
    provider._speaker_identity_engine = engine
    return provider


def _silent_chunk(seconds: float = 5.0) -> AudioChunk:
    sample_count = int(seconds * 16000)
    return AudioChunk(pcm16=b"\x00\x00" * sample_count, sample_rate=16000, channels=1)


class ResolveLocalSpeakerTests(unittest.TestCase):
    def test_drops_sliver_and_matches_dominant_cluster(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SpeakerProfileStore(path=Path(tmp) / "profiles.json")
            profile_id = _seed(store, _vec(1.0, 0.0), seconds=10.0, label="A")
            engine = _engine(store, backend_embedding=_vec(1.0, 0.0))
            provider = _provider(engine)
            # Dominant local speaker talks 0-4s (4s, survives a 1.5s floor); a phantom
            # sliver talks 4.0-4.3s (0.3s, must be filtered out before the dominant pick).
            provider.diarize_whole_file_turns = lambda chunk, channel_mode="mono", emit_status=True: [
                {"start": 0.0, "end": 4.0, "speaker": "LOCAL_A"},
                {"start": 4.0, "end": 4.3, "speaker": "LOCAL_B"},
            ]

            before_centroid = list(store.profile_summaries()[0]["centroid"])
            result = provider.resolve_local_speaker(
                _silent_chunk(), sliver_floor_seconds=1.5, assign_threshold=0.65
            )

            self.assertEqual(result, profile_id)
            # Read-only: store untouched by the call.
            self.assertEqual(store.profile_count(), 1)
            self.assertEqual(list(store.profile_summaries()[0]["centroid"]), before_centroid)

    def test_returns_none_when_all_clusters_are_slivers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SpeakerProfileStore(path=Path(tmp) / "profiles.json")
            _seed(store, _vec(1.0, 0.0), seconds=10.0, label="A")
            engine = _engine(store, backend_embedding=_vec(1.0, 0.0))
            provider = _provider(engine)
            provider.diarize_whole_file_turns = lambda chunk, channel_mode="mono", emit_status=True: [
                {"start": 0.0, "end": 0.5, "speaker": "LOCAL_A"},
                {"start": 0.5, "end": 0.9, "speaker": "LOCAL_B"},
            ]

            result = provider.resolve_local_speaker(
                _silent_chunk(), sliver_floor_seconds=1.5, assign_threshold=0.65
            )

            self.assertIsNone(result)
            self.assertEqual(store.profile_count(), 1)

    def test_returns_none_on_no_confident_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SpeakerProfileStore(path=Path(tmp) / "profiles.json")
            _seed(store, _vec(1.0, 0.0), seconds=10.0, label="A")
            # Orthogonal embedding -> cosine 0.0, well below any sane threshold.
            engine = _engine(store, backend_embedding=_vec(0.0, 1.0))
            provider = _provider(engine)
            provider.diarize_whole_file_turns = lambda chunk, channel_mode="mono", emit_status=True: [
                {"start": 0.0, "end": 4.0, "speaker": "LOCAL_A"},
            ]

            result = provider.resolve_local_speaker(
                _silent_chunk(), sliver_floor_seconds=1.5, assign_threshold=0.65
            )

            self.assertIsNone(result)
            self.assertEqual(store.profile_count(), 1)

    def test_returns_none_when_diarization_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SpeakerProfileStore(path=Path(tmp) / "profiles.json")
            _seed(store, _vec(1.0, 0.0), seconds=10.0, label="A")
            engine = _engine(store, backend_embedding=_vec(1.0, 0.0))
            provider = _provider(engine, enable_diarization=False)
            provider.diarize_whole_file_turns = lambda chunk, channel_mode="mono", emit_status=True: [
                {"start": 0.0, "end": 4.0, "speaker": "LOCAL_A"},
            ]

            result = provider.resolve_local_speaker(_silent_chunk())

            self.assertIsNone(result)

    def test_skips_when_pipeline_lock_held(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SpeakerProfileStore(path=Path(tmp) / "profiles.json")
            _seed(store, _vec(1.0, 0.0), seconds=10.0, label="A")
            engine = _engine(store, backend_embedding=_vec(1.0, 0.0))
            provider = _provider(engine)
            provider.diarize_whole_file_turns = lambda chunk, channel_mode="mono", emit_status=True: [
                {"start": 0.0, "end": 4.0, "speaker": "LOCAL_A"},
            ]
            provider._speaker_inventory_refresh_lock.acquire()
            try:
                result = provider.resolve_local_speaker(_silent_chunk())
            finally:
                provider._speaker_inventory_refresh_lock.release()

            self.assertIsNone(result)

    def test_target_span_picks_overlapping_cluster_not_window_dominant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SpeakerProfileStore(path=Path(tmp) / "profiles.json")
            profile_id = _seed(store, _vec(1.0, 0.0), seconds=10.0, label="A")
            engine = _engine(store, backend_embedding=_vec(1.0, 0.0))
            provider = _provider(engine)
            # LOCAL_A dominates the 20s window (0-10s) but the target batch (14-16s) sits
            # inside LOCAL_B's turn (12-17s): borrow-the-partition must pick LOCAL_B and
            # embed its 5s of audio, not LOCAL_A's 10s.
            provider.diarize_whole_file_turns = lambda chunk, channel_mode="mono", emit_status=True: [
                {"start": 0.0, "end": 10.0, "speaker": "LOCAL_A"},
                {"start": 12.0, "end": 17.0, "speaker": "LOCAL_B"},
            ]

            stats: dict = {}
            result = provider.resolve_local_speaker(
                _silent_chunk(20.0),
                sliver_floor_seconds=1.5,
                assign_threshold=0.65,
                target_start_seconds=14.0,
                target_end_seconds=16.0,
                stats=stats,
            )

            self.assertEqual(result, profile_id)
            self.assertEqual(stats.get("reason"), "matched")
            # 5s of LOCAL_B at 16kHz, not 10s of LOCAL_A.
            self.assertEqual(engine._backend.seen_clip_sizes, [5 * 16000])

    def test_target_span_without_overlap_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SpeakerProfileStore(path=Path(tmp) / "profiles.json")
            _seed(store, _vec(1.0, 0.0), seconds=10.0, label="A")
            engine = _engine(store, backend_embedding=_vec(1.0, 0.0))
            provider = _provider(engine)
            provider.diarize_whole_file_turns = lambda chunk, channel_mode="mono", emit_status=True: [
                {"start": 0.0, "end": 10.0, "speaker": "LOCAL_A"},
            ]

            stats: dict = {}
            result = provider.resolve_local_speaker(
                _silent_chunk(20.0),
                sliver_floor_seconds=1.5,
                assign_threshold=0.65,
                target_start_seconds=14.0,
                target_end_seconds=16.0,
                stats=stats,
            )

            self.assertIsNone(result)
            self.assertEqual(stats.get("reason"), "no-target-overlap")

    def test_turns_cache_reuses_diarization_for_identical_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SpeakerProfileStore(path=Path(tmp) / "profiles.json")
            profile_id = _seed(store, _vec(1.0, 0.0), seconds=10.0, label="A")
            engine = _engine(store, backend_embedding=_vec(1.0, 0.0))
            provider = _provider(engine)
            diarize_calls: list[int] = []

            def counting_diarize(chunk, channel_mode="mono", emit_status=True):
                diarize_calls.append(1)
                return [{"start": 0.0, "end": 4.0, "speaker": "LOCAL_A"}]

            provider.diarize_whole_file_turns = counting_diarize
            chunk = _silent_chunk(5.0)

            first_stats: dict = {}
            second_stats: dict = {}
            first = provider.resolve_local_speaker(chunk, stats=first_stats)
            second = provider.resolve_local_speaker(chunk, stats=second_stats)

            self.assertEqual(first, profile_id)
            self.assertEqual(second, profile_id)
            self.assertEqual(len(diarize_calls), 1)
            self.assertFalse(first_stats.get("turns_cached"))
            self.assertTrue(second_stats.get("turns_cached"))

    def test_exception_in_diarize_never_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SpeakerProfileStore(path=Path(tmp) / "profiles.json")
            _seed(store, _vec(1.0, 0.0), seconds=10.0, label="A")
            engine = _engine(store, backend_embedding=_vec(1.0, 0.0))
            provider = _provider(engine)

            def boom(chunk, channel_mode="mono", emit_status=True):
                raise RuntimeError("synthetic diarization failure")

            provider.diarize_whole_file_turns = boom

            result = provider.resolve_local_speaker(_silent_chunk())

            self.assertIsNone(result)


class ResolveLocalSpeakerTurnsTests(unittest.TestCase):
    """Round 0052: turn-aware resolver returns labeled spans with full score maps."""

    def test_returns_span_entries_clipped_to_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SpeakerProfileStore(path=Path(tmp) / "profiles.json")
            profile_id = _seed(store, _vec(1.0, 0.0), seconds=10.0, label="A")
            engine = _engine(store, backend_embedding=_vec(1.0, 0.0))
            provider = _provider(engine)
            provider.diarize_whole_file_turns = lambda chunk, channel_mode="mono", emit_status=True: [
                {"start": 0.0, "end": 10.0, "speaker": "LOCAL_A"},
                {"start": 12.0, "end": 17.0, "speaker": "LOCAL_B"},
            ]

            stats: dict = {}
            entries = provider.resolve_local_speaker_turns(
                _silent_chunk(20.0),
                sliver_floor_seconds=1.5,
                assign_threshold=0.65,
                target_start_seconds=8.0,
                target_end_seconds=14.0,
                stats=stats,
            )

            self.assertIsNotNone(entries)
            self.assertEqual(stats.get("reason"), "matched-2-clusters")
            # LOCAL_A's turn clipped to 8..10, LOCAL_B's to 12..14, sorted by start.
            self.assertEqual([(e["start"], e["end"]) for e in entries], [(8.0, 10.0), (12.0, 14.0)])
            for entry in entries:
                self.assertEqual(entry["resolved"], profile_id)
                self.assertIn(profile_id, entry["scores"])
            self.assertEqual(store.profile_count(), 1)

    def test_below_threshold_clusters_are_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SpeakerProfileStore(path=Path(tmp) / "profiles.json")
            _seed(store, _vec(1.0, 0.0), seconds=10.0, label="A")
            engine = _engine(store, backend_embedding=_vec(0.0, 1.0))  # orthogonal -> cosine 0
            provider = _provider(engine)
            provider.diarize_whole_file_turns = lambda chunk, channel_mode="mono", emit_status=True: [
                {"start": 0.0, "end": 10.0, "speaker": "LOCAL_A"},
            ]

            stats: dict = {}
            entries = provider.resolve_local_speaker_turns(
                _silent_chunk(20.0),
                target_start_seconds=2.0,
                target_end_seconds=6.0,
                stats=stats,
            )

            self.assertIsNone(entries)
            self.assertEqual(stats.get("reason"), "no-confident-overlap")

    def test_no_overlap_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SpeakerProfileStore(path=Path(tmp) / "profiles.json")
            _seed(store, _vec(1.0, 0.0), seconds=10.0, label="A")
            engine = _engine(store, backend_embedding=_vec(1.0, 0.0))
            provider = _provider(engine)
            provider.diarize_whole_file_turns = lambda chunk, channel_mode="mono", emit_status=True: [
                {"start": 0.0, "end": 10.0, "speaker": "LOCAL_A"},
            ]

            entries = provider.resolve_local_speaker_turns(
                _silent_chunk(20.0),
                target_start_seconds=14.0,
                target_end_seconds=16.0,
            )

            self.assertIsNone(entries)


class ScoreProfilesReadonlyTests(unittest.TestCase):
    def test_scores_all_profiles_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SpeakerProfileStore(path=Path(tmp) / "profiles.json")
            id_a = _seed(store, _vec(1.0, 0.0), seconds=10.0, label="A")
            id_b = _seed(store, _vec(0.0, 1.0), seconds=10.0, label="B")
            engine = SpeakerIdentityEngine.__new__(SpeakerIdentityEngine)
            engine._enabled = True
            engine._profile_store = store

            before = store.profile_summaries()
            scores = engine.score_profiles_readonly(_vec(1.0, 0.0))

            self.assertAlmostEqual(scores[id_a], 1.0, places=5)
            self.assertAlmostEqual(scores[id_b], 0.0, places=5)
            self.assertEqual(store.profile_summaries(), before)

    def test_empty_when_no_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SpeakerProfileStore(path=Path(tmp) / "profiles.json")
            engine = SpeakerIdentityEngine.__new__(SpeakerIdentityEngine)
            engine._enabled = True
            engine._profile_store = store

            self.assertEqual(engine.score_profiles_readonly(_vec(1.0, 0.0)), {})


class MatchProfileReadonlyTests(unittest.TestCase):
    def test_never_mutates_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SpeakerProfileStore(path=Path(tmp) / "profiles.json")
            profile_id = _seed(store, _vec(1.0, 0.0), seconds=10.0, label="A")
            engine = SpeakerIdentityEngine.__new__(SpeakerIdentityEngine)
            engine._enabled = True
            engine._profile_store = store

            before_summaries = store.profile_summaries()
            result = engine.match_profile_readonly(_vec(1.0, 0.0), threshold=0.65)

            self.assertEqual(result, profile_id)
            self.assertEqual(store.profile_count(), 1)
            self.assertEqual(store.profile_summaries(), before_summaries)

    def test_returns_none_below_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SpeakerProfileStore(path=Path(tmp) / "profiles.json")
            _seed(store, _vec(1.0, 0.0), seconds=10.0, label="A")
            engine = SpeakerIdentityEngine.__new__(SpeakerIdentityEngine)
            engine._enabled = True
            engine._profile_store = store

            result = engine.match_profile_readonly(_vec(0.0, 1.0), threshold=0.65)

            self.assertIsNone(result)

    def test_returns_none_when_no_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SpeakerProfileStore(path=Path(tmp) / "profiles.json")
            engine = SpeakerIdentityEngine.__new__(SpeakerIdentityEngine)
            engine._enabled = True
            engine._profile_store = store

            result = engine.match_profile_readonly(_vec(1.0, 0.0), threshold=0.65)

            self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
