from __future__ import annotations

import threading
import unittest

from voice2text.capture import AudioChunk
from voice2text.stt.whisperx_provider import WhisperXTranscriber


class _SpyProfileStore:
    def __init__(self, profile_count: int = 0) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        self._profile_count = int(profile_count)

    def match_or_create(self, *args, **kwargs):
        self.calls.append(("match_or_create", args, kwargs))

    def merge_profiles(self, *args, **kwargs):
        self.calls.append(("merge_profiles", args, kwargs))

    def blend_centroid(self, *args, **kwargs):
        self.calls.append(("blend_centroid", args, kwargs))

    def set_soft_speaker_cap(self, *args, **kwargs):
        self.calls.append(("set_soft_speaker_cap", args, kwargs))

    def profile_count(self) -> int:
        self.calls.append(("profile_count", (), {}))
        return self._profile_count


class _SpyEngine:
    def __init__(self, store: _SpyProfileStore) -> None:
        self._profile_store = store


def _provider(*, enable_diarization: bool = True, store: _SpyProfileStore | None = None) -> WhisperXTranscriber:
    provider = WhisperXTranscriber.__new__(WhisperXTranscriber)
    provider._speaker_inventory_refresh_lock = threading.Lock()
    provider._enable_diarization = enable_diarization
    provider._speaker_identity_engine = _SpyEngine(store or _SpyProfileStore())
    return provider


def _silent_chunk(seconds: float = 5.0) -> AudioChunk:
    return AudioChunk(pcm16=b"\x00\x00" * int(seconds * 16000), sample_rate=16000, channels=1)


class EstimateSpeakerCountTests(unittest.TestCase):
    def test_counts_surviving_speakers_after_sliver_filter(self) -> None:
        store = _SpyProfileStore()
        provider = _provider(store=store)
        provider.diarize_whole_file_turns = lambda chunk, channel_mode="mono", emit_status=True: [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
            {"start": 1.2, "end": 2.0, "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_01"},
            {"start": 4.0, "end": 4.4, "speaker": "PHANTOM"},
        ]

        count = provider.estimate_speaker_count(_silent_chunk(), sliver_floor_seconds=1.5)

        self.assertEqual(count, 2)
        self.assertEqual(store.calls, [])

    def test_returns_none_when_no_turns(self) -> None:
        provider = _provider()
        provider.diarize_whole_file_turns = lambda chunk, channel_mode="mono", emit_status=True: []

        self.assertIsNone(provider.estimate_speaker_count(_silent_chunk()))

    def test_returns_none_when_diarization_disabled(self) -> None:
        provider = _provider(enable_diarization=False)
        calls: list[int] = []

        def diarize(chunk, channel_mode="mono", emit_status=True):
            calls.append(1)
            return [{"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"}]

        provider.diarize_whole_file_turns = diarize

        self.assertIsNone(provider.estimate_speaker_count(_silent_chunk()))
        self.assertEqual(calls, [])

    def test_returns_none_when_lock_held(self) -> None:
        provider = _provider()
        calls: list[int] = []
        provider.diarize_whole_file_turns = lambda chunk, channel_mode="mono", emit_status=True: calls.append(1)
        provider._speaker_inventory_refresh_lock.acquire()
        try:
            result = provider.estimate_speaker_count(_silent_chunk())
        finally:
            provider._speaker_inventory_refresh_lock.release()

        self.assertIsNone(result)
        self.assertEqual(calls, [])

    def test_returns_none_on_exception(self) -> None:
        provider = _provider()

        def boom(chunk, channel_mode="mono", emit_status=True):
            raise RuntimeError("synthetic diarization failure")

        provider.diarize_whole_file_turns = boom

        self.assertIsNone(provider.estimate_speaker_count(_silent_chunk()))

    def test_returns_none_when_all_clusters_are_slivers(self) -> None:
        provider = _provider()
        provider.diarize_whole_file_turns = lambda chunk, channel_mode="mono", emit_status=True: [
            {"start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"},
            {"start": 0.5, "end": 1.0, "speaker": "SPEAKER_01"},
        ]

        self.assertIsNone(provider.estimate_speaker_count(_silent_chunk(), sliver_floor_seconds=1.5))

    def test_set_speaker_count_cap_forwards_to_store_only(self) -> None:
        store = _SpyProfileStore()
        provider = _provider(store=store)

        provider.set_speaker_count_cap(3)

        self.assertEqual(store.calls, [("set_soft_speaker_cap", (3,), {})])

    def test_get_speaker_profile_count_forwards_to_store(self) -> None:
        store = _SpyProfileStore(profile_count=4)
        provider = _provider(store=store)

        count = provider.get_speaker_profile_count()

        self.assertEqual(count, 4)
        self.assertEqual(store.calls, [("profile_count", (), {})])

    def test_get_speaker_profile_count_returns_zero_when_engine_or_store_absent(self) -> None:
        provider = WhisperXTranscriber.__new__(WhisperXTranscriber)
        provider._speaker_identity_engine = None
        self.assertEqual(provider.get_speaker_profile_count(), 0)

        provider._speaker_identity_engine = _SpyEngine(None)  # type: ignore[arg-type]
        self.assertEqual(provider.get_speaker_profile_count(), 0)


if __name__ == "__main__":
    unittest.main()
