from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from voice2text.capture import AudioChunk
from voice2text.stt.speaker_identity import SpeakerIdentityConfig, SpeakerIdentityEngine
from voice2text.stt.whisperx_provider import WhisperXTranscriber


class _Rows:
    def itertuples(self, index=False):
        return iter([type("Row", (), {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"})()])


class _Pipeline:
    def __init__(self) -> None:
        self.kwargs: list[dict[str, object]] = []

    def __call__(self, audio, **kwargs):
        self.kwargs.append(dict(kwargs))
        return _Rows()


class _WhisperX:
    @staticmethod
    def assign_word_speakers(diarize_segments, payload):
        return payload


class _SpySpeakerProfileStore:
    instances: list["_SpySpeakerProfileStore"] = []

    def __init__(self, *args, **kwargs) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        _SpySpeakerProfileStore.instances.append(self)

    def set_soft_speaker_cap(self, *args, **kwargs) -> None:
        self.calls.append(("set_soft_speaker_cap", args, kwargs))

    def set_merge_grace(self, *args, **kwargs) -> None:
        self.calls.append(("set_merge_grace", args, kwargs))

    def set_merge_preserve_centroid(self, *args, **kwargs) -> None:
        self.calls.append(("set_merge_preserve_centroid", args, kwargs))

    def set_max_exemplars(self, *args, **kwargs) -> None:
        self.calls.append(("set_max_exemplars", args, kwargs))


def _provider(*, min_speakers: int = 0, max_speakers: int = 0) -> tuple[WhisperXTranscriber, _Pipeline]:
    provider = WhisperXTranscriber.__new__(WhisperXTranscriber)
    pipeline = _Pipeline()
    provider._diar_min_speakers = min_speakers
    provider._diar_max_speakers = max_speakers
    provider._enable_diarization = True
    provider._diarization_disabled_reason = None
    provider._diarization_device = "cpu"
    provider._diarization_pipeline = pipeline
    provider._whisperx = _WhisperX()
    provider._emit = lambda message: None
    provider._ensure_diarization_pipeline_loaded = lambda: None
    provider._ensure_whole_file_diarization_pipeline = lambda: pipeline
    return provider, pipeline


def _chunk(seconds: float = 1.0) -> AudioChunk:
    sample_count = int(seconds * 16000)
    pcm = (np.ones(sample_count, dtype=np.int16) * 1000).tobytes()
    return AudioChunk(pcm16=pcm, sample_rate=16000, channels=1)


class DiarizationSpeakerHintTests(unittest.TestCase):
    def test_kwargs_builder_omits_unset_values(self) -> None:
        provider, _pipeline = _provider()
        self.assertEqual(provider._diar_speaker_count_kwargs(), {})

    def test_kwargs_builder_includes_positive_values_only(self) -> None:
        provider, _pipeline = _provider(min_speakers=2, max_speakers=4)
        self.assertEqual(provider._diar_speaker_count_kwargs(), {"min_speakers": 2, "max_speakers": 4})

        provider, _pipeline = _provider(min_speakers=0, max_speakers=3)
        self.assertEqual(provider._diar_speaker_count_kwargs(), {"max_speakers": 3})

    def test_attach_speaker_labels_passes_no_kwargs_when_unset(self) -> None:
        provider, pipeline = _provider()

        provider._attach_speaker_labels(np.ones(16000, dtype=np.float32), [{"start": 0.0, "end": 1.0}])

        self.assertEqual(pipeline.kwargs, [{}])

    def test_attach_speaker_labels_passes_hints_when_set(self) -> None:
        provider, pipeline = _provider(min_speakers=2, max_speakers=2)

        provider._attach_speaker_labels(np.ones(16000, dtype=np.float32), [{"start": 0.0, "end": 1.0}])

        self.assertEqual(pipeline.kwargs, [{"min_speakers": 2, "max_speakers": 2}])

    def test_whole_file_diarization_passes_no_kwargs_when_unset(self) -> None:
        provider, pipeline = _provider()

        turns = provider.diarize_whole_file_turns(_chunk())

        self.assertEqual(pipeline.kwargs, [{}])
        self.assertEqual(turns[0]["speaker"], "SPEAKER_00")

    def test_whole_file_diarization_passes_hints_when_set(self) -> None:
        provider, pipeline = _provider(min_speakers=3, max_speakers=3)

        provider.diarize_whole_file_turns(_chunk())

        self.assertEqual(pipeline.kwargs, [{"min_speakers": 3, "max_speakers": 3}])


class SpeakerIdentityHintTests(unittest.TestCase):
    def test_engine_applies_max_speakers_hint_to_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = SpeakerIdentityEngine(
                SpeakerIdentityConfig(
                    enabled=True,
                    backend="pyannote",
                    store_path=str(Path(tmp) / "profiles.json"),
                    match_threshold=0.99,
                    min_seconds=2.0,
                    reconcile_threshold=0.52,
                    model_root=str(Path(tmp)),
                    device="cpu",
                    hf_token="",
                    pyannote_model="pyannote/embedding",
                    speechbrain_model="speechbrain/spkrec-ecapa-voxceleb",
                    nemo_model="nvidia/speakerverification_en_titanet_large",
                    max_speakers_hint=1,
                )
            )

            store = engine._profile_store
            self.assertIsNotNone(store)
            first = store.match_or_create(
                embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                threshold=0.99,
                observed_label="A",
                duration_seconds=2.0,
            )
            second = store.match_or_create(
                embedding=np.asarray([0.0, 1.0], dtype=np.float32),
                threshold=0.99,
                observed_label="B",
                duration_seconds=2.0,
            )

            self.assertTrue(first.created)
            self.assertFalse(second.created)
            self.assertEqual(store.profile_count(), 1)

    def test_engine_applies_merge_grace_to_store(self) -> None:
        _SpySpeakerProfileStore.instances = []
        with tempfile.TemporaryDirectory() as tmp:
            with patch("voice2text.stt.speaker_identity.SpeakerProfileStore", _SpySpeakerProfileStore):
                SpeakerIdentityEngine(
                    SpeakerIdentityConfig(
                        enabled=True,
                        backend="pyannote",
                        store_path=str(Path(tmp) / "profiles.json"),
                        match_threshold=0.99,
                        min_seconds=2.0,
                        reconcile_threshold=0.52,
                        model_root=str(Path(tmp)),
                        device="cpu",
                        hf_token="",
                        pyannote_model="pyannote/embedding",
                        speechbrain_model="speechbrain/spkrec-ecapa-voxceleb",
                        nemo_model="nvidia/speakerverification_en_titanet_large",
                        merge_grace_windows=30,
                        merge_grace_relief=0.15,
                    )
                )

        self.assertEqual(len(_SpySpeakerProfileStore.instances), 1)
        self.assertIn(("set_soft_speaker_cap", (0,), {}), _SpySpeakerProfileStore.instances[0].calls)
        self.assertIn(("set_merge_grace", (30, 0.15), {}), _SpySpeakerProfileStore.instances[0].calls)

    def test_engine_applies_merge_preserve_centroid_to_store(self) -> None:
        _SpySpeakerProfileStore.instances = []
        with tempfile.TemporaryDirectory() as tmp:
            with patch("voice2text.stt.speaker_identity.SpeakerProfileStore", _SpySpeakerProfileStore):
                SpeakerIdentityEngine(
                    SpeakerIdentityConfig(
                        enabled=True,
                        backend="pyannote",
                        store_path=str(Path(tmp) / "profiles.json"),
                        match_threshold=0.99,
                        min_seconds=2.0,
                        reconcile_threshold=0.52,
                        model_root=str(Path(tmp)),
                        device="cpu",
                        hf_token="",
                        pyannote_model="pyannote/embedding",
                        speechbrain_model="speechbrain/spkrec-ecapa-voxceleb",
                        nemo_model="nvidia/speakerverification_en_titanet_large",
                        merge_preserve_centroid=True,
                    )
                )

        self.assertEqual(len(_SpySpeakerProfileStore.instances), 1)
        self.assertIn(
            ("set_merge_preserve_centroid", (True,), {}),
            _SpySpeakerProfileStore.instances[0].calls,
        )

    def test_engine_applies_max_exemplars_to_store(self) -> None:
        _SpySpeakerProfileStore.instances = []
        with tempfile.TemporaryDirectory() as tmp:
            with patch("voice2text.stt.speaker_identity.SpeakerProfileStore", _SpySpeakerProfileStore):
                SpeakerIdentityEngine(
                    SpeakerIdentityConfig(
                        enabled=True,
                        backend="pyannote",
                        store_path=str(Path(tmp) / "profiles.json"),
                        match_threshold=0.99,
                        min_seconds=2.0,
                        reconcile_threshold=0.52,
                        model_root=str(Path(tmp)),
                        device="cpu",
                        hf_token="",
                        pyannote_model="pyannote/embedding",
                        speechbrain_model="speechbrain/spkrec-ecapa-voxceleb",
                        nemo_model="nvidia/speakerverification_en_titanet_large",
                        max_exemplars=4,
                        exemplar_diversity_threshold=0.85,
                    )
                )

        self.assertEqual(len(_SpySpeakerProfileStore.instances), 1)
        self.assertIn(
            ("set_max_exemplars", (4, 0.85), {}),
            _SpySpeakerProfileStore.instances[0].calls,
        )

    def test_engine_applies_default_max_exemplars_to_store(self) -> None:
        _SpySpeakerProfileStore.instances = []
        with tempfile.TemporaryDirectory() as tmp:
            with patch("voice2text.stt.speaker_identity.SpeakerProfileStore", _SpySpeakerProfileStore):
                SpeakerIdentityEngine(
                    SpeakerIdentityConfig(
                        enabled=True,
                        backend="pyannote",
                        store_path=str(Path(tmp) / "profiles.json"),
                        match_threshold=0.99,
                        min_seconds=2.0,
                        reconcile_threshold=0.52,
                        model_root=str(Path(tmp)),
                        device="cpu",
                        hf_token="",
                        pyannote_model="pyannote/embedding",
                        speechbrain_model="speechbrain/spkrec-ecapa-voxceleb",
                        nemo_model="nvidia/speakerverification_en_titanet_large",
                    )
                )

        self.assertEqual(len(_SpySpeakerProfileStore.instances), 1)
        self.assertIn(
            ("set_max_exemplars", (1, 0.90), {}),
            _SpySpeakerProfileStore.instances[0].calls,
        )


if __name__ == "__main__":
    unittest.main()
