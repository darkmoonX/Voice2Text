"""Round 0020 Phase A: session recorder decorator."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
import wave

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice2text.audio_capture import AudioChunk
from voice2text.capture.session_recorder import (
    RecordingAudioCapture,
    apply_replay_session,
    load_session_manifest,
    redact_config_snapshot,
    replay_config_overrides,
)


class _FakeCapture:
    """Minimal AudioCaptureBase-like source yielding preset chunks then None."""

    def __init__(self, chunks, *, sample_rate=16000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self._chunks = list(chunks)
        self._i = 0
        self.started = False
        self.stopped = False
        self.backend_label = "fake"  # exercised by __getattr__ delegation

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def read_chunk(self, timeout=0.2):
        if self._i >= len(self._chunks):
            return None
        c = self._chunks[self._i]
        self._i += 1
        return c


def _wrap(inner, tmp, config_snapshot=None):
    return RecordingAudioCapture(inner, out_dir=Path(tmp) / "rec", config_snapshot=config_snapshot)


class SessionRecorderTests(unittest.TestCase):
    def test_forwards_and_records_chunks(self):
        pcms = [b"\x01\x02" * 100, b"\x03\x04" * 150, b"\x05\x06" * 80]
        chunks = [AudioChunk(p, 16000, 1) for p in pcms]
        with tempfile.TemporaryDirectory() as tmp:
            rec = _wrap(_FakeCapture(chunks), tmp)
            rec.start()
            got = []
            while True:
                c = rec.read_chunk()
                if c is None:
                    break
                got.append(c)
            rec.stop()

            # forwarded the inner chunks unchanged
            self.assertEqual([c.pcm16 for c in got], pcms)
            self.assertTrue(rec._inner.started and rec._inner.stopped)

            out = Path(tmp) / "rec"
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["chunk_count"], 3)
            self.assertEqual(manifest["total_pcm_bytes"], sum(len(p) for p in pcms))
            self.assertEqual(manifest["sample_rate"], 16000)
            self.assertEqual(manifest["channels"], 1)

            # WAV round-trips to the exact concatenated PCM
            with wave.open(str(out / "session.wav"), "rb") as w:
                self.assertEqual(w.getframerate(), 16000)
                self.assertEqual(w.getnchannels(), 1)
                self.assertEqual(w.getsampwidth(), 2)
                frames = w.readframes(w.getnframes())
            self.assertEqual(frames, b"".join(pcms))

    def test_skips_empty_and_none_chunks(self):
        chunks = [AudioChunk(b"\x01\x02" * 50, 16000, 1), AudioChunk(b"", 16000, 1)]
        with tempfile.TemporaryDirectory() as tmp:
            rec = _wrap(_FakeCapture(chunks), tmp)
            rec.start()
            while rec.read_chunk() is not None:
                pass
            rec.stop()
            manifest = json.loads((Path(tmp) / "rec" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["chunk_count"], 1)  # empty chunk not recorded

    def test_empty_session_writes_valid_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = _wrap(_FakeCapture([]), tmp)
            rec.start()
            self.assertIsNone(rec.read_chunk())
            rec.stop()
            manifest = json.loads((Path(tmp) / "rec" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["chunk_count"], 0)
        self.assertEqual(manifest["total_pcm_bytes"], 0)
        self.assertEqual(manifest["duration_seconds"], 0.0)

    def test_config_snapshot_redacts_token(self):
        snap = {"model_size": "medium", "whisperx_hf_token": "secret123", "segment_seconds": 10.0}
        with tempfile.TemporaryDirectory() as tmp:
            rec = _wrap(_FakeCapture([AudioChunk(b"\x00\x00" * 10, 16000, 1)]), tmp, config_snapshot=snap)
            rec.start()
            while rec.read_chunk() is not None:
                pass
            rec.stop()
            manifest = json.loads((Path(tmp) / "rec" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["config"]["whisperx_hf_token"], "<redacted>")
        self.assertNotIn("secret123", json.dumps(manifest))
        self.assertEqual(manifest["config"]["model_size"], "medium")

    def test_getattr_delegates_to_inner(self):
        with tempfile.TemporaryDirectory() as tmp:
            inner = _FakeCapture([])
            rec = _wrap(inner, tmp)
            self.assertEqual(rec.backend_label, "fake")  # delegated

    def test_redact_helper_keeps_empty_token_empty(self):
        self.assertEqual(redact_config_snapshot({"whisperx_hf_token": ""})["whisperx_hf_token"], "")
        self.assertEqual(redact_config_snapshot({"whisperx_hf_token": "x"})["whisperx_hf_token"], "<redacted>")


def _record_a_session(tmp, config_snapshot):
    rec = RecordingAudioCapture(
        _FakeCapture([AudioChunk(b"\x01\x02" * 200, 16000, 1)]),
        out_dir=Path(tmp) / "rec",
        config_snapshot=config_snapshot,
    )
    rec.start()
    while rec.read_chunk() is not None:
        pass
    rec.stop()
    return Path(tmp) / "rec"


class ReplaySessionTests(unittest.TestCase):
    _SNAP = {
        "model_size": "large-v2",
        "compute_type": "float16",
        "segment_seconds": 10.0,
        "hop_seconds": 2.0,
        "whisperx_enable_diarization": True,
        "whisperx_hf_token": "secret",
        "log_dir": "should-not-be-restored",
        "source_file_path": "old-path",
    }

    def test_replay_config_overrides_subset(self):
        manifest = {"config": dict(self._SNAP)}
        ov = replay_config_overrides(manifest)
        self.assertEqual(ov["model_size"], "large-v2")
        self.assertEqual(ov["segment_seconds"], 10.0)
        self.assertTrue(ov["whisperx_enable_diarization"])
        # infra / secrets are NOT in the replay subset
        self.assertNotIn("log_dir", ov)
        self.assertNotIn("whisperx_hf_token", ov)
        self.assertNotIn("source_file_path", ov)

    def test_load_manifest_resolves_wav(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec_dir = _record_a_session(tmp, self._SNAP)
            m = load_session_manifest(rec_dir)
            self.assertTrue(Path(m["_wav_path"]).exists())
            self.assertEqual(Path(m["_wav_path"]).name, "session.wav")
            # accepts the manifest path directly too
            m2 = load_session_manifest(rec_dir / "manifest.json")
            self.assertEqual(m2["_wav_path"], m["_wav_path"])

    def test_apply_replay_session_sets_source_and_config(self):
        from voice2text.config import RuntimeConfig

        with tempfile.TemporaryDirectory() as tmp:
            rec_dir = _record_a_session(tmp, self._SNAP)
            cfg = RuntimeConfig()
            manifest = apply_replay_session(cfg, rec_dir)
            self.assertEqual(cfg.source_mode, "file")
            self.assertTrue(cfg.source_file_path.endswith("session.wav"))
            self.assertEqual(cfg.model_size, "large-v2")
            self.assertEqual(cfg.segment_seconds, 10.0)
            self.assertTrue(cfg.whisperx_enable_diarization)
            self.assertEqual(manifest["chunk_count"], 1)
            # token redacted in manifest -> never restored as a real secret
            self.assertNotEqual(cfg.whisperx_hf_token, "secret")

    def test_apply_replay_session_missing_wav_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec_dir = Path(tmp) / "rec"
            rec_dir.mkdir()
            (rec_dir / "manifest.json").write_text(
                json.dumps({"wav": "session.wav", "config": {}}), encoding="utf-8"
            )
            from voice2text.config import RuntimeConfig

            with self.assertRaises(FileNotFoundError):
                apply_replay_session(RuntimeConfig(), rec_dir)


if __name__ == "__main__":
    unittest.main()
