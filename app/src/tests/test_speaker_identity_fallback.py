from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from voice2text.stt.speaker_identity import _BaseEmbeddingBackend, _FallbackEmbeddingBackend


class _FakeBackend(_BaseEmbeddingBackend):
    def __init__(self, *, name: str, ready: bool, reason: str = "") -> None:
        super().__init__(device="cpu", model_root=Path("."), on_status=None)
        self.backend_name = name
        self._ready = ready
        self._disabled_reason = reason
        self.ensure_count = 0

    def ensure_loaded(self) -> bool:
        self.ensure_count += 1
        return bool(self._ready)

    def extract_embedding(self, clip: np.ndarray) -> np.ndarray | None:
        if not self.ensure_loaded():
            return None
        return np.ones((4,), dtype=np.float32)


class SpeakerIdentityFallbackTests(unittest.TestCase):
    def test_pyannote_gated_error_falls_back_to_speechbrain(self) -> None:
        messages: list[str] = []
        primary = _FakeBackend(
            name="pyannote",
            ready=False,
            reason=(
                "403 Client Error. Cannot access gated repo for url "
                "https://huggingface.co/pyannote/embedding/resolve/main/pytorch_model.bin."
            ),
        )
        fallback = _FakeBackend(name="speechbrain_ecapa", ready=True)
        backend = _FallbackEmbeddingBackend(primary=primary, fallback=fallback, on_status=messages.append)

        self.assertTrue(backend.ensure_loaded())
        self.assertEqual(backend.active_backend_name, "speechbrain_ecapa")
        self.assertTrue(any("pyannote/embedding gated access denied" in msg for msg in messages))
        self.assertNotIn("huggingface.co", " ".join(messages))


if __name__ == "__main__":
    unittest.main()
