"""Round 0072 — 'auto' model default resolves by effective device."""
from __future__ import annotations

import unittest

from voice2text.stt.model_resolution import (
    DEFAULT_CPU_MODEL,
    DEFAULT_CUDA_MODEL,
    is_auto_model,
    resolve_model_size,
)


class ModelResolutionTests(unittest.TestCase):
    def test_auto_on_cuda_is_large_v3(self) -> None:
        self.assertEqual(resolve_model_size("auto", "cuda"), "large-v3")
        self.assertEqual(DEFAULT_CUDA_MODEL, "large-v3")

    def test_auto_on_cpu_is_small(self) -> None:
        self.assertEqual(resolve_model_size("auto", "cpu"), "small")
        self.assertEqual(DEFAULT_CPU_MODEL, "small")

    def test_empty_model_treated_as_auto(self) -> None:
        self.assertEqual(resolve_model_size("", "cuda"), "large-v3")
        self.assertEqual(resolve_model_size(None, "cpu"), "small")

    def test_explicit_model_always_honored(self) -> None:
        for device in ("cuda", "cpu", ""):
            self.assertEqual(resolve_model_size("medium", device), "medium")
            self.assertEqual(resolve_model_size("large-v2", device), "large-v2")

    def test_auto_detection_is_case_insensitive(self) -> None:
        self.assertTrue(is_auto_model("AUTO"))
        self.assertTrue(is_auto_model(" auto "))
        self.assertFalse(is_auto_model("small"))

    def test_cuda_device_index_still_counts_as_cuda(self) -> None:
        self.assertEqual(resolve_model_size("auto", "cuda:0"), "large-v3")

    def test_runtime_config_default_is_auto(self) -> None:
        from voice2text.config import RuntimeConfig
        self.assertEqual(RuntimeConfig().model_size, "auto")


if __name__ == "__main__":
    unittest.main()
