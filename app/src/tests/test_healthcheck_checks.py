"""Round 0022 Phase A: structured health checks (CUDA/FFmpeg/HF/bridge/cache) + redaction."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice2text.config import RuntimeConfig
from voice2text.stt.healthcheck import (
    ProviderHealthReport,
    check_capture_bridge,
    check_cuda,
    check_ffmpeg,
    check_hf_token,
    check_model_cache,
    summarize_health_reports,
)


def _cfg(**overrides) -> RuntimeConfig:
    cfg = RuntimeConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class CudaCheckTests(unittest.TestCase):
    def test_cpu_variant_ok(self):
        c = check_cuda(_cfg(stt_variant="cpu"), cublas_probe=lambda: False)
        self.assertEqual(c.status, "ok")

    def test_gpu_with_cublas_ok(self):
        c = check_cuda(_cfg(stt_variant="gpu"), cublas_probe=lambda: True)
        self.assertEqual(c.status, "ok")

    def test_gpu_without_cublas_warn(self):
        c = check_cuda(_cfg(stt_variant="gpu"), cublas_probe=lambda: False)
        self.assertEqual(c.status, "warn")
        self.assertTrue(c.fix_hint)


class FfmpegCheckTests(unittest.TestCase):
    def test_found_via_which_ok(self):
        c = check_ffmpeg(_cfg(ffmpeg_dll_dir=""), which=lambda name: "/usr/bin/ffmpeg")
        self.assertEqual(c.status, "ok")
        self.assertIn("ffmpeg", c.detail)

    def test_missing_warn(self):
        c = check_ffmpeg(_cfg(ffmpeg_dll_dir=""), which=lambda name: None)
        self.assertEqual(c.status, "warn")
        self.assertTrue(c.fix_hint)


class HfTokenCheckTests(unittest.TestCase):
    def test_diarization_off_ok(self):
        c = check_hf_token(_cfg(whisperx_enable_diarization=False, whisperx_hf_token="secret"))
        self.assertEqual(c.status, "ok")

    def test_diarization_on_with_token_ok_and_redacted(self):
        c = check_hf_token(_cfg(whisperx_enable_diarization=True, whisperx_hf_token="secret-token-123"))
        self.assertEqual(c.status, "ok")
        # Token value must never appear anywhere in the check.
        self.assertNotIn("secret-token-123", json.dumps(c.as_dict()))

    def test_diarization_on_without_token_warn(self):
        c = check_hf_token(_cfg(whisperx_enable_diarization=True, whisperx_hf_token=""))
        self.assertEqual(c.status, "warn")
        self.assertTrue(c.fix_hint)


class BridgeCheckTests(unittest.TestCase):
    def test_missing_bridge_warn_with_fallback(self):
        c = check_capture_bridge(resolve=lambda: None, health=lambda p: (True, "unused"))
        self.assertEqual(c.status, "warn")
        self.assertIn("fallback", c.detail.lower())

    def test_present_healthy_ok(self):
        c = check_capture_bridge(resolve=lambda: Path("bridge.exe"), health=lambda p: (True, "probe-exit=0"))
        self.assertEqual(c.status, "ok")

    def test_present_unhealthy_warn(self):
        c = check_capture_bridge(resolve=lambda: Path("bridge.exe"), health=lambda p: (False, "probe-crashed"))
        self.assertEqual(c.status, "warn")
        self.assertTrue(c.fix_hint)


class ModelCacheCheckTests(unittest.TestCase):
    def test_empty_cache_warn(self):
        c = check_model_cache(_cfg(), summary=lambda: {"total_bytes": 0, "entry_count": 0})
        self.assertEqual(c.status, "warn")

    def test_populated_cache_ok(self):
        c = check_model_cache(_cfg(), summary=lambda: {"total_bytes": 1048576, "entry_count": 3})
        self.assertEqual(c.status, "ok")
        self.assertIn("3", c.detail)


class SummarizeTests(unittest.TestCase):
    def test_summary_renders_checks_and_redacts(self):
        report = ProviderHealthReport(provider="whisperx", ok=True)
        report.details["model_ref"] = "medium"
        report.checks = [
            check_hf_token(_cfg(whisperx_enable_diarization=True, whisperx_hf_token="my-secret")),
            check_ffmpeg(_cfg(ffmpeg_dll_dir=""), which=lambda n: None),
        ]
        text = summarize_health_reports([report])
        self.assertIn("check[ok] hf_token", text)
        self.assertIn("check[warn] ffmpeg", text)
        self.assertIn("-> fix:", text)              # warn rows show the fix hint
        self.assertNotIn("my-secret", text)         # token never leaks


if __name__ == "__main__":
    unittest.main()
