"""Unit tests for STT provider health-check diagnostics and warnings."""
from __future__ import annotations

from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice2text.config import RuntimeConfig
from voice2text.stt.healthcheck import run_provider_health_check


def _module(name: str) -> types.ModuleType:
    return types.ModuleType(name)


class STTHealthCheckTests(unittest.TestCase):
    def test_active_scope_returns_single_provider(self) -> None:
        cfg = RuntimeConfig(stt_provider="whisper")
        with patch.dict(sys.modules, {"whisperx": _module("whisperx")}):
            reports = run_provider_health_check(cfg, scope="active")

        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0].provider, "whisperx")

    def test_all_scope_uses_current_providers_only(self) -> None:
        with patch.dict(
            sys.modules,
            {
                "whisperx": _module("whisperx"),
            },
        ):
            reports = run_provider_health_check(RuntimeConfig(), scope="all")

        self.assertEqual([report.provider for report in reports], ["whisperx", "whispercpp"])

    def test_whisperx_diarization_without_token_warns(self) -> None:
        cfg = RuntimeConfig(
            stt_provider="whisperx",
            whisperx_enable_diarization=True,
            whisperx_hf_token="",
        )

        with patch.dict(sys.modules, {"whisperx": _module("whisperx")}):
            reports = run_provider_health_check(cfg, scope="active")

        self.assertEqual(reports[0].provider, "whisperx")
        self.assertTrue(any("HF token is empty" in warning for warning in reports[0].warnings))


if __name__ == "__main__":
    unittest.main()
