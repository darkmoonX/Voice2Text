"""Round 0069: auto-crash-bundle-on-uncaught-exception guard logic in bootstrap_runtime.py.

Only exercises `_write_auto_crash_bundle` directly (not the full `_install_python_exception_hooks`,
which mutates `sys.excepthook` process-globally and would be invasive to install/restore around a
unit test). The guard/gating logic is what round 0069 actually added; the hook wiring itself is
unchanged plumbing.
"""
from __future__ import annotations

import logging
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import voice2text.bootstrap_runtime as bootstrap_runtime
from voice2text.config import RuntimeConfig

_LOGGER = logging.getLogger("test-crash-bundle")


class AutoCrashBundleGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self._original = bootstrap_runtime._CRASH_BUNDLE_WRITTEN
        bootstrap_runtime._CRASH_BUNDLE_WRITTEN = False

    def tearDown(self) -> None:
        bootstrap_runtime._CRASH_BUNDLE_WRITTEN = self._original

    def test_writes_bundle_once_when_enabled(self) -> None:
        cfg = RuntimeConfig(crash_bundle_on_uncaught_exception=True)
        with patch("voice2text.crash_bundle.create_crash_bundle", return_value=Path("C:/fake/crash.zip")) as fake:
            bootstrap_runtime._write_auto_crash_bundle(cfg, _LOGGER, "test reason 1")
            bootstrap_runtime._write_auto_crash_bundle(cfg, _LOGGER, "test reason 2")
        # Second call must be a no-op: a cascading crash loop must not spam bundles.
        fake.assert_called_once_with(cfg, reason="test reason 1")

    def test_disabled_by_config_never_calls_create_crash_bundle(self) -> None:
        cfg = RuntimeConfig(crash_bundle_on_uncaught_exception=False)
        with patch("voice2text.crash_bundle.create_crash_bundle") as fake:
            bootstrap_runtime._write_auto_crash_bundle(cfg, _LOGGER, "test reason")
        fake.assert_not_called()

    def test_failure_inside_create_crash_bundle_is_swallowed(self) -> None:
        cfg = RuntimeConfig(crash_bundle_on_uncaught_exception=True)
        with patch("voice2text.crash_bundle.create_crash_bundle", side_effect=RuntimeError("boom")):
            # Must not raise -- this runs inside an exception handler; a second failure there
            # would be catastrophic.
            bootstrap_runtime._write_auto_crash_bundle(cfg, _LOGGER, "test reason")
        # Guard still flips even on failure, so a broken bundle path can't retry-loop either.
        self.assertTrue(bootstrap_runtime._CRASH_BUNDLE_WRITTEN)


if __name__ == "__main__":
    unittest.main()
