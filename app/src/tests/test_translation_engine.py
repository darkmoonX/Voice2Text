"""Round 0026: TranslationEngine — inline passthrough, off-thread timeout/retry, credential redaction."""
from __future__ import annotations

from pathlib import Path
import sys
import threading
import time
import unittest

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice2text.config import RuntimeConfig
from voice2text.translation import TranslationEngine, build_translation_engine
from voice2text.translation.base import TranslationState
from voice2text.capture.session_recorder import redact_config_snapshot


class FakeBackend:
    """Minimal TranslationBackend for tests (configurable delay / failures)."""

    def __init__(self, *, enabled=True, result="OUT", delay=0.0, fail_times=0, name="fake"):
        self._enabled = enabled
        self._result = result
        self._delay = float(delay)
        self._fail_times = int(fail_times)
        self._name = name
        self.calls = 0
        self._lock = threading.Lock()
        self.state = TranslationState(enabled, "fake backend")

    @property
    def name(self):
        return self._name

    @property
    def enabled(self):
        return self._enabled

    def translate(self, text, source_code=None):
        with self._lock:
            self.calls += 1
            n = self.calls
        if self._delay:
            time.sleep(self._delay)
        if n <= self._fail_times:
            return None
        return f"{self._result}:{text}"


class InlinePassthroughTests(unittest.TestCase):
    def test_disabled_policy_is_direct_passthrough(self):
        backend = FakeBackend(result="T")
        engine = TranslationEngine(backend, queue_max=0)
        self.assertFalse(engine.policy_active)
        self.assertEqual(engine.translate("hi", "en"), "T:hi")
        self.assertEqual(backend.calls, 1)  # called exactly once, inline (no retry wrapping)

    def test_disabled_backend_returns_none(self):
        engine = TranslationEngine(FakeBackend(enabled=False), queue_max=0)
        self.assertFalse(engine.enabled)
        self.assertIsNone(engine.translate("hi", "en"))

    def test_none_backend_is_safe(self):
        engine = TranslationEngine(None, queue_max=0)
        self.assertFalse(engine.enabled)
        self.assertIsNone(engine.translate("hi"))


class AsyncPolicyTests(unittest.TestCase):
    def test_async_fast_backend_returns_result(self):
        backend = FakeBackend(result="OK")
        engine = TranslationEngine(backend, queue_max=4, timeout_seconds=2.0)
        self.addCleanup(engine.shutdown)
        self.assertTrue(engine.policy_active)
        self.assertEqual(engine.translate("hi", "en"), "OK:hi")

    def test_hanging_backend_cannot_stall_loop(self):
        # The key acceptance: a slow backend returns None within ~timeout, never blocking the caller.
        backend = FakeBackend(result="LATE", delay=5.0)
        engine = TranslationEngine(backend, queue_max=2, timeout_seconds=0.15)
        self.addCleanup(engine.shutdown)
        started = time.monotonic()
        out = engine.translate("hi", "en")
        elapsed = time.monotonic() - started
        self.assertIsNone(out)              # gave up -> source-only subtitle
        self.assertLess(elapsed, 1.0)       # returned promptly (well under the 5s backend delay)

    def test_retry_recovers_after_transient_failures(self):
        backend = FakeBackend(result="OK", fail_times=2)
        engine = TranslationEngine(backend, queue_max=2, timeout_seconds=2.0, max_retries=2)
        self.addCleanup(engine.shutdown)
        self.assertEqual(engine.translate("hi", "en"), "OK:hi")
        self.assertEqual(backend.calls, 3)  # 2 failures + 1 success

    def test_no_retry_gives_up_on_failure(self):
        backend = FakeBackend(result="OK", fail_times=1)
        engine = TranslationEngine(backend, queue_max=2, timeout_seconds=2.0, max_retries=0)
        self.addCleanup(engine.shutdown)
        self.assertIsNone(engine.translate("hi", "en"))

    def test_concurrent_slow_calls_all_return_bounded(self):
        # Saturate a size-1 queue with a slow backend from several threads; none may hang.
        backend = FakeBackend(result="X", delay=3.0)
        engine = TranslationEngine(backend, queue_max=1, timeout_seconds=0.2)
        self.addCleanup(engine.shutdown)
        results: list = []
        lock = threading.Lock()

        def worker():
            started = time.monotonic()
            r = engine.translate("hi", "en")
            with lock:
                results.append((r, time.monotonic() - started))

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=3.0)
        self.assertEqual(len(results), 5)
        for _r, elapsed in results:
            self.assertLess(elapsed, 1.5)  # bounded by timeout, not the 3s backend delay


class BuildEngineDefaultsTests(unittest.TestCase):
    def test_default_config_keeps_engine_inline(self):
        cfg = RuntimeConfig()
        cfg.translation_enabled = False
        engine = build_translation_engine(cfg)
        # Default translation_queue_max == 0 -> inline passthrough (byte-identical pre-0026 behavior).
        self.assertFalse(engine.policy_active)
        self.assertEqual(engine.name, "argos")

    def test_queue_max_enables_policy(self):
        cfg = RuntimeConfig()
        cfg.translation_enabled = False
        cfg.translation_queue_max = 4
        engine = build_translation_engine(cfg)
        self.addCleanup(engine.shutdown)
        self.assertTrue(engine.policy_active)


class RedactionTests(unittest.TestCase):
    def test_future_credential_fields_are_redacted(self):
        snap = {
            "translation_backend": "cloud",
            "translation_api_key": "sk-secret-123",
            "cloud_access_token": "tok-abc",
            "some_password": "p@ss",
            "whisperx_hf_token": "hf_xxx",
            "translation_to": "zh",
        }
        out = redact_config_snapshot(snap)
        self.assertEqual(out["translation_api_key"], "<redacted>")
        self.assertEqual(out["cloud_access_token"], "<redacted>")
        self.assertEqual(out["some_password"], "<redacted>")
        self.assertEqual(out["whisperx_hf_token"], "<redacted>")
        # Non-credential fields are untouched.
        self.assertEqual(out["translation_backend"], "cloud")
        self.assertEqual(out["translation_to"], "zh")

    def test_empty_credential_stays_empty(self):
        out = redact_config_snapshot({"translation_api_key": "", "whisperx_hf_token": ""})
        self.assertEqual(out["translation_api_key"], "")
        self.assertEqual(out["whisperx_hf_token"], "")


if __name__ == "__main__":
    unittest.main()
