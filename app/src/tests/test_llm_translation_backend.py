"""Round 0074 — LlmTranslator backend unit tests (no real llama-server needed)."""
from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from voice2text.config import RuntimeConfig
from voice2text.translation import llm_backend
from voice2text.translation.llm_backend import (
    CONTEXT_TIERS,
    LlmTranslator,
    build_system_prompt,
    choose_context_tier,
    estimate_asr_reserve_mib,
    llm_footprint_mib,
)
from voice2text.translation.registry import UnavailableBackend, build_backend


class _FakeLlamaHandler(BaseHTTPRequestHandler):
    canned = "這是譯文"

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            body = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length).decode("utf-8"))
        type(self).last_request = request
        body = json.dumps({
            "choices": [{"message": {"content": type(self).canned}}]
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence
        pass


def _wait_ready(backend: LlmTranslator, timeout: float = 8.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if backend.enabled:
            return True
        time.sleep(0.05)
    return False


class SystemPromptTests(unittest.TestCase):
    def test_target_labels(self) -> None:
        self.assertIn("Traditional Chinese", build_system_prompt("zh-hant"))
        self.assertIn("Traditional Chinese", build_system_prompt("zh-tw"))
        self.assertIn("Simplified Chinese", build_system_prompt("zh"))
        self.assertIn("natural English", build_system_prompt("en"))

    def test_number_guard_present(self) -> None:
        # round 0073: "4到5個月" -> "April to May" slip class; the prompt carries a guard.
        self.assertIn("numbers", build_system_prompt("zh-hant"))


class UnavailableStatesTests(unittest.TestCase):
    def test_disabled_by_config(self) -> None:
        backend = LlmTranslator(enabled=False, source_code="auto", target_code="zh-hant")
        self.assertFalse(backend.enabled)
        self.assertIn("disabled", backend.state.message.lower())
        self.assertIsNone(backend.translate("hello"))

    def test_missing_server_path(self) -> None:
        backend = LlmTranslator(enabled=True, source_code="auto", target_code="zh-hant",
                                server_path="", model_path="")
        self.assertFalse(backend.enabled)
        self.assertIn("translation_llm_server_path", backend.state.message)

    def test_missing_model_path(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False) as fake_server:
            server_path = fake_server.name
        backend = LlmTranslator(enabled=True, source_code="auto", target_code="zh-hant",
                                server_path=server_path, model_path="Z:\\nope\\model.gguf")
        self.assertFalse(backend.enabled)
        self.assertIn("translation_llm_model_path", backend.state.message)
        Path(server_path).unlink(missing_ok=True)


class FakeServerRoundTripTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.httpd = HTTPServer(("127.0.0.1", 0), _FakeLlamaHandler)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.server_file = Path(cls.tmp.name) / "llama-server.exe"
        cls.model_file = Path(cls.tmp.name) / "model.gguf"
        cls.server_file.write_bytes(b"x")
        cls.model_file.write_bytes(b"x")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.httpd.shutdown()
        cls.tmp.cleanup()

    def _backend(self, **kwargs) -> LlmTranslator:
        defaults = dict(
            enabled=True, source_code="auto", target_code="zh-hant",
            server_path=str(self.server_file), model_path=str(self.model_file),
            port=self.port,
        )
        defaults.update(kwargs)
        return LlmTranslator(**defaults)

    def test_reuses_existing_healthy_server_and_translates(self) -> None:
        backend = self._backend()
        self.assertTrue(_wait_ready(backend), backend.state.message)
        # A healthy server on the port is reused, never owned/killed.
        self.assertFalse(backend._owns_server)
        out = backend.translate("Hello world", source_code="en")
        self.assertEqual(out, "這是譯文")
        request = _FakeLlamaHandler.last_request
        self.assertEqual(request["temperature"], 0.0)
        self.assertEqual(request["messages"][1]["content"], "Hello world")
        self.assertIn("Traditional Chinese", request["messages"][0]["content"])
        backend.shutdown()  # no-op for reused server; must not raise

    def test_same_language_family_skipped(self) -> None:
        backend = self._backend()
        self.assertTrue(_wait_ready(backend), backend.state.message)
        self.assertIsNone(backend.translate("你好世界", source_code="zh"))
        self.assertIsNone(backend.translate("你好世界", source_code="zh-hant"))
        self.assertEqual(backend.translate("Hello", source_code="en"), "這是譯文")

    def test_empty_text_skipped(self) -> None:
        backend = self._backend()
        self.assertTrue(_wait_ready(backend), backend.state.message)
        self.assertIsNone(backend.translate("   "))

    def test_state_message_reports_active(self) -> None:
        backend = self._backend()
        self.assertTrue(_wait_ready(backend), backend.state.message)
        self.assertTrue(backend.state.active)
        self.assertIn("model.gguf", backend.state.message)


def _cheap_backend(**kwargs) -> LlmTranslator:
    """Constructor-only backend (enabled=False -> no warmup thread, no real paths)."""
    defaults = dict(enabled=False, source_code="auto", target_code="zh-hant")
    defaults.update(kwargs)
    return LlmTranslator(**defaults)


class AutoSizingTests(unittest.TestCase):
    def test_choose_context_tier_prefers_largest_fitting(self) -> None:
        weights = 2400
        self.assertEqual(choose_context_tier(llm_footprint_mib(weights, 4096), weights), (4096, True))
        self.assertEqual(choose_context_tier(llm_footprint_mib(weights, 2048), weights), (2048, True))
        self.assertEqual(choose_context_tier(llm_footprint_mib(weights, 1024), weights), (1024, True))
        # below the smallest tier's footprint -> CPU fallback
        self.assertEqual(choose_context_tier(llm_footprint_mib(weights, 1024) - 1, weights),
                         (CONTEXT_TIERS[-1], False))

    def test_asr_reserve_estimates(self) -> None:
        cfg = RuntimeConfig()  # whisperx, cuda, model auto->large-v3, diar off
        self.assertEqual(estimate_asr_reserve_mib(cfg), 4200 + 600)
        cfg.whisperx_enable_diarization = True
        self.assertEqual(estimate_asr_reserve_mib(cfg), 4200 + 600 + 1300)
        cfg.whisperx_enable_diarization = False
        cfg.model_size = "small"
        self.assertEqual(estimate_asr_reserve_mib(cfg), 1800 + 600)
        cfg.model_device = "cpu"
        self.assertEqual(estimate_asr_reserve_mib(cfg), 0)
        cfg2 = RuntimeConfig()
        cfg2.stt_provider = "whispercpp"
        self.assertEqual(estimate_asr_reserve_mib(cfg2), 1800)

    def test_manual_context_pin_is_never_resized(self) -> None:
        backend = _cheap_backend(context_size=3000, gpu_layers=99)
        original = llm_backend.probe_free_vram_mib
        llm_backend.probe_free_vram_mib = lambda: 500  # would force CPU tier if consulted
        try:
            backend._resolve_auto_sizing()
        finally:
            llm_backend.probe_free_vram_mib = original
        self.assertEqual(backend._effective_context, 3000)
        self.assertEqual(backend._effective_gpu_layers, 99)

    def test_auto_context_uses_budget(self) -> None:
        backend = _cheap_backend(context_size=0, gpu_layers=99, asr_reserve_mib=6100)
        original = llm_backend.probe_free_vram_mib
        llm_backend.probe_free_vram_mib = lambda: 12000  # budget 5900, weights fallback 2500
        try:
            backend._resolve_auto_sizing()
        finally:
            llm_backend.probe_free_vram_mib = original
        self.assertEqual(backend._effective_context, 4096)
        self.assertEqual(backend._effective_gpu_layers, 99)

    def test_auto_context_falls_to_cpu_when_budget_tiny(self) -> None:
        backend = _cheap_backend(context_size=0, gpu_layers=99, asr_reserve_mib=6100)
        original = llm_backend.probe_free_vram_mib
        llm_backend.probe_free_vram_mib = lambda: 8000  # budget 1900 < smallest footprint
        try:
            backend._resolve_auto_sizing()
        finally:
            llm_backend.probe_free_vram_mib = original
        self.assertEqual(backend._effective_context, CONTEXT_TIERS[-1])
        self.assertEqual(backend._effective_gpu_layers, 0)

    def test_probe_unavailable_uses_conservative_default(self) -> None:
        backend = _cheap_backend(context_size=0, gpu_layers=99)
        original = llm_backend.probe_free_vram_mib
        llm_backend.probe_free_vram_mib = lambda: None
        try:
            backend._resolve_auto_sizing()
        finally:
            llm_backend.probe_free_vram_mib = original
        self.assertEqual(backend._effective_context, 2048)
        self.assertEqual(backend._effective_gpu_layers, 99)


class RetryLadderTests(unittest.TestCase):
    def test_auto_ladder_halves_context_then_cpu(self) -> None:
        backend = _cheap_backend(context_size=0, gpu_layers=99)
        backend._effective_context = 4096
        backend._effective_gpu_layers = 99
        self.assertEqual(backend._attempt_ladder(),
                         [(4096, 99), (2048, 99), (1024, 99), (1024, 0)])

    def test_manual_ladder_only_degrades_gpu_layers(self) -> None:
        backend = _cheap_backend(context_size=3000, gpu_layers=99)
        backend._effective_context = 3000
        backend._effective_gpu_layers = 99
        self.assertEqual(backend._attempt_ladder(), [(3000, 99), (3000, 0)])

    def test_cpu_only_ladder_has_single_attempt(self) -> None:
        backend = _cheap_backend(context_size=3000, gpu_layers=0)
        backend._effective_context = 3000
        backend._effective_gpu_layers = 0
        self.assertEqual(backend._attempt_ladder(), [(3000, 0)])

    def test_degraded_tier_auto_halves_then_manual_drops_gpu(self) -> None:
        auto = _cheap_backend(context_size=0, gpu_layers=99)
        auto._effective_context, auto._effective_gpu_layers = 4096, 99
        self.assertEqual(auto._degraded_tier(), (2048, 99))
        auto._effective_context = 1024
        self.assertEqual(auto._degraded_tier(), (1024, 0))
        manual = _cheap_backend(context_size=3000, gpu_layers=99)
        manual._effective_context, manual._effective_gpu_layers = 3000, 99
        self.assertEqual(manual._degraded_tier(), (3000, 0))


class RegistryTests(unittest.TestCase):
    def test_llm_token_builds_real_backend(self) -> None:
        cfg = RuntimeConfig()
        cfg.translation_enabled = False  # constructor stays cheap, no warmup
        backend = build_backend("llm", cfg)
        self.assertIsInstance(backend, LlmTranslator)
        self.assertEqual(backend.name, "llm")

    def test_cloud_still_reserved_stub(self) -> None:
        backend = build_backend("cloud", RuntimeConfig())
        self.assertIsInstance(backend, UnavailableBackend)

    def test_config_defaults_exist(self) -> None:
        cfg = RuntimeConfig()
        self.assertEqual(cfg.translation_llm_port, 8474)
        self.assertEqual(cfg.translation_llm_server_path, "")
        self.assertEqual(cfg.translation_llm_gpu_layers, 99)
        # round 0075: context default is the AUTO sentinel (VRAM-aware sizing)
        self.assertEqual(cfg.translation_llm_context_size, 0)


if __name__ == "__main__":
    unittest.main()
