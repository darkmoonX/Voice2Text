"""Round 0026: translation backend registry — name -> backend, reserved stubs, safe fallback."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice2text.config import RuntimeConfig
from voice2text.translation import ArgosTranslator, UnavailableBackend, build_backend


def _config(**over) -> RuntimeConfig:
    cfg = RuntimeConfig()
    cfg.translation_enabled = False  # keep Argos from touching the network in tests
    for key, value in over.items():
        setattr(cfg, key, value)
    return cfg


class RegistryTests(unittest.TestCase):
    def test_argos_by_name(self):
        backend = build_backend("argos", _config())
        self.assertIsInstance(backend, ArgosTranslator)
        self.assertEqual(backend.name, "argos")

    def test_default_none_is_argos(self):
        self.assertIsInstance(build_backend(None, _config()), ArgosTranslator)
        self.assertIsInstance(build_backend("", _config()), ArgosTranslator)

    def test_reserved_backends_are_disabled_stubs(self):
        # round 0074: 'llm' became a real backend (LlmTranslator); only 'cloud' remains reserved.
        for name in ("cloud",):
            backend = build_backend(name, _config())
            self.assertIsInstance(backend, UnavailableBackend)
            self.assertEqual(backend.name, name)
            self.assertFalse(backend.enabled)
            self.assertFalse(backend.state.active)
            self.assertIn("not yet implemented", backend.state.message)
            # translate never raises — returns None so the loop is unaffected.
            self.assertIsNone(backend.translate("hello", "en"))

    def test_llm_token_builds_real_backend(self):
        from voice2text.translation.llm_backend import LlmTranslator

        backend = build_backend("llm", _config())
        self.assertIsInstance(backend, LlmTranslator)
        self.assertEqual(backend.name, "llm")

    def test_unknown_falls_back_to_argos_with_warning(self):
        warnings: list[str] = []
        backend = build_backend("bogus-backend", _config(), on_status=warnings.append)
        self.assertIsInstance(backend, ArgosTranslator)
        self.assertTrue(any("bogus-backend" in w and "argos" in w for w in warnings))

    def test_reserved_backend_emits_no_fallback_warning(self):
        warnings: list[str] = []
        build_backend("llm", _config(), on_status=warnings.append)
        self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()
