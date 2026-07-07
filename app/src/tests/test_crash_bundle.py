"""Round 0025: crash bundle generator (redacted diagnostics zip)."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice2text.config import RuntimeConfig
from voice2text.crash_bundle import collect_environment, create_crash_bundle


def _build_runtime_tree(root: Path) -> RuntimeConfig:
    """A synthetic app/src-like tree: <root>/logs, <root>/debug_logs, <root>/runtime_settings.json."""
    logs = root / "logs"
    debug = root / "debug_logs"
    logs.mkdir(parents=True, exist_ok=True)
    debug.mkdir(parents=True, exist_ok=True)
    (logs / "app.log").write_text("hello log\n", encoding="utf-8")
    (logs / "python_crash_trace.log").write_text("traceback...\n", encoding="utf-8")
    (debug / "debug_trace_20260618.jsonl").write_text('{"window": 1}\n', encoding="utf-8")
    (root / "runtime_settings.json").write_text(
        json.dumps({"model_size": "small", "whisperx_hf_token": "super-secret-token"}),
        encoding="utf-8",
    )
    cfg = RuntimeConfig()
    cfg.log_dir = str(logs)
    cfg.whisperx_hf_token = "super-secret-token"
    return cfg


class CreateBundleTests(unittest.TestCase):
    def test_bundle_contents_and_redaction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "src"
            cfg = _build_runtime_tree(root)
            out = Path(tmp) / "out"
            zip_path = create_crash_bundle(cfg, out_dir=out, reason="unit-test")
            self.assertTrue(zip_path.exists())

            with zipfile.ZipFile(zip_path, "r") as zf:
                names = set(zf.namelist())
                blob = zf.read(zf.namelist()[0])  # touch one entry
                self.assertIn("manifest.json", names)
                self.assertIn("runtime_settings.json", names)
                self.assertIn("logs/app.log", names)
                self.assertIn("logs/python_crash_trace.log", names)
                self.assertIn("debug_logs/debug_trace_20260618.jsonl", names)

                manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
                settings = zf.read("runtime_settings.json").decode("utf-8")
                whole = b"".join(zf.read(n) for n in names).decode("utf-8", errors="replace")

            # Token never appears anywhere in the bundle.
            self.assertNotIn("super-secret-token", whole)
            self.assertEqual(manifest["config"].get("whisperx_hf_token"), "<redacted>")
            self.assertEqual(json.loads(settings).get("whisperx_hf_token"), "<redacted>")
            self.assertEqual(manifest["reason"], "unit-test")
            self.assertIn("environment", manifest)
            _ = blob

    def test_missing_inputs_are_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = RuntimeConfig()
            cfg.log_dir = str(Path(tmp) / "nonexistent" / "logs")  # nothing exists
            zip_path = create_crash_bundle(cfg, out_dir=Path(tmp) / "out", reason="empty")
            self.assertTrue(zip_path.exists())
            with zipfile.ZipFile(zip_path, "r") as zf:
                manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
            # Still produces a valid manifest with an environment report, just no collected files.
            self.assertIn("environment", manifest)
            self.assertIn("runtime_settings.json (missing/unreadable)", manifest["skipped"])

    def test_collect_environment_keys(self):
        env = collect_environment(RuntimeConfig())
        for key in ("platform", "python_version", "cuda_available", "ffmpeg", "capture_bridge", "model_cache"):
            self.assertIn(key, env)


if __name__ == "__main__":
    unittest.main()
