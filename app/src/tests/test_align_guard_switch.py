"""Unit tests for the WhisperX alignment CUDA safety guard runtime switch (round 0028)."""
from __future__ import annotations

from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest import mock

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice2text.stt import whisperx_provider
from voice2text.stt.align_probe_cache import read_cached_verdict, write_cached_verdict
from voice2text.stt.whisperx_provider import WhisperXTranscriber


class AlignGuardNormalizeTests(unittest.TestCase):
    def test_default_is_safe(self) -> None:
        self.assertEqual(WhisperXTranscriber._normalize_align_guard(None), "safe")
        self.assertEqual(WhisperXTranscriber._normalize_align_guard(""), "safe")
        self.assertEqual(WhisperXTranscriber._normalize_align_guard("safe"), "safe")
        self.assertEqual(WhisperXTranscriber._normalize_align_guard("probe"), "probe")

    def test_unsafe_aliases(self) -> None:
        for value in ("unsafe-cuda", "unsafe_cuda", "UNSAFE-CUDA", " unsafe ", "cuda"):
            self.assertEqual(WhisperXTranscriber._normalize_align_guard(value), "unsafe-cuda")

    def test_unknown_falls_back_to_safe(self) -> None:
        self.assertEqual(WhisperXTranscriber._normalize_align_guard("garbage"), "safe")


class AlignGuardPolicyTests(unittest.TestCase):
    def _stub(self, guard: str, *, model_root: Path | None = None) -> WhisperXTranscriber:
        inst = WhisperXTranscriber.__new__(WhisperXTranscriber)
        inst._align_guard = WhisperXTranscriber._normalize_align_guard(guard)
        inst._alignment_language = "en"
        inst._source_language_hint = None
        inst._alignment_model = "facebook/wav2vec2-base-960h"
        inst._english_align_large = False
        inst._model_root = model_root or Path(tempfile.gettempdir()) / "voice2text-align-probe-test"
        inst._hf_token = "secret-token-must-not-be-passed"
        inst._messages: list[str] = []
        inst._emit = inst._messages.append  # type: ignore[assignment]
        return inst

    def test_non_cuda_unchanged(self) -> None:
        inst = self._stub("safe")
        with mock.patch.object(whisperx_provider.os, "name", "nt"):
            self.assertEqual(inst._apply_alignment_device_safety_policy("cpu"), "cpu")
            self.assertEqual(inst._apply_alignment_device_safety_policy(""), "cpu")
        self.assertEqual(inst._messages, [])

    def test_safe_downgrades_cuda_to_cpu_on_windows(self) -> None:
        inst = self._stub("safe")
        with mock.patch.object(whisperx_provider.os, "name", "nt"), \
                mock.patch.dict(whisperx_provider.os.environ, {}, clear=False):
            whisperx_provider.os.environ.pop("VOICE2TEXT_WHISPERX_ALLOW_UNSAFE_CUDA_ALIGN", None)
            self.assertEqual(inst._apply_alignment_device_safety_policy("cuda"), "cpu")
        self.assertTrue(any("downgraded to CPU" in m for m in inst._messages))

    def test_unsafe_cuda_keeps_cuda_with_warning_on_windows(self) -> None:
        inst = self._stub("unsafe-cuda")
        with mock.patch.object(whisperx_provider.os, "name", "nt"):
            self.assertEqual(inst._apply_alignment_device_safety_policy("cuda"), "cuda")
        self.assertTrue(any("unsafe-cuda" in m for m in inst._messages))

    def test_env_var_overrides_safe_default(self) -> None:
        inst = self._stub("safe")
        with mock.patch.object(whisperx_provider.os, "name", "nt"), \
                mock.patch.dict(
                    whisperx_provider.os.environ,
                    {"VOICE2TEXT_WHISPERX_ALLOW_UNSAFE_CUDA_ALIGN": "1"},
                    clear=False,
                ):
            self.assertEqual(inst._apply_alignment_device_safety_policy("cuda"), "cuda")
        self.assertTrue(any("bypassed by" in m for m in inst._messages))

    def test_non_windows_keeps_cuda_regardless_of_guard(self) -> None:
        for guard in ("safe", "unsafe-cuda", "probe"):
            inst = self._stub(guard)
            with mock.patch.object(whisperx_provider.os, "name", "posix"):
                self.assertEqual(inst._apply_alignment_device_safety_policy("cuda"), "cuda")
            self.assertEqual(inst._messages, [])

    def test_probe_cache_safe_keeps_cuda_without_spawning(self) -> None:
        signature = {
            "torch_version": "2.test",
            "cuda_version": "12.test",
            "gpu_name": "Test GPU",
            "align_model_repo": "facebook/wav2vec2-base-960h",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_cached_verdict(root / "align_cuda_probe.json", signature, cuda_safe=True, reason="ok")
            inst = self._stub("probe", model_root=root)
            with mock.patch.object(whisperx_provider.os, "name", "nt"), \
                    mock.patch.object(whisperx_provider, "collect_probe_signature", return_value=signature), \
                    mock.patch.object(whisperx_provider.subprocess, "run") as run:
                self.assertEqual(inst._apply_alignment_device_safety_policy("cuda"), "cuda")
                run.assert_not_called()
        self.assertTrue(any("cache hit" in m for m in inst._messages))

    def test_probe_cache_unsafe_downgrades_without_spawning(self) -> None:
        signature = {
            "torch_version": "2.test",
            "cuda_version": "12.test",
            "gpu_name": "Test GPU",
            "align_model_repo": "facebook/wav2vec2-base-960h",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_cached_verdict(root / "align_cuda_probe.json", signature, cuda_safe=False, reason="returncode=3221225477")
            inst = self._stub("probe", model_root=root)
            with mock.patch.object(whisperx_provider.os, "name", "nt"), \
                    mock.patch.object(whisperx_provider, "collect_probe_signature", return_value=signature), \
                    mock.patch.object(whisperx_provider.subprocess, "run") as run:
                self.assertEqual(inst._apply_alignment_device_safety_policy("cuda"), "cpu")
                run.assert_not_called()
        self.assertTrue(any("cache hit" in m for m in inst._messages))

    def test_probe_cache_miss_success_spawns_once_and_caches(self) -> None:
        signature = {
            "torch_version": "2.test",
            "cuda_version": "12.test",
            "gpu_name": "Test GPU",
            "align_model_repo": "facebook/wav2vec2-base-960h",
        }
        completed = mock.Mock(returncode=0, stdout='{"ok": true, "word_count": 3}\n')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inst = self._stub("probe", model_root=root)
            with mock.patch.object(whisperx_provider.os, "name", "nt"), \
                    mock.patch.dict(whisperx_provider.os.environ, {}, clear=False), \
                    mock.patch.object(whisperx_provider, "collect_probe_signature", return_value=signature), \
                    mock.patch.object(whisperx_provider.subprocess, "run", return_value=completed) as run:
                whisperx_provider.os.environ.pop("VOICE2TEXT_WHISPERX_ALIGN_PROBE_FORCE", None)
                self.assertEqual(inst._apply_alignment_device_safety_policy("cuda"), "cuda")
                self.assertEqual(inst._apply_alignment_device_safety_policy("cuda"), "cuda")
                run.assert_called_once()
                cmd = run.call_args.args[0]
                self.assertNotIn("secret-token-must-not-be-passed", " ".join(map(str, cmd)))
                # Must invoke the probe by FILE path (cwd-independent), never `-m` (ModuleNotFound
                # under the launching app's cwd would false-fail rc=1).
                self.assertNotIn("-m", cmd)
                self.assertTrue(any(str(part).endswith("align_cuda_probe.py") for part in cmd))
        self.assertTrue(any("probe passed" in m for m in inst._messages))

    def test_probe_timeout_downgrades_to_cpu(self) -> None:
        signature = {
            "torch_version": "2.test",
            "cuda_version": "12.test",
            "gpu_name": "Test GPU",
            "align_model_repo": "facebook/wav2vec2-base-960h",
        }
        with tempfile.TemporaryDirectory() as tmp:
            inst = self._stub("probe", model_root=Path(tmp))
            with mock.patch.object(whisperx_provider.os, "name", "nt"), \
                    mock.patch.object(whisperx_provider, "collect_probe_signature", return_value=signature), \
                    mock.patch.object(
                        whisperx_provider.subprocess,
                        "run",
                        side_effect=subprocess.TimeoutExpired(cmd=["probe"], timeout=1.0),
                    ):
                self.assertEqual(inst._apply_alignment_device_safety_policy("cuda"), "cpu")
        self.assertTrue(any("timeout" in m for m in inst._messages))

    def test_probe_spawn_error_retries_then_downgrades_without_caching(self) -> None:
        signature = {
            "torch_version": "2.test",
            "cuda_version": "12.test",
            "gpu_name": "Test GPU",
            "align_model_repo": "facebook/wav2vec2-base-960h",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inst = self._stub("probe", model_root=root)
            with mock.patch.object(whisperx_provider.os, "name", "nt"), \
                    mock.patch.object(whisperx_provider.time, "sleep"), \
                    mock.patch.object(whisperx_provider, "collect_probe_signature", return_value=signature), \
                    mock.patch.object(
                        whisperx_provider.subprocess, "run", side_effect=OSError("spawn failed")
                    ) as run:
                self.assertEqual(inst._apply_alignment_device_safety_policy("cuda"), "cpu")
                # transient failure -> retried once (two spawns)
                self.assertEqual(run.call_count, 2)
            # an inconclusive failure must NOT poison the cache
            self.assertIsNone(read_cached_verdict(root / "align_cuda_probe.json", signature))
        self.assertTrue(any("spawn-error" in m for m in inst._messages))
        self.assertTrue(any("without caching" in m for m in inst._messages))

    def test_probe_transient_then_success_self_heals_and_caches(self) -> None:
        signature = {
            "torch_version": "2.test",
            "cuda_version": "12.test",
            "gpu_name": "Test GPU",
            "align_model_repo": "facebook/wav2vec2-base-960h",
        }
        first = mock.Mock(returncode=1, stdout="")
        second = mock.Mock(returncode=0, stdout='{"ok": true, "word_count": 3}\n')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inst = self._stub("probe", model_root=root)
            with mock.patch.object(whisperx_provider.os, "name", "nt"), \
                    mock.patch.object(whisperx_provider.time, "sleep"), \
                    mock.patch.object(whisperx_provider, "collect_probe_signature", return_value=signature), \
                    mock.patch.object(
                        whisperx_provider.subprocess, "run", side_effect=[first, second]
                    ) as run:
                self.assertEqual(inst._apply_alignment_device_safety_policy("cuda"), "cuda")
                self.assertEqual(run.call_count, 2)
            # the clean retry is trusted and cached True (self-heal)
            self.assertIs(read_cached_verdict(root / "align_cuda_probe.json", signature), True)
        self.assertTrue(any("retrying once" in m for m in inst._messages))
        self.assertTrue(any("probe passed" in m for m in inst._messages))

    def test_probe_access_violation_is_cached_unsafe_without_retry(self) -> None:
        signature = {
            "torch_version": "2.test",
            "cuda_version": "12.test",
            "gpu_name": "Test GPU",
            "align_model_repo": "facebook/wav2vec2-base-960h",
        }
        crash = mock.Mock(returncode=3221225477, stdout="")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inst = self._stub("probe", model_root=root)
            with mock.patch.object(whisperx_provider.os, "name", "nt"), \
                    mock.patch.object(whisperx_provider.time, "sleep") as sleep, \
                    mock.patch.object(whisperx_provider, "collect_probe_signature", return_value=signature), \
                    mock.patch.object(
                        whisperx_provider.subprocess, "run", return_value=crash
                    ) as run:
                self.assertEqual(inst._apply_alignment_device_safety_policy("cuda"), "cpu")
                # access violation is deterministic: single spawn, no retry/sleep
                run.assert_called_once()
                sleep.assert_not_called()
            # deterministic crash IS cached so later starts short-circuit to CPU
            self.assertIs(read_cached_verdict(root / "align_cuda_probe.json", signature), False)
        self.assertTrue(any("3221225477" in m for m in inst._messages))


if __name__ == "__main__":
    unittest.main()
