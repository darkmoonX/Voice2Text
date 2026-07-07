"""Round 0039: multi-app simultaneous capture wiring in build_cpp_capture_from_config."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice2text.audio_capture import MixedAudioCapture
from voice2text.capture import cpp_backend
from voice2text.capture.cpp_backend import CppBridgeCapture, build_cpp_capture_from_config
from voice2text.config import RuntimeConfig


def _app_cfg(app_names, *, weights=None, channel_mode="mono") -> RuntimeConfig:
    cfg = RuntimeConfig()
    cfg.source_mode = "app"
    cfg.source_app_names = list(app_names)
    cfg.source_mix_weights = list(weights or [])
    cfg.source_channel_mode = channel_mode
    return cfg


class MultiAppCaptureWiringTest(unittest.TestCase):
    def setUp(self) -> None:
        # The bridge exe need not exist: CppBridgeCapture only spawns on start(); we build, not start.
        self._patchers = [
            patch.object(cpp_backend, "resolve_capture_bridge_executable", lambda: Path("fake_bridge.exe")),
            patch.object(cpp_backend, "check_bridge_health", lambda _exe: (True, "")),
            patch.object(cpp_backend, "check_process_loopback_support", lambda _exe: (True, "")),
        ]
        for p in self._patchers:
            p.start()

    def tearDown(self) -> None:
        for p in self._patchers:
            p.stop()

    def test_single_app_returns_single_bridge_capture(self) -> None:
        capture = build_cpp_capture_from_config(_app_cfg(["chrome.exe"]))
        self.assertIsInstance(capture, CppBridgeCapture)
        self.assertEqual(capture._app_names, ["chrome.exe"])

    def test_two_apps_return_mixed_of_two_single_app_bridges(self) -> None:
        capture = build_cpp_capture_from_config(_app_cfg(["chrome.exe", "vlc.exe"]))
        self.assertIsInstance(capture, MixedAudioCapture)
        inner = capture._captures
        self.assertEqual(len(inner), 2)
        for c in inner:
            self.assertIsInstance(c, CppBridgeCapture)
        # each inner bridge targets exactly one app
        self.assertEqual([c._app_names for c in inner], [["chrome.exe"], ["vlc.exe"]])
        # mix target is the STT input format
        self.assertEqual(capture.sample_rate, 16000)

    def test_three_apps_mixed(self) -> None:
        capture = build_cpp_capture_from_config(_app_cfg(["chrome.exe", "msedge.exe", "vlc.exe"]))
        self.assertIsInstance(capture, MixedAudioCapture)
        self.assertEqual(len(capture._captures), 3)

    def test_duplicate_app_names_collapse_to_single(self) -> None:
        # _normalize_app_target_names dedupes -> one app -> single capture, not a mixer.
        capture = build_cpp_capture_from_config(_app_cfg(["chrome.exe", "chrome.exe"]))
        self.assertIsInstance(capture, CppBridgeCapture)
        self.assertEqual(capture._app_names, ["chrome.exe"])

    def test_matching_weights_are_passed_through(self) -> None:
        capture = build_cpp_capture_from_config(_app_cfg(["chrome.exe", "vlc.exe"], weights=[0.7, 1.3]))
        self.assertIsInstance(capture, MixedAudioCapture)
        self.assertEqual(capture._weights, [0.7, 1.3])

    def test_mismatched_weights_fall_back_to_equal(self) -> None:
        capture = build_cpp_capture_from_config(_app_cfg(["chrome.exe", "vlc.exe"], weights=[0.5]))
        self.assertIsInstance(capture, MixedAudioCapture)
        # equal weights default inside MixedAudioCapture (one per capture)
        self.assertEqual(capture._weights, [1.0, 1.0])

    def test_channel_mode_forwarded(self) -> None:
        capture = build_cpp_capture_from_config(_app_cfg(["a.exe", "b.exe"], channel_mode="stereo"))
        self.assertIsInstance(capture, MixedAudioCapture)
        self.assertEqual(capture._channel_mode, "stereo")


if __name__ == "__main__":
    unittest.main()
