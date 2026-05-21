"""Manual smoke helper to switch providers and validate basic runtime wiring."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import QCoreApplication

APP_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = APP_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from app.audio_capture import list_active_app_sessions
from app.config import RuntimeConfig
from app.controller import TranscriptionController


class _DummyTranscriber:
    def has_enough_signal(self, *_args: Any, **_kwargs: Any) -> bool:
        return False

    def transcribe(self, *_args: Any, **_kwargs: Any) -> str:
        return ""


def _pump(app: QCoreApplication, seconds: float) -> None:
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.02)


def _apply_and_restart(
    app: QCoreApplication,
    controller: TranscriptionController,
    cfg: RuntimeConfig,
    updates: dict[str, Any],
    label: str,
) -> None:
    for key, value in updates.items():
        setattr(cfg, key, value)

    controller.restart()
    _pump(app, 0.8)
    print(f"[ok] {label}")


def main() -> int:
    app = QCoreApplication.instance() or QCoreApplication([])

    sessions = list_active_app_sessions()
    preferred_app = "msedge.exe"
    if preferred_app not in sessions:
        preferred_app = sessions[0] if sessions else "System Sounds"

    cfg = RuntimeConfig(
        model_size="small",
        model_device="cpu",
        compute_type="int8",
        source_mode="loopback",
        source_device_indices=[],
        source_app_name="",
        source_app_names=[],
        source_language=None,
        segment_seconds=6.0,
        hop_seconds=1.5,
        translation_enabled=False,
        translation_from="auto",
        translation_to="zh",
        bilingual_style="stacked",
    )

    controller = TranscriptionController(cfg)
    controller._create_transcriber_with_fallback = lambda: _DummyTranscriber()  # type: ignore[attr-defined]

    errors: list[str] = []
    controller.error_message.connect(errors.append)

    controller.start()
    _pump(app, 0.8)

    _apply_and_restart(
        app,
        controller,
        cfg,
        {
            "source_mode": "app",
            "source_app_name": preferred_app,
            "source_app_names": [preferred_app],
            "source_device_indices": [],
        },
        "switch to app mode",
    )

    _apply_and_restart(
        app,
        controller,
        cfg,
        {
            "source_mode": "loopback",
            "source_app_name": "",
            "source_app_names": [],
            "source_device_indices": [],
        },
        "switch back to loopback mode",
    )

    _apply_and_restart(
        app,
        controller,
        cfg,
        {"segment_seconds": 7.0, "hop_seconds": 1.7},
        "increase segment/hop",
    )

    _apply_and_restart(
        app,
        controller,
        cfg,
        {"segment_seconds": 5.0, "hop_seconds": 1.2},
        "decrease segment/hop",
    )

    _apply_and_restart(
        app,
        controller,
        cfg,
        {
            "translation_enabled": True,
            "translation_from": "en",
            "translation_to": "zh",
            "bilingual_style": "stacked",
        },
        "enable translation stacked",
    )

    _apply_and_restart(
        app,
        controller,
        cfg,
        {
            "translation_enabled": True,
            "translation_from": "en",
            "translation_to": "ja",
            "bilingual_style": "translation-only",
        },
        "change translation target/style",
    )

    _apply_and_restart(
        app,
        controller,
        cfg,
        {"translation_enabled": False},
        "disable translation",
    )

    controller.stop()
    _pump(app, 0.2)

    if errors:
        print("[warn] controller emitted errors:")
        for item in errors:
            print(f"  - {item}")

    print("[done] smoke switch test completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
