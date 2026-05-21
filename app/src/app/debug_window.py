from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QPlainTextEdit, QVBoxLayout, QWidget


class STTDebugWindow(QWidget):
    def __init__(self, debug_log_dir: str | Path | None = None) -> None:
        super().__init__(None)
        self.setWindowTitle("Voice2Text Debug Trace")
        self.resize(960, 620)
        self.setFont(QFont("Segoe UI", 10))
        layout = QVBoxLayout(self)
        self._log = QPlainTextEdit(self)
        self._log.setReadOnly(True)
        layout.addWidget(self._log)

        root = Path(debug_log_dir) if debug_log_dir else (Path(__file__).resolve().parents[1] / "debug_logs")
        root.mkdir(parents=True, exist_ok=True)
        self._debug_log_file = root / f"debug_trace_{datetime.now().strftime('%Y%m%d')}.jsonl"

    def append_event(self, payload: dict[str, object]) -> None:
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        lines = [
            f"[{ts}] provider={payload.get('provider', '')}",
            f"raw={payload.get('raw_text', '')}",
            f"merged={payload.get('merged_text', '')}",
            f"history={payload.get('history_text', '')}",
            f"stable={payload.get('stable_text', '')}",
            f"partial_state={payload.get('partial_state', [])}",
            f"meta={payload.get('meta', {})}",
        ]
        self._log.appendPlainText("\n".join(lines) + "\n" + ("-" * 72))

        record = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "provider": payload.get("provider", ""),
            "raw_text": payload.get("raw_text", ""),
            "merged_text": payload.get("merged_text", ""),
            "history_text": payload.get("history_text", ""),
            "history_state": payload.get("history_state", []),
            "stable_text": payload.get("stable_text", ""),
            "stable_state": payload.get("stable_state", []),
            "partial_state": payload.get("partial_state", []),
            "meta": payload.get("meta", {}),
        }
        try:
            with self._debug_log_file.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass
