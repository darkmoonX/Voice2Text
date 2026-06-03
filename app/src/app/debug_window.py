from __future__ import annotations

from datetime import datetime
import json
import logging
from pathlib import Path
import threading
from collections import deque
from typing import Deque

from PySide6.QtCore import QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QPlainTextEdit, QVBoxLayout, QWidget


class DebugWindowLogHandler(logging.Handler):
    """Thread-safe logger handler that forwards formatted lines to debug window queue."""

    def __init__(self, append_line) -> None:
        super().__init__(level=logging.INFO)
        self._append_line = append_line

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:
            return
        try:
            self._append_line(message)
        except Exception:
            return


class STTDebugWindow(QWidget):
    def __init__(self, debug_log_dir: str | Path | None = None) -> None:
        super().__init__(None)
        self.setWindowTitle("Voice2Text Debug Trace")
        self.resize(960, 620)
        self.setFont(QFont("Segoe UI", 10))
        layout = QVBoxLayout(self)
        self._log = QPlainTextEdit(self)
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(8000)
        layout.addWidget(self._log)
        self._pending_lines: Deque[str] = deque()
        self._pending_lock = threading.Lock()
        self._flush_timer = QTimer(self)
        self._flush_timer.setInterval(50)
        self._flush_timer.timeout.connect(self._flush_pending_lines)
        self._flush_timer.start()

        root = Path(debug_log_dir) if debug_log_dir else (Path(__file__).resolve().parents[1] / "debug_logs")
        root.mkdir(parents=True, exist_ok=True)
        self._debug_log_file = root / f"debug_trace_{datetime.now().strftime('%Y%m%d')}.jsonl"

    def append_event(self, payload: dict[str, object]) -> None:
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        meta = payload.get("meta", {})
        speaker_profile_stats = {}
        if isinstance(meta, dict):
            raw_stats = meta.get("speaker_profile_stats")
            if isinstance(raw_stats, dict):
                speaker_profile_stats = raw_stats
        lines = [
            f"[{ts}] provider={payload.get('provider', '')}",
            f"raw={payload.get('raw_text', '')}",
            f"merged={payload.get('merged_text', '')}",
            f"history={payload.get('history_text', '')}",
            f"stable={payload.get('stable_text', '')}",
            f"partial_state={payload.get('partial_state', [])}",
            f"speaker_profile_stats={speaker_profile_stats}",
            f"meta={meta}",
        ]
        self._enqueue_line("\n".join(lines) + "\n" + ("-" * 72))

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
            "speaker_profile_stats": speaker_profile_stats,
            "meta": meta,
        }
        try:
            with self._debug_log_file.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def append_log_line(self, line: str) -> None:
        self._enqueue_line(str(line or ""))

    def load_runtime_history(self, log_dir: str | Path, *, max_lines: int = 2500) -> None:
        root = Path(log_dir)
        if not root.exists() or (not root.is_dir()):
            return
        try:
            candidates = sorted(
                [path for path in root.glob("voice2text.log*") if path.is_file()],
                key=lambda path: path.stat().st_mtime,
            )
        except Exception:
            return
        merged: list[str] = []
        for path in candidates:
            try:
                merged.extend(path.read_text(encoding="utf-8", errors="replace").splitlines())
            except Exception:
                continue
        if max_lines > 0 and len(merged) > max_lines:
            merged = merged[-max_lines:]
        if not merged:
            return
        self._enqueue_line("===== Runtime log history loaded =====")
        for line in merged:
            self._enqueue_line(line)
        self._enqueue_line("===== End of history =====")

    def closeEvent(self, event) -> None:
        try:
            self._flush_timer.stop()
            self._flush_pending_lines()
        except Exception:
            pass
        super().closeEvent(event)

    def _enqueue_line(self, line: str) -> None:
        text = line.strip("\r\n")
        if not text:
            return
        with self._pending_lock:
            self._pending_lines.append(text)

    def _flush_pending_lines(self) -> None:
        batch: list[str] = []
        with self._pending_lock:
            while self._pending_lines and len(batch) < 200:
                batch.append(self._pending_lines.popleft())
        if not batch:
            return
        self._log.appendPlainText("\n".join(batch))
