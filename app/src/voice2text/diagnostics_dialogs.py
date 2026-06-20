"""Qt dialogs for the health check + model/cache manager (round 0022 Phase B).

Thin presentation over the headless cores (`stt.healthcheck`, `stt.model_cache`). The slow work runs on a
background thread and results return to the UI thread via Qt signals (queued connection) — the UI thread
never blocks. `_populate_*` are split out so the rendering is unit-testable with synthetic data.
"""
from __future__ import annotations

import threading
from typing import Callable

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from .config import RuntimeConfig

_I18N = {
    "zh": {
        "health_title": "執行環境健檢",
        "cache_title": "模型 / 快取管理",
        "running": "檢查中…",
        "scanning": "掃描中…",
        "rerun": "重新檢查",
        "refresh": "重新整理",
        "delete": "刪除選取",
        "close": "關閉",
        "col_status": "狀態", "col_check": "項目", "col_detail": "說明", "col_fix": "建議修復",
        "col_name": "名稱", "col_lang": "語言", "col_kind": "類型", "col_size": "大小", "col_ready": "就緒",
        "total": "總計", "folders": "個資料夾", "yes": "是", "no": "否",
        "confirm_delete": "確定刪除這個快取項目嗎？", "freed": "已釋放",
        "failed": "失敗",
    },
    "en": {
        "health_title": "Runtime Health Check",
        "cache_title": "Model / Cache Manager",
        "running": "Running…",
        "scanning": "Scanning…",
        "rerun": "Re-run",
        "refresh": "Refresh",
        "delete": "Delete selected",
        "close": "Close",
        "col_status": "Status", "col_check": "Check", "col_detail": "Detail", "col_fix": "Fix",
        "col_name": "Name", "col_lang": "Lang", "col_kind": "Kind", "col_size": "Size", "col_ready": "Ready",
        "total": "Total", "folders": "folders", "yes": "Yes", "no": "No",
        "confirm_delete": "Delete this cache entry?", "freed": "Freed",
        "failed": "Failed",
    },
}

_STATUS_COLOR = {"ok": "#2E7D32", "warn": "#B26A00", "fail": "#C62828"}


class _Worker(QObject):
    """Runs a callable on a plain thread and marshals the result back via a queued signal."""

    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn: Callable[[], object]) -> None:
        super().__init__()
        self._fn = fn

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        try:
            result = self._fn()
        except Exception as exc:  # surface to the UI, never crash the thread
            self.failed.emit(str(exc))
            return
        self.done.emit(result)


def _lang(config: RuntimeConfig) -> str:
    value = str(getattr(config, "ui_language", "zh") or "zh").strip().lower()
    return value if value in _I18N else "zh"


class HealthCheckDialog(QDialog):
    def __init__(self, config: RuntimeConfig, *, parent=None, scope: str = "active") -> None:
        super().__init__(parent)
        self._config = config
        self._scope = scope
        self._t = _I18N[_lang(config)]
        self._worker: _Worker | None = None
        self.setWindowTitle(self._t["health_title"])
        self.resize(720, 360)

        layout = QVBoxLayout(self)
        self._status_label = QLabel(self._t["running"])
        layout.addWidget(self._status_label)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(
            [self._t["col_status"], self._t["col_check"], self._t["col_detail"], self._t["col_fix"]]
        )
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self._table)

        buttons = QHBoxLayout()
        self._rerun_btn = QPushButton(self._t["rerun"])
        self._rerun_btn.clicked.connect(self._run)
        close_btn = QPushButton(self._t["close"])
        close_btn.clicked.connect(self.accept)
        buttons.addWidget(self._rerun_btn)
        buttons.addStretch(1)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)

        self._run()

    def _run(self) -> None:
        self._rerun_btn.setEnabled(False)
        self._status_label.setText(self._t["running"])
        config, scope = self._config, self._scope

        def work():
            from .stt.healthcheck import run_provider_health_check

            return run_provider_health_check(config, scope=scope)

        self._worker = _Worker(work)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_failed(self, message: str) -> None:
        self._rerun_btn.setEnabled(True)
        self._status_label.setText(f"{self._t['failed']}: {message}")

    def _on_done(self, reports) -> None:
        self._rerun_btn.setEnabled(True)
        self._populate_health(reports)

    def _populate_health(self, reports) -> None:
        """Render `ProviderHealthReport` rows into the table (separated for testability)."""
        rows = []
        multi = len(reports) > 1
        any_fail = False
        for report in reports:
            if not getattr(report, "ok", True):
                any_fail = True
            for check in getattr(report, "checks", []) or []:
                label = check.label if not multi else f"{report.provider}: {check.label}"
                rows.append((check.status, label, check.detail, check.fix_hint))
        self._table.setRowCount(len(rows))
        for r, (status, label, detail, fix) in enumerate(rows):
            status_item = QTableWidgetItem(status.upper())
            status_item.setData(Qt.ItemDataRole.UserRole, status)
            status_item.setForeground(QBrush(QColor(_STATUS_COLOR.get(status, "#444444"))))
            self._table.setItem(r, 0, status_item)
            self._table.setItem(r, 1, QTableWidgetItem(str(label)))
            self._table.setItem(r, 2, QTableWidgetItem(str(detail)))
            self._table.setItem(r, 3, QTableWidgetItem(str(fix) if status != "ok" else ""))
        overall = "FAIL" if any_fail else "OK"
        self._status_label.setText(f"{overall}  ({len(rows)})")
        return rows


class ModelCacheDialog(QDialog):
    def __init__(self, config: RuntimeConfig, *, parent=None) -> None:
        super().__init__(parent)
        self._config = config
        self._t = _I18N[_lang(config)]
        self._worker: _Worker | None = None
        self.setWindowTitle(self._t["cache_title"])
        self.resize(760, 420)

        layout = QVBoxLayout(self)
        self._header = QLabel(self._t["scanning"])
        layout.addWidget(self._header)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            [self._t["col_name"], self._t["col_lang"], self._t["col_kind"], self._t["col_size"], self._t["col_ready"]]
        )
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self._table)

        buttons = QHBoxLayout()
        self._refresh_btn = QPushButton(self._t["refresh"])
        self._refresh_btn.clicked.connect(self._run)
        self._delete_btn = QPushButton(self._t["delete"])
        self._delete_btn.clicked.connect(self._delete_selected)
        close_btn = QPushButton(self._t["close"])
        close_btn.clicked.connect(self.accept)
        buttons.addWidget(self._refresh_btn)
        buttons.addWidget(self._delete_btn)
        buttons.addStretch(1)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)

        self._run()

    def _run(self) -> None:
        self._refresh_btn.setEnabled(False)
        self._header.setText(self._t["scanning"])

        def work():
            from .stt.model_cache import scan_model_cache

            return scan_model_cache()

        self._worker = _Worker(work)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_failed(self, message: str) -> None:
        self._refresh_btn.setEnabled(True)
        self._header.setText(f"{self._t['failed']}: {message}")

    def _on_done(self, scan) -> None:
        self._refresh_btn.setEnabled(True)
        self._populate_cache(scan)

    def _populate_cache(self, scan) -> None:
        """Render a `ModelCacheScan` into the table + header (separated for testability)."""
        from .stt.model_cache import human_size

        entries = list(getattr(scan, "entries", []))
        self._table.setRowCount(len(entries))
        for r, entry in enumerate(entries):
            name_item = QTableWidgetItem(str(entry.name))
            name_item.setData(Qt.ItemDataRole.UserRole, str(entry.path))
            self._table.setItem(r, 0, name_item)
            self._table.setItem(r, 1, QTableWidgetItem(str(entry.lang)))
            self._table.setItem(r, 2, QTableWidgetItem(str(entry.kind)))
            self._table.setItem(r, 3, QTableWidgetItem(human_size(entry.size_bytes)))
            self._table.setItem(r, 4, QTableWidgetItem(self._t["yes"] if entry.ready else self._t["no"]))
        self._header.setText(
            f"{self._t['total']}: {human_size(getattr(scan, 'total_bytes', 0))}  "
            f"({len(entries)} {self._t['folders']})"
        )
        return entries

    def _selected_path(self) -> str:
        items = self._table.selectedItems()
        if not items:
            return ""
        name_item = self._table.item(items[0].row(), 0)
        return str(name_item.data(Qt.ItemDataRole.UserRole) or "") if name_item else ""

    def _delete_selected(self) -> None:
        path = self._selected_path()
        if not path:
            return
        if QMessageBox.question(self, self._t["cache_title"], f"{self._t['confirm_delete']}\n\n{path}") != QMessageBox.StandardButton.Yes:
            return
        from .stt.model_cache import delete_cache_entry, human_size

        try:
            freed = delete_cache_entry(path)
        except Exception as exc:
            QMessageBox.warning(self, self._t["cache_title"], f"{self._t['failed']}: {exc}")
            return
        QMessageBox.information(self, self._t["cache_title"], f"{self._t['freed']}: {human_size(freed)}")
        self._run()
