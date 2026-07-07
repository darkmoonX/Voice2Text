"""System tray controller that bridges user actions to overlay/controller runtime updates."""
from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QAction, QColor, QFont, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from .capture import list_active_app_sessions, list_audio_devices
from .config import RuntimeConfig
from .settings_dialog import SettingsDialog

_I18N = {
    "zh": {
        "show": "顯示",
        "minimize": "最小化",
        "settings": "設定",
        "health": "執行環境健檢…",
        "cache": "模型 / 快取管理…",
        "crash_bundle": "建立診斷壓縮包…",
        "exit": "離開",
        "applied": "設定已套用。",
        "crash_bundle_writing": "正在建立診斷壓縮包…",
        "crash_bundle_done": "診斷壓縮包已建立",
        "crash_bundle_failed": "建立診斷壓縮包失敗",
    },
    "en": {
        "show": "Show",
        "minimize": "Minimize",
        "settings": "Settings",
        "health": "Runtime health check…",
        "cache": "Model / cache manager…",
        "crash_bundle": "Create diagnostics bundle…",
        "exit": "Exit",
        "applied": "Settings applied.",
        "crash_bundle_writing": "Writing diagnostics bundle…",
        "crash_bundle_done": "Diagnostics bundle created",
        "crash_bundle_failed": "Failed to create diagnostics bundle",
    },
}


class Voice2TextTrayController(QObject):
    _bundle_created = Signal(str)
    _bundle_failed = Signal(str)

    def __init__(
        self,
        app: QApplication,
        overlay,
        config: RuntimeConfig,
        on_settings_applied,
        on_export_transcript=None,
        on_import_audio=None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._app = app
        self._overlay = overlay
        self._config = config
        self._on_settings_applied = on_settings_applied
        self._on_export_transcript = on_export_transcript
        self._on_import_audio = on_import_audio

        self._tray = QSystemTrayIcon(self)
        self._icon = self._build_icon()
        self._app.setWindowIcon(self._icon)
        self._overlay.setWindowIcon(self._icon)
        self._tray.setIcon(self._icon)
        self._tray.setToolTip("Voice2Text")
        self._tray.setContextMenu(self._build_menu())
        self._tray.activated.connect(self._on_tray_activated)
        self._tray.show()
        self._bundle_created.connect(self._on_bundle_created)
        self._bundle_failed.connect(self._on_bundle_failed)

    def _lang(self) -> str:
        value = (self._config.ui_language or "zh").strip().lower()
        return value if value in _I18N else "zh"

    def _t(self, key: str) -> str:
        return _I18N[self._lang()][key]

    def refresh_locale(self) -> None:
        self._tray.setContextMenu(self._build_menu())

    def _build_menu(self) -> QMenu:
        menu = QMenu()
        show_action = QAction(self._t("show"), menu)
        minimize_action = QAction(self._t("minimize"), menu)
        settings_action = QAction(self._t("settings"), menu)
        health_action = QAction(self._t("health"), menu)
        cache_action = QAction(self._t("cache"), menu)
        crash_bundle_action = QAction(self._t("crash_bundle"), menu)
        exit_action = QAction(self._t("exit"), menu)
        show_action.triggered.connect(self.show_overlay)
        minimize_action.triggered.connect(self.hide_overlay)
        settings_action.triggered.connect(self.open_settings)
        health_action.triggered.connect(self.open_health_check)
        cache_action.triggered.connect(self.open_cache_manager)
        crash_bundle_action.triggered.connect(self.create_diagnostics_bundle)
        exit_action.triggered.connect(QApplication.quit)
        menu.addAction(show_action)
        menu.addAction(minimize_action)
        menu.addSeparator()
        menu.addAction(settings_action)
        menu.addAction(health_action)
        menu.addAction(cache_action)
        menu.addAction(crash_bundle_action)
        menu.addSeparator()
        menu.addAction(exit_action)
        return menu

    def show_overlay(self) -> None:
        self._overlay.show()
        self._overlay.raise_()
        self._overlay.activateWindow()

    def hide_overlay(self) -> None:
        self._overlay.hide()

    def open_settings(self) -> None:
        old_lang = self._lang()
        dialog = SettingsDialog(
            config=self._config,
            devices=list_audio_devices(),
            app_sessions=list_active_app_sessions(),
            device_provider=list_audio_devices,
            app_session_provider=list_active_app_sessions,
            export_transcript_callback=self._on_export_transcript,
            import_audio_callback=self._on_import_audio,
            parent=None,
        )
        if dialog.exec() != SettingsDialog.DialogCode.Accepted:
            return

        self._on_settings_applied(dialog.updates)
        if self._lang() != old_lang:
            self.refresh_locale()
        self._tray.showMessage("Voice2Text", self._t("applied"), QSystemTrayIcon.MessageIcon.Information, 1800)

    def open_health_check(self) -> None:
        from .diagnostics_dialogs import HealthCheckDialog

        HealthCheckDialog(self._config, parent=None).exec()

    def open_cache_manager(self) -> None:
        from .diagnostics_dialogs import ModelCacheDialog

        ModelCacheDialog(self._config, parent=None).exec()

    def create_diagnostics_bundle(self) -> None:
        """Manual tray trigger for round 0025 Phase B: build the same redacted diagnostics zip
        `--crash-bundle` writes, off the UI thread, and surface the result via a tray balloon."""
        self._tray.showMessage(
            "Voice2Text", self._t("crash_bundle_writing"), QSystemTrayIcon.MessageIcon.Information, 1800
        )
        config = self._config

        def work() -> None:
            from .crash_bundle import create_crash_bundle

            try:
                path = create_crash_bundle(config, reason="manual (tray)")
            except Exception as exc:
                self._bundle_failed.emit(str(exc))
                return
            self._bundle_created.emit(str(path))

        threading.Thread(target=work, daemon=True).start()

    def _on_bundle_created(self, path: str) -> None:
        self._tray.showMessage(
            "Voice2Text", f"{self._t('crash_bundle_done')}: {path}", QSystemTrayIcon.MessageIcon.Information, 4000
        )

    def _on_bundle_failed(self, message: str) -> None:
        self._tray.showMessage(
            "Voice2Text", f"{self._t('crash_bundle_failed')}: {message}", QSystemTrayIcon.MessageIcon.Warning, 4000
        )

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            if self._overlay.isVisible():
                self.hide_overlay()
            else:
                self.show_overlay()

    @staticmethod
    def _build_icon() -> QIcon:
        pixmap = QPixmap(64, 64)
        pixmap.fill(QColor(0, 0, 0, 0))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(QColor("#0A101A"))
        painter.setPen(QPen(QColor("#8CC8FF"), 3))
        painter.drawRoundedRect(6, 6, 52, 52, 12, 12)
        painter.setPen(QPen(QColor("#E8F2FF"), 3))
        painter.drawLine(16, 23, 48, 23)
        painter.drawLine(16, 32, 40, 32)
        painter.drawLine(16, 41, 46, 41)
        painter.setPen(QColor("#8CC8FF"))
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.drawText(10, 58, "V2T")
        painter.end()
        return QIcon(pixmap)

