from __future__ import annotations

from PySide6.QtCore import QObject
from PySide6.QtGui import QAction, QColor, QFont, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from .audio_capture import list_active_app_sessions, list_audio_devices
from .config import RuntimeConfig
from .settings_dialog import SettingsDialog


class Voice2TextTrayController(QObject):
    def __init__(
        self,
        app: QApplication,
        overlay,
        config: RuntimeConfig,
        on_settings_applied,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._app = app
        self._overlay = overlay
        self._config = config
        self._on_settings_applied = on_settings_applied

        self._tray = QSystemTrayIcon(self)
        self._icon = self._build_icon()

        self._app.setWindowIcon(self._icon)
        self._overlay.setWindowIcon(self._icon)
        self._tray.setIcon(self._icon)
        self._tray.setToolTip("Voice2Text")
        self._tray.setContextMenu(self._build_menu())
        self._tray.activated.connect(self._on_tray_activated)
        self._tray.show()

    def _build_menu(self) -> QMenu:
        menu = QMenu()

        show_action = QAction("顯示", menu)
        minimize_action = QAction("縮小", menu)
        settings_action = QAction("設定", menu)
        exit_action = QAction("離開", menu)

        show_action.triggered.connect(self.show_overlay)
        minimize_action.triggered.connect(self.hide_overlay)
        settings_action.triggered.connect(self.open_settings)
        exit_action.triggered.connect(QApplication.quit)

        menu.addAction(show_action)
        menu.addAction(minimize_action)
        menu.addSeparator()
        menu.addAction(settings_action)
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
        dialog = SettingsDialog(
            config=self._config,
            devices=list_audio_devices(),
            app_sessions=list_active_app_sessions(),
            parent=self._overlay,
        )

        if dialog.exec() != SettingsDialog.DialogCode.Accepted:
            return

        self._on_settings_applied(dialog.updates)
        self._tray.showMessage(
            "Voice2Text",
            "設定已套用。",
            QSystemTrayIcon.MessageIcon.Information,
            1800,
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