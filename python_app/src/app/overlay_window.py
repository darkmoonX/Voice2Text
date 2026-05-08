from __future__ import annotations

from collections import deque
from typing import Deque

from PySide6.QtCore import QPoint, QRect, QSize, Qt, QTimer
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QKeyEvent,
    QMouseEvent,
    QPaintEvent,
    QPainter,
    QPen,
)
from PySide6.QtWidgets import QApplication, QWidget

from .config import RuntimeConfig


class SubtitleOverlayWindow(QWidget):
    _EDGE_LEFT = 1
    _EDGE_RIGHT = 2
    _EDGE_TOP = 4
    _EDGE_BOTTOM = 8

    def __init__(self, config: RuntimeConfig) -> None:
        super().__init__(None)
        self._config = config

        # line tuple: (text, kind), kind in {source, translated, status, error}
        self._lines: Deque[tuple[str, str]] = deque()
        self._line_height = 32
        self._scroll_offset = 0.0
        self._scroll_speed = 2.6

        self._dragging = False
        self._drag_pos = QPoint()

        self._resizing = False
        self._resize_edges = 0
        self._resize_start_geometry = QRect()
        self._resize_start_pos = QPoint()
        self._resize_margin = 8
        self._minimum_size = QSize(480, 160)

        self._init_ui()

        self._tick = QTimer(self)
        self._tick.timeout.connect(self._on_tick)
        self._tick.start(16)

    def push_subtitle(self, source_text: str, translated_text: str = "") -> None:
        source_text = self._normalize_inline_text(source_text)
        translated_text = self._normalize_inline_text(translated_text)

        if not source_text and not translated_text:
            return

        entries: list[tuple[str, str]] = []
        if not self._config.translation_enabled:
            if source_text:
                entries.append((source_text, "source"))
        else:
            style = self._config.bilingual_style.lower().strip()
            if style == "translation-only":
                if translated_text:
                    entries.append((translated_text, "translated"))
                elif source_text:
                    entries.append((source_text, "source"))
            else:
                if source_text:
                    entries.append((source_text, "source"))
                if translated_text:
                    entries.append((translated_text, "translated"))

        self._replace_subtitle_entries(entries)

    def push_status(self, text: str) -> None:
        entries = [(line, "status") for line in self._split_lines(f"[status] {text}")]
        self._append_entries(entries)

    def push_error(self, text: str) -> None:
        entries = [(line, "error") for line in self._split_lines(f"[error] {text}")]
        self._append_entries(entries)

    def apply_runtime_config(self, config: RuntimeConfig) -> None:
        self._config = config

        current_font = self.font()
        current_font.setPointSize(max(10, config.font_size))
        current_font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
        self.setFont(current_font)
        self._line_height = QFontMetrics(self.font()).height() + 8

        self.setGeometry(
            self.x(),
            self.y(),
            max(self._minimum_size.width(), config.overlay_width),
            max(self._minimum_size.height(), config.overlay_height),
        )

        self._trim_history()
        self.update()

    def _init_ui(self) -> None:
        self.setWindowTitle("Voice2Text Overlay")
        self.setGeometry(
            self._config.overlay_x,
            self._config.overlay_y,
            self._config.overlay_width,
            self._config.overlay_height,
        )

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Window
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMouseTracking(True)
        self.setMinimumSize(self._minimum_size)

        font = QFont("Segoe UI", max(10, self._config.font_size))
        font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
        self.setFont(font)
        self._line_height = QFontMetrics(self.font()).height() + 8

        self.push_status("Overlay ready. Drag to move, edge resize enabled, ESC to quit.")

    def _split_lines(self, text: str) -> list[str]:
        return [part.strip() for part in text.splitlines() if part.strip()]

    @staticmethod
    def _normalize_inline_text(text: str) -> str:
        return " ".join(text.split()).strip()

    def _append_entries(self, entries: list[tuple[str, str]]) -> None:
        if not entries:
            return

        fm = QFontMetrics(self.font())
        content_width = max(1, self._content_rect().width())
        added_height = 0
        for item in entries:
            self._lines.append(item)
            added_height += self._measure_entry_height(item[0], fm, content_width)

        self._trim_history()

        self._scroll_offset += float(added_height)
        self._scroll_offset = min(self._scroll_offset, float(self.height()))
        self.update()

    def _replace_subtitle_entries(self, entries: list[tuple[str, str]]) -> None:
        kept: Deque[tuple[str, str]] = deque(
            (text, kind) for text, kind in self._lines if kind in {"status", "error"}
        )
        kept.extend(entries)

        self._lines = kept
        self._trim_history()
        self._scroll_offset = 0.0
        self.update()

    def _trim_history(self) -> None:
        fm = QFontMetrics(self.font())
        content = self._content_rect()
        width = max(1, content.width())
        total_height = sum(self._measure_entry_height(text, fm, width) for text, _ in self._lines)
        history_limit_height = max(content.height() * 20, 3200)

        while self._lines and total_height > history_limit_height:
            old_text, _ = self._lines[0]
            total_height -= self._measure_entry_height(old_text, fm, width)
            self._lines.popleft()

    def _visible_line_capacity(self) -> int:
        content = self._content_rect()
        return max(1, content.height() // max(1, self._line_height))

    def _content_rect(self) -> QRect:
        return self.rect().adjusted(20, 16, -20, -18)

    def _on_tick(self) -> None:
        if self._scroll_offset > 0.0:
            self._scroll_offset = max(0.0, self._scroll_offset - self._scroll_speed)
            self.update()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            QApplication.instance().quit()
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        edges = self._hit_test_edges(event.pos())
        if edges:
            self._resizing = True
            self._resize_edges = edges
            self._resize_start_geometry = self.geometry()
            self._resize_start_pos = event.globalPosition().toPoint()
            event.accept()
            return

        self._dragging = True
        self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._resizing and (event.buttons() & Qt.MouseButton.LeftButton):
            self._perform_resize(event.globalPosition().toPoint())
            event.accept()
            return

        if self._dragging and (event.buttons() & Qt.MouseButton.LeftButton):
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
            return

        edges = self._hit_test_edges(event.pos())
        self._update_cursor(edges)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self._resizing = False
            self._resize_edges = 0
            self._config.overlay_width = self.width()
            self._config.overlay_height = self.height()
            self._update_cursor(self._hit_test_edges(event.pos()))
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        if not self._dragging and not self._resizing:
            self.unsetCursor()
        super().leaveEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        container = self.rect().adjusted(0, 0, -1, -1)
        bg = self._parse_color(self._config.background_color, QColor(10, 16, 26, 255))
        bg.setAlpha(int(255 * self._config.overlay_opacity))

        painter.setBrush(bg)
        painter.setPen(QPen(QColor(255, 255, 255, 90), 1))
        painter.drawRoundedRect(container, 16, 16)

        painter.setFont(self.font())
        fm = QFontMetrics(self.font())

        content = self._content_rect()
        y = float(content.bottom()) + self._scroll_offset
        text_flags = int(
            Qt.AlignmentFlag.AlignLeft
            | Qt.AlignmentFlag.AlignVCenter
            | Qt.TextFlag.TextWordWrap
        )

        for line, kind in reversed(self._lines):
            block_height = self._measure_entry_height(line, fm, content.width())
            line_rect = QRect(
                content.left(),
                int(y - block_height),
                content.width(),
                block_height,
            )

            painter.setPen(self._line_color(kind))
            painter.drawText(
                line_rect,
                text_flags,
                line,
            )

            y -= block_height
            if y < content.top() - block_height:
                break

    def _measure_entry_height(self, text: str, fm: QFontMetrics, width: int) -> int:
        if width <= 0:
            return self._line_height

        flags = int(
            Qt.AlignmentFlag.AlignLeft
            | Qt.AlignmentFlag.AlignVCenter
            | Qt.TextFlag.TextWordWrap
        )
        rect = fm.boundingRect(
            QRect(0, 0, width, 10000),
            flags,
            text,
        )
        return max(self._line_height, rect.height() + 6)

    def _line_color(self, kind: str) -> QColor:
        if kind == "translated":
            return self._parse_color(
                self._config.translated_text_color,
                QColor(255, 217, 138, 255),
            )
        if kind == "status":
            return self._parse_color(self._config.status_color, QColor(120, 215, 255, 255))
        if kind == "error":
            return self._parse_color(self._config.error_color, QColor(255, 120, 120, 255))

        return self._parse_color(self._config.source_text_color, QColor(240, 242, 245, 255))

    def _hit_test_edges(self, pos: QPoint) -> int:
        rect = self.rect()
        x = pos.x()
        y = pos.y()

        edges = 0
        if x <= self._resize_margin:
            edges |= self._EDGE_LEFT
        elif x >= rect.width() - self._resize_margin:
            edges |= self._EDGE_RIGHT

        if y <= self._resize_margin:
            edges |= self._EDGE_TOP
        elif y >= rect.height() - self._resize_margin:
            edges |= self._EDGE_BOTTOM

        return edges

    def _update_cursor(self, edges: int) -> None:
        if edges in (self._EDGE_LEFT | self._EDGE_TOP, self._EDGE_RIGHT | self._EDGE_BOTTOM):
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        elif edges in (self._EDGE_RIGHT | self._EDGE_TOP, self._EDGE_LEFT | self._EDGE_BOTTOM):
            self.setCursor(Qt.CursorShape.SizeBDiagCursor)
        elif edges in (self._EDGE_LEFT, self._EDGE_RIGHT):
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        elif edges in (self._EDGE_TOP, self._EDGE_BOTTOM):
            self.setCursor(Qt.CursorShape.SizeVerCursor)
        else:
            self.unsetCursor()

    def _perform_resize(self, global_pos: QPoint) -> None:
        delta = global_pos - self._resize_start_pos
        rect = QRect(self._resize_start_geometry)

        if self._resize_edges & self._EDGE_LEFT:
            rect.setLeft(rect.left() + delta.x())
        if self._resize_edges & self._EDGE_RIGHT:
            rect.setRight(rect.right() + delta.x())
        if self._resize_edges & self._EDGE_TOP:
            rect.setTop(rect.top() + delta.y())
        if self._resize_edges & self._EDGE_BOTTOM:
            rect.setBottom(rect.bottom() + delta.y())

        if rect.width() < self._minimum_size.width():
            if self._resize_edges & self._EDGE_LEFT:
                rect.setLeft(rect.right() - self._minimum_size.width() + 1)
            else:
                rect.setRight(rect.left() + self._minimum_size.width() - 1)

        if rect.height() < self._minimum_size.height():
            if self._resize_edges & self._EDGE_TOP:
                rect.setTop(rect.bottom() - self._minimum_size.height() + 1)
            else:
                rect.setBottom(rect.top() + self._minimum_size.height() - 1)

        self.setGeometry(rect.normalized())
        self._trim_history()

    @staticmethod
    def _parse_color(raw: str, fallback: QColor) -> QColor:
        color = QColor(raw)
        if not color.isValid():
            return fallback
        return color