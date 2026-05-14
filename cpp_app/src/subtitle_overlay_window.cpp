#include "subtitle_overlay_window.h"

#include <algorithm>

#include <QApplication>
#include <QChar>
#include <QEvent>
#include <QFont>
#include <QFontMetrics>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QPaintEvent>
#include <QPainter>
#include <QPen>
#include <QStyle>
#include <QToolButton>
#include <QWheelEvent>

SubtitleOverlayWindow::SubtitleOverlayWindow(QWidget *parent) : QWidget(parent) {
    setWindowTitle("Voice2Text Overlay");
    setWindowFlags(Qt::FramelessWindowHint | Qt::WindowStaysOnTopHint | Qt::Window);
    setAttribute(Qt::WA_TranslucentBackground, true);
    setMouseTracking(true);
    setMinimumSize(minimumSize_);

    setGeometry(40, 700, 1200, 320);

    QFont font("Segoe UI", 17);
    setFont(font);
    lineHeight_ = QFontMetrics(this->font()).height() + 8;

    connect(&tickTimer_, &QTimer::timeout, this, &SubtitleOverlayWindow::onTick);
    tickTimer_.start(16);
    jumpBottomButton_ = new QToolButton(this);
    jumpBottomButton_->setAutoRaise(true);
    jumpBottomButton_->setIcon(style()->standardIcon(QStyle::SP_ArrowDown));
    jumpBottomButton_->setToolTip("Back to latest");
    connect(jumpBottomButton_, &QToolButton::clicked, this, &SubtitleOverlayWindow::scrollToBottom);
    updateJumpBottomButtonGeometry();
    updateJumpBottomButtonVisibility();

    pushStatus("Overlay ready. ESC to quit. Drag window to move.");
}

void SubtitleOverlayWindow::setTranslationEnabled(bool enabled) {
    translationEnabled_ = enabled;
}

void SubtitleOverlayWindow::setTranslationStyle(const QString &style) {
    if (style == "translation-only") {
        translationStyle_ = "translation-only";
    } else {
        translationStyle_ = "stacked";
    }
}

void SubtitleOverlayWindow::setTextColors(const QColor &sourceColor, const QColor &translatedColor) {
    sourceColor_ = sourceColor;
    translatedColor_ = translatedColor;
}

void SubtitleOverlayWindow::setBackgroundColor(const QColor &background) {
    backgroundColor_ = background;
}

float SubtitleOverlayWindow::overlayOpacity() const {
    return std::clamp(static_cast<float>(backgroundColor_.alpha()) / 255.0F, 0.2F, 1.0F);
}

int SubtitleOverlayWindow::overlayFontSize() const {
    return std::max(10, font().pointSize());
}

void SubtitleOverlayWindow::applyVisualSettings(bool translationEnabled,
                                                const QString &translationStyle,
                                                int fontSize,
                                                float opacity,
                                                const QColor &sourceColor,
                                                const QColor &translatedColor,
                                                const QColor &backgroundColor) {
    setTranslationEnabled(translationEnabled);
    setTranslationStyle(translationStyle);
    setTextColors(sourceColor, translatedColor);

    QColor bg = backgroundColor;
    const float clampedOpacity = std::clamp(opacity, 0.2F, 1.0F);
    bg.setAlpha(static_cast<int>(clampedOpacity * 255.0F));
    setBackgroundColor(bg);

    QFont current = font();
    current.setPointSize(std::max(10, fontSize));
    setFont(current);
    lineHeight_ = QFontMetrics(this->font()).height() + 8;

    trimHistory();
    update();
}

void SubtitleOverlayWindow::pushSubtitle(const QString &sourceText, const QString &translatedText) {
    const QString source = normalizeInlineText(sourceText);
    const QString translated = normalizeInlineText(translatedText);

    if (source.isEmpty() && translated.isEmpty()) {
        return;
    }

    QList<LineEntry> entries;

    if (translationEnabled_) {
        if (translationStyle_ == "translation-only") {
            if (!translated.isEmpty()) {
                entries.push_back(LineEntry{translated, LineEntry::Kind::Translated});
            } else {
                entries.push_back(LineEntry{source, LineEntry::Kind::Source});
            }
            replaceSubtitleEntries(entries);
            return;
        }

        if (!source.isEmpty()) {
            entries.push_back(LineEntry{source, LineEntry::Kind::Source});
        }
        if (!translated.isEmpty()) {
            entries.push_back(LineEntry{translated, LineEntry::Kind::Translated});
        }
        replaceSubtitleEntries(entries);
        return;
    }

    if (!source.isEmpty()) {
        entries.push_back(LineEntry{source, LineEntry::Kind::Source});
    } else {
        entries.push_back(LineEntry{translated, LineEntry::Kind::Source});
    }

    replaceSubtitleEntries(entries);
}

void SubtitleOverlayWindow::pushStatus(const QString &text) {
    appendLines("[status] " + text, LineEntry::Kind::Status);
}

void SubtitleOverlayWindow::pushError(const QString &text) {
    appendLines("[error] " + text, LineEntry::Kind::Error);
}

void SubtitleOverlayWindow::appendLines(const QString &text, LineEntry::Kind kind) {
    const QStringList parts = text.split('\n', Qt::SkipEmptyParts);
    int added = 0;
    int addedHeight = 0;
    const QFontMetrics fm(font());
    const int contentWidth = std::max(1, rect().adjusted(20, 44, -20, -18).width());

    for (const QString &raw : parts) {
        const QString line = raw.trimmed();
        if (line.isEmpty()) {
            continue;
        }

        lines_.push_back(LineEntry{line, kind});
        ++added;
        addedHeight += measureLineHeight(line, fm, contentWidth);
    }

    if (added > 0) {
        const bool wasAtBottom = isAtBottom();
        trimHistory();
        if (wasAtBottom) {
            scrollOffset_ += static_cast<float>(addedHeight);
            if (scrollOffset_ > static_cast<float>(height())) {
                scrollOffset_ = static_cast<float>(height());
            }
            historyScrollOffset_ = 0.0F;
        } else {
            historyScrollOffset_ = std::min(historyScrollOffset_ + static_cast<float>(addedHeight),
                                            maxHistoryScrollOffset());
        }
        updateJumpBottomButtonVisibility();
        update();
    }
}

void SubtitleOverlayWindow::trimHistory() {
    const QRect content = rect().adjusted(20, 44, -20, -18);
    const QFontMetrics fm(font());
    const int width = std::max(1, content.width());

    int totalHeight = 0;
    for (const auto &entry : lines_) {
        totalHeight += measureLineHeight(entry.text, fm, width);
    }

    const int keepHeight = std::max(content.height() * 20, 3200);
    while (!lines_.empty() && totalHeight > keepHeight) {
        totalHeight -= measureLineHeight(lines_.front().text, fm, width);
        lines_.pop_front();
    }
}

int SubtitleOverlayWindow::visibleLineCapacity() const {
    const QRect content = rect().adjusted(20, 44, -20, -18);
    if (lineHeight_ <= 0) {
        return 1;
    }
    return std::max(1, content.height() / lineHeight_);
}

int SubtitleOverlayWindow::measureLineHeight(const QString &text,
                                             const QFontMetrics &fm,
                                             int width) const {
    if (width <= 0) {
        return lineHeight_;
    }

    const QRect bounds = fm.boundingRect(
        QRect(0, 0, width, 10000),
        Qt::AlignLeft | Qt::AlignVCenter | Qt::TextWordWrap,
        text);
    return std::max(lineHeight_, bounds.height() + 6);
}

QColor SubtitleOverlayWindow::colorForKind(LineEntry::Kind kind) const {
    switch (kind) {
        case LineEntry::Kind::Source:
            return sourceColor_;
        case LineEntry::Kind::Translated:
            return translatedColor_;
        case LineEntry::Kind::Status:
            return statusColor_;
        case LineEntry::Kind::Error:
            return errorColor_;
    }
    return sourceColor_;
}

void SubtitleOverlayWindow::onTick() {
    if (scrollOffset_ > 0.0F) {
        scrollOffset_ -= scrollSpeed_;
        if (scrollOffset_ < 0.0F) {
            scrollOffset_ = 0.0F;
        }
        update();
    }
}

void SubtitleOverlayWindow::keyPressEvent(QKeyEvent *event) {
    if (event->key() == Qt::Key_Escape) {
        QApplication::quit();
        return;
    }
    QWidget::keyPressEvent(event);
}

void SubtitleOverlayWindow::mousePressEvent(QMouseEvent *event) {
    if (event->button() == Qt::LeftButton) {
        const int edges = hitTestEdges(event->pos());
        if (edges != EdgeNone) {
            resizing_ = true;
            resizeEdges_ = edges;
            resizeStartGeometry_ = geometry();
            resizeStartPos_ = event->globalPosition().toPoint();
            event->accept();
            return;
        }

        dragging_ = true;
        dragOrigin_ = event->globalPosition().toPoint() - frameGeometry().topLeft();
        event->accept();
        return;
    }
    QWidget::mousePressEvent(event);
}

void SubtitleOverlayWindow::mouseMoveEvent(QMouseEvent *event) {
    if (resizing_ && (event->buttons() & Qt::LeftButton)) {
        performResize(event->globalPosition().toPoint());
        event->accept();
        return;
    }

    if (dragging_ && (event->buttons() & Qt::LeftButton)) {
        move(event->globalPosition().toPoint() - dragOrigin_);
        event->accept();
        return;
    }

    updateCursor(hitTestEdges(event->pos()));
    QWidget::mouseMoveEvent(event);
}

void SubtitleOverlayWindow::mouseReleaseEvent(QMouseEvent *event) {
    if (event->button() == Qt::LeftButton) {
        dragging_ = false;
        resizing_ = false;
        resizeEdges_ = EdgeNone;
        updateCursor(hitTestEdges(event->pos()));
    }
    QWidget::mouseReleaseEvent(event);
}

void SubtitleOverlayWindow::leaveEvent(QEvent *event) {
    if (!dragging_ && !resizing_) {
        unsetCursor();
    }
    QWidget::leaveEvent(event);
}

void SubtitleOverlayWindow::wheelEvent(QWheelEvent *event) {
    const int delta = event->angleDelta().y();
    if (delta == 0) {
        QWidget::wheelEvent(event);
        return;
    }
    const float step = std::max(12.0F, static_cast<float>(lineHeight_) * 0.9F);
    historyScrollOffset_ += delta > 0 ? step : -step;
    historyScrollOffset_ = std::clamp(historyScrollOffset_, 0.0F, maxHistoryScrollOffset());
    updateJumpBottomButtonVisibility();
    update();
    event->accept();
}

void SubtitleOverlayWindow::paintEvent(QPaintEvent *event) {
    Q_UNUSED(event);

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);

    const QRect container = rect().adjusted(0, 0, -1, -1);

    painter.setPen(QPen(QColor(255, 255, 255, 90), 1));
    painter.setBrush(backgroundColor_);
    painter.drawRoundedRect(container, 16, 16);

    // painter.setPen(QColor(180, 210, 255, 220));
    // painter.setFont(QFont("Segoe UI", 10));
    // painter.drawText(container.adjusted(14, 10, -14, -container.height() + 34),
    //                  Qt::AlignLeft | Qt::AlignVCenter,
    //                  "Voice2Text Overlay | ESC: exit | Drag: move");

    painter.setFont(font());
    const QFontMetrics fm(font());
    const QRect content = rect().adjusted(20, 44, -20, -18);

    float y = static_cast<float>(content.bottom()) + scrollOffset_ + historyScrollOffset_;
    for (auto it = lines_.rbegin(); it != lines_.rend(); ++it) {
        const int blockHeight = measureLineHeight(it->text, fm, content.width());
        const QRect lineRect(content.left(),
                             static_cast<int>(y - blockHeight),
                             content.width(),
                             blockHeight);
        painter.setPen(colorForKind(it->kind));
        painter.drawText(lineRect,
                         Qt::AlignLeft | Qt::AlignVCenter | Qt::TextWordWrap,
                         it->text);

        y -= static_cast<float>(blockHeight);
        if (y < static_cast<float>(content.top() - blockHeight)) {
            break;
        }
    }
}

int SubtitleOverlayWindow::hitTestEdges(const QPoint &pos) const {
    int edges = EdgeNone;
    if (pos.x() <= resizeMargin_) {
        edges |= EdgeLeft;
    } else if (pos.x() >= width() - resizeMargin_) {
        edges |= EdgeRight;
    }

    if (pos.y() <= resizeMargin_) {
        edges |= EdgeTop;
    } else if (pos.y() >= height() - resizeMargin_) {
        edges |= EdgeBottom;
    }

    return edges;
}

void SubtitleOverlayWindow::updateCursor(int edges) {
    if (edges == (EdgeLeft | EdgeTop) || edges == (EdgeRight | EdgeBottom)) {
        setCursor(Qt::SizeFDiagCursor);
        return;
    }

    if (edges == (EdgeRight | EdgeTop) || edges == (EdgeLeft | EdgeBottom)) {
        setCursor(Qt::SizeBDiagCursor);
        return;
    }

    if (edges == EdgeLeft || edges == EdgeRight) {
        setCursor(Qt::SizeHorCursor);
        return;
    }

    if (edges == EdgeTop || edges == EdgeBottom) {
        setCursor(Qt::SizeVerCursor);
        return;
    }

    unsetCursor();
}

void SubtitleOverlayWindow::performResize(const QPoint &globalPos) {
    const QPoint delta = globalPos - resizeStartPos_;
    QRect rect = resizeStartGeometry_;

    if (resizeEdges_ & EdgeLeft) {
        rect.setLeft(rect.left() + delta.x());
    }
    if (resizeEdges_ & EdgeRight) {
        rect.setRight(rect.right() + delta.x());
    }
    if (resizeEdges_ & EdgeTop) {
        rect.setTop(rect.top() + delta.y());
    }
    if (resizeEdges_ & EdgeBottom) {
        rect.setBottom(rect.bottom() + delta.y());
    }

    if (rect.width() < minimumSize_.width()) {
        if (resizeEdges_ & EdgeLeft) {
            rect.setLeft(rect.right() - minimumSize_.width() + 1);
        } else {
            rect.setRight(rect.left() + minimumSize_.width() - 1);
        }
    }

    if (rect.height() < minimumSize_.height()) {
        if (resizeEdges_ & EdgeTop) {
            rect.setTop(rect.bottom() - minimumSize_.height() + 1);
        } else {
            rect.setBottom(rect.top() + minimumSize_.height() - 1);
        }
    }

    setGeometry(rect.normalized());
    updateJumpBottomButtonGeometry();
    updateJumpBottomButtonVisibility();
}

void SubtitleOverlayWindow::replaceSubtitleEntries(const QList<LineEntry> &entries) {
    const bool wasAtBottom = isAtBottom();
    std::deque<LineEntry> kept;
    for (const auto &entry : lines_) {
        if (entry.kind == LineEntry::Kind::Status || entry.kind == LineEntry::Kind::Error) {
            kept.push_back(entry);
        }
    }

    for (const auto &entry : entries) {
        if (entry.text.trimmed().isEmpty()) {
            continue;
        }
        kept.push_back(entry);
    }

    lines_.swap(kept);
    trimHistory();
    scrollOffset_ = 0.0F;
    if (wasAtBottom) {
        historyScrollOffset_ = 0.0F;
    } else {
        historyScrollOffset_ = std::min(historyScrollOffset_, maxHistoryScrollOffset());
    }
    updateJumpBottomButtonVisibility();
    update();
}

float SubtitleOverlayWindow::maxHistoryScrollOffset() const {
    const QRect content = rect().adjusted(20, 44, -20, -18);
    const QFontMetrics fm(font());
    const int width = std::max(1, content.width());
    int totalHeight = 0;
    for (const auto &entry : lines_) {
        totalHeight += measureLineHeight(entry.text, fm, width);
    }
    return std::max(0.0F, static_cast<float>(totalHeight - content.height()));
}

bool SubtitleOverlayWindow::isAtBottom() const {
    return historyScrollOffset_ <= 1.0F;
}

void SubtitleOverlayWindow::scrollToBottom() {
    historyScrollOffset_ = 0.0F;
    updateJumpBottomButtonVisibility();
    update();
}

void SubtitleOverlayWindow::updateJumpBottomButtonGeometry() {
    if (jumpBottomButton_ == nullptr) {
        return;
    }
    constexpr int size = 26;
    constexpr int margin = 14;
    const int x = std::max(0, width() - size - margin);
    const int y = std::max(0, height() - size - margin);
    jumpBottomButton_->setGeometry(x, y, size, size);
}

void SubtitleOverlayWindow::updateJumpBottomButtonVisibility() {
    if (jumpBottomButton_ == nullptr) {
        return;
    }
    jumpBottomButton_->setVisible(!isAtBottom());
}

QString SubtitleOverlayWindow::normalizeInlineText(const QString &text) {
    QString out;
    out.reserve(text.size());

    bool inSpace = false;
    for (const QChar ch : text) {
        if (ch.isSpace()) {
            if (!inSpace) {
                out.push_back(' ');
                inSpace = true;
            }
            continue;
        }

        out.push_back(ch);
        inSpace = false;
    }

    return out.trimmed();
}
