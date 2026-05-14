#pragma once

#include <QColor>
#include <QList>
#include <QPoint>
#include <QRect>
#include <QSize>
#include <QString>
#include <QTimer>
#include <QToolButton>
#include <QWidget>

#include <deque>

class QEvent;
class QWheelEvent;

class SubtitleOverlayWindow : public QWidget {
    Q_OBJECT

public:
    explicit SubtitleOverlayWindow(QWidget *parent = nullptr);

    void setTranslationEnabled(bool enabled);
    void setTranslationStyle(const QString &style);
    void setTextColors(const QColor &sourceColor, const QColor &translatedColor);
    void setBackgroundColor(const QColor &background);
    bool translationEnabled() const { return translationEnabled_; }
    QString translationStyle() const { return translationStyle_; }
    QColor sourceTextColor() const { return sourceColor_; }
    QColor translatedTextColor() const { return translatedColor_; }
    QColor backgroundColor() const { return backgroundColor_; }
    float overlayOpacity() const;
    int overlayFontSize() const;
    void applyVisualSettings(bool translationEnabled,
                             const QString &translationStyle,
                             int fontSize,
                             float opacity,
                             const QColor &sourceColor,
                             const QColor &translatedColor,
                             const QColor &backgroundColor);

public slots:
    void pushSubtitle(const QString &sourceText, const QString &translatedText = QString());
    void pushStatus(const QString &text);
    void pushError(const QString &text);

protected:
    void paintEvent(QPaintEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void leaveEvent(QEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;

private slots:
    void onTick();

private:
    struct LineEntry {
        QString text;
        enum class Kind { Source, Translated, Status, Error } kind;
    };

    enum EdgeMask {
        EdgeNone = 0,
        EdgeLeft = 1,
        EdgeRight = 2,
        EdgeTop = 4,
        EdgeBottom = 8,
    };

    void appendLines(const QString &text, LineEntry::Kind kind);
    void replaceSubtitleEntries(const QList<LineEntry> &entries);
    static QString normalizeInlineText(const QString &text);
    void trimHistory();
    int visibleLineCapacity() const;
    float maxHistoryScrollOffset() const;
    bool isAtBottom() const;
    void scrollToBottom();
    void updateJumpBottomButtonGeometry();
    void updateJumpBottomButtonVisibility();
    int measureLineHeight(const QString &text, const QFontMetrics &fm, int width) const;
    QColor colorForKind(LineEntry::Kind kind) const;
    int hitTestEdges(const QPoint &pos) const;
    void updateCursor(int edges);
    void performResize(const QPoint &globalPos);

    std::deque<LineEntry> lines_;
    int lineHeight_{32};
    float scrollOffset_{0.0F};
    float historyScrollOffset_{0.0F};
    float scrollSpeed_{2.8F};
    QToolButton *jumpBottomButton_{nullptr};

    bool translationEnabled_{false};
    QString translationStyle_{"stacked"};

    QColor sourceColor_{240, 242, 245, 255};
    QColor translatedColor_{255, 217, 138, 255};
    QColor statusColor_{120, 215, 255, 255};
    QColor errorColor_{255, 125, 125, 255};
    QColor backgroundColor_{10, 16, 26, 184};

    bool dragging_{false};
    QPoint dragOrigin_;

    bool resizing_{false};
    int resizeEdges_{EdgeNone};
    int resizeMargin_{8};
    QRect resizeStartGeometry_;
    QPoint resizeStartPos_;
    QSize minimumSize_{480, 160};

    QTimer tickTimer_;
};
