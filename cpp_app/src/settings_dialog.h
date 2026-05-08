#pragma once

#include <QColor>
#include <QDialog>
#include <QString>
#include <QStringList>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QPushButton;
class QSlider;
class QSpinBox;

struct RuntimeSettings {
    QString sourceMode{"loopback"};
    QString loopbackDeviceId;
    QStringList sourceApps;
    QString sourceLanguage{"auto"};
    float segmentSeconds{6.0F};
    float hopSeconds{1.5F};
    QString overlapMergeMethod{"replace-window"};

    bool translationEnabled{false};
    QString fromLang{"auto"};
    QString toLang{"zh"};
    QString translationStyle{"stacked"};

    int fontSize{18};
    float opacity{0.8F};
    QColor sourceColor{QStringLiteral("#F0F2F5")};
    QColor translatedColor{QStringLiteral("#FFD98A")};
    QColor backgroundColor{QStringLiteral("#0A101A")};
};

struct SourceDeviceEntry {
    QString id;
    QString label;
};

class SettingsDialog : public QDialog {
    Q_OBJECT

public:
    explicit SettingsDialog(const RuntimeSettings &initial,
                            const QList<SourceDeviceEntry> &loopbackDevices,
                            const QList<SourceDeviceEntry> &appSessions,
                            QWidget *parent = nullptr);

    RuntimeSettings settings() const;

private slots:
    void onModeChanged();
    void openSourceSelection();
    void pickSourceColor();
    void pickTranslatedColor();
    void pickBackgroundColor();
    void onTranslationToggle();

private:
    void refreshSourceSummary();
    static void setComboData(QComboBox *combo, const QString &value);
    void setButtonColor(QPushButton *button, const QColor &color) const;
    QColor buttonColor(const QPushButton *button) const;

    QList<SourceDeviceEntry> appSessions_;
    QList<SourceDeviceEntry> loopbackDevices_;
    QString selectedLoopbackDeviceId_;
    QStringList selectedAppNames_;

    QComboBox *sourceModeCombo_{nullptr};
    QPushButton *sourceSelectButton_{nullptr};
    QLabel *sourceSummaryLabel_{nullptr};
    QDoubleSpinBox *segmentSpin_{nullptr};
    QDoubleSpinBox *hopSpin_{nullptr};
    QComboBox *overlapMergeMethodCombo_{nullptr};
    QComboBox *sourceLanguageCombo_{nullptr};

    QCheckBox *translationEnabledCheck_{nullptr};
    QComboBox *translationStyleCombo_{nullptr};
    QComboBox *translationLanguageCombo_{nullptr};
    QSpinBox *fontSizeSpin_{nullptr};
    QSlider *opacitySlider_{nullptr};
    QLabel *opacityValueLabel_{nullptr};
    QPushButton *sourceColorButton_{nullptr};
    QPushButton *translatedColorButton_{nullptr};
    QPushButton *backgroundColorButton_{nullptr};
};
