#pragma once

#include <QColor>
#include <QDialog>
#include <QString>
#include <QStringList>

#include "audio/discovery.h"
#include "runtime/runtime_settings.h"

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QPushButton;
class QSlider;
class QSpinBox;

class SettingsDialog : public QDialog {
    Q_OBJECT

public:
    explicit SettingsDialog(const RuntimeSettings &initial,
                            const QList<SourceDeviceEntry> &loopbackDevices,
                            const QList<SourceDeviceEntry> &appSessions,
                            const QStringList &modelCandidates,
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
    QString uiLang_{"zh"};
    void refreshSourceSummary();
    static void setComboData(QComboBox *combo, const QString &value);
    void setButtonColor(QPushButton *button, const QColor &color) const;
    QColor buttonColor(const QPushButton *button) const;

    QList<SourceDeviceEntry> appSessions_;
    QList<SourceDeviceEntry> loopbackDevices_;
    QString selectedLoopbackDeviceId_;
    QStringList selectedAppNames_;

    QComboBox *sourceModeCombo_{nullptr};
    QComboBox *uiLanguageCombo_{nullptr};
    QPushButton *sourceSelectButton_{nullptr};
    QLabel *sourceSummaryLabel_{nullptr};
    QDoubleSpinBox *segmentSpin_{nullptr};
    QDoubleSpinBox *hopSpin_{nullptr};
    QComboBox *overlapMergeMethodCombo_{nullptr};
    QComboBox *sourceLanguageCombo_{nullptr};
    QComboBox *modelCombo_{nullptr};
    QCheckBox *vadEnabledCheck_{nullptr};
    QCheckBox *vadAdaptiveCheck_{nullptr};
    QDoubleSpinBox *vadThresholdSpin_{nullptr};

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
