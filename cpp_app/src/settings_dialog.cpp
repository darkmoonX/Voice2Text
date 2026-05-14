#include "settings_dialog.h"

#include <algorithm>
#include <cmath>

#include <QCheckBox>
#include <QColorDialog>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QScrollArea>
#include <QSlider>
#include <QSpinBox>
#include <QVBoxLayout>
#include <QWidget>

#include "settings/i18n.h"
#include "settings/mapping.h"

namespace {

class SourceSelectionDialog : public QDialog {
public:
    SourceSelectionDialog(const QString &title,
                          const QList<SourceDeviceEntry> &entries,
                          const QStringList &selected,
                          const QString &uiLang,
                          QWidget *parent)
        : QDialog(parent) {
        setWindowTitle(title);
        setMinimumSize(460, 420);

        auto *root = new QVBoxLayout(this);

        selectAll_ = new QCheckBox(settings_i18n::selectAll(uiLang), this);
        root->addWidget(selectAll_);

        auto *body = new QWidget(this);
        auto *bodyLayout = new QVBoxLayout(body);
        bodyLayout->setContentsMargins(0, 0, 0, 0);
        bodyLayout->setSpacing(4);

        if (entries.isEmpty()) {
            bodyLayout->addWidget(new QLabel(settings_i18n::noSources(uiLang), body));
        } else {
            for (const SourceDeviceEntry &entry : entries) {
                auto *cb = new QCheckBox(entry.label, body);
                cb->setProperty("value", entry.id);
                cb->setChecked(selected.contains(entry.id));
                checks_.push_back(cb);
                bodyLayout->addWidget(cb);
                connect(cb, &QCheckBox::checkStateChanged, this, [this](int) {
                    refreshSelectAllState();
                });
            }
        }

        bodyLayout->addStretch(1);

        auto *scroll = new QScrollArea(this);
        scroll->setWidgetResizable(true);
        scroll->setWidget(body);
        root->addWidget(scroll, 1);

        auto *buttons =
            new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
        connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
        connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
        root->addWidget(buttons);

        connect(selectAll_, &QCheckBox::checkStateChanged, this, [this](int state) {
            const bool checked = state == Qt::Checked;
            for (QCheckBox *cb : checks_) {
                cb->blockSignals(true);
                cb->setChecked(checked);
                cb->blockSignals(false);
            }
            refreshSelectAllState();
        });

        refreshSelectAllState();
    }

    QStringList selectedValues() const {
        QStringList out;
        for (const QCheckBox *cb : checks_) {
            if (cb->isChecked()) {
                out.push_back(cb->property("value").toString());
            }
        }
        return out;
    }

private:
    void refreshSelectAllState() {
        if (checks_.empty()) {
            selectAll_->setEnabled(false);
            selectAll_->setChecked(false);
            return;
        }

        int checkedCount = 0;
        for (const QCheckBox *cb : checks_) {
            if (cb->isChecked()) {
                ++checkedCount;
            }
        }

        selectAll_->blockSignals(true);
        if (checkedCount == 0) {
            selectAll_->setTristate(false);
            selectAll_->setCheckState(Qt::Unchecked);
        } else if (checkedCount == static_cast<int>(checks_.size())) {
            selectAll_->setTristate(false);
            selectAll_->setCheckState(Qt::Checked);
        } else {
            selectAll_->setTristate(true);
            selectAll_->setCheckState(Qt::PartiallyChecked);
        }
        selectAll_->blockSignals(false);
    }

    QCheckBox *selectAll_{nullptr};
    QList<QCheckBox *> checks_;
};

}  // namespace

SettingsDialog::SettingsDialog(const RuntimeSettings &initial,
                               const QList<SourceDeviceEntry> &loopbackDevices,
                               const QList<SourceDeviceEntry> &appSessions,
                               const QStringList &modelCandidates,
                               QWidget *parent)
    : QDialog(parent) {
    uiLang_ = settings_i18n::normalizeUiLanguage(initial.uiLanguage);
    loopbackDevices_ = loopbackDevices;
    appSessions_ = appSessions;
    selectedLoopbackDeviceId_ = initial.loopbackDeviceId;
    selectedAppNames_ = initial.sourceApps;

    setWindowTitle(settings_i18n::settingsTitle(uiLang_));
    setMinimumWidth(580);

    auto *root = new QVBoxLayout(this);
    auto *form = new QFormLayout();

    sourceModeCombo_ = new QComboBox(this);
    sourceModeCombo_->addItem(settings_i18n::modeLoopbackLabel(uiLang_), "loopback");
    sourceModeCombo_->addItem(settings_i18n::modeMicrophoneLabel(uiLang_), "microphone");
    sourceModeCombo_->addItem(settings_i18n::modeAppLabel(uiLang_), "app");
    setComboData(sourceModeCombo_, initial.sourceMode);
    uiLanguageCombo_ = new QComboBox(this);
    uiLanguageCombo_->addItem("繁體中文", "zh");
    uiLanguageCombo_->addItem("English", "en");
    setComboData(uiLanguageCombo_, uiLang_);

    sourceSelectButton_ = new QPushButton(settings_i18n::selectSource(uiLang_), this);
    sourceSummaryLabel_ = new QLabel(this);
    sourceSummaryLabel_->setWordWrap(true);

    segmentSpin_ = new QDoubleSpinBox(this);
    segmentSpin_->setDecimals(2);
    segmentSpin_->setRange(1.0, 12.0);
    segmentSpin_->setSingleStep(0.1);
    segmentSpin_->setValue(static_cast<double>(initial.segmentSeconds));

    hopSpin_ = new QDoubleSpinBox(this);
    hopSpin_->setDecimals(2);
    hopSpin_->setRange(0.1, 6.0);
    hopSpin_->setSingleStep(0.1);
    hopSpin_->setValue(static_cast<double>(initial.hopSeconds));

    overlapMergeMethodCombo_ = new QComboBox(this);
    overlapMergeMethodCombo_->addItem("stable-tail", "stable-tail");
    overlapMergeMethodCombo_->addItem("commit-on-break", "commit-on-break");
    setComboData(overlapMergeMethodCombo_, normalizeOverlapMergeMethod(initial.overlapMergeMethod));

    sourceLanguageCombo_ = new QComboBox(this);
    sourceLanguageCombo_->addItem("auto", "auto");
    sourceLanguageCombo_->addItem("en", "en");
    sourceLanguageCombo_->addItem("zh-hant", "zh-hant");
    sourceLanguageCombo_->addItem("zh-hans", "zh-hans");
    sourceLanguageCombo_->addItem("ja", "ja");
    sourceLanguageCombo_->addItem("ko", "ko");
    setComboData(sourceLanguageCombo_, initial.sourceLanguage);

    modelCombo_ = new QComboBox(this);
    modelCombo_->setEditable(true);
    for (const QString &candidate : modelCandidates) {
        modelCombo_->addItem(candidate, candidate);
    }
    if (!initial.modelPath.trimmed().isEmpty() && modelCombo_->findText(initial.modelPath.trimmed()) < 0) {
        modelCombo_->addItem(initial.modelPath.trimmed(), initial.modelPath.trimmed());
    }
    modelCombo_->setCurrentText(initial.modelPath.trimmed());

    vadEnabledCheck_ = new QCheckBox(this);
    vadEnabledCheck_->setChecked(initial.vadEnabled);
    vadAdaptiveCheck_ = new QCheckBox(this);
    vadAdaptiveCheck_->setChecked(initial.vadAdaptiveEnabled);
    vadThresholdSpin_ = new QDoubleSpinBox(this);
    vadThresholdSpin_->setDecimals(3);
    vadThresholdSpin_->setRange(0.001, 0.200);
    vadThresholdSpin_->setSingleStep(0.001);
    vadThresholdSpin_->setValue(initial.vadRmsThreshold);

    translationEnabledCheck_ = new QCheckBox(this);
    translationEnabledCheck_->setChecked(initial.translationEnabled);

    translationStyleCombo_ = new QComboBox(this);
    translationStyleCombo_->addItem(settings_i18n::translationStyleStacked(uiLang_), "stacked");
    translationStyleCombo_->addItem(settings_i18n::translationStyleTranslatedOnly(uiLang_), "translation-only");
    setComboData(translationStyleCombo_, initial.translationStyle);

    translationLanguageCombo_ = new QComboBox(this);
    translationLanguageCombo_->addItem("en", "en");
    translationLanguageCombo_->addItem("zh", "zh");
    translationLanguageCombo_->addItem("ja", "ja");
    translationLanguageCombo_->addItem("ko", "ko");
    setComboData(translationLanguageCombo_, initial.toLang);

    fontSizeSpin_ = new QSpinBox(this);
    fontSizeSpin_->setRange(10, 64);
    fontSizeSpin_->setValue(initial.fontSize);

    opacitySlider_ = new QSlider(Qt::Horizontal, this);
    opacitySlider_->setRange(20, 100);
    opacitySlider_->setValue(std::clamp(static_cast<int>(std::lround(initial.opacity * 100.0F)), 20, 100));

    opacityValueLabel_ = new QLabel(this);
    opacityValueLabel_->setMinimumWidth(48);
    opacityValueLabel_->setAlignment(Qt::AlignRight | Qt::AlignVCenter);

    sourceColorButton_ = new QPushButton(this);
    translatedColorButton_ = new QPushButton(this);
    backgroundColorButton_ = new QPushButton(this);

    setButtonColor(sourceColorButton_, initial.sourceColor);
    setButtonColor(translatedColorButton_, initial.translatedColor);
    setButtonColor(backgroundColorButton_, initial.backgroundColor);

    form->addRow(settings_i18n::uiLanguageLabel(uiLang_), uiLanguageCombo_);
    form->addRow(settings_i18n::modeLabel(uiLang_), sourceModeCombo_);

    auto *sourceRow = new QHBoxLayout();
    sourceRow->addWidget(sourceSelectButton_);
    sourceRow->addWidget(sourceSummaryLabel_, 1);
    form->addRow(settings_i18n::sourceLabel(uiLang_), sourceRow);

    form->addRow(settings_i18n::segmentLabel(uiLang_), segmentSpin_);
    form->addRow(settings_i18n::hopLabel(uiLang_), hopSpin_);
    form->addRow(settings_i18n::mergeLabel(uiLang_), overlapMergeMethodCombo_);
    form->addRow(settings_i18n::sourceLanguageLabel(uiLang_), sourceLanguageCombo_);
    form->addRow("STT Model", modelCombo_);
    form->addRow("VAD Enabled", vadEnabledCheck_);
    form->addRow("Adaptive VAD", vadAdaptiveCheck_);
    form->addRow("VAD RMS Threshold", vadThresholdSpin_);

    auto *translationLabel = new QWidget(this);
    auto *translationLabelLayout = new QHBoxLayout(translationLabel);
    translationLabelLayout->setContentsMargins(0, 0, 0, 0);
    translationLabelLayout->setSpacing(6);
    translationLabelLayout->addWidget(new QLabel(settings_i18n::translationLabel(uiLang_), translationLabel));
    translationLabelLayout->addWidget(translationEnabledCheck_);
    translationLabelLayout->addStretch(1);

    form->addRow(translationLabel, translationStyleCombo_);
    form->addRow(settings_i18n::translationLanguageLabel(uiLang_), translationLanguageCombo_);
    form->addRow(settings_i18n::fontSizeLabel(uiLang_), fontSizeSpin_);

    auto *opacityRow = new QHBoxLayout();
    opacityRow->addWidget(opacitySlider_, 1);
    opacityRow->addWidget(opacityValueLabel_);
    form->addRow(settings_i18n::opacityLabel(uiLang_), opacityRow);

    form->addRow(settings_i18n::sourceColorLabel(uiLang_), sourceColorButton_);
    form->addRow(settings_i18n::translatedColorLabel(uiLang_), translatedColorButton_);
    form->addRow(settings_i18n::backgroundColorLabel(uiLang_), backgroundColorButton_);

    root->addLayout(form);

    auto *buttons =
        new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    root->addWidget(buttons);

    connect(sourceColorButton_, &QPushButton::clicked, this, &SettingsDialog::pickSourceColor);
    connect(translatedColorButton_,
            &QPushButton::clicked,
            this,
            &SettingsDialog::pickTranslatedColor);
    connect(backgroundColorButton_,
            &QPushButton::clicked,
            this,
            &SettingsDialog::pickBackgroundColor);
    connect(translationEnabledCheck_,
            &QCheckBox::checkStateChanged,
            this,
            &SettingsDialog::onTranslationToggle);
    connect(sourceModeCombo_, &QComboBox::currentIndexChanged, this, [this](int) {
        onModeChanged();
    });
    connect(sourceSelectButton_, &QPushButton::clicked, this, &SettingsDialog::openSourceSelection);
    connect(segmentSpin_, &QDoubleSpinBox::valueChanged, this, [this](double segmentValue) {
        const double hopMax = std::max(0.1, segmentValue - 0.1);
        hopSpin_->setMaximum(hopMax);
        if (hopSpin_->value() > hopMax) {
            hopSpin_->setValue(hopMax);
        }
    });
    connect(opacitySlider_, &QSlider::valueChanged, this, [this](int value) {
        opacityValueLabel_->setText(QString("%1%").arg(value));
    });
    connect(vadEnabledCheck_, &QCheckBox::checkStateChanged, this, [this](int) {
        const bool enabled = vadEnabledCheck_->isChecked();
        vadAdaptiveCheck_->setEnabled(enabled);
        vadThresholdSpin_->setEnabled(enabled);
    });

    onModeChanged();
    segmentSpin_->valueChanged(segmentSpin_->value());
    opacitySlider_->valueChanged(opacitySlider_->value());
    onTranslationToggle();
    vadEnabledCheck_->checkStateChanged(vadEnabledCheck_->checkState());
}

RuntimeSettings SettingsDialog::settings() const {
    SettingsInputPayload payload;
    payload.uiLanguage = uiLanguageCombo_->currentData().toString();
    payload.sourceMode = sourceModeCombo_->currentData().toString();
    payload.loopbackDeviceId = selectedLoopbackDeviceId_;
    payload.appNames = selectedAppNames_;
    payload.sourceLanguage = sourceLanguageCombo_->currentData().toString();
    payload.modelPath = modelCombo_->currentText().trimmed();
    payload.segmentSeconds = static_cast<float>(segmentSpin_->value());
    payload.hopSeconds = static_cast<float>(hopSpin_->value());
    payload.overlapMergeMethod = overlapMergeMethodCombo_->currentData().toString();
    payload.vadEnabled = vadEnabledCheck_->isChecked();
    payload.vadAdaptiveEnabled = vadAdaptiveCheck_->isChecked();
    payload.vadRmsThreshold = static_cast<float>(vadThresholdSpin_->value());
    payload.translationEnabled = translationEnabledCheck_->isChecked();
    payload.translationTo = translationLanguageCombo_->currentData().toString();
    payload.translationStyle = translationStyleCombo_->currentData().toString();
    payload.fontSize = fontSizeSpin_->value();
    payload.opacity = static_cast<float>(opacitySlider_->value()) / 100.0F;
    payload.sourceColor = buttonColor(sourceColorButton_);
    payload.translatedColor = buttonColor(translatedColorButton_);
    payload.backgroundColor = buttonColor(backgroundColorButton_);
    return buildRuntimeSettingsFromPayload(payload);
}

void SettingsDialog::pickSourceColor() {
    const QColor picked = QColorDialog::getColor(
        buttonColor(sourceColorButton_), this, settings_i18n::pickSourceColorTitle(uiLang_));
    if (picked.isValid()) {
        setButtonColor(sourceColorButton_, picked);
    }
}

void SettingsDialog::pickTranslatedColor() {
    const QColor picked =
        QColorDialog::getColor(buttonColor(translatedColorButton_),
                               this,
                               settings_i18n::pickTranslatedColorTitle(uiLang_));
    if (picked.isValid()) {
        setButtonColor(translatedColorButton_, picked);
    }
}

void SettingsDialog::pickBackgroundColor() {
    const QColor picked = QColorDialog::getColor(
        buttonColor(backgroundColorButton_), this, settings_i18n::pickBackgroundColorTitle(uiLang_));
    if (picked.isValid()) {
        setButtonColor(backgroundColorButton_, picked);
    }
}

void SettingsDialog::onTranslationToggle() {
    const bool enabled = translationEnabledCheck_->isChecked();
    translationStyleCombo_->setEnabled(enabled);
    translationLanguageCombo_->setEnabled(enabled);
    translatedColorButton_->setEnabled(enabled);
}

void SettingsDialog::onModeChanged() {
    const QString mode = sourceModeCombo_->currentData().toString();
    const bool selectable = (mode == "loopback" || mode == "app");
    sourceSelectButton_->setVisible(selectable);
    sourceSummaryLabel_->setVisible(true);
    refreshSourceSummary();
}

void SettingsDialog::openSourceSelection() {
    const QString mode = sourceModeCombo_->currentData().toString();
    if (mode == "loopback") {
        QList<SourceDeviceEntry> entries;
        entries.reserve(loopbackDevices_.size());
        for (const SourceDeviceEntry &dev : loopbackDevices_) {
            entries.push_back(dev);
        }

        const QStringList selected = selectedLoopbackDeviceId_.isEmpty()
                                         ? QStringList{}
                                         : QStringList{selectedLoopbackDeviceId_};
        SourceSelectionDialog dialog(settings_i18n::selectLoopback(uiLang_), entries, selected, uiLang_, this);
        if (dialog.exec() != QDialog::Accepted) {
            return;
        }

        const QStringList picked = dialog.selectedValues();
        selectedLoopbackDeviceId_ = picked.isEmpty() ? QString{} : picked.front();
        refreshSourceSummary();
        return;
    }

    if (mode != "app") {
        return;
    }

    SourceSelectionDialog dialog(settings_i18n::selectApps(uiLang_), appSessions_, selectedAppNames_, uiLang_, this);
    if (dialog.exec() != QDialog::Accepted) {
        return;
    }

    selectedAppNames_ = dialog.selectedValues();
    refreshSourceSummary();
}

void SettingsDialog::refreshSourceSummary() {
    const QString mode = sourceModeCombo_->currentData().toString();
    if (mode == "microphone") {
        sourceSummaryLabel_->setText(settings_i18n::microphoneDefaultSummary(uiLang_));
        return;
    }

    if (mode == "loopback") {
        if (selectedLoopbackDeviceId_.isEmpty()) {
            sourceSummaryLabel_->setText(settings_i18n::loopbackDefaultSummary(uiLang_));
            return;
        }

        QString label;
        for (const SourceDeviceEntry &dev : loopbackDevices_) {
            if (dev.id == selectedLoopbackDeviceId_) {
                label = dev.label;
                break;
            }
        }

        if (label.isEmpty()) {
            sourceSummaryLabel_->setText(settings_i18n::loopbackMissingSummary(uiLang_));
        } else {
            sourceSummaryLabel_->setText(
                QString("%1 %2").arg(settings_i18n::loopbackSelectedPrefix(uiLang_), label));
        }
        return;
    }

    if (selectedAppNames_.isEmpty()) {
        sourceSummaryLabel_->setText(settings_i18n::appNoneSummary(uiLang_));
    } else {
        QStringList labels;
        for (const QString &selected : selectedAppNames_) {
            QString label;
            for (const SourceDeviceEntry &entry : appSessions_) {
                if (entry.id == selected) {
                    label = entry.label;
                    break;
                }
            }
            labels.push_back(label.isEmpty() ? selected : label);
        }
        sourceSummaryLabel_->setText(
            QString("%1 %2").arg(settings_i18n::appSelectedPrefix(uiLang_), labels.join(", ")));
    }
}

void SettingsDialog::setComboData(QComboBox *combo, const QString &value) {
    const int index = combo->findData(value);
    combo->setCurrentIndex(index >= 0 ? index : 0);
}

void SettingsDialog::setButtonColor(QPushButton *button, const QColor &color) const {
    const QColor valid = color.isValid() ? color : QColor("#FFFFFF");
    button->setText(valid.name(QColor::HexRgb));

    const QColor textColor = valid.lightness() > 120 ? QColor("#000000") : QColor("#FFFFFF");
    button->setStyleSheet(
        QString("background:%1; color:%2;")
            .arg(valid.name(QColor::HexRgb), textColor.name(QColor::HexRgb)));
}

QColor SettingsDialog::buttonColor(const QPushButton *button) const {
    const QColor value(button->text().trimmed());
    if (value.isValid()) {
        return value;
    }
    return QColor("#FFFFFF");
}


