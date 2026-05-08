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

namespace {

class SourceSelectionDialog : public QDialog {
public:
    SourceSelectionDialog(const QString &title,
                          const QList<SourceDeviceEntry> &entries,
                          const QStringList &selected,
                          QWidget *parent)
        : QDialog(parent) {
        setWindowTitle(title);
        setMinimumSize(460, 420);

        auto *root = new QVBoxLayout(this);

        selectAll_ = new QCheckBox("全選", this);
        root->addWidget(selectAll_);

        auto *body = new QWidget(this);
        auto *bodyLayout = new QVBoxLayout(body);
        bodyLayout->setContentsMargins(0, 0, 0, 0);
        bodyLayout->setSpacing(4);

        if (entries.isEmpty()) {
            bodyLayout->addWidget(new QLabel("(沒有可選來源)", body));
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
                               QWidget *parent)
    : QDialog(parent) {
    loopbackDevices_ = loopbackDevices;
    appSessions_ = appSessions;
    selectedLoopbackDeviceId_ = initial.loopbackDeviceId;
    selectedAppNames_ = initial.sourceApps;

    setWindowTitle("Voice2Text 設定");
    setMinimumWidth(580);

    auto *root = new QVBoxLayout(this);
    auto *form = new QFormLayout();

    sourceModeCombo_ = new QComboBox(this);
    sourceModeCombo_->addItem("輸出回放", "loopback");
    sourceModeCombo_->addItem("麥克風", "microphone");
    sourceModeCombo_->addItem("指定應用", "app");
    setComboData(sourceModeCombo_, initial.sourceMode);

    sourceSelectButton_ = new QPushButton("選擇來源", this);
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
    overlapMergeMethodCombo_->addItem("覆蓋最近視窗（推薦）", "replace-window");
    overlapMergeMethodCombo_->addItem("字尾重疊合併", "suffix-overlap");
    overlapMergeMethodCombo_->addItem("模糊重疊合併", "fuzzy-overlap");
    overlapMergeMethodCombo_->addItem("僅追加（舊行為）", "append-only");
    setComboData(overlapMergeMethodCombo_, initial.overlapMergeMethod);

    sourceLanguageCombo_ = new QComboBox(this);
    sourceLanguageCombo_->addItem("自動", "auto");
    sourceLanguageCombo_->addItem("英文", "en");
    sourceLanguageCombo_->addItem("中文（繁體）", "zh-hant");
    sourceLanguageCombo_->addItem("中文（簡體）", "zh-hans");
    sourceLanguageCombo_->addItem("日文", "ja");
    sourceLanguageCombo_->addItem("韓文", "ko");
    setComboData(sourceLanguageCombo_, initial.sourceLanguage);

    translationEnabledCheck_ = new QCheckBox(this);
    translationEnabledCheck_->setChecked(initial.translationEnabled);

    translationStyleCombo_ = new QComboBox(this);
    translationStyleCombo_->addItem("上下分行", "stacked");
    translationStyleCombo_->addItem("僅翻譯", "translation-only");
    setComboData(translationStyleCombo_, initial.translationStyle);

    translationLanguageCombo_ = new QComboBox(this);
    translationLanguageCombo_->addItem("英文", "en");
    translationLanguageCombo_->addItem("中文", "zh");
    translationLanguageCombo_->addItem("日文", "ja");
    translationLanguageCombo_->addItem("韓文", "ko");
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

    form->addRow("聲音來源模式", sourceModeCombo_);

    auto *sourceRow = new QHBoxLayout();
    sourceRow->addWidget(sourceSelectButton_);
    sourceRow->addWidget(sourceSummaryLabel_, 1);
    form->addRow("來源選擇", sourceRow);

    form->addRow("分段秒數", segmentSpin_);
    form->addRow("滑動步長秒數", hopSpin_);
    form->addRow("重疊整合方法", overlapMergeMethodCombo_);
    form->addRow("偵測語言", sourceLanguageCombo_);

    auto *translationLabel = new QWidget(this);
    auto *translationLabelLayout = new QHBoxLayout(translationLabel);
    translationLabelLayout->setContentsMargins(0, 0, 0, 0);
    translationLabelLayout->setSpacing(6);
    translationLabelLayout->addWidget(new QLabel("翻譯", translationLabel));
    translationLabelLayout->addWidget(translationEnabledCheck_);
    translationLabelLayout->addStretch(1);

    form->addRow(translationLabel, translationStyleCombo_);
    form->addRow("翻譯語言", translationLanguageCombo_);
    form->addRow("字體大小", fontSizeSpin_);

    auto *opacityRow = new QHBoxLayout();
    opacityRow->addWidget(opacitySlider_, 1);
    opacityRow->addWidget(opacityValueLabel_);
    form->addRow("透明度", opacityRow);

    form->addRow("來源文字顏色", sourceColorButton_);
    form->addRow("翻譯文字顏色", translatedColorButton_);
    form->addRow("背景顏色", backgroundColorButton_);

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

    onModeChanged();
    segmentSpin_->valueChanged(segmentSpin_->value());
    opacitySlider_->valueChanged(opacitySlider_->value());
    onTranslationToggle();
}

RuntimeSettings SettingsDialog::settings() const {
    RuntimeSettings out;
    out.sourceMode = sourceModeCombo_->currentData().toString();
    out.loopbackDeviceId = selectedLoopbackDeviceId_;
    if (out.sourceMode == "app") {
        out.sourceApps = selectedAppNames_;
    }

    out.sourceLanguage = sourceLanguageCombo_->currentData().toString();
    if (out.sourceLanguage.isEmpty()) {
        out.sourceLanguage = "auto";
    }

    out.segmentSeconds = static_cast<float>(segmentSpin_->value());
    out.hopSeconds = static_cast<float>(hopSpin_->value());
    if (out.hopSeconds > out.segmentSeconds) {
        out.hopSeconds = out.segmentSeconds;
    }
    out.overlapMergeMethod = overlapMergeMethodCombo_->currentData().toString();

    out.translationEnabled = translationEnabledCheck_->isChecked();
    out.toLang = translationLanguageCombo_->currentData().toString();

    if (out.sourceLanguage == "auto") {
        out.fromLang = "auto";
    } else if (out.sourceLanguage == "zh-hant" || out.sourceLanguage == "zh-hans") {
        out.fromLang = "zh";
    } else {
        out.fromLang = out.sourceLanguage;
    }

    out.translationStyle = translationStyleCombo_->currentData().toString();
    out.fontSize = fontSizeSpin_->value();
    out.opacity = static_cast<float>(opacitySlider_->value()) / 100.0F;
    out.sourceColor = buttonColor(sourceColorButton_);
    out.translatedColor = buttonColor(translatedColorButton_);
    out.backgroundColor = buttonColor(backgroundColorButton_);
    return out;
}

void SettingsDialog::pickSourceColor() {
    const QColor picked = QColorDialog::getColor(buttonColor(sourceColorButton_), this, "來源文字顏色");
    if (picked.isValid()) {
        setButtonColor(sourceColorButton_, picked);
    }
}

void SettingsDialog::pickTranslatedColor() {
    const QColor picked =
        QColorDialog::getColor(buttonColor(translatedColorButton_), this, "翻譯文字顏色");
    if (picked.isValid()) {
        setButtonColor(translatedColorButton_, picked);
    }
}

void SettingsDialog::pickBackgroundColor() {
    const QColor picked = QColorDialog::getColor(buttonColor(backgroundColorButton_), this, "背景顏色");
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
        SourceSelectionDialog dialog("選擇輸出回放來源", entries, selected, this);
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

    SourceSelectionDialog dialog("選擇應用來源", appSessions_, selectedAppNames_, this);
    if (dialog.exec() != QDialog::Accepted) {
        return;
    }

    selectedAppNames_ = dialog.selectedValues();
    refreshSourceSummary();
}

void SettingsDialog::refreshSourceSummary() {
    const QString mode = sourceModeCombo_->currentData().toString();
    if (mode == "microphone") {
        sourceSummaryLabel_->setText("使用預設麥克風來源");
        return;
    }

    if (mode == "loopback") {
        if (selectedLoopbackDeviceId_.isEmpty()) {
            sourceSummaryLabel_->setText("使用預設輸出回放來源");
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
            sourceSummaryLabel_->setText("已選擇裝置不可用，將回退預設來源");
        } else {
            sourceSummaryLabel_->setText(QString("已選擇裝置: %1").arg(label));
        }
        return;
    }

    if (selectedAppNames_.isEmpty()) {
        sourceSummaryLabel_->setText("尚未選擇應用");
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
        sourceSummaryLabel_->setText(QString("已選擇應用: %1").arg(labels.join(", ")));
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
