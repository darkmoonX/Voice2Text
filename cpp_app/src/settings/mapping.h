#pragma once

#include <QColor>
#include <QString>
#include <QStringList>

#include "runtime/runtime_settings.h"

struct SettingsInputPayload {
    QString uiLanguage;
    QString sourceMode;
    QString loopbackDeviceId;
    QStringList appNames;
    QString sourceLanguage;
    QString modelPath;
    float segmentSeconds{6.0F};
    float hopSeconds{1.5F};
    QString overlapMergeMethod;
    bool vadEnabled{true};
    bool vadAdaptiveEnabled{true};
    float vadRmsThreshold{0.010F};
    bool translationEnabled{false};
    QString translationTo;
    QString translationStyle;
    int fontSize{18};
    float opacity{0.8F};
    QColor sourceColor;
    QColor translatedColor;
    QColor backgroundColor;
};

RuntimeSettings buildRuntimeSettingsFromPayload(const SettingsInputPayload &payload);
