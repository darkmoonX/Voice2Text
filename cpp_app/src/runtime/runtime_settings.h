#pragma once

#include <QColor>
#include <QString>
#include <QStringList>

struct RuntimeSettings {
    QString uiLanguage{"zh"};
    QString sourceMode{"loopback"};
    QString loopbackDeviceId;
    QStringList sourceApps;
    QString sourceLanguage{"auto"};
    QString modelPath;
    float segmentSeconds{6.0F};
    float hopSeconds{1.5F};
    QString overlapMergeMethod{"stable-tail"};
    bool vadEnabled{true};
    bool vadAdaptiveEnabled{true};
    float vadRmsThreshold{0.010F};

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

QString normalizeOverlapMergeMethod(const QString &rawMethod);
QString effectiveModelLabel(const QString &modelPath);
