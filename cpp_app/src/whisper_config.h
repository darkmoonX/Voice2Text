#pragma once

#include <QString>
#include <QStringList>

struct WhisperRuntimeParams {
    int maxContext{96};
    float entropyThold{2.4F};
    float logprobThold{-1.0F};
    float noSpeechThold{0.6F};
    float temperature{0.0F};
    int beamSize{1};
    int bestOf{1};
};

QString resolveWhisperConfigPath();
WhisperRuntimeParams loadWhisperRuntimeParams(const QString &path,
                                              QStringList *warnings = nullptr);
