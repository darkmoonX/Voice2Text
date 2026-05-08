#pragma once

#include <QObject>
#include <QString>
#include <QStringList>

#include "audio/loopback_capture.h"
#include "stt/whisper_engine.h"
#include "translation/translation_engine.h"
#include "whisper_config.h"

class AppController : public QObject {
    Q_OBJECT

public:
    AppController(QString modelPath,
                  CaptureSourceMode sourceMode,
                  QString loopbackDeviceId,
                  QStringList sourceAppNames,
                  QString sourceLanguage,
                  float segmentSeconds,
                  QString overlapMergeMethod,
                  float hopSeconds,
                  WhisperRuntimeParams whisperParams,
                  bool translationEnabled,
                  QString fromLang,
                  QString toLang,
                  QObject *parent = nullptr);
    ~AppController() override;

    bool start();
    void stop();

signals:
    void subtitleReady(const QString &sourceText, const QString &translatedText);
    void statusReady(const QString &text);
    void errorReady(const QString &text);

private slots:
    void onTranscript(const QString &text);

private:
    LoopbackCapture *capture_{nullptr};
    WhisperEngine *whisper_{nullptr};
    TranslationEngine translator_;
    bool running_{false};
};