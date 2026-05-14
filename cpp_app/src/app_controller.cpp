#include "app_controller.h"

#include <utility>

AppController::AppController(QString modelPath,
                 CaptureSourceMode sourceMode,
                 QString loopbackDeviceId,
                 QStringList sourceAppNames,
                 QString sourceLanguage,
                 float segmentSeconds,
                 QString overlapMergeMethod,
                 float hopSeconds,
                 bool vadEnabled,
                 bool vadAdaptiveEnabled,
                 float vadRmsThreshold,
                 WhisperRuntimeParams whisperParams,
                 bool translationEnabled,
                 QString fromLang,
                 QString toLang,
                 QObject *parent)
    : QObject(parent),
      capture_(new LoopbackCapture(sourceMode,
                   std::move(loopbackDeviceId),
                   std::move(sourceAppNames),
                   this)),
      whisper_(new WhisperEngine(std::move(modelPath),
                 std::move(sourceLanguage),
                 segmentSeconds,
                 std::move(overlapMergeMethod),
                 hopSeconds,
                 vadEnabled,
                 vadAdaptiveEnabled,
                 vadRmsThreshold,
                 whisperParams,
                 this)),
      translator_(translationEnabled, std::move(fromLang), std::move(toLang)) {
    connect(capture_, &LoopbackCapture::pcmChunkReady, whisper_, &WhisperEngine::onPcmChunk);

    connect(capture_, &LoopbackCapture::statusRaised, this, &AppController::statusReady);
    connect(capture_, &LoopbackCapture::errorRaised, this, &AppController::errorReady);

    connect(whisper_, &WhisperEngine::statusRaised, this, &AppController::statusReady);
    connect(whisper_, &WhisperEngine::errorRaised, this, &AppController::errorReady);
    connect(whisper_, &WhisperEngine::transcriptReady, this, &AppController::onTranscript);
}

AppController::~AppController() {
    stop();
}

bool AppController::start() {
    if (running_) {
        return true;
    }

    emit statusReady(translator_.info());

    if (!whisper_->initialize()) {
        emit errorReady("Whisper engine initialization failed.");
        return false;
    }

    if (whisper_->isTranscriptionEnabled()) {
        if (!capture_->start()) {
            emit errorReady("Loopback capture start failed.");
            return false;
        }
    } else {
        emit statusReady("Controller started with STT disabled.");
    }

    running_ = true;
    emit statusReady("Controller started.");
    return true;
}

void AppController::stop() {
    if (!running_) {
        return;
    }
    capture_->stop();
    running_ = false;
}

void AppController::onTranscript(const QString &text) {
    const QString source = text.trimmed();
    if (source.isEmpty()) {
        return;
    }

    QString translated;
    if (translator_.isEnabled()) {
        translated = translator_.translate(source).trimmed();
    }

    emit subtitleReady(source, translated);
}
