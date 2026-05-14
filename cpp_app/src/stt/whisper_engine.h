#pragma once

#include <QByteArray>
#include <QObject>
#include <QRegularExpression>
#include <QString>

#include <cstdint>
#include <vector>

#include "whisper_config.h"

class WhisperEngine : public QObject {
    Q_OBJECT

public:
    explicit WhisperEngine(QString modelPath,
                           QString sourceLanguage = "auto",
                           float segmentSeconds = 6.0F,
                           QString overlapMergeMethod = "stable-tail",
                           float hopSeconds = 1.5F,
                           bool vadEnabled = true,
                           bool vadAdaptiveEnabled = true,
                           float vadRmsThreshold = 0.010F,
                           WhisperRuntimeParams runtimeParams = {},
                           QObject *parent = nullptr);
    ~WhisperEngine() override;

    bool initialize();
    bool isTranscriptionEnabled() const { return sttEnabled_; }

public slots:
    void onPcmChunk(const QByteArray &pcm16Mono, int sampleRate);

signals:
    void transcriptReady(const QString &text);
    void statusRaised(const QString &message);
    void errorRaised(const QString &message);

private:
    void processIfReady();
    std::vector<float> resampleTo16k(const std::vector<float> &input, int sampleRate) const;
    QString mergeIncremental(const QString &incoming);
    float lockRatio() const;
    QString mergeStableTail(const QString &overlapTail,
                            const QString &incoming,
                            float lockRatio) const;
    QString mergeByExactOverlap(const QString &base, const QString &incoming) const;
    QString mergeByFuzzyOverlap(const QString &base, const QString &incoming) const;
    QString normalizeOutputText(const QString &text) const;
    QString collapseRepeatedPhrases(const QString &text) const;
    QString collapseRepeatedCharSpans(const QString &text) const;
    int softSuffixPrefixOverlap(const QString &base, const QString &incoming) const;
    bool containsCjk(const QString &text) const;
    bool shouldProcessByVad(const std::vector<float> &audio16k);
    QString joinSeparator(const QString &base, const QString &incoming) const;
    QString normalizeChineseScript(const QString &text) const;
    static int overlapPrefixSuffix(const QString &base, const QString &incoming);

    QString modelPath_;
    QString sourceLanguage_{"auto"};
    QString chineseScript_{"none"};
    QString overlapMergeMethod_{"stable-tail"};
    WhisperRuntimeParams runtimeParams_{};
    std::vector<int16_t> pcmBuffer_;
    int inputSampleRate_{16000};
    int windowMs_{6000};
    int hopMs_{1500};
    QString frozenTranscript_;
    QString activeWindowTranscript_;
    QString lastEmittedTranscript_;
    bool sttEnabled_{false};
    bool vadEnabled_{true};
    bool vadAdaptiveEnabled_{true};
    float vadRmsThreshold_{0.010F};
    float vadAdaptiveNoiseFloor_{0.0F};

#if HAS_WHISPER
    struct whisper_context *ctx_{nullptr};
#endif
};
