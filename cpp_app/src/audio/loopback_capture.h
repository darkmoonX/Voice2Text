#pragma once

#include <QByteArray>
#include <QObject>
#include <QString>
#include <QStringList>

#include <atomic>
#include <thread>

enum class CaptureSourceMode {
    Loopback,
    Microphone,
    App,
};

class LoopbackCapture : public QObject {
    Q_OBJECT

public:
    explicit LoopbackCapture(CaptureSourceMode mode = CaptureSourceMode::Loopback,
                             QString targetDeviceId = {},
                             QStringList targetAppNames = {},
                             QObject *parent = nullptr);
    ~LoopbackCapture() override;

    bool start();
    void stop();

signals:
    void pcmChunkReady(const QByteArray &pcm16Mono, int sampleRate);
    void statusRaised(const QString &message);
    void errorRaised(const QString &message);

private:
    void captureLoop();

    CaptureSourceMode mode_{CaptureSourceMode::Loopback};
    QString targetDeviceId_;
    QStringList targetAppNames_;
    std::atomic_bool running_{false};
    std::thread worker_;
};