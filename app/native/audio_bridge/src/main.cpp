#include <cstdio>
#include <algorithm>

#include <QByteArray>
#include <QCommandLineOption>
#include <QCommandLineParser>
#include <QCoreApplication>
#include <QStringList>
#include <QTextStream>

#include "loopback_capture.h"

#ifdef Q_OS_WIN
#include <fcntl.h>
#include <io.h>
#include <audioclient.h>
#if defined(__has_include)
#if __has_include(<audioclientactivationparams.h>)
#include <audioclientactivationparams.h>
#define VOICE2TEXT_HAS_PROCESS_LOOPBACK_HEADER 1
#else
#define VOICE2TEXT_HAS_PROCESS_LOOPBACK_HEADER 0
#endif
#else
#define VOICE2TEXT_HAS_PROCESS_LOOPBACK_HEADER 0
#endif
#if VOICE2TEXT_HAS_PROCESS_LOOPBACK_HEADER
#define VOICE2TEXT_HAS_PROCESS_LOOPBACK 1
#else
#define VOICE2TEXT_HAS_PROCESS_LOOPBACK 0
#endif
#endif

static CaptureSourceMode captureModeFromString(const QString &modeRaw) {
    const QString mode = modeRaw.trimmed().toLower();
    if (mode == "microphone") {
        return CaptureSourceMode::Microphone;
    }
    if (mode == "app") {
        return CaptureSourceMode::App;
    }
    return CaptureSourceMode::Loopback;
}

int main(int argc, char *argv[]) {
    QCoreApplication app(argc, argv);
    app.setApplicationName("Voice2Text Capture Bridge");

    QCommandLineParser parser;
    parser.setApplicationDescription("Headless audio capture bridge for Python runtime.");
    parser.addHelpOption();

    QCommandLineOption sourceModeOption("source-mode", "Source mode: loopback|microphone|app.", "mode", "loopback");
    QCommandLineOption sourceAppsOption("source-apps", "Comma-separated app names for app source mode.", "names", "");
    QCommandLineOption sourceDeviceIdOption(
        "source-device-id",
        "Windows endpoint device id used by loopback/app capture fallback.",
        "id",
        "");
    QCommandLineOption probeProcessLoopbackOption(
        "probe-process-loopback",
        "Probe whether this build/toolchain supports Application Loopback Capture APIs.");
    parser.addOption(sourceModeOption);
    parser.addOption(sourceAppsOption);
    parser.addOption(sourceDeviceIdOption);
    parser.addOption(probeProcessLoopbackOption);
    parser.process(app);

    if (parser.isSet(probeProcessLoopbackOption)) {
        QTextStream err(stderr);
#if defined(Q_OS_WIN) && VOICE2TEXT_HAS_PROCESS_LOOPBACK
        err << "process-loopback-supported\n";
        err.flush();
        return 0;
#else
        err << "process-loopback-unsupported\n";
        err.flush();
        return 2;
#endif
    }

#ifdef Q_OS_WIN
    _setmode(_fileno(stdout), _O_BINARY);
#endif

    CaptureSourceMode mode = captureModeFromString(parser.value(sourceModeOption));
    QStringList sourceApps;
    for (const QString &piece : parser.value(sourceAppsOption).split(',')) {
        const QString item = piece.trimmed();
        if (!item.isEmpty()) {
            sourceApps.push_back(item);
        }
    }
    const QString deviceId = parser.value(sourceDeviceIdOption).trimmed();

    LoopbackCapture capture(mode, deviceId, sourceApps);
    QTextStream err(stderr);
    QObject::connect(&capture, &LoopbackCapture::statusRaised, &app, [&err](const QString &message) {
        err << "[capture-status] " << message << "\n";
        err.flush();
    });
    QObject::connect(&capture, &LoopbackCapture::errorRaised, &app, [&err, &app](const QString &message) {
        err << "[capture-error] " << message << "\n";
        err.flush();
        app.exit(2);
    });
    QObject::connect(&capture, &LoopbackCapture::pcmChunkReady, &app, [&app](const QByteArray &pcm16, int sampleRate, int channels) {
        if (pcm16.isEmpty()) {
            return;
        }
        char header[16] = {'V', '2', 'T', 'B', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        const quint32 sr = static_cast<quint32>(sampleRate);
        const quint32 ch = static_cast<quint32>((std::max)(1, channels));
        const quint32 len = static_cast<quint32>(pcm16.size());
        header[4] = static_cast<char>(sr & 0xFFU);
        header[5] = static_cast<char>((sr >> 8U) & 0xFFU);
        header[6] = static_cast<char>((sr >> 16U) & 0xFFU);
        header[7] = static_cast<char>((sr >> 24U) & 0xFFU);
        header[8] = static_cast<char>(ch & 0xFFU);
        header[9] = static_cast<char>((ch >> 8U) & 0xFFU);
        header[10] = static_cast<char>((ch >> 16U) & 0xFFU);
        header[11] = static_cast<char>((ch >> 24U) & 0xFFU);
        header[12] = static_cast<char>(len & 0xFFU);
        header[13] = static_cast<char>((len >> 8U) & 0xFFU);
        header[14] = static_cast<char>((len >> 16U) & 0xFFU);
        header[15] = static_cast<char>((len >> 24U) & 0xFFU);

        if (std::fwrite(header, sizeof(char), sizeof(header), stdout) != sizeof(header)) {
            app.exit(3);
            return;
        }
        if (std::fwrite(pcm16.constData(), sizeof(char), static_cast<size_t>(pcm16.size()), stdout) !=
            static_cast<size_t>(pcm16.size())) {
            app.exit(3);
            return;
        }
        std::fflush(stdout);
    });
    QObject::connect(&app, &QCoreApplication::aboutToQuit, [&capture]() {
        capture.stop();
    });

    if (!capture.start()) {
        err << "[capture-error] failed to start capture bridge\n";
        err.flush();
        return 2;
    }
    return app.exec();
}
