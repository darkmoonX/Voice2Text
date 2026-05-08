#include <algorithm>
#include <cmath>
#include <memory>
#include <new>

#include <QApplication>
#include <QColor>
#include <QCommandLineOption>
#include <QCommandLineParser>
#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QFont>
#include <QThread>
#include <QSet>
#include <QStringList>
#include <QTextStream>

#ifdef Q_OS_WIN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <audiopolicy.h>
#include <mmdeviceapi.h>
#include <functiondiscoverykeys_devpkey.h>
#endif

#include "app_controller.h"
#include "runtime_logger.h"
#include "settings_dialog.h"
#include "subtitle_overlay_window.h"
#include "tray_controller.h"
#include "whisper_config.h"

#ifdef Q_OS_WIN
template <typename T>
void safeRelease(T **ptr) {
    if (ptr != nullptr && *ptr != nullptr) {
        (*ptr)->Release();
        *ptr = nullptr;
    }
}
#endif

static QString processNameFromPid(DWORD pid) {
    if (pid == 0) {
        return {};
    }

#ifdef Q_OS_WIN
    const HANDLE process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
    if (process == nullptr || process == INVALID_HANDLE_VALUE) {
        return {};
    }

    DWORD length = 32767;
    std::unique_ptr<wchar_t[]> pathBuffer(new (std::nothrow) wchar_t[length + 1]);
    QString name;
    if (pathBuffer != nullptr &&
        QueryFullProcessImageNameW(process, 0, pathBuffer.get(), &length)) {
        const QString fullPath = QString::fromWCharArray(pathBuffer.get(), static_cast<int>(length));
        const int slash = std::max(fullPath.lastIndexOf('\\'), fullPath.lastIndexOf('/'));
        name = (slash >= 0 ? fullPath.mid(slash + 1) : fullPath).trimmed();
    }
    CloseHandle(process);
    return name;
#endif

    return {};
}

static QList<SourceDeviceEntry> listMixerAppSessionEntries() {
    QList<SourceDeviceEntry> entries;

#ifdef Q_OS_WIN
    HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    const bool comReady = SUCCEEDED(hr) || hr == RPC_E_CHANGED_MODE;
    const bool shouldUninitialize = SUCCEEDED(hr);
    if (!comReady) {
        return entries;
    }

    IMMDeviceEnumerator *enumerator = nullptr;
    IMMDevice *device = nullptr;
    IAudioSessionManager2 *sessionManager = nullptr;
    IAudioSessionEnumerator *sessionEnumerator = nullptr;
    int sessionCount = 0;
    QSet<QString> seenIds;

    hr = CoCreateInstance(__uuidof(MMDeviceEnumerator),
                          nullptr,
                          CLSCTX_ALL,
                          __uuidof(IMMDeviceEnumerator),
                          reinterpret_cast<void **>(&enumerator));
    if (FAILED(hr)) {
        goto cleanup;
    }

    hr = enumerator->GetDefaultAudioEndpoint(eRender, eConsole, &device);
    if (FAILED(hr)) {
        goto cleanup;
    }

    hr = device->Activate(__uuidof(IAudioSessionManager2),
                          CLSCTX_ALL,
                          nullptr,
                          reinterpret_cast<void **>(&sessionManager));
    if (FAILED(hr)) {
        goto cleanup;
    }

    hr = sessionManager->GetSessionEnumerator(&sessionEnumerator);
    if (FAILED(hr)) {
        goto cleanup;
    }

    hr = sessionEnumerator->GetCount(&sessionCount);
    if (FAILED(hr)) {
        goto cleanup;
    }

    for (int index = 0; index < sessionCount; ++index) {
        IAudioSessionControl *control = nullptr;
        IAudioSessionControl2 *control2 = nullptr;
        LPWSTR displayNameRaw = nullptr;

        if (FAILED(sessionEnumerator->GetSession(index, &control)) || control == nullptr) {
            continue;
        }

        AudioSessionState state = AudioSessionStateExpired;
        if (FAILED(control->GetState(&state)) || state == AudioSessionStateExpired) {
            safeRelease(&control);
            continue;
        }

        control->GetDisplayName(&displayNameRaw);

        DWORD processId = 0;
        if (SUCCEEDED(control->QueryInterface(__uuidof(IAudioSessionControl2),
                                              reinterpret_cast<void **>(&control2))) &&
            control2 != nullptr) {
            control2->GetProcessId(&processId);
        }

        const QString processName = processNameFromPid(processId).trimmed();
        const QString displayName =
            (displayNameRaw != nullptr) ? QString::fromWCharArray(displayNameRaw).trimmed() : QString{};

        if (!processName.isEmpty()) {
            const QString id = processName.toLower();
            if (!seenIds.contains(id)) {
                seenIds.insert(id);

                QString label = processName;
                if (!displayName.isEmpty() &&
                    displayName.compare(processName, Qt::CaseInsensitive) != 0) {
                    label = QString("%1 (%2)").arg(displayName, processName);
                }

                entries.push_back(SourceDeviceEntry{processName, label});
            }
        }

        if (displayNameRaw != nullptr) {
            CoTaskMemFree(displayNameRaw);
            displayNameRaw = nullptr;
        }
        safeRelease(&control2);
        safeRelease(&control);
    }

cleanup:
    safeRelease(&sessionEnumerator);
    safeRelease(&sessionManager);
    safeRelease(&device);
    safeRelease(&enumerator);
    if (shouldUninitialize) {
        CoUninitialize();
    }
#endif

    std::sort(entries.begin(), entries.end(), [](const SourceDeviceEntry &lhs, const SourceDeviceEntry &rhs) {
        return lhs.label.compare(rhs.label, Qt::CaseInsensitive) < 0;
    });
    return entries;
}

static QStringList listMixerAppSessionProcessNames() {
    QStringList names;
    const QList<SourceDeviceEntry> entries = listMixerAppSessionEntries();
    for (const SourceDeviceEntry &entry : entries) {
        const QString id = entry.id.trimmed();
        if (!id.isEmpty()) {
            names.push_back(id);
        }
    }

    names.removeDuplicates();
    std::sort(names.begin(), names.end(), [](const QString &lhs, const QString &rhs) {
        return lhs.compare(rhs, Qt::CaseInsensitive) < 0;
    });
    return names;
}

static QList<SourceDeviceEntry> listLoopbackDeviceEntries() {
    QList<SourceDeviceEntry> devices;

#ifdef Q_OS_WIN
    HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    const bool comReady = SUCCEEDED(hr) || hr == RPC_E_CHANGED_MODE;
    const bool shouldUninitialize = SUCCEEDED(hr);
    if (!comReady) {
        return devices;
    }

    IMMDeviceEnumerator *enumerator = nullptr;
    IMMDeviceCollection *collection = nullptr;
    UINT count = 0;

    hr = CoCreateInstance(__uuidof(MMDeviceEnumerator),
                          nullptr,
                          CLSCTX_ALL,
                          __uuidof(IMMDeviceEnumerator),
                          reinterpret_cast<void **>(&enumerator));
    if (FAILED(hr)) {
        goto cleanup;
    }

    hr = enumerator->EnumAudioEndpoints(eRender, DEVICE_STATE_ACTIVE, &collection);
    if (FAILED(hr)) {
        goto cleanup;
    }

    hr = collection->GetCount(&count);
    if (FAILED(hr)) {
        goto cleanup;
    }

    for (UINT index = 0; index < count; ++index) {
        IMMDevice *device = nullptr;
        IPropertyStore *props = nullptr;
        LPWSTR idRaw = nullptr;
        PROPVARIANT friendlyName;
        PropVariantInit(&friendlyName);

        if (FAILED(collection->Item(index, &device))) {
            PropVariantClear(&friendlyName);
            continue;
        }

        QString id;
        QString label;

        if (SUCCEEDED(device->GetId(&idRaw)) && idRaw != nullptr) {
            id = QString::fromWCharArray(idRaw).trimmed();
        }

        if (SUCCEEDED(device->OpenPropertyStore(STGM_READ, &props)) && props != nullptr) {
            if (SUCCEEDED(props->GetValue(PKEY_Device_FriendlyName, &friendlyName)) &&
                friendlyName.vt == VT_LPWSTR && friendlyName.pwszVal != nullptr) {
                label = QString::fromWCharArray(friendlyName.pwszVal).trimmed();
            }
        }

        if (!id.isEmpty()) {
            if (label.isEmpty()) {
                label = id;
            }
            devices.push_back(SourceDeviceEntry{id, label});
        }

        PropVariantClear(&friendlyName);
        if (idRaw != nullptr) {
            CoTaskMemFree(idRaw);
            idRaw = nullptr;
        }
        safeRelease(&props);
        safeRelease(&device);
    }

cleanup:
    safeRelease(&collection);
    safeRelease(&enumerator);
    if (shouldUninitialize) {
        CoUninitialize();
    }
#endif

    std::sort(devices.begin(), devices.end(), [](const SourceDeviceEntry &lhs, const SourceDeviceEntry &rhs) {
        return lhs.label.compare(rhs.label, Qt::CaseInsensitive) < 0;
    });
    return devices;
}

static QString resolveModelPath(const QString &cliValue) {
    const QString cliPath = cliValue.trimmed();
    if (!cliPath.isEmpty() && QFileInfo::exists(cliPath)) {
        return QFileInfo(cliPath).absoluteFilePath();
    }

    const QString envPath = qEnvironmentVariable("WHISPER_MODEL_PATH").trimmed();
    if (!envPath.isEmpty() && QFileInfo::exists(envPath)) {
        return QFileInfo(envPath).absoluteFilePath();
    }

    const QString appDir = QCoreApplication::applicationDirPath();
    const QString cwd = QDir::currentPath();

    QStringList roots;
    roots << appDir << QDir(appDir).absoluteFilePath("..") << QDir(appDir).absoluteFilePath("../..")
          << cwd << QDir(cwd).absoluteFilePath("..");
    roots.removeDuplicates();

    const QStringList relDirs = {
        "models",
        "third_party/whisper.cpp/models",
        "../models",
        "../third_party/whisper.cpp/models",
        "../../models",
        "../../third_party/whisper.cpp/models",
    };

    const QStringList modelNames = {
        "ggml-large-v3-turbo.bin",
        "ggml-large-v3.bin",
        "ggml-medium.bin",
        "ggml-small.bin",
        "ggml-base.bin",
        "ggml-tiny.bin",
        "ggml-medium.en.bin",
        "ggml-small.en.bin",
        "ggml-base.en.bin",
        "ggml-tiny.en.bin",
    };

    for (const QString &root : roots) {
        const QDir rootDir(root);
        for (const QString &relDir : relDirs) {
            const QDir candidateDir(rootDir.absoluteFilePath(relDir));
            for (const QString &modelName : modelNames) {
                const QString modelPath = candidateDir.absoluteFilePath(modelName);
                if (QFileInfo::exists(modelPath)) {
                    return QFileInfo(modelPath).absoluteFilePath();
                }
            }
        }
    }

    return {};
}

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
    QApplication app(argc, argv);
    app.setApplicationName("Voice2Text C++");
    app.setQuitOnLastWindowClosed(false);

    QCommandLineParser parser;
    parser.setApplicationDescription("Speech-to-text overlay with whisper.cpp and Qt");
    parser.addHelpOption();

    const QString whisperConfigPath = resolveWhisperConfigPath();
    QStringList whisperConfigWarnings;
    WhisperRuntimeParams whisperParams =
        loadWhisperRuntimeParams(whisperConfigPath, &whisperConfigWarnings);

    QCommandLineOption modelPathOption(QStringList() << "m" << "model-path",
                                       "Path to whisper.cpp model file. Optional when auto-discovery succeeds or WHISPER_MODEL_PATH is set.",
                                       "path");
    QCommandLineOption translateOption("translate",
                                       "Enable translation stage (Python Argos bridge).");
    QCommandLineOption fromLangOption("from-lang", "Source language code.", "code", "auto");
    QCommandLineOption toLangOption("to-lang", "Target language code.", "code", "zh");

    QCommandLineOption sourceModeOption("source-mode", "Source mode: loopback|microphone|app (process-loopback).",
                                        "mode", "loopback");
    QCommandLineOption sourceAppsOption("source-apps",
                                        "Comma-separated app names for app source mode.",
                                        "names",
                                        "");
    QCommandLineOption listAppSessionsOption(
        "list-app-sessions",
        "List active Windows mixer sessions for app process-loopback source selection and exit.");
    QCommandLineOption smokeSwitchTestOption(
        "smoke-switch-test",
        "Run automatic runtime settings switch smoke test and exit.");
    QCommandLineOption sourceLanguageOption("source-language",
                                            "STT language hint: auto|en|zh|zh-hant|zh-hans|ja|ko.",
                                            "lang",
                                            "auto");
    QCommandLineOption segmentSecondsOption("segment-seconds",
                                            "Transcription segment window in seconds.",
                                            "sec",
                                            "6.0");
    QCommandLineOption hopSecondsOption("hop-seconds",
                                        "Sliding hop interval in seconds.",
                                        "sec",
                                        "1.5");
    QCommandLineOption overlapMergeMethodOption(
        "overlap-merge-method",
        "Overlap merge method: replace-window|suffix-overlap|fuzzy-overlap|append-only.",
        "method",
        "replace-window");
    QCommandLineOption maxContextOption(QStringList() << "mc" << "max-context",
                                        "Whisper max context tokens (0 disables context carry).",
                                        "tokens",
                                        QString::number(whisperParams.maxContext));
    QCommandLineOption entropyTholdOption("entropy-thold",
                                          "Whisper entropy threshold.",
                                          "value",
                                          QString::number(whisperParams.entropyThold, 'f', 3));
    QCommandLineOption logprobTholdOption("logprob-thold",
                                          "Whisper log probability threshold.",
                                          "value",
                                          QString::number(whisperParams.logprobThold, 'f', 3));
    QCommandLineOption noSpeechTholdOption("no-speech-thold",
                                           "Whisper no-speech threshold.",
                                           "value",
                                           QString::number(whisperParams.noSpeechThold, 'f', 3));
    QCommandLineOption temperatureOption("temperature",
                                         "Whisper decode temperature.",
                                         "value",
                                         QString::number(whisperParams.temperature, 'f', 3));
    QCommandLineOption beamSizeOption("beam-size",
                                      "Whisper beam size.",
                                      "value",
                                      QString::number(whisperParams.beamSize));
    QCommandLineOption bestOfOption("best-of",
                                    "Whisper best-of decode count.",
                                    "value",
                                    QString::number(whisperParams.bestOf));

    QCommandLineOption bilingualStyleOption("bilingual-style",
                                            "Subtitle style: stacked|translation-only.",
                                            "style",
                                            "stacked");
    QCommandLineOption sourceTextColorOption("source-text-color", "Source text color.", "hex",
                                             "#F0F2F5");
    QCommandLineOption translatedTextColorOption("translated-text-color", "Translated text color.",
                                                 "hex", "#FFD98A");
    QCommandLineOption backgroundColorOption("background-color", "Overlay background color.", "hex",
                                             "#0A101A");
    QCommandLineOption overlayOpacityOption("overlay-opacity", "Overlay opacity (0.2-1.0).", "value",
                                            "0.8");
    QCommandLineOption fontSizeOption("font-size", "Overlay font size.", "size", "18");

    parser.addOption(modelPathOption);
    parser.addOption(translateOption);
    parser.addOption(fromLangOption);
    parser.addOption(toLangOption);
    parser.addOption(sourceModeOption);
    parser.addOption(sourceAppsOption);
    parser.addOption(listAppSessionsOption);
    parser.addOption(smokeSwitchTestOption);
    parser.addOption(sourceLanguageOption);
    parser.addOption(segmentSecondsOption);
    parser.addOption(hopSecondsOption);
    parser.addOption(overlapMergeMethodOption);
    parser.addOption(maxContextOption);
    parser.addOption(entropyTholdOption);
    parser.addOption(logprobTholdOption);
    parser.addOption(noSpeechTholdOption);
    parser.addOption(temperatureOption);
    parser.addOption(beamSizeOption);
    parser.addOption(bestOfOption);
    parser.addOption(bilingualStyleOption);
    parser.addOption(sourceTextColorOption);
    parser.addOption(translatedTextColorOption);
    parser.addOption(backgroundColorOption);
    parser.addOption(overlayOpacityOption);
    parser.addOption(fontSizeOption);
    parser.process(app);

    if (parser.isSet(listAppSessionsOption)) {
        QTextStream out(stdout);
        for (const QString &name : listMixerAppSessionProcessNames()) {
            out << name << "\n";
        }
        return 0;
    }

    const QString modelPath = resolveModelPath(parser.value(modelPathOption));

    RuntimeSettings runtime;
    runtime.translationEnabled = parser.isSet(translateOption);
    runtime.fromLang = parser.value(fromLangOption).trimmed().toLower();
    runtime.toLang = parser.value(toLangOption).trimmed().toLower();
    runtime.sourceLanguage = parser.value(sourceLanguageOption).trimmed().toLower();
    if (runtime.sourceLanguage.isEmpty()) {
        runtime.sourceLanguage = "auto";
    }

    runtime.segmentSeconds = std::clamp(parser.value(segmentSecondsOption).toFloat(), 1.0F, 12.0F);
    runtime.hopSeconds =
        std::clamp(parser.value(hopSecondsOption).toFloat(), 0.1F, std::max(0.1F, runtime.segmentSeconds - 0.1F));

    runtime.overlapMergeMethod = parser.value(overlapMergeMethodOption).trimmed().toLower();
    if (runtime.overlapMergeMethod.isEmpty()) {
        runtime.overlapMergeMethod = "replace-window";
    }

    whisperParams.maxContext = std::clamp(parser.value(maxContextOption).toInt(), 0, 8192);
    whisperParams.entropyThold = std::clamp(parser.value(entropyTholdOption).toFloat(), 0.0F, 10.0F);
    whisperParams.logprobThold = std::clamp(parser.value(logprobTholdOption).toFloat(), -10.0F, 2.0F);
    whisperParams.noSpeechThold = std::clamp(parser.value(noSpeechTholdOption).toFloat(), 0.0F, 1.0F);
    whisperParams.temperature = std::clamp(parser.value(temperatureOption).toFloat(), 0.0F, 2.0F);
    whisperParams.beamSize = std::clamp(parser.value(beamSizeOption).toInt(), 1, 32);
    whisperParams.bestOf = std::clamp(parser.value(bestOfOption).toInt(), 1, 32);
    if (whisperParams.bestOf < whisperParams.beamSize) {
        whisperParams.bestOf = whisperParams.beamSize;
    }

    runtime.sourceMode = parser.value(sourceModeOption).trimmed().toLower();
    if (runtime.sourceMode != "loopback" && runtime.sourceMode != "microphone" && runtime.sourceMode != "app") {
        runtime.sourceMode = "loopback";
    }

    for (const QString &piece : parser.value(sourceAppsOption).split(',')) {
        const QString item = piece.trimmed();
        if (!item.isEmpty()) {
            runtime.sourceApps.push_back(item);
        }
    }

    runtime.translationStyle = parser.value(bilingualStyleOption).trimmed().toLower();
    if (runtime.translationStyle != "translation-only") {
        runtime.translationStyle = "stacked";
    }

    runtime.sourceColor = QColor(parser.value(sourceTextColorOption));
    if (!runtime.sourceColor.isValid()) {
        runtime.sourceColor = QColor("#F0F2F5");
    }

    runtime.translatedColor = QColor(parser.value(translatedTextColorOption));
    if (!runtime.translatedColor.isValid()) {
        runtime.translatedColor = QColor("#FFD98A");
    }

    runtime.backgroundColor = QColor(parser.value(backgroundColorOption));
    if (!runtime.backgroundColor.isValid()) {
        runtime.backgroundColor = QColor("#0A101A");
    }

    runtime.opacity = std::clamp(parser.value(overlayOpacityOption).toFloat(), 0.2F, 1.0F);
    runtime.fontSize = std::max(parser.value(fontSizeOption).toInt(), 10);

    SubtitleOverlayWindow overlay;
    overlay.applyVisualSettings(runtime.translationEnabled,
                                runtime.translationStyle,
                                runtime.fontSize,
                                runtime.opacity,
                                runtime.sourceColor,
                                runtime.translatedColor,
                                runtime.backgroundColor);

    if (modelPath.isEmpty()) {
        overlay.pushStatus(
            "No model file detected. STT is disabled. Provide --model-path or set WHISPER_MODEL_PATH.");
    } else {
        overlay.pushStatus(QString("Using model: %1").arg(modelPath));
    }
    if (!whisperConfigPath.trimmed().isEmpty()) {
        overlay.pushStatus(QString("Whisper config loaded: %1").arg(whisperConfigPath));
    }
    for (const QString &warning : whisperConfigWarnings) {
        overlay.pushStatus(QString("[whisper-config] %1").arg(warning));
    }

    RuntimeLogger logger("logs");
    std::unique_ptr<AppController> controller;

    const auto stopController = [&controller]() {
        if (controller != nullptr) {
            controller->stop();
            controller.reset();
        }
    };

    auto connectControllerSignals = [&overlay, &logger](AppController *instance) {
        QObject::connect(instance, &AppController::subtitleReady, &overlay,
                         &SubtitleOverlayWindow::pushSubtitle);
        QObject::connect(instance, &AppController::statusReady, &overlay,
                         &SubtitleOverlayWindow::pushStatus);
        QObject::connect(instance, &AppController::errorReady, &overlay,
                         &SubtitleOverlayWindow::pushError);
        QObject::connect(instance, &AppController::statusReady, &logger, &RuntimeLogger::info);
        QObject::connect(instance, &AppController::errorReady, &logger, &RuntimeLogger::error);
        QObject::connect(instance,
                         &AppController::subtitleReady,
                         &logger,
                         [&logger](const QString &sourceText, const QString &translatedText) {
                             if (!sourceText.trimmed().isEmpty()) {
                                 logger.info(QString("STT: %1").arg(sourceText));
                             }
                             if (!translatedText.trimmed().isEmpty()) {
                                 logger.info(QString("TRANSLATE: %1").arg(translatedText));
                             }
                         });
    };

    const auto startController = [&]() -> bool {
        stopController();

        controller = std::make_unique<AppController>(modelPath,
                                                     captureModeFromString(runtime.sourceMode),
                                                     runtime.loopbackDeviceId,
                                                     runtime.sourceApps,
                                                     runtime.sourceLanguage,
                                                     runtime.segmentSeconds,
                                                     runtime.overlapMergeMethod,
                                                     runtime.hopSeconds,
                                                     whisperParams,
                                                     runtime.translationEnabled,
                                                     runtime.fromLang,
                                                     runtime.toLang);
        connectControllerSignals(controller.get());

        if (!controller->start()) {
            overlay.pushError("Controller failed to start.");
            return false;
        }

        return true;
    };

    const auto applySettings = [&](const RuntimeSettings &updated) {
        const RuntimeSettings previous = runtime;
        runtime = updated;

        overlay.applyVisualSettings(runtime.translationEnabled,
                                    runtime.translationStyle,
                                    runtime.fontSize,
                                    runtime.opacity,
                                    runtime.sourceColor,
                                    runtime.translatedColor,
                                    runtime.backgroundColor);

        const auto changedFloat = [](float lhs, float rhs) {
            return std::abs(lhs - rhs) > 0.0001F;
        };

        const bool requiresRestart =
            previous.sourceMode != runtime.sourceMode ||
            previous.loopbackDeviceId != runtime.loopbackDeviceId ||
            previous.sourceApps != runtime.sourceApps ||
            previous.sourceLanguage != runtime.sourceLanguage ||
            changedFloat(previous.segmentSeconds, runtime.segmentSeconds) ||
            changedFloat(previous.hopSeconds, runtime.hopSeconds) ||
            previous.overlapMergeMethod != runtime.overlapMergeMethod ||
            previous.translationEnabled != runtime.translationEnabled ||
            previous.fromLang != runtime.fromLang || previous.toLang != runtime.toLang;

        if (requiresRestart) {
            if (startController()) {
                overlay.pushStatus("Runtime settings applied. Capture restarted.");
            } else {
                overlay.pushError("Capture restart failed after settings update.");
            }
            return;
        }

        overlay.pushStatus("UI settings applied.");
    };

    TrayController tray(&app,
                        &overlay,
                        [&runtime]() {
                            return runtime;
                        },
                        applySettings,
                        []() {
                            return listLoopbackDeviceEntries();
                        },
                        []() {
                            return listMixerAppSessionEntries();
                        });
    Q_UNUSED(tray);

    overlay.show();
    startController();

    if (parser.isSet(smokeSwitchTestOption)) {
        const auto settle = [&]() {
            for (int i = 0; i < 16; ++i) {
                QCoreApplication::processEvents();
                QThread::msleep(60);
            }
        };

        auto chooseAppTarget = []() {
            const QList<SourceDeviceEntry> running = listMixerAppSessionEntries();
            for (const SourceDeviceEntry &entry : running) {
                const QString lowered = entry.id.trimmed().toLower();
                if (lowered.isEmpty()) {
                    continue;
                }
                return entry.id;
            }
            return running.isEmpty() ? QString{} : running.front().id;
        };

        settle();

        const QString appTarget = chooseAppTarget();
        if (!appTarget.isEmpty()) {
            RuntimeSettings next = runtime;
            next.sourceMode = "app";
            next.sourceApps = QStringList{appTarget};
            applySettings(next);
            settle();

            next = runtime;
            next.sourceMode = "loopback";
            next.sourceApps.clear();
            applySettings(next);
            settle();
        }

        RuntimeSettings next = runtime;
        next.segmentSeconds = std::clamp(runtime.segmentSeconds + 1.0F, 1.0F, 12.0F);
        next.hopSeconds = std::clamp(runtime.hopSeconds + 0.2F, 0.1F, std::max(0.1F, next.segmentSeconds - 0.1F));
        applySettings(next);
        settle();

        next = runtime;
        next.segmentSeconds = std::clamp(runtime.segmentSeconds - 1.0F, 1.0F, 12.0F);
        next.hopSeconds = std::clamp(runtime.hopSeconds - 0.2F, 0.1F, std::max(0.1F, next.segmentSeconds - 0.1F));
        applySettings(next);
        settle();

        next = runtime;
        next.translationEnabled = true;
        next.toLang = "zh";
        next.translationStyle = "stacked";
        applySettings(next);
        settle();

        next = runtime;
        next.translationEnabled = true;
        next.toLang = "ja";
        next.translationStyle = "translation-only";
        applySettings(next);
        settle();

        next = runtime;
        next.translationEnabled = false;
        applySettings(next);
        settle();

        stopController();
        return 0;
    }

    QObject::connect(&app, &QCoreApplication::aboutToQuit, [&]() {
        stopController();
    });

    return app.exec();
}
