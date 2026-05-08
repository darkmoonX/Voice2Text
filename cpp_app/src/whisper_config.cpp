#include "whisper_config.h"

#include <algorithm>

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonValue>

namespace {

QJsonValue pickValue(const QJsonObject &object, const QStringList &keys) {
    for (const QString &key : keys) {
        if (object.contains(key)) {
            return object.value(key);
        }
    }
    return {};
}

bool parseIntValue(const QJsonObject &object,
                   const QStringList &keys,
                   int *outValue,
                   int minValue,
                   int maxValue) {
    if (outValue == nullptr) {
        return false;
    }

    const QJsonValue value = pickValue(object, keys);
    if (value.isUndefined() || value.isNull()) {
        return false;
    }

    bool ok = false;
    int parsed = 0;
    if (value.isDouble()) {
        parsed = value.toInt();
        ok = true;
    } else if (value.isString()) {
        parsed = value.toString().trimmed().toInt(&ok);
    }

    if (!ok) {
        return false;
    }

    *outValue = std::clamp(parsed, minValue, maxValue);
    return true;
}

bool parseFloatValue(const QJsonObject &object,
                     const QStringList &keys,
                     float *outValue,
                     float minValue,
                     float maxValue) {
    if (outValue == nullptr) {
        return false;
    }

    const QJsonValue value = pickValue(object, keys);
    if (value.isUndefined() || value.isNull()) {
        return false;
    }

    bool ok = false;
    float parsed = 0.0F;
    if (value.isDouble()) {
        parsed = static_cast<float>(value.toDouble());
        ok = true;
    } else if (value.isString()) {
        parsed = value.toString().trimmed().toFloat(&ok);
    }

    if (!ok) {
        return false;
    }

    *outValue = std::clamp(parsed, minValue, maxValue);
    return true;
}

QJsonObject extractConfigObject(const QJsonDocument &document) {
    if (!document.isObject()) {
        return {};
    }

    const QJsonObject root = document.object();
    if (root.contains("whisper") && root.value("whisper").isObject()) {
        return root.value("whisper").toObject();
    }

    return root;
}

} // namespace

QString resolveWhisperConfigPath() {
    const QString appDir = QCoreApplication::applicationDirPath();
    const QString cwd = QDir::currentPath();

    QStringList roots;
    roots << appDir << QDir(appDir).absoluteFilePath("..") << QDir(appDir).absoluteFilePath("../..")
          << cwd << QDir(cwd).absoluteFilePath("..");
    roots.removeDuplicates();

    const QStringList relativeCandidates = {
        "src/whisper_config.json",
        "../src/whisper_config.json",
        "../../src/whisper_config.json",
        "cpp_app/src/whisper_config.json",
    };

    for (const QString &root : roots) {
        const QDir rootDir(root);
        for (const QString &relativePath : relativeCandidates) {
            const QString candidate = QFileInfo(rootDir.absoluteFilePath(relativePath)).absoluteFilePath();
            if (QFileInfo::exists(candidate)) {
                return candidate;
            }
        }
    }

    return {};
}

WhisperRuntimeParams loadWhisperRuntimeParams(const QString &path,
                                              QStringList *warnings) {
    WhisperRuntimeParams params;

    if (path.trimmed().isEmpty()) {
        if (warnings != nullptr) {
            warnings->push_back("Whisper config not found. Using built-in defaults.");
        }
        return params;
    }

    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        if (warnings != nullptr) {
            warnings->push_back(QString("Unable to open whisper config: %1").arg(path));
        }
        return params;
    }

    QJsonParseError error;
    const QJsonDocument doc = QJsonDocument::fromJson(file.readAll(), &error);
    file.close();

    if (error.error != QJsonParseError::NoError) {
        if (warnings != nullptr) {
            warnings->push_back(
                QString("Whisper config parse error (%1). Using defaults.").arg(error.errorString()));
        }
        return params;
    }

    const QJsonObject object = extractConfigObject(doc);
    if (object.isEmpty()) {
        if (warnings != nullptr) {
            warnings->push_back("Whisper config is empty. Using defaults.");
        }
        return params;
    }

    parseIntValue(object,
                  {"max-context", "max_context", "mc", "-mc"},
                  &params.maxContext,
                  0,
                  8192);
    parseFloatValue(object,
                    {"entropy-thold", "entropy_thold"},
                    &params.entropyThold,
                    0.0F,
                    10.0F);
    parseFloatValue(object,
                    {"logprob-thold", "logprob_thold"},
                    &params.logprobThold,
                    -10.0F,
                    2.0F);
    parseFloatValue(object,
                    {"no-speech-thold", "no_speech_thold"},
                    &params.noSpeechThold,
                    0.0F,
                    1.0F);
    parseFloatValue(object,
                    {"temperature"},
                    &params.temperature,
                    0.0F,
                    2.0F);
    parseIntValue(object,
                  {"beam-size", "beam_size"},
                  &params.beamSize,
                  1,
                  32);
    parseIntValue(object,
                  {"best-of", "best_of"},
                  &params.bestOf,
                  1,
                  32);

    if (params.bestOf < params.beamSize) {
        params.bestOf = params.beamSize;
    }

    return params;
}
