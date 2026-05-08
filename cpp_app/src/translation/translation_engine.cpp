#include "translation/translation_engine.h"

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QProcess>

namespace {
constexpr int kProbeTimeoutMs = 180000;
constexpr int kTranslateTimeoutMs = 12000;
constexpr int kPythonCheckTimeoutMs = 5000;
}  // namespace

QString TranslationEngine::normalizeLanguageCode(const QString &code) {
    const QString normalized = code.trimmed().toLower();
    if (normalized == "zh-hant" || normalized == "zh-hans") {
        return "zh";
    }
    if (normalized.isEmpty()) {
        return "auto";
    }
    return normalized;
}

QString TranslationEngine::findBridgeScript() {
    const QString appDir = QCoreApplication::applicationDirPath();
    const QString cwd = QDir::currentPath();

    QStringList roots;
    roots << appDir << QDir(appDir).absoluteFilePath("..") << QDir(appDir).absoluteFilePath("../..")
          << cwd << QDir(cwd).absoluteFilePath("..");
    roots.removeDuplicates();

    const QStringList relativeCandidates = {
        "tools/argos_translate_bridge.py",
        "../tools/argos_translate_bridge.py",
        "../../tools/argos_translate_bridge.py",
        "cpp_app/tools/argos_translate_bridge.py",
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

bool TranslationEngine::runProcess(const PythonCommand &command,
                                   const QStringList &args,
                                   const QByteArray &stdinData,
                                   int timeoutMs,
                                   QByteArray *stdoutData,
                                   QByteArray *stderrData,
                                   int *exitCode) {
    QProcess process;
    process.start(command.executable, command.baseArgs + args);
    if (!process.waitForStarted(timeoutMs)) {
        return false;
    }

    if (!stdinData.isEmpty()) {
        process.write(stdinData);
    }
    process.closeWriteChannel();

    if (!process.waitForFinished(timeoutMs)) {
        process.kill();
        process.waitForFinished();
        return false;
    }

    if (stdoutData != nullptr) {
        *stdoutData = process.readAllStandardOutput();
    }
    if (stderrData != nullptr) {
        *stderrData = process.readAllStandardError();
    }
    if (exitCode != nullptr) {
        *exitCode = process.exitCode();
    }

    return true;
}

TranslationEngine::TranslationEngine(bool enabled, QString fromLang, QString toLang)
    : enabled_(enabled), fromLang_(normalizeLanguageCode(fromLang)), toLang_(normalizeLanguageCode(toLang)) {
    if (!enabled_) {
        statusMessage_ = "Translation disabled by config.";
        return;
    }

    if (toLang_.isEmpty()) {
        toLang_ = "zh";
    }

    if (fromLang_ != "auto" && fromLang_ == toLang_) {
        statusMessage_ = "Translation source and target are identical.";
        return;
    }

    bridgeScript_ = findBridgeScript();
    if (bridgeScript_.isEmpty()) {
        statusMessage_ = "Translation helper script not found (tools/argos_translate_bridge.py).";
        return;
    }

    if (!resolvePythonCommand()) {
        statusMessage_ = "Python runtime not found for translation bridge.";
        return;
    }

    if (!probeBridge(true)) {
        return;
    }

    active_ = true;
}

bool TranslationEngine::resolvePythonCommand() {
    const QString envPython = qEnvironmentVariable("VOICE2TEXT_PYTHON").trimmed();

    QList<PythonCommand> candidates;
    if (!envPython.isEmpty()) {
        candidates.push_back(PythonCommand{envPython, {}});
    }
    candidates.push_back(PythonCommand{"python", {}});
    candidates.push_back(PythonCommand{"py", {"-3"}});

    for (const PythonCommand &candidate : candidates) {
        QByteArray stdoutData;
        QByteArray stderrData;
        int exitCode = -1;
        const bool ok = runProcess(candidate,
                                   {"-c", "import sys;print(sys.version_info.major)"},
                                   {},
                                   kPythonCheckTimeoutMs,
                                   &stdoutData,
                                   &stderrData,
                                   &exitCode);
        if (!ok || exitCode != 0) {
            continue;
        }

        const QString versionLine = QString::fromUtf8(stdoutData).trimmed();
        if (versionLine == "3") {
            pythonCommand_ = candidate;
            return true;
        }
    }

    return false;
}

bool TranslationEngine::probeBridge(bool autoInstall) {
    QByteArray stdoutData;
    QByteArray stderrData;
    int exitCode = -1;

    const QStringList args = {
        bridgeScript_,
        "--mode",
        "probe",
        "--from-code",
        fromLang_,
        "--to-code",
        toLang_,
        "--auto-install",
        autoInstall ? "1" : "0",
    };

    const bool ok = runProcess(
        pythonCommand_, args, {}, kProbeTimeoutMs, &stdoutData, &stderrData, &exitCode);
    if (!ok) {
        statusMessage_ = "Translation probe timed out or failed to start Python process.";
        return false;
    }

    const QJsonDocument doc = QJsonDocument::fromJson(stdoutData);
    if (!doc.isObject()) {
        const QString stderrText = QString::fromUtf8(stderrData).trimmed();
        statusMessage_ = stderrText.isEmpty() ? "Translation probe returned invalid response."
                                              : QString("Translation probe error: %1").arg(stderrText);
        return false;
    }

    const QJsonObject obj = doc.object();
    const bool active = obj.value("active").toBool(false);
    const QString message = obj.value("message").toString().trimmed();
    if (!message.isEmpty()) {
        statusMessage_ = message;
    } else {
        statusMessage_ = active ? "Translation active." : "Translation unavailable.";
    }

    return active && exitCode == 0;
}

bool TranslationEngine::isEnabled() const {
    return enabled_ && active_;
}

QString TranslationEngine::info() const {
    if (!enabled_) {
        return "Translation disabled.";
    }

    return statusMessage_.isEmpty() ? QString("Translation configured (%1->%2).")
                                          .arg(fromLang_, toLang_)
                                    : statusMessage_;
}

QString TranslationEngine::translate(const QString &source) const {
    if (!isEnabled()) {
        return {};
    }

    const QString trimmed = source.trimmed();
    if (trimmed.isEmpty()) {
        return {};
    }

    QByteArray stdoutData;
    QByteArray stderrData;
    int exitCode = -1;

    const QStringList args = {
        bridgeScript_,
        "--mode",
        "translate",
        "--from-code",
        fromLang_,
        "--to-code",
        toLang_,
        "--auto-install",
        "0",
    };

    const bool ok = runProcess(pythonCommand_,
                               args,
                               trimmed.toUtf8(),
                               kTranslateTimeoutMs,
                               &stdoutData,
                               &stderrData,
                               &exitCode);
    if (!ok || exitCode != 0) {
        return {};
    }

    const QJsonDocument doc = QJsonDocument::fromJson(stdoutData);
    if (!doc.isObject()) {
        return {};
    }

    const QString translated = doc.object().value("translated").toString().trimmed();
    return translated;
}