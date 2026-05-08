#pragma once

#include <QString>
#include <QStringList>

class TranslationEngine {
public:
    TranslationEngine(bool enabled, QString fromLang, QString toLang);

    bool isEnabled() const;
    QString info() const;
    QString translate(const QString &source) const;

private:
    struct PythonCommand {
        QString executable;
        QStringList baseArgs;
    };

    static QString normalizeLanguageCode(const QString &code);
    static QString findBridgeScript();
    static bool runProcess(const PythonCommand &command,
                           const QStringList &args,
                           const QByteArray &stdinData,
                           int timeoutMs,
                           QByteArray *stdoutData,
                           QByteArray *stderrData,
                           int *exitCode);

    bool resolvePythonCommand();
    bool probeBridge(bool autoInstall);

    bool enabled_{false};
    bool active_{false};
    QString fromLang_;
    QString toLang_;
    QString statusMessage_;
    QString bridgeScript_;
    PythonCommand pythonCommand_;
};