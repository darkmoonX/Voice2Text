#pragma once

#include <QObject>
#include <QString>

class QFile;
class QMutex;

class RuntimeLogger : public QObject {
    Q_OBJECT

public:
    explicit RuntimeLogger(const QString &logDir = "logs", QObject *parent = nullptr);
    ~RuntimeLogger() override;

public slots:
    void info(const QString &message);
    void error(const QString &message);

private:
    void write(const QString &level, const QString &message);

    QFile *file_{nullptr};
    QMutex *mutex_{nullptr};
};