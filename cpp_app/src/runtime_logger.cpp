#include "runtime_logger.h"

#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QMutex>
#include <QMutexLocker>
#include <QTextStream>

RuntimeLogger::RuntimeLogger(const QString &logDir, QObject *parent) : QObject(parent) {
    QDir dir;
    dir.mkpath(logDir);

    file_ = new QFile(logDir + "/voice2text_cpp.log", this);
    file_->open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text);

    mutex_ = new QMutex();
}

RuntimeLogger::~RuntimeLogger() {
    if (file_ != nullptr && file_->isOpen()) {
        file_->flush();
        file_->close();
    }
    delete mutex_;
    mutex_ = nullptr;
}

void RuntimeLogger::info(const QString &message) {
    write("INFO", message);
}

void RuntimeLogger::error(const QString &message) {
    write("ERROR", message);
}

void RuntimeLogger::write(const QString &level, const QString &message) {
    if (file_ == nullptr || !file_->isOpen() || mutex_ == nullptr) {
        return;
    }

    QMutexLocker lock(mutex_);
    QTextStream stream(file_);
    stream << QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss")
           << " | " << level << " | " << message << '\n';
    stream.flush();
}