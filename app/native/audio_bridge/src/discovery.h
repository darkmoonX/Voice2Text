#pragma once

#include <QList>
#include <QString>
#include <QStringList>

struct SourceDeviceEntry {
    QString id;
    QString label;
};

QList<SourceDeviceEntry> discoverMixerAppSessionEntries();
QStringList discoverMixerAppSessionProcessNames();
QList<SourceDeviceEntry> discoverLoopbackDeviceEntries();
QString discoverVirtualCableLoopbackDeviceId();
bool isVirtualCableName(const QString &name);
