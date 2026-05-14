#include "audio/discovery.h"

#include <algorithm>
#include <memory>
#include <new>

#include <QSet>

#ifdef Q_OS_WIN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <audiopolicy.h>
#include <mmdeviceapi.h>
#include <functiondiscoverykeys_devpkey.h>
#endif

#ifdef Q_OS_WIN
template <typename T>
void safeRelease(T **ptr) {
    if (ptr != nullptr && *ptr != nullptr) {
        (*ptr)->Release();
        *ptr = nullptr;
    }
}
#endif

QString processNameFromPid(DWORD pid) {
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
    if (pathBuffer != nullptr && QueryFullProcessImageNameW(process, 0, pathBuffer.get(), &length)) {
        const QString fullPath = QString::fromWCharArray(pathBuffer.get(), static_cast<int>(length));
        const int slash = std::max(fullPath.lastIndexOf('\\'), fullPath.lastIndexOf('/'));
        name = (slash >= 0 ? fullPath.mid(slash + 1) : fullPath).trimmed();
    }
    CloseHandle(process);
    return name;
#else
    return {};
#endif
}

bool isVirtualCableKeywordMatch(const QString &name) {
    const QString lowered = name.trimmed().toLower();
    static const QStringList keywords = {"vb cable", "vb-cable", "vb-audio", "virtual audio cable",
                                         "virtual cable", "cable input", "cable output", "voicemeeter"};
    for (const QString &kw : keywords) {
        if (lowered.contains(kw)) {
            return true;
        }
    }
    return false;
}

QList<SourceDeviceEntry> discoverMixerAppSessionEntries() {
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

    hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, __uuidof(IMMDeviceEnumerator),
                          reinterpret_cast<void **>(&enumerator));
    if (FAILED(hr)) {
        goto cleanup;
    }
    hr = enumerator->GetDefaultAudioEndpoint(eRender, eConsole, &device);
    if (FAILED(hr)) {
        goto cleanup;
    }
    hr = device->Activate(__uuidof(IAudioSessionManager2), CLSCTX_ALL, nullptr,
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

QStringList discoverMixerAppSessionProcessNames() {
    QStringList names;
    const QList<SourceDeviceEntry> entries = discoverMixerAppSessionEntries();
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

QList<SourceDeviceEntry> discoverLoopbackDeviceEntries() {
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
    hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, __uuidof(IMMDeviceEnumerator),
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

QString discoverVirtualCableLoopbackDeviceId() {
    const QList<SourceDeviceEntry> devices = discoverLoopbackDeviceEntries();
    for (const SourceDeviceEntry &dev : devices) {
        if (isVirtualCableKeywordMatch(dev.label) || isVirtualCableKeywordMatch(dev.id)) {
            return dev.id;
        }
    }
    return {};
}

bool isVirtualCableName(const QString &name) {
    return isVirtualCableKeywordMatch(name);
}
