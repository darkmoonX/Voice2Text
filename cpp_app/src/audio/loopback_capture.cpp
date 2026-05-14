#include "audio/loopback_capture.h"
#include "audio/discovery.h"

#include <Windows.h>
#include <audioclient.h>
#include <mmdeviceapi.h>
#include <tlhelp32.h>

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

#if VOICE2TEXT_HAS_PROCESS_LOOPBACK_HEADER && defined(VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK) && \
    defined(AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK)
#define VOICE2TEXT_HAS_PROCESS_LOOPBACK 1
#else
#define VOICE2TEXT_HAS_PROCESS_LOOPBACK 0
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <new>
#include <utility>
#include <vector>

namespace {

template <typename T>
void safeRelease(T **ptr) {
    if (ptr != nullptr && *ptr != nullptr) {
        (*ptr)->Release();
        *ptr = nullptr;
    }
}

QByteArray convertToMonoPcm16(const BYTE *data, UINT32 frames, const WAVEFORMATEX *format,
                              bool isSilent) {
    QByteArray out;
    out.resize(static_cast<int>(frames * sizeof(int16_t)));
    auto *dst = reinterpret_cast<int16_t *>(out.data());

    if (isSilent || data == nullptr || format == nullptr) {
        std::fill(dst, dst + frames, 0);
        return out;
    }

    const int channels = std::max<int>(1, static_cast<int>(format->nChannels));

    if (format->wBitsPerSample == 16) {
        const auto *src = reinterpret_cast<const int16_t *>(data);
        for (UINT32 i = 0; i < frames; ++i) {
            float sum = 0.0F;
            for (int ch = 0; ch < channels; ++ch) {
                const int idx = static_cast<int>(i) * channels + ch;
                sum += static_cast<float>(src[idx]) / 32768.0F;
            }
            const float mono = sum / static_cast<float>(channels);
            const int sample = std::clamp(static_cast<int>(std::lround(mono * 32767.0F)), -32768, 32767);
            dst[i] = static_cast<int16_t>(sample);
        }
        return out;
    }

    if (format->wBitsPerSample == 32) {
        // Most WASAPI loopback streams are float32 in shared mode.
        const auto *src = reinterpret_cast<const float *>(data);
        for (UINT32 i = 0; i < frames; ++i) {
            float sum = 0.0F;
            for (int ch = 0; ch < channels; ++ch) {
                const int idx = static_cast<int>(i) * channels + ch;
                sum += src[idx];
            }
            const float mono = std::clamp(sum / static_cast<float>(channels), -1.0F, 1.0F);
            const int sample = std::clamp(static_cast<int>(std::lround(mono * 32767.0F)), -32768, 32767);
            dst[i] = static_cast<int16_t>(sample);
        }
        return out;
    }

    std::fill(dst, dst + frames, 0);
    return out;
}

QString normalizeProcessName(const QString &name) {
    return name.trimmed().toLower();
}

bool processNameMatchesTarget(const QString &processName, const QString &targetRaw) {
    const QString process = normalizeProcessName(processName);
    const QString target = normalizeProcessName(targetRaw);
    if (process.isEmpty() || target.isEmpty()) {
        return false;
    }

    if (process == target) {
        return true;
    }

    if (!target.endsWith(".exe") && process == target + ".exe") {
        return true;
    }

    if (target.endsWith(".exe") && process == target.left(target.size() - 4)) {
        return true;
    }

    return false;
}

DWORD findTargetProcessId(const QStringList &targets, QString *matchedNameOut) {
    if (matchedNameOut != nullptr) {
        matchedNameOut->clear();
    }

    if (targets.isEmpty()) {
        return 0;
    }

    const HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snapshot == INVALID_HANDLE_VALUE) {
        return 0;
    }

    struct ProcessInfo {
        QString name;
        DWORD pid{0};
    };

    std::vector<ProcessInfo> running;
    PROCESSENTRY32W entry{};
    entry.dwSize = sizeof(PROCESSENTRY32W);
    if (Process32FirstW(snapshot, &entry)) {
        do {
            const QString name = QString::fromWCharArray(entry.szExeFile).trimmed();
            if (!name.isEmpty() && entry.th32ProcessID != 0) {
                running.push_back({name, entry.th32ProcessID});
            }
        } while (Process32NextW(snapshot, &entry));
    }
    CloseHandle(snapshot);

    for (const QString &target : targets) {
        for (const ProcessInfo &proc : running) {
            if (!processNameMatchesTarget(proc.name, target)) {
                continue;
            }
            if (matchedNameOut != nullptr) {
                *matchedNameOut = proc.name;
            }
            return proc.pid;
        }
    }

    return 0;
}

#if VOICE2TEXT_HAS_PROCESS_LOOPBACK

class ProcessLoopbackActivationHandler : public IActivateAudioInterfaceCompletionHandler {
public:
    ProcessLoopbackActivationHandler() = default;

    STDMETHODIMP QueryInterface(REFIID iid, void **ppv) override {
        if (ppv == nullptr) {
            return E_POINTER;
        }

        if (iid == __uuidof(IUnknown) || iid == __uuidof(IActivateAudioInterfaceCompletionHandler)) {
            *ppv = static_cast<IActivateAudioInterfaceCompletionHandler *>(this);
            AddRef();
            return S_OK;
        }

        *ppv = nullptr;
        return E_NOINTERFACE;
    }

    STDMETHODIMP_(ULONG) AddRef() override {
        return static_cast<ULONG>(InterlockedIncrement(&refCount_));
    }

    STDMETHODIMP_(ULONG) Release() override {
        const ULONG count = static_cast<ULONG>(InterlockedDecrement(&refCount_));
        if (count == 0U) {
            delete this;
        }
        return count;
    }

    STDMETHODIMP ActivateCompleted(IActivateAudioInterfaceAsyncOperation *operation) override {
        if (operation == nullptr) {
            activateResult_ = E_POINTER;
            SetEvent(event_);
            return S_OK;
        }

        IUnknown *activated = nullptr;
        HRESULT hr = operation->GetActivateResult(&activateResult_, &activated);
        if (FAILED(hr)) {
            activateResult_ = hr;
        }

        if (SUCCEEDED(activateResult_) && activated != nullptr) {
            activateResult_ =
                activated->QueryInterface(__uuidof(IAudioClient), reinterpret_cast<void **>(&audioClient_));
        }

        if (activated != nullptr) {
            activated->Release();
        }

        SetEvent(event_);
        return S_OK;
    }

    HANDLE eventHandle() const {
        return event_;
    }

    HRESULT activateResult() const {
        return activateResult_;
    }

    IAudioClient *detachAudioClient() {
        IAudioClient *client = audioClient_;
        audioClient_ = nullptr;
        return client;
    }

private:
    ~ProcessLoopbackActivationHandler() override {
        safeRelease(&audioClient_);
        if (event_ != nullptr) {
            CloseHandle(event_);
            event_ = nullptr;
        }
    }

    volatile LONG refCount_{1};
    HANDLE event_{CreateEventW(nullptr, FALSE, FALSE, nullptr)};
    HRESULT activateResult_{E_FAIL};
    IAudioClient *audioClient_{nullptr};
};

#endif

bool activateProcessLoopbackClient(DWORD targetPid,
                                   IAudioClient **outAudioClient,
                                   QString *errorMessageOut) {
    if (outAudioClient == nullptr) {
        return false;
    }
    *outAudioClient = nullptr;

#if !VOICE2TEXT_HAS_PROCESS_LOOPBACK
    if (errorMessageOut != nullptr) {
        *errorMessageOut =
            "Process loopback activation APIs are unavailable in this build toolchain/Windows SDK.";
    }
    return false;
#else
    AUDIOCLIENT_ACTIVATION_PARAMS activationParams{};
    activationParams.ActivationType = AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK;
    activationParams.ProcessLoopbackParams.TargetProcessId = targetPid;
    activationParams.ProcessLoopbackParams.ProcessLoopbackMode =
        PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE;

    PROPVARIANT activateParams{};
    activateParams.vt = VT_BLOB;
    activateParams.blob.cbSize = sizeof(activationParams);
    activateParams.blob.pBlobData = reinterpret_cast<BYTE *>(&activationParams);

    auto *handler = new (std::nothrow) ProcessLoopbackActivationHandler();
    if (handler == nullptr) {
        if (errorMessageOut != nullptr) {
            *errorMessageOut = "Failed to allocate process loopback activation handler.";
        }
        return false;
    }

    IActivateAudioInterfaceAsyncOperation *asyncOperation = nullptr;
    const HRESULT startHr = ActivateAudioInterfaceAsync(
        VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK,
        __uuidof(IAudioClient),
        &activateParams,
        handler,
        &asyncOperation);
    if (FAILED(startHr)) {
        if (errorMessageOut != nullptr) {
            *errorMessageOut =
                QString("ActivateAudioInterfaceAsync failed (hr=0x%1)")
                    .arg(static_cast<quint32>(startHr), 8, 16, QLatin1Char('0'));
        }
        safeRelease(&asyncOperation);
        handler->Release();
        return false;
    }

    const DWORD waitRc = WaitForSingleObject(handler->eventHandle(), 5000);
    if (waitRc != WAIT_OBJECT_0) {
        if (errorMessageOut != nullptr) {
            *errorMessageOut = "Timed out waiting for process loopback activation.";
        }
        safeRelease(&asyncOperation);
        handler->Release();
        return false;
    }

    const HRESULT activateHr = handler->activateResult();
    if (FAILED(activateHr)) {
        if (errorMessageOut != nullptr) {
            *errorMessageOut =
                QString("Process loopback activation returned hr=0x%1")
                    .arg(static_cast<quint32>(activateHr), 8, 16, QLatin1Char('0'));
        }
        safeRelease(&asyncOperation);
        handler->Release();
        return false;
    }

    IAudioClient *client = handler->detachAudioClient();
    safeRelease(&asyncOperation);
    handler->Release();

    if (client == nullptr) {
        if (errorMessageOut != nullptr) {
            *errorMessageOut = "Process loopback activation returned null IAudioClient.";
        }
        return false;
    }

    *outAudioClient = client;
    return true;
#endif
}

} // namespace

LoopbackCapture::LoopbackCapture(CaptureSourceMode mode,
                                                                 QString targetDeviceId,
                                                                 QStringList targetAppNames,
                                                                 QObject *parent)
        : QObject(parent),
            mode_(mode),
            targetDeviceId_(std::move(targetDeviceId)),
            targetAppNames_(std::move(targetAppNames)) {}

LoopbackCapture::~LoopbackCapture() {
    stop();
}

bool LoopbackCapture::start() {
    if (running_.exchange(true)) {
        return true;
    }

    try {
        worker_ = std::thread(&LoopbackCapture::captureLoop, this);
    } catch (...) {
        running_.store(false);
        emit errorRaised("Failed to start loopback worker thread.");
        return false;
    }

    return true;
}

void LoopbackCapture::stop() {
    running_.store(false);
    if (worker_.joinable()) {
        worker_.join();
    }
}

void LoopbackCapture::captureLoop() {
    HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    const bool coInitialized = SUCCEEDED(hr) || hr == RPC_E_CHANGED_MODE;
    if (!coInitialized) {
        emit errorRaised("COM initialization failed for loopback capture.");
        running_.store(false);
        return;
    }

    IMMDeviceEnumerator *enumerator = nullptr;
    IMMDevice *device = nullptr;
    IAudioClient *audioClient = nullptr;
    IAudioCaptureClient *captureClient = nullptr;
    WAVEFORMATEX *mixFormat = nullptr;
    EDataFlow dataFlow = eRender;
    DWORD streamFlags = AUDCLNT_STREAMFLAGS_LOOPBACK;
    QString modeLabel = "loopback";
    const REFERENCE_TIME bufferDuration = 2000000;
    bool useDefaultEndpoint = true;
    QString requestedDeviceId = targetDeviceId_.trimmed();

    if (mode_ == CaptureSourceMode::App) {
        if (targetAppNames_.isEmpty()) {
            emit errorRaised("App mode requires at least one target process name.");
            goto cleanup;
        }

        QString matchedProcessName;
        const DWORD targetPid = findTargetProcessId(targetAppNames_, &matchedProcessName);
        if (targetPid == 0) {
            emit errorRaised(
                QString("No running process matches selected app source(s): %1")
                    .arg(targetAppNames_.join(", ")));
            goto cleanup;
        }

        QString activateError;
        if (!activateProcessLoopbackClient(targetPid, &audioClient, &activateError)) {
            emit statusRaised(
                QString("Process loopback unavailable for pid %1: %2")
                    .arg(static_cast<qulonglong>(targetPid))
                    .arg(activateError));
            if (requestedDeviceId.isEmpty()) {
                requestedDeviceId = discoverVirtualCableLoopbackDeviceId();
            }
            if (requestedDeviceId.isEmpty()) {
                emit errorRaised(
                    "App mode requires process loopback or VB-CABLE endpoint for strict isolation. Capture aborted.");
                goto cleanup;
            }
            emit statusRaised("Process loopback unavailable; fallback to VB-CABLE loopback endpoint.");
            modeLabel = "app-virtual-cable-loopback";
            useDefaultEndpoint = true;
        } else {
            useDefaultEndpoint = false;
            streamFlags = 0;
            modeLabel = "app-process-loopback";
            emit statusRaised(
                QString("App process loopback target: %1 (pid=%2)")
                    .arg(matchedProcessName)
                    .arg(static_cast<qulonglong>(targetPid)));

            if (targetAppNames_.size() > 1) {
                emit statusRaised(
                    "App mode currently captures one running target process (first match). "
                    "Start additional instances if you need multiple isolated captures.");
            }
        }
    }

    if (mode_ != CaptureSourceMode::App || useDefaultEndpoint) {
        hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL,
                              __uuidof(IMMDeviceEnumerator), reinterpret_cast<void **>(&enumerator));
        if (FAILED(hr)) {
            emit errorRaised("Unable to create MMDeviceEnumerator.");
            goto cleanup;
        }

        if (mode_ == CaptureSourceMode::Microphone) {
            dataFlow = eCapture;
            streamFlags = 0;
            modeLabel = "microphone";
        } else if (mode_ == CaptureSourceMode::App && modeLabel != "app-fallback-loopback") {
            modeLabel = "loopback";
        }

        const bool canUseConfiguredDevice =
            !requestedDeviceId.isEmpty() &&
            (mode_ == CaptureSourceMode::Loopback ||
             (mode_ == CaptureSourceMode::App && useDefaultEndpoint));
        if (canUseConfiguredDevice) {
            hr = enumerator->GetDevice(reinterpret_cast<LPCWSTR>(requestedDeviceId.utf16()), &device);
            if (FAILED(hr)) {
                if (mode_ == CaptureSourceMode::App) {
                    emit errorRaised("App mode VB-CABLE endpoint unavailable. Capture aborted.");
                    goto cleanup;
                }
                emit statusRaised("Selected output device unavailable, fallback to default endpoint.");
            } else if (mode_ == CaptureSourceMode::Loopback) {
                modeLabel = "loopback-selected";
            } else if (mode_ == CaptureSourceMode::App && useDefaultEndpoint) {
                modeLabel = "app-virtual-cable-loopback";
            }
        }

        if (device == nullptr) {
            if (mode_ == CaptureSourceMode::App) {
                emit errorRaised(
                    "App mode strict isolation refused default endpoint fallback. Configure VB-CABLE and retry.");
                goto cleanup;
            }
            hr = enumerator->GetDefaultAudioEndpoint(dataFlow, eConsole, &device);
            if (FAILED(hr)) {
                emit errorRaised("Unable to access default audio endpoint.");
                goto cleanup;
            }
        }

        hr = device->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr,
                              reinterpret_cast<void **>(&audioClient));
        if (FAILED(hr)) {
            emit errorRaised("Unable to activate IAudioClient.");
            goto cleanup;
        }
    }

    hr = audioClient->GetMixFormat(&mixFormat);
    if (FAILED(hr) || mixFormat == nullptr) {
        emit errorRaised("Unable to query mix format.");
        goto cleanup;
    }

    // 200 ms shared buffer for responsive subtitle updates.
    hr = audioClient->Initialize(
        AUDCLNT_SHAREMODE_SHARED,
        streamFlags,
        bufferDuration,
        0,
        mixFormat,
        nullptr);
    if (FAILED(hr)) {
        emit errorRaised("IAudioClient initialize failed.");
        goto cleanup;
    }

    hr = audioClient->GetService(__uuidof(IAudioCaptureClient),
                                 reinterpret_cast<void **>(&captureClient));
    if (FAILED(hr)) {
        emit errorRaised("Unable to get IAudioCaptureClient service.");
        goto cleanup;
    }

    hr = audioClient->Start();
    if (FAILED(hr)) {
        emit errorRaised("Unable to start loopback capture stream.");
        goto cleanup;
    }

    emit statusRaised(QString("%1 capture started @ %2 Hz, %3 ch")
                          .arg(modeLabel)
                          .arg(static_cast<int>(mixFormat->nSamplesPerSec))
                          .arg(static_cast<int>(mixFormat->nChannels)));

    while (running_.load()) {
        UINT32 packetLength = 0;
        hr = captureClient->GetNextPacketSize(&packetLength);
        if (FAILED(hr)) {
            emit errorRaised("GetNextPacketSize failed.");
            break;
        }

        if (packetLength == 0) {
            Sleep(10);
            continue;
        }

        while (packetLength > 0 && running_.load()) {
            BYTE *data = nullptr;
            UINT32 frames = 0;
            DWORD flags = 0;

            hr = captureClient->GetBuffer(&data, &frames, &flags, nullptr, nullptr);
            if (FAILED(hr)) {
                emit errorRaised("GetBuffer failed during loopback capture.");
                running_.store(false);
                break;
            }

            const bool isSilent = (flags & AUDCLNT_BUFFERFLAGS_SILENT) != 0;
            const QByteArray monoPcm = convertToMonoPcm16(data, frames, mixFormat, isSilent);

            emit pcmChunkReady(monoPcm, static_cast<int>(mixFormat->nSamplesPerSec));
            captureClient->ReleaseBuffer(frames);

            hr = captureClient->GetNextPacketSize(&packetLength);
            if (FAILED(hr)) {
                emit errorRaised("GetNextPacketSize failed while draining packets.");
                running_.store(false);
                break;
            }
        }
    }

    if (audioClient != nullptr) {
        audioClient->Stop();
    }

    emit statusRaised("Capture stopped.");

cleanup:
    safeRelease(&captureClient);
    safeRelease(&audioClient);
    safeRelease(&device);
    safeRelease(&enumerator);

    if (mixFormat != nullptr) {
        CoTaskMemFree(mixFormat);
    }

    if (coInitialized) {
        CoUninitialize();
    }

    running_.store(false);
}
