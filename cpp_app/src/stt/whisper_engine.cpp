#include "stt/whisper_engine.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <thread>

#include <QStringList>

#if defined(Q_OS_WIN)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#endif

#if HAS_WHISPER
#include <whisper.h>
#endif

WhisperEngine::WhisperEngine(QString modelPath,
                                                         QString sourceLanguage,
                                                         float segmentSeconds,
                                                         QString overlapMergeMethod,
                                                         float hopSeconds,
                                                         WhisperRuntimeParams runtimeParams,
                                                         QObject *parent)
        : QObject(parent),
            modelPath_(std::move(modelPath)),
            sourceLanguage_(sourceLanguage.trimmed().toLower()),
            overlapMergeMethod_(overlapMergeMethod.trimmed().toLower()),
            runtimeParams_(runtimeParams) {
        if (sourceLanguage_.isEmpty()) {
                sourceLanguage_ = "auto";
        }

        if (sourceLanguage_ == "zh-hant" || sourceLanguage_ == "zh-tw" || sourceLanguage_ == "zh-hk") {
            sourceLanguage_ = "zh";
            chineseScript_ = "hant";
        } else if (sourceLanguage_ == "zh-hans" || sourceLanguage_ == "zh-cn" || sourceLanguage_ == "zh-sg") {
            sourceLanguage_ = "zh";
            chineseScript_ = "hans";
        }

        if (overlapMergeMethod_.isEmpty()) {
            overlapMergeMethod_ = "replace-window";
        }

        if (overlapMergeMethod_ != "replace-window" && overlapMergeMethod_ != "suffix-overlap" &&
            overlapMergeMethod_ != "fuzzy-overlap" && overlapMergeMethod_ != "append-only") {
            overlapMergeMethod_ = "replace-window";
        }

        windowMs_ = std::max(500, static_cast<int>(segmentSeconds * 1000.0F));
        hopMs_ = std::max(100, static_cast<int>(hopSeconds * 1000.0F));
        hopMs_ = std::min(hopMs_, std::max(100, windowMs_ - 100));
}

WhisperEngine::~WhisperEngine() {
#if HAS_WHISPER
    if (ctx_ != nullptr) {
        whisper_free(ctx_);
        ctx_ = nullptr;
    }
#endif
}

bool WhisperEngine::initialize() {
#if HAS_WHISPER
    frozenTranscript_.clear();
    activeWindowTranscript_.clear();
    lastEmittedTranscript_.clear();
    pcmBuffer_.clear();

    if (modelPath_.trimmed().isEmpty()) {
        sttEnabled_ = false;
        emit statusRaised(
            "No model path configured. STT disabled. Use --model-path or WHISPER_MODEL_PATH to enable.");
        return true;
    }

    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = true;

    const QByteArray pathBytes = modelPath_.toUtf8();
    ctx_ = whisper_init_from_file_with_params(pathBytes.constData(), cparams);
    if (ctx_ == nullptr) {
        emit errorRaised("Failed to initialize whisper.cpp context from model path.");
        sttEnabled_ = false;
        return false;
    }

    sttEnabled_ = true;
    emit statusRaised("whisper.cpp initialized.");
    return true;
#else
    sttEnabled_ = false;
    emit statusRaised("whisper.cpp backend not linked. Running in demo mode.");
    return true;
#endif
}

void WhisperEngine::onPcmChunk(const QByteArray &pcm16Mono, int sampleRate) {
    if (!sttEnabled_) {
        return;
    }

    if (pcm16Mono.isEmpty()) {
        return;
    }

    inputSampleRate_ = std::max(1, sampleRate);

    const auto *src = reinterpret_cast<const int16_t *>(pcm16Mono.constData());
    const std::size_t count = static_cast<std::size_t>(pcm16Mono.size() / static_cast<int>(sizeof(int16_t)));
    pcmBuffer_.insert(pcmBuffer_.end(), src, src + count);

    processIfReady();
}

void WhisperEngine::processIfReady() {
    const std::size_t windowSamples =
        static_cast<std::size_t>(std::max(1, inputSampleRate_) * windowMs_ / 1000);
    const std::size_t hopSamples = std::min<std::size_t>(
        windowSamples,
        static_cast<std::size_t>(std::max(1, inputSampleRate_) * hopMs_ / 1000));

    while (pcmBuffer_.size() >= windowSamples) {
        std::vector<int16_t> chunk(pcmBuffer_.begin(), pcmBuffer_.begin() + windowSamples);
        const std::size_t eraseCount = std::min<std::size_t>(hopSamples, pcmBuffer_.size());
        pcmBuffer_.erase(pcmBuffer_.begin(), pcmBuffer_.begin() + eraseCount);

        std::vector<float> mono;
        mono.resize(chunk.size());
        for (std::size_t i = 0; i < chunk.size(); ++i) {
            mono[i] = static_cast<float>(chunk[i]) / 32768.0F;
        }

        const std::vector<float> audio16k = resampleTo16k(mono, inputSampleRate_);
        if (audio16k.empty()) {
            continue;
        }

#if HAS_WHISPER
        if (ctx_ == nullptr) {
            break;
        }

        const whisper_sampling_strategy samplingStrategy =
            (runtimeParams_.beamSize > 1) ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY;
        whisper_full_params params = whisper_full_default_params(samplingStrategy);
        params.print_progress = false;
        params.print_realtime = false;
        params.print_timestamps = false;
        params.translate = false;
        params.no_context = runtimeParams_.maxContext <= 0;
        params.single_segment = false;
        params.n_threads = std::max(1U, std::thread::hardware_concurrency() / 2U);
        params.temperature = runtimeParams_.temperature;
        params.entropy_thold = runtimeParams_.entropyThold;
        params.logprob_thold = runtimeParams_.logprobThold;
        params.no_speech_thold = runtimeParams_.noSpeechThold;
        params.greedy.best_of = std::max(1, runtimeParams_.bestOf);
        params.beam_search.beam_size = std::max(1, runtimeParams_.beamSize);
        if (runtimeParams_.maxContext > 0) {
            params.n_max_text_ctx = runtimeParams_.maxContext;
        }

        QByteArray languageBytes;
        if (sourceLanguage_ != "auto") {
            languageBytes = sourceLanguage_.toUtf8();
            params.language = languageBytes.constData();
        }

        const int rc = whisper_full(ctx_, params, audio16k.data(), static_cast<int>(audio16k.size()));
        if (rc != 0) {
            emit errorRaised("whisper_full returned non-zero.");
            continue;
        }

        const int segCount = whisper_full_n_segments(ctx_);
        QStringList parts;
        for (int i = 0; i < segCount; ++i) {
            const char *raw = whisper_full_get_segment_text(ctx_, i);
            if (raw == nullptr) {
                continue;
            }
            const QString piece = QString::fromUtf8(raw).trimmed();
            if (!piece.isEmpty()) {
                parts.push_back(piece);
            }
        }

        const QString merged = parts.join(" ").simplified();
        if (!merged.isEmpty()) {
            const QString normalized = normalizeChineseScript(merged);
            const QString rolling = mergeIncremental(normalized);
            if (!rolling.isEmpty()) {
                emit transcriptReady(rolling);
            }
        }
#else
        float rms = 0.0F;
        for (float v : audio16k) {
            rms += v * v;
        }
        rms = std::sqrt(rms / static_cast<float>(audio16k.size()));
        if (rms > 0.02F) {
            emit transcriptReady("[demo] Speech detected. Link whisper.cpp model to decode text.");
        }
#endif
    }
}

std::vector<float> WhisperEngine::resampleTo16k(const std::vector<float> &input, int sampleRate) const {
    constexpr int targetRate = 16000;
    if (input.empty()) {
        return {};
    }
    if (sampleRate == targetRate) {
        return input;
    }

    const std::size_t targetSize =
        static_cast<std::size_t>(static_cast<double>(input.size()) * targetRate / sampleRate);
    if (targetSize < 2) {
        return {};
    }

    std::vector<float> out;
    out.resize(targetSize);

    const double scale = static_cast<double>(input.size() - 1) / static_cast<double>(targetSize - 1);
    for (std::size_t i = 0; i < targetSize; ++i) {
        const double srcPos = static_cast<double>(i) * scale;
        const std::size_t idx = static_cast<std::size_t>(srcPos);
        const std::size_t next = std::min<std::size_t>(idx + 1, input.size() - 1);
        const float frac = static_cast<float>(srcPos - static_cast<double>(idx));
        out[i] = input[idx] + (input[next] - input[idx]) * frac;
    }

    return out;
}

QString WhisperEngine::mergeIncremental(const QString &incoming) {
    const QString cleaned = incoming.simplified();
    if (cleaned.isEmpty()) {
        return {};
    }

    if (activeWindowTranscript_.isEmpty()) {
        activeWindowTranscript_ = cleaned;
        lastEmittedTranscript_ = mergeByExactOverlap(frozenTranscript_, activeWindowTranscript_);
        return lastEmittedTranscript_;
    }

    if (overlapMergeMethod_ == "append-only") {
        const QString combinedPrev = mergeByExactOverlap(frozenTranscript_, activeWindowTranscript_);
        QString combined = mergeByExactOverlap(combinedPrev, cleaned);
        if (combined.size() > 1800) {
            combined = combined.right(1800);
        }
        if (combined == lastEmittedTranscript_) {
            return {};
        }
        frozenTranscript_.clear();
        activeWindowTranscript_ = combined;
        lastEmittedTranscript_ = combined;
        return combined;
    }

    const int lockChars =
        std::clamp(static_cast<int>(std::round(activeWindowTranscript_.size() * lockRatio())),
                   0,
                   static_cast<int>(activeWindowTranscript_.size()));

    const QString lockChunk = activeWindowTranscript_.left(lockChars).trimmed();
    const QString overlapTail = activeWindowTranscript_.mid(lockChars).trimmed();

    if (!lockChunk.isEmpty()) {
        frozenTranscript_ = mergeByExactOverlap(frozenTranscript_, lockChunk);
        if (frozenTranscript_.size() > 1800) {
            frozenTranscript_ = frozenTranscript_.right(1800);
        }
    }

    if (overlapMergeMethod_ == "suffix-overlap") {
        activeWindowTranscript_ = mergeByExactOverlap(overlapTail, cleaned);
    } else if (overlapMergeMethod_ == "fuzzy-overlap") {
        activeWindowTranscript_ = mergeByFuzzyOverlap(overlapTail, cleaned);
    } else {
        activeWindowTranscript_ = mergeReplaceWindow(overlapTail, cleaned, lockRatio());
    }

    if (activeWindowTranscript_.size() > 1200) {
        activeWindowTranscript_ = activeWindowTranscript_.right(1200);
    }

    QString combined = mergeByExactOverlap(frozenTranscript_, activeWindowTranscript_);
    if (combined.size() > 1800) {
        combined = combined.right(1800);
    }

    if (combined == lastEmittedTranscript_) {
        return {};
    }

    lastEmittedTranscript_ = combined;
    return combined;
}

float WhisperEngine::lockRatio() const {
    if (windowMs_ <= 0) {
        return 0.2F;
    }

    const float ratio = static_cast<float>(hopMs_) / static_cast<float>(windowMs_);
    return std::clamp(ratio, 0.05F, 0.95F);
}

QString WhisperEngine::mergeReplaceWindow(const QString &overlapTail,
                                          const QString &incoming,
                                          float lockRatio) const {
    const QString previous = overlapTail.simplified();
    const QString latest = incoming.simplified();

    if (previous.isEmpty()) {
        return latest;
    }
    if (latest.isEmpty()) {
        return previous;
    }

    const float preserveRatio = std::clamp(lockRatio * 2.0F, 0.22F, 0.55F);
    int keepChars = static_cast<int>(std::round(previous.size() * preserveRatio));
    keepChars = std::clamp(keepChars, 10, static_cast<int>(previous.size()));

    const QString stableHead = previous.left(keepChars).trimmed();
    const QString mutableTail = previous.mid(keepChars).trimmed();

    QString reconciledTail = mergeByFuzzyOverlap(mutableTail, latest);
    if (reconciledTail == latest && !mutableTail.isEmpty()) {
        // If overlap is weak, avoid trusting unstable leading tokens from latest chunk.
        const int skipChars = std::max(4, static_cast<int>(std::round(latest.size() * 0.18F)));
        const QString conservative = latest.mid(skipChars).trimmed();
        if (!conservative.isEmpty()) {
            reconciledTail = mergeByExactOverlap(mutableTail, conservative);
        }
        if (reconciledTail.isEmpty()) {
            reconciledTail = mutableTail;
        }
    }

    const QString merged = mergeByExactOverlap(stableHead, reconciledTail);
    if (!merged.isEmpty()) {
        return merged;
    }

    return previous;
}

QString WhisperEngine::mergeByExactOverlap(const QString &base, const QString &incoming) const {
    const QString left = base.simplified();
    const QString right = incoming.simplified();

    if (left.isEmpty()) {
        return right;
    }
    if (right.isEmpty()) {
        return left;
    }

    const int overlap = overlapPrefixSuffix(left, right);
    if (overlap >= right.size()) {
        return left;
    }

    if (overlap > 0) {
        return (left + right.mid(overlap)).trimmed();
    }

    const int tailLen = std::max(16, static_cast<int>(right.size()) * 2);
    const QString tail = left.right(tailLen);
    if (tail.contains(right)) {
        return left;
    }

    QString separator;
    if (!left.isEmpty()) {
        const QChar last = left.at(left.size() - 1);
        if (QStringLiteral("。！？，,. ").contains(last)) {
            separator = "";
        } else {
            separator = " ";
        }
    }

    return (left + separator + right).trimmed();
}

QString WhisperEngine::mergeByFuzzyOverlap(const QString &base, const QString &incoming) const {
    const QString left = base.simplified();
    const QString right = incoming.simplified();

    if (left.isEmpty()) {
        return right;
    }
    if (right.isEmpty()) {
        return left;
    }

    const int exact = overlapPrefixSuffix(left, right);
    if (exact > 0) {
        return mergeByExactOverlap(left, right);
    }

    auto lcsSimilarity = [](const QString &a, const QString &b) -> float {
        const int n = a.size();
        const int m = b.size();
        if (n <= 0 || m <= 0) {
            return 0.0F;
        }

        std::vector<int> prev(static_cast<std::size_t>(m + 1), 0);
        std::vector<int> curr(static_cast<std::size_t>(m + 1), 0);

        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j <= m; ++j) {
                if (a.at(i - 1) == b.at(j - 1)) {
                    curr[static_cast<std::size_t>(j)] =
                        prev[static_cast<std::size_t>(j - 1)] + 1;
                } else {
                    curr[static_cast<std::size_t>(j)] =
                        std::max(prev[static_cast<std::size_t>(j)],
                                 curr[static_cast<std::size_t>(j - 1)]);
                }
            }
            std::swap(prev, curr);
            std::fill(curr.begin(), curr.end(), 0);
        }

        const int lcs = prev[static_cast<std::size_t>(m)];
        const int denom = std::max(n, m);
        if (denom <= 0) {
            return 0.0F;
        }
        return static_cast<float>(lcs) / static_cast<float>(denom);
    };

    const int maxLen =
        std::min(std::min(static_cast<int>(left.size()), static_cast<int>(right.size())), 120);
    const int minLen = std::min(6, maxLen);
    int bestSize = 0;
    float bestScore = 0.0F;

    for (int size = maxLen; size >= minLen; --size) {
        const QString tail = left.right(size);
        const QString head = right.left(size);
        const float score = lcsSimilarity(tail, head);

        if (score > bestScore) {
            bestScore = score;
            bestSize = size;
        }

        if (score >= 0.76F) {
            return (left + right.mid(size)).trimmed();
        }
    }

    if (bestSize >= 8 && bestScore >= 0.62F) {
        return (left + right.mid(bestSize)).trimmed();
    }

    return right;
}

int WhisperEngine::overlapPrefixSuffix(const QString &base, const QString &incoming) {
    const int maxLen =
        std::min(static_cast<int>(base.size()), static_cast<int>(incoming.size()));
    for (int size = maxLen; size > 0; --size) {
        if (base.endsWith(incoming.left(size))) {
            return size;
        }
    }
    return 0;
}

QString WhisperEngine::normalizeChineseScript(const QString &text) const {
    if (text.isEmpty()) {
        return text;
    }

#if defined(Q_OS_WIN)
    DWORD mapFlag = 0;
    if (chineseScript_ == "hant") {
        mapFlag = LCMAP_TRADITIONAL_CHINESE;
    } else if (chineseScript_ == "hans") {
        mapFlag = LCMAP_SIMPLIFIED_CHINESE;
    }

    if (mapFlag == 0) {
        return text;
    }

    const std::wstring input = text.toStdWString();
    if (input.empty()) {
        return text;
    }

    const int required = LCMapStringEx(LOCALE_NAME_SYSTEM_DEFAULT,
                                       mapFlag,
                                       input.c_str(),
                                       -1,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr,
                                       0);
    if (required <= 1) {
        return text;
    }

    std::wstring output(static_cast<std::size_t>(required), L'\0');
    const int written = LCMapStringEx(LOCALE_NAME_SYSTEM_DEFAULT,
                                      mapFlag,
                                      input.c_str(),
                                      -1,
                                      output.data(),
                                      required,
                                      nullptr,
                                      nullptr,
                                      0);
    if (written <= 1) {
        return text;
    }

    output.resize(static_cast<std::size_t>(written - 1));
    return QString::fromStdWString(output);
#else
    return text;
#endif
}