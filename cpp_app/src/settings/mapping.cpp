#include "settings/mapping.h"

#include <algorithm>
#include "settings/i18n.h"

RuntimeSettings buildRuntimeSettingsFromPayload(const SettingsInputPayload &payload) {
    RuntimeSettings out;
    out.uiLanguage = settings_i18n::normalizeUiLanguage(payload.uiLanguage);
    out.sourceMode = payload.sourceMode.trimmed().toLower();
    out.loopbackDeviceId = payload.loopbackDeviceId.trimmed();
    if (out.sourceMode == "app") {
        out.sourceApps = payload.appNames;
    }

    out.sourceLanguage = payload.sourceLanguage.trimmed().toLower();
    out.modelPath = payload.modelPath.trimmed();
    if (out.sourceLanguage.isEmpty()) {
        out.sourceLanguage = "auto";
    }

    out.segmentSeconds = std::clamp(payload.segmentSeconds, 1.0F, 12.0F);
    out.hopSeconds = std::clamp(payload.hopSeconds, 0.1F, out.segmentSeconds);
    out.overlapMergeMethod = normalizeOverlapMergeMethod(payload.overlapMergeMethod);
    out.vadEnabled = payload.vadEnabled;
    out.vadAdaptiveEnabled = payload.vadAdaptiveEnabled;
    out.vadRmsThreshold = std::clamp(payload.vadRmsThreshold, 0.001F, 0.2F);

    out.translationEnabled = payload.translationEnabled;
    out.toLang = payload.translationTo.trimmed().toLower();
    if (out.toLang.isEmpty()) {
        out.toLang = "zh";
    }

    if (out.sourceLanguage == "auto") {
        out.fromLang = "auto";
    } else if (out.sourceLanguage == "zh-hant" || out.sourceLanguage == "zh-hans") {
        out.fromLang = "zh";
    } else {
        out.fromLang = out.sourceLanguage;
    }

    out.translationStyle = payload.translationStyle == "translation-only" ? "translation-only" : "stacked";
    out.fontSize = std::max(payload.fontSize, 10);
    out.opacity = std::clamp(payload.opacity, 0.2F, 1.0F);
    out.sourceColor = payload.sourceColor.isValid() ? payload.sourceColor : QColor("#F0F2F5");
    out.translatedColor = payload.translatedColor.isValid() ? payload.translatedColor : QColor("#FFD98A");
    out.backgroundColor = payload.backgroundColor.isValid() ? payload.backgroundColor : QColor("#0A101A");
    return out;
}
