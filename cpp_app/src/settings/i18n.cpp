#include "settings/i18n.h"

namespace settings_i18n {

QString normalizeUiLanguage(const QString &rawLang) {
    const QString lang = rawLang.trimmed().toLower();
    return lang == "en" ? "en" : "zh";
}

static bool isEn(const QString &lang) { return normalizeUiLanguage(lang) == "en"; }

QString modeLoopbackLabel(const QString &lang) { return isEn(lang) ? "Loopback" : "系統輸出"; }
QString modeMicrophoneLabel(const QString &lang) { return isEn(lang) ? "Microphone" : "麥克風"; }
QString modeAppLabel(const QString &lang) { return isEn(lang) ? "App Session" : "App Session"; }
QString settingsTitle(const QString &lang) { return isEn(lang) ? "Voice2Text Settings" : "Voice2Text 設定"; }
QString selectAll(const QString &lang) { return isEn(lang) ? "Select All" : "全選"; }
QString noSources(const QString &lang) { return isEn(lang) ? "(No available sources)" : "(沒有可用來源)"; }
QString selectSource(const QString &lang) { return isEn(lang) ? "Select Source" : "選擇來源"; }
QString selectLoopback(const QString &lang) { return isEn(lang) ? "Select Loopback Device" : "選擇 Loopback 裝置"; }
QString selectApps(const QString &lang) { return isEn(lang) ? "Select App Sessions" : "選擇 App Session"; }
QString modeLabel(const QString &lang) { return isEn(lang) ? "Source Mode" : "來源模式"; }
QString uiLanguageLabel(const QString &lang) { return isEn(lang) ? "UI Language" : "介面語言"; }
QString sourceLabel(const QString &lang) { return isEn(lang) ? "Source" : "來源"; }
QString segmentLabel(const QString &lang) { return isEn(lang) ? "Segment Seconds" : "片段秒數"; }
QString hopLabel(const QString &lang) { return isEn(lang) ? "Hop Seconds" : "滑動步進秒數"; }
QString mergeLabel(const QString &lang) { return isEn(lang) ? "Merge Method" : "重疊整合方法"; }
QString sourceLanguageLabel(const QString &lang) { return isEn(lang) ? "Source Language" : "偵測語言"; }
QString translationLabel(const QString &lang) { return isEn(lang) ? "Translation" : "翻譯"; }
QString translationStyleStacked(const QString &lang) { return isEn(lang) ? "Stacked" : "上下分行"; }
QString translationStyleTranslatedOnly(const QString &lang) { return isEn(lang) ? "Translated only" : "僅翻譯"; }
QString translationLanguageLabel(const QString &lang) { return isEn(lang) ? "Translation Target" : "翻譯語言"; }
QString fontSizeLabel(const QString &lang) { return isEn(lang) ? "Font Size" : "字體大小"; }
QString opacityLabel(const QString &lang) { return isEn(lang) ? "Opacity" : "透明度"; }
QString sourceColorLabel(const QString &lang) { return isEn(lang) ? "Source Text Color" : "原文顏色"; }
QString translatedColorLabel(const QString &lang) { return isEn(lang) ? "Translated Text Color" : "譯文顏色"; }
QString backgroundColorLabel(const QString &lang) { return isEn(lang) ? "Background Color" : "背景顏色"; }
QString settingsApplied(const QString &lang) { return isEn(lang) ? "Settings applied." : "設定已套用。"; }
QString microphoneDefaultSummary(const QString &lang) { return isEn(lang) ? "Using default microphone source." : "使用預設麥克風來源。"; }
QString loopbackDefaultSummary(const QString &lang) { return isEn(lang) ? "Using default loopback source." : "使用預設 loopback 裝置。"; }
QString loopbackMissingSummary(const QString &lang) { return isEn(lang) ? "Selected loopback device is unavailable; default source will be used." : "所選 loopback 裝置不存在，將使用預設來源。"; }
QString appNoneSummary(const QString &lang) { return isEn(lang) ? "No app sources selected." : "尚未選擇 App 來源"; }
QString loopbackSelectedPrefix(const QString &lang) { return isEn(lang) ? "Selected:" : "已選擇:"; }
QString appSelectedPrefix(const QString &lang) { return isEn(lang) ? "Selected apps:" : "已選擇 App:"; }
QString pickSourceColorTitle(const QString &lang) { return isEn(lang) ? "Pick source text color" : "選擇原文顏色"; }
QString pickTranslatedColorTitle(const QString &lang) { return isEn(lang) ? "Pick translated text color" : "選擇譯文顏色"; }
QString pickBackgroundColorTitle(const QString &lang) { return isEn(lang) ? "Pick background color" : "選擇背景顏色"; }

}  // namespace settings_i18n
