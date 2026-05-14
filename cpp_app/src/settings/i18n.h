#pragma once

#include <QString>

namespace settings_i18n {

QString normalizeUiLanguage(const QString &rawLang);
QString modeLoopbackLabel(const QString &lang);
QString modeMicrophoneLabel(const QString &lang);
QString modeAppLabel(const QString &lang);
QString settingsTitle(const QString &lang);
QString selectAll(const QString &lang);
QString noSources(const QString &lang);
QString selectSource(const QString &lang);
QString selectLoopback(const QString &lang);
QString selectApps(const QString &lang);
QString modeLabel(const QString &lang);
QString uiLanguageLabel(const QString &lang);
QString sourceLabel(const QString &lang);
QString segmentLabel(const QString &lang);
QString hopLabel(const QString &lang);
QString mergeLabel(const QString &lang);
QString sourceLanguageLabel(const QString &lang);
QString translationLabel(const QString &lang);
QString translationStyleStacked(const QString &lang);
QString translationStyleTranslatedOnly(const QString &lang);
QString translationLanguageLabel(const QString &lang);
QString fontSizeLabel(const QString &lang);
QString opacityLabel(const QString &lang);
QString sourceColorLabel(const QString &lang);
QString translatedColorLabel(const QString &lang);
QString backgroundColorLabel(const QString &lang);
QString settingsApplied(const QString &lang);
QString microphoneDefaultSummary(const QString &lang);
QString loopbackDefaultSummary(const QString &lang);
QString loopbackMissingSummary(const QString &lang);
QString appNoneSummary(const QString &lang);
QString loopbackSelectedPrefix(const QString &lang);
QString appSelectedPrefix(const QString &lang);
QString pickSourceColorTitle(const QString &lang);
QString pickTranslatedColorTitle(const QString &lang);
QString pickBackgroundColorTitle(const QString &lang);

}  // namespace settings_i18n
