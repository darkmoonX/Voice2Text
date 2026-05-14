#pragma once

#include <functional>

#include <QIcon>
#include <QList>
#include <QObject>
#include <QSystemTrayIcon>

#include "settings_dialog.h"

class QAction;
class QApplication;
class QMenu;
class SubtitleOverlayWindow;

class TrayController : public QObject {
    Q_OBJECT

public:
    TrayController(QApplication *app,
                   SubtitleOverlayWindow *overlay,
                   std::function<RuntimeSettings()> currentSettingsProvider,
                   std::function<void(const RuntimeSettings &)> applySettings,
                   std::function<QList<SourceDeviceEntry>()> loopbackDeviceProvider,
                   std::function<QList<SourceDeviceEntry>()> appSessionProvider,
                   std::function<QStringList()> modelCandidatesProvider,
                   QObject *parent = nullptr);

private slots:
    void showOverlay();
    void hideOverlay();
    void openSettings();
    void onTrayActivated(QSystemTrayIcon::ActivationReason reason);

private:
    static QIcon buildIcon();

    QApplication *app_{nullptr};
    SubtitleOverlayWindow *overlay_{nullptr};
    std::function<RuntimeSettings()> currentSettingsProvider_;
    std::function<void(const RuntimeSettings &)> applySettings_;
    std::function<QList<SourceDeviceEntry>()> loopbackDeviceProvider_;
    std::function<QList<SourceDeviceEntry>()> appSessionProvider_;
    std::function<QStringList()> modelCandidatesProvider_;

    QSystemTrayIcon tray_;
    QMenu *menu_{nullptr};
    QAction *showAction_{nullptr};
    QAction *hideAction_{nullptr};
    QAction *settingsAction_{nullptr};
    QAction *exitAction_{nullptr};
};
