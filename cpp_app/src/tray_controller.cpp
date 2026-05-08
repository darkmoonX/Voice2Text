#include "tray_controller.h"

#include <utility>

#include <QAction>
#include <QApplication>
#include <QFont>
#include <QMenu>
#include <QPainter>
#include <QPen>
#include <QPixmap>

#include "settings_dialog.h"
#include "subtitle_overlay_window.h"

TrayController::TrayController(QApplication *app,
                               SubtitleOverlayWindow *overlay,
                               std::function<RuntimeSettings()> currentSettingsProvider,
                               std::function<void(const RuntimeSettings &)> applySettings,
                               std::function<QList<SourceDeviceEntry>()> loopbackDeviceProvider,
                               std::function<QList<SourceDeviceEntry>()> appSessionProvider,
                               QObject *parent)
        : QObject(parent),
            app_(app),
            overlay_(overlay),
            currentSettingsProvider_(std::move(currentSettingsProvider)),
            applySettings_(std::move(applySettings)),
            loopbackDeviceProvider_(std::move(loopbackDeviceProvider)),
            appSessionProvider_(std::move(appSessionProvider)),
            tray_(this) {
    if (app_ == nullptr || overlay_ == nullptr) {
        return;
    }

    const QIcon icon = buildIcon();
    app_->setWindowIcon(icon);
    overlay_->setWindowIcon(icon);

    if (!QSystemTrayIcon::isSystemTrayAvailable()) {
        overlay_->pushStatus("System tray unavailable on this desktop session.");
        return;
    }

    menu_ = new QMenu();
    showAction_ = menu_->addAction("顯示");
    hideAction_ = menu_->addAction("縮小");
    menu_->addSeparator();
    settingsAction_ = menu_->addAction("設定");
    menu_->addSeparator();
    exitAction_ = menu_->addAction("離開");

    connect(showAction_, &QAction::triggered, this, &TrayController::showOverlay);
    connect(hideAction_, &QAction::triggered, this, &TrayController::hideOverlay);
    connect(settingsAction_, &QAction::triggered, this, &TrayController::openSettings);
    connect(exitAction_, &QAction::triggered, app_, &QApplication::quit);

    tray_.setIcon(icon);
    tray_.setToolTip("Voice2Text");
    tray_.setContextMenu(menu_);
    connect(&tray_, &QSystemTrayIcon::activated, this, &TrayController::onTrayActivated);
    tray_.show();
}

void TrayController::showOverlay() {
    if (overlay_ == nullptr) {
        return;
    }
    overlay_->show();
    overlay_->raise();
    overlay_->activateWindow();
}

void TrayController::hideOverlay() {
    if (overlay_ == nullptr) {
        return;
    }
    overlay_->hide();
}

void TrayController::openSettings() {
    if (overlay_ == nullptr || !currentSettingsProvider_ || !applySettings_) {
        return;
    }

    RuntimeSettings initial = currentSettingsProvider_();
    const QList<SourceDeviceEntry> loopbackDevices =
        loopbackDeviceProvider_ ? loopbackDeviceProvider_() : QList<SourceDeviceEntry>{};
    const QList<SourceDeviceEntry> appSessions =
        appSessionProvider_ ? appSessionProvider_() : QList<SourceDeviceEntry>{};

    SettingsDialog dialog(initial, loopbackDevices, appSessions, overlay_);
    if (dialog.exec() != QDialog::Accepted) {
        return;
    }

    applySettings_(dialog.settings());
    tray_.showMessage("Voice2Text", "設定已套用。", QSystemTrayIcon::Information, 1800);
}

void TrayController::onTrayActivated(QSystemTrayIcon::ActivationReason reason) {
    if (reason != QSystemTrayIcon::Trigger) {
        return;
    }

    if (overlay_ == nullptr) {
        return;
    }

    if (overlay_->isVisible()) {
        hideOverlay();
    } else {
        showOverlay();
    }
}

QIcon TrayController::buildIcon() {
    QPixmap pixmap(64, 64);
    pixmap.fill(Qt::transparent);

    QPainter painter(&pixmap);
    painter.setRenderHint(QPainter::Antialiasing, true);

    painter.setBrush(QColor("#0A101A"));
    painter.setPen(QPen(QColor("#8CC8FF"), 3));
    painter.drawRoundedRect(6, 6, 52, 52, 12, 12);

    painter.setPen(QPen(QColor("#E8F2FF"), 3));
    painter.drawLine(16, 23, 48, 23);
    painter.drawLine(16, 32, 40, 32);
    painter.drawLine(16, 41, 46, 41);

    painter.setPen(QColor("#8CC8FF"));
    painter.setFont(QFont("Segoe UI", 8, QFont::Bold));
    painter.drawText(10, 58, "V2T");

    painter.end();
    return QIcon(pixmap);
}
