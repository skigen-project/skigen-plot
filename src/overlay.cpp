#include "overlay.h"

#include <skigen/plot/plotview.h>

#include <QFileDialog>
#include <QHBoxLayout>
#include <QPainter>
#include <QPropertyAnimation>
#include <QPushButton>
#include <QTimer>

namespace Skigen::Plot {

PlotOverlay::PlotOverlay(PlotView* parent)
    : QWidget(parent)
    , m_plotView(parent)
{
    setAttribute(Qt::WA_TransparentForMouseEvents, false);
    setMouseTracking(true);

    auto* layout = new QHBoxLayout(this);
    layout->setContentsMargins(6, 4, 6, 4);
    layout->setSpacing(4);

    auto makeBtn = [&](const QString& text) {
        auto* btn = new QPushButton(text, this);
        btn->setFixedSize(28, 28);
        btn->setCursor(Qt::PointingHandCursor);
        btn->setFocusPolicy(Qt::NoFocus);
        return btn;
    };

    m_themeBtn = makeBtn(QStringLiteral("\u263E"));
    m_saveBtn  = makeBtn(QStringLiteral("\u2B07"));

    layout->addWidget(m_themeBtn);
    layout->addWidget(m_saveBtn);
    setLayout(layout);
    setFixedSize(sizeHint());

    m_fadeAnim = new QPropertyAnimation(this, "overlayOpacity", this);
    m_fadeAnim->setDuration(200);

    m_hideTimer = new QTimer(this);
    m_hideTimer->setSingleShot(true);
    m_hideTimer->setInterval(2000);
    connect(m_hideTimer, &QTimer::timeout, this, [this] {
        m_fadeAnim->stop();
        m_fadeAnim->setStartValue(m_opacity);
        m_fadeAnim->setEndValue(0.0);
        m_fadeAnim->start();
    });

    connect(m_themeBtn, &QPushButton::clicked, this, [this] {
        bool wasDark = m_isDark;
        m_plotView->setTheme(wasDark ? Theme::light() : Theme::dark());
        updateThemeColors(!wasDark);
        showTemporarily();
    });

    connect(m_saveBtn, &QPushButton::clicked, this, [this] {
        QString path = QFileDialog::getSaveFileName(
            m_plotView,
            QStringLiteral("Save Plot"),
            QStringLiteral("plot.png"),
            QStringLiteral("PNG Images (*.png)"));
        if (!path.isEmpty())
            m_plotView->savePng(path, m_plotView->width() * 2, m_plotView->height() * 2);
    });

    updateThemeColors(true);
    setOverlayOpacity(0.0);
}

void PlotOverlay::showTemporarily() {
    m_hideTimer->stop();
    m_fadeAnim->stop();
    m_fadeAnim->setStartValue(m_opacity);
    m_fadeAnim->setEndValue(1.0);
    m_fadeAnim->start();
    m_hideTimer->start();
}

void PlotOverlay::updateThemeColors(bool isDark) {
    m_isDark = isDark;
    m_themeBtn->setText(isDark ? QStringLiteral("\u2600") : QStringLiteral("\u263E"));

    QString bg = isDark
        ? QStringLiteral("rgba(255,255,255,30)")
        : QStringLiteral("rgba(0,0,0,25)");
    QString fg = isDark
        ? QStringLiteral("#e0e0e8")
        : QStringLiteral("#334155");
    QString hover = isDark
        ? QStringLiteral("rgba(255,255,255,50)")
        : QStringLiteral("rgba(0,0,0,45)");

    QString ss = QStringLiteral(
        "QPushButton {"
        "  background: %1;"
        "  color: %2;"
        "  border: none;"
        "  border-radius: 6px;"
        "  font-size: 14px;"
        "}"
        "QPushButton:hover {"
        "  background: %3;"
        "}").arg(bg, fg, hover);

    m_themeBtn->setStyleSheet(ss);
    m_saveBtn->setStyleSheet(ss);
    repaint();
}

void PlotOverlay::setOverlayOpacity(qreal opacity) {
    m_opacity = opacity;
    setVisible(m_opacity > 0.01);
    repaint();
}

void PlotOverlay::paintEvent(QPaintEvent*) {
    if (m_opacity < 0.01)
        return;

    QPainter p(this);
    p.setOpacity(m_opacity);
    p.setRenderHint(QPainter::Antialiasing);

    QColor bg = m_isDark ? QColor(20, 20, 35, 140) : QColor(240, 242, 248, 180);
    p.setBrush(bg);
    p.setPen(Qt::NoPen);
    p.drawRoundedRect(rect(), 10, 10);
}

} // namespace Skigen::Plot
