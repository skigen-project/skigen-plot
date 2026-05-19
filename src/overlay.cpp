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
    layout->setContentsMargins(7, 7, 7, 7);
    layout->setSpacing(6);

    auto makeBtn = [&](const QString& text) {
        auto* btn = new QPushButton(text, this);
        btn->setFixedSize(32, 32);
        btn->setCursor(Qt::PointingHandCursor);
        btn->setFocusPolicy(Qt::NoFocus);
        btn->setFlat(true);
        return btn;
    };

    m_rotateBtn = makeBtn(QStringLiteral("\u27F3"));
    m_panBtn    = makeBtn(QStringLiteral("\u2725"));
    m_zoomBtn   = makeBtn(QStringLiteral("+"));
    m_resetBtn  = makeBtn(QStringLiteral("\u2302"));
    m_themeBtn  = makeBtn(QStringLiteral("\u263E"));
    m_saveBtn   = makeBtn(QStringLiteral("\u2193"));

    m_rotateBtn->setToolTip(QStringLiteral("Rotate"));
    m_panBtn->setToolTip(QStringLiteral("Pan"));
    m_zoomBtn->setToolTip(QStringLiteral("Zoom"));
    m_resetBtn->setToolTip(QStringLiteral("Reset view"));
    m_themeBtn->setToolTip(QStringLiteral("Toggle theme"));
    m_saveBtn->setToolTip(QStringLiteral("Save PNG"));

    layout->addWidget(m_rotateBtn);
    layout->addWidget(m_panBtn);
    layout->addWidget(m_zoomBtn);
    layout->addWidget(m_resetBtn);
    layout->addWidget(m_themeBtn);
    layout->addWidget(m_saveBtn);
    setLayout(layout);
    setFixedSize(236, 46);

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

    connect(m_rotateBtn, &QPushButton::clicked, this, [this] {
        setTool(InteractionTool::Rotate);
    });
    connect(m_panBtn, &QPushButton::clicked, this, [this] {
        setTool(InteractionTool::Pan);
    });
    connect(m_zoomBtn, &QPushButton::clicked, this, [this] {
        setTool(InteractionTool::Zoom);
    });
    connect(m_resetBtn, &QPushButton::clicked, this, [this] {
        m_plotView->resetCameraView();
        showTemporarily();
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

void PlotOverlay::setTool(InteractionTool tool) {
    m_plotView->setInteractionTool(tool);
    restyleButtons();
    showTemporarily();
}

void PlotOverlay::showTemporarily() {
    updateMode();
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
    restyleButtons();
    repaint();
}

void PlotOverlay::restyleButtons() {
    updateMode();
    QString bg = m_isDark
        ? QStringLiteral("rgba(255,255,255,30)")
        : QStringLiteral("rgba(0,0,0,25)");
    QString activeBg = m_isDark
        ? QStringLiteral("rgba(6,182,212,90)")
        : QStringLiteral("rgba(6,182,212,55)");
    QString fg = m_isDark
        ? QStringLiteral("#e0e0e8")
        : QStringLiteral("#334155");
    QString hover = m_isDark
        ? QStringLiteral("rgba(255,255,255,50)")
        : QStringLiteral("rgba(0,0,0,45)");

    auto style = [&](bool active) {
        return QStringLiteral(
        "QPushButton {"
        "  background: %1;"
        "  color: %2;"
        "  border: none;"
        "  border-radius: 6px;"
        "  font-size: 15px;"
        "  padding: 0px;"
        "  text-align: center;"
        "}"
        "QPushButton:hover {"
        "  background: %3;"
        "}").arg(active ? activeBg : bg, fg, hover);
    };

    auto tool = m_plotView->interactionTool();
    m_rotateBtn->setStyleSheet(style(tool == InteractionTool::Rotate));
    m_panBtn->setStyleSheet(style(tool == InteractionTool::Pan));
    m_zoomBtn->setStyleSheet(style(tool == InteractionTool::Zoom));
    m_resetBtn->setStyleSheet(style(false));
    m_themeBtn->setStyleSheet(style(false));
    m_saveBtn->setStyleSheet(style(false));
}

void PlotOverlay::updateMode() {
    bool is3D = m_plotView->is3DView();
    m_rotateBtn->setVisible(is3D);
    if (!is3D && m_plotView->interactionTool() == InteractionTool::Rotate)
        m_plotView->setInteractionTool(InteractionTool::Pan);

    int visibleCount = is3D ? 6 : 5;
    setFixedSize(14 + visibleCount * 32 + (visibleCount - 1) * 6, 46);
    if (auto* owner = parentWidget())
        move(owner->width() - width() - 12,
             owner->height() - height() - 12);
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
