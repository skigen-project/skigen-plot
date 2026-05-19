#pragma once

#include <QWidget>

class QPropertyAnimation;
class QTimer;
class QPushButton;

namespace Skigen::Plot {

class PlotView;

class PlotOverlay : public QWidget {
    Q_OBJECT
    Q_PROPERTY(qreal overlayOpacity READ overlayOpacity WRITE setOverlayOpacity)

public:
    explicit PlotOverlay(PlotView* parent);

    void showTemporarily();
    void updateThemeColors(bool isDark);

    auto overlayOpacity() const -> qreal { return m_opacity; }
    void setOverlayOpacity(qreal opacity);

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    PlotView* m_plotView;
    QPushButton* m_themeBtn;
    QPushButton* m_saveBtn;
    QPropertyAnimation* m_fadeAnim;
    QTimer* m_hideTimer;
    qreal m_opacity = 0.0;
    bool m_isDark = true;
};

} // namespace Skigen::Plot
