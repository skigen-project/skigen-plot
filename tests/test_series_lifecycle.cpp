// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#include <skigen/plot/plotview.h>

#include <QApplication>
#include <QImage>

#include <Eigen/Core>

#include <array>
#include <limits>

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);
    Skigen::Plot::PlotView view;

    if (view.xLimits() != std::pair{0.0f, 1.0f}
        || view.yLimits() != std::pair{0.0f, 1.0f}) {
        return 30;
    }

    Eigen::VectorXf x(3);
    x << 0.0f, 1.0f, 2.0f;
    Eigen::VectorXf y(3);
    y << 1.0f, 2.0f, 3.0f;

    const auto line = view.plot(x, y, {.label = "line"});
    const auto points = view.scatter(x, y, {.label = "points"});
    if (!line || !points || line == points)
        return 1;
    if (!view.containsSeries(line) || !view.containsSeries(points))
        return 2;

    if (!view.setXLimits(10.0f, -2.0f) || !view.setYLimits(-4.0f, 8.0f)
        || view.xLimits() != std::pair{10.0f, -2.0f}
        || view.yLimits() != std::pair{-4.0f, 8.0f}) {
        return 26;
    }
    if (view.setXLimits(1.0f, 1.0f)
        || view.setYLimits(0.0f, std::numeric_limits<float>::infinity())
        || view.xLimits() != std::pair{10.0f, -2.0f}
        || view.yLimits() != std::pair{-4.0f, 8.0f}) {
        return 27;
    }
    view.resetXLimits();
    if (view.xLimits() == std::pair{10.0f, -2.0f}
        || view.yLimits() != std::pair{-4.0f, 8.0f}) {
        return 28;
    }
    view.resetAxisLimits();
    if (view.yLimits() == std::pair{-4.0f, 8.0f})
        return 29;

    view.resize(640, 480);
    QImage withoutLegend(view.size(), QImage::Format_ARGB32_Premultiplied);
    withoutLegend.fill(Qt::transparent);
    static_cast<QWidget&>(view).render(&withoutLegend);

    view.setLegendVisible(true);
    view.setLegendPosition(Skigen::Plot::LegendPosition::LowerLeft);
    if (!view.legendVisible()
        || view.legendPosition() != Skigen::Plot::LegendPosition::LowerLeft) {
        return 13;
    }

    QImage withLegend(view.size(), QImage::Format_ARGB32_Premultiplied);
    withLegend.fill(Qt::transparent);
    static_cast<QWidget&>(view).render(&withLegend);
    int changedPixels = 0;
    for (int yPixel = 0; yPixel < withLegend.height(); ++yPixel) {
        for (int xPixel = 0; xPixel < withLegend.width(); ++xPixel) {
            if (withLegend.pixel(xPixel, yPixel)
                != withoutLegend.pixel(xPixel, yPixel)) {
                ++changedPixels;
            }
        }
    }
    if (changedPixels < 100)
        return 14;

    Eigen::VectorXf updatedY(2);
    updatedY << 4.0f, 5.0f;
    if (!view.updateSeriesData(line, x, updatedY))
        return 3;
    if (!view.setSeriesStyle(points, {.pointSize = 9.0f, .hollow = true}))
        return 4;
    constexpr std::array markerShapes{
        Skigen::Plot::MarkerShape::Circle,
        Skigen::Plot::MarkerShape::Square,
        Skigen::Plot::MarkerShape::Triangle,
        Skigen::Plot::MarkerShape::Plus,
        Skigen::Plot::MarkerShape::Cross
    };
    for (auto marker : markerShapes) {
        if (!view.setSeriesStyle(points, {.pointSize = 9.0f, .marker = marker}))
            return 31;
    }
    if (!view.setSeriesVisible(line, false) || !view.is2DView())
        return 5;
    if (!view.setSeriesVisible(points, false) || view.is2DView())
        return 6;
    if (!view.setSeriesVisible(line, true) || !view.is2DView())
        return 7;

    if (!view.removeSeries(line) || view.containsSeries(line))
        return 8;
    if (view.updateSeriesData(line, x, y)
        || view.setSeriesVisible(line, true)
        || view.removeSeries(line)) {
        return 9;
    }
    if (!view.containsSeries(points))
        return 10;

    Skigen::Plot::PlotView otherView;
    if (otherView.containsSeries(points) || otherView.removeSeries(points))
        return 11;

    const auto histogram = view.hist(y, 2, false, {.label = "histogram"});
    if (!histogram || !view.containsSeries(histogram))
        return 15;
    if (!view.setSeriesStyle(histogram, {.label = "updated histogram"})
        || !view.setSeriesVisible(histogram, false)
        || !view.removeSeries(histogram)
        || view.containsSeries(histogram)) {
        return 16;
    }

    Eigen::VectorXf empty;
    if (view.hist(empty))
        return 17;

    const auto bars = view.bar(x, y, 0.8f, {.label = "bars"});
    const auto horizontalBars = view.barh(x, y);
    if (!bars || !horizontalBars || bars == horizontalBars
        || !view.containsSeries(bars) || !view.containsSeries(horizontalBars)) {
        return 18;
    }
    if (!view.removeSeries(bars) || !view.containsSeries(horizontalBars))
        return 19;
    if (view.bar(empty, y) || view.barh(y, empty))
        return 20;

    const auto band = view.fillBetween(x, y, x, {.label = "band"});
    const auto stairs = view.step(x, y, {.label = "step"});
    if (!band || !stairs || band == stairs
        || !view.containsSeries(band) || !view.containsSeries(stairs)) {
        return 21;
    }
    if (!view.setSeriesVisible(band, false) || !view.removeSeries(stairs))
        return 22;
    if (view.fillBetween(empty, y, x) || view.step(empty, y))
        return 23;

    const auto errors = view.errorbar(x, y, x, {.label = "errors"});
    const auto vectors = view.quiver(x, y, y, x, {.label = "vectors"});
    if (!errors || !vectors || errors == vectors
        || !view.setSeriesVisible(errors, false)
        || !view.removeSeries(vectors)) {
        return 24;
    }
    if (view.errorbar(empty, y, x) || view.quiver(x, empty, y, x))
        return 25;

    view.clear();
    return view.containsSeries(points) ? 12 : 0;
}