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

    Eigen::VectorXf nonFiniteX(3);
    nonFiniteX << 0.0f, std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN();
    Eigen::VectorXf nonFiniteY = Eigen::VectorXf::Ones(3);
    Skigen::Plot::PlotView validationView;
    const auto filtered = validationView.scatter(nonFiniteX, nonFiniteY);
    nonFiniteX(0) = std::numeric_limits<float>::quiet_NaN();
    if (!filtered || validationView.scatter(nonFiniteX, nonFiniteY)
        || validationView.plot(Eigen::VectorXf{}, Eigen::VectorXf{})) {
        return 33;
    }
    if (!validationView.updateSeriesData(filtered, nonFiniteX, nonFiniteY))
        return 34;

    const auto line = view.plot(x, y, {.label = "line"});
    const auto points = view.scatter(x, y, {.label = "points"});
    if (!line || !points || line == points)
        return 1;
    const auto [autoXMin, autoXMax] = view.xLimits();
    const auto [autoYMin, autoYMax] = view.yLimits();
    if (std::abs(autoXMin + 0.1f) > 1e-5f
        || std::abs(autoXMax - 2.1f) > 1e-5f
        || std::abs(autoYMin - 0.9f) > 1e-5f
        || std::abs(autoYMax - 3.1f) > 1e-5f) {
        return 56;
    }
    if (!view.containsSeries(line) || !view.containsSeries(points))
        return 2;
    constexpr std::array lineStyles{
        Skigen::Plot::LineStyle::Solid,
        Skigen::Plot::LineStyle::Dashed,
        Skigen::Plot::LineStyle::Dotted,
        Skigen::Plot::LineStyle::DashDot
    };
    for (const auto lineStyle : lineStyles) {
        if (!view.setSeriesStyle(line, {.lineWidth = 3.0f,
                                        .label = "line",
                                        .lineStyle = lineStyle})) {
            return 51;
        }
    }

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

    if (!view.setXLimits(-2.0f, 10.0f))
        return 52;
    view.setXScale(Skigen::Plot::AxisScale::Log10);
    if (view.xScale() != Skigen::Plot::AxisScale::Log10
        || view.yScale() != Skigen::Plot::AxisScale::Linear
        || view.xLimits() == std::pair{-2.0f, 10.0f}
        || view.setXLimits(0.0f, 10.0f)
        || view.setXLimits(-1.0f, -10.0f)
        || !view.setXLimits(100.0f, 0.1f)
        || view.xLimits() != std::pair{100.0f, 0.1f}) {
        return 55;
    }
    view.setYScale(Skigen::Plot::AxisScale::Log10);
    if (view.yScale() != Skigen::Plot::AxisScale::Log10
        || view.setYLimits(-1.0f, 10.0f)
        || !view.setYLimits(0.01f, 1000.0f)) {
        return 53;
    }
    view.setXScale(Skigen::Plot::AxisScale::Linear);
    view.setYScale(Skigen::Plot::AxisScale::Linear);
    view.resetAxisLimits();
    if (view.xScale() != Skigen::Plot::AxisScale::Linear
        || view.yScale() != Skigen::Plot::AxisScale::Linear) {
        return 54;
    }

    Skigen::Plot::PlotView fieldView;
    Eigen::MatrixXf autoscaleField = Eigen::MatrixXf::Ones(3, 4);
    if (!fieldView.imshow(autoscaleField)
        || fieldView.xLimits() != std::pair{0.0f, 4.0f}
        || fieldView.yLimits() != std::pair{0.0f, 3.0f}) {
        return 57;
    }

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
    Eigen::VectorXf histogramValues(4);
    histogramValues << 1.0f, 2.0f,
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity();
    if (!view.hist(histogramValues, 2))
        return 35;
    histogramValues.setConstant(std::numeric_limits<float>::quiet_NaN());
    if (view.hist(histogramValues, 2))
        return 36;

    Eigen::VectorXf invalidValues = Eigen::VectorXf::Constant(
        3, std::numeric_limits<float>::quiet_NaN());
    if (view.bar(x, invalidValues) || view.barh(x, invalidValues)
        || view.fillBetween(x, invalidValues, y)
        || view.step(x, invalidValues)
        || view.errorbar(x, y, invalidValues)
        || view.stem(x, invalidValues)
        || view.quiver(x, y, invalidValues, y)) {
        return 37;
    }

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

    Skigen::Plot::PlotView invalid3DView;
    Eigen::MatrixXf invalidVertices(2, 2);
    invalidVertices.setZero();
    invalid3DView.pointCloud(invalidVertices);
    Eigen::MatrixXf validVertices(3, 3);
    validVertices << 0.0f, 0.0f, 0.0f,
                     1.0f, 0.0f, 0.0f,
                     0.0f, 1.0f, 0.0f;
    Eigen::MatrixXi invalidIndices(1, 3);
    invalidIndices << 0, 1, 3;
    invalid3DView.mesh(validVertices, invalidIndices);
    if (invalid3DView.is3DView())
        return 38;

    Skigen::Plot::PlotView matrixView;
    Eigen::MatrixXf invalidMatrix = Eigen::MatrixXf::Constant(
        2, 2, std::numeric_limits<float>::quiet_NaN());
    if (matrixView.imshow(invalidMatrix)
        || matrixView.contourf(invalidMatrix)
        || matrixView.contour(invalidMatrix)) {
        return 39;
    }
    if (matrixView.is2DView())
        return 39;
    invalidMatrix(0, 0) = 1.0f;
    Eigen::MatrixXf lifecycleField(2, 2);
    lifecycleField << 0.0f, 1.0f,
                      1.0f, 0.0f;
    const auto image = matrixView.imshow(invalidMatrix);
    const auto filledContours = matrixView.contourf(lifecycleField, 3);
    const auto contours = matrixView.contour(
        lifecycleField, 3,
        {.color = Eigen::Vector4f(0.2f, 0.3f, 0.4f, 1.0f)});
    if (!image || !filledContours || !contours
        || image == filledContours || image == contours
        || filledContours == contours
        || !matrixView.containsSeries(image)
        || !matrixView.containsSeries(filledContours)
        || !matrixView.containsSeries(contours)
        || !matrixView.is2DView()) {
        return 40;
    }
    if (matrixView.updateSeriesData(image, x, y)
        || !matrixView.setSeriesStyle(image, {.opacity = 0.45f})
        || !matrixView.setSeriesStyle(
            contours, {.color = Eigen::Vector4f(0.8f, 0.1f, 0.2f, 1.0f),
                       .opacity = 0.6f})
        || !matrixView.setSeriesVisible(image, false)
        || !matrixView.removeSeries(filledContours)
        || matrixView.containsSeries(filledContours)
        || !matrixView.containsSeries(contours)) {
        return 56;
    }
    Skigen::Plot::PlotView foreignFieldView;
    if (foreignFieldView.containsSeries(image)
        || foreignFieldView.setSeriesVisible(image, false)
        || foreignFieldView.removeSeries(image)) {
        return 57;
    }

    Skigen::Plot::PlotView hexbinView;
    hexbinView.hexbin(invalidValues, invalidValues);
    if (hexbinView.is2DView())
        return 41;
    Eigen::VectorXf mixedHexbinY = invalidValues;
    mixedHexbinY[0] = 1.0f;
    const auto hexagons = hexbinView.hexbin(x, mixedHexbinY);
    if (!hexagons || !hexbinView.containsSeries(hexagons)
        || !hexbinView.is2DView()) {
        return 42;
    }
    if (!hexbinView.setSeriesVisible(hexagons, false)
        || hexbinView.is2DView()
        || !hexbinView.setSeriesVisible(hexagons, true)
        || !hexbinView.removeSeries(hexagons)
        || hexbinView.containsSeries(hexagons)
        || hexbinView.setSeriesStyle(hexagons, {.opacity = 0.5f})) {
        return 58;
    }

    Skigen::Plot::PlotView pieView;
    if (pieView.pie(invalidValues) || pieView.is2DView())
        return 43;
    Eigen::VectorXf pieValues(3);
    pieValues << 1.0f, std::numeric_limits<float>::infinity(), 2.0f;
    const auto pie = pieView.pie(pieValues);
    if (!pie || !pieView.is2DView()
        || pieView.updateSeriesData(pie, x, y)
        || !pieView.setSeriesVisible(pie, false)
        || !pieView.setSeriesStyle(pie, {.opacity = 0.5f})
        || !pieView.removeSeries(pie)
        || pieView.containsSeries(pie)) {
        return 44;
    }

    Skigen::Plot::PlotView statisticsView;
    const std::vector<Eigen::VectorXf> invalidGroups{invalidValues};
    if (statisticsView.boxplot(invalidGroups)
        || statisticsView.violinplot(invalidGroups)
        || statisticsView.is2DView()) {
        return 45;
    }
    const std::vector<Eigen::VectorXf> mixedGroups{invalidValues, y};
    const auto boxes = statisticsView.boxplot(mixedGroups, {.label = "boxes"});
    const auto violins = statisticsView.violinplot(mixedGroups,
                                                    {.label = "violins"});
    if (!boxes || !violins || boxes == violins || !statisticsView.is2DView()
        || statisticsView.updateSeriesData(boxes, x, y)
        || statisticsView.updateSeriesData(violins, x, y)
        || !statisticsView.setSeriesStyle(boxes, {.opacity = 0.4f})
        || !statisticsView.setSeriesVisible(violins, false)
        || !statisticsView.removeSeries(boxes)
        || statisticsView.containsSeries(boxes)) {
        return 46;
    }

    Skigen::Plot::PlotView parameterView;
    if (!parameterView.hist(y, -1))
        return 47;
    Eigen::MatrixXf field(2, 2);
    field << 0.0f, 1.0f,
             1.0f, 0.0f;
    Skigen::Plot::PlotView contourParameterView;
    contourParameterView.contour(field, 0);
    if (!contourParameterView.is2DView())
        return 48;
    Skigen::Plot::PlotView contourfParameterView;
    contourfParameterView.contourf(field, -1);
    if (!contourfParameterView.is2DView())
        return 49;
    Skigen::Plot::PlotView hexbinParameterView;
    hexbinParameterView.hexbin(x, y, 0);
    if (!hexbinParameterView.is2DView())
        return 50;

    const auto stems = view.stem(x, y, {.label = "stems"});
    if (!stems || !view.containsSeries(stems)
        || !view.updateSeriesData(stems, x, updatedY)
        || !view.updateSeriesData(stems, x, invalidValues)
        || !view.setSeriesStyle(stems, {.label = "updated stems",
                                        .marker = Skigen::Plot::MarkerShape::Cross})
        || !view.setSeriesVisible(stems, false)
        || !view.removeSeries(stems)
        || view.containsSeries(stems)
        || view.stem(empty, y)) {
        return 32;
    }

    matrixView.clear();
    if (matrixView.containsSeries(image) || matrixView.containsSeries(contours)
        || matrixView.is2DView()) {
        return 59;
    }

    view.clear();
    return view.containsSeries(points) ? 12 : 0;
}