// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#include <skigen/plot/plotview.h>

#include <QApplication>

#include <Eigen/Core>

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);
    Skigen::Plot::PlotView view;

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

    Eigen::VectorXf updatedY(2);
    updatedY << 4.0f, 5.0f;
    if (!view.updateSeriesData(line, x, updatedY))
        return 3;
    if (!view.setSeriesStyle(points, {.pointSize = 9.0f, .hollow = true}))
        return 4;
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

    view.clear();
    return view.containsSeries(points) ? 12 : 0;
}