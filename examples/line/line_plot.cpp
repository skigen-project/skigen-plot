// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// line_plot.cpp — Plot a sine wave with SkigenPlot
#include <skigen/plot/plotview.h>
#include <Eigen/Core>
#include <QApplication>
#include <cmath>
#include <numbers>

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);

    //! [example_line_plot]
    int n = 500;
    Eigen::VectorXf x = Eigen::VectorXf::LinSpaced(n, 0.f,
        4.f * std::numbers::pi_v<float>);
    Eigen::VectorXf y = x.array().sin();

    Skigen::Plot::PlotView view;
    view.plot(x, y);
    view.setLineColor({0.2f, 0.8f, 0.4f, 1.0f});
    view.setBackgroundColor({0.05f, 0.05f, 0.08f, 1.0f});
    view.resize(800, 500);
    view.setWindowTitle("SkigenPlot — Sine Wave");
    view.show();
    //! [example_line_plot]

    return app.exec();
}
