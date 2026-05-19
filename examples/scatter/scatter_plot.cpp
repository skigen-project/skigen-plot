// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// scatter_plot.cpp — Scatter plot of random 2D points
#include <skigen/plot/plotview.h>
#include <Eigen/Core>
#include <QApplication>
#include <random>

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);

    //! [example_scatter_plot]
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.f, 1.f);

    int n = 200;
    Eigen::VectorXf x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x(i) = dist(rng);
        y(i) = dist(rng);
    }

    Skigen::Plot::PlotView view;
    view.scatter(x, y);
    view.setLineColor({1.0f, 0.4f, 0.3f, 1.0f});
    view.resize(600, 600);
    view.setWindowTitle("SkigenPlot — Scatter");
    view.show();
    //! [example_scatter_plot]

    return app.exec();
}
