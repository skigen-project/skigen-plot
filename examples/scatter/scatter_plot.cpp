// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#include <skigen/plot/plotview.h>
#include <Eigen/Core>
#include <QApplication>
#include <random>

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.f, 1.f);

    int n = 150;
    Eigen::VectorXf x1(n), y1(n), x2(n), y2(n);
    for (int i = 0; i < n; ++i) {
        x1(i) = dist(rng) - 1.5f;
        y1(i) = dist(rng);
        x2(i) = dist(rng) + 1.5f;
        y2(i) = dist(rng);
    }

    Skigen::Plot::PlotView view;
    view.setTitle("Gaussian Clusters");
    view.scatter(x1, y1, {.pointSize = 6.0f, .label = "Cluster A"});
    view.scatter(x2, y2, {.pointSize = 6.0f, .label = "Cluster B"});
    view.resize(700, 600);
    view.setWindowTitle("SkigenPlot — Scatter");
    view.show();

    return app.exec();
}
