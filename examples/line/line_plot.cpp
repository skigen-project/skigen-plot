// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#include <skigen/plot/plotview.h>
#include <Eigen/Core>
#include <QApplication>
#include <cmath>
#include <numbers>

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);

    constexpr float pi = std::numbers::pi_v<float>;
    int n = 500;
    Eigen::VectorXf x = Eigen::VectorXf::LinSpaced(n, 0.f, 4.f * pi);
    Eigen::VectorXf sinY = x.array().sin();
    Eigen::VectorXf cosY = x.array().cos();

    Skigen::Plot::PlotView view;
    view.setTitle("Trigonometric Functions");
    view.plot(x, sinY, {.label = "sin(x)"});
    view.plot(x, cosY, {.label = "cos(x)"});
    view.resize(800, 500);
    view.setWindowTitle("SkigenPlot — Line Plot");
    view.show();

    return app.exec();
}
