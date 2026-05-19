// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#include <skigen/plot/plotview.h>
#include <Eigen/Core>
#include <QApplication>
#include <random>

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);

    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.f, 1.f);

    int n = 1000;
    Eigen::MatrixXf vertices(n, 3);
    for (int i = 0; i < n; ++i) {
        vertices(i, 0) = dist(rng);
        vertices(i, 1) = dist(rng);
        vertices(i, 2) = dist(rng);
    }

    Skigen::Plot::PlotView view;
    view.setTitle("3D Point Cloud");
    view.pointCloud(vertices);

    Skigen::Plot::Camera3D camera;
    camera.lookAt({3.f, 3.f, 3.f}, {0.f, 0.f, 0.f});
    camera.setPerspective(45.f, 4.f / 3.f, 0.1f, 50.f);
    view.setCamera(camera);

    view.resize(800, 600);
    view.setWindowTitle("SkigenPlot — Point Cloud");
    view.show();

    return app.exec();
}
