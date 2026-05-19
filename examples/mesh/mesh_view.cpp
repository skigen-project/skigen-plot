// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

// mesh_view.cpp — 3D mesh rendering of a simple cube
#include <skigen/plot/plotview.h>
#include <Eigen/Core>
#include <QApplication>

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);

    //! [example_mesh_view]
    Eigen::MatrixXf vertices(8, 3);
    vertices << -1.f, -1.f, -1.f,
                 1.f, -1.f, -1.f,
                 1.f,  1.f, -1.f,
                -1.f,  1.f, -1.f,
                -1.f, -1.f,  1.f,
                 1.f, -1.f,  1.f,
                 1.f,  1.f,  1.f,
                -1.f,  1.f,  1.f;

    Eigen::MatrixXi indices(12, 3);
    indices << 0, 1, 2,  0, 2, 3,   // front
               4, 6, 5,  4, 7, 6,   // back
               0, 4, 5,  0, 5, 1,   // bottom
               2, 6, 7,  2, 7, 3,   // top
               0, 3, 7,  0, 7, 4,   // left
               1, 5, 6,  1, 6, 2;   // right

    Skigen::Plot::PlotView view;
    view.mesh(vertices, indices);

    Skigen::Plot::Camera3D camera;
    camera.lookAt({3.f, 2.f, 4.f}, {0.f, 0.f, 0.f});
    camera.setPerspective(45.f, 4.f / 3.f, 0.1f, 50.f);
    view.setCamera(camera);

    view.setBackgroundColor({0.08f, 0.08f, 0.12f, 1.0f});
    view.resize(800, 600);
    view.setWindowTitle("SkigenPlot — Cube Mesh");
    view.show();
    //! [example_mesh_view]

    return app.exec();
}
