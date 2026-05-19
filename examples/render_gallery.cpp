// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#include <skigen/plot/plotview.h>

#include <Eigen/Core>
#include <QApplication>
#include <QDebug>
#include <QDir>
#include <QTimer>

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <numbers>
#include <random>
#include <vector>

namespace {

struct RenderStep {
    QString filename;
    std::function<void(Skigen::Plot::PlotView&)> setup;
};

auto lineStep(QString filename, Skigen::Plot::Theme theme) -> RenderStep {
    return {
        std::move(filename),
        [theme](Skigen::Plot::PlotView& view) {
            constexpr float pi = std::numbers::pi_v<float>;
            int n = 500;
            Eigen::VectorXf x = Eigen::VectorXf::LinSpaced(n, 0.f, 4.f * pi);
            Eigen::VectorXf sinY = x.array().sin();
            Eigen::VectorXf cosY = x.array().cos();

            view.clear();
            view.setTheme(theme);
            view.setTitle(QStringLiteral("Trigonometric Functions"));
            view.setCaption(QStringLiteral("High-resolution line rendering with crisp scientific axes"));
            view.setAxisLabels(QStringLiteral("x [rad]"), QStringLiteral("amplitude"));
            view.plot(x, sinY, {.label = "sin(x)"});
            view.plot(x, cosY, {.label = "cos(x)"});
        }
    };
}

auto scatterStep(QString filename, Skigen::Plot::Theme theme) -> RenderStep {
    return {
        std::move(filename),
        [theme](Skigen::Plot::PlotView& view) {
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

            view.clear();
            view.setTheme(theme);
            view.setTitle(QStringLiteral("Gaussian Clusters"));
            view.setCaption(QStringLiteral("Soft-edged points with translucent grid and axis labels"));
            view.setAxisLabels(QStringLiteral("feature 1"), QStringLiteral("feature 2"));
            view.scatter(x1, y1, {.pointSize = 6.0f, .label = "Cluster A"});
            view.scatter(x2, y2, {.pointSize = 6.0f, .label = "Cluster B"});
        }
    };
}

auto pointCloudStep(QString filename, Skigen::Plot::Theme theme) -> RenderStep {
    return {
        std::move(filename),
        [theme](Skigen::Plot::PlotView& view) {
            std::mt19937 rng(123);
            std::normal_distribution<float> dist(0.f, 1.f);

            int n = 720;
            Eigen::MatrixXf vertices(n, 3);
            for (int i = 0; i < n; ++i) {
                vertices(i, 0) = std::clamp(dist(rng), -2.25f, 2.25f);
                vertices(i, 1) = std::clamp(dist(rng), -2.25f, 2.25f);
                vertices(i, 2) = std::clamp(dist(rng), -2.25f, 2.25f);
            }

            Skigen::Plot::Camera3D camera;
            camera.lookAt({4.8f, 4.0f, 5.0f}, {0.f, -0.15f, 0.f});
            camera.setPerspective(32.f, 4.f / 3.f, 0.1f, 50.f);

            view.clear();
            view.setTheme(theme);
            view.setTitle(QStringLiteral("3D Point Cloud"));
            view.setCaption(QStringLiteral("Depth-tested point rendering"));
            view.setAxisLabels(QStringLiteral("x"), QStringLiteral("y"), QStringLiteral("z"));
            view.pointCloud(vertices);
            view.setCamera(camera);
        }
    };
}

auto meshStep(QString filename, Skigen::Plot::Theme theme) -> RenderStep {
    return {
        std::move(filename),
        [theme](Skigen::Plot::PlotView& view) {
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
            indices << 0, 1, 2,  0, 2, 3,
                       4, 6, 5,  4, 7, 6,
                       0, 4, 5,  0, 5, 1,
                       2, 6, 7,  2, 7, 3,
                       0, 3, 7,  0, 7, 4,
                       1, 5, 6,  1, 6, 2;

            Skigen::Plot::Camera3D camera;
            camera.lookAt({3.7f, 2.7f, 4.6f}, {0.f, 0.f, 0.f});
            camera.setPerspective(38.f, 4.f / 3.f, 0.1f, 50.f);

            view.clear();
            view.setTheme(theme);
            view.setTitle(QStringLiteral("Cube Mesh"));
            view.setCaption(QStringLiteral("Flat-shaded scientific mesh with sharp-edge overlay"));
            view.setAxisLabels(QStringLiteral("x"), QStringLiteral("y"), QStringLiteral("z"));
            view.mesh(vertices, indices, {.color = Eigen::Vector4f(0.04f, 0.48f, 0.70f, 1.0f)});
            view.setCamera(camera);
        }
    };
}

} // namespace

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);

    QString outDir = argc > 1 ? QString::fromLocal8Bit(argv[1])
                              : QStringLiteral("renderings");
    QDir().mkpath(outDir);

    std::vector<RenderStep> steps = {
        lineStep(QStringLiteral("line_dark.png"), Skigen::Plot::Theme::dark()),
        lineStep(QStringLiteral("line_light.png"), Skigen::Plot::Theme::light()),
        scatterStep(QStringLiteral("scatter_dark.png"), Skigen::Plot::Theme::dark()),
        scatterStep(QStringLiteral("scatter_light.png"), Skigen::Plot::Theme::light()),
        pointCloudStep(QStringLiteral("point_cloud_dark.png"), Skigen::Plot::Theme::dark()),
        meshStep(QStringLiteral("mesh_dark.png"), Skigen::Plot::Theme::dark()),
    };

    Skigen::Plot::PlotView view;
    view.setOverlayVisible(false);
    view.resize(1200, 800);
    steps.front().setup(view);
    view.show();

    auto index = std::make_shared<std::size_t>(0);
    auto runNext = std::make_shared<std::function<void()>>();
    *runNext = [&]() {
        if (*index >= steps.size()) {
            app.quit();
            return;
        }

        const auto& step = steps[*index];
        step.setup(view);
        view.update();

        QTimer::singleShot(180, &view, [&, step]() {
            QString path = outDir + QDir::separator() + step.filename;
            if (!view.savePng(path, 1200, 800))
                qWarning() << "Failed to render" << path;
            ++(*index);
            (*runNext)();
        });
    };

    QTimer::singleShot(180, &view, [&]() { (*runNext)(); });
    return app.exec();
}
