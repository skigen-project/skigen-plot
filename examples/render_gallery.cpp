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

auto histStep(QString filename, Skigen::Plot::Theme theme) -> RenderStep {
    return {
        std::move(filename),
        [theme](Skigen::Plot::PlotView& view) {
            std::mt19937 rng(7);
            std::normal_distribution<float> dist(0.f, 1.f);
            int n = 2000;
            Eigen::VectorXf v(n);
            for (int i = 0; i < n; ++i) v(i) = dist(rng);

            view.clear();
            view.setTheme(theme);
            view.setTitle(QStringLiteral("Histogram"));
            view.setCaption(QStringLiteral("Gaussian sample, Sturges-rule binning"));
            view.setAxisLabels(QStringLiteral("value"), QStringLiteral("count"));
            view.hist(v, 30);
        }
    };
}

auto barStep(QString filename, Skigen::Plot::Theme theme) -> RenderStep {
    return {
        std::move(filename),
        [theme](Skigen::Plot::PlotView& view) {
            Eigen::VectorXf x(6), h(6);
            x << 0, 1, 2, 3, 4, 5;
            h << 0.32f, 0.18f, 0.27f, 0.09f, 0.08f, 0.06f;

            view.clear();
            view.setTheme(theme);
            view.setTitle(QStringLiteral("Bar Chart"));
            view.setCaption(QStringLiteral("Feature importances (illustrative)"));
            view.setAxisLabels(QStringLiteral("feature"), QStringLiteral("importance"));
            view.bar(x, h, 0.7f);
        }
    };
}

auto errorbarStep(QString filename, Skigen::Plot::Theme theme) -> RenderStep {
    return {
        std::move(filename),
        [theme](Skigen::Plot::PlotView& view) {
            constexpr float pi = std::numbers::pi_v<float>;
            int n = 12;
            Eigen::VectorXf x = Eigen::VectorXf::LinSpaced(n, 0.f, 2.f * pi);
            Eigen::VectorXf y = x.array().sin();
            Eigen::VectorXf err = Eigen::VectorXf::Constant(n, 0.12f);
            // Shaded confidence band + mean line + error bars.
            Eigen::VectorXf lo = y.array() - err.array();
            Eigen::VectorXf hi = y.array() + err.array();

            view.clear();
            view.setTheme(theme);
            view.setTitle(QStringLiteral("Error Bars & Confidence Band"));
            view.setCaption(QStringLiteral("Cross-validation-style score with uncertainty"));
            view.setAxisLabels(QStringLiteral("x"), QStringLiteral("score"));
            view.fillBetween(x, lo, hi);
            view.plot(x, y, {.label = "mean"});
            view.errorbar(x, y, err);
        }
    };
}

auto stemStep(QString filename, Skigen::Plot::Theme theme) -> RenderStep {
    return {
        std::move(filename),
        [theme](Skigen::Plot::PlotView& view) {
            constexpr float pi = std::numbers::pi_v<float>;
            int n = 24;
            Eigen::VectorXf x = Eigen::VectorXf::LinSpaced(n, 0.f, 2.f * pi);
            Eigen::VectorXf y = (x.array() * 1.5f).sin() * (-x.array() * 0.25f).exp();

            view.clear();
            view.setTheme(theme);
            view.setTitle(QStringLiteral("Stem Plot"));
            view.setCaption(QStringLiteral("Damped oscillation — baseline-anchored impulses"));
            view.setAxisLabels(QStringLiteral("n"), QStringLiteral("amplitude"));
            view.stem(x, y);
        }
    };
}

auto boxplotStep(QString filename, Skigen::Plot::Theme theme) -> RenderStep {
    return {
        std::move(filename),
        [theme](Skigen::Plot::PlotView& view) {
            std::mt19937 rng(11);
            std::vector<Eigen::VectorXf> groups;
            for (int g = 0; g < 4; ++g) {
                std::normal_distribution<float> dist(static_cast<float>(g) * 0.6f,
                                                     0.5f + 0.15f * static_cast<float>(g));
                int m = 80;
                Eigen::VectorXf v(m);
                for (int i = 0; i < m; ++i) v(i) = dist(rng);
                // Inject a couple of outliers in the first group.
                if (g == 0) { v(0) = 3.5f; v(1) = -3.0f; }
                groups.push_back(v);
            }

            view.clear();
            view.setTheme(theme);
            view.setTitle(QStringLiteral("Box Plot"));
            view.setCaption(QStringLiteral("Distribution comparison across 4 groups"));
            view.setAxisLabels(QStringLiteral("group"), QStringLiteral("value"));
            view.boxplot(groups);
        }
    };
}

auto contourStep(QString filename, Skigen::Plot::Theme theme) -> RenderStep {
    return {
        std::move(filename),
        [theme](Skigen::Plot::PlotView& view) {
            // A smooth two-bump field (sum of two Gaussians) on a 48x48 grid —
            // reads like a classifier decision surface.
            int n = 48;
            Eigen::MatrixXf z(n, n);
            auto bump = [](float x, float y, float cx, float cy, float s) {
                float dx = (x - cx) / s, dy = (y - cy) / s;
                return std::exp(-(dx * dx + dy * dy));
            };
            for (int r = 0; r < n; ++r)
                for (int c = 0; c < n; ++c) {
                    float x = static_cast<float>(c);
                    float y = static_cast<float>(r);
                    z(r, c) = bump(x, y, n * 0.34f, n * 0.40f, n * 0.20f)
                            - 0.8f * bump(x, y, n * 0.68f, n * 0.62f, n * 0.22f);
                }

            view.clear();
            view.setTheme(theme);
            view.setGridVisible(false);
            view.setAxisArrowsVisible(false);
            view.setTitle(QStringLiteral("Contour (filled + lines)"));
            view.setCaption(QStringLiteral("Decision-surface-style field, coolwarm fill + iso-lines"));
            view.setAxisLabels(QStringLiteral("x"), QStringLiteral("y"));
            view.contourf(z, 12, Skigen::Plot::Colormap::Coolwarm);
            view.contour(z, 8, {.color = Eigen::Vector4f(0.12f, 0.12f, 0.14f, 0.8f)});
        }
    };
}

auto heatmapStep(QString filename, Skigen::Plot::Theme theme,
                 Skigen::Plot::Colormap cmap = Skigen::Plot::Colormap::Viridis,
                 QString cmapName = QStringLiteral("viridis")) -> RenderStep {
    return {
        std::move(filename),
        [theme, cmap, cmapName](Skigen::Plot::PlotView& view) {
            // A smooth 2-D field (Gaussian bump) sampled on a 24x24 grid so
            // the colormap gradient is clearly visible.
            int n = 24;
            Eigen::MatrixXf m(n, n);
            for (int r = 0; r < n; ++r)
                for (int c = 0; c < n; ++c) {
                    float x = (static_cast<float>(c) - n * 0.5f) / (n * 0.32f);
                    float y = (static_cast<float>(r) - n * 0.5f) / (n * 0.32f);
                    m(r, c) = std::exp(-(x * x + y * y));
                }

            view.clear();
            view.setTheme(theme);
            view.setGridVisible(false);
            view.setAxisArrowsVisible(false);  // image plots use plain spines
            view.setTitle(QStringLiteral("Heatmap (imshow)"));
            view.setCaption(QStringLiteral("Gaussian field, %1 colormap").arg(cmapName));
            view.setAxisLabels(QStringLiteral("x"), QStringLiteral("y"));
            view.imshow(m, cmap);
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
        // `_paper` = clean white layout + Skigen palette (the default look).
        // `_mpl`   = the same layout with matplotlib's `tab10` palette.
        lineStep(QStringLiteral("line_paper.png"), Skigen::Plot::Theme::paper()),
        lineStep(QStringLiteral("line_mpl.png"), Skigen::Plot::Theme::matplotlibStyle()),
        lineStep(QStringLiteral("line_dark.png"), Skigen::Plot::Theme::dark()),
        lineStep(QStringLiteral("line_light.png"), Skigen::Plot::Theme::light()),
        scatterStep(QStringLiteral("scatter_paper.png"), Skigen::Plot::Theme::paper()),
        scatterStep(QStringLiteral("scatter_mpl.png"), Skigen::Plot::Theme::matplotlibStyle()),
        scatterStep(QStringLiteral("scatter_dark.png"), Skigen::Plot::Theme::dark()),
        scatterStep(QStringLiteral("scatter_light.png"), Skigen::Plot::Theme::light()),
        histStep(QStringLiteral("hist_paper.png"), Skigen::Plot::Theme::paper()),
        histStep(QStringLiteral("hist_dark.png"), Skigen::Plot::Theme::dark()),
        barStep(QStringLiteral("bar_paper.png"), Skigen::Plot::Theme::paper()),
        barStep(QStringLiteral("bar_dark.png"), Skigen::Plot::Theme::dark()),
        errorbarStep(QStringLiteral("errorbar_paper.png"), Skigen::Plot::Theme::paper()),
        errorbarStep(QStringLiteral("errorbar_dark.png"), Skigen::Plot::Theme::dark()),
        stemStep(QStringLiteral("stem_paper.png"), Skigen::Plot::Theme::paper()),
        stemStep(QStringLiteral("stem_dark.png"), Skigen::Plot::Theme::dark()),
        boxplotStep(QStringLiteral("boxplot_paper.png"), Skigen::Plot::Theme::paper()),
        boxplotStep(QStringLiteral("boxplot_dark.png"), Skigen::Plot::Theme::dark()),
        contourStep(QStringLiteral("contour_paper.png"), Skigen::Plot::Theme::paper()),
        contourStep(QStringLiteral("contour_dark.png"), Skigen::Plot::Theme::dark()),
        heatmapStep(QStringLiteral("heatmap_paper.png"), Skigen::Plot::Theme::paper()),
        heatmapStep(QStringLiteral("heatmap_dark.png"), Skigen::Plot::Theme::dark()),
        heatmapStep(QStringLiteral("heatmap_jet.png"), Skigen::Plot::Theme::paper(),
                    Skigen::Plot::Colormap::Jet, QStringLiteral("jet")),
        heatmapStep(QStringLiteral("heatmap_hot.png"), Skigen::Plot::Theme::dark(),
                    Skigen::Plot::Colormap::Hot, QStringLiteral("hot")),
        pointCloudStep(QStringLiteral("point_cloud_paper.png"), Skigen::Plot::Theme::paper()),
        pointCloudStep(QStringLiteral("point_cloud_dark.png"), Skigen::Plot::Theme::dark()),
        meshStep(QStringLiteral("mesh_paper.png"), Skigen::Plot::Theme::paper()),
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
        view.setGridVisible(true);  // default; a step may turn it off
        step.setup(view);
        // The clean paper/matplotlib look uses plain spines (no arrowheads).
        const bool plainSpines = step.filename.contains(QStringLiteral("_mpl"))
                                 || step.filename.contains(QStringLiteral("_paper"));
        view.setAxisArrowsVisible(!plainSpines);
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
