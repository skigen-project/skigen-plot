// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#include <skigen/plot/figure.h>

#include <QApplication>
#include <QCoreApplication>
#include <QElapsedTimer>
#include <QString>

#include <algorithm>
#include <array>
#include <memory>
#include <vector>

namespace Skigen::Plot {

namespace {

// Static argc/argv backing for an implicitly-created QApplication. Qt
// requires these references to remain valid for the lifetime of the
// QApplication; making them file-static satisfies that.
int        g_dummy_argc = 1;
char       g_dummy_arg0[] = "skigen-plot";
char*      g_dummy_argv[] = {g_dummy_arg0, nullptr};

// Process events until `predicate` returns true or `timeout_ms` elapses.
// Used to wait for the QRhiWidget pipeline to come online before we can
// read pixels out of it via savePng().
bool spin(int timeout_ms, auto predicate) {
    QElapsedTimer t;
    t.start();
    while (t.elapsed() < timeout_ms) {
        QCoreApplication::processEvents(QEventLoop::AllEvents, 16);
        if (predicate())
            return true;
    }
    return predicate();
}

} // namespace

struct Figure::Impl {
    std::unique_ptr<QApplication> ownedApp;
    std::unique_ptr<PlotView>     view;

    Impl() {
        // Adopt an existing QApplication if the host already created one
        // (e.g. embedded inside another Qt app); otherwise spin one up
        // transparently so the user never has to write main()-level Qt
        // boilerplate.
        if (QCoreApplication::instance() == nullptr) {
            ownedApp = std::make_unique<QApplication>(g_dummy_argc, g_dummy_argv);
        }
        view = std::make_unique<PlotView>();
    }
};

Figure::Figure() : d(std::make_unique<Impl>()) {}
Figure::~Figure() = default;

auto Figure::view() -> PlotView&             { return *d->view; }
auto Figure::view() const -> const PlotView& { return *d->view; }

auto Figure::title(const QString& s)   -> Figure& { d->view->setTitle(s);       return *this; }
auto Figure::caption(const QString& s) -> Figure& { d->view->setCaption(s);     return *this; }
auto Figure::xlabel(const QString& s)  -> Figure& { d->view->setXAxisLabel(s);  return *this; }
auto Figure::ylabel(const QString& s)  -> Figure& { d->view->setYAxisLabel(s);  return *this; }
auto Figure::theme(const Theme& t)     -> Figure& { d->view->setTheme(t);       return *this; }
auto Figure::clear()                   -> Figure& { d->view->clear();           return *this; }

void Figure::scatterLabeled(std::span<const float> xy, int n,
                            std::span<const int> labels) {
    if (n <= 0 || labels.size() != static_cast<std::size_t>(n) ||
        xy.size() != static_cast<std::size_t>(n) * 2)
        return;

    // Collect unique labels in sorted order so the palette mapping is
    // deterministic across runs.
    std::vector<int> uniq(labels.begin(), labels.end());
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());

    for (int lbl : uniq) {
        std::vector<float> xs, ys;
        xs.reserve(static_cast<std::size_t>(n));
        ys.reserve(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            if (labels[i] == lbl) {
                xs.push_back(xy[i * 2 + 0]);
                ys.push_back(xy[i * 2 + 1]);
            }
        }
        Eigen::Map<const Eigen::VectorXf> xv(xs.data(),
            static_cast<Eigen::Index>(xs.size()));
        Eigen::Map<const Eigen::VectorXf> yv(ys.data(),
            static_cast<Eigen::Index>(ys.size()));
        d->view->scatter(xv, yv, {
            .pointSize = 7.0f,
            .label = QStringLiteral("%1").arg(lbl),
        });
    }
}

auto Figure::show(int width, int height) -> int {
    d->view->setOverlayVisible(true);
    d->view->resize(width, height);
    d->view->show();
    return QCoreApplication::exec();
}

auto Figure::save(const QString& path, int width, int height) -> bool {
    // Realise the widget without ever putting pixels on screen — RHI
    // initialisation still runs because the widget is technically shown,
    // but no window manager surface is created.
    d->view->setOverlayVisible(false);
    d->view->resize(width, height);
    d->view->setAttribute(Qt::WA_DontShowOnScreen, true);
    d->view->show();

    bool ok = false;
    spin(/*timeout_ms=*/3000, [&] {
        ok = d->view->savePng(path, width, height);
        return ok;
    });

    d->view->hide();
    d->view->setAttribute(Qt::WA_DontShowOnScreen, false);
    return ok;
}

auto Figure::saveThemed(const QString& stem, int width, int height) -> bool {
    const Theme original = d->view->theme();

    struct Variant { const char* suffix; Theme theme; };
    const std::array<Variant, 2> variants = {{
        {"_dark.png",  Theme::dark()},
        {"_light.png", Theme::light()},
    }};

    bool all_ok = true;
    for (const auto& v : variants) {
        d->view->setTheme(v.theme);
        if (!save(stem + QString::fromLatin1(v.suffix), width, height))
            all_ok = false;
    }

    d->view->setTheme(original);
    return all_ok;
}

} // namespace Skigen::Plot
