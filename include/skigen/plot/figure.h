#pragma once

#include <skigen/plot/export.h>
#include <skigen/plot/plotview.h>
#include <skigen/plot/series.h>
#include <skigen/plot/theme.h>

#include <Eigen/Core>
#include <QString>

#include <memory>
#include <span>
#include <type_traits>

namespace Skigen::Plot {

// Standalone plotting façade. Hides Qt boilerplate (QApplication, event
// loop, RHI initialisation, theme cycling, headless rendering) behind a
// matplotlib-flavoured chainable API:
//
//     Skigen::Plot::Figure fig;
//     fig.scatter(X, labels).title("KMeans Clustering");
//     fig.show();                  // interactive — blocks until closed
//     fig.save("out.png");         // headless render to a single file
//     fig.saveThemed("out");       // writes out_dark.png + out_light.png
//
// To embed a plot inside an existing Qt application, instantiate
// `PlotView` directly and add it to a layout — Figure is only needed for
// standalone use.
class SKIGENPLOT_EXPORT Figure {
public:
    Figure();
    ~Figure();
    Figure(const Figure&) = delete;
    Figure& operator=(const Figure&) = delete;

    // Underlying widget — useful for advanced styling or embedding.
    auto view() -> PlotView&;
    auto view() const -> const PlotView&;

    // ── Plot primitives (chainable) ────────────────────────────────────

    template <typename DerivedX, typename DerivedY>
    auto plot(const Eigen::MatrixBase<DerivedX>& x,
              const Eigen::MatrixBase<DerivedY>& y,
              const PlotStyle& style = {}) -> Figure&;

    template <typename DerivedX, typename DerivedY>
        requires std::is_floating_point_v<typename DerivedY::Scalar>
    auto scatter(const Eigen::MatrixBase<DerivedX>& x,
                 const Eigen::MatrixBase<DerivedY>& y,
                 const PlotStyle& style = {}) -> Figure&;

    // Labeled scatter — N×2 coordinates + N integer labels. Emits one
    // series per unique label so the theme palette colors cycle
    // automatically. Equivalent to matplotlib's `scatter(x, y, c=labels)`.
    template <typename DerivedXY, typename DerivedL>
        requires std::is_integral_v<typename DerivedL::Scalar>
    auto scatter(const Eigen::MatrixBase<DerivedXY>& points,
                 const Eigen::MatrixBase<DerivedL>& labels) -> Figure&;

    // ── Labels & theme (chainable) ─────────────────────────────────────

    auto title(const QString& s) -> Figure&;
    auto caption(const QString& s) -> Figure&;
    auto xlabel(const QString& s) -> Figure&;
    auto ylabel(const QString& s) -> Figure&;
    auto theme(const Theme& t) -> Figure&;

    auto clear() -> Figure&;

    // ── Output ─────────────────────────────────────────────────────────

    // Opens an interactive window with the toolbar overlay and blocks
    // until the user closes it. Returns the event loop exit code.
    auto show(int width = 1200, int height = 800) -> int;

    // Renders the current figure to a PNG file without showing a visible
    // window. Returns false on failure.
    auto save(const QString& path, int width = 1200, int height = 800) -> bool;

    // Renders the figure twice with dark and light themes into
    // <stem>_dark.png and <stem>_light.png. Restores the previous theme.
    auto saveThemed(const QString& stem, int width = 1200, int height = 800) -> bool;

private:
    void scatterLabeled(std::span<const float> xy_rowmajor, int n,
                        std::span<const int> labels);

    struct Impl;
    std::unique_ptr<Impl> d;
};

// ── Template implementations ──────────────────────────────────────────

template <typename DerivedX, typename DerivedY>
auto Figure::plot(const Eigen::MatrixBase<DerivedX>& x,
                  const Eigen::MatrixBase<DerivedY>& y,
                  const PlotStyle& style) -> Figure& {
    view().plot(x, y, style);
    return *this;
}

template <typename DerivedX, typename DerivedY>
    requires std::is_floating_point_v<typename DerivedY::Scalar>
auto Figure::scatter(const Eigen::MatrixBase<DerivedX>& x,
                     const Eigen::MatrixBase<DerivedY>& y,
                     const PlotStyle& style) -> Figure& {
    view().scatter(x, y, style);
    return *this;
}

template <typename DerivedXY, typename DerivedL>
    requires std::is_integral_v<typename DerivedL::Scalar>
auto Figure::scatter(const Eigen::MatrixBase<DerivedXY>& points,
                     const Eigen::MatrixBase<DerivedL>& labels) -> Figure& {
    // Materialise the inputs as float (RowMajor) + int so the impl can
    // iterate by row regardless of the caller's storage order.
    using RowMajorXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::RowMajor>;
    RowMajorXf P = points.derived().template cast<float>();
    Eigen::VectorXi L = labels.derived().template cast<int>().eval();
    scatterLabeled({P.data(), static_cast<std::size_t>(P.size())},
                   static_cast<int>(P.rows()),
                   {L.data(), static_cast<std::size_t>(L.size())});
    return *this;
}

} // namespace Skigen::Plot
