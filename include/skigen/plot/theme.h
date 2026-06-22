#pragma once

#include <skigen/plot/export.h>

#include <Eigen/Core>

#include <array>

namespace Skigen::Plot {

/// @brief Selectable categorical colour palette for plotted series.
///
/// A palette only governs the per-series colours; it is orthogonal to a
/// theme's layout (background, grid, axes). Use Theme::withPalette() or
/// PlotView::setPalette() to apply a palette to any theme.
enum class Palette {
    Skigen,      ///< Vivid cyan / violet / emerald categorical palette (default).
    Matplotlib,  ///< Canonical matplotlib `tab10` palette (first six colours).
};

/// @brief The six categorical series colours for the given palette.
SKIGENPLOT_EXPORT auto seriesPalette(Palette palette) -> std::array<Eigen::Vector4f, 6>;

struct SKIGENPLOT_EXPORT Theme {
    Eigen::Vector4f background;
    Eigen::Vector4f gridColor;
    Eigen::Vector4f axisColor;
    Eigen::Vector4f textColor;
    std::array<Eigen::Vector4f, 6> seriesColors;

    static auto dark() -> Theme;
    static auto light() -> Theme;

    /// @brief Clean "paper" light theme: white background, light-gray solid
    ///   gridlines, muted dark-gray axes/text, and the vivid Skigen
    ///   categorical series palette. This is the matplotlib-style *layout*
    ///   paired with Skigen's own colours — ideal for publication figures.
    ///   Use withPalette() / PlotView::setPalette() to swap the palette.
    static auto paper() -> Theme;

    /// @brief Faithful matplotlib look: the clean white paper() layout with
    ///   the canonical matplotlib `tab10` series palette. Equivalent to
    ///   `Theme::paper().withPalette(Palette::Matplotlib)`. Use this when you
    ///   want output that matches `matplotlib.pyplot` defaults exactly.
    static auto matplotlibStyle() -> Theme;

    /// @brief A copy of this theme with its series colours replaced by the
    ///   given @p palette. Layout (background, grid, axes, text) is kept.
    [[nodiscard]] auto withPalette(Palette palette) const -> Theme;
};

} // namespace Skigen::Plot
