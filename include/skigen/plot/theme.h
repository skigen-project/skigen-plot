#pragma once

#include <skigen/plot/export.h>

#include <Eigen/Core>

#include <array>

namespace Skigen::Plot {

struct SKIGENPLOT_EXPORT Theme {
    Eigen::Vector4f background;
    Eigen::Vector4f gridColor;
    Eigen::Vector4f axisColor;
    Eigen::Vector4f textColor;
    std::array<Eigen::Vector4f, 6> seriesColors;

    static auto dark() -> Theme;
    static auto light() -> Theme;

    /// @brief Matplotlib-style light theme: white background, light-gray
    ///   solid gridlines, muted dark-gray axes/text, and the canonical
    ///   matplotlib `tab10` categorical series palette. Produces figures
    ///   that match the look of `matplotlib.pyplot` defaults.
    static auto matplotlib() -> Theme;
};

} // namespace Skigen::Plot
