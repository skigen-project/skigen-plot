#pragma once

#include <skigen/plot/export.h>

#include <Eigen/Core>
#include <QString>

#include <optional>

namespace Skigen::Plot {

struct SKIGENPLOT_EXPORT PlotStyle {
    std::optional<Eigen::Vector4f> color;
    float lineWidth = 1.5f;
    float pointSize = 5.0f;
    float opacity = 1.0f;
    bool hollow = false;
    QString label;
};

} // namespace Skigen::Plot
