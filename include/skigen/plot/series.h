#pragma once

#include <skigen/plot/export.h>

#include <Eigen/Core>
#include <QString>

#include <cstdint>
#include <optional>

namespace Skigen::Plot {

class PlotView;

class SKIGENPLOT_EXPORT SeriesHandle {
public:
    SeriesHandle() = default;

    explicit operator bool() const noexcept { return m_id != 0; }
    friend bool operator==(SeriesHandle, SeriesHandle) = default;

private:
    explicit SeriesHandle(std::uint64_t id) : m_id(id) {}

    std::uint64_t m_id = 0;

    friend class PlotView;
};

struct SKIGENPLOT_EXPORT PlotStyle {
    std::optional<Eigen::Vector4f> color;
    float lineWidth = 1.5f;
    float pointSize = 5.0f;
    float opacity = 1.0f;
    bool hollow = false;
    QString label;
};

} // namespace Skigen::Plot
