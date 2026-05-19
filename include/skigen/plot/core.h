#pragma once

#include <skigen/plot/export.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <concepts>
#include <limits>
#include <numbers>
#include <ranges>
#include <span>

namespace Skigen::Plot {

// ── Range → Eigen bridge ─────────────────────────────────────────────────

template <std::ranges::contiguous_range R>
    requires std::is_arithmetic_v<std::ranges::range_value_t<R>>
auto mapRange(R&& range) {
    using Scalar = std::ranges::range_value_t<R>;
    using MapType = Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>;
    return MapType(std::ranges::data(range),
                   static_cast<Eigen::Index>(std::ranges::size(range)));
}

// ── 2D Bounding Box ──────────────────────────────────────────────────────

struct SKIGENPLOT_EXPORT BoundingBox2D {
    Eigen::Vector2f min = Eigen::Vector2f::Constant(
        std::numeric_limits<float>::max());
    Eigen::Vector2f max = Eigen::Vector2f::Constant(
        std::numeric_limits<float>::lowest());

    auto width()  const -> float { return max.x() - min.x(); }
    auto height() const -> float { return max.y() - min.y(); }
    auto center() const -> Eigen::Vector2f { return (min + max) * 0.5f; }
    auto expanded(float margin) const -> BoundingBox2D;

    template <typename DerivedX, typename DerivedY>
    static auto fromXY(const Eigen::MatrixBase<DerivedX>& x,
                       const Eigen::MatrixBase<DerivedY>& y) -> BoundingBox2D {
        BoundingBox2D bb;
        bb.min.x() = static_cast<float>(x.derived().minCoeff());
        bb.max.x() = static_cast<float>(x.derived().maxCoeff());
        bb.min.y() = static_cast<float>(y.derived().minCoeff());
        bb.max.y() = static_cast<float>(y.derived().maxCoeff());
        return bb;
    }
};

// ── 3D Bounding Box ──────────────────────────────────────────────────────

struct SKIGENPLOT_EXPORT BoundingBox3D {
    Eigen::Vector3f min = Eigen::Vector3f::Constant(
        std::numeric_limits<float>::max());
    Eigen::Vector3f max = Eigen::Vector3f::Constant(
        std::numeric_limits<float>::lowest());

    auto diagonal() const -> float;
    auto center()   const -> Eigen::Vector3f { return (min + max) * 0.5f; }
    auto expanded(float margin) const -> BoundingBox3D;

    template <typename Derived>
    static auto fromVertices(const Eigen::MatrixBase<Derived>& verts)
        -> BoundingBox3D
    {
        BoundingBox3D bb;
        for (Eigen::Index i = 0; i < verts.rows(); ++i) {
            for (int a = 0; a < 3; ++a) {
                auto v = static_cast<float>(verts(i, a));
                if (v < bb.min[a]) bb.min[a] = v;
                if (v > bb.max[a]) bb.max[a] = v;
            }
        }
        return bb;
    }
};

// ── Orthographic projection (2D data → NDC) ─────────────────────────────

SKIGENPLOT_EXPORT
auto orthoProjection(const BoundingBox2D& bounds,
                     float margin = 0.05f) -> Eigen::Matrix4f;

// ── 3D Camera ────────────────────────────────────────────────────────────

class SKIGENPLOT_EXPORT Camera3D {
public:
    Camera3D();

    void lookAt(const Eigen::Vector3f& eye,
                const Eigen::Vector3f& target,
                const Eigen::Vector3f& up = Eigen::Vector3f::UnitY());

    void setPerspective(float fovDegrees, float aspect,
                        float nearPlane, float farPlane);

    auto viewMatrix()           const -> Eigen::Matrix4f;
    auto projectionMatrix()     const -> Eigen::Matrix4f;
    auto viewProjectionMatrix() const -> Eigen::Matrix4f;

    auto position() const -> const Eigen::Vector3f& { return m_eye; }
    auto target()   const -> const Eigen::Vector3f& { return m_target; }

private:
    Eigen::Vector3f m_eye    {0.f, 0.f, 5.f};
    Eigen::Vector3f m_target {0.f, 0.f, 0.f};
    Eigen::Vector3f m_up     {0.f, 1.f, 0.f};
    float m_fov    = 45.f;
    float m_aspect = 1.f;
    float m_near   = 0.1f;
    float m_far    = 100.f;
};

// ── Normalization ────────────────────────────────────────────────────────

template <typename Derived>
auto normalizeMinMax(const Eigen::MatrixBase<Derived>& data)
    -> Eigen::MatrixXf
{
    float lo = static_cast<float>(data.derived().minCoeff());
    float hi = static_cast<float>(data.derived().maxCoeff());
    float range = hi - lo;
    if (range < std::numeric_limits<float>::epsilon())
        return Eigen::MatrixXf::Zero(data.rows(), data.cols());
    return (data.derived().template cast<float>().array() - lo) / range;
}

} // namespace Skigen::Plot
