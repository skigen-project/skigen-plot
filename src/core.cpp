#include <skigen/plot/core.h>
#include <skigen/plot/theme.h>

#include <algorithm>
#include <cmath>

namespace Skigen::Plot {

// ── BoundingBox2D ────────────────────────────────────────────────────────

auto BoundingBox2D::expanded(float margin) const -> BoundingBox2D {
    float mx = width()  * margin;
    float my = height() * margin;
    return {min - Eigen::Vector2f(mx, my),
            max + Eigen::Vector2f(mx, my)};
}

auto BoundingBox2D::merge(const BoundingBox2D& other) const -> BoundingBox2D {
    BoundingBox2D result;
    result.min.x() = std::min(min.x(), other.min.x());
    result.min.y() = std::min(min.y(), other.min.y());
    result.max.x() = std::max(max.x(), other.max.x());
    result.max.y() = std::max(max.y(), other.max.y());
    return result;
}

// ── BoundingBox3D ────────────────────────────────────────────────────────

auto BoundingBox3D::diagonal() const -> float {
    return (max - min).norm();
}

auto BoundingBox3D::expanded(float margin) const -> BoundingBox3D {
    float d = diagonal() * margin;
    Eigen::Vector3f pad = Eigen::Vector3f::Constant(d);
    return {min - pad, max + pad};
}

// ── Orthographic projection ──────────────────────────────────────────────

auto orthoProjection(const BoundingBox2D& bounds,
                     float margin) -> Eigen::Matrix4f
{
    auto b = bounds.expanded(margin);
    float l = b.min.x(), r = b.max.x();
    float bot = b.min.y(), top = b.max.y();

    Eigen::Matrix4f m = Eigen::Matrix4f::Zero();
    m(0, 0) =  2.f / (r - l);
    m(0, 3) = -(r + l) / (r - l);
    m(1, 1) =  2.f / (top - bot);
    m(1, 3) = -(top + bot) / (top - bot);
    m(2, 2) = -1.f;
    m(3, 3) =  1.f;
    return m;
}

// ── Camera3D ─────────────────────────────────────────────────────────────

Camera3D::Camera3D() = default;

void Camera3D::lookAt(const Eigen::Vector3f& eye,
                      const Eigen::Vector3f& target,
                      const Eigen::Vector3f& up) {
    m_eye    = eye;
    m_target = target;
    m_up     = up;
}

void Camera3D::setPerspective(float fovDegrees, float aspect,
                              float nearPlane, float farPlane) {
    m_fov    = fovDegrees;
    m_aspect = aspect;
    m_near   = nearPlane;
    m_far    = farPlane;
}

auto Camera3D::viewMatrix() const -> Eigen::Matrix4f {
    Eigen::Vector3f f = (m_target - m_eye).normalized();
    Eigen::Vector3f r = f.cross(m_up).normalized();
    Eigen::Vector3f u = r.cross(f);

    Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
    m(0, 0) =  r.x(); m(0, 1) =  r.y(); m(0, 2) =  r.z();
    m(1, 0) =  u.x(); m(1, 1) =  u.y(); m(1, 2) =  u.z();
    m(2, 0) = -f.x(); m(2, 1) = -f.y(); m(2, 2) = -f.z();
    m(0, 3) = -r.dot(m_eye);
    m(1, 3) = -u.dot(m_eye);
    m(2, 3) =  f.dot(m_eye);
    return m;
}

auto Camera3D::projectionMatrix() const -> Eigen::Matrix4f {
    float t = std::tan(m_fov * 0.5f * std::numbers::pi_v<float> / 180.f);

    Eigen::Matrix4f m = Eigen::Matrix4f::Zero();
    m(0, 0) = 1.f / (m_aspect * t);
    m(1, 1) = 1.f / t;
    m(2, 2) = -(m_far + m_near) / (m_far - m_near);
    m(2, 3) = -(2.f * m_far * m_near) / (m_far - m_near);
    m(3, 2) = -1.f;
    return m;
}

auto Camera3D::viewProjectionMatrix() const -> Eigen::Matrix4f {
    return projectionMatrix() * viewMatrix();
}

// ── Tick computation ────────────────────────────────────────────────────

static auto niceNumber(float x, bool round) -> float {
    float exp = std::floor(std::log10(x));
    float frac = x / std::pow(10.f, exp);
    float nice;
    if (round) {
        if (frac < 1.5f)      nice = 1.f;
        else if (frac < 3.f)  nice = 2.f;
        else if (frac < 7.f)  nice = 5.f;
        else                  nice = 10.f;
    } else {
        if (frac <= 1.f)      nice = 1.f;
        else if (frac <= 2.f) nice = 2.f;
        else if (frac <= 5.f) nice = 5.f;
        else                  nice = 10.f;
    }
    return nice * std::pow(10.f, exp);
}

auto computeTicks(float lo, float hi, int maxTicks) -> TickResult {
    TickResult result;
    float range = hi - lo;
    if (range < std::numeric_limits<float>::epsilon()) {
        result.ticks.push_back(lo);
        result.spacing = 1.f;
        return result;
    }
    float niceRange = niceNumber(range, false);
    result.spacing = niceNumber(niceRange / static_cast<float>(maxTicks - 1), true);
    float tickMin = std::floor(lo / result.spacing) * result.spacing;
    float tickMax = std::ceil(hi / result.spacing) * result.spacing;
    for (float t = tickMin; t <= tickMax + result.spacing * 0.5f; t += result.spacing)
        result.ticks.push_back(t);
    return result;
}

// ── Vertex normal computation ───────────────────────────────────────────

auto computeVertexNormals(std::span<const float> vertices, int vertexCount,
                          std::span<const uint32_t> indices, int triangleCount)
    -> std::vector<float>
{
    std::vector<float> normals(static_cast<std::size_t>(vertexCount) * 3, 0.f);

    for (int t = 0; t < triangleCount; ++t) {
        auto ti = static_cast<std::size_t>(t) * 3;
        uint32_t i0 = indices[ti], i1 = indices[ti + 1], i2 = indices[ti + 2];

        Eigen::Vector3f v0(vertices[i0 * 3], vertices[i0 * 3 + 1], vertices[i0 * 3 + 2]);
        Eigen::Vector3f v1(vertices[i1 * 3], vertices[i1 * 3 + 1], vertices[i1 * 3 + 2]);
        Eigen::Vector3f v2(vertices[i2 * 3], vertices[i2 * 3 + 1], vertices[i2 * 3 + 2]);

        Eigen::Vector3f fn = (v1 - v0).cross(v2 - v0);

        for (uint32_t idx : {i0, i1, i2}) {
            normals[idx * 3]     += fn.x();
            normals[idx * 3 + 1] += fn.y();
            normals[idx * 3 + 2] += fn.z();
        }
    }

    std::vector<float> interleaved(static_cast<std::size_t>(vertexCount) * 6);
    for (int i = 0; i < vertexCount; ++i) {
        auto vi = static_cast<std::size_t>(i);
        Eigen::Vector3f n(normals[vi * 3], normals[vi * 3 + 1], normals[vi * 3 + 2]);
        float len = n.norm();
        if (len > 1e-8f) n /= len;
        else n = Eigen::Vector3f(0.f, 1.f, 0.f);

        interleaved[vi * 6]     = vertices[vi * 3];
        interleaved[vi * 6 + 1] = vertices[vi * 3 + 1];
        interleaved[vi * 6 + 2] = vertices[vi * 3 + 2];
        interleaved[vi * 6 + 3] = n.x();
        interleaved[vi * 6 + 4] = n.y();
        interleaved[vi * 6 + 5] = n.z();
    }
    return interleaved;
}

// ── Theme presets ───────────────────────────────────────────────────────

auto Theme::dark() -> Theme {
    Theme t;
    t.background = {0.035f, 0.031f, 0.082f, 1.0f};
    t.gridColor  = {0.12f, 0.12f, 0.22f, 0.25f};
    t.axisColor  = {0.30f, 0.30f, 0.48f, 0.7f};
    t.textColor  = {0.85f, 0.85f, 0.90f, 1.0f};
    t.seriesColors = {{
        {0.024f, 0.714f, 0.831f, 1.0f},  // #06b6d4 cyan
        {0.486f, 0.228f, 0.929f, 1.0f},  // #7c3aed violet
        {0.659f, 0.333f, 0.969f, 1.0f},  // #a855f7 purple
        {0.957f, 0.447f, 0.714f, 1.0f},  // #f472b6 pink
        {0.204f, 0.827f, 0.600f, 1.0f},  // #34d399 emerald
        {0.984f, 0.749f, 0.141f, 1.0f},  // #fbbf24 amber
    }};
    return t;
}

auto Theme::light() -> Theme {
    Theme t;
    t.background = {0.976f, 0.980f, 0.992f, 1.0f};
    t.gridColor  = {0.85f, 0.87f, 0.92f, 0.35f};
    t.axisColor  = {0.35f, 0.40f, 0.50f, 0.8f};
    t.textColor  = {0.15f, 0.23f, 0.33f, 1.0f};
    t.seriesColors = {{
        {0.016f, 0.565f, 0.659f, 1.0f},  // darker cyan
        {0.376f, 0.176f, 0.718f, 1.0f},  // darker violet
        {0.498f, 0.208f, 0.776f, 1.0f},  // darker purple
        {0.839f, 0.267f, 0.553f, 1.0f},  // darker pink
        {0.122f, 0.639f, 0.459f, 1.0f},  // darker emerald
        {0.816f, 0.584f, 0.047f, 1.0f},  // darker amber
    }};
    return t;
}

} // namespace Skigen::Plot
