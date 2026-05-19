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
    t.background = {0.047f, 0.039f, 0.102f, 1.0f};  // #0c0a1a
    t.gridColor  = {0.424f, 0.361f, 0.620f, 0.18f};
    t.axisColor  = {0.729f, 0.678f, 0.918f, 0.58f};
    t.textColor  = {0.925f, 0.914f, 0.984f, 1.0f};
    t.seriesColors = {{
        {0.024f, 0.714f, 0.831f, 0.98f}, // #06b6d4 cyan
        {0.549f, 0.361f, 0.965f, 0.98f}, // #8b5cf6 violet
        {0.204f, 0.827f, 0.600f, 0.98f}, // #34d399 emerald
        {0.937f, 0.341f, 0.196f, 0.98f}, // #ef5732 vermillion
        {0.984f, 0.749f, 0.141f, 0.98f}, // #fbbf24 amber
        {0.376f, 0.647f, 0.980f, 0.98f}, // #60a5fa blue
    }};
    return t;
}

auto Theme::light() -> Theme {
    Theme t;
    t.background = {0.984f, 0.988f, 0.996f, 1.0f};  // #fbfcfe
    t.gridColor  = {0.506f, 0.569f, 0.682f, 0.22f};
    t.axisColor  = {0.235f, 0.282f, 0.376f, 0.58f};
    t.textColor  = {0.118f, 0.161f, 0.231f, 1.0f};
    t.seriesColors = {{
        {0.000f, 0.549f, 0.663f, 0.98f}, // #008ca9 deep cyan
        {0.431f, 0.192f, 0.855f, 0.98f}, // #6e31da violet
        {0.035f, 0.584f, 0.408f, 0.98f}, // #099568 emerald
        {0.765f, 0.188f, 0.125f, 0.98f}, // #c33020 vermillion
        {0.776f, 0.482f, 0.000f, 0.98f}, // #c67b00 amber
        {0.145f, 0.388f, 0.922f, 0.98f}, // #2563eb blue
    }};
    return t;
}

} // namespace Skigen::Plot
