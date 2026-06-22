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

// ── Colormaps ────────────────────────────────────────────────────────────

namespace {

// Piecewise-linear interpolation over a small table of RGB anchor points
// sampled at evenly-spaced positions in [0, 1].
template <std::size_t N>
auto interpAnchors(const std::array<Eigen::Vector3f, N>& anchors, float t)
    -> Eigen::Vector4f {
    t = std::clamp(t, 0.0f, 1.0f);
    float scaled = t * static_cast<float>(N - 1);
    auto i0 = static_cast<std::size_t>(scaled);
    if (i0 >= N - 1) return {anchors[N - 1].x(), anchors[N - 1].y(),
                             anchors[N - 1].z(), 1.0f};
    float f = scaled - static_cast<float>(i0);
    Eigen::Vector3f c = anchors[i0] * (1.0f - f) + anchors[i0 + 1] * f;
    return {c.x(), c.y(), c.z(), 1.0f};
}

// MATLAB-classic colormaps (jet/hot/cool/bone), ported from MNE-CPP's
// DISPLIB ColorMap fuzzy-set formulation. linearSlope(x, m, b) = m*x + b,
// clamped to [0, 1]; channels are evaluated analytically (no LUT).
float linearSlope(float x, float m, float b) {
    return std::clamp(m * x + b, 0.0f, 1.0f);
}

Eigen::Vector4f jetColor(float x) {
    float r, g, bch;
    // Red
    if (x < 0.375f) r = 0.0f;
    else if (x < 0.625f) r = linearSlope(x, 4.0f, -1.5f);
    else if (x < 0.875f) r = 1.0f;
    else r = linearSlope(x, -4.0f, 4.5f);
    // Green
    if (x < 0.125f) g = 0.0f;
    else if (x < 0.375f) g = linearSlope(x, 4.0f, -0.5f);
    else if (x < 0.625f) g = 1.0f;
    else if (x < 0.875f) g = linearSlope(x, -4.0f, 3.5f);
    else g = 0.0f;
    // Blue
    if (x < 0.125f) bch = linearSlope(x, 4.0f, 0.5f);
    else if (x < 0.375f) bch = 1.0f;
    else if (x < 0.625f) bch = linearSlope(x, -4.0f, 2.5f);
    else bch = 0.0f;
    return {r, g, bch, 1.0f};
}

Eigen::Vector4f hotColor(float x) {
    float r = (x < 0.375f) ? linearSlope(x, 2.5621f, 0.0392f) : 1.0f;
    float g;
    if (x < 0.375f) g = 0.0f;
    else if (x < 0.75f) g = linearSlope(x, 2.6667f, -1.0f);
    else g = 1.0f;
    float bch = (x < 0.75f) ? 0.0f : linearSlope(x, 4.0f, -3.0f);
    return {r, g, bch, 1.0f};
}

Eigen::Vector4f coolColor(float x) {
    return {linearSlope(x, 1.0f, 0.0f), linearSlope(x, -1.0f, 1.0f), 1.0f, 1.0f};
}

Eigen::Vector4f boneColor(float x) {
    float r, g, bch;
    if (x < 0.375f) r = linearSlope(x, 0.8471f, 0.0f);
    else if (x < 0.75f) r = linearSlope(x, 0.8889f, -0.0157f);
    else r = linearSlope(x, 1.396f, -0.396f);
    if (x < 0.375f) g = linearSlope(x, 0.8471f, 0.0f);
    else if (x < 0.75f) g = linearSlope(x, 1.2237f, -0.1413f);
    else g = linearSlope(x, 0.894f, 0.106f);
    if (x < 0.375f) bch = linearSlope(x, 1.1712f, 0.0039f);
    else if (x < 0.75f) bch = linearSlope(x, 0.8889f, 0.1098f);
    else bch = linearSlope(x, 0.8941f, 0.1059f);
    return {r, g, bch, 1.0f};
}

} // namespace

auto sampleColormap(Colormap map, float t) -> Eigen::Vector4f {
    switch (map) {
        case Colormap::Viridis: {
            // matplotlib `viridis` anchors (dark blue → green → yellow).
            static const std::array<Eigen::Vector3f, 6> a = {{
                {0.267f, 0.005f, 0.329f}, {0.283f, 0.141f, 0.458f},
                {0.254f, 0.265f, 0.530f}, {0.164f, 0.471f, 0.558f},
                {0.135f, 0.659f, 0.518f}, {0.993f, 0.906f, 0.144f},
            }};
            return interpAnchors(a, t);
        }
        case Colormap::Magma: {
            static const std::array<Eigen::Vector3f, 6> a = {{
                {0.001f, 0.000f, 0.014f}, {0.232f, 0.060f, 0.438f},
                {0.551f, 0.161f, 0.506f}, {0.870f, 0.288f, 0.409f},
                {0.987f, 0.591f, 0.385f}, {0.987f, 0.991f, 0.749f},
            }};
            return interpAnchors(a, t);
        }
        case Colormap::Plasma: {
            static const std::array<Eigen::Vector3f, 6> a = {{
                {0.050f, 0.030f, 0.528f}, {0.417f, 0.000f, 0.658f},
                {0.692f, 0.165f, 0.564f}, {0.881f, 0.392f, 0.383f},
                {0.988f, 0.652f, 0.211f}, {0.940f, 0.975f, 0.131f},
            }};
            return interpAnchors(a, t);
        }
        case Colormap::Coolwarm: {
            static const std::array<Eigen::Vector3f, 3> a = {{
                {0.230f, 0.299f, 0.754f}, {0.865f, 0.865f, 0.865f},
                {0.706f, 0.016f, 0.150f},
            }};
            return interpAnchors(a, t);
        }
        case Colormap::Jet:  return jetColor(std::clamp(t, 0.0f, 1.0f));
        case Colormap::Hot:  return hotColor(std::clamp(t, 0.0f, 1.0f));
        case Colormap::Cool: return coolColor(std::clamp(t, 0.0f, 1.0f));
        case Colormap::Bone: return boneColor(std::clamp(t, 0.0f, 1.0f));
        case Colormap::Gray:
        default: {
            float g = std::clamp(t, 0.0f, 1.0f);
            return {g, g, g, 1.0f};
        }
    }
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

// ── Palettes ──────────────────────────────────────────────────────────────

auto seriesPalette(Palette palette) -> std::array<Eigen::Vector4f, 6> {
    switch (palette) {
        case Palette::Matplotlib:
            // Canonical matplotlib `tab10`, first six colours.
            return {{
                {0.122f, 0.467f, 0.706f, 1.0f}, // #1f77b4 tab:blue
                {1.000f, 0.498f, 0.055f, 1.0f}, // #ff7f0e tab:orange
                {0.173f, 0.627f, 0.173f, 1.0f}, // #2ca02c tab:green
                {0.839f, 0.153f, 0.157f, 1.0f}, // #d62728 tab:red
                {0.580f, 0.404f, 0.741f, 1.0f}, // #9467bd tab:purple
                {0.549f, 0.337f, 0.294f, 1.0f}, // #8c564b tab:brown
            }};
        case Palette::Skigen:
        default:
            // Vivid Skigen categorical palette (tuned for both dark and
            // light backgrounds).
            return {{
                {0.000f, 0.549f, 0.663f, 0.98f}, // #008ca9 deep cyan
                {0.431f, 0.192f, 0.855f, 0.98f}, // #6e31da violet
                {0.035f, 0.584f, 0.408f, 0.98f}, // #099568 emerald
                {0.765f, 0.188f, 0.125f, 0.98f}, // #c33020 vermillion
                {0.776f, 0.482f, 0.000f, 0.98f}, // #c67b00 amber
                {0.145f, 0.388f, 0.922f, 0.98f}, // #2563eb blue
            }};
    }
}

auto Theme::withPalette(Palette palette) const -> Theme {
    Theme t = *this;
    t.seriesColors = seriesPalette(palette);
    return t;
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

auto Theme::paper() -> Theme {
    // Clean matplotlib-style layout — white figure, light-gray gridlines,
    // muted dark-gray spines/text — paired with Skigen's own vivid
    // cyan/violet/emerald categorical palette (more qualitative on white
    // than the muted `tab10` defaults).
    Theme t;
    t.background = {1.000f, 1.000f, 1.000f, 1.0f};  // white #ffffff
    t.gridColor  = {0.690f, 0.690f, 0.690f, 0.42f}; // #b0b0b0 @ ~0.4 alpha
    t.axisColor  = {0.150f, 0.150f, 0.150f, 0.85f}; // #262626 spine gray
    t.textColor  = {0.150f, 0.150f, 0.150f, 1.0f};  // #262626 near-black
    t.seriesColors = seriesPalette(Palette::Skigen);
    return t;
}

auto Theme::matplotlibStyle() -> Theme {
    // The clean paper() layout with matplotlib's own `tab10` palette, so the
    // result matches `matplotlib.pyplot` defaults as closely as possible.
    return paper().withPalette(Palette::Matplotlib);
}

} // namespace Skigen::Plot
