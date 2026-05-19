#include <skigen/plot/core.h>

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

} // namespace Skigen::Plot
