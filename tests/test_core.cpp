// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#include <skigen/plot/core.h>
#include <skigen/plot/theme.h>

#include <Eigen/Core>

#include <cmath>
#include <functional>
#include <iostream>
#include <numbers>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Minimal test harness (matches skigen convention)
// ---------------------------------------------------------------------------

static int g_passed = 0;
static int g_failed = 0;

struct TestFailure : std::exception {
    std::string msg;
    TestFailure(std::string m) : msg(std::move(m)) {}
    const char* what() const noexcept override { return msg.c_str(); }
};

#define ASSERT_TRUE(cond)                                                     \
    do {                                                                       \
        if (!(cond)) {                                                         \
            throw TestFailure(std::string(__FILE__) + ":" +                    \
                              std::to_string(__LINE__) + ": ASSERT_TRUE(" +    \
                              #cond + ") failed");                             \
        }                                                                      \
    } while (false)

template <typename Scalar>
void assert_near(Scalar a, Scalar b, Scalar tol, const char* file, int line) {
    if (std::abs(a - b) > tol) {
        std::ostringstream oss;
        oss << file << ":" << line << ": ASSERT_NEAR failed: " << a
            << " vs " << b << " (tol=" << tol << ")";
        throw TestFailure(oss.str());
    }
}
#define ASSERT_NEAR(a, b, tol) assert_near(a, b, tol, __FILE__, __LINE__)

template <typename VecA, typename VecB, typename Scalar>
void assert_vector_near(const VecA& A, const VecB& B, Scalar tol,
                        const char* file, int line) {
    if (A.size() != B.size())
        throw TestFailure(std::string(file) + ":" + std::to_string(line) +
                          ": Size mismatch");
    for (Eigen::Index i = 0; i < A.size(); ++i)
        assert_near(A(i), B(i), tol, file, line);
}
#define ASSERT_VECTOR_NEAR(A, B, tol) assert_vector_near(A, B, tol, __FILE__, __LINE__)

static void run_test(const std::string& name, std::function<void()> fn) {
    try {
        fn();
        ++g_passed;
        std::cout << "  PASS  " << name << "\n";
    } catch (const TestFailure& e) {
        ++g_failed;
        std::cout << "  FAIL  " << name << "\n        " << e.what() << "\n";
    } catch (const std::exception& e) {
        ++g_failed;
        std::cout << "  FAIL  " << name << "\n        exception: " << e.what()
                  << "\n";
    }
}

// ---------------------------------------------------------------------------
// BoundingBox2D tests
// ---------------------------------------------------------------------------

void test_bbox2d_from_xy() {
    Eigen::VectorXf x(4);
    x << 1.f, 3.f, -2.f, 5.f;
    Eigen::VectorXf y(4);
    y << 0.f, 4.f, -1.f, 2.f;

    auto bb = Skigen::Plot::BoundingBox2D::fromXY(x, y);
    ASSERT_NEAR(bb.min.x(), -2.f, 1e-6f);
    ASSERT_NEAR(bb.max.x(),  5.f, 1e-6f);
    ASSERT_NEAR(bb.min.y(), -1.f, 1e-6f);
    ASSERT_NEAR(bb.max.y(),  4.f, 1e-6f);
}

void test_bbox2d_width_height_center() {
    Skigen::Plot::BoundingBox2D bb;
    bb.min = Eigen::Vector2f(1.f, 2.f);
    bb.max = Eigen::Vector2f(5.f, 8.f);

    ASSERT_NEAR(bb.width(), 4.f, 1e-6f);
    ASSERT_NEAR(bb.height(), 6.f, 1e-6f);
    ASSERT_NEAR(bb.center().x(), 3.f, 1e-6f);
    ASSERT_NEAR(bb.center().y(), 5.f, 1e-6f);
}

void test_bbox2d_expanded() {
    Skigen::Plot::BoundingBox2D bb;
    bb.min = Eigen::Vector2f(0.f, 0.f);
    bb.max = Eigen::Vector2f(10.f, 20.f);

    auto expanded = bb.expanded(0.1f);
    ASSERT_NEAR(expanded.min.x(), -1.f, 1e-6f);
    ASSERT_NEAR(expanded.max.x(), 11.f, 1e-6f);
    ASSERT_NEAR(expanded.min.y(), -2.f, 1e-6f);
    ASSERT_NEAR(expanded.max.y(), 22.f, 1e-6f);

    auto center = expanded.center();
    ASSERT_NEAR(center.x(), 5.f, 1e-6f);
    ASSERT_NEAR(center.y(), 10.f, 1e-6f);
}

// ---------------------------------------------------------------------------
// BoundingBox3D tests
// ---------------------------------------------------------------------------

void test_bbox3d_from_vertices() {
    Eigen::MatrixXf verts(4, 3);
    verts << 0.f, 0.f, 0.f,
             1.f, 2.f, 3.f,
            -1.f, 0.f, 1.f,
             0.5f, 1.f, -0.5f;

    auto bb = Skigen::Plot::BoundingBox3D::fromVertices(verts);
    ASSERT_NEAR(bb.min.x(), -1.f, 1e-6f);
    ASSERT_NEAR(bb.max.x(),  1.f, 1e-6f);
    ASSERT_NEAR(bb.min.y(),  0.f, 1e-6f);
    ASSERT_NEAR(bb.max.y(),  2.f, 1e-6f);
    ASSERT_NEAR(bb.min.z(), -0.5f, 1e-6f);
    ASSERT_NEAR(bb.max.z(),  3.f, 1e-6f);
}

void test_bbox3d_diagonal() {
    Skigen::Plot::BoundingBox3D bb;
    bb.min = Eigen::Vector3f(0.f, 0.f, 0.f);
    bb.max = Eigen::Vector3f(1.f, 1.f, 1.f);

    ASSERT_NEAR(bb.diagonal(), std::sqrt(3.f), 1e-5f);
}

void test_bbox3d_center() {
    Skigen::Plot::BoundingBox3D bb;
    bb.min = Eigen::Vector3f(-3.f, -2.f, -1.f);
    bb.max = Eigen::Vector3f(3.f, 2.f, 1.f);

    auto c = bb.center();
    ASSERT_NEAR(c.x(), 0.f, 1e-6f);
    ASSERT_NEAR(c.y(), 0.f, 1e-6f);
    ASSERT_NEAR(c.z(), 0.f, 1e-6f);
}

void test_bbox3d_expanded() {
    Skigen::Plot::BoundingBox3D bb;
    bb.min = Eigen::Vector3f(0.f, 0.f, 0.f);
    bb.max = Eigen::Vector3f(1.f, 1.f, 1.f);

    auto expanded = bb.expanded(0.5f);
    float pad = bb.diagonal() * 0.5f;
    ASSERT_NEAR(expanded.min.x(), -pad, 1e-5f);
    ASSERT_NEAR(expanded.max.x(), 1.f + pad, 1e-5f);
}

// ---------------------------------------------------------------------------
// Projection tests
// ---------------------------------------------------------------------------

void test_ortho_projection_unit_box() {
    Skigen::Plot::BoundingBox2D bb;
    bb.min = Eigen::Vector2f(-1.f, -1.f);
    bb.max = Eigen::Vector2f(1.f, 1.f);

    auto m = Skigen::Plot::orthoProjection(bb, 0.f);
    Eigen::Vector4f corner(1.f, 1.f, 0.f, 1.f);
    Eigen::Vector4f ndc = m * corner;
    ASSERT_NEAR(ndc.x(), 1.f, 1e-5f);
    ASSERT_NEAR(ndc.y(), 1.f, 1e-5f);
}

void test_ortho_projection_with_margin() {
    Skigen::Plot::BoundingBox2D bb;
    bb.min = Eigen::Vector2f(0.f, 0.f);
    bb.max = Eigen::Vector2f(10.f, 10.f);

    auto m = Skigen::Plot::orthoProjection(bb, 0.1f);
    Eigen::Vector4f center(5.f, 5.f, 0.f, 1.f);
    Eigen::Vector4f ndc = m * center;
    ASSERT_NEAR(ndc.x(), 0.f, 1e-5f);
    ASSERT_NEAR(ndc.y(), 0.f, 1e-5f);
}

// ---------------------------------------------------------------------------
// Camera3D tests
// ---------------------------------------------------------------------------

void test_camera3d_default_position() {
    Skigen::Plot::Camera3D cam;
    ASSERT_NEAR(cam.position().z(), 5.f, 1e-6f);
    ASSERT_NEAR(cam.target().x(), 0.f, 1e-6f);
}

void test_camera3d_look_at_orthonormal() {
    Skigen::Plot::Camera3D cam;
    cam.lookAt({3.f, 4.f, 5.f}, {0.f, 0.f, 0.f});

    auto V = cam.viewMatrix();
    Eigen::Matrix3f R = V.block<3, 3>(0, 0);
    Eigen::Matrix3f identity = R * R.transpose();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            ASSERT_NEAR(identity(i, j), (i == j) ? 1.f : 0.f, 1e-5f);
}

void test_camera3d_perspective_fov() {
    Skigen::Plot::Camera3D cam;
    cam.setPerspective(90.f, 1.f, 0.1f, 100.f);
    auto P = cam.projectionMatrix();
    float expected = 1.f / std::tan(45.f * std::numbers::pi_v<float> / 180.f);
    ASSERT_NEAR(P(0, 0), expected, 1e-5f);
    ASSERT_NEAR(P(1, 1), expected, 1e-5f);
}

void test_camera3d_vp_product() {
    Skigen::Plot::Camera3D cam;
    cam.lookAt({1.f, 2.f, 3.f}, {0.f, 0.f, 0.f});
    cam.setPerspective(60.f, 1.5f, 0.1f, 50.f);

    auto VP = cam.viewProjectionMatrix();
    Eigen::Matrix4f VP_manual = cam.projectionMatrix() * cam.viewMatrix();

    for (int i = 0; i < 16; ++i)
        ASSERT_NEAR(VP.data()[i], VP_manual.data()[i], 1e-5f);
}

// ---------------------------------------------------------------------------
// Normalization tests
// ---------------------------------------------------------------------------

void test_normalize_min_max_basic() {
    Eigen::VectorXf v(3);
    v << 1.f, 3.f, 5.f;
    auto n = Skigen::Plot::normalizeMinMax(v);
    ASSERT_NEAR(n(0), 0.f, 1e-6f);
    ASSERT_NEAR(n(1), 0.5f, 1e-6f);
    ASSERT_NEAR(n(2), 1.f, 1e-6f);
}

void test_normalize_min_max_constant() {
    Eigen::VectorXf v = Eigen::VectorXf::Constant(5, 3.f);
    auto n = Skigen::Plot::normalizeMinMax(v);
    for (Eigen::Index i = 0; i < n.size(); ++i)
        ASSERT_NEAR(n(i), 0.f, 1e-6f);
}

// ---------------------------------------------------------------------------
// mapRange tests
// ---------------------------------------------------------------------------

void test_map_range() {
    std::vector<float> data = {1.f, 2.f, 3.f, 4.f, 5.f};
    auto mapped = Skigen::Plot::mapRange(data);
    ASSERT_TRUE(mapped.size() == 5);
    ASSERT_NEAR(mapped(0), 1.f, 1e-6f);
    ASSERT_NEAR(mapped(4), 5.f, 1e-6f);
}

// ---------------------------------------------------------------------------
// Tick computation tests
// ---------------------------------------------------------------------------

void test_compute_ticks_basic() {
    auto result = Skigen::Plot::computeTicks(0.f, 10.f);
    ASSERT_TRUE(result.ticks.size() >= 3);
    ASSERT_NEAR(result.ticks.front(), 0.f, 1e-6f);
    ASSERT_NEAR(result.ticks.back(), 10.f, 1e-6f);
    for (std::size_t i = 1; i < result.ticks.size(); ++i)
        ASSERT_NEAR(result.ticks[i] - result.ticks[i - 1], result.spacing, 1e-5f);
}

void test_compute_ticks_symmetric() {
    auto result = Skigen::Plot::computeTicks(-5.f, 5.f);
    ASSERT_TRUE(result.ticks.size() >= 3);
    bool hasZero = false;
    for (float t : result.ticks)
        if (std::abs(t) < 1e-5f) hasZero = true;
    ASSERT_TRUE(hasZero);
}

void test_compute_ticks_small_range() {
    auto result = Skigen::Plot::computeTicks(0.001f, 0.009f);
    ASSERT_TRUE(result.ticks.size() >= 2);
    ASSERT_TRUE(result.spacing > 0.f);
}

void test_compute_ticks_degenerate() {
    auto result = Skigen::Plot::computeTicks(3.f, 3.f);
    ASSERT_TRUE(result.ticks.size() >= 1);
    ASSERT_NEAR(result.ticks[0], 3.f, 1e-6f);
}

// ---------------------------------------------------------------------------
// Vertex normal tests
// ---------------------------------------------------------------------------

void test_vertex_normals_single_triangle() {
    std::vector<float> verts = {0.f, 0.f, 0.f,  1.f, 0.f, 0.f,  0.f, 1.f, 0.f};
    std::vector<uint32_t> idx = {0, 1, 2};
    auto result = Skigen::Plot::computeVertexNormals(verts, 3, idx, 1);
    ASSERT_TRUE(result.size() == 18);
    // All normals should point in +Z direction
    for (int i = 0; i < 3; ++i) {
        ASSERT_NEAR(result[static_cast<std::size_t>(i) * 6 + 3], 0.f, 1e-5f);
        ASSERT_NEAR(result[static_cast<std::size_t>(i) * 6 + 4], 0.f, 1e-5f);
        ASSERT_NEAR(result[static_cast<std::size_t>(i) * 6 + 5], 1.f, 1e-5f);
    }
}

void test_vertex_normals_preserves_positions() {
    std::vector<float> verts = {1.f, 2.f, 3.f,  4.f, 5.f, 6.f,  7.f, 8.f, 9.f};
    std::vector<uint32_t> idx = {0, 1, 2};
    auto result = Skigen::Plot::computeVertexNormals(verts, 3, idx, 1);
    for (int i = 0; i < 3; ++i) {
        auto vi = static_cast<std::size_t>(i);
        ASSERT_NEAR(result[vi * 6],     verts[vi * 3],     1e-6f);
        ASSERT_NEAR(result[vi * 6 + 1], verts[vi * 3 + 1], 1e-6f);
        ASSERT_NEAR(result[vi * 6 + 2], verts[vi * 3 + 2], 1e-6f);
    }
}

// ---------------------------------------------------------------------------
// Theme tests
// ---------------------------------------------------------------------------

void test_theme_dark() {
    auto t = Skigen::Plot::Theme::dark();
    ASSERT_TRUE(t.background.norm() > 0.f);
    ASSERT_TRUE(t.seriesColors[0].norm() > 0.f);
}

void test_theme_light() {
    auto t = Skigen::Plot::Theme::light();
    ASSERT_TRUE(t.background.x() > 0.9f);
    ASSERT_TRUE(t.seriesColors[0].norm() > 0.f);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    std::cout << "SkigenPlot — core math tests\n";
    std::cout << std::string(40, '-') << "\n";

    run_test("bbox2d_from_xy",             test_bbox2d_from_xy);
    run_test("bbox2d_width_height_center", test_bbox2d_width_height_center);
    run_test("bbox2d_expanded",            test_bbox2d_expanded);
    run_test("bbox3d_from_vertices",       test_bbox3d_from_vertices);
    run_test("bbox3d_diagonal",            test_bbox3d_diagonal);
    run_test("bbox3d_center",              test_bbox3d_center);
    run_test("bbox3d_expanded",            test_bbox3d_expanded);
    run_test("ortho_unit_box",             test_ortho_projection_unit_box);
    run_test("ortho_with_margin",          test_ortho_projection_with_margin);
    run_test("camera3d_default_position",  test_camera3d_default_position);
    run_test("camera3d_look_at_ortho",     test_camera3d_look_at_orthonormal);
    run_test("camera3d_perspective_fov",   test_camera3d_perspective_fov);
    run_test("camera3d_vp_product",        test_camera3d_vp_product);
    run_test("normalize_min_max_basic",    test_normalize_min_max_basic);
    run_test("normalize_min_max_constant", test_normalize_min_max_constant);
    run_test("map_range",                  test_map_range);
    run_test("compute_ticks_basic",        test_compute_ticks_basic);
    run_test("compute_ticks_symmetric",    test_compute_ticks_symmetric);
    run_test("compute_ticks_small_range",  test_compute_ticks_small_range);
    run_test("compute_ticks_degenerate",   test_compute_ticks_degenerate);
    run_test("vertex_normals_triangle",    test_vertex_normals_single_triangle);
    run_test("vertex_normals_positions",   test_vertex_normals_preserves_positions);
    run_test("theme_dark",                 test_theme_dark);
    run_test("theme_light",                test_theme_light);

    std::cout << std::string(40, '-') << "\n";
    std::cout << g_passed << " passed, " << g_failed << " failed\n";
    return g_failed > 0 ? 1 : 0;
}
