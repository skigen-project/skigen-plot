// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Skigen Contributors

#include <skigen/plot/core.h>

#include <Eigen/Core>

#include <cmath>
#include <functional>
#include <iostream>
#include <numbers>
#include <sstream>
#include <string>

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
// Tests: BoundingBox math used by plot()/scatter() data paths
// ---------------------------------------------------------------------------

void test_plot_bounds_sine_wave() {
    int n = 1000;
    Eigen::VectorXf x = Eigen::VectorXf::LinSpaced(n, 0.f,
        2.f * std::numbers::pi_v<float>);
    Eigen::VectorXf y = x.array().sin();

    auto bb = Skigen::Plot::BoundingBox2D::fromXY(x, y);
    ASSERT_NEAR(bb.min.x(), 0.f, 1e-4f);
    ASSERT_NEAR(bb.max.x(), 2.f * std::numbers::pi_v<float>, 1e-2f);
    ASSERT_TRUE(bb.min.y() >= -1.f - 1e-4f);
    ASSERT_TRUE(bb.max.y() <=  1.f + 1e-4f);
}

void test_plot_bounds_single_point() {
    Eigen::VectorXf x(1);
    x << 5.f;
    Eigen::VectorXf y(1);
    y << 3.f;

    auto bb = Skigen::Plot::BoundingBox2D::fromXY(x, y);
    ASSERT_NEAR(bb.width(), 0.f, 1e-6f);
    ASSERT_NEAR(bb.height(), 0.f, 1e-6f);
    ASSERT_NEAR(bb.center().x(), 5.f, 1e-6f);
    ASSERT_NEAR(bb.center().y(), 3.f, 1e-6f);
}

void test_plot_bounds_negative_range() {
    Eigen::VectorXf x(3);
    x << -10.f, -5.f, -1.f;
    Eigen::VectorXf y(3);
    y << -20.f, -15.f, -8.f;

    auto bb = Skigen::Plot::BoundingBox2D::fromXY(x, y);
    ASSERT_NEAR(bb.min.x(), -10.f, 1e-6f);
    ASSERT_NEAR(bb.max.x(),  -1.f, 1e-6f);
    ASSERT_NEAR(bb.min.y(), -20.f, 1e-6f);
    ASSERT_NEAR(bb.max.y(),  -8.f, 1e-6f);
}

// ---------------------------------------------------------------------------
// Tests: Eigen expression template compatibility
// ---------------------------------------------------------------------------

void test_bounds_from_expression() {
    Eigen::VectorXf base(5);
    base << 1.f, 2.f, 3.f, 4.f, 5.f;

    auto bb = Skigen::Plot::BoundingBox2D::fromXY(
        base.head(3),
        base.tail(3));

    ASSERT_NEAR(bb.min.x(), 1.f, 1e-6f);
    ASSERT_NEAR(bb.max.x(), 3.f, 1e-6f);
    ASSERT_NEAR(bb.min.y(), 3.f, 1e-6f);
    ASSERT_NEAR(bb.max.y(), 5.f, 1e-6f);
}

void test_normalize_with_expression() {
    Eigen::MatrixXf M(2, 3);
    M << 0.f, 5.f, 10.f,
         2.f, 4.f, 6.f;

    auto n = Skigen::Plot::normalizeMinMax(M.row(1));
    ASSERT_NEAR(n(0), 0.f, 1e-6f);
    ASSERT_NEAR(n(1), 0.5f, 1e-6f);
    ASSERT_NEAR(n(2), 1.f, 1e-6f);
}

// ---------------------------------------------------------------------------
// Tests: Camera round-trips
// ---------------------------------------------------------------------------

void test_camera_view_matrix_invertible() {
    Skigen::Plot::Camera3D cam;
    cam.lookAt({5.f, 5.f, 5.f}, {0.f, 0.f, 0.f});
    cam.setPerspective(60.f, 16.f / 9.f, 0.1f, 100.f);

    auto VP = cam.viewProjectionMatrix();
    float det = VP.determinant();
    ASSERT_TRUE(std::abs(det) > 1e-6f);
}

void test_camera_origin_maps_to_center() {
    Skigen::Plot::Camera3D cam;
    cam.lookAt({0.f, 0.f, 5.f}, {0.f, 0.f, 0.f});
    cam.setPerspective(60.f, 1.f, 0.1f, 100.f);

    Eigen::Vector4f origin(0.f, 0.f, 0.f, 1.f);
    Eigen::Vector4f clip = cam.viewProjectionMatrix() * origin;
    float ndcX = clip.x() / clip.w();
    float ndcY = clip.y() / clip.w();
    ASSERT_NEAR(ndcX, 0.f, 1e-4f);
    ASSERT_NEAR(ndcY, 0.f, 1e-4f);
}

// ---------------------------------------------------------------------------
// Tests: Double → Float conversion via template API
// ---------------------------------------------------------------------------

void test_bounds_from_double_data() {
    Eigen::VectorXd xd(3);
    xd << 1.0, 2.0, 3.0;
    Eigen::VectorXd yd(3);
    yd << 4.0, 5.0, 6.0;

    auto bb = Skigen::Plot::BoundingBox2D::fromXY(xd, yd);
    ASSERT_NEAR(bb.min.x(), 1.f, 1e-5f);
    ASSERT_NEAR(bb.max.y(), 6.f, 1e-5f);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    std::cout << "SkigenPlot — plotview data-layer tests\n";
    std::cout << std::string(40, '-') << "\n";

    run_test("plot_bounds_sine_wave",       test_plot_bounds_sine_wave);
    run_test("plot_bounds_single_point",    test_plot_bounds_single_point);
    run_test("plot_bounds_negative_range",  test_plot_bounds_negative_range);
    run_test("bounds_from_expression",      test_bounds_from_expression);
    run_test("normalize_with_expression",   test_normalize_with_expression);
    run_test("camera_view_invertible",      test_camera_view_matrix_invertible);
    run_test("camera_origin_maps_center",   test_camera_origin_maps_to_center);
    run_test("bounds_from_double_data",     test_bounds_from_double_data);

    std::cout << std::string(40, '-') << "\n";
    std::cout << g_passed << " passed, " << g_failed << " failed\n";
    return g_failed > 0 ? 1 : 0;
}
