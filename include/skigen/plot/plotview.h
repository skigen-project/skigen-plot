#pragma once

#include <skigen/plot/core.h>
#include <skigen/plot/export.h>
#include <skigen/plot/theme.h>

#include <Eigen/Core>
#include <QRhiWidget>

#include <memory>
#include <span>

class QRhiResourceUpdateBatch;
class QRhiRenderTarget;

namespace Skigen::Plot {

class SKIGENPLOT_EXPORT PlotView : public QRhiWidget {
    Q_OBJECT

public:
    explicit PlotView(QWidget* parent = nullptr);
    ~PlotView() override;

    // ── 2D line plot ─────────────────────────────────────────────────────

    template <typename DerivedX, typename DerivedY>
    void plot(const Eigen::MatrixBase<DerivedX>& x,
              const Eigen::MatrixBase<DerivedY>& y)
    {
        Eigen::VectorXf xf = x.derived().template cast<float>().eval();
        Eigen::VectorXf yf = y.derived().template cast<float>().eval();
        setLineData({xf.data(), static_cast<std::size_t>(xf.size())},
                    {yf.data(), static_cast<std::size_t>(yf.size())});
    }

    // ── 2D scatter plot ──────────────────────────────────────────────────

    template <typename DerivedX, typename DerivedY>
    void scatter(const Eigen::MatrixBase<DerivedX>& x,
                 const Eigen::MatrixBase<DerivedY>& y)
    {
        Eigen::VectorXf xf = x.derived().template cast<float>().eval();
        Eigen::VectorXf yf = y.derived().template cast<float>().eval();
        setScatterData({xf.data(), static_cast<std::size_t>(xf.size())},
                       {yf.data(), static_cast<std::size_t>(yf.size())});
    }

    // ── 3D point cloud (N×3 matrix) ──────────────────────────────────────

    template <typename Derived>
    void pointCloud(const Eigen::MatrixBase<Derived>& vertices) {
        Eigen::MatrixXf vf = vertices.derived().template cast<float>().eval();
        setPointCloudData({vf.data(), static_cast<std::size_t>(vf.size())},
                          vf.rows());
    }

    // ── 3D mesh (N×3 vertices, M×3 indices) ─────────────────────────────

    template <typename DerivedV, typename DerivedI>
    void mesh(const Eigen::MatrixBase<DerivedV>& vertices,
              const Eigen::MatrixBase<DerivedI>& indices)
    {
        Eigen::MatrixXf vf = vertices.derived().template cast<float>().eval();
        Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic> idx =
            indices.derived().template cast<uint32_t>().eval();
        setMeshData({vf.data(), static_cast<std::size_t>(vf.size())},
                    vf.rows(),
                    {idx.data(), static_cast<std::size_t>(idx.size())},
                    idx.rows());
    }

    // ── Appearance ───────────────────────────────────────────────────────

    void setLineColor(const Eigen::Vector4f& rgba);
    void setBackgroundColor(const Eigen::Vector4f& rgba);
    void setPointSize(float size);

    // ── Theme ────────────────────────────────────────────────────────────

    void setTheme(const Theme& theme);
    auto theme() const -> const Theme&;

    // ── Grid and axes ────────────────────────────────────────────────────

    void setGridVisible(bool visible);
    void setAxesVisible(bool visible);

    // ── Camera (3D modes) ────────────────────────────────────────────────

    void setCamera(const Camera3D& camera);
    auto camera() const -> const Camera3D&;

    // ── Export ───────────────────────────────────────────────────────────

    auto savePng(const QString& path, int width, int height) -> bool;

protected:
    void initialize(QRhiCommandBuffer* cb) override;
    void render(QRhiCommandBuffer* cb) override;

private:
    void setLineData(std::span<const float> x, std::span<const float> y);
    void setScatterData(std::span<const float> x, std::span<const float> y);
    void setPointCloudData(std::span<const float> data, int vertexCount);
    void setMeshData(std::span<const float> verts, int vertexCount,
                     std::span<const uint32_t> idx, int triangleCount);

    void computeGridVertices();
    void renderToTarget(QRhiCommandBuffer* cb,
                        QRhiRenderTarget* rt,
                        QRhiResourceUpdateBatch* u,
                        const QSize& sz);

    struct Impl;
    std::unique_ptr<Impl> d;
};

} // namespace Skigen::Plot
