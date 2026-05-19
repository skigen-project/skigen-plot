#pragma once

#include <skigen/plot/core.h>
#include <skigen/plot/export.h>
#include <skigen/plot/series.h>
#include <skigen/plot/theme.h>

#include <Eigen/Core>
#include <QRhiWidget>

#include <memory>
#include <span>

class QRhiResourceUpdateBatch;
class QRhiRenderTarget;
class QPainter;

namespace Skigen::Plot {

enum class InteractionTool {
    Select,
    Pan,
    Rotate,
    Zoom
};

class SKIGENPLOT_EXPORT PlotView : public QRhiWidget {
    Q_OBJECT

public:
    explicit PlotView(QWidget* parent = nullptr);
    ~PlotView() override;

    // ── 2D line plot (adds a series) ────────────────────────────────

    template <typename DerivedX, typename DerivedY>
    void plot(const Eigen::MatrixBase<DerivedX>& x,
              const Eigen::MatrixBase<DerivedY>& y,
              const PlotStyle& style = {})
    {
        Eigen::VectorXf xf = x.derived().template cast<float>().eval();
        Eigen::VectorXf yf = y.derived().template cast<float>().eval();
        addLineSeries({xf.data(), static_cast<std::size_t>(xf.size())},
                      {yf.data(), static_cast<std::size_t>(yf.size())},
                      style);
    }

    // ── 2D scatter plot (adds a series) ─────────────────────────────

    template <typename DerivedX, typename DerivedY>
    void scatter(const Eigen::MatrixBase<DerivedX>& x,
                 const Eigen::MatrixBase<DerivedY>& y,
                 const PlotStyle& style = {})
    {
        Eigen::VectorXf xf = x.derived().template cast<float>().eval();
        Eigen::VectorXf yf = y.derived().template cast<float>().eval();
        addScatterSeries({xf.data(), static_cast<std::size_t>(xf.size())},
                         {yf.data(), static_cast<std::size_t>(yf.size())},
                         style);
    }

    // ── 3D point cloud (N×3 matrix) ─────────────────────────────────

    template <typename Derived>
    void pointCloud(const Eigen::MatrixBase<Derived>& vertices,
                    const PlotStyle& style = {}) {
        Eigen::MatrixXf vf = vertices.derived().template cast<float>().eval();
        setPointCloudData({vf.data(), static_cast<std::size_t>(vf.size())},
                          vf.rows(), style);
    }

    // ── 3D mesh (N×3 vertices, M×3 indices) ─────────────────────────

    template <typename DerivedV, typename DerivedI>
    void mesh(const Eigen::MatrixBase<DerivedV>& vertices,
              const Eigen::MatrixBase<DerivedI>& indices,
              const PlotStyle& style = {})
    {
        Eigen::MatrixXf vf = vertices.derived().template cast<float>().eval();
        Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic> idx =
            indices.derived().template cast<uint32_t>().eval();
        setMeshData({vf.data(), static_cast<std::size_t>(vf.size())},
                    vf.rows(),
                    {idx.data(), static_cast<std::size_t>(idx.size())},
                    idx.rows(), style);
    }

    // ── Scene management ────────────────────────────────────────────

    void clear();
    void setTitle(const QString& title);
    void setCaption(const QString& caption);

    // ── Appearance ──────────────────────────────────────────────────

    void setBackgroundColor(const Eigen::Vector4f& rgba);
    void setPointSize(float size);

    // ── Theme ───────────────────────────────────────────────────────

    void setTheme(const Theme& theme);
    auto theme() const -> const Theme&;

    // ── Grid and axes ───────────────────────────────────────────────

    void setGridVisible(bool visible);
    void setAxesVisible(bool visible);
    void setAxisArrowsVisible(bool visible);
    void setAxisLabels(const QString& xLabel, const QString& yLabel);
    void setAxisLabels(const QString& xLabel, const QString& yLabel, const QString& zLabel);
    void setXAxisLabel(const QString& label);
    void setYAxisLabel(const QString& label);
    void setZAxisLabel(const QString& label);

    // ── Camera (3D modes) ───────────────────────────────────────────

    void setCamera(const Camera3D& camera);
    auto camera() const -> const Camera3D&;

    // ── Overlay ─────────────────────────────────────────────────────

    void setOverlayVisible(bool visible);
    void setInteractionTool(InteractionTool tool);
    auto interactionTool() const -> InteractionTool;
    void resetCameraView();
    void zoomCamera(float factor);
    auto is3DView() const -> bool;
    auto is2DView() const -> bool;

    // ── Export ───────────────────────────────────────────────────────

    auto savePng(const QString& path, int width, int height) -> bool;

protected:
    void initialize(QRhiCommandBuffer* cb) override;
    void render(QRhiCommandBuffer* cb) override;
    void resizeEvent(QResizeEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;

private:
    void addLineSeries(std::span<const float> x, std::span<const float> y,
                       const PlotStyle& style);
    void addScatterSeries(std::span<const float> x, std::span<const float> y,
                          const PlotStyle& style);
    void addSeriesImpl(int kind, std::span<const float> x,
                       std::span<const float> y, const PlotStyle& style);

    void setPointCloudData(std::span<const float> data, int vertexCount,
                           const PlotStyle& style);
    void setMeshData(std::span<const float> verts, int vertexCount,
                     std::span<const uint32_t> idx, int triangleCount,
                     const PlotStyle& style);

    void computeGridVertices();
    void recomputeBounds();
    void layoutChildren();
    void paintTextOverlay(QPainter& painter, const QSize& size) const;

    void renderToTarget(QRhiCommandBuffer* cb,
                        QRhiRenderTarget* rt,
                        QRhiResourceUpdateBatch* u,
                        const QSize& sz);

    friend class PlotTextOverlay;

    struct Impl;
    std::unique_ptr<Impl> d;
};

} // namespace Skigen::Plot
