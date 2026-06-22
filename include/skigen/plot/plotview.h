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

    // ── 2D histogram (adds a filled series) ─────────────────────────

    /// @brief Bin @p values into @p bins uniform bins and draw the counts as
    ///   filled bars. When @p density is true, bars are normalised so their
    ///   total area is 1. Use @p bins <= 0 for a Sturges-rule default.
    template <typename Derived>
    void hist(const Eigen::MatrixBase<Derived>& values,
              int bins = 0,
              bool density = false,
              const PlotStyle& style = {})
    {
        Eigen::VectorXf v = values.derived().template cast<float>().eval();
        histImpl({v.data(), static_cast<std::size_t>(v.size())}, bins, density, style);
    }

    // ── 2D bar / horizontal bar (adds a filled series) ──────────────

    /// @brief Vertical bars of height @p heights at positions @p x.
    template <typename DerivedX, typename DerivedH>
    void bar(const Eigen::MatrixBase<DerivedX>& x,
             const Eigen::MatrixBase<DerivedH>& heights,
             float width = 0.8f,
             const PlotStyle& style = {})
    {
        Eigen::VectorXf xf = x.derived().template cast<float>().eval();
        Eigen::VectorXf hf = heights.derived().template cast<float>().eval();
        barImpl({xf.data(), static_cast<std::size_t>(xf.size())},
                {hf.data(), static_cast<std::size_t>(hf.size())},
                width, /*horizontal=*/false, style);
    }

    /// @brief Horizontal bars of length @p widths at positions @p y.
    template <typename DerivedY, typename DerivedW>
    void barh(const Eigen::MatrixBase<DerivedY>& y,
              const Eigen::MatrixBase<DerivedW>& widths,
              float thickness = 0.8f,
              const PlotStyle& style = {})
    {
        Eigen::VectorXf yf = y.derived().template cast<float>().eval();
        Eigen::VectorXf wf = widths.derived().template cast<float>().eval();
        barImpl({yf.data(), static_cast<std::size_t>(yf.size())},
                {wf.data(), static_cast<std::size_t>(wf.size())},
                thickness, /*horizontal=*/true, style);
    }

    // ── 2D filled area between two curves ───────────────────────────

    /// @brief Fill the region between @p y0 and @p y1 over @p x.
    template <typename DerivedX, typename DerivedY0, typename DerivedY1>
    void fillBetween(const Eigen::MatrixBase<DerivedX>& x,
                     const Eigen::MatrixBase<DerivedY0>& y0,
                     const Eigen::MatrixBase<DerivedY1>& y1,
                     const PlotStyle& style = {})
    {
        Eigen::VectorXf xf = x.derived().template cast<float>().eval();
        Eigen::VectorXf y0f = y0.derived().template cast<float>().eval();
        Eigen::VectorXf y1f = y1.derived().template cast<float>().eval();
        fillBetweenImpl({xf.data(), static_cast<std::size_t>(xf.size())},
                        {y0f.data(), static_cast<std::size_t>(y0f.size())},
                        {y1f.data(), static_cast<std::size_t>(y1f.size())},
                        style);
    }

    // ── 2D step plot (piecewise-constant line) ──────────────────────

    /// @brief Piecewise-constant line through (@p x, @p y).
    template <typename DerivedX, typename DerivedY>
    void step(const Eigen::MatrixBase<DerivedX>& x,
              const Eigen::MatrixBase<DerivedY>& y,
              const PlotStyle& style = {})
    {
        Eigen::VectorXf xf = x.derived().template cast<float>().eval();
        Eigen::VectorXf yf = y.derived().template cast<float>().eval();
        stepImpl({xf.data(), static_cast<std::size_t>(xf.size())},
                 {yf.data(), static_cast<std::size_t>(yf.size())}, style);
    }

    // ── 2D error bars ───────────────────────────────────────────────

    /// @brief Symmetric vertical error bars of half-height @p yerr at (x, y).
    template <typename DerivedX, typename DerivedY, typename DerivedE>
    void errorbar(const Eigen::MatrixBase<DerivedX>& x,
                  const Eigen::MatrixBase<DerivedY>& y,
                  const Eigen::MatrixBase<DerivedE>& yerr,
                  const PlotStyle& style = {})
    {
        Eigen::VectorXf xf = x.derived().template cast<float>().eval();
        Eigen::VectorXf yf = y.derived().template cast<float>().eval();
        Eigen::VectorXf ef = yerr.derived().template cast<float>().eval();
        errorbarImpl({xf.data(), static_cast<std::size_t>(xf.size())},
                     {yf.data(), static_cast<std::size_t>(yf.size())},
                     {ef.data(), static_cast<std::size_t>(ef.size())}, style);
    }

    // ── 2D heatmap / image (colormapped R×C matrix) ─────────────────

    /// @brief Draw matrix @p m as a colormapped grid of cells (row 0 at the
    ///   top, matching matplotlib's `imshow`). The data range is auto-scaled
    ///   to [min, max] across @p m unless @p vmin < @p vmax is given.
    template <typename Derived>
    void imshow(const Eigen::MatrixBase<Derived>& m,
                Colormap cmap = Colormap::Viridis,
                float vmin = 0.0f, float vmax = 0.0f)
    {
        Eigen::MatrixXf mf = m.derived().template cast<float>().eval();
        imshowImpl({mf.data(), static_cast<std::size_t>(mf.size())},
                   static_cast<int>(mf.rows()), static_cast<int>(mf.cols()),
                   cmap, vmin, vmax);
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

    /// @brief Replace only the series colours of the current theme with the
    ///   given palette, keeping the existing layout (background, grid, axes).
    void setPalette(Palette palette);

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
    void addFillSeries(std::span<const float> triangleVertices,
                       const PlotStyle& style);

    void histImpl(std::span<const float> values, int bins, bool density,
                  const PlotStyle& style);
    void barImpl(std::span<const float> positions, std::span<const float> sizes,
                 float width, bool horizontal, const PlotStyle& style);
    void fillBetweenImpl(std::span<const float> x, std::span<const float> y0,
                         std::span<const float> y1, const PlotStyle& style);
    void stepImpl(std::span<const float> x, std::span<const float> y,
                  const PlotStyle& style);
    void errorbarImpl(std::span<const float> x, std::span<const float> y,
                      std::span<const float> yerr, const PlotStyle& style);
    void imshowImpl(std::span<const float> data, int rows, int cols,
                    Colormap cmap, float vmin, float vmax);

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
