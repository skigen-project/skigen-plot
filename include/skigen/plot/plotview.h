#pragma once

#include <skigen/plot/core.h>
#include <skigen/plot/export.h>
#include <skigen/plot/series.h>
#include <skigen/plot/theme.h>

#include <Eigen/Core>
#include <QRhiWidget>

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

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
    auto plot(const Eigen::MatrixBase<DerivedX>& x,
              const Eigen::MatrixBase<DerivedY>& y,
              const PlotStyle& style = {}) -> SeriesHandle
    {
        Eigen::VectorXf xf = x.derived().template cast<float>().eval();
        Eigen::VectorXf yf = y.derived().template cast<float>().eval();
        return addLineSeries({xf.data(), static_cast<std::size_t>(xf.size())},
                             {yf.data(), static_cast<std::size_t>(yf.size())},
                             style);
    }

    // ── 2D scatter plot (adds a series) ─────────────────────────────

    template <typename DerivedX, typename DerivedY>
    auto scatter(const Eigen::MatrixBase<DerivedX>& x,
                 const Eigen::MatrixBase<DerivedY>& y,
                 const PlotStyle& style = {}) -> SeriesHandle
    {
        Eigen::VectorXf xf = x.derived().template cast<float>().eval();
        Eigen::VectorXf yf = y.derived().template cast<float>().eval();
        return addScatterSeries({xf.data(), static_cast<std::size_t>(xf.size())},
                                {yf.data(), static_cast<std::size_t>(yf.size())},
                                style);
    }

    template <typename DerivedX, typename DerivedY>
    auto updateSeriesData(SeriesHandle handle,
                          const Eigen::MatrixBase<DerivedX>& x,
                          const Eigen::MatrixBase<DerivedY>& y) -> bool
    {
        Eigen::VectorXf xf = x.derived().template cast<float>().eval();
        Eigen::VectorXf yf = y.derived().template cast<float>().eval();
        return updateSeriesDataImpl(
            handle,
            {xf.data(), static_cast<std::size_t>(xf.size())},
            {yf.data(), static_cast<std::size_t>(yf.size())});
    }

    auto setSeriesStyle(SeriesHandle handle, const PlotStyle& style) -> bool;
    auto setSeriesVisible(SeriesHandle handle, bool visible) -> bool;
    auto containsSeries(SeriesHandle handle) const -> bool;
    auto removeSeries(SeriesHandle handle) -> bool;

    // -- Scrolling telemetry -----------------------------------------

    /// @brief Start a line series that retains the latest @p windowSize
    ///   samples. Starting a new telemetry stream replaces the current one.
    void startTelemetry(std::size_t windowSize,
                        const PlotStyle& style = {});

    /// @brief Append a sample using a monotonically increasing x coordinate.
    void appendTelemetry(float value);

    /// @brief Append a sample with an explicit x coordinate.
    void appendTelemetry(float x, float y);

    /// @brief Number of samples currently retained by the telemetry stream.
    auto telemetryPointCount() const -> std::size_t;

    /// @brief Remove the active telemetry stream and its line series.
    void clearTelemetry();

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

    // ── 2D stem plot (baseline-anchored impulses) ──────────────────

    /// @brief Draw vertical stems from y=0 to each (@p x, @p y) with a marker
    ///   at the top of each stem.
    template <typename DerivedX, typename DerivedY>
    void stem(const Eigen::MatrixBase<DerivedX>& x,
              const Eigen::MatrixBase<DerivedY>& y,
              const PlotStyle& style = {})
    {
        Eigen::VectorXf xf = x.derived().template cast<float>().eval();
        Eigen::VectorXf yf = y.derived().template cast<float>().eval();
        stemImpl({xf.data(), static_cast<std::size_t>(xf.size())},
                 {yf.data(), static_cast<std::size_t>(yf.size())}, style);
    }

    // ── 2D box plot (one box per group) ─────────────────────────────

    /// @brief Draw a box-and-whisker for each group in @p groups, placed at
    ///   integer positions 0, 1, 2, … Box spans Q1–Q3, with the median line,
    ///   1.5·IQR whiskers, and outliers as points.
    void boxplot(const std::vector<Eigen::VectorXf>& groups,
                 const PlotStyle& style = {});

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

    // ── 2D contour lines / filled contour ───────────────────────────

    /// @brief Draw iso-lines of scalar field @p z at @p levels evenly-spaced
    ///   levels between the data min and max (marching squares). Cell (r, c)
    ///   maps to grid coordinate (c, rows-1-r), matching imshow().
    template <typename Derived>
    void contour(const Eigen::MatrixBase<Derived>& z, int levels = 8,
                 const PlotStyle& style = {})
    {
        Eigen::MatrixXf zf = z.derived().template cast<float>().eval();
        contourImpl({zf.data(), static_cast<std::size_t>(zf.size())},
                    static_cast<int>(zf.rows()), static_cast<int>(zf.cols()),
                    levels, style);
    }

    /// @brief Filled contour: colour each cell by its value band using @p cmap
    ///   (cell-level quantisation). Companion to contour().
    template <typename Derived>
    void contourf(const Eigen::MatrixBase<Derived>& z, int levels = 10,
                  Colormap cmap = Colormap::Viridis)
    {
        Eigen::MatrixXf zf = z.derived().template cast<float>().eval();
        contourfImpl({zf.data(), static_cast<std::size_t>(zf.size())},
                     static_cast<int>(zf.rows()), static_cast<int>(zf.cols()),
                     levels, cmap);
    }

    // ── 2D violin plot (KDE density per group) ──────────────────────

    /// @brief Draw a violin (mirrored Gaussian-KDE density) for each group in
    ///   @p groups, placed at integer positions 0, 1, 2, …
    void violinplot(const std::vector<Eigen::VectorXf>& groups,
                    const PlotStyle& style = {});

    // ── 2D quiver (vector field) ────────────────────────────────────

    /// @brief Draw arrows at (@p x, @p y) with components (@p u, @p v).
    template <typename DX, typename DY, typename DU, typename DV>
    void quiver(const Eigen::MatrixBase<DX>& x, const Eigen::MatrixBase<DY>& y,
                const Eigen::MatrixBase<DU>& u, const Eigen::MatrixBase<DV>& v,
                const PlotStyle& style = {})
    {
        Eigen::VectorXf xf = x.derived().template cast<float>().eval();
        Eigen::VectorXf yf = y.derived().template cast<float>().eval();
        Eigen::VectorXf uf = u.derived().template cast<float>().eval();
        Eigen::VectorXf vf = v.derived().template cast<float>().eval();
        quiverImpl({xf.data(), static_cast<std::size_t>(xf.size())},
                   {yf.data(), static_cast<std::size_t>(yf.size())},
                   {uf.data(), static_cast<std::size_t>(uf.size())},
                   {vf.data(), static_cast<std::size_t>(vf.size())}, style);
    }

    // ── 2D hexbin (hexagonally-binned density) ──────────────────────

    /// @brief Hexagonally bin points (@p x, @p y) into a @p gridsize-wide grid
    ///   and colour each hexagon by its count using @p cmap.
    template <typename DX, typename DY>
    void hexbin(const Eigen::MatrixBase<DX>& x, const Eigen::MatrixBase<DY>& y,
                int gridsize = 20, Colormap cmap = Colormap::Viridis)
    {
        Eigen::VectorXf xf = x.derived().template cast<float>().eval();
        Eigen::VectorXf yf = y.derived().template cast<float>().eval();
        hexbinImpl({xf.data(), static_cast<std::size_t>(xf.size())},
                   {yf.data(), static_cast<std::size_t>(yf.size())},
                   gridsize, cmap);
    }

    // ── 2D pie chart ────────────────────────────────────────────────

    /// @brief Draw proportional wedges for @p values (normalised to their sum),
    ///   starting at the top and proceeding clockwise.
    template <typename Derived>
    void pie(const Eigen::MatrixBase<Derived>& values)
    {
        Eigen::VectorXf vf = values.derived().template cast<float>().eval();
        pieImpl({vf.data(), static_cast<std::size_t>(vf.size())});
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

    /// @brief Show a vertical colorbar legend for colormapped data (imshow /
    ///   contourf): a colormapped strip with min/mid/max tick labels.
    void setColorbarVisible(bool visible);

    /// @brief When true, 1 data-unit maps to the same pixel length on both
    ///   axes (keeps circles round — e.g. for pie charts). Default: false.
    void setAspectEqual(bool equal);
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
    auto addLineSeries(std::span<const float> x, std::span<const float> y,
                       const PlotStyle& style) -> SeriesHandle;
    auto addScatterSeries(std::span<const float> x, std::span<const float> y,
                          const PlotStyle& style) -> SeriesHandle;
    auto addSeriesImpl(int kind, std::span<const float> x,
                       std::span<const float> y, const PlotStyle& style)
        -> SeriesHandle;
    auto updateSeriesDataImpl(SeriesHandle handle, std::span<const float> x,
                              std::span<const float> y) -> bool;
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
    void stemImpl(std::span<const float> x, std::span<const float> y,
                  const PlotStyle& style);
    void errorbarImpl(std::span<const float> x, std::span<const float> y,
                      std::span<const float> yerr, const PlotStyle& style);
    void imshowImpl(std::span<const float> data, int rows, int cols,
                    Colormap cmap, float vmin, float vmax);
    void contourImpl(std::span<const float> data, int rows, int cols,
                     int levels, const PlotStyle& style);
    void contourfImpl(std::span<const float> data, int rows, int cols,
                      int levels, Colormap cmap);
    void quiverImpl(std::span<const float> x, std::span<const float> y,
                    std::span<const float> u, std::span<const float> v,
                    const PlotStyle& style);
    void hexbinImpl(std::span<const float> x, std::span<const float> y,
                    int gridsize, Colormap cmap);
    void pieImpl(std::span<const float> values);

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
