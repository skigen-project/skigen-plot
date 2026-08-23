#include <skigen/plot/plotview.h>

#include "overlay.h"

#include <rhi/qrhi.h>

#include <QFile>
#include <QFontMetrics>
#include <QImage>
#include <QLabel>
#include <QMouseEvent>
#include <QPainter>
#include <QResizeEvent>
#include <QWheelEvent>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <expected>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

using namespace Qt::StringLiterals;

namespace Skigen::Plot {

// ── Render mode (for 3D, which stays single-dataset) ───────────────

enum class RenderMode3D { None, PointCloud, Mesh };
enum class SeriesKind { Line, Scatter, Fill };

class PlotTextOverlay : public QWidget {
public:
    explicit PlotTextOverlay(PlotView* parent)
        : QWidget(parent)
        , m_plotView(parent)
    {
        setAttribute(Qt::WA_TransparentForMouseEvents);
        setAttribute(Qt::WA_TranslucentBackground);
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing);
        painter.setRenderHint(QPainter::TextAntialiasing);
        m_plotView->paintTextOverlay(painter, size());
    }

private:
    PlotView* m_plotView;
};

// ── Shader loading ─────────────────────────────────────────────────

static auto loadShader(const QString& path)
    -> std::expected<QShader, QString>
{
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly))
        return std::unexpected(u"Failed to open shader: "_s + path);
    auto shader = QShader::fromSerialized(f.readAll());
    if (!shader.isValid())
        return std::unexpected(u"Invalid shader: "_s + path);
    return shader;
}

// ── Per-series GPU state ───────────────────────────────────────────

struct Series2D {
    std::uint64_t id = 0;
    SeriesKind kind;
    std::vector<float> vertices;
    int vertexCount = 0;
    Eigen::Vector4f color;
    float pointSize = 5.0f;
    bool hollow = false;
    bool visible = true;
    QString label;
    bool dirty = true;

    QRhiBuffer* vb = nullptr;
    QRhiBuffer* ub = nullptr;
    QRhiShaderResourceBindings* srb = nullptr;
    int vbCapacity = 0;
};

// ── PIMPL ──────────────────────────────────────────────────────────

struct PlotView::Impl {
    // 2D series
    std::vector<Series2D> series2d;
    int nextColorIndex = 0;
    std::optional<SeriesHandle> telemetrySeries;
    std::vector<float> telemetryRing;
    std::size_t telemetryHead = 0;
    std::size_t telemetrySize = 0;
    float telemetryNextX = 0.0f;

    // Shared pipelines (all Line2D series share linePipeline, etc.)
    std::unique_ptr<QRhiGraphicsPipeline> linePipeline;
    std::unique_ptr<QRhiGraphicsPipeline> pointPipeline;
    std::unique_ptr<QRhiGraphicsPipeline> fillPipeline;
    std::unique_ptr<QRhiGraphicsPipeline> heatmapPipeline;
    std::unique_ptr<QRhiGraphicsPipeline> gridPipeline;

    // Heatmap (imshow) — single per-view dataset of per-vertex coloured
    // triangles: interleaved (vec2 position, vec4 colour), stride 24.
    std::unique_ptr<QRhiBuffer> heatmapBuffer;
    std::unique_ptr<QRhiBuffer> heatmapUniformBuffer;
    std::unique_ptr<QRhiShaderResourceBindings> heatmapSrb;
    std::vector<float> heatmapVertices;
    int heatmapVertexCount = 0;
    int heatmapCapacity = 0;
    bool hasHeatmap = false;
    bool heatmapDirty = false;

    // Colorbar legend state (set by imshow / contourf).
    bool showColorbar = false;
    bool hasColormappedData = false;
    Colormap colorbarMap = Colormap::Viridis;
    float colorbarVmin = 0.0f;
    float colorbarVmax = 1.0f;

    // Contour lines — vec2 line segments drawn with the grid pipeline,
    // uniform colour via contourUniformBuffer.
    std::unique_ptr<QRhiBuffer> contourBuffer;
    std::unique_ptr<QRhiBuffer> contourUniformBuffer;
    std::unique_ptr<QRhiShaderResourceBindings> contourSrb;
    std::vector<float> contourVertices;
    Eigen::Vector4f contourColor{0.1f, 0.1f, 0.1f, 0.9f};
    int contourVertexCount = 0;
    int contourCapacity = 0;
    bool hasContour = false;
    bool contourDirty = false;

    // 3D pipelines
    std::unique_ptr<QRhiGraphicsPipeline> point3dPipeline;
    std::unique_ptr<QRhiGraphicsPipeline> meshPipeline;
    std::unique_ptr<QRhiGraphicsPipeline> meshEdgePipeline;
    std::unique_ptr<QRhiGraphicsPipeline> guide3dPipeline;

    // 3D data (single dataset)
    RenderMode3D mode3d = RenderMode3D::None;
    std::unique_ptr<QRhiBuffer> vertex3dBuffer;
    std::vector<float> vertices3d;
    int vertex3dCount = 0;
    int vertex3dCapacity = 0;
    bool data3dDirty = false;
    Eigen::Vector4f data3dColor{0.024f, 0.714f, 0.831f, 1.0f};
    float data3dPointSize = 5.0f;

    // Mesh index data
    std::unique_ptr<QRhiBuffer> indexBuffer;
    std::unique_ptr<QRhiBuffer> meshEdgeBuffer;
    std::unique_ptr<QRhiBuffer> guide3dBuffer;
    std::vector<uint32_t> indices;
    std::vector<float> meshEdgeVertices;
    std::vector<float> guide3dVertices;
    int indexCount = 0;
    int indexCapacity = 0;
    int meshEdgeVertexCount = 0;
    int meshEdgeCapacity = 0;
    int guide3dVertexCount = 0;
    int guide3dCapacity = 0;
    bool indexDirty = false;
    bool meshEdgeDirty = false;
    bool guide3dDirty = false;

    // 3D uniform buffers
    std::unique_ptr<QRhiBuffer> point3dUniformBuffer;
    std::unique_ptr<QRhiBuffer> meshUniformBuffer;
    std::unique_ptr<QRhiBuffer> meshEdgeUniformBuffer;
    std::unique_ptr<QRhiBuffer> guide3dUniformBuffer;
    std::unique_ptr<QRhiShaderResourceBindings> point3dSrb;
    std::unique_ptr<QRhiShaderResourceBindings> meshSrb;
    std::unique_ptr<QRhiShaderResourceBindings> meshEdgeSrb;
    std::unique_ptr<QRhiShaderResourceBindings> guide3dSrb;

    // Grid vertex data
    std::unique_ptr<QRhiBuffer> gridVertexBuffer;
    std::vector<float> gridVertices;
    std::vector<float> axisVertices;
    std::vector<float> xTickValues;
    std::vector<float> yTickValues;
    std::vector<float> xTickValues3d;
    std::vector<float> yTickValues3d;
    std::vector<float> zTickValues3d;
    int gridVertexCount = 0;
    int axisVertexCount = 0;
    int gridVertexCapacity = 0;
    bool gridDirty = false;

    // Last render-target size in pixels — used to size 2D axis arrowheads
    // consistently in screen space across both axes.
    float viewportW = 1.f;
    float viewportH = 1.f;

    // Grid uniform buffers
    std::unique_ptr<QRhiBuffer> gridUniformBuffer;
    std::unique_ptr<QRhiBuffer> axisUniformBuffer;
    std::unique_ptr<QRhiShaderResourceBindings> gridSrb;
    std::unique_ptr<QRhiShaderResourceBindings> axisSrb;

    bool pipelineReady = false;

    // Spatial state
    BoundingBox2D bounds2d;
    BoundingBox2D viewBounds;
    bool userViewBounds2d = false;
    BoundingBox3D bounds3d;
    Camera3D camera;
    Camera3D homeCamera;

    // Appearance
    Theme theme = Theme::dark();
    std::optional<Eigen::Vector4f> userBgColor;
    float defaultPointSize = 5.0f;
    bool showGrid = true;
    bool showAxes = true;
    bool showAxisArrows = true;
    bool showLegend = false;
    LegendPosition legendPosition = LegendPosition::Auto;
    bool aspectEqual = false;
    QString title;
    QString caption;
    QString xAxisLabel;
    QString yAxisLabel;
    QString zAxisLabel;

    // UI
    QLabel* titleLabel = nullptr;
    PlotTextOverlay* textOverlay = nullptr;
    PlotOverlay* overlay = nullptr;
    bool overlayEnabled = true;
    InteractionTool interactionTool = InteractionTool::Rotate;
    bool dragging = false;
    QPoint lastMousePos;

    bool has2D() const {
        return std::ranges::any_of(series2d, [](const Series2D& series) {
            return series.visible && series.vertexCount > 0;
        }) || hasHeatmap || hasContour;
    }
    bool has3D() const { return mode3d != RenderMode3D::None; }
    bool hasData() const { return has2D() || has3D(); }
};

// ── Helpers ────────────────────────────────────────────────────────

static auto makeSrb(QRhi* r, QRhiBuffer* ub)
    -> QRhiShaderResourceBindings*
{
    auto* srb = r->newShaderResourceBindings();
    srb->setBindings({
        QRhiShaderResourceBinding::uniformBuffer(
            0,
            QRhiShaderResourceBinding::VertexStage
                | QRhiShaderResourceBinding::FragmentStage,
            ub)
    });
    srb->create();
    return srb;
}

static auto nextSeriesId() -> std::uint64_t {
    static std::atomic<std::uint64_t> nextId{1};
    return nextId.fetch_add(1, std::memory_order_relaxed);
}

static auto makeUniqueSrb(QRhi* r, QRhiBuffer* ub)
    -> std::unique_ptr<QRhiShaderResourceBindings>
{
    return std::unique_ptr<QRhiShaderResourceBindings>(makeSrb(r, ub));
}

static auto makeUB(QRhi* r, quint32 size) -> std::unique_ptr<QRhiBuffer> {
    auto buf = std::unique_ptr<QRhiBuffer>(
        r->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer, size));
    buf->create();
    return buf;
}

static auto makeDynBuf(QRhi* r, QRhiBuffer::UsageFlag usage, quint32 size)
    -> std::unique_ptr<QRhiBuffer>
{
    auto buf = std::unique_ptr<QRhiBuffer>(
        r->newBuffer(QRhiBuffer::Dynamic, usage, size));
    buf->create();
    return buf;
}

static auto makeRawDynBuf(QRhi* r, QRhiBuffer::UsageFlag usage, quint32 size)
    -> QRhiBuffer*
{
    auto* buf = r->newBuffer(QRhiBuffer::Dynamic, usage, size);
    buf->create();
    return buf;
}

static auto makeRawUB(QRhi* r, quint32 size) -> QRhiBuffer* {
    auto* buf = r->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer, size);
    buf->create();
    return buf;
}

static auto alphaBlend() -> QRhiGraphicsPipeline::TargetBlend {
    QRhiGraphicsPipeline::TargetBlend blend;
    blend.enable = true;
    blend.srcColor = QRhiGraphicsPipeline::SrcAlpha;
    blend.dstColor = QRhiGraphicsPipeline::OneMinusSrcAlpha;
    blend.srcAlpha = QRhiGraphicsPipeline::One;
    blend.dstAlpha = QRhiGraphicsPipeline::OneMinusSrcAlpha;
    return blend;
}

struct EdgeInfo {
    Eigen::Vector3f firstNormal{0.f, 0.f, 0.f};
    bool hasSecond = false;
    bool include = false;
};

static auto edgeKey(uint32_t a, uint32_t b) -> uint64_t {
    auto lo = static_cast<uint64_t>(std::min(a, b));
    auto hi = static_cast<uint64_t>(std::max(a, b));
    return (lo << 32u) | hi;
}

static void appendEdge(std::vector<float>& out,
                       std::span<const float> vertices,
                       uint32_t a,
                       uint32_t b) {
    for (uint32_t idx : {a, b}) {
        auto offset = static_cast<std::size_t>(idx) * 3;
        out.push_back(vertices[offset]);
        out.push_back(vertices[offset + 1]);
        out.push_back(vertices[offset + 2]);
    }
}

static auto computeSharpEdgeVertices(std::span<const float> vertices,
                                     std::span<const uint32_t> indices,
                                     int triangleCount) -> std::vector<float>
{
    static constexpr float kCoplanarCos = 0.995f;

    std::unordered_map<uint64_t, EdgeInfo> edges;
    edges.reserve(static_cast<std::size_t>(triangleCount) * 3);

    for (int t = 0; t < triangleCount; ++t) {
        auto ti = static_cast<std::size_t>(t) * 3;
        std::array<uint32_t, 3> tri = {indices[ti], indices[ti + 1], indices[ti + 2]};

        auto vertex = [&](uint32_t idx) {
            auto offset = static_cast<std::size_t>(idx) * 3;
            return Eigen::Vector3f(vertices[offset], vertices[offset + 1], vertices[offset + 2]);
        };

        Eigen::Vector3f normal = (vertex(tri[1]) - vertex(tri[0]))
            .cross(vertex(tri[2]) - vertex(tri[0]));
        if (normal.squaredNorm() > 1e-12f)
            normal.normalize();

        for (int e = 0; e < 3; ++e) {
            uint32_t a = tri[static_cast<std::size_t>(e)];
            uint32_t b = tri[static_cast<std::size_t>((e + 1) % 3)];
            auto& info = edges[edgeKey(a, b)];
            if (!info.hasSecond && info.firstNormal.squaredNorm() == 0.f) {
                info.firstNormal = normal;
            } else if (!info.hasSecond) {
                info.hasSecond = true;
                info.include = info.firstNormal.dot(normal) < kCoplanarCos;
            } else {
                info.include = true;
            }
        }
    }

    std::vector<float> result;
    result.reserve(edges.size() * 6);
    for (const auto& [key, info] : edges) {
        if (!info.hasSecond || info.include) {
            uint32_t a = static_cast<uint32_t>(key >> 32u);
            uint32_t b = static_cast<uint32_t>(key & 0xffffffffu);
            appendEdge(result, vertices, a, b);
        }
    }
    return result;
}

static void appendLine3D(std::vector<float>& out,
                         const Eigen::Vector3f& a,
                         const Eigen::Vector3f& b) {
    out.insert(out.end(), {a.x(), a.y(), a.z(), b.x(), b.y(), b.z()});
}

static void appendCone3D(std::vector<float>& out,
                         const Eigen::Vector3f& tip,
                         const Eigen::Vector3f& direction,
                         float length,
                         float radius) {
    Eigen::Vector3f axis = direction.normalized();
    Eigen::Vector3f helper = std::abs(axis.dot(Eigen::Vector3f::UnitY())) > 0.92f
        ? Eigen::Vector3f::UnitX()
        : Eigen::Vector3f::UnitY();
    Eigen::Vector3f u = axis.cross(helper).normalized();
    Eigen::Vector3f v = axis.cross(u).normalized();
    Eigen::Vector3f baseCenter = tip - axis * length;

    static constexpr int kSegments = 14;
    std::array<Eigen::Vector3f, kSegments> ring;
    for (int i = 0; i < kSegments; ++i) {
        float t = 2.0f * std::numbers::pi_v<float>
            * static_cast<float>(i) / static_cast<float>(kSegments);
        ring[static_cast<std::size_t>(i)] = baseCenter
            + radius * (std::cos(t) * u + std::sin(t) * v);
    }

    for (int i = 0; i < kSegments; ++i) {
        const auto& a = ring[static_cast<std::size_t>(i)];
        const auto& b = ring[static_cast<std::size_t>((i + 1) % kSegments)];
        appendLine3D(out, a, b);
        if (i % 2 == 0)
            appendLine3D(out, tip, a);
    }
}

static auto computeGuide3DVertices(const BoundingBox3D& bounds,
                                   const Camera3D& camera) -> std::vector<float> {
    auto b = bounds.expanded(0.04f);
    Eigen::Vector3f lo = b.min;
    Eigen::Vector3f hi = b.max;
    Eigen::Vector3f center = b.center();
    Eigen::Vector3f eye = camera.position();
    float backX = eye.x() >= center.x() ? lo.x() : hi.x();
    float backY = eye.y() >= center.y() ? lo.y() : hi.y();
    float backZ = eye.z() >= center.z() ? lo.z() : hi.z();

    std::vector<float> out;
    out.reserve(360);

    auto p = [](float x, float y, float z) { return Eigen::Vector3f(x, y, z); };

    auto gridPlaneX = [&](float x) {
        appendLine3D(out, p(x, lo.y(), lo.z()), p(x, hi.y(), lo.z()));
        appendLine3D(out, p(x, hi.y(), lo.z()), p(x, hi.y(), hi.z()));
        appendLine3D(out, p(x, hi.y(), hi.z()), p(x, lo.y(), hi.z()));
        appendLine3D(out, p(x, lo.y(), hi.z()), p(x, lo.y(), lo.z()));
    };
    auto gridPlaneY = [&](float y) {
        appendLine3D(out, p(lo.x(), y, lo.z()), p(hi.x(), y, lo.z()));
        appendLine3D(out, p(hi.x(), y, lo.z()), p(hi.x(), y, hi.z()));
        appendLine3D(out, p(hi.x(), y, hi.z()), p(lo.x(), y, hi.z()));
        appendLine3D(out, p(lo.x(), y, hi.z()), p(lo.x(), y, lo.z()));
    };
    auto gridPlaneZ = [&](float z) {
        appendLine3D(out, p(lo.x(), lo.y(), z), p(hi.x(), lo.y(), z));
        appendLine3D(out, p(hi.x(), lo.y(), z), p(hi.x(), hi.y(), z));
        appendLine3D(out, p(hi.x(), hi.y(), z), p(lo.x(), hi.y(), z));
        appendLine3D(out, p(lo.x(), hi.y(), z), p(lo.x(), lo.y(), z));
    };

    gridPlaneX(backX);
    gridPlaneY(backY);
    gridPlaneZ(backZ);

    static constexpr int kDivisions = 4;
    for (int i = 1; i < kDivisions; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(kDivisions);
        float x = std::lerp(lo.x(), hi.x(), t);
        float y = std::lerp(lo.y(), hi.y(), t);
        float z = std::lerp(lo.z(), hi.z(), t);

        appendLine3D(out, p(backX, y, lo.z()), p(backX, y, hi.z()));
        appendLine3D(out, p(backX, lo.y(), z), p(backX, hi.y(), z));

        appendLine3D(out, p(x, backY, lo.z()), p(x, backY, hi.z()));
        appendLine3D(out, p(lo.x(), backY, z), p(hi.x(), backY, z));

        appendLine3D(out, p(x, lo.y(), backZ), p(x, hi.y(), backZ));
        appendLine3D(out, p(lo.x(), y, backZ), p(hi.x(), y, backZ));
    }

    float dx = (hi.x() - lo.x()) * 0.08f;
    float dy = (hi.y() - lo.y()) * 0.08f;
    float dz = (hi.z() - lo.z()) * 0.08f;

    float xTip = eye.x() >= center.x() ? hi.x() + dx : lo.x() - dx;
    float xBase = eye.x() >= center.x() ? hi.x() : lo.x();
    float xSign = eye.x() >= center.x() ? 1.f : -1.f;
    float xConeLength = dx * 0.48f;
    Eigen::Vector3f xConeTip = p(xTip, backY, backZ);
    Eigen::Vector3f xConeAxis(xSign, 0.f, 0.f);
    appendLine3D(out, p(xBase, backY, backZ), xConeTip - xConeAxis * xConeLength);
    appendCone3D(out, xConeTip, xConeAxis, xConeLength,
                 std::max(dy, dz) * 0.14f);

    float yTip = eye.y() >= center.y() ? hi.y() + dy : lo.y() - dy;
    float yBase = eye.y() >= center.y() ? hi.y() : lo.y();
    float ySign = eye.y() >= center.y() ? 1.f : -1.f;
    float yConeLength = dy * 0.48f;
    Eigen::Vector3f yConeTip = p(backX, yTip, backZ);
    Eigen::Vector3f yConeAxis(0.f, ySign, 0.f);
    appendLine3D(out, p(backX, yBase, backZ), yConeTip - yConeAxis * yConeLength);
    appendCone3D(out, yConeTip, yConeAxis, yConeLength,
                 std::max(dx, dz) * 0.14f);

    float zTip = eye.z() >= center.z() ? hi.z() + dz : lo.z() - dz;
    float zBase = eye.z() >= center.z() ? hi.z() : lo.z();
    float zSign = eye.z() >= center.z() ? 1.f : -1.f;
    float zConeLength = dz * 0.48f;
    Eigen::Vector3f zConeTip = p(backX, backY, zTip);
    Eigen::Vector3f zConeAxis(0.f, 0.f, zSign);
    appendLine3D(out, p(backX, backY, zBase), zConeTip - zConeAxis * zConeLength);
    appendCone3D(out, zConeTip, zConeAxis, zConeLength,
                 std::max(dx, dy) * 0.14f);

    return out;
}

// ── Construction ───────────────────────────────────────────────────

PlotView::PlotView(QWidget* parent)
    : QRhiWidget(parent)
    , d(std::make_unique<Impl>())
{
    setMouseTracking(true);
    setSampleCount(4);

    d->titleLabel = new QLabel(this);
    d->titleLabel->setAlignment(Qt::AlignHCenter | Qt::AlignTop);
    d->titleLabel->hide();

    d->textOverlay = new PlotTextOverlay(this);
    d->overlay = new PlotOverlay(this);
}

PlotView::~PlotView() {
    for (auto& s : d->series2d) {
        delete s.vb;
        delete s.ub;
        delete s.srb;
    }
}

// ── Data setters ───────────────────────────────────────────────────

auto PlotView::addLineSeries(std::span<const float> x,
                             std::span<const float> y,
                             const PlotStyle& style) -> SeriesHandle {
    return addSeriesImpl(static_cast<int>(SeriesKind::Line), x, y, style);
}

auto PlotView::addScatterSeries(std::span<const float> x,
                                std::span<const float> y,
                                const PlotStyle& style) -> SeriesHandle {
    return addSeriesImpl(static_cast<int>(SeriesKind::Scatter), x, y, style);
}

auto PlotView::addSeriesImpl(int kindInt,
                             std::span<const float> x,
                             std::span<const float> y,
                             const PlotStyle& style) -> SeriesHandle {
    auto kind = static_cast<SeriesKind>(kindInt);
    auto n = static_cast<int>(std::min(x.size(), y.size()));
    if (!d->has2D() && !d->has3D())
        d->interactionTool = InteractionTool::Pan;

    Series2D series;
    series.id = nextSeriesId();
    series.kind = kind;
    series.vertices.resize(static_cast<std::size_t>(n) * 2);
    for (int i = 0; i < n; ++i) {
        series.vertices[static_cast<std::size_t>(i) * 2]     = x[static_cast<std::size_t>(i)];
        series.vertices[static_cast<std::size_t>(i) * 2 + 1] = y[static_cast<std::size_t>(i)];
    }
    series.vertexCount = n;

    int colorIdx = d->nextColorIndex;
    d->nextColorIndex++;
    Eigen::Vector4f resolvedColor = style.color.value_or(
        d->theme.seriesColors[static_cast<std::size_t>(colorIdx) % d->theme.seriesColors.size()]);
    if (style.opacity < 1.0f)
        resolvedColor.w() = style.opacity;
    series.color = resolvedColor;
    series.pointSize = style.pointSize;
    series.hollow = style.hollow;
    series.label = style.label;
    series.dirty = true;

    if (d->pipelineReady) {
        auto* r = rhi();
        quint32 ubSize = (kind == SeriesKind::Scatter) ? 96u : 80u;
        series.ub = makeRawUB(r, ubSize);
        series.srb = makeSrb(r, series.ub);
        int floats = std::max(static_cast<int>(series.vertices.size()), 1);
        series.vb = makeRawDynBuf(r, QRhiBuffer::VertexBuffer,
                                   quint32(floats * sizeof(float)));
        series.vbCapacity = floats;
    }

    const SeriesHandle handle(series.id);
    d->series2d.push_back(std::move(series));
    recomputeBounds();
    d->gridDirty = true;
    update();
    return handle;
}

auto PlotView::updateSeriesDataImpl(SeriesHandle handle,
                                    std::span<const float> x,
                                    std::span<const float> y) -> bool {
    auto seriesIt = std::ranges::find(d->series2d, handle.m_id, &Series2D::id);
    if (seriesIt == d->series2d.end() || seriesIt->kind == SeriesKind::Fill)
        return false;

    const auto count = std::min(x.size(), y.size());
    seriesIt->vertices.resize(count * 2);
    for (std::size_t i = 0; i < count; ++i) {
        seriesIt->vertices[i * 2] = x[i];
        seriesIt->vertices[i * 2 + 1] = y[i];
    }
    seriesIt->vertexCount = static_cast<int>(count);
    seriesIt->dirty = true;
    recomputeBounds();
    d->gridDirty = true;
    update();
    return true;
}

auto PlotView::setSeriesStyle(SeriesHandle handle,
                              const PlotStyle& style) -> bool {
    auto seriesIt = std::ranges::find(d->series2d, handle.m_id, &Series2D::id);
    if (seriesIt == d->series2d.end())
        return false;

    Eigen::Vector4f resolvedColor = style.color.value_or(seriesIt->color);
    if (style.opacity < 1.0f)
        resolvedColor.w() = style.opacity;
    seriesIt->color = resolvedColor;
    seriesIt->pointSize = style.pointSize;
    seriesIt->hollow = style.hollow;
    seriesIt->label = style.label;
    if (d->textOverlay) d->textOverlay->update();
    update();
    return true;
}

auto PlotView::setSeriesVisible(SeriesHandle handle, bool visible) -> bool {
    auto seriesIt = std::ranges::find(d->series2d, handle.m_id, &Series2D::id);
    if (seriesIt == d->series2d.end())
        return false;

    seriesIt->visible = visible;
    recomputeBounds();
    d->gridDirty = true;
    if (d->textOverlay) d->textOverlay->update();
    update();
    return true;
}

auto PlotView::containsSeries(SeriesHandle handle) const -> bool {
    return std::ranges::any_of(d->series2d, [handle](const Series2D& series) {
        return series.id == handle.m_id;
    });
}

auto PlotView::removeSeries(SeriesHandle handle) -> bool {
    auto seriesIt = std::ranges::find(d->series2d, handle.m_id, &Series2D::id);
    if (seriesIt == d->series2d.end())
        return false;

    delete seriesIt->vb;
    delete seriesIt->ub;
    delete seriesIt->srb;
    d->series2d.erase(seriesIt);
    recomputeBounds();
    d->gridDirty = true;
    if (d->textOverlay) d->textOverlay->update();
    update();
    return true;
}

void PlotView::startTelemetry(std::size_t windowSize,
                              const PlotStyle& style) {
    if (windowSize == 0)
        throw std::invalid_argument("telemetry window size must be positive");

    clearTelemetry();
    d->telemetrySeries = addSeriesImpl(
        static_cast<int>(SeriesKind::Line), {}, {}, style);
    d->telemetryRing.resize(windowSize * 2);
    d->telemetryHead = 0;
    d->telemetrySize = 0;
    d->telemetryNextX = 0.0f;
    auto seriesIt = std::ranges::find(
        d->series2d, d->telemetrySeries->m_id, &Series2D::id);
    seriesIt->vertices.reserve(windowSize * 2);
}

void PlotView::appendTelemetry(float value) {
    const float x = d->telemetryNextX;
    appendTelemetry(x, value);
}

void PlotView::appendTelemetry(float x, float y) {
    if (!d->telemetrySeries)
        throw std::logic_error("startTelemetry must be called before appending samples");

    const std::size_t capacity = d->telemetryRing.size() / 2;
    std::size_t position = (d->telemetryHead + d->telemetrySize) % capacity;
    if (d->telemetrySize == capacity) {
        position = d->telemetryHead;
        d->telemetryHead = (d->telemetryHead + 1) % capacity;
    } else {
        ++d->telemetrySize;
    }
    d->telemetryRing[position * 2] = x;
    d->telemetryRing[position * 2 + 1] = y;

    auto seriesIt = std::ranges::find(
        d->series2d, d->telemetrySeries->m_id, &Series2D::id);
    auto& series = *seriesIt;
    series.vertices.resize(d->telemetrySize * 2);
    for (std::size_t i = 0; i < d->telemetrySize; ++i) {
        const std::size_t source = (d->telemetryHead + i) % capacity;
        series.vertices[i * 2] = d->telemetryRing[source * 2];
        series.vertices[i * 2 + 1] = d->telemetryRing[source * 2 + 1];
    }
    series.vertexCount = static_cast<int>(d->telemetrySize);
    series.dirty = true;
    d->telemetryNextX = x + 1.0f;

    recomputeBounds();
    d->gridDirty = true;
    update();
}

auto PlotView::telemetryPointCount() const -> std::size_t {
    return d->telemetrySize;
}

void PlotView::clearTelemetry() {
    if (!d->telemetrySeries)
        return;

    removeSeries(*d->telemetrySeries);
    d->telemetrySeries.reset();
    d->telemetryRing.clear();
    d->telemetryHead = 0;
    d->telemetrySize = 0;
    d->telemetryNextX = 0.0f;
    recomputeBounds();
    d->gridDirty = true;
    update();
}

void PlotView::addFillSeries(std::span<const float> triangleVertices,
                             const PlotStyle& style) {
    if (!d->has2D() && !d->has3D())
        d->interactionTool = InteractionTool::Pan;

    Series2D series;
    series.kind = SeriesKind::Fill;
    series.vertices.assign(triangleVertices.begin(), triangleVertices.end());
    series.vertexCount = static_cast<int>(series.vertices.size() / 2);

    int colorIdx = d->nextColorIndex;
    d->nextColorIndex++;
    Eigen::Vector4f resolvedColor = style.color.value_or(
        d->theme.seriesColors[static_cast<std::size_t>(colorIdx) % d->theme.seriesColors.size()]);
    if (style.opacity < 1.0f)
        resolvedColor.w() = style.opacity;
    series.color = resolvedColor;
    series.dirty = true;

    if (d->pipelineReady) {
        auto* r = rhi();
        series.ub = makeRawUB(r, 80u);
        series.srb = makeSrb(r, series.ub);
        int floats = std::max(static_cast<int>(series.vertices.size()), 1);
        series.vb = makeRawDynBuf(r, QRhiBuffer::VertexBuffer,
                                  quint32(floats * sizeof(float)));
        series.vbCapacity = floats;
    }

    d->series2d.push_back(std::move(series));
    recomputeBounds();
    d->gridDirty = true;
    update();
}

namespace {

// Append the two triangles of an axis-aligned rectangle (6 vec2 vertices).
void appendQuad(std::vector<float>& out,
                float x0, float y0, float x1, float y1) {
    out.insert(out.end(), {
        x0, y0,  x1, y0,  x1, y1,   // triangle 1
        x0, y0,  x1, y1,  x0, y1,   // triangle 2
    });
}

} // namespace

void PlotView::histImpl(std::span<const float> values, int bins, bool density,
                        const PlotStyle& style) {
    const int n = static_cast<int>(values.size());
    if (n == 0) return;

    float lo = values[0], hi = values[0];
    for (float v : values) { lo = std::min(lo, v); hi = std::max(hi, v); }
    if (hi <= lo) hi = lo + 1.0f;

    // Sturges' rule default: ceil(log2(n)) + 1.
    if (bins <= 0)
        bins = std::max(1, static_cast<int>(std::ceil(std::log2(std::max(1, n)) + 1.0)));

    std::vector<int> counts(static_cast<std::size_t>(bins), 0);
    const float binW = (hi - lo) / static_cast<float>(bins);
    for (float v : values) {
        int b = static_cast<int>((v - lo) / binW);
        b = std::clamp(b, 0, bins - 1);
        counts[static_cast<std::size_t>(b)]++;
    }

    // Bar height: raw count or probability density (area sums to 1).
    auto height = [&](int c) -> float {
        if (!density) return static_cast<float>(c);
        return static_cast<float>(c) / (static_cast<float>(n) * binW);
    };

    std::vector<float> verts;
    verts.reserve(static_cast<std::size_t>(bins) * 12);
    // Tiny gap between bars for visual separation (matplotlib uses rwidth).
    const float gap = binW * 0.02f;
    for (int b = 0; b < bins; ++b) {
        float x0 = lo + static_cast<float>(b) * binW + gap;
        float x1 = lo + static_cast<float>(b + 1) * binW - gap;
        appendQuad(verts, x0, 0.0f, x1, height(counts[static_cast<std::size_t>(b)]));
    }
    addFillSeries(verts, style);
}

void PlotView::barImpl(std::span<const float> positions,
                       std::span<const float> sizes,
                       float width, bool horizontal, const PlotStyle& style) {
    const int n = static_cast<int>(std::min(positions.size(), sizes.size()));
    if (n == 0) return;

    const float half = width * 0.5f;
    std::vector<float> verts;
    verts.reserve(static_cast<std::size_t>(n) * 12);
    for (int i = 0; i < n; ++i) {
        float p = positions[static_cast<std::size_t>(i)];
        float s = sizes[static_cast<std::size_t>(i)];
        if (horizontal)
            appendQuad(verts, 0.0f, p - half, s, p + half);
        else
            appendQuad(verts, p - half, 0.0f, p + half, s);
    }
    addFillSeries(verts, style);
}

void PlotView::fillBetweenImpl(std::span<const float> x,
                               std::span<const float> y0,
                               std::span<const float> y1,
                               const PlotStyle& style) {
    const int n = static_cast<int>(std::min({x.size(), y0.size(), y1.size()}));
    if (n < 2) return;

    std::vector<float> verts;
    verts.reserve(static_cast<std::size_t>(n - 1) * 12);
    for (int i = 0; i < n - 1; ++i) {
        float xa = x[static_cast<std::size_t>(i)];
        float xb = x[static_cast<std::size_t>(i + 1)];
        float a0 = y0[static_cast<std::size_t>(i)],     a1 = y1[static_cast<std::size_t>(i)];
        float b0 = y0[static_cast<std::size_t>(i + 1)], b1 = y1[static_cast<std::size_t>(i + 1)];
        // Quad (xa,a0)-(xb,b0)-(xb,b1)-(xa,a1) as two triangles.
        verts.insert(verts.end(), {
            xa, a0,  xb, b0,  xb, b1,
            xa, a0,  xb, b1,  xa, a1,
        });
    }
    PlotStyle s = style;
    if (s.opacity >= 1.0f && !s.color) s.opacity = 0.4f; // translucent band by default
    addFillSeries(verts, s);
}

void PlotView::stepImpl(std::span<const float> x, std::span<const float> y,
                        const PlotStyle& style) {
    const int n = static_cast<int>(std::min(x.size(), y.size()));
    if (n == 0) return;
    // Expand to a piecewise-constant polyline: (x0,y0)-(x1,y0)-(x1,y1)-...
    std::vector<float> xs, ys;
    xs.reserve(static_cast<std::size_t>(n) * 2);
    ys.reserve(static_cast<std::size_t>(n) * 2);
    for (int i = 0; i < n; ++i) {
        if (i > 0) {  // horizontal tread to the new x at the previous y
            xs.push_back(x[static_cast<std::size_t>(i)]);
            ys.push_back(y[static_cast<std::size_t>(i - 1)]);
        }
        xs.push_back(x[static_cast<std::size_t>(i)]);
        ys.push_back(y[static_cast<std::size_t>(i)]);
    }
    addSeriesImpl(static_cast<int>(SeriesKind::Line),
                  {xs.data(), xs.size()}, {ys.data(), ys.size()}, style);
}

void PlotView::errorbarImpl(std::span<const float> x, std::span<const float> y,
                            std::span<const float> yerr, const PlotStyle& style) {
    const int n = static_cast<int>(std::min({x.size(), y.size(), yerr.size()}));
    if (n == 0) return;

    // Whisker thickness / cap width as a fraction of the data x-range.
    float xlo = x[0], xhi = x[0];
    for (int i = 0; i < n; ++i) { xlo = std::min(xlo, x[static_cast<std::size_t>(i)]);
                                  xhi = std::max(xhi, x[static_cast<std::size_t>(i)]); }
    float xspan = std::max(xhi - xlo, 1e-6f);
    const float stemHalf = xspan * 0.0015f;  // half-thickness of the vertical stem
    const float capHalf  = xspan * 0.010f;   // half-width of the end caps
    const float capThick = xspan * 0.0015f;

    std::vector<float> verts;
    verts.reserve(static_cast<std::size_t>(n) * 36);
    for (int i = 0; i < n; ++i) {
        float xi = x[static_cast<std::size_t>(i)];
        float yi = y[static_cast<std::size_t>(i)];
        float ei = std::abs(yerr[static_cast<std::size_t>(i)]);
        appendQuad(verts, xi - stemHalf, yi - ei, xi + stemHalf, yi + ei);   // vertical stem
        appendQuad(verts, xi - capHalf, yi + ei - capThick, xi + capHalf, yi + ei + capThick); // top cap
        appendQuad(verts, xi - capHalf, yi - ei - capThick, xi + capHalf, yi - ei + capThick); // bottom cap
    }
    addFillSeries(verts, style);
}

void PlotView::stemImpl(std::span<const float> x, std::span<const float> y,
                        const PlotStyle& style) {
    const int n = static_cast<int>(std::min(x.size(), y.size()));
    if (n == 0) return;

    float xlo = x[0], xhi = x[0];
    for (int i = 0; i < n; ++i) { xlo = std::min(xlo, x[static_cast<std::size_t>(i)]);
                                  xhi = std::max(xhi, x[static_cast<std::size_t>(i)]); }
    const float stemHalf = std::max(xhi - xlo, 1e-6f) * 0.0015f;

    // Thin vertical quads from the baseline (y=0) to each sample.
    std::vector<float> verts;
    verts.reserve(static_cast<std::size_t>(n) * 12);
    std::vector<float> mx(static_cast<std::size_t>(n)), my(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        float xi = x[static_cast<std::size_t>(i)];
        float yi = y[static_cast<std::size_t>(i)];
        appendQuad(verts, xi - stemHalf, 0.0f, xi + stemHalf, yi);
        mx[static_cast<std::size_t>(i)] = xi;
        my[static_cast<std::size_t>(i)] = yi;
    }
    addFillSeries(verts, style);
    // Markers at the stem tops (re-use the resolved series colour).
    PlotStyle markerStyle = style;
    markerStyle.pointSize = (style.pointSize > 0.0f) ? style.pointSize : 6.0f;
    addSeriesImpl(static_cast<int>(SeriesKind::Scatter),
                  {mx.data(), mx.size()}, {my.data(), my.size()}, markerStyle);
}

void PlotView::boxplot(const std::vector<Eigen::VectorXf>& groups,
                       const PlotStyle& style) {
    if (groups.empty()) return;
    if (!d->has2D() && !d->has3D())
        d->interactionTool = InteractionTool::Pan;

    const float boxHalf = 0.3f;     // half box width in group-position units
    const float capHalf = 0.15f;    // whisker-cap half width

    // Percentile of a *sorted* sample via linear interpolation.
    auto percentile = [](std::vector<float>& s, float p) -> float {
        if (s.empty()) return 0.0f;
        if (s.size() == 1) return s[0];
        float idx = p * static_cast<float>(s.size() - 1);
        auto i0 = static_cast<std::size_t>(idx);
        float f = idx - static_cast<float>(i0);
        if (i0 + 1 >= s.size()) return s.back();
        return s[i0] * (1.0f - f) + s[i0 + 1] * f;
    };

    std::vector<float> boxVerts;     // filled box bodies + median/whisker lines
    std::vector<float> outX, outY;   // outlier markers

    // Line thickness expressed in x-units (positions are integers, span ~G).
    const float lineHalf = std::max(static_cast<float>(groups.size()), 1.0f) * 0.004f;

    for (std::size_t g = 0; g < groups.size(); ++g) {
        std::vector<float> s(groups[g].data(), groups[g].data() + groups[g].size());
        if (s.empty()) continue;
        std::sort(s.begin(), s.end());

        float q1 = percentile(s, 0.25f);
        float med = percentile(s, 0.50f);
        float q3 = percentile(s, 0.75f);
        float iqr = q3 - q1;
        float loFence = q1 - 1.5f * iqr;
        float hiFence = q3 + 1.5f * iqr;

        // Whisker ends = most extreme samples within the fences.
        float whiskLo = q1, whiskHi = q3;
        for (float v : s) {
            if (v >= loFence && v < whiskLo) whiskLo = v;
            if (v <= hiFence && v > whiskHi) whiskHi = v;
            if (v < loFence || v > hiFence) {
                outX.push_back(static_cast<float>(g));
                outY.push_back(v);
            }
        }

        float cx = static_cast<float>(g);
        float x0 = cx - boxHalf, x1 = cx + boxHalf;

        // Box body (translucent fill is applied via the series colour alpha).
        appendQuad(boxVerts, x0, q1, x1, q3);
        // Box outline (four thin quads) + median line.
        appendQuad(boxVerts, x0, q1 - lineHalf, x1, q1 + lineHalf);          // bottom
        appendQuad(boxVerts, x0, q3 - lineHalf, x1, q3 + lineHalf);          // top
        appendQuad(boxVerts, x0 - lineHalf, q1, x0 + lineHalf, q3);          // left
        appendQuad(boxVerts, x1 - lineHalf, q1, x1 + lineHalf, q3);          // right
        appendQuad(boxVerts, x0, med - lineHalf, x1, med + lineHalf);        // median
        // Whiskers (vertical stems + caps).
        appendQuad(boxVerts, cx - lineHalf, whiskHi, cx + lineHalf, q3);     // upper stem
        appendQuad(boxVerts, cx - lineHalf, q1, cx + lineHalf, whiskLo);     // lower stem
        appendQuad(boxVerts, cx - capHalf, whiskHi - lineHalf, cx + capHalf, whiskHi + lineHalf);
        appendQuad(boxVerts, cx - capHalf, whiskLo - lineHalf, cx + capHalf, whiskLo + lineHalf);
    }

    PlotStyle boxStyle = style;
    if (boxStyle.opacity >= 1.0f && !boxStyle.color)
        boxStyle.opacity = 0.55f;  // translucent body so the median reads
    addFillSeries(boxVerts, boxStyle);

    if (!outX.empty()) {
        PlotStyle outStyle = style;
        outStyle.pointSize = 5.0f;
        addSeriesImpl(static_cast<int>(SeriesKind::Scatter),
                      {outX.data(), outX.size()}, {outY.data(), outY.size()}, outStyle);
    }
}

void PlotView::violinplot(const std::vector<Eigen::VectorXf>& groups,
                          const PlotStyle& style) {
    if (groups.empty()) return;
    if (!d->has2D() && !d->has3D())
        d->interactionTool = InteractionTool::Pan;

    constexpr int kResolution = 48;     // vertical density samples
    const float maxHalfWidth = 0.38f;   // max violin half-width in position units

    std::vector<float> verts;
    std::vector<float> medX, medY;      // median tick markers
    for (std::size_t g = 0; g < groups.size(); ++g) {
        const Eigen::VectorXf& gv = groups[g];
        if (gv.size() == 0) continue;
        std::vector<float> s(gv.data(), gv.data() + gv.size());
        std::sort(s.begin(), s.end());
        const auto n = static_cast<float>(s.size());

        float lo = s.front(), hi = s.back();
        if (hi <= lo) hi = lo + 1.0f;
        // Silverman's rule-of-thumb KDE bandwidth.
        float mean = 0.0f; for (float v : s) mean += v; mean /= n;
        float var = 0.0f; for (float v : s) var += (v - mean) * (v - mean);
        float sd = std::sqrt(var / std::max(1.0f, n - 1.0f));
        float bw = 1.06f * std::max(sd, 1e-6f) * std::pow(n, -0.2f);

        // Evaluate the Gaussian KDE on a vertical grid and find the peak.
        std::vector<float> dens(kResolution + 1), yv(kResolution + 1);
        float peak = 0.0f;
        for (int i = 0; i <= kResolution; ++i) {
            float y = lo + (hi - lo) * static_cast<float>(i) / kResolution;
            yv[static_cast<std::size_t>(i)] = y;
            float acc = 0.0f;
            for (float xi : s) {
                float z = (y - xi) / bw;
                acc += std::exp(-0.5f * z * z);
            }
            float d = acc / (n * bw * 2.5066283f); // sqrt(2*pi)
            dens[static_cast<std::size_t>(i)] = d;
            peak = std::max(peak, d);
        }
        if (peak <= 0.0f) continue;

        float cx = static_cast<float>(g);
        // Build the symmetric body as a triangle strip of trapezoids.
        for (int i = 0; i < kResolution; ++i) {
            float w0 = maxHalfWidth * dens[static_cast<std::size_t>(i)] / peak;
            float w1 = maxHalfWidth * dens[static_cast<std::size_t>(i + 1)] / peak;
            float y0 = yv[static_cast<std::size_t>(i)];
            float y1 = yv[static_cast<std::size_t>(i + 1)];
            // Quad spanning [cx-w0,cx+w0]@y0 to [cx-w1,cx+w1]@y1.
            verts.insert(verts.end(), {
                cx - w0, y0,  cx + w0, y0,  cx + w1, y1,
                cx - w0, y0,  cx + w1, y1,  cx - w1, y1,
            });
        }
        // Median marker.
        float med = s[s.size() / 2];
        medX.push_back(cx);
        medY.push_back(med);
    }

    PlotStyle vStyle = style;
    if (vStyle.opacity >= 1.0f && !vStyle.color)
        vStyle.opacity = 0.6f;
    addFillSeries(verts, vStyle);

    if (!medX.empty()) {
        PlotStyle medStyle = style;
        medStyle.pointSize = 5.0f;
        addSeriesImpl(static_cast<int>(SeriesKind::Scatter),
                      {medX.data(), medX.size()}, {medY.data(), medY.size()}, medStyle);
    }
}

void PlotView::quiverImpl(std::span<const float> x, std::span<const float> y,
                          std::span<const float> u, std::span<const float> v,
                          const PlotStyle& style) {
    const int n = static_cast<int>(std::min({x.size(), y.size(), u.size(), v.size()}));
    if (n == 0) return;

    // Shaft thickness / arrowhead size as a fraction of the mean vector length.
    float meanLen = 0.0f;
    for (int i = 0; i < n; ++i) {
        float ui = u[static_cast<std::size_t>(i)], vi = v[static_cast<std::size_t>(i)];
        meanLen += std::sqrt(ui * ui + vi * vi);
    }
    meanLen = std::max(meanLen / static_cast<float>(n), 1e-6f);
    const float shaftHalf = meanLen * 0.04f;
    const float headLen = meanLen * 0.30f;
    const float headHalf = meanLen * 0.16f;

    std::vector<float> verts;
    verts.reserve(static_cast<std::size_t>(n) * 18);
    for (int i = 0; i < n; ++i) {
        float px = x[static_cast<std::size_t>(i)], py = y[static_cast<std::size_t>(i)];
        float ux = u[static_cast<std::size_t>(i)], uy = v[static_cast<std::size_t>(i)];
        float len = std::sqrt(ux * ux + uy * uy);
        if (len < 1e-9f) continue;
        float dx = ux / len, dy = uy / len;     // unit direction
        float nx = -dy, ny = dx;                 // unit normal
        float tipX = px + ux, tipY = py + uy;
        float baseX = tipX - dx * headLen, baseY = tipY - dy * headLen;
        // Shaft as a thin quad from origin to the head base.
        verts.insert(verts.end(), {
            px + nx * shaftHalf, py + ny * shaftHalf,
            px - nx * shaftHalf, py - ny * shaftHalf,
            baseX - nx * shaftHalf, baseY - ny * shaftHalf,
            px + nx * shaftHalf, py + ny * shaftHalf,
            baseX - nx * shaftHalf, baseY - ny * shaftHalf,
            baseX + nx * shaftHalf, baseY + ny * shaftHalf,
        });
        // Arrowhead triangle.
        verts.insert(verts.end(), {
            tipX, tipY,
            baseX + nx * headHalf, baseY + ny * headHalf,
            baseX - nx * headHalf, baseY - ny * headHalf,
        });
    }
    addFillSeries(verts, style);
}

void PlotView::pieImpl(std::span<const float> values) {
    const int n = static_cast<int>(values.size());
    if (n == 0) return;
    if (!d->has2D() && !d->has3D())
        d->interactionTool = InteractionTool::Pan;

    float total = 0.0f;
    for (float v : values) total += std::max(0.0f, v);
    if (total <= 0.0f) return;

    constexpr float pi = 3.14159265358979323846f;
    constexpr int kArcSteps = 48;       // arc tessellation per full circle
    const float radius = 1.0f;
    const float cx = 0.0f, cy = 0.0f;

    // Each wedge becomes its own Fill series so it picks up a palette colour.
    float angle = pi * 0.5f;            // start at the top
    for (int i = 0; i < n; ++i) {
        float frac = std::max(0.0f, values[static_cast<std::size_t>(i)]) / total;
        float sweep = frac * 2.0f * pi;
        int steps = std::max(1, static_cast<int>(std::ceil(frac * kArcSteps)));
        float a0 = angle;
        std::vector<float> verts;
        verts.reserve(static_cast<std::size_t>(steps) * 6);
        for (int s = 0; s < steps; ++s) {
            float t0 = a0 - sweep * static_cast<float>(s) / steps;       // clockwise
            float t1 = a0 - sweep * static_cast<float>(s + 1) / steps;
            verts.insert(verts.end(), {
                cx, cy,
                cx + radius * std::cos(t0), cy + radius * std::sin(t0),
                cx + radius * std::cos(t1), cy + radius * std::sin(t1),
            });
        }
        addFillSeries(verts, {});
        angle -= sweep;
    }

    // Square the data bounds and force equal aspect so the pie stays round.
    BoundingBox2D bb;
    bb.min = Eigen::Vector2f(-radius * 1.15f, -radius * 1.15f);
    bb.max = Eigen::Vector2f(radius * 1.15f, radius * 1.15f);
    d->bounds2d = d->bounds2d.merge(bb);
    d->aspectEqual = true;
    d->gridDirty = true;
    update();
}

void PlotView::imshowImpl(std::span<const float> data, int rows, int cols,
                          Colormap cmap, float vmin, float vmax) {
    if (rows <= 0 || cols <= 0) return;
    if (!d->has2D() && !d->has3D())
        d->interactionTool = InteractionTool::Pan;

    // Eigen storage is column-major: element (r, c) is at index c*rows + r.
    auto at = [&](int r, int c) -> float {
        return data[static_cast<std::size_t>(c) * static_cast<std::size_t>(rows)
                    + static_cast<std::size_t>(r)];
    };

    // Auto data range unless an explicit [vmin, vmax) was supplied.
    if (!(vmin < vmax)) {
        vmin = at(0, 0); vmax = at(0, 0);
        for (int c = 0; c < cols; ++c)
            for (int r = 0; r < rows; ++r) {
                float v = at(r, c);
                vmin = std::min(vmin, v); vmax = std::max(vmax, v);
            }
    }
    const float range = std::max(vmax - vmin, 1e-12f);

    // Interleaved (vec2 pos, vec4 colour) per vertex, 6 vertices per cell.
    auto& verts = d->heatmapVertices;
    verts.clear();
    verts.reserve(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols) * 36);
    auto push = [&](float x, float y, const Eigen::Vector4f& col) {
        verts.insert(verts.end(), {x, y, col.x(), col.y(), col.z(), col.w()});
    };
    for (int r = 0; r < rows; ++r) {
        // Row 0 at the top (matplotlib imshow convention).
        float y0 = static_cast<float>(rows - 1 - r);
        float y1 = static_cast<float>(rows - r);
        for (int c = 0; c < cols; ++c) {
            float t = (at(r, c) - vmin) / range;
            Eigen::Vector4f col = sampleColormap(cmap, t);
            float x0 = static_cast<float>(c);
            float x1 = static_cast<float>(c + 1);
            push(x0, y0, col); push(x1, y0, col); push(x1, y1, col);
            push(x0, y0, col); push(x1, y1, col); push(x0, y1, col);
        }
    }
    d->heatmapVertexCount = static_cast<int>(verts.size() / 6);
    d->hasHeatmap = true;
    d->heatmapDirty = true;
    d->hasColormappedData = true;
    d->colorbarMap = cmap;
    d->colorbarVmin = vmin;
    d->colorbarVmax = vmax;

    // Bounds: the full cell grid.
    d->bounds2d = BoundingBox2D();
    BoundingBox2D bb;
    bb.min = Eigen::Vector2f(0.0f, 0.0f);
    bb.max = Eigen::Vector2f(static_cast<float>(cols), static_cast<float>(rows));
    d->bounds2d = d->bounds2d.merge(bb);

    if (d->pipelineReady) {
        auto* r = rhi();
        int floats = std::max(static_cast<int>(verts.size()), 1);
        d->heatmapBuffer = makeDynBuf(r, QRhiBuffer::VertexBuffer,
                                      quint32(floats * sizeof(float)));
        d->heatmapCapacity = floats;
    }
    d->gridDirty = true;
    update();
}

void PlotView::contourfImpl(std::span<const float> data, int rows, int cols,
                            int levels, Colormap cmap) {
    if (rows <= 0 || cols <= 0) return;
    levels = std::max(2, levels);

    auto at = [&](int r, int c) -> float {
        return data[static_cast<std::size_t>(c) * static_cast<std::size_t>(rows)
                    + static_cast<std::size_t>(r)];
    };
    float vmin = at(0, 0), vmax = at(0, 0);
    for (int c = 0; c < cols; ++c)
        for (int r = 0; r < rows; ++r) {
            float v = at(r, c); vmin = std::min(vmin, v); vmax = std::max(vmax, v);
        }
    const float range = std::max(vmax - vmin, 1e-12f);

    auto& verts = d->heatmapVertices;
    verts.clear();
    verts.reserve(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols) * 36);
    auto push = [&](float x, float y, const Eigen::Vector4f& col) {
        verts.insert(verts.end(), {x, y, col.x(), col.y(), col.z(), col.w()});
    };
    const auto fLevels = static_cast<float>(levels);
    for (int r = 0; r < rows; ++r) {
        float y0 = static_cast<float>(rows - 1 - r);
        float y1 = static_cast<float>(rows - r);
        for (int c = 0; c < cols; ++c) {
            float tRaw = (at(r, c) - vmin) / range;
            // Quantise to discrete bands (filled-contour look).
            int band = std::min(levels - 1, static_cast<int>(tRaw * fLevels));
            float t = (static_cast<float>(band) + 0.5f) / fLevels;
            Eigen::Vector4f col = sampleColormap(cmap, t);
            float x0 = static_cast<float>(c);
            float x1 = static_cast<float>(c + 1);
            push(x0, y0, col); push(x1, y0, col); push(x1, y1, col);
            push(x0, y0, col); push(x1, y1, col); push(x0, y1, col);
        }
    }
    d->heatmapVertexCount = static_cast<int>(verts.size() / 6);
    d->hasHeatmap = true;
    d->heatmapDirty = true;
    d->hasColormappedData = true;
    d->colorbarMap = cmap;
    d->colorbarVmin = vmin;
    d->colorbarVmax = vmax;

    d->bounds2d = BoundingBox2D();
    BoundingBox2D bb;
    bb.min = Eigen::Vector2f(0.0f, 0.0f);
    bb.max = Eigen::Vector2f(static_cast<float>(cols), static_cast<float>(rows));
    d->bounds2d = d->bounds2d.merge(bb);

    if (d->pipelineReady) {
        auto* r = rhi();
        int floats = std::max(static_cast<int>(verts.size()), 1);
        d->heatmapBuffer = makeDynBuf(r, QRhiBuffer::VertexBuffer,
                                      quint32(floats * sizeof(float)));
        d->heatmapCapacity = floats;
    }
    d->gridDirty = true;
    update();
}

void PlotView::contourImpl(std::span<const float> data, int rows, int cols,
                           int levels, const PlotStyle& style) {
    if (rows < 2 || cols < 2) return;
    levels = std::max(1, levels);
    if (!d->has2D() && !d->has3D())
        d->interactionTool = InteractionTool::Pan;

    auto at = [&](int r, int c) -> float {
        return data[static_cast<std::size_t>(c) * static_cast<std::size_t>(rows)
                    + static_cast<std::size_t>(r)];
    };
    float vmin = at(0, 0), vmax = at(0, 0);
    for (int c = 0; c < cols; ++c)
        for (int r = 0; r < rows; ++r) {
            float v = at(r, c); vmin = std::min(vmin, v); vmax = std::max(vmax, v);
        }
    if (vmax <= vmin) return;

    // Node position for grid index (r, c) — cell-centred to align with imshow.
    auto nodeX = [&](int c) { return static_cast<float>(c) + 0.5f; };
    auto nodeY = [&](int r) { return static_cast<float>(rows - 1 - r) + 0.5f; };

    auto& verts = d->contourVertices;
    verts.clear();
    auto seg = [&](float xa, float ya, float xb, float yb) {
        verts.insert(verts.end(), {xa, ya, xb, yb});
    };
    // Linear interpolation of the crossing point along an edge.
    auto lerp = [](float a, float b, float iso, float pa, float pb) {
        float d = (b - a);
        float t = (std::abs(d) < 1e-12f) ? 0.5f : (iso - a) / d;
        return pa + std::clamp(t, 0.0f, 1.0f) * (pb - pa);
    };

    for (int li = 1; li <= levels; ++li) {
        float iso = vmin + (vmax - vmin) * static_cast<float>(li) / static_cast<float>(levels + 1);
        // Marching squares over each 2x2 cell.
        for (int r = 0; r < rows - 1; ++r) {
            for (int c = 0; c < cols - 1; ++c) {
                float tl = at(r, c),     tr = at(r, c + 1);
                float bl = at(r + 1, c), br = at(r + 1, c + 1);
                int code = (tl > iso ? 8 : 0) | (tr > iso ? 4 : 0) |
                           (br > iso ? 2 : 0) | (bl > iso ? 1 : 0);
                if (code == 0 || code == 15) continue;

                float xL = nodeX(c), xR = nodeX(c + 1);
                float yT = nodeY(r), yB = nodeY(r + 1);
                // Edge crossing points (top, right, bottom, left).
                float topX = lerp(tl, tr, iso, xL, xR), topY = yT;
                float rightX = xR,                       rightY = lerp(tr, br, iso, yT, yB);
                float botX = lerp(bl, br, iso, xL, xR), botY = yB;
                float leftX = xL,                        leftY = lerp(tl, bl, iso, yT, yB);

                switch (code) {
                    case 1: case 14: seg(leftX, leftY, botX, botY); break;
                    case 2: case 13: seg(botX, botY, rightX, rightY); break;
                    case 3: case 12: seg(leftX, leftY, rightX, rightY); break;
                    case 4: case 11: seg(topX, topY, rightX, rightY); break;
                    case 6: case 9:  seg(topX, topY, botX, botY); break;
                    case 7: case 8:  seg(leftX, leftY, topX, topY); break;
                    case 5:  // saddle: two segments
                        seg(leftX, leftY, topX, topY);
                        seg(botX, botY, rightX, rightY);
                        break;
                    case 10: // saddle
                        seg(leftX, leftY, botX, botY);
                        seg(topX, topY, rightX, rightY);
                        break;
                    default: break;
                }
            }
        }
    }

    d->contourVertexCount = static_cast<int>(verts.size() / 2);
    d->hasContour = true;
    d->contourDirty = true;
    d->contourColor = style.color.value_or(
        d->theme.seriesColors[static_cast<std::size_t>(d->nextColorIndex++)
                              % d->theme.seriesColors.size()]);
    if (style.opacity < 1.0f) d->contourColor.w() = style.opacity;

    // Bounds cover the cell grid (so contour-only views still frame nicely).
    BoundingBox2D bb;
    bb.min = Eigen::Vector2f(0.0f, 0.0f);
    bb.max = Eigen::Vector2f(static_cast<float>(cols), static_cast<float>(rows));
    d->bounds2d = d->bounds2d.merge(bb);

    if (d->pipelineReady) {
        auto* r = rhi();
        int floats = std::max(static_cast<int>(verts.size()), 1);
        d->contourBuffer = makeDynBuf(r, QRhiBuffer::VertexBuffer,
                                      quint32(floats * sizeof(float)));
        d->contourCapacity = floats;
    }
    d->gridDirty = true;
    update();
}

void PlotView::hexbinImpl(std::span<const float> x, std::span<const float> y,
                          int gridsize, Colormap cmap) {
    const int n = static_cast<int>(std::min(x.size(), y.size()));
    if (n == 0) return;
    gridsize = std::max(2, gridsize);
    if (!d->has2D() && !d->has3D())
        d->interactionTool = InteractionTool::Pan;

    float xlo = x[0], xhi = x[0], ylo = y[0], yhi = y[0];
    for (int i = 0; i < n; ++i) {
        xlo = std::min(xlo, x[static_cast<std::size_t>(i)]);
        xhi = std::max(xhi, x[static_cast<std::size_t>(i)]);
        ylo = std::min(ylo, y[static_cast<std::size_t>(i)]);
        yhi = std::max(yhi, y[static_cast<std::size_t>(i)]);
    }
    float xrange = std::max(xhi - xlo, 1e-6f);
    float yrange = std::max(yhi - ylo, 1e-6f);

    // Pointy-top hex grid with horizontal spacing dx and vertical spacing dy.
    const float dx = xrange / static_cast<float>(gridsize);
    const float radius = dx / std::sqrt(3.0f);   // hexagon circumradius
    const float dyRow = radius * 1.5f;            // vertical row spacing
    const int rows = std::max(2, static_cast<int>(std::ceil(yrange / dyRow)) + 1);

    // Bin: assign each point to the nearest hex centre (offset rows).
    std::vector<int> counts(static_cast<std::size_t>(gridsize + 1) *
                            static_cast<std::size_t>(rows + 1), 0);
    auto idx = [&](int c, int r) { return static_cast<std::size_t>(r) *
                                          static_cast<std::size_t>(gridsize + 1) +
                                          static_cast<std::size_t>(c); };
    auto cellCenter = [&](int c, int r) -> std::pair<float, float> {
        float ox = (r & 1) ? dx * 0.5f : 0.0f;     // offset odd rows
        return {xlo + dx * static_cast<float>(c) + ox,
                ylo + dyRow * static_cast<float>(r)};
    };

    int maxCount = 0;
    for (int i = 0; i < n; ++i) {
        float px = x[static_cast<std::size_t>(i)], py = y[static_cast<std::size_t>(i)];
        int r = std::clamp(static_cast<int>(std::round((py - ylo) / dyRow)), 0, rows);
        float ox = (r & 1) ? dx * 0.5f : 0.0f;
        int c = std::clamp(static_cast<int>(std::round((px - xlo - ox) / dx)), 0, gridsize);
        int& cnt = counts[idx(c, r)];
        cnt++;
        maxCount = std::max(maxCount, cnt);
    }
    if (maxCount == 0) return;

    // Emit a coloured hexagon for every non-empty bin.
    auto& verts = d->heatmapVertices;
    verts.clear();
    constexpr float pi = 3.14159265358979323846f;
    for (int r = 0; r <= rows; ++r) {
        for (int c = 0; c <= gridsize; ++c) {
            int cnt = counts[idx(c, r)];
            if (cnt == 0) continue;
            auto [hx, hy] = cellCenter(c, r);
            float t = static_cast<float>(cnt) / static_cast<float>(maxCount);
            Eigen::Vector4f col = sampleColormap(cmap, t);
            // Pointy-top hexagon: 6 triangles fanning from the centre.
            float prevX = 0, prevY = 0;
            for (int k = 0; k <= 6; ++k) {
                float a = pi / 6.0f + pi / 3.0f * static_cast<float>(k);
                float vx = hx + radius * std::cos(a);
                float vy = hy + radius * std::sin(a);
                if (k > 0) {
                    verts.insert(verts.end(), {
                        hx, hy, col.x(), col.y(), col.z(), col.w(),
                        prevX, prevY, col.x(), col.y(), col.z(), col.w(),
                        vx, vy, col.x(), col.y(), col.z(), col.w(),
                    });
                }
                prevX = vx; prevY = vy;
            }
        }
    }
    d->heatmapVertexCount = static_cast<int>(verts.size() / 6);
    d->hasHeatmap = true;
    d->heatmapDirty = true;
    d->hasColormappedData = true;
    d->colorbarMap = cmap;
    d->colorbarVmin = 0.0f;
    d->colorbarVmax = static_cast<float>(maxCount);

    BoundingBox2D bb;
    bb.min = Eigen::Vector2f(xlo - dx, ylo - dyRow);
    bb.max = Eigen::Vector2f(xhi + dx, yhi + dyRow);
    d->bounds2d = d->bounds2d.merge(bb);

    if (d->pipelineReady) {
        auto* rr = rhi();
        int floats = std::max(static_cast<int>(verts.size()), 1);
        d->heatmapBuffer = makeDynBuf(rr, QRhiBuffer::VertexBuffer,
                                      quint32(floats * sizeof(float)));
        d->heatmapCapacity = floats;
    }
    d->gridDirty = true;
    update();
}

void PlotView::setPointCloudData(std::span<const float> data,
                                  int vertexCount,
                                  const PlotStyle& style) {
    d->vertices3d.resize(static_cast<std::size_t>(vertexCount) * 3);
    if (!d->has3D())
        d->interactionTool = InteractionTool::Rotate;
    for (int i = 0; i < vertexCount; ++i) {
        d->vertices3d[static_cast<std::size_t>(i) * 3]     = data[static_cast<std::size_t>(i)];
        d->vertices3d[static_cast<std::size_t>(i) * 3 + 1] = data[static_cast<std::size_t>(vertexCount + i)];
        d->vertices3d[static_cast<std::size_t>(i) * 3 + 2] = data[static_cast<std::size_t>(2 * vertexCount + i)];
    }
    d->vertex3dCount = vertexCount;

    d->bounds3d = BoundingBox3D();
    for (int i = 0; i < vertexCount; ++i) {
        for (int a = 0; a < 3; ++a) {
            float v = d->vertices3d[static_cast<std::size_t>(i) * 3 + static_cast<std::size_t>(a)];
            if (v < d->bounds3d.min[a]) d->bounds3d.min[a] = v;
            if (v > d->bounds3d.max[a]) d->bounds3d.max[a] = v;
        }
    }

    d->data3dColor = style.color.value_or(d->theme.seriesColors[0]);
    d->data3dPointSize = style.pointSize;
    d->mode3d = RenderMode3D::PointCloud;
    d->meshEdgeVertices.clear();
    d->meshEdgeVertexCount = 0;
    d->guide3dVertices = computeGuide3DVertices(d->bounds3d, d->camera);
    d->guide3dVertexCount = static_cast<int>(d->guide3dVertices.size() / 3);
    d->xTickValues3d = computeTicks(d->bounds3d.min.x(), d->bounds3d.max.x(), 5).ticks;
    d->yTickValues3d = computeTicks(d->bounds3d.min.y(), d->bounds3d.max.y(), 5).ticks;
    d->zTickValues3d = computeTicks(d->bounds3d.min.z(), d->bounds3d.max.z(), 5).ticks;
    d->data3dDirty = true;
    d->guide3dDirty = true;
    update();
}

void PlotView::setMeshData(std::span<const float> verts, int vertexCount,
                            std::span<const uint32_t> idx, int triangleCount,
                            const PlotStyle& style) {
    std::vector<float> rowMajorVerts(static_cast<std::size_t>(vertexCount) * 3);
    if (!d->has3D())
        d->interactionTool = InteractionTool::Rotate;
    for (int i = 0; i < vertexCount; ++i) {
        rowMajorVerts[static_cast<std::size_t>(i) * 3]     = verts[static_cast<std::size_t>(i)];
        rowMajorVerts[static_cast<std::size_t>(i) * 3 + 1] = verts[static_cast<std::size_t>(vertexCount + i)];
        rowMajorVerts[static_cast<std::size_t>(i) * 3 + 2] = verts[static_cast<std::size_t>(2 * vertexCount + i)];
    }

    d->indices.resize(static_cast<std::size_t>(triangleCount) * 3);
    for (int t = 0; t < triangleCount; ++t) {
        d->indices[static_cast<std::size_t>(t) * 3]     = idx[static_cast<std::size_t>(t)];
        d->indices[static_cast<std::size_t>(t) * 3 + 1] = idx[static_cast<std::size_t>(triangleCount + t)];
        d->indices[static_cast<std::size_t>(t) * 3 + 2] = idx[static_cast<std::size_t>(2 * triangleCount + t)];
    }
    d->indexCount = triangleCount * 3;

    d->meshEdgeVertices = computeSharpEdgeVertices(rowMajorVerts,
                                                   d->indices,
                                                   triangleCount);
    d->meshEdgeVertexCount = static_cast<int>(d->meshEdgeVertices.size() / 3);
    d->vertices3d = computeVertexNormals(rowMajorVerts, vertexCount,
                                         d->indices, triangleCount);
    d->vertex3dCount = vertexCount;

    d->bounds3d = BoundingBox3D();
    for (int i = 0; i < vertexCount; ++i) {
        for (int a = 0; a < 3; ++a) {
            float v = rowMajorVerts[static_cast<std::size_t>(i) * 3 + static_cast<std::size_t>(a)];
            if (v < d->bounds3d.min[a]) d->bounds3d.min[a] = v;
            if (v > d->bounds3d.max[a]) d->bounds3d.max[a] = v;
        }
    }

    d->data3dColor = style.color.value_or(d->theme.seriesColors[0]);
    d->mode3d = RenderMode3D::Mesh;
    d->guide3dVertices = computeGuide3DVertices(d->bounds3d, d->camera);
    d->guide3dVertexCount = static_cast<int>(d->guide3dVertices.size() / 3);
    d->xTickValues3d = computeTicks(d->bounds3d.min.x(), d->bounds3d.max.x(), 5).ticks;
    d->yTickValues3d = computeTicks(d->bounds3d.min.y(), d->bounds3d.max.y(), 5).ticks;
    d->zTickValues3d = computeTicks(d->bounds3d.min.z(), d->bounds3d.max.z(), 5).ticks;
    d->data3dDirty = true;
    d->indexDirty = true;
    d->meshEdgeDirty = true;
    d->guide3dDirty = true;
    update();
}

// ── Scene management ───────────────────────────────────────────────

void PlotView::clear() {
    for (auto& s : d->series2d) {
        delete s.vb;
        delete s.ub;
        delete s.srb;
    }
    d->series2d.clear();
    d->nextColorIndex = 0;
    d->telemetrySeries.reset();
    d->telemetryRing.clear();
    d->telemetryHead = 0;
    d->telemetrySize = 0;
    d->telemetryNextX = 0.0f;
    d->mode3d = RenderMode3D::None;
    d->hasHeatmap = false;
    d->heatmapVertices.clear();
    d->heatmapVertexCount = 0;
    d->heatmapBuffer.reset();
    d->hasColormappedData = false;
    d->aspectEqual = false;
    d->hasContour = false;
    d->contourVertices.clear();
    d->contourVertexCount = 0;
    d->contourBuffer.reset();
    d->meshEdgeVertices.clear();
    d->meshEdgeVertexCount = 0;
    d->guide3dVertices.clear();
    d->guide3dVertexCount = 0;
    d->xTickValues3d.clear();
    d->yTickValues3d.clear();
    d->zTickValues3d.clear();
    d->bounds2d = BoundingBox2D();
    d->viewBounds = BoundingBox2D();
    d->userViewBounds2d = false;
    d->xTickValues.clear();
    d->yTickValues.clear();
    d->gridDirty = true;
    update();
}

void PlotView::setTitle(const QString& title) {
    d->title = title;
    d->titleLabel->hide();
    d->textOverlay->update();
    update();
}

void PlotView::setCaption(const QString& caption) {
    d->caption = caption;
    d->textOverlay->update();
    update();
}

// ── Appearance ─────────────────────────────────────────────────────

void PlotView::setBackgroundColor(const Eigen::Vector4f& rgba) {
    d->userBgColor = rgba;
    update();
}

void PlotView::setPointSize(float size) {
    d->defaultPointSize = size;
    update();
}

void PlotView::setTheme(const Theme& theme) {
    d->theme = theme;
    d->gridDirty = true;
    if (d->interactionTool == InteractionTool::Rotate && !d->has3D())
        d->interactionTool = InteractionTool::Pan;

    bool isDark = theme.background.x() < 0.5f;
    d->overlay->updateThemeColors(isDark);

    if (d->titleLabel->isVisible()) {
        QPalette pal = d->titleLabel->palette();
        auto tc = d->theme.textColor;
        pal.setColor(QPalette::WindowText,
                     QColor::fromRgbF(tc.x(), tc.y(), tc.z(), tc.w()));
        d->titleLabel->setPalette(pal);
    }

    d->textOverlay->update();
    update();
}

auto PlotView::theme() const -> const Theme& {
    return d->theme;
}

void PlotView::setPalette(Palette palette) {
    setTheme(d->theme.withPalette(palette));
}

void PlotView::setGridVisible(bool visible) {
    d->showGrid = visible;
    update();
}

void PlotView::setAxesVisible(bool visible) {
    d->showAxes = visible;
    update();
}

void PlotView::setAxisArrowsVisible(bool visible) {
    d->showAxisArrows = visible;
    d->gridDirty = true;
    update();
}

void PlotView::setLegendVisible(bool visible) {
    d->showLegend = visible;
    if (d->textOverlay) d->textOverlay->update();
    update();
}

auto PlotView::legendVisible() const -> bool {
    return d->showLegend;
}

void PlotView::setLegendPosition(LegendPosition position) {
    d->legendPosition = position;
    if (d->textOverlay) d->textOverlay->update();
    update();
}

auto PlotView::legendPosition() const -> LegendPosition {
    return d->legendPosition;
}

void PlotView::setColorbarVisible(bool visible) {
    d->showColorbar = visible;
    if (d->textOverlay) d->textOverlay->update();
    update();
}

void PlotView::setAspectEqual(bool equal) {
    d->aspectEqual = equal;
    d->gridDirty = true;
    update();
}

void PlotView::setAxisLabels(const QString& xLabel, const QString& yLabel) {
    d->xAxisLabel = xLabel;
    d->yAxisLabel = yLabel;
    d->textOverlay->update();
    update();
}

void PlotView::setAxisLabels(const QString& xLabel,
                             const QString& yLabel,
                             const QString& zLabel) {
    d->xAxisLabel = xLabel;
    d->yAxisLabel = yLabel;
    d->zAxisLabel = zLabel;
    d->textOverlay->update();
    update();
}

void PlotView::setXAxisLabel(const QString& label) {
    d->xAxisLabel = label;
    d->textOverlay->update();
    update();
}

void PlotView::setYAxisLabel(const QString& label) {
    d->yAxisLabel = label;
    d->textOverlay->update();
    update();
}

void PlotView::setZAxisLabel(const QString& label) {
    d->zAxisLabel = label;
    d->textOverlay->update();
    update();
}

void PlotView::setCamera(const Camera3D& camera) {
    d->camera = camera;
    d->homeCamera = camera;
    if (d->has3D()) {
        d->guide3dVertices = computeGuide3DVertices(d->bounds3d, d->camera);
        d->guide3dVertexCount = static_cast<int>(d->guide3dVertices.size() / 3);
        d->guide3dDirty = true;
    }
    update();
}

auto PlotView::camera() const -> const Camera3D& {
    return d->camera;
}

void PlotView::setOverlayVisible(bool visible) {
    d->overlayEnabled = visible;
    if (!visible)
        d->overlay->hide();
}

void PlotView::setInteractionTool(InteractionTool tool) {
    if (tool == InteractionTool::Rotate && !d->has3D())
        tool = InteractionTool::Pan;
    d->interactionTool = tool;
}

auto PlotView::interactionTool() const -> InteractionTool {
    return d->interactionTool;
}

void PlotView::resetCameraView() {
    if (d->has2D() && !d->has3D()) {
        d->userViewBounds2d = false;
        d->gridDirty = true;
        update();
        return;
    }
    d->camera = d->homeCamera;
    if (d->has3D()) {
        d->guide3dVertices = computeGuide3DVertices(d->bounds3d, d->camera);
        d->guide3dVertexCount = static_cast<int>(d->guide3dVertices.size() / 3);
        d->guide3dDirty = true;
    }
    update();
}

void PlotView::zoomCamera(float factor) {
    if (d->has2D() && !d->has3D()) {
        if (!d->userViewBounds2d) {
            computeGridVertices();
            d->userViewBounds2d = true;
        }

        factor = std::clamp(factor, 0.1f, 10.0f);
        Eigen::Vector2f center = d->viewBounds.center();
        Eigen::Vector2f halfSize(d->viewBounds.width() * factor * 0.5f,
                                 d->viewBounds.height() * factor * 0.5f);
        d->viewBounds.min = center - halfSize;
        d->viewBounds.max = center + halfSize;
        d->gridDirty = true;
        update();
        return;
    }

    if (!d->has3D())
        return;

    Eigen::Vector3f target = d->camera.target();
    Eigen::Vector3f eye = d->camera.position();
    Eigen::Vector3f offset = eye - target;
    float minDistance = std::max(0.05f, d->bounds3d.diagonal() * 0.04f);
    float distance = std::max(minDistance, offset.norm() * factor);
    if (offset.squaredNorm() < 1e-8f)
        offset = Eigen::Vector3f(0.f, 0.f, 1.f);
    offset.normalize();
    d->camera.lookAt(target + offset * distance, target);
    d->guide3dVertices = computeGuide3DVertices(d->bounds3d, d->camera);
    d->guide3dVertexCount = static_cast<int>(d->guide3dVertices.size() / 3);
    d->guide3dDirty = true;
    update();
}

auto PlotView::is3DView() const -> bool {
    return d->has3D();
}

auto PlotView::is2DView() const -> bool {
    return d->has2D() && !d->has3D();
}

// ── Bounds ─────────────────────────────────────────────────────────

void PlotView::recomputeBounds() {
    d->bounds2d = BoundingBox2D();
    for (const auto& s : d->series2d) {
        if (!s.visible || s.vertexCount == 0) continue;
        BoundingBox2D sb;
        for (int i = 0; i < s.vertexCount; ++i) {
            float x = s.vertices[static_cast<std::size_t>(i) * 2];
            float y = s.vertices[static_cast<std::size_t>(i) * 2 + 1];
            if (x < sb.min.x()) sb.min.x() = x;
            if (x > sb.max.x()) sb.max.x() = x;
            if (y < sb.min.y()) sb.min.y() = y;
            if (y > sb.max.y()) sb.max.y() = y;
        }
        d->bounds2d = d->bounds2d.merge(sb);
    }
}

// ── Grid computation ───────────────────────────────────────────────

void PlotView::computeGridVertices() {
    d->gridVertices.clear();
    d->axisVertices.clear();
    d->xTickValues.clear();
    d->yTickValues.clear();

    BoundingBox2D sourceBounds = d->userViewBounds2d ? d->viewBounds : d->bounds2d;
    auto xTicks = computeTicks(sourceBounds.min.x(), sourceBounds.max.x());
    auto yTicks = computeTicks(sourceBounds.min.y(), sourceBounds.max.y());

    if (xTicks.ticks.empty() || yTicks.ticks.empty()) return;

    if (d->userViewBounds2d) {
        d->viewBounds = sourceBounds;
    } else {
        d->viewBounds.min = Eigen::Vector2f(xTicks.ticks.front(), yTicks.ticks.front());
        d->viewBounds.max = Eigen::Vector2f(xTicks.ticks.back(), yTicks.ticks.back());
    }

    float ylo = d->viewBounds.min.y();
    float yhi = d->viewBounds.max.y();
    float xlo = d->viewBounds.min.x();
    float xhi = d->viewBounds.max.x();

    auto visibleTicks = [](const std::vector<float>& ticks, float lo, float hi) {
        std::vector<float> result;
        result.reserve(ticks.size());
        float eps = std::max(1e-6f, (hi - lo) * 1e-5f);
        for (float tick : ticks) {
            if (tick >= lo - eps && tick <= hi + eps)
                result.push_back(tick);
        }
        return result;
    };

    d->xTickValues = d->userViewBounds2d ? visibleTicks(xTicks.ticks, xlo, xhi) : xTicks.ticks;
    d->yTickValues = d->userViewBounds2d ? visibleTicks(yTicks.ticks, ylo, yhi) : yTicks.ticks;

    for (float xt : d->xTickValues) {
        d->gridVertices.push_back(xt); d->gridVertices.push_back(ylo);
        d->gridVertices.push_back(xt); d->gridVertices.push_back(yhi);
    }
    for (float yt : d->yTickValues) {
        d->gridVertices.push_back(xlo); d->gridVertices.push_back(yt);
        d->gridVertices.push_back(xhi); d->gridVertices.push_back(yt);
    }
    d->gridVertexCount = static_cast<int>(d->gridVertices.size()) / 2;

    d->axisVertices.push_back(xlo); d->axisVertices.push_back(ylo);
    d->axisVertices.push_back(xhi); d->axisVertices.push_back(ylo);
    d->axisVertices.push_back(xlo); d->axisVertices.push_back(ylo);
    d->axisVertices.push_back(xlo); d->axisVertices.push_back(yhi);

    if (d->showAxisArrows) {
        auto appendLine2D = [&](float ax, float ay, float bx, float by) {
            d->axisVertices.push_back(ax); d->axisVertices.push_back(ay);
            d->axisVertices.push_back(bx); d->axisVertices.push_back(by);
        };

        // Arrowheads must look identical across both axes, i.e. share the
        // same length and apex angle in *screen* (pixel) space. The data
        // ranges and the widget aspect differ per axis, so convert a fixed
        // pixel length/half-width into data units separately for each axis.
        // Data->pixel scale: 1 data-unit-x -> (W / xRange) px (the ortho
        // projection maps the data range to NDC [-1,1], NDC maps to the
        // viewport), and likewise 1 data-unit-y -> (H / yRange) px.
        const float xRange = std::max(1e-12f, xhi - xlo);
        const float yRange = std::max(1e-12f, yhi - ylo);
        const float dataPerPxX = xRange / d->viewportW;
        const float dataPerPxY = yRange / d->viewportH;

        constexpr float kArrowLenPx = 13.0f;       // tip-to-base length
        constexpr float kArrowHalfWidthPx = 5.0f;  // half of the base width

        // Solid (filled) arrowheads: the grid/axis pipeline draws line
        // segments only, so fill each triangular head with a fan of
        // closely-spaced segments sweeping from the tip across the base.
        constexpr int kArrowFanSteps = 14;
        auto fillArrow2D = [&](float tipX, float tipY,
                               float baseAX, float baseAY,
                               float baseBX, float baseBY) {
            for (int i = 0; i <= kArrowFanSteps; ++i) {
                float s = static_cast<float>(i) / static_cast<float>(kArrowFanSteps);
                float px = baseAX + (baseBX - baseAX) * s;
                float py = baseAY + (baseBY - baseAY) * s;
                appendLine2D(tipX, tipY, px, py);
            }
        };

        // X-axis arrow (points +x): length along x, half-width along y.
        const float xHeadLen = kArrowLenPx * dataPerPxX;
        const float xHeadHalf = kArrowHalfWidthPx * dataPerPxY;
        fillArrow2D(xhi, ylo,
                    xhi - xHeadLen, ylo - xHeadHalf,
                    xhi - xHeadLen, ylo + xHeadHalf);

        // Y-axis arrow (points +y): length along y, half-width along x.
        const float yHeadLen = kArrowLenPx * dataPerPxY;
        const float yHeadHalf = kArrowHalfWidthPx * dataPerPxX;
        fillArrow2D(xlo, yhi,
                    xlo - yHeadHalf, yhi - yHeadLen,
                    xlo + yHeadHalf, yhi - yHeadLen);
    }

    float txLen = (yhi - ylo) * 0.010f;
    float tyLen = (xhi - xlo) * 0.010f;
    for (float xt : d->xTickValues) {
        d->axisVertices.push_back(xt); d->axisVertices.push_back(ylo);
        d->axisVertices.push_back(xt); d->axisVertices.push_back(ylo + txLen);
    }
    for (float yt : d->yTickValues) {
        d->axisVertices.push_back(xlo); d->axisVertices.push_back(yt);
        d->axisVertices.push_back(xlo + tyLen); d->axisVertices.push_back(yt);
    }
    d->axisVertexCount = static_cast<int>(d->axisVertices.size()) / 2;
    d->gridDirty = false;
}

// ── Layout ─────────────────────────────────────────────────────────

void PlotView::layoutChildren() {
    if (d->textOverlay)
        d->textOverlay->setGeometry(rect());

    if (d->titleLabel && d->titleLabel->isVisible()) {
        d->titleLabel->setGeometry(0, 8, width(), 30);
    }
    if (d->overlay) {
        int ox = width() - d->overlay->width() - 12;
        int oy = height() - d->overlay->height() - 12;
        d->overlay->move(ox, oy);
    }
}

static auto colorFromVec(const Eigen::Vector4f& rgba, float alphaScale = 1.0f) -> QColor {
    return QColor::fromRgbF(rgba.x(), rgba.y(), rgba.z(),
                            std::clamp(rgba.w() * alphaScale, 0.0f, 1.0f));
}

static auto tickLabel(float value) -> QString {
    if (std::abs(value) < 1e-6f)
        value = 0.f;
    return QString::number(value, 'g', 4);
}

static auto plotAreaFor(const QSize& size,
                        bool hasTitle,
                        bool hasCaption,
                        bool hasXAxisLabel,
                        bool hasYAxisLabel,
                        bool hasColorbar = false) -> QRectF
{
    double left = hasYAxisLabel ? 68.0 : 46.0;
    double top = hasTitle ? (hasCaption ? 58.0 : 38.0) : (hasCaption ? 38.0 : 18.0);
    double right = hasColorbar ? 80.0 : 24.0;
    double bottom = hasXAxisLabel ? 50.0 : 34.0;
    return QRectF(left,
                  top,
                  std::max(20.0, static_cast<double>(size.width()) - left - right),
                  std::max(20.0, static_cast<double>(size.height()) - top - bottom));
}

static auto dataToPixel(const BoundingBox2D& bounds,
                        const QRectF& plotArea,
                        const Eigen::Vector2f& point) -> QPointF
{
    auto b = bounds.expanded(0.02f);
    float x = (point.x() - b.min.x()) / b.width();
    float y = (point.y() - b.min.y()) / b.height();
    return QPointF(plotArea.left() + x * plotArea.width(),
                   plotArea.top() + (1.0f - y) * plotArea.height());
}

static auto project3D(const Eigen::Matrix4f& mvp,
                      const QSize& size,
                      const Eigen::Vector3f& point) -> std::optional<QPointF>
{
    Eigen::Vector4f clip = mvp * Eigen::Vector4f(point.x(), point.y(), point.z(), 1.0f);
    if (std::abs(clip.w()) < 1e-6f)
        return std::nullopt;

    Eigen::Vector3f ndc = clip.head<3>() / clip.w();
    if (ndc.z() < -1.2f || ndc.z() > 1.2f)
        return std::nullopt;

    return QPointF((ndc.x() * 0.5f + 0.5f) * static_cast<float>(size.width()),
                   (0.5f - ndc.y() * 0.5f) * static_cast<float>(size.height()));
}

void PlotView::paintTextOverlay(QPainter& painter, const QSize& size) const {
    if (size.isEmpty())
        return;

    auto textColor = colorFromVec(d->theme.textColor, 0.92f);
    auto mutedColor = colorFromVec(d->theme.textColor, 0.60f);
    auto axisColor = colorFromVec(d->theme.axisColor, 0.92f);

    painter.save();
    painter.setPen(textColor);

    if (!d->title.isEmpty()) {
        QFont titleFont = painter.font();
        titleFont.setPointSize(13);
        titleFont.setWeight(QFont::DemiBold);
        painter.setFont(titleFont);
        QRect titleRect(24, 9, std::max(0, size.width() - 48), 24);
        painter.drawText(titleRect, Qt::AlignHCenter | Qt::AlignVCenter, d->title);
    }

    if (!d->caption.isEmpty()) {
        QFont captionFont = painter.font();
        captionFont.setPointSize(10);
        captionFont.setWeight(QFont::Normal);
        painter.setFont(captionFont);
        painter.setPen(mutedColor);
        QRect captionRect(32, d->title.isEmpty() ? 10 : 32,
                          std::max(0, size.width() - 64), 20);
        painter.drawText(captionRect, Qt::AlignHCenter | Qt::AlignVCenter, d->caption);
    }

    bool is2D = d->has2D() && !d->has3D() && d->showAxes
        && !d->xTickValues.empty() && !d->yTickValues.empty();
    if (is2D) {
        QRectF plotArea = plotAreaFor(size,
                                      !d->title.isEmpty(),
                                      !d->caption.isEmpty(),
                                      !d->xAxisLabel.isEmpty(),
                                      !d->yAxisLabel.isEmpty(),
                                      d->showColorbar && d->hasColormappedData);

        QFont tickFont = painter.font();
        tickFont.setPointSize(9);
        tickFont.setWeight(QFont::Normal);
        painter.setFont(tickFont);
        QFontMetrics tickMetrics(tickFont);
        painter.setPen(mutedColor);
        double xTickY = plotArea.bottom() + 4.0;

        for (std::size_t i = 0; i < d->xTickValues.size(); ++i) {
            if (i == 0 || i + 1 == d->xTickValues.size())
                continue;
            float xt = d->xTickValues[i];
            QPointF pos = dataToPixel(d->viewBounds, plotArea,
                                      Eigen::Vector2f(xt, d->viewBounds.min.y()));
            QString label = tickLabel(xt);
            QRectF rect(pos.x() - 32.0, xTickY, 64.0, 16.0);
            painter.drawText(rect, Qt::AlignHCenter | Qt::AlignVCenter, label);
        }

        double yTickX = plotArea.left() - 42.0;
        for (std::size_t i = 0; i < d->yTickValues.size(); ++i) {
            if (i == 0 || i + 1 == d->yTickValues.size())
                continue;
            float yt = d->yTickValues[i];
            QPointF pos = dataToPixel(d->viewBounds, plotArea,
                                      Eigen::Vector2f(d->viewBounds.min.x(), yt));
            QString label = tickLabel(yt);
            QRectF rect(yTickX, pos.y() - tickMetrics.height() * 0.5,
                        34.0, tickMetrics.height() + 2.0);
            painter.drawText(rect, Qt::AlignRight | Qt::AlignVCenter, label);
        }

        QFont labelFont = painter.font();
        labelFont.setPointSize(10);
        labelFont.setWeight(QFont::DemiBold);
        painter.setFont(labelFont);
        painter.setPen(axisColor);

        if (!d->xAxisLabel.isEmpty()) {
            QRectF rect(plotArea.left(), size.height() - 22.0,
                        plotArea.width(), 16.0);
            painter.drawText(rect, Qt::AlignHCenter | Qt::AlignVCenter, d->xAxisLabel);
        }

        if (!d->yAxisLabel.isEmpty()) {
            painter.save();
            painter.translate(16.0, plotArea.center().y());
            painter.rotate(-90.0);
            QRectF rect(-plotArea.height() * 0.5, -8.0, plotArea.height(), 16.0);
            painter.drawText(rect, Qt::AlignHCenter | Qt::AlignVCenter, d->yAxisLabel);
            painter.restore();
        }

        // ── Colorbar legend (imshow / contourf) ────────────────────
        if (d->showColorbar && d->hasColormappedData) {
            const double barW = 14.0;
            const double barX = plotArea.right() + 18.0;
            const double barTop = plotArea.top();
            const double barH = plotArea.height();

            // Colormapped gradient strip (top = vmax, bottom = vmin).
            const int steps = 128;
            for (int i = 0; i < steps; ++i) {
                float t = 1.0f - (static_cast<float>(i) + 0.5f) / static_cast<float>(steps);
                Eigen::Vector4f col = sampleColormap(d->colorbarMap, t);
                double y = barTop + barH * static_cast<double>(i) / steps;
                double h = barH / steps + 1.0;
                painter.fillRect(QRectF(barX, y, barW, h),
                                 colorFromVec(col));
            }
            // Strip border.
            painter.setPen(QPen(mutedColor, 1.0));
            painter.setBrush(Qt::NoBrush);
            painter.drawRect(QRectF(barX, barTop, barW, barH));

            // Min / mid / max tick labels.
            QFont cbFont = painter.font();
            cbFont.setPointSize(8);
            cbFont.setWeight(QFont::Normal);
            painter.setFont(cbFont);
            painter.setPen(mutedColor);
            const double lx = barX + barW + 4.0;
            auto cbLabel = [&](double frac, float value) {
                double y = barTop + barH * (1.0 - frac);
                painter.drawText(QRectF(lx, y - 8.0, 44.0, 16.0),
                                 Qt::AlignLeft | Qt::AlignVCenter, tickLabel(value));
            };
            cbLabel(0.0, d->colorbarVmin);
            cbLabel(0.5, 0.5f * (d->colorbarVmin + d->colorbarVmax));
            cbLabel(1.0, d->colorbarVmax);
        }
    }

    if (d->showLegend && d->has2D() && !d->has3D()) {
        std::vector<const Series2D*> entries;
        for (const auto& series : d->series2d) {
            if (series.visible && !series.label.isEmpty())
                entries.push_back(&series);
        }

        if (!entries.empty()) {
            const QRectF plotArea = plotAreaFor(
                size,
                !d->title.isEmpty(),
                !d->caption.isEmpty(),
                !d->xAxisLabel.isEmpty(),
                !d->yAxisLabel.isEmpty(),
                d->showColorbar && d->hasColormappedData);

            QFont legendFont = painter.font();
            legendFont.setPointSize(9);
            legendFont.setWeight(QFont::Normal);
            painter.setFont(legendFont);
            const QFontMetrics metrics(legendFont);

            constexpr double padding = 8.0;
            constexpr double sampleWidth = 24.0;
            constexpr double gap = 7.0;
            const double rowHeight = std::max(18.0, metrics.height() + 4.0);
            double labelWidth = 0.0;
            for (const auto* entry : entries)
                labelWidth = std::max(labelWidth,
                                      static_cast<double>(metrics.horizontalAdvance(entry->label)));
            const QSizeF legendSize(
                padding * 2.0 + sampleWidth + gap + labelWidth,
                padding * 2.0 + rowHeight * static_cast<double>(entries.size()));

            LegendPosition position = d->legendPosition;
            if (position == LegendPosition::Auto) {
                std::array<std::size_t, 4> occupancy{};
                const Eigen::Vector2f center = d->viewBounds.center();
                for (const auto& series : d->series2d) {
                    if (!series.visible)
                        continue;
                    for (int i = 0; i < series.vertexCount; ++i) {
                        const float x = series.vertices[static_cast<std::size_t>(i) * 2];
                        const float y = series.vertices[static_cast<std::size_t>(i) * 2 + 1];
                        const bool right = x >= center.x();
                        const bool bottom = y < center.y();
                        const std::size_t quadrant = bottom
                            ? (right ? 3u : 2u)
                            : (right ? 1u : 0u);
                        ++occupancy[quadrant];
                    }
                }
                const auto leastOccupied = static_cast<std::size_t>(
                    std::distance(occupancy.begin(),
                                  std::min_element(occupancy.begin(), occupancy.end())));
                constexpr std::array positions{
                    LegendPosition::UpperLeft,
                    LegendPosition::UpperRight,
                    LegendPosition::LowerLeft,
                    LegendPosition::LowerRight
                };
                position = positions[leastOccupied];
            }

            constexpr double inset = 10.0;
            const bool right = position == LegendPosition::UpperRight
                || position == LegendPosition::LowerRight;
            const bool bottom = position == LegendPosition::LowerLeft
                || position == LegendPosition::LowerRight;
            const QPointF topLeft(
                right ? plotArea.right() - legendSize.width() - inset
                      : plotArea.left() + inset,
                bottom ? plotArea.bottom() - legendSize.height() - inset
                       : plotArea.top() + inset);
            const QRectF legendRect(topLeft, legendSize);

            QColor background = colorFromVec(d->theme.background);
            background.setAlphaF(0.92);
            painter.setPen(QPen(mutedColor, 1.0));
            painter.setBrush(background);
            painter.drawRoundedRect(legendRect, 4.0, 4.0);

            for (std::size_t i = 0; i < entries.size(); ++i) {
                const auto& entry = *entries[i];
                const double rowCenter = legendRect.top() + padding
                    + rowHeight * (static_cast<double>(i) + 0.5);
                const double sampleLeft = legendRect.left() + padding;
                painter.setPen(QPen(colorFromVec(entry.color), 2.0));
                painter.setBrush(colorFromVec(entry.color));
                if (entry.kind == SeriesKind::Scatter) {
                    painter.drawEllipse(
                        QPointF(sampleLeft + sampleWidth * 0.5, rowCenter),
                        3.5, 3.5);
                } else if (entry.kind == SeriesKind::Fill) {
                    painter.drawRect(QRectF(sampleLeft, rowCenter - 3.0,
                                            sampleWidth, 6.0));
                } else {
                    painter.drawLine(QPointF(sampleLeft, rowCenter),
                                     QPointF(sampleLeft + sampleWidth, rowCenter));
                }
                painter.setPen(textColor);
                painter.setBrush(Qt::NoBrush);
                painter.drawText(
                    QRectF(sampleLeft + sampleWidth + gap,
                           rowCenter - rowHeight * 0.5,
                           labelWidth, rowHeight),
                    Qt::AlignLeft | Qt::AlignVCenter,
                    entry.label);
            }
        }
    }

    if (d->has3D() && d->showAxes) {
        auto b = d->bounds3d.expanded(0.04f);
        Eigen::Vector3f lo = b.min;
        Eigen::Vector3f hi = b.max;
        Eigen::Vector3f center = b.center();
        Eigen::Vector3f eye = d->camera.position();
        float backX = eye.x() >= center.x() ? lo.x() : hi.x();
        float backY = eye.y() >= center.y() ? lo.y() : hi.y();
        float backZ = eye.z() >= center.z() ? lo.z() : hi.z();
        float frontX = eye.x() >= center.x() ? hi.x() : lo.x();
        float frontZ = eye.z() >= center.z() ? hi.z() : lo.z();
        float lowerY = eye.y() >= center.y() ? lo.y() : hi.y();
        float xTip = eye.x() >= center.x()
            ? hi.x() + (hi.x() - lo.x()) * 0.08f
            : lo.x() - (hi.x() - lo.x()) * 0.08f;
        float yTip = eye.y() >= center.y()
            ? hi.y() + (hi.y() - lo.y()) * 0.08f
            : lo.y() - (hi.y() - lo.y()) * 0.08f;
        float zTip = eye.z() >= center.z()
            ? hi.z() + (hi.z() - lo.z()) * 0.08f
            : lo.z() - (hi.z() - lo.z()) * 0.08f;
        Eigen::Matrix4f mvp = d->camera.viewProjectionMatrix();

        QRectF occupiedMeshRect;
        bool hasOccupiedMeshRect = false;
        for (float x : {lo.x(), hi.x()}) {
            for (float y : {lo.y(), hi.y()}) {
                for (float z : {lo.z(), hi.z()}) {
                    auto projected = project3D(mvp, size, Eigen::Vector3f(x, y, z));
                    if (!projected)
                        continue;
                    QRectF pointRect(projected->x(), projected->y(), 1.0, 1.0);
                    occupiedMeshRect = hasOccupiedMeshRect
                        ? occupiedMeshRect.united(pointRect)
                        : pointRect;
                    hasOccupiedMeshRect = true;
                }
            }
        }
        occupiedMeshRect = occupiedMeshRect.adjusted(-10.0, -10.0, 10.0, 10.0);
        QPointF screenCenter(size.width() * 0.5, size.height() * 0.5);

        QFont labelFont = painter.font();
        labelFont.setPointSize(10);
        labelFont.setWeight(QFont::DemiBold);
        QFont tickFont = painter.font();
        tickFont.setPointSize(8);
        tickFont.setWeight(QFont::Normal);

        auto placeOutsideMesh = [&](QPointF pos, const QSizeF& textSize) {
            QPointF outward = pos - screenCenter;
            double length = std::hypot(outward.x(), outward.y());
            if (length <= 1.0) {
                outward = QPointF(0.0, 1.0);
                length = 1.0;
            }
            QPointF direction = outward / length;
            pos += direction * 30.0;

            auto rectFor = [&](const QPointF& centerPoint) {
                return QRectF(centerPoint.x() - textSize.width() * 0.5,
                              centerPoint.y() - textSize.height() * 0.5,
                              textSize.width(),
                              textSize.height());
            };
            for (int i = 0; hasOccupiedMeshRect && i < 10; ++i) {
                if (!rectFor(pos).intersects(occupiedMeshRect))
                    break;
                pos += direction * 14.0;
            }

            pos.setX(std::clamp(pos.x(),
                                textSize.width() * 0.5 + 8.0,
                                static_cast<double>(size.width()) - textSize.width() * 0.5 - 8.0));
            pos.setY(std::clamp(pos.y(),
                                textSize.height() * 0.5 + 8.0,
                                static_cast<double>(size.height()) - textSize.height() * 0.5 - 8.0));
            return pos;
        };

        auto drawTick = [&](float value, const Eigen::Vector3f& point, QPointF offset) {
            auto projected = project3D(mvp, size, point);
            if (!projected)
                return;
            painter.setFont(tickFont);
            painter.setPen(mutedColor);
            QString label = tickLabel(value);
            QFontMetrics metrics(tickFont);
            QSizeF textSize(std::max(38, metrics.horizontalAdvance(label) + 12),
                            metrics.height() + 4);
            QPointF pos = placeOutsideMesh(*projected + offset, textSize);
            QRectF rect(pos.x() - textSize.width() * 0.5,
                        pos.y() - textSize.height() * 0.5,
                        textSize.width(),
                        textSize.height());
            painter.drawText(rect, Qt::AlignHCenter | Qt::AlignVCenter, label);
        };

        auto interiorTick = [](const std::vector<float>& ticks, std::size_t i) {
            return i > 0 && i + 1 < ticks.size();
        };

        for (std::size_t i = 0; i < d->xTickValues3d.size(); ++i) {
            if (!interiorTick(d->xTickValues3d, i))
                continue;
            float xt = d->xTickValues3d[i];
            drawTick(xt, Eigen::Vector3f(xt, lowerY, frontZ), QPointF(0.0, 18.0));
        }
        for (std::size_t i = 0; i < d->yTickValues3d.size(); ++i) {
            if (!interiorTick(d->yTickValues3d, i))
                continue;
            float yt = d->yTickValues3d[i];
            drawTick(yt, Eigen::Vector3f(frontX, yt, backZ), QPointF(28.0, -2.0));
        }
        for (std::size_t i = 0; i < d->zTickValues3d.size(); ++i) {
            if (!interiorTick(d->zTickValues3d, i))
                continue;
            float zt = d->zTickValues3d[i];
            drawTick(zt, Eigen::Vector3f(backX, backY, zt), QPointF(-24.0, 14.0));
        }

        auto drawLabel = [&](const QString& text, const Eigen::Vector3f& point) {
            if (text.isEmpty())
                return;
            auto projected = project3D(mvp, size, point);
            if (!projected)
                return;
            painter.setFont(labelFont);
            painter.setPen(axisColor);
            QFontMetrics metrics(labelFont);
            QSizeF textSize(std::max(56, metrics.horizontalAdvance(text) + 18),
                            metrics.height() + 4);
            QPointF pos = placeOutsideMesh(*projected, textSize);
            QRectF rect(pos.x() - textSize.width() * 0.5,
                        pos.y() - textSize.height() * 0.5,
                        textSize.width(),
                        textSize.height());
            painter.drawText(rect, Qt::AlignHCenter | Qt::AlignVCenter, text);
        };

        drawLabel(d->xAxisLabel, Eigen::Vector3f(xTip, backY, backZ));
        drawLabel(d->yAxisLabel, Eigen::Vector3f(backX, yTip, backZ));
        drawLabel(d->zAxisLabel, Eigen::Vector3f(backX, backY, zTip));
    }

    painter.restore();
}

void PlotView::resizeEvent(QResizeEvent* event) {
    QRhiWidget::resizeEvent(event);
    layoutChildren();
}

void PlotView::mouseMoveEvent(QMouseEvent* event) {
    if (d->overlayEnabled)
        d->overlay->showTemporarily();

    if (d->dragging && d->has3D()) {
        QPoint delta = event->pos() - d->lastMousePos;
        d->lastMousePos = event->pos();

        Eigen::Vector3f eye = d->camera.position();
        Eigen::Vector3f target = d->camera.target();
        Eigen::Vector3f offset = eye - target;
        float distance = std::max(offset.norm(), 1e-4f);
        Eigen::Vector3f forward = (target - eye).normalized();
        Eigen::Vector3f right = forward.cross(Eigen::Vector3f::UnitY());
        if (right.squaredNorm() < 1e-8f)
            right = Eigen::Vector3f::UnitX();
        right.normalize();
        Eigen::Vector3f up = right.cross(forward).normalized();

        if (d->interactionTool == InteractionTool::Rotate) {
            float yaw = -static_cast<float>(delta.x()) * 0.008f;
            float pitch = -static_cast<float>(delta.y()) * 0.008f;
            Eigen::AngleAxisf yawRot(yaw, Eigen::Vector3f::UnitY());
            Eigen::AngleAxisf pitchRot(pitch, right);
            Eigen::Vector3f rotated = yawRot * pitchRot * offset;
            d->camera.lookAt(target + rotated, target);
            d->guide3dVertices = computeGuide3DVertices(d->bounds3d, d->camera);
            d->guide3dVertexCount = static_cast<int>(d->guide3dVertices.size() / 3);
            d->guide3dDirty = true;
            update();
            return;
        }

        if (d->interactionTool == InteractionTool::Pan) {
            float scale = distance * 0.0018f;
            Eigen::Vector3f shift = (-right * static_cast<float>(delta.x())
                + up * static_cast<float>(delta.y())) * scale;
            d->camera.lookAt(eye + shift, target + shift);
            d->guide3dVertices = computeGuide3DVertices(d->bounds3d, d->camera);
            d->guide3dVertexCount = static_cast<int>(d->guide3dVertices.size() / 3);
            d->guide3dDirty = true;
            update();
            return;
        }

        if (d->interactionTool == InteractionTool::Zoom) {
            float factor = std::exp(static_cast<float>(delta.y()) * 0.01f);
            zoomCamera(factor);
            return;
        }
    }

    if (d->dragging && d->has2D() && !d->has3D()) {
        QPoint delta = event->pos() - d->lastMousePos;
        d->lastMousePos = event->pos();

        if (!d->userViewBounds2d) {
            computeGridVertices();
            d->userViewBounds2d = true;
        }

        if (d->interactionTool == InteractionTool::Pan) {
            QRectF plotArea = plotAreaFor(size(),
                                          !d->title.isEmpty(),
                                          !d->caption.isEmpty(),
                                          !d->xAxisLabel.isEmpty(),
                                          !d->yAxisLabel.isEmpty(),
                                          d->showColorbar && d->hasColormappedData);
            float dx = -static_cast<float>(delta.x())
                / static_cast<float>(std::max(1.0, plotArea.width())) * d->viewBounds.width();
            float dy = static_cast<float>(delta.y())
                / static_cast<float>(std::max(1.0, plotArea.height())) * d->viewBounds.height();
            Eigen::Vector2f shift(dx, dy);
            d->viewBounds.min += shift;
            d->viewBounds.max += shift;
            d->gridDirty = true;
            update();
            return;
        }

        if (d->interactionTool == InteractionTool::Zoom) {
            float factor = std::exp(static_cast<float>(delta.y()) * 0.01f);
            zoomCamera(factor);
            return;
        }
    }

    QRhiWidget::mouseMoveEvent(event);
}

void PlotView::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        d->dragging = true;
        d->lastMousePos = event->pos();
        if (d->overlayEnabled)
            d->overlay->showTemporarily();
        event->accept();
        return;
    }
    QRhiWidget::mousePressEvent(event);
}

void PlotView::mouseReleaseEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        d->dragging = false;
        event->accept();
        return;
    }
    QRhiWidget::mouseReleaseEvent(event);
}

void PlotView::wheelEvent(QWheelEvent* event) {
    if (d->hasData()) {
        float factor = std::pow(0.88f, static_cast<float>(event->angleDelta().y()) / 120.0f);
        zoomCamera(factor);
        if (d->overlayEnabled)
            d->overlay->showTemporarily();
        event->accept();
        return;
    }
    QRhiWidget::wheelEvent(event);
}

// ── QRhiWidget overrides ───────────────────────────────────────────

void PlotView::initialize(QRhiCommandBuffer* /*cb*/) {
    auto* r = rhi();
    d->pipelineReady = false;

    static constexpr int kInitialFloats = 128 * 1024;
    int sc = renderTarget()->sampleCount();

    // ── 3D vertex + index buffers ──────────────────────────────────
    d->vertex3dCapacity = kInitialFloats;
    d->vertex3dBuffer = makeDynBuf(r, QRhiBuffer::VertexBuffer,
                                    quint32(kInitialFloats * sizeof(float)));

    d->indexCapacity = kInitialFloats;
    d->indexBuffer = makeDynBuf(r, QRhiBuffer::IndexBuffer,
                                 quint32(kInitialFloats * sizeof(uint32_t)));

    d->meshEdgeCapacity = kInitialFloats;
    d->meshEdgeBuffer = makeDynBuf(r, QRhiBuffer::VertexBuffer,
                                    quint32(kInitialFloats * sizeof(float)));

    d->guide3dCapacity = kInitialFloats;
    d->guide3dBuffer = makeDynBuf(r, QRhiBuffer::VertexBuffer,
                                   quint32(kInitialFloats * sizeof(float)));

    // ── Grid vertex buffer ─────────────────────────────────────────
    d->gridVertexCapacity = kInitialFloats;
    d->gridVertexBuffer = makeDynBuf(r, QRhiBuffer::VertexBuffer,
                                      quint32(kInitialFloats * sizeof(float)));

    // ── Uniform buffers (grid, axis, 3D) ───────────────────────────
    d->point3dUniformBuffer = makeUB(r, 96);
    d->meshUniformBuffer    = makeUB(r, 112);
    d->meshEdgeUniformBuffer = makeUB(r, 80);
    d->guide3dUniformBuffer = makeUB(r, 80);
    d->gridUniformBuffer    = makeUB(r, 80);
    d->axisUniformBuffer    = makeUB(r, 80);
    d->heatmapUniformBuffer = makeUB(r, 80);
    d->contourUniformBuffer = makeUB(r, 80);

    // ── SRBs ───────────────────────────────────────────────────────
    d->point3dSrb = makeUniqueSrb(r, d->point3dUniformBuffer.get());
    d->heatmapSrb = makeUniqueSrb(r, d->heatmapUniformBuffer.get());
    d->contourSrb = makeUniqueSrb(r, d->contourUniformBuffer.get());
    d->meshSrb    = makeUniqueSrb(r, d->meshUniformBuffer.get());
    d->meshEdgeSrb = makeUniqueSrb(r, d->meshEdgeUniformBuffer.get());
    d->guide3dSrb = makeUniqueSrb(r, d->guide3dUniformBuffer.get());
    d->gridSrb    = makeUniqueSrb(r, d->gridUniformBuffer.get());
    d->axisSrb    = makeUniqueSrb(r, d->axisUniformBuffer.get());

    // ── Create per-series GPU resources ────────────────────────────
    for (auto& s : d->series2d) {
        quint32 ubSize = (s.kind == SeriesKind::Scatter) ? 96u : 80u;
        s.ub = makeRawUB(r, ubSize);
        s.srb = makeSrb(r, s.ub);
        int floats = static_cast<int>(s.vertices.size());
        s.vb = makeRawDynBuf(r, QRhiBuffer::VertexBuffer,
                              quint32(std::max(floats, 1) * sizeof(float)));
        s.vbCapacity = std::max(floats, 1);
        s.dirty = true;
    }
    if (d->hasHeatmap && !d->heatmapVertices.empty()) {
        int floats = static_cast<int>(d->heatmapVertices.size());
        d->heatmapBuffer = makeDynBuf(r, QRhiBuffer::VertexBuffer,
                                      quint32(floats * sizeof(float)));
        d->heatmapCapacity = floats;
        d->heatmapDirty = true;
    }
    if (d->hasContour && !d->contourVertices.empty()) {
        int floats = static_cast<int>(d->contourVertices.size());
        d->contourBuffer = makeDynBuf(r, QRhiBuffer::VertexBuffer,
                                      quint32(floats * sizeof(float)));
        d->contourCapacity = floats;
        d->contourDirty = true;
    }

    // ── Load shaders ───────────────────────────────────────────────
    auto lineVs  = loadShader(u":/skigen/plot/line2d.vert.qsb"_s);
    auto lineFs  = loadShader(u":/skigen/plot/line2d.frag.qsb"_s);
    auto fillVs  = loadShader(u":/skigen/plot/fill2d.vert.qsb"_s);
    auto fillFs  = loadShader(u":/skigen/plot/fill2d.frag.qsb"_s);
    auto pointVs = loadShader(u":/skigen/plot/point2d.vert.qsb"_s);
    auto pointFs = loadShader(u":/skigen/plot/point2d.frag.qsb"_s);
    auto edgeVs  = loadShader(u":/skigen/plot/edge3d.vert.qsb"_s);
    auto pt3dVs  = loadShader(u":/skigen/plot/point3d.vert.qsb"_s);
    auto pt3dFs  = loadShader(u":/skigen/plot/point3d.frag.qsb"_s);
    auto meshVs  = loadShader(u":/skigen/plot/mesh3d.vert.qsb"_s);
    auto meshFs  = loadShader(u":/skigen/plot/mesh3d.frag.qsb"_s);
    if (!lineVs || !lineFs || !fillVs || !fillFs || !pointVs || !pointFs ||
        !edgeVs || !pt3dVs || !pt3dFs || !meshVs  || !meshFs) {
        qWarning("SkigenPlot: shader loading failed");
        return;
    }

    auto* rpDesc = renderTarget()->renderPassDescriptor();

    // ── 2D vertex layout (vec2, stride 8) ──────────────────────────
    QRhiVertexInputLayout layout2d;
    layout2d.setBindings({QRhiVertexInputBinding(2 * sizeof(float))});
    layout2d.setAttributes({
        QRhiVertexInputAttribute(0, 0, QRhiVertexInputAttribute::Float2, 0)
    });

    // ── Heatmap layout (vec2 pos + vec4 colour, stride 24) ─────────
    QRhiVertexInputLayout layoutHeatmap;
    layoutHeatmap.setBindings({QRhiVertexInputBinding(6 * sizeof(float))});
    layoutHeatmap.setAttributes({
        QRhiVertexInputAttribute(0, 0, QRhiVertexInputAttribute::Float2, 0),
        QRhiVertexInputAttribute(0, 1, QRhiVertexInputAttribute::Float4,
                                 2 * sizeof(float))
    });

    // ── 3D point layout (vec3, stride 12) ──────────────────────────
    QRhiVertexInputLayout layout3dPoint;
    layout3dPoint.setBindings({QRhiVertexInputBinding(3 * sizeof(float))});
    layout3dPoint.setAttributes({
        QRhiVertexInputAttribute(0, 0, QRhiVertexInputAttribute::Float3, 0)
    });

    // ── 3D mesh layout (vec3 pos + vec3 normal, stride 24) ─────────
    QRhiVertexInputLayout layoutMesh;
    layoutMesh.setBindings({QRhiVertexInputBinding(6 * sizeof(float))});
    layoutMesh.setAttributes({
        QRhiVertexInputAttribute(0, 0, QRhiVertexInputAttribute::Float3, 0),
        QRhiVertexInputAttribute(0, 1, QRhiVertexInputAttribute::Float3,
                                 3 * sizeof(float))
    });

    // We need a reference SRB for pipeline creation — use gridSrb
    // (all 2D SRBs are layout-compatible: single UB at binding 0)

    // ── Line pipeline (LineStrip, no depth) ────────────────────────
    d->linePipeline.reset(r->newGraphicsPipeline());
    d->linePipeline->setTopology(QRhiGraphicsPipeline::LineStrip);
    d->linePipeline->setShaderStages({
        {QRhiShaderStage::Vertex, *lineVs},
        {QRhiShaderStage::Fragment, *lineFs}
    });
    d->linePipeline->setVertexInputLayout(layout2d);
    d->linePipeline->setShaderResourceBindings(d->gridSrb.get());
    d->linePipeline->setRenderPassDescriptor(rpDesc);
    d->linePipeline->setSampleCount(sc);
    d->linePipeline->setLineWidth(1.2f);
    d->linePipeline->setTargetBlends({alphaBlend()});
    d->linePipeline->create();

    // ── Grid pipeline (Lines, no depth, alpha blending) ────────────
    d->gridPipeline.reset(r->newGraphicsPipeline());
    d->gridPipeline->setTopology(QRhiGraphicsPipeline::Lines);
    d->gridPipeline->setShaderStages({
        {QRhiShaderStage::Vertex, *lineVs},
        {QRhiShaderStage::Fragment, *lineFs}
    });
    d->gridPipeline->setVertexInputLayout(layout2d);
    d->gridPipeline->setShaderResourceBindings(d->gridSrb.get());
    d->gridPipeline->setRenderPassDescriptor(rpDesc);
    d->gridPipeline->setSampleCount(sc);
    d->gridPipeline->setLineWidth(1.0f);
    d->gridPipeline->setTargetBlends({alphaBlend()});
    d->gridPipeline->create();

    // ── Point pipeline (Points, no depth) ──────────────────────────
    d->pointPipeline.reset(r->newGraphicsPipeline());
    d->pointPipeline->setTopology(QRhiGraphicsPipeline::Points);
    d->pointPipeline->setShaderStages({
        {QRhiShaderStage::Vertex, *pointVs},
        {QRhiShaderStage::Fragment, *pointFs}
    });
    d->pointPipeline->setVertexInputLayout(layout2d);
    d->pointPipeline->setShaderResourceBindings(d->gridSrb.get());
    d->pointPipeline->setRenderPassDescriptor(rpDesc);
    d->pointPipeline->setSampleCount(sc);
    d->pointPipeline->setTargetBlends({alphaBlend()});
    d->pointPipeline->create();

    // ── Fill pipeline (Triangles, no depth, alpha blend) ───────────
    // Reuses the line2d shaders (uniform mvp + colour, vec2 input) to draw
    // filled 2D geometry: bars, histogram, area fills, error-bar caps.
    d->fillPipeline.reset(r->newGraphicsPipeline());
    d->fillPipeline->setTopology(QRhiGraphicsPipeline::Triangles);
    d->fillPipeline->setShaderStages({
        {QRhiShaderStage::Vertex, *lineVs},
        {QRhiShaderStage::Fragment, *lineFs}
    });
    d->fillPipeline->setVertexInputLayout(layout2d);
    d->fillPipeline->setShaderResourceBindings(d->gridSrb.get());
    d->fillPipeline->setRenderPassDescriptor(rpDesc);
    d->fillPipeline->setSampleCount(sc);
    d->fillPipeline->setTargetBlends({alphaBlend()});
    d->fillPipeline->create();

    // ── Heatmap pipeline (Triangles, per-vertex colour, no depth) ──
    d->heatmapPipeline.reset(r->newGraphicsPipeline());
    d->heatmapPipeline->setTopology(QRhiGraphicsPipeline::Triangles);
    d->heatmapPipeline->setShaderStages({
        {QRhiShaderStage::Vertex, *fillVs},
        {QRhiShaderStage::Fragment, *fillFs}
    });
    d->heatmapPipeline->setVertexInputLayout(layoutHeatmap);
    d->heatmapPipeline->setShaderResourceBindings(d->heatmapSrb.get());
    d->heatmapPipeline->setRenderPassDescriptor(rpDesc);
    d->heatmapPipeline->setSampleCount(sc);
    d->heatmapPipeline->setTargetBlends({alphaBlend()});
    d->heatmapPipeline->create();

    // ── 3D Point pipeline (Points, depth enabled) ──────────────────
    d->point3dPipeline.reset(r->newGraphicsPipeline());
    d->point3dPipeline->setTopology(QRhiGraphicsPipeline::Points);
    d->point3dPipeline->setShaderStages({
        {QRhiShaderStage::Vertex, *pt3dVs},
        {QRhiShaderStage::Fragment, *pt3dFs}
    });
    d->point3dPipeline->setVertexInputLayout(layout3dPoint);
    d->point3dPipeline->setShaderResourceBindings(d->point3dSrb.get());
    d->point3dPipeline->setRenderPassDescriptor(rpDesc);
    d->point3dPipeline->setSampleCount(sc);
    d->point3dPipeline->setDepthTest(true);
    d->point3dPipeline->setDepthWrite(true);
    d->point3dPipeline->setTargetBlends({alphaBlend()});
    d->point3dPipeline->create();

    // ── Mesh pipeline (Triangles, depth enabled) ───────────────────
    d->meshPipeline.reset(r->newGraphicsPipeline());
    d->meshPipeline->setTopology(QRhiGraphicsPipeline::Triangles);
    d->meshPipeline->setShaderStages({
        {QRhiShaderStage::Vertex, *meshVs},
        {QRhiShaderStage::Fragment, *meshFs}
    });
    d->meshPipeline->setVertexInputLayout(layoutMesh);
    d->meshPipeline->setShaderResourceBindings(d->meshSrb.get());
    d->meshPipeline->setRenderPassDescriptor(rpDesc);
    d->meshPipeline->setSampleCount(sc);
    d->meshPipeline->setDepthTest(true);
    d->meshPipeline->setDepthWrite(true);
    d->meshPipeline->create();

    // ── Mesh edge pipeline (3D sharp-edge overlay) ─────────────────
    d->meshEdgePipeline.reset(r->newGraphicsPipeline());
    d->meshEdgePipeline->setTopology(QRhiGraphicsPipeline::Lines);
    d->meshEdgePipeline->setShaderStages({
        {QRhiShaderStage::Vertex, *edgeVs},
        {QRhiShaderStage::Fragment, *lineFs}
    });
    d->meshEdgePipeline->setVertexInputLayout(layout3dPoint);
    d->meshEdgePipeline->setShaderResourceBindings(d->meshEdgeSrb.get());
    d->meshEdgePipeline->setRenderPassDescriptor(rpDesc);
    d->meshEdgePipeline->setSampleCount(sc);
    d->meshEdgePipeline->setLineWidth(1.35f);
    d->meshEdgePipeline->setDepthTest(true);
    d->meshEdgePipeline->setDepthWrite(false);
    d->meshEdgePipeline->setDepthOp(QRhiGraphicsPipeline::LessOrEqual);
    d->meshEdgePipeline->setTargetBlends({alphaBlend()});
    d->meshEdgePipeline->create();

    // ── 3D guide pipeline (axes, grid, and arrowheads) ─────────────
    d->guide3dPipeline.reset(r->newGraphicsPipeline());
    d->guide3dPipeline->setTopology(QRhiGraphicsPipeline::Lines);
    d->guide3dPipeline->setShaderStages({
        {QRhiShaderStage::Vertex, *edgeVs},
        {QRhiShaderStage::Fragment, *lineFs}
    });
    d->guide3dPipeline->setVertexInputLayout(layout3dPoint);
    d->guide3dPipeline->setShaderResourceBindings(d->guide3dSrb.get());
    d->guide3dPipeline->setRenderPassDescriptor(rpDesc);
    d->guide3dPipeline->setSampleCount(sc);
    d->guide3dPipeline->setLineWidth(1.0f);
    d->guide3dPipeline->setDepthTest(true);
    d->guide3dPipeline->setDepthWrite(false);
    d->guide3dPipeline->setDepthOp(QRhiGraphicsPipeline::LessOrEqual);
    d->guide3dPipeline->setTargetBlends({alphaBlend()});
    d->guide3dPipeline->create();

    d->pipelineReady = true;
    d->data3dDirty = true;
    d->indexDirty = true;
    d->meshEdgeDirty = true;
    d->guide3dDirty = true;
    d->gridDirty = true;
    for (auto& s : d->series2d)
        s.dirty = true;
}

void PlotView::render(QRhiCommandBuffer* cb) {
    if (!d->pipelineReady || !d->hasData())
        return;

    auto* r = rhi();
    auto* u = r->nextResourceUpdateBatch();
    const QSize sz = renderTarget()->pixelSize();

    renderToTarget(cb, renderTarget(), u, sz);
}

void PlotView::renderToTarget(QRhiCommandBuffer* cb,
                               QRhiRenderTarget* rt,
                               QRhiResourceUpdateBatch* u,
                               const QSize& sz) {
    auto* r = rhi();
    bool is2D = d->has2D() && !d->has3D();

    d->viewportW = std::max(1.0f, static_cast<float>(sz.width()));
    d->viewportH = std::max(1.0f, static_cast<float>(sz.height()));

    // ── Prepare grid (2D only) ─────────────────────────────────────
    if (is2D && d->gridDirty)
        computeGridVertices();

    // ── Compute MVP ────────────────────────────────────────────────
    Eigen::Matrix4f mvp;
    if (is2D) {
        BoundingBox2D projBounds = d->viewBounds;
        if (d->aspectEqual) {
            // Expand the shorter data axis so 1 data-unit maps to the same
            // pixel length on both axes (keeps circles/pies round).
            QRectF pa = plotAreaFor(sz, !d->title.isEmpty(), !d->caption.isEmpty(),
                                    !d->xAxisLabel.isEmpty(), !d->yAxisLabel.isEmpty(),
                                    d->showColorbar && d->hasColormappedData);
            double paAspect = std::max(1.0, pa.width()) / std::max(1.0, pa.height());
            float w = projBounds.width(), h = projBounds.height();
            if (w > 1e-6f && h > 1e-6f) {
                double dataAspect = static_cast<double>(w) / static_cast<double>(h);
                if (dataAspect < paAspect) {
                    float targetW = static_cast<float>(h * paAspect);
                    float pad = (targetW - w) * 0.5f;
                    projBounds.min.x() -= pad; projBounds.max.x() += pad;
                } else {
                    float targetH = static_cast<float>(w / paAspect);
                    float pad = (targetH - h) * 0.5f;
                    projBounds.min.y() -= pad; projBounds.max.y() += pad;
                }
            }
        }
        mvp = orthoProjection(projBounds, 0.02f);
    } else {
        mvp = d->camera.viewProjectionMatrix();
    }

    QMatrix4x4 correction = r->clipSpaceCorrMatrix();
    Eigen::Matrix4f corr;
    for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 4; ++col)
            corr(row, col) = correction(row, col);
    mvp = corr * mvp;

    Eigen::Vector4f bgColor = d->userBgColor.value_or(d->theme.background);

    // ── Upload 2D series vertex data ───────────────────────────────
    for (auto& s : d->series2d) {
        auto requiredFloats = static_cast<int>(s.vertices.size());
        if (requiredFloats > s.vbCapacity) {
            delete s.vb;
            s.vbCapacity = requiredFloats * 2;
            s.vb = makeRawDynBuf(r, QRhiBuffer::VertexBuffer,
                                  quint32(s.vbCapacity * sizeof(float)));
            s.dirty = true;
        }
        if (s.dirty && !s.vertices.empty()) {
            u->updateDynamicBuffer(s.vb, 0,
                                   quint32(s.vertices.size() * sizeof(float)),
                                   s.vertices.data());
            s.dirty = false;
        }
    }

    // ── Upload 3D vertex data ──────────────────────────────────────
    if (d->has3D()) {
        auto requiredFloats = static_cast<int>(d->vertices3d.size());
        if (requiredFloats > d->vertex3dCapacity) {
            d->vertex3dCapacity = requiredFloats * 2;
            d->vertex3dBuffer = makeDynBuf(r, QRhiBuffer::VertexBuffer,
                                            quint32(d->vertex3dCapacity * sizeof(float)));
            d->data3dDirty = true;
        }
        if (d->data3dDirty) {
            u->updateDynamicBuffer(d->vertex3dBuffer.get(), 0,
                                   quint32(d->vertices3d.size() * sizeof(float)),
                                   d->vertices3d.data());
            d->data3dDirty = false;
        }
        if (d->mode3d == RenderMode3D::Mesh && d->indexDirty) {
            auto requiredIdx = static_cast<int>(d->indices.size());
            if (requiredIdx > d->indexCapacity) {
                d->indexCapacity = requiredIdx * 2;
                d->indexBuffer = makeDynBuf(r, QRhiBuffer::IndexBuffer,
                                             quint32(d->indexCapacity * sizeof(uint32_t)));
            }
            u->updateDynamicBuffer(d->indexBuffer.get(), 0,
                                   quint32(d->indices.size() * sizeof(uint32_t)),
                                   d->indices.data());
            d->indexDirty = false;
        }
        if (d->mode3d == RenderMode3D::Mesh && d->meshEdgeDirty) {
            auto requiredFloats = static_cast<int>(d->meshEdgeVertices.size());
            if (requiredFloats > d->meshEdgeCapacity) {
                d->meshEdgeCapacity = requiredFloats * 2;
                d->meshEdgeBuffer = makeDynBuf(r, QRhiBuffer::VertexBuffer,
                                                quint32(d->meshEdgeCapacity * sizeof(float)));
            }
            if (!d->meshEdgeVertices.empty()) {
                u->updateDynamicBuffer(d->meshEdgeBuffer.get(), 0,
                                       quint32(d->meshEdgeVertices.size() * sizeof(float)),
                                       d->meshEdgeVertices.data());
            }
            d->meshEdgeDirty = false;
        }
        if (d->guide3dDirty) {
            auto requiredFloats = static_cast<int>(d->guide3dVertices.size());
            if (requiredFloats > d->guide3dCapacity) {
                d->guide3dCapacity = requiredFloats * 2;
                d->guide3dBuffer = makeDynBuf(r, QRhiBuffer::VertexBuffer,
                                               quint32(d->guide3dCapacity * sizeof(float)));
            }
            if (!d->guide3dVertices.empty()) {
                u->updateDynamicBuffer(d->guide3dBuffer.get(), 0,
                                       quint32(d->guide3dVertices.size() * sizeof(float)),
                                       d->guide3dVertices.data());
            }
            d->guide3dDirty = false;
        }
    }

    // ── Upload grid vertex data (2D only) ──────────────────────────
    if (is2D && (d->showGrid || d->showAxes)) {
        auto totalGridFloats = static_cast<int>(
            d->gridVertices.size() + d->axisVertices.size());
        if (totalGridFloats > d->gridVertexCapacity) {
            d->gridVertexCapacity = totalGridFloats * 2;
            d->gridVertexBuffer = makeDynBuf(r, QRhiBuffer::VertexBuffer,
                                              quint32(d->gridVertexCapacity * sizeof(float)));
        }
        if (!d->gridVertices.empty()) {
            u->updateDynamicBuffer(d->gridVertexBuffer.get(), 0,
                                   quint32(d->gridVertices.size() * sizeof(float)),
                                   d->gridVertices.data());
        }
        if (!d->axisVertices.empty()) {
            u->updateDynamicBuffer(d->gridVertexBuffer.get(),
                                   quint32(d->gridVertices.size() * sizeof(float)),
                                   quint32(d->axisVertices.size() * sizeof(float)),
                                   d->axisVertices.data());
        }
    }

    // ── Upload grid uniforms ───────────────────────────────────────
    if (is2D && d->showGrid) {
        u->updateDynamicBuffer(d->gridUniformBuffer.get(), 0, 64, mvp.data());
        u->updateDynamicBuffer(d->gridUniformBuffer.get(), 64, 16,
                               d->theme.gridColor.data());
    }
    if (is2D && d->showAxes) {
        u->updateDynamicBuffer(d->axisUniformBuffer.get(), 0, 64, mvp.data());
        u->updateDynamicBuffer(d->axisUniformBuffer.get(), 64, 16,
                               d->theme.axisColor.data());
    }

    // ── Upload per-series uniforms ─────────────────────────────────
    for (auto& s : d->series2d) {
        u->updateDynamicBuffer(s.ub, 0, 64, mvp.data());
        u->updateDynamicBuffer(s.ub, 64, 16, s.color.data());
        if (s.kind == SeriesKind::Scatter) {
            Eigen::Vector4f params(s.pointSize, s.hollow ? 1.f : 0.f, 0.f, 0.f);
            u->updateDynamicBuffer(s.ub, 80, 16, params.data());
        }
    }

    // ── Upload heatmap uniform + vertex data ───────────────────────
    if (is2D && d->hasHeatmap && d->heatmapBuffer) {
        u->updateDynamicBuffer(d->heatmapUniformBuffer.get(), 0, 64, mvp.data());
        if (d->heatmapDirty && !d->heatmapVertices.empty()) {
            u->updateDynamicBuffer(d->heatmapBuffer.get(), 0,
                                   quint32(d->heatmapVertices.size() * sizeof(float)),
                                   d->heatmapVertices.data());
            d->heatmapDirty = false;
        }
    }

    // ── Upload contour uniform + vertex data ───────────────────────
    if (is2D && d->hasContour && d->contourBuffer) {
        u->updateDynamicBuffer(d->contourUniformBuffer.get(), 0, 64, mvp.data());
        u->updateDynamicBuffer(d->contourUniformBuffer.get(), 64, 16,
                               d->contourColor.data());
        if (d->contourDirty && !d->contourVertices.empty()) {
            u->updateDynamicBuffer(d->contourBuffer.get(), 0,
                                   quint32(d->contourVertices.size() * sizeof(float)),
                                   d->contourVertices.data());
            d->contourDirty = false;
        }
    }

    // ── Upload 3D uniforms ─────────────────────────────────────────
    if (d->mode3d == RenderMode3D::PointCloud) {
        Eigen::Vector4f params(d->data3dPointSize, 0.f, 0.f, 0.f);
        u->updateDynamicBuffer(d->point3dUniformBuffer.get(), 0, 64, mvp.data());
        u->updateDynamicBuffer(d->point3dUniformBuffer.get(), 64, 16,
                               d->data3dColor.data());
        u->updateDynamicBuffer(d->point3dUniformBuffer.get(), 80, 16, params.data());
    }
    if (d->mode3d == RenderMode3D::Mesh) {
        Eigen::Vector3f eye = d->camera.position();
        Eigen::Vector3f target = d->camera.target();
        Eigen::Vector3f forward = (target - eye).normalized();
        Eigen::Vector3f right = forward.cross(Eigen::Vector3f::UnitY());
        if (right.squaredNorm() < 1e-8f)
            right = Eigen::Vector3f::UnitX();
        right.normalize();
        Eigen::Vector3f up = right.cross(forward).normalized();
        Eigen::Vector3f keyLight = (-forward + 0.50f * up + 0.24f * right).normalized();
        Eigen::Vector4f lightDir(keyLight.x(), keyLight.y(), keyLight.z(), 0.f);
        Eigen::Vector4f lightParams(0.30f, 0.78f, 0.46f, 0.52f);
        bool darkBg = bgColor.x() < 0.5f;
        Eigen::Vector4f edgeColor = darkBg
            ? Eigen::Vector4f(0.82f, 0.96f, 1.00f, 0.48f)
            : Eigen::Vector4f(0.05f, 0.12f, 0.20f, 0.34f);
        u->updateDynamicBuffer(d->meshUniformBuffer.get(), 0, 64, mvp.data());
        u->updateDynamicBuffer(d->meshUniformBuffer.get(), 64, 16,
                               d->data3dColor.data());
        u->updateDynamicBuffer(d->meshUniformBuffer.get(), 80, 16, lightDir.data());
        u->updateDynamicBuffer(d->meshUniformBuffer.get(), 96, 16, lightParams.data());

        u->updateDynamicBuffer(d->meshEdgeUniformBuffer.get(), 0, 64, mvp.data());
        u->updateDynamicBuffer(d->meshEdgeUniformBuffer.get(), 64, 16,
                               edgeColor.data());
    }
    if (d->has3D() && d->guide3dVertexCount > 0) {
        bool darkBg = bgColor.x() < 0.5f;
        Eigen::Vector4f guideColor = darkBg
            ? Eigen::Vector4f(0.82f, 0.86f, 0.95f, 0.30f)
            : Eigen::Vector4f(0.18f, 0.22f, 0.28f, 0.34f);
        u->updateDynamicBuffer(d->guide3dUniformBuffer.get(), 0, 64, mvp.data());
        u->updateDynamicBuffer(d->guide3dUniformBuffer.get(), 64, 16,
                               guideColor.data());
    }

    // ── Begin render pass ──────────────────────────────────────────
    cb->beginPass(rt,
                  QColor::fromRgbF(bgColor.x(), bgColor.y(), bgColor.z(), bgColor.w()),
                  {1.0f, 0}, u);

    if (is2D) {
        QRectF plotArea = plotAreaFor(sz,
                                      !d->title.isEmpty(),
                                      !d->caption.isEmpty(),
                                      !d->xAxisLabel.isEmpty(),
                                      !d->yAxisLabel.isEmpty(),
                                      d->showColorbar && d->hasColormappedData);
        cb->setViewport({static_cast<float>(plotArea.x()),
                         static_cast<float>(plotArea.y()),
                         static_cast<float>(plotArea.width()),
                         static_cast<float>(plotArea.height())});
    } else {
        cb->setViewport({0, 0,
                         static_cast<float>(sz.width()),
                         static_cast<float>(sz.height())});
    }

    // ── Draw 3D guide frame under data ─────────────────────────────
    if (d->has3D() && d->guide3dVertexCount > 0) {
        cb->setGraphicsPipeline(d->guide3dPipeline.get());
        cb->setShaderResources(d->guide3dSrb.get());
        const QRhiCommandBuffer::VertexInput guideBuf(d->guide3dBuffer.get(), 0);
        cb->setVertexInput(0, 1, &guideBuf);
        cb->draw(d->guide3dVertexCount);
    }

    // ── Draw heatmap (2D, under grid/axes) ─────────────────────────
    if (is2D && d->hasHeatmap && d->heatmapBuffer && d->heatmapVertexCount > 0) {
        cb->setGraphicsPipeline(d->heatmapPipeline.get());
        cb->setShaderResources(d->heatmapSrb.get());
        const QRhiCommandBuffer::VertexInput hmBuf(d->heatmapBuffer.get(), 0);
        cb->setVertexInput(0, 1, &hmBuf);
        cb->draw(d->heatmapVertexCount);
    }

    // ── Draw contour lines (2D, over any filled heatmap) ───────────
    if (is2D && d->hasContour && d->contourBuffer && d->contourVertexCount > 0) {
        cb->setGraphicsPipeline(d->gridPipeline.get());
        cb->setShaderResources(d->contourSrb.get());
        const QRhiCommandBuffer::VertexInput cbuf(d->contourBuffer.get(), 0);
        cb->setVertexInput(0, 1, &cbuf);
        cb->draw(d->contourVertexCount);
    }

    // ── Draw grid (2D only) ────────────────────────────────────────
    if (is2D && d->showGrid && d->gridVertexCount > 0) {
        cb->setGraphicsPipeline(d->gridPipeline.get());
        cb->setShaderResources(d->gridSrb.get());
        const QRhiCommandBuffer::VertexInput gridBuf(d->gridVertexBuffer.get(), 0);
        cb->setVertexInput(0, 1, &gridBuf);
        cb->draw(d->gridVertexCount);
    }

    // ── Draw axes + ticks (2D only) ────────────────────────────────
    if (is2D && d->showAxes && d->axisVertexCount > 0) {
        cb->setGraphicsPipeline(d->gridPipeline.get());
        cb->setShaderResources(d->axisSrb.get());
        quint32 axisOffset = quint32(d->gridVertices.size() * sizeof(float));
        const QRhiCommandBuffer::VertexInput axisBuf(d->gridVertexBuffer.get(), axisOffset);
        cb->setVertexInput(0, 1, &axisBuf);
        cb->draw(d->axisVertexCount);
    }

    // ── Draw 2D series ─────────────────────────────────────────────
    // Two passes so filled geometry (bars, histogram, area) renders *under*
    // line/scatter series, matching matplotlib's z-ordering.
    auto drawSeries2D = [&](const Series2D& s) {
        if (!s.visible || s.vertexCount <= 0) return;
        if (s.kind == SeriesKind::Line) {
            cb->setGraphicsPipeline(d->linePipeline.get());
        } else if (s.kind == SeriesKind::Fill) {
            cb->setGraphicsPipeline(d->fillPipeline.get());
        } else {
            cb->setGraphicsPipeline(d->pointPipeline.get());
        }
        cb->setShaderResources(s.srb);
        const QRhiCommandBuffer::VertexInput vbuf(s.vb, 0);
        cb->setVertexInput(0, 1, &vbuf);
        cb->draw(s.vertexCount);
    };
    for (const auto& s : d->series2d)
        if (s.kind == SeriesKind::Fill) drawSeries2D(s);
    for (const auto& s : d->series2d)
        if (s.kind != SeriesKind::Fill) drawSeries2D(s);

    // ── Draw 3D data ───────────────────────────────────────────────
    if (d->mode3d == RenderMode3D::PointCloud) {
        cb->setGraphicsPipeline(d->point3dPipeline.get());
        cb->setShaderResources(d->point3dSrb.get());
        const QRhiCommandBuffer::VertexInput vbuf(d->vertex3dBuffer.get(), 0);
        cb->setVertexInput(0, 1, &vbuf);
        cb->draw(d->vertex3dCount);
    }
    if (d->mode3d == RenderMode3D::Mesh) {
        cb->setGraphicsPipeline(d->meshPipeline.get());
        cb->setShaderResources(d->meshSrb.get());
        const QRhiCommandBuffer::VertexInput vbuf(d->vertex3dBuffer.get(), 0);
        cb->setVertexInput(0, 1, &vbuf, d->indexBuffer.get(), 0,
                           QRhiCommandBuffer::IndexUInt32);
        cb->drawIndexed(d->indexCount);

        if (d->meshEdgeVertexCount > 0) {
            cb->setGraphicsPipeline(d->meshEdgePipeline.get());
            cb->setShaderResources(d->meshEdgeSrb.get());
            const QRhiCommandBuffer::VertexInput edgeBuf(d->meshEdgeBuffer.get(), 0);
            cb->setVertexInput(0, 1, &edgeBuf);
            cb->draw(d->meshEdgeVertexCount);
        }
    }

    cb->endPass();
}

// ── PNG export ─────────────────────────────────────────────────────

auto PlotView::savePng(const QString& path, int width, int height) -> bool {
    auto* r = rhi();
    if (!r || !d->pipelineReady || !d->hasData())
        return false;

    QSize sz(width, height);

    std::unique_ptr<QRhiTexture> tex(
        r->newTexture(QRhiTexture::RGBA8, sz, 1,
                      QRhiTexture::RenderTarget | QRhiTexture::UsedAsTransferSource));
    if (!tex->create()) return false;

    std::unique_ptr<QRhiRenderBuffer> ds(
        r->newRenderBuffer(QRhiRenderBuffer::DepthStencil, sz));
    if (!ds->create()) return false;

    QRhiTextureRenderTargetDescription rtDesc;
    rtDesc.setColorAttachments({QRhiColorAttachment(tex.get())});
    rtDesc.setDepthStencilBuffer(ds.get());

    std::unique_ptr<QRhiTextureRenderTarget> rt(
        r->newTextureRenderTarget(rtDesc));
    std::unique_ptr<QRhiRenderPassDescriptor> rpDesc(
        rt->newCompatibleRenderPassDescriptor());
    rt->setRenderPassDescriptor(rpDesc.get());
    if (!rt->create()) return false;

    QRhiCommandBuffer* cb = nullptr;
    if (r->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess)
        return false;

    auto* u = r->nextResourceUpdateBatch();

    d->data3dDirty = true;
    d->indexDirty = true;
    d->meshEdgeDirty = true;
    d->gridDirty = true;
    for (auto& s : d->series2d)
        s.dirty = true;

    renderToTarget(cb, rt.get(), u, sz);

    QRhiReadbackResult readResult;
    bool readComplete = false;
    readResult.completed = [&readComplete] { readComplete = true; };

    auto* readBatch = r->nextResourceUpdateBatch();
    readBatch->readBackTexture(QRhiReadbackDescription(tex.get()), &readResult);
    cb->resourceUpdate(readBatch);

    r->endOffscreenFrame();

    if (!readComplete || readResult.data.isEmpty())
        return false;

    QImage img(reinterpret_cast<const uchar*>(readResult.data.constData()),
               width, height, QImage::Format_RGBA8888);
    QImage annotated = img.copy();
    QPainter painter(&annotated);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setRenderHint(QPainter::TextAntialiasing);
    paintTextOverlay(painter, annotated.size());
    painter.end();
    return annotated.save(path, "PNG");
}

} // namespace Skigen::Plot
