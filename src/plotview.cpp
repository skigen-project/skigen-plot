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
#include <cmath>
#include <expected>
#include <numbers>
#include <optional>
#include <unordered_map>
#include <vector>

using namespace Qt::StringLiterals;

namespace Skigen::Plot {

// ── Render mode (for 3D, which stays single-dataset) ───────────────

enum class RenderMode3D { None, PointCloud, Mesh };
enum class SeriesKind { Line, Scatter };

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
    SeriesKind kind;
    std::vector<float> vertices;
    int vertexCount = 0;
    Eigen::Vector4f color;
    float pointSize = 5.0f;
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

    // Shared pipelines (all Line2D series share linePipeline, etc.)
    std::unique_ptr<QRhiGraphicsPipeline> linePipeline;
    std::unique_ptr<QRhiGraphicsPipeline> pointPipeline;
    std::unique_ptr<QRhiGraphicsPipeline> gridPipeline;

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

    bool has2D() const { return !series2d.empty(); }
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

void PlotView::addLineSeries(std::span<const float> x,
                              std::span<const float> y,
                              const PlotStyle& style) {
    addSeriesImpl(static_cast<int>(SeriesKind::Line), x, y, style);
}

void PlotView::addScatterSeries(std::span<const float> x,
                                 std::span<const float> y,
                                 const PlotStyle& style) {
    addSeriesImpl(static_cast<int>(SeriesKind::Scatter), x, y, style);
}

void PlotView::addSeriesImpl(int kindInt,
                              std::span<const float> x,
                              std::span<const float> y,
                              const PlotStyle& style) {
    auto kind = static_cast<SeriesKind>(kindInt);
    auto n = static_cast<int>(std::min(x.size(), y.size()));
    if (!d->has2D() && !d->has3D())
        d->interactionTool = InteractionTool::Pan;

    Series2D series;
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
    series.dirty = true;

    if (d->pipelineReady) {
        auto* r = rhi();
        quint32 ubSize = (kind == SeriesKind::Scatter) ? 96u : 80u;
        series.ub = makeRawUB(r, ubSize);
        series.srb = makeSrb(r, series.ub);
        int floats = static_cast<int>(series.vertices.size());
        series.vb = makeRawDynBuf(r, QRhiBuffer::VertexBuffer,
                                   quint32(floats * sizeof(float)));
        series.vbCapacity = floats;
    }

    d->series2d.push_back(std::move(series));
    recomputeBounds();
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
    d->mode3d = RenderMode3D::None;
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
        if (s.vertexCount == 0) continue;
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
        float arrowX = (xhi - xlo) * 0.018f;
        float arrowY = (yhi - ylo) * 0.018f;

        auto appendLine2D = [&](float ax, float ay, float bx, float by) {
            d->axisVertices.push_back(ax); d->axisVertices.push_back(ay);
            d->axisVertices.push_back(bx); d->axisVertices.push_back(by);
        };

        float xArrowHalfHeight = arrowY * 0.24f;
        appendLine2D(xhi, ylo, xhi - arrowX, ylo + xArrowHalfHeight);
        appendLine2D(xhi, ylo, xhi - arrowX, ylo - xArrowHalfHeight);
        appendLine2D(xhi - arrowX, ylo - xArrowHalfHeight,
                     xhi - arrowX, ylo + xArrowHalfHeight);

        float yArrowHalfWidth = arrowX * 0.24f;
        appendLine2D(xlo, yhi, xlo - yArrowHalfWidth, yhi - arrowY);
        appendLine2D(xlo, yhi, xlo + yArrowHalfWidth, yhi - arrowY);
        appendLine2D(xlo - yArrowHalfWidth, yhi - arrowY,
                     xlo + yArrowHalfWidth, yhi - arrowY);
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
                        bool hasYAxisLabel) -> QRectF
{
    double left = hasYAxisLabel ? 68.0 : 46.0;
    double top = hasTitle ? (hasCaption ? 58.0 : 38.0) : (hasCaption ? 38.0 : 18.0);
    double right = 24.0;
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
                                      !d->yAxisLabel.isEmpty());

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
                                          !d->yAxisLabel.isEmpty());
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

    // ── SRBs ───────────────────────────────────────────────────────
    d->point3dSrb = makeUniqueSrb(r, d->point3dUniformBuffer.get());
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

    // ── Load shaders ───────────────────────────────────────────────
    auto lineVs  = loadShader(u":/skigen/plot/line2d.vert.qsb"_s);
    auto lineFs  = loadShader(u":/skigen/plot/line2d.frag.qsb"_s);
    auto pointVs = loadShader(u":/skigen/plot/point2d.vert.qsb"_s);
    auto pointFs = loadShader(u":/skigen/plot/point2d.frag.qsb"_s);
    auto edgeVs  = loadShader(u":/skigen/plot/edge3d.vert.qsb"_s);
    auto pt3dVs  = loadShader(u":/skigen/plot/point3d.vert.qsb"_s);
    auto pt3dFs  = loadShader(u":/skigen/plot/point3d.frag.qsb"_s);
    auto meshVs  = loadShader(u":/skigen/plot/mesh3d.vert.qsb"_s);
    auto meshFs  = loadShader(u":/skigen/plot/mesh3d.frag.qsb"_s);
    if (!lineVs || !lineFs || !pointVs || !pointFs || !edgeVs ||
        !pt3dVs || !pt3dFs || !meshVs  || !meshFs) {
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

    // ── Prepare grid (2D only) ─────────────────────────────────────
    if (is2D && d->gridDirty)
        computeGridVertices();

    // ── Compute MVP ────────────────────────────────────────────────
    Eigen::Matrix4f mvp;
    if (is2D)
        mvp = orthoProjection(d->viewBounds, 0.02f);
    else
        mvp = d->camera.viewProjectionMatrix();

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
        if (s.dirty) {
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
            Eigen::Vector4f params(s.pointSize, 0.f, 0.f, 0.f);
            u->updateDynamicBuffer(s.ub, 80, 16, params.data());
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
                                      !d->yAxisLabel.isEmpty());
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
    for (const auto& s : d->series2d) {
        if (s.vertexCount <= 0) continue;

        if (s.kind == SeriesKind::Line) {
            cb->setGraphicsPipeline(d->linePipeline.get());
        } else {
            cb->setGraphicsPipeline(d->pointPipeline.get());
        }
        cb->setShaderResources(s.srb);
        const QRhiCommandBuffer::VertexInput vbuf(s.vb, 0);
        cb->setVertexInput(0, 1, &vbuf);
        cb->draw(s.vertexCount);
    }

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
