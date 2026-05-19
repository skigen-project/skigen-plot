#include <skigen/plot/plotview.h>

#include <rhi/qrhi.h>

#include <QFile>
#include <QImage>

#include <algorithm>
#include <cmath>
#include <expected>
#include <optional>
#include <vector>

using namespace Qt::StringLiterals;

namespace Skigen::Plot {

// ── Render mode ─────────────────────────────────────────────────────────

enum class RenderMode { None, Line2D, Scatter2D, PointCloud3D, Mesh3D };

// ── Shader loading ──────────────────────────────────────────────────────

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

// ── PIMPL ───────────────────────────────────────────────────────────────

struct PlotView::Impl {
    RenderMode mode = RenderMode::None;

    // 2D pipelines
    std::unique_ptr<QRhiGraphicsPipeline> linePipeline;
    std::unique_ptr<QRhiGraphicsPipeline> pointPipeline;
    std::unique_ptr<QRhiGraphicsPipeline> gridPipeline;

    // 3D pipelines
    std::unique_ptr<QRhiGraphicsPipeline> point3dPipeline;
    std::unique_ptr<QRhiGraphicsPipeline> meshPipeline;

    // 2D vertex data (interleaved x, y pairs)
    std::unique_ptr<QRhiBuffer> vertexBuffer;
    std::vector<float> vertices;
    int vertexCount = 0;
    int vertexCapacity = 0;
    bool dataDirty = false;

    // 3D vertex data (interleaved x, y, z)
    std::unique_ptr<QRhiBuffer> vertex3dBuffer;
    std::vector<float> vertices3d;
    int vertex3dCount = 0;
    int vertex3dCapacity = 0;
    bool data3dDirty = false;

    // Mesh index data
    std::unique_ptr<QRhiBuffer> indexBuffer;
    std::vector<uint32_t> indices;
    int indexCount = 0;
    int indexCapacity = 0;
    bool indexDirty = false;

    // Grid vertex data
    std::unique_ptr<QRhiBuffer> gridVertexBuffer;
    std::vector<float> gridVertices;
    std::vector<float> axisVertices;
    int gridVertexCount = 0;
    int axisVertexCount = 0;
    int gridVertexCapacity = 0;
    bool gridDirty = false;

    // Uniform buffers (separate per draw layer)
    std::unique_ptr<QRhiBuffer> lineUniformBuffer;     // 80 B
    std::unique_ptr<QRhiBuffer> scatterUniformBuffer;  // 96 B
    std::unique_ptr<QRhiBuffer> point3dUniformBuffer;  // 96 B
    std::unique_ptr<QRhiBuffer> meshUniformBuffer;     // 112 B
    std::unique_ptr<QRhiBuffer> gridUniformBuffer;     // 80 B
    std::unique_ptr<QRhiBuffer> axisUniformBuffer;     // 80 B

    // Shader resource bindings
    std::unique_ptr<QRhiShaderResourceBindings> lineSrb;
    std::unique_ptr<QRhiShaderResourceBindings> scatterSrb;
    std::unique_ptr<QRhiShaderResourceBindings> point3dSrb;
    std::unique_ptr<QRhiShaderResourceBindings> meshSrb;
    std::unique_ptr<QRhiShaderResourceBindings> gridSrb;
    std::unique_ptr<QRhiShaderResourceBindings> axisSrb;

    bool pipelineReady = false;

    // Spatial state
    BoundingBox2D bounds2d;
    BoundingBox2D viewBounds;
    BoundingBox3D bounds3d;
    Camera3D camera;

    // Appearance
    Theme theme = Theme::dark();
    std::optional<Eigen::Vector4f> userLineColor;
    std::optional<Eigen::Vector4f> userBgColor;
    float pointSize = 4.0f;
    bool showGrid = true;
    bool showAxes = true;
};

// ── Helpers ─────────────────────────────────────────────────────────────

static void interleave2D(std::span<const float> x, std::span<const float> y,
                         std::vector<float>& out, int& count,
                         BoundingBox2D& bounds)
{
    auto n = static_cast<int>(std::min(x.size(), y.size()));
    out.resize(static_cast<std::size_t>(n) * 2);
    for (int i = 0; i < n; ++i) {
        out[static_cast<std::size_t>(i) * 2]     = x[static_cast<std::size_t>(i)];
        out[static_cast<std::size_t>(i) * 2 + 1] = y[static_cast<std::size_t>(i)];
    }
    count = n;
    bounds = BoundingBox2D::fromXY(
        Eigen::Map<const Eigen::VectorXf>(x.data(), static_cast<Eigen::Index>(x.size())),
        Eigen::Map<const Eigen::VectorXf>(y.data(), static_cast<Eigen::Index>(y.size())));
}

static auto makeSrb(QRhi* r, QRhiBuffer* ub)
    -> std::unique_ptr<QRhiShaderResourceBindings>
{
    auto srb = std::unique_ptr<QRhiShaderResourceBindings>(
        r->newShaderResourceBindings());
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

// ── Construction ────────────────────────────────────────────────────────

PlotView::PlotView(QWidget* parent)
    : QRhiWidget(parent)
    , d(std::make_unique<Impl>())
{
}

PlotView::~PlotView() = default;

// ── Data setters ────────────────────────────────────────────────────────

void PlotView::setLineData(std::span<const float> x,
                           std::span<const float> y) {
    interleave2D(x, y, d->vertices, d->vertexCount, d->bounds2d);
    d->mode = RenderMode::Line2D;
    d->dataDirty = true;
    d->gridDirty = true;
    update();
}

void PlotView::setScatterData(std::span<const float> x,
                              std::span<const float> y) {
    interleave2D(x, y, d->vertices, d->vertexCount, d->bounds2d);
    d->mode = RenderMode::Scatter2D;
    d->dataDirty = true;
    d->gridDirty = true;
    update();
}

void PlotView::setPointCloudData(std::span<const float> data,
                                 int vertexCount) {
    d->vertices3d.resize(static_cast<std::size_t>(vertexCount) * 3);
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

    d->mode = RenderMode::PointCloud3D;
    d->data3dDirty = true;
    update();
}

void PlotView::setMeshData(std::span<const float> verts, int vertexCount,
                           std::span<const uint32_t> idx, int triangleCount) {
    std::vector<float> rowMajorVerts(static_cast<std::size_t>(vertexCount) * 3);
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

    d->mode = RenderMode::Mesh3D;
    d->data3dDirty = true;
    d->indexDirty = true;
    update();
}

// ── Appearance ──────────────────────────────────────────────────────────

void PlotView::setLineColor(const Eigen::Vector4f& rgba) {
    d->userLineColor = rgba;
    update();
}

void PlotView::setBackgroundColor(const Eigen::Vector4f& rgba) {
    d->userBgColor = rgba;
    update();
}

void PlotView::setPointSize(float size) {
    d->pointSize = size;
    update();
}

void PlotView::setTheme(const Theme& theme) {
    d->theme = theme;
    d->gridDirty = true;
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

void PlotView::setCamera(const Camera3D& camera) {
    d->camera = camera;
    update();
}

auto PlotView::camera() const -> const Camera3D& {
    return d->camera;
}

// ── Grid computation ────────────────────────────────────────────────────

void PlotView::computeGridVertices() {
    d->gridVertices.clear();
    d->axisVertices.clear();

    auto xTicks = computeTicks(d->bounds2d.min.x(), d->bounds2d.max.x());
    auto yTicks = computeTicks(d->bounds2d.min.y(), d->bounds2d.max.y());

    if (xTicks.ticks.empty() || yTicks.ticks.empty()) return;

    d->viewBounds.min = Eigen::Vector2f(xTicks.ticks.front(), yTicks.ticks.front());
    d->viewBounds.max = Eigen::Vector2f(xTicks.ticks.back(), yTicks.ticks.back());

    float ylo = d->viewBounds.min.y();
    float yhi = d->viewBounds.max.y();
    float xlo = d->viewBounds.min.x();
    float xhi = d->viewBounds.max.x();

    for (float xt : xTicks.ticks) {
        d->gridVertices.push_back(xt); d->gridVertices.push_back(ylo);
        d->gridVertices.push_back(xt); d->gridVertices.push_back(yhi);
    }
    for (float yt : yTicks.ticks) {
        d->gridVertices.push_back(xlo); d->gridVertices.push_back(yt);
        d->gridVertices.push_back(xhi); d->gridVertices.push_back(yt);
    }
    d->gridVertexCount = static_cast<int>(d->gridVertices.size()) / 2;

    float axisY = (ylo <= 0.f && yhi >= 0.f) ? 0.f : ylo;
    float axisX = (xlo <= 0.f && xhi >= 0.f) ? 0.f : xlo;

    d->axisVertices.push_back(xlo); d->axisVertices.push_back(axisY);
    d->axisVertices.push_back(xhi); d->axisVertices.push_back(axisY);
    d->axisVertices.push_back(axisX); d->axisVertices.push_back(ylo);
    d->axisVertices.push_back(axisX); d->axisVertices.push_back(yhi);

    float txLen = (yhi - ylo) * 0.015f;
    float tyLen = (xhi - xlo) * 0.015f;
    for (float xt : xTicks.ticks) {
        d->axisVertices.push_back(xt); d->axisVertices.push_back(axisY - txLen);
        d->axisVertices.push_back(xt); d->axisVertices.push_back(axisY + txLen);
    }
    for (float yt : yTicks.ticks) {
        d->axisVertices.push_back(axisX - tyLen); d->axisVertices.push_back(yt);
        d->axisVertices.push_back(axisX + tyLen); d->axisVertices.push_back(yt);
    }
    d->axisVertexCount = static_cast<int>(d->axisVertices.size()) / 2;
    d->gridDirty = false;
}

// ── QRhiWidget overrides ────────────────────────────────────────────────

void PlotView::initialize(QRhiCommandBuffer* /*cb*/) {
    auto* r = rhi();
    d->pipelineReady = false;

    static constexpr int kInitialFloats = 128 * 1024;
    int sc = renderTarget()->sampleCount();

    // ── Vertex buffers ──────────────────────────────────────────────────
    d->vertexCapacity = kInitialFloats;
    d->vertexBuffer = makeDynBuf(r, QRhiBuffer::VertexBuffer,
                                  quint32(kInitialFloats * sizeof(float)));

    d->vertex3dCapacity = kInitialFloats;
    d->vertex3dBuffer = makeDynBuf(r, QRhiBuffer::VertexBuffer,
                                    quint32(kInitialFloats * sizeof(float)));

    d->indexCapacity = kInitialFloats;
    d->indexBuffer = makeDynBuf(r, QRhiBuffer::IndexBuffer,
                                 quint32(kInitialFloats * sizeof(uint32_t)));

    d->gridVertexCapacity = kInitialFloats;
    d->gridVertexBuffer = makeDynBuf(r, QRhiBuffer::VertexBuffer,
                                      quint32(kInitialFloats * sizeof(float)));

    // ── Uniform buffers ─────────────────────────────────────────────────
    d->lineUniformBuffer    = makeUB(r, 80);
    d->scatterUniformBuffer = makeUB(r, 96);
    d->point3dUniformBuffer = makeUB(r, 96);
    d->meshUniformBuffer    = makeUB(r, 112);
    d->gridUniformBuffer    = makeUB(r, 80);
    d->axisUniformBuffer    = makeUB(r, 80);

    // ── SRBs ────────────────────────────────────────────────────────────
    d->lineSrb    = makeSrb(r, d->lineUniformBuffer.get());
    d->scatterSrb = makeSrb(r, d->scatterUniformBuffer.get());
    d->point3dSrb = makeSrb(r, d->point3dUniformBuffer.get());
    d->meshSrb    = makeSrb(r, d->meshUniformBuffer.get());
    d->gridSrb    = makeSrb(r, d->gridUniformBuffer.get());
    d->axisSrb    = makeSrb(r, d->axisUniformBuffer.get());

    // ── Load shaders ────────────────────────────────────────────────────
    auto lineVs  = loadShader(u":/skigen/plot/line2d.vert.qsb"_s);
    auto lineFs  = loadShader(u":/skigen/plot/line2d.frag.qsb"_s);
    auto pointVs = loadShader(u":/skigen/plot/point2d.vert.qsb"_s);
    auto pointFs = loadShader(u":/skigen/plot/point2d.frag.qsb"_s);
    auto pt3dVs  = loadShader(u":/skigen/plot/point3d.vert.qsb"_s);
    auto pt3dFs  = loadShader(u":/skigen/plot/point3d.frag.qsb"_s);
    auto meshVs  = loadShader(u":/skigen/plot/mesh3d.vert.qsb"_s);
    auto meshFs  = loadShader(u":/skigen/plot/mesh3d.frag.qsb"_s);
    if (!lineVs || !lineFs || !pointVs || !pointFs ||
        !pt3dVs || !pt3dFs || !meshVs  || !meshFs) {
        qWarning("SkigenPlot: shader loading failed");
        return;
    }

    auto* rpDesc = renderTarget()->renderPassDescriptor();

    // ── 2D vertex layout (vec2, stride 8) ───────────────────────────────
    QRhiVertexInputLayout layout2d;
    layout2d.setBindings({QRhiVertexInputBinding(2 * sizeof(float))});
    layout2d.setAttributes({
        QRhiVertexInputAttribute(0, 0, QRhiVertexInputAttribute::Float2, 0)
    });

    // ── 3D point layout (vec3, stride 12) ───────────────────────────────
    QRhiVertexInputLayout layout3dPoint;
    layout3dPoint.setBindings({QRhiVertexInputBinding(3 * sizeof(float))});
    layout3dPoint.setAttributes({
        QRhiVertexInputAttribute(0, 0, QRhiVertexInputAttribute::Float3, 0)
    });

    // ── 3D mesh layout (vec3 pos + vec3 normal, stride 24) ──────────────
    QRhiVertexInputLayout layoutMesh;
    layoutMesh.setBindings({QRhiVertexInputBinding(6 * sizeof(float))});
    layoutMesh.setAttributes({
        QRhiVertexInputAttribute(0, 0, QRhiVertexInputAttribute::Float3, 0),
        QRhiVertexInputAttribute(0, 1, QRhiVertexInputAttribute::Float3,
                                 3 * sizeof(float))
    });

    // ── Line pipeline (LineStrip, no depth) ─────────────────────────────
    d->linePipeline.reset(r->newGraphicsPipeline());
    d->linePipeline->setTopology(QRhiGraphicsPipeline::LineStrip);
    d->linePipeline->setShaderStages({
        {QRhiShaderStage::Vertex, *lineVs},
        {QRhiShaderStage::Fragment, *lineFs}
    });
    d->linePipeline->setVertexInputLayout(layout2d);
    d->linePipeline->setShaderResourceBindings(d->lineSrb.get());
    d->linePipeline->setRenderPassDescriptor(rpDesc);
    d->linePipeline->setSampleCount(sc);
    d->linePipeline->create();

    // ── Grid pipeline (Lines, no depth) ─────────────────────────────────
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
    QRhiGraphicsPipeline::TargetBlend blend;
    blend.enable = true;
    blend.srcColor = QRhiGraphicsPipeline::SrcAlpha;
    blend.dstColor = QRhiGraphicsPipeline::OneMinusSrcAlpha;
    blend.srcAlpha = QRhiGraphicsPipeline::One;
    blend.dstAlpha = QRhiGraphicsPipeline::OneMinusSrcAlpha;
    d->gridPipeline->setTargetBlends({blend});
    d->gridPipeline->create();

    // ── Point pipeline (Points, no depth) ───────────────────────────────
    d->pointPipeline.reset(r->newGraphicsPipeline());
    d->pointPipeline->setTopology(QRhiGraphicsPipeline::Points);
    d->pointPipeline->setShaderStages({
        {QRhiShaderStage::Vertex, *pointVs},
        {QRhiShaderStage::Fragment, *pointFs}
    });
    d->pointPipeline->setVertexInputLayout(layout2d);
    d->pointPipeline->setShaderResourceBindings(d->scatterSrb.get());
    d->pointPipeline->setRenderPassDescriptor(rpDesc);
    d->pointPipeline->setSampleCount(sc);
    d->pointPipeline->create();

    // ── 3D Point pipeline (Points, depth enabled) ───────────────────────
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
    d->point3dPipeline->create();

    // ── Mesh pipeline (Triangles, depth enabled) ────────────────────────
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

    d->pipelineReady = true;
    d->dataDirty = true;
    d->data3dDirty = true;
    d->indexDirty = true;
    d->gridDirty = true;
}

void PlotView::render(QRhiCommandBuffer* cb) {
    if (!d->pipelineReady || d->mode == RenderMode::None)
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
    bool is2D = (d->mode == RenderMode::Line2D || d->mode == RenderMode::Scatter2D);

    // ── Prepare grid (2D only) ──────────────────────────────────────────
    if (is2D && d->gridDirty)
        computeGridVertices();

    // ── Compute MVP ─────────────────────────────────────────────────────
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

    Eigen::Vector4f dataColor = d->userLineColor.value_or(d->theme.seriesColors[0]);
    Eigen::Vector4f bgColor = d->userBgColor.value_or(d->theme.background);

    // ── Upload vertex data ──────────────────────────────────────────────
    if (is2D) {
        auto requiredFloats = static_cast<int>(d->vertices.size());
        if (requiredFloats > d->vertexCapacity) {
            d->vertexCapacity = requiredFloats * 2;
            d->vertexBuffer = makeDynBuf(r, QRhiBuffer::VertexBuffer,
                                          quint32(d->vertexCapacity * sizeof(float)));
            d->dataDirty = true;
        }
        if (d->dataDirty) {
            u->updateDynamicBuffer(d->vertexBuffer.get(), 0,
                                   quint32(d->vertices.size() * sizeof(float)),
                                   d->vertices.data());
            d->dataDirty = false;
        }
    } else {
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
        if (d->mode == RenderMode::Mesh3D && d->indexDirty) {
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
    }

    // ── Upload grid vertex data (2D only) ───────────────────────────────
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

    // ── Upload uniforms ─────────────────────────────────────────────────
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

    switch (d->mode) {
    case RenderMode::Line2D:
        u->updateDynamicBuffer(d->lineUniformBuffer.get(), 0, 64, mvp.data());
        u->updateDynamicBuffer(d->lineUniformBuffer.get(), 64, 16, dataColor.data());
        break;
    case RenderMode::Scatter2D: {
        Eigen::Vector4f params(d->pointSize, 0.f, 0.f, 0.f);
        u->updateDynamicBuffer(d->scatterUniformBuffer.get(), 0, 64, mvp.data());
        u->updateDynamicBuffer(d->scatterUniformBuffer.get(), 64, 16, dataColor.data());
        u->updateDynamicBuffer(d->scatterUniformBuffer.get(), 80, 16, params.data());
        break;
    }
    case RenderMode::PointCloud3D: {
        Eigen::Vector4f params(d->pointSize, 0.f, 0.f, 0.f);
        u->updateDynamicBuffer(d->point3dUniformBuffer.get(), 0, 64, mvp.data());
        u->updateDynamicBuffer(d->point3dUniformBuffer.get(), 64, 16, dataColor.data());
        u->updateDynamicBuffer(d->point3dUniformBuffer.get(), 80, 16, params.data());
        break;
    }
    case RenderMode::Mesh3D: {
        Eigen::Vector4f lightDir = Eigen::Vector4f(1.f, 1.f, 1.f, 0.f).normalized();
        Eigen::Vector4f lightParams(0.3f, 0.7f, 0.f, 0.f);
        u->updateDynamicBuffer(d->meshUniformBuffer.get(), 0, 64, mvp.data());
        u->updateDynamicBuffer(d->meshUniformBuffer.get(), 64, 16, dataColor.data());
        u->updateDynamicBuffer(d->meshUniformBuffer.get(), 80, 16, lightDir.data());
        u->updateDynamicBuffer(d->meshUniformBuffer.get(), 96, 16, lightParams.data());
        break;
    }
    default:
        break;
    }

    // ── Begin render pass ───────────────────────────────────────────────
    cb->beginPass(rt,
                  QColor::fromRgbF(bgColor.x(), bgColor.y(), bgColor.z(), bgColor.w()),
                  {1.0f, 0}, u);

    cb->setViewport({0, 0,
                     static_cast<float>(sz.width()),
                     static_cast<float>(sz.height())});

    // ── Draw grid (2D only) ─────────────────────────────────────────────
    if (is2D && d->showGrid && d->gridVertexCount > 0) {
        cb->setGraphicsPipeline(d->gridPipeline.get());
        cb->setShaderResources(d->gridSrb.get());
        const QRhiCommandBuffer::VertexInput gridBuf(d->gridVertexBuffer.get(), 0);
        cb->setVertexInput(0, 1, &gridBuf);
        cb->draw(d->gridVertexCount);
    }

    // ── Draw axes + ticks (2D only) ─────────────────────────────────────
    if (is2D && d->showAxes && d->axisVertexCount > 0) {
        cb->setGraphicsPipeline(d->gridPipeline.get());
        cb->setShaderResources(d->axisSrb.get());
        quint32 axisOffset = quint32(d->gridVertices.size() * sizeof(float));
        const QRhiCommandBuffer::VertexInput axisBuf(d->gridVertexBuffer.get(), axisOffset);
        cb->setVertexInput(0, 1, &axisBuf);
        cb->draw(d->axisVertexCount);
    }

    // ── Draw data ───────────────────────────────────────────────────────
    switch (d->mode) {
    case RenderMode::Line2D: {
        cb->setGraphicsPipeline(d->linePipeline.get());
        cb->setShaderResources(d->lineSrb.get());
        const QRhiCommandBuffer::VertexInput vbuf(d->vertexBuffer.get(), 0);
        cb->setVertexInput(0, 1, &vbuf);
        cb->draw(d->vertexCount);
        break;
    }
    case RenderMode::Scatter2D: {
        cb->setGraphicsPipeline(d->pointPipeline.get());
        cb->setShaderResources(d->scatterSrb.get());
        const QRhiCommandBuffer::VertexInput vbuf(d->vertexBuffer.get(), 0);
        cb->setVertexInput(0, 1, &vbuf);
        cb->draw(d->vertexCount);
        break;
    }
    case RenderMode::PointCloud3D: {
        cb->setGraphicsPipeline(d->point3dPipeline.get());
        cb->setShaderResources(d->point3dSrb.get());
        const QRhiCommandBuffer::VertexInput vbuf(d->vertex3dBuffer.get(), 0);
        cb->setVertexInput(0, 1, &vbuf);
        cb->draw(d->vertex3dCount);
        break;
    }
    case RenderMode::Mesh3D: {
        cb->setGraphicsPipeline(d->meshPipeline.get());
        cb->setShaderResources(d->meshSrb.get());
        const QRhiCommandBuffer::VertexInput vbuf(d->vertex3dBuffer.get(), 0);
        cb->setVertexInput(0, 1, &vbuf, d->indexBuffer.get(), 0,
                           QRhiCommandBuffer::IndexUInt32);
        cb->drawIndexed(d->indexCount);
        break;
    }
    default:
        break;
    }

    cb->endPass();
}

// ── PNG export ──────────────────────────────────────────────────────────

auto PlotView::savePng(const QString& path, int width, int height) -> bool {
    auto* r = rhi();
    if (!r || !d->pipelineReady || d->mode == RenderMode::None)
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

    d->dataDirty = true;
    d->data3dDirty = true;
    d->indexDirty = true;
    d->gridDirty = true;

    renderToTarget(cb, rt.get(), u, sz);

    QRhiReadbackResult readResult;
    bool readComplete = false;
    readResult.completed = [&readComplete] { readComplete = true; };

    auto* readBatch = r->nextResourceUpdateBatch();
    readBatch->readBackTexture(QRhiReadbackDescription(tex.get()), &readResult);
    cb->beginPass(rt.get(), Qt::black, {1.0f, 0}, nullptr);
    cb->endPass(readBatch);

    r->endOffscreenFrame();

    if (!readComplete || readResult.data.isEmpty())
        return false;

    QImage img(reinterpret_cast<const uchar*>(readResult.data.constData()),
               width, height, QImage::Format_RGBA8888);
    return img.save(path, "PNG");
}

} // namespace Skigen::Plot
