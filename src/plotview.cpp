#include <skigen/plot/plotview.h>

#include <rhi/qrhi.h>

#include <QFile>

#include <algorithm>
#include <expected>
#include <vector>

namespace Skigen::Plot {

// ── Shader loading ───────────────────────────────────────────────────────

static auto loadShader(const QString& path)
    -> std::expected<QShader, QString>
{
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly))
        return std::unexpected(u"Failed to open shader: "_qs + path);
    auto shader = QShader::fromSerialized(f.readAll());
    if (!shader.isValid())
        return std::unexpected(u"Invalid shader: "_qs + path);
    return shader;
}

// ── PIMPL ────────────────────────────────────────────────────────────────

struct PlotView::Impl {
    // GPU resources
    std::unique_ptr<QRhiBuffer> vertexBuffer;
    std::unique_ptr<QRhiBuffer> uniformBuffer;
    std::unique_ptr<QRhiShaderResourceBindings> srb;
    std::unique_ptr<QRhiGraphicsPipeline> linePipeline;

    // Vertex data (interleaved x, y pairs)
    std::vector<float> vertices;
    int vertexCount = 0;
    int vertexCapacity = 0;
    bool dataDirty = false;
    bool pipelineReady = false;

    // Appearance
    Eigen::Vector4f lineColor{0.2f, 0.6f, 1.0f, 1.0f};
    Eigen::Vector4f bgColor  {0.05f, 0.05f, 0.08f, 1.0f};

    // Spatial state
    BoundingBox2D bounds;
    Camera3D camera;
};

// ── Construction ─────────────────────────────────────────────────────────

PlotView::PlotView(QWidget* parent)
    : QRhiWidget(parent)
    , d(std::make_unique<Impl>())
{
}

PlotView::~PlotView() = default;

// ── Data setters ─────────────────────────────────────────────────────────

void PlotView::setLineData(std::span<const float> x,
                           std::span<const float> y) {
    auto n = static_cast<int>(std::min(x.size(), y.size()));
    d->vertices.resize(static_cast<std::size_t>(n) * 2);
    for (int i = 0; i < n; ++i) {
        d->vertices[static_cast<std::size_t>(i) * 2]     = x[static_cast<std::size_t>(i)];
        d->vertices[static_cast<std::size_t>(i) * 2 + 1] = y[static_cast<std::size_t>(i)];
    }
    d->vertexCount = n;
    d->bounds = BoundingBox2D::fromXY(
        Eigen::Map<const Eigen::VectorXf>(x.data(), static_cast<Eigen::Index>(x.size())),
        Eigen::Map<const Eigen::VectorXf>(y.data(), static_cast<Eigen::Index>(y.size())));
    d->dataDirty = true;
    update();
}

void PlotView::setScatterData(std::span<const float> x,
                              std::span<const float> y) {
    setLineData(x, y);
}

void PlotView::setPointCloudData(std::span<const float> /*data*/,
                                 int /*vertexCount*/) {
}

void PlotView::setMeshData(std::span<const float> /*verts*/, int /*vertexCount*/,
                           std::span<const uint32_t> /*idx*/, int /*triangleCount*/) {
}

// ── Appearance ───────────────────────────────────────────────────────────

void PlotView::setLineColor(const Eigen::Vector4f& rgba) {
    d->lineColor = rgba;
    update();
}

void PlotView::setBackgroundColor(const Eigen::Vector4f& rgba) {
    d->bgColor = rgba;
    update();
}

void PlotView::setCamera(const Camera3D& camera) {
    d->camera = camera;
    update();
}

auto PlotView::camera() const -> const Camera3D& {
    return d->camera;
}

// ── QRhiWidget overrides ─────────────────────────────────────────────────

void PlotView::initialize(QRhiCommandBuffer* /*cb*/) {
    auto* r = rhi();
    d->pipelineReady = false;

    // Vertex buffer — dynamic for real-time updates
    static constexpr int kInitialFloats = 128 * 1024;
    d->vertexCapacity = kInitialFloats;
    d->vertexBuffer.reset(
        r->newBuffer(QRhiBuffer::Dynamic,
                     QRhiBuffer::VertexBuffer,
                     quint32(kInitialFloats * sizeof(float))));
    d->vertexBuffer->create();

    // Uniform buffer: mat4 (64 bytes) + vec4 (16 bytes) = 80 bytes
    d->uniformBuffer.reset(
        r->newBuffer(QRhiBuffer::Dynamic,
                     QRhiBuffer::UniformBuffer, 80));
    d->uniformBuffer->create();

    // Shader resource bindings
    d->srb.reset(r->newShaderResourceBindings());
    d->srb->setBindings({
        QRhiShaderResourceBinding::uniformBuffer(
            0,
            QRhiShaderResourceBinding::VertexStage
                | QRhiShaderResourceBinding::FragmentStage,
            d->uniformBuffer.get())
    });
    d->srb->create();

    // Load compiled shaders from Qt resources
    auto vs = loadShader(u":/skigen/plot/line2d.vert.qsb"_qs);
    auto fs = loadShader(u":/skigen/plot/line2d.frag.qsb"_qs);
    if (!vs || !fs) {
        qWarning("SkigenPlot: shader loading failed");
        return;
    }

    // Vertex input: vec2 position (stride = 8 bytes)
    QRhiVertexInputLayout inputLayout;
    inputLayout.setBindings({QRhiVertexInputBinding(2 * sizeof(float))});
    inputLayout.setAttributes({
        QRhiVertexInputAttribute(0, 0,
                                 QRhiVertexInputAttribute::Float2, 0)
    });

    // Graphics pipeline (line strip)
    d->linePipeline.reset(r->newGraphicsPipeline());
    d->linePipeline->setTopology(QRhiGraphicsPipeline::LineStrip);
    d->linePipeline->setShaderStages({
        {QRhiShaderStage::Vertex,   *vs},
        {QRhiShaderStage::Fragment, *fs}
    });
    d->linePipeline->setVertexInputLayout(inputLayout);
    d->linePipeline->setShaderResourceBindings(d->srb.get());
    d->linePipeline->setRenderPassDescriptor(
        renderTarget()->renderPassDescriptor());
    d->linePipeline->create();

    d->pipelineReady = true;
    d->dataDirty = true;
}

void PlotView::render(QRhiCommandBuffer* cb) {
    if (!d->pipelineReady || d->vertexCount == 0)
        return;

    auto* r = rhi();
    const QSize sz = renderTarget()->pixelSize();

    // Grow vertex buffer if data exceeds capacity
    auto requiredFloats = static_cast<int>(d->vertices.size());
    if (requiredFloats > d->vertexCapacity) {
        d->vertexCapacity = requiredFloats * 2;
        d->vertexBuffer.reset(
            r->newBuffer(QRhiBuffer::Dynamic,
                         QRhiBuffer::VertexBuffer,
                         quint32(d->vertexCapacity * sizeof(float))));
        d->vertexBuffer->create();
        d->dataDirty = true;
    }

    // Compute MVP (ortho for 2D, perspective for 3D)
    Eigen::Matrix4f mvp = orthoProjection(d->bounds);

    // Apply QRhi clip-space correction (handles Vulkan Y-flip, depth range)
    QMatrix4x4 correction = r->clipSpaceCorrMatrix();
    Eigen::Matrix4f corr;
    for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 4; ++col)
            corr(row, col) = correction(row, col);
    mvp = corr * mvp;

    // Resource update batch
    QRhiResourceUpdateBatch* u = r->nextResourceUpdateBatch();

    if (d->dataDirty) {
        u->updateDynamicBuffer(
            d->vertexBuffer.get(), 0,
            quint32(d->vertices.size() * sizeof(float)),
            d->vertices.data());
        d->dataDirty = false;
    }

    // Upload uniforms: mat4 mvp at offset 0, vec4 color at offset 64
    u->updateDynamicBuffer(d->uniformBuffer.get(), 0, 64, mvp.data());
    u->updateDynamicBuffer(d->uniformBuffer.get(), 64, 16,
                           d->lineColor.data());

    // Begin render pass
    const auto& bg = d->bgColor;
    cb->beginPass(renderTarget(),
                  QColor::fromRgbF(bg.x(), bg.y(), bg.z(), bg.w()),
                  {1.0f, 0}, u);

    cb->setGraphicsPipeline(d->linePipeline.get());
    cb->setViewport({0, 0,
                     static_cast<float>(sz.width()),
                     static_cast<float>(sz.height())});
    cb->setShaderResources(d->srb.get());

    const QRhiCommandBuffer::VertexInput vbufBinding(
        d->vertexBuffer.get(), 0);
    cb->setVertexInput(0, 1, &vbufBinding);
    cb->draw(d->vertexCount);

    cb->endPass();
}

} // namespace Skigen::Plot
