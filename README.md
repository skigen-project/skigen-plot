# SkigenPlot

**Hardware-accelerated C++ plotting for scientific and ML visualization.**

[![Release](https://github.com/skigen-project/skigen-plot/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/skigen-project/skigen-plot/actions/workflows/main.yml)
[![Staging](https://github.com/skigen-project/skigen-plot/actions/workflows/staging.yml/badge.svg?branch=staging)](https://github.com/skigen-project/skigen-plot/actions/workflows/staging.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://skigen-project.github.io/skigen-plot/)
[![Docs Dev](https://img.shields.io/badge/docs-dev-orange.svg)](https://skigen-project.github.io/skigen-plot/dev/)

## Overview

SkigenPlot is the visualization component of the [Skigen](https://github.com/skigen-project/skigen) ecosystem. It renders 2D and 3D scientific data through Qt's RHI abstraction layer, mapping directly to Vulkan, Metal, or Direct3D 12 without intermediate software rasterization.

The library accepts Eigen expression templates natively and retains plot data in GPU-ready float storage. Dynamic vertex buffers and a bounded telemetry API support real-time sensor streams, EEG displays, and simulation output.

## Example

```cpp
#include <skigen/plot/plotview.h>
#include <Eigen/Core>
#include <QApplication>
#include <numbers>

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);

    Eigen::VectorXf x = Eigen::VectorXf::LinSpaced(500, 0.f,
        4.f * std::numbers::pi_v<float>);
    Eigen::VectorXf y = x.array().sin();

    Skigen::Plot::PlotView view;
    auto series = view.plot(
      x, y, {.color = Eigen::Vector4f{0.2f, 0.8f, 0.4f, 1.0f},
             .label = "sin(x)"});
    view.setSeriesVisible(series, true);
    view.setLegendVisible(true);
    view.resize(800, 500);
    view.show();

    return app.exec();
}
```

`plot()`, `scatter()`, `hist()`, `bar()`, `barh()`, `fillBetween()`, `step()`,
`errorbar()`, `stem()`, `boxplot()`, `violinplot()`, `quiver()`, `pie()`,
`imshow()`, `contour()`, `contourf()`, and `hexbin()` return stable
`SeriesHandle` values.
Use `updateSeriesData()`, `setSeriesStyle()`, `setSeriesVisible()`, and
`removeSeries()` to modify one series without clearing the rest of the view.
Field-plot data is immutable in v1, so `updateSeriesData()` returns `false`
for those handles; style, visibility, and removal remain supported.
Visible labelled series can be shown in a theme-aware legend with automatic
or explicit corner placement.

Vector inputs are truncated to the shortest required vector. Non-finite
samples are discarded by handle-returning 2D plots, while filled areas omit
segments with invalid endpoints. Histograms ignore non-finite observations.
Creation returns an invalid handle when no usable data remains.
`computeHistogram()` exposes the same finite filtering, uniform bin edges,
counts, and probability-density normalization used by `hist()` for numerical
inspection without constructing a widget.
Point clouds and meshes require finite `N x 3` vertices. Mesh faces require an
`M x 3` index matrix whose values refer to existing vertices; invalid 3D input
leaves the current view unchanged.
Heatmaps and contours derive color ranges from finite cells and omit invalid
cells. An all-invalid matrix leaves the current view unchanged.
Hexbin omits non-finite coordinate pairs. Pie charts omit non-finite and
non-positive weights; an all-invalid input leaves the view unchanged.
Box and violin plots omit non-finite samples and empty groups while preserving
the positions of valid groups. Their component geometry, and all wedges of a
pie chart, share one handle for styling, visibility, and removal.

Use `setXLimits()` and `setYLimits()` for explicit view bounds; endpoint order
is preserved, so reversed endpoints invert an axis. `resetXLimits()`,
`resetYLimits()`, or `resetAxisLimits()` restore automatic limits.
Use `setXScale(AxisScale::Log10)` and `setYScale(AxisScale::Log10)` independently
for base-10 logarithmic axes. Limits remain expressed in data units; logarithmic
limits must be positive, and non-positive samples are omitted on logarithmic axes.
Scatter series support circle, square, triangle, plus, and cross markers.
Line series support solid, dashed, dotted, and dash-dot strokes. Line width and
dash spacing are measured in screen pixels and remain stable while zooming.
Non-finite line samples create visible gaps instead of connecting adjacent runs.

## Plot Types

| Type | Dimension | Input |
|------|-----------|-------|
| Line plot | 2D | `VectorXf` x, y |
| Scatter plot | 2D | `VectorXf` x, y |
| Scrolling telemetry | 2D | Bounded sample stream |
| Histogram / bar chart | 2D | `VectorXf` |
| Error bars / filled area / step / stem | 2D | `VectorXf` series |
| Box / violin plot | 2D | Groups of `VectorXf` |
| Heatmap / contour | 2D | `MatrixXf` |
| Quiver / hexbin / pie | 2D | Vector or matrix data |
| Point cloud | 3D | `MatrixXf` (N x 3) |
| Surface mesh | 3D | Vertices (N x 3) + indices (M x 3) |

## Requirements

| Dependency | Version |
|------------|---------|
| C++ | C++23 (GCC >= 13, Clang >= 17, MSVC >= 19.38) |
| [Eigen 3](https://eigen.tuxfamily.org/) | >= 3.4 |
| [Qt 6](https://www.qt.io/) | >= 6.7 (Core, Gui, Widgets, ShaderTools) |
| CMake | >= 3.20 |

## Building

```bash
git clone https://github.com/skigen-project/skigen-plot.git
cd skigen-plot
cmake -B build -DSKIGENPLOT_BUILD_TESTS=ON -DSKIGENPLOT_BUILD_EXAMPLES=ON
cmake --build build
ctest --test-dir build
```

Eigen is discovered from the system or from a sibling `../eigen` directory.

## CMake Integration

```cmake
# As a subdirectory
add_subdirectory(skigen-plot)
target_link_libraries(my_app PRIVATE SkigenPlot::SkigenPlot)

# As an installed package
find_package(SkigenPlot REQUIRED)
target_link_libraries(my_app PRIVATE SkigenPlot::SkigenPlot)
```

## Architecture

All rendering passes through `QRhiWidget`, which maps directly to the platform's native GPU API. Public headers forward-declare QRhi types; the `<rhi/qrhi.h>` include is confined to `.cpp` files.

```
User Code -> PlotView API -> Vertex Buffer Upload -> QRhi Pipeline -> Vulkan / Metal / D3D12
                 ^
         core.h: BoundingBox, Camera3D, Projections
```

## Project Structure

```
include/skigen/plot/
  core.h          BoundingBox2D/3D, Camera3D, projections, normalization
  plotview.h      PlotView widget: plot(), scatter(), pointCloud(), mesh()
  export.h        Shared library export macros

src/
  core.cpp        Projection and camera matrix implementations
  plotview.cpp    QRhi pipeline, shader loading, GPU buffer management
  shaders/        GLSL 440 vertex/fragment shaders (compiled to .qsb)

tests/            Headless math tests
examples/         Line, scatter, point cloud, and mesh examples
doc/              API registry, guide pages
```

## License

[MIT](LICENSE)
