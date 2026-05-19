# SkigenPlot

**Hardware-accelerated C++ plotting for scientific and ML visualization.**

[![Release](https://github.com/skigen-project/skigen-plot/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/skigen-project/skigen-plot/actions/workflows/main.yml)
[![Staging](https://github.com/skigen-project/skigen-plot/actions/workflows/staging.yml/badge.svg?branch=staging)](https://github.com/skigen-project/skigen-plot/actions/workflows/staging.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

SkigenPlot is the visualization component of the [Skigen](https://github.com/skigen-project/skigen) ecosystem. It renders 2D and 3D scientific data through Qt's RHI abstraction layer, mapping directly to Vulkan, Metal, or Direct3D 12 without intermediate software rasterization.

The library accepts Eigen expression templates natively — any `Eigen::MatrixBase<Derived>` is a valid input with zero-copy semantics for contiguous data. Dynamic vertex buffers sustain 60 Hz+ update rates at 1M+ data points, suitable for real-time telemetry, EEG streams, and simulation output.

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
    view.plot(x, y);
    view.setLineColor({0.2f, 0.8f, 0.4f, 1.0f});
    view.resize(800, 500);
    view.show();

    return app.exec();
}
```

## Plot Types

| Type | Dimension | Input |
|------|-----------|-------|
| Line plot | 2D | `VectorXf` x, y |
| Scatter plot | 2D | `VectorXf` x, y |
| Scrolling telemetry | 2D | Streaming `VectorXf` |
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

## Documentation

Integrated into the [Skigen Project website](https://skigen-project.github.io) under the **Plot** navigation entry.

## License

[MIT](LICENSE)
