# SkigenPlot

**Hardware-accelerated C++ plotting for scientific and ML visualization.**

SkigenPlot is the plotting library for the [Skigen](https://github.com/skigen-project/skigen) ecosystem — a *matplotlib for C++* built on Vulkan, Metal, and Direct3D 12 from day one. It provides zero-friction Eigen integration, real-time 2D/3D rendering, and a clean API designed for both interactive research and production-grade visualization pipelines.

## Features

- **Native Eigen expressions** — pass any `Eigen::MatrixBase<Derived>` directly; no manual conversion or memory copies for contiguous data.
- **GPU-accelerated rendering** — all drawing through Qt's RHI abstraction (`QRhiWidget`), supporting Vulkan, Metal, and Direct3D 12.
- **Real-time capable** — dynamic vertex buffers sustain 60 Hz+ at 1M+ data points; designed for live telemetry, EEG streams, and simulation output.
- **LGPLv3/MIT safe** — no QPainter, no Qt Canvas; the library is fully MIT-licensed.
- **Modern C++23** — `std::span`, `std::expected`, `std::ranges`, `std::numbers`.
- **Cross-platform** — Linux, macOS, Windows, with GPU backend selected automatically.

## Quick Example

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

## Supported Plot Types

| Type | Dimension | Input |
|------|-----------|-------|
| Line plot | 2D | `VectorXf` × 2 (x, y) |
| Scatter plot | 2D | `VectorXf` × 2 (x, y) |
| Scrolling telemetry | 2D | Streaming `VectorXf` |
| Point cloud | 3D | `MatrixXf` (N × 3) |
| Surface mesh | 3D | Vertices (N × 3) + indices (M × 3) |

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| [Eigen 3](https://eigen.tuxfamily.org/) | ≥ 3.4 | Matrix and vector types |
| [Qt 6](https://www.qt.io/) | ≥ 6.7 | Widgets, QRhi rendering, shader compilation |
| C++ compiler | C++23 | GCC ≥ 13, Clang ≥ 17, MSVC ≥ 2022 17.8 |
| CMake | ≥ 3.20 | Build system |

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

## Project Structure

```
include/skigen/plot/
  core.h          Math: BoundingBox2D/3D, Camera3D, projections, normalization
  plotview.h      Widget: PlotView with plot(), scatter(), pointCloud(), mesh()
  export.h        Shared library export macros

src/
  core.cpp        Projection and camera matrix implementations
  plotview.cpp     QRhi pipeline, shader loading, GPU buffer management
  shaders/        GLSL 440 vertex/fragment shaders (compiled to .qsb)

tests/            Headless math tests (zero-dependency harness)
examples/         Line, scatter, point cloud, and mesh examples
doc/              Doxygen config, API registry, guide pages (MDX)
```

## Architecture

All rendering passes through `QRhiWidget`, which maps directly to the platform's native GPU API. Public headers forward-declare all QRhi types; the `<rhi/qrhi.h>` include is confined to `.cpp` files, keeping compile times low and the public API clean.

```
User Code ──▶ PlotView API ──▶ Vertex Buffer Upload ──▶ QRhi Pipeline ──▶ Vulkan / Metal / D3D12
                 ▲
         core.h: BoundingBox, Camera3D, Projections
```

## Documentation

SkigenPlot documentation is integrated into the unified [Skigen Project website](https://skigen-project.github.io) under the **Plot** navigation entry.

## License

[MIT](LICENSE)
