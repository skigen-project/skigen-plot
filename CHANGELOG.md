# Changelog

All notable changes to SkigenPlot will be documented in this file.

## [1.0.0] - Unreleased

### Highlights

- Hardware-accelerated 2D and 3D plotting through Qt RHI on Vulkan, Metal,
  and Direct3D.
- Native Eigen expression input for line, scatter, statistical, filled,
  field, point-cloud, and mesh plots.
- Fixed-capacity scrolling telemetry with automatic and explicit sample
  coordinates.
- Stable line and scatter handles with in-place data/style updates, visibility
  control, and independent removal.
- Histogram, bar, horizontal-bar, filled-area, and step creation return stable
  handles with the same style, visibility, and removal operations.
- Theme-aware legends for labelled 2D series with automatic or fixed-corner
  placement.
- Dark, light, paper, and matplotlib-style themes with configurable palettes.
- Offscreen PNG export and themed gallery generation.

### Plot Types

- Line, scatter, step, stem, error-bar, and filled-area plots.
- Histogram, vertical and horizontal bars, box plots, and violin plots.
- Heatmaps, contour lines, filled contours, quiver, hexbin, and pie charts.
- Depth-tested 3D point clouds and surface meshes.

### Packaging

- CMake package export as `SkigenPlot::SkigenPlot`.
- Buildable source archives with SHA-256 checksums for tagged releases.

### Known Limitations

- Performance targets still require recorded measurements across the Vulkan,
  Metal, and Direct3D release runners.
- Plot data is retained in float vertex storage; non-float Eigen expressions
  are converted when submitted.