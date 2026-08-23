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
- Histogram, bar, horizontal-bar, filled-area, step, error-bar, and quiver
  creation return stable handles with the same style, visibility, and removal
  operations.
- Composite stem plots return one stable handle for updating, styling, hiding,
  or removing their stems and markers together.
- Heatmap, contour, filled-contour, and hexbin plots return independent stable
  handles for styling, visibility, and removal, allowing field plots to compose
  without replacing earlier field geometry.
- Theme-aware legends for labelled 2D series with automatic or fixed-corner
  placement.
- Independent x/y axis limits with getters, autoscale reset, validation, and
  intentional axis inversion through reversed endpoints.
- Data-relative autoscaling with restrained padding for ordinary series and
  exact sticky edges for fields, keeping axes aligned with rendered content.
- Independent linear and base-10 logarithmic x/y scales with decade ticks,
  mixed-scale rendering, transformed pan/zoom, and positive-domain validation.
- Circle, square, triangle, plus, and cross scatter markers, including hollow
  rendering for closed shapes.
- Portable screen-space triangle markers for 2D scatter and depth-tested 3D
  point clouds, preserving requested pixel sizes on Direct3D, Vulkan, and Metal.
- Refined gallery composition with high-contrast scatter clusters, legends,
  publication-style spines, clearer 3D framing, and a shaded damped-wave
  surface mesh.
- Deterministic finite-data filtering for line, scatter, series updates, and
  histograms; creation fails predictably when no usable values remain.
- Public `computeHistogram()` data-layer results with reference tests for
  finite-count conservation, unit-integral density, and constant ranges.
- Finite-sample filtering for bars, filled areas, steps, error bars, stems,
  and quiver fields, including grouped stem updates.
- Shape, finite-coordinate, and index-range validation for 3D point clouds
  and triangle meshes.
- NaN/Inf masking and finite color-range calculation for heatmaps and filled
  or line contours.
- Finite coordinate filtering for hexbin and safe positive-weight
  normalization for pie charts.
- Finite sample filtering and all-invalid no-op behavior for box and violin
  plots.
- Documented and tested automatic/default lower bounds for histogram bins,
  contour levels, and hexbin grid size.
- Portable triangle-based line strokes with effective per-series width, solid,
  dashed, dotted, and dash-dot styles, and preserved non-finite gaps.
- Grouped stable handles for box plots, violin plots, and pie charts, with
  composite style, visibility, and removal operations.
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