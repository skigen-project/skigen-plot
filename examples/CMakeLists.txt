cmake_minimum_required(VERSION 3.20)

# ---------------------------------------------------------------------------
# Helper: add an example target with standard flags
# ---------------------------------------------------------------------------
function(skigenplot_add_example name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE SkigenPlot::SkigenPlot)
    target_compile_options(${name} PRIVATE ${SKIGENPLOT_WARNING_FLAGS})
endfunction()

# ---------------------------------------------------------------------------
# 2D line examples
# ---------------------------------------------------------------------------
skigenplot_add_example(ex_line_plot       line/line_plot.cpp)

# ---------------------------------------------------------------------------
# 2D scatter examples
# ---------------------------------------------------------------------------
skigenplot_add_example(ex_scatter_plot    scatter/scatter_plot.cpp)

# ---------------------------------------------------------------------------
# 3D point cloud examples
# ---------------------------------------------------------------------------
skigenplot_add_example(ex_point_cloud     pointcloud/point_cloud.cpp)

# ---------------------------------------------------------------------------
# 3D mesh examples
# ---------------------------------------------------------------------------
skigenplot_add_example(ex_mesh_view       mesh/mesh_view.cpp)
