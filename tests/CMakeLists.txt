cmake_minimum_required(VERSION 3.20)

# ---------------------------------------------------------------------------
# Core math tests (headless — no Qt, no GPU)
# ---------------------------------------------------------------------------
add_executable(skigenplot_test_core test_core.cpp)
target_link_libraries(skigenplot_test_core PRIVATE SkigenPlot::SkigenPlot)
target_compile_options(skigenplot_test_core PRIVATE ${SKIGENPLOT_WARNING_FLAGS})
add_test(NAME skigenplot_test_core COMMAND skigenplot_test_core)

# ---------------------------------------------------------------------------
# PlotView data-layer tests (headless — core.h math only)
# ---------------------------------------------------------------------------
add_executable(skigenplot_test_plotview_data test_plotview_data.cpp)
target_link_libraries(skigenplot_test_plotview_data PRIVATE SkigenPlot::SkigenPlot)
target_compile_options(skigenplot_test_plotview_data PRIVATE ${SKIGENPLOT_WARNING_FLAGS})
add_test(NAME skigenplot_test_plotview_data COMMAND skigenplot_test_plotview_data)

# ---------------------------------------------------------------------------
# PlotView streaming API tests (offscreen Qt widget, no visible window)
# ---------------------------------------------------------------------------
add_executable(skigenplot_test_telemetry test_telemetry.cpp)
target_link_libraries(skigenplot_test_telemetry PRIVATE SkigenPlot::SkigenPlot)
target_compile_options(skigenplot_test_telemetry PRIVATE ${SKIGENPLOT_WARNING_FLAGS})
add_test(NAME skigenplot_test_telemetry COMMAND skigenplot_test_telemetry)
set_tests_properties(skigenplot_test_telemetry PROPERTIES
	ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
)

# ---------------------------------------------------------------------------
# Stable series handle and mutation tests (offscreen Qt widget)
# ---------------------------------------------------------------------------
add_executable(skigenplot_test_series_lifecycle test_series_lifecycle.cpp)
target_link_libraries(skigenplot_test_series_lifecycle PRIVATE SkigenPlot::SkigenPlot)
target_compile_options(skigenplot_test_series_lifecycle PRIVATE ${SKIGENPLOT_WARNING_FLAGS})
add_test(NAME skigenplot_test_series_lifecycle COMMAND skigenplot_test_series_lifecycle)
set_tests_properties(skigenplot_test_series_lifecycle PROPERTIES
	ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
)
