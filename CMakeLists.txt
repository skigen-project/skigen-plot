cmake_minimum_required(VERSION 3.20)
project(SkigenPlot
    VERSION 1.0.0
    LANGUAGES CXX
    DESCRIPTION "Hardware-accelerated C++ plotting library for scientific and ML visualization"
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_AUTOMOC ON)

# ---------------------------------------------------------------------------
# Eigen dependency (same fallback strategy as skigen)
# ---------------------------------------------------------------------------
if(NOT TARGET Eigen3::Eigen)
    find_package(Eigen3 3.4 QUIET NO_MODULE)
    if(NOT Eigen3_FOUND)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/eigen/Eigen/Core")
            set(_eigen_dir "${CMAKE_CURRENT_SOURCE_DIR}/eigen")
        elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../eigen/Eigen/Core")
            set(_eigen_dir "${CMAKE_CURRENT_SOURCE_DIR}/../eigen")
        endif()
        if(_eigen_dir)
            add_library(Eigen3_source INTERFACE)
            target_include_directories(Eigen3_source INTERFACE "${_eigen_dir}")
            add_library(Eigen3::Eigen ALIAS Eigen3_source)
            message(STATUS "Using Eigen from source tree: ${_eigen_dir}")
        else()
            message(FATAL_ERROR
                "Eigen3 not found. Install Eigen3 or place the Eigen source tree "
                "at ./eigen or ../eigen")
        endif()
    endif()
endif()

# ---------------------------------------------------------------------------
# Qt 6 dependencies
# ---------------------------------------------------------------------------
find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets ShaderTools)
find_package(Qt6 QUIET COMPONENTS GuiPrivate)

# ---------------------------------------------------------------------------
# Library target
# ---------------------------------------------------------------------------
add_library(SkigenPlot
    src/core.cpp
    src/plotview.cpp
    src/overlay.cpp
    src/overlay.h
    src/figure.cpp
    include/skigen/plot/plotview.h
    include/skigen/plot/figure.h
)
add_library(SkigenPlot::SkigenPlot ALIAS SkigenPlot)

target_include_directories(SkigenPlot
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(SkigenPlot
    PUBLIC
        Eigen3::Eigen
        Qt6::Widgets
    PRIVATE
        Qt6::GuiPrivate
)

target_compile_features(SkigenPlot PUBLIC cxx_std_23)

target_compile_definitions(SkigenPlot PRIVATE SKIGENPLOT_BUILDING_LIBRARY)

# For static builds, SKIGENPLOT_EXPORT must be empty (no dllimport/dllexport)
get_target_property(_skigenplot_type SkigenPlot TYPE)
if(_skigenplot_type STREQUAL "STATIC_LIBRARY")
    target_compile_definitions(SkigenPlot PUBLIC SKIGENPLOT_STATIC)
endif()

# ---------------------------------------------------------------------------
# Shader compilation (GLSL → .qsb via Qt Shader Tools)
# ---------------------------------------------------------------------------
qt_add_shaders(SkigenPlot "shaders"
    PREFIX "/skigen/plot"
    BASE "src/shaders"
    FILES
        src/shaders/line2d.vert
        src/shaders/line2d.frag
        src/shaders/edge3d.vert
        src/shaders/point2d.vert
        src/shaders/point2d.frag
        src/shaders/point3d.vert
        src/shaders/point3d.frag
        src/shaders/mesh3d.vert
        src/shaders/mesh3d.frag
)

# ---------------------------------------------------------------------------
# Strict warnings
# ---------------------------------------------------------------------------
set(SKIGENPLOT_WARNING_FLAGS "")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    set(SKIGENPLOT_WARNING_FLAGS -Wall -Wextra -Wpedantic)
elseif(MSVC)
    set(SKIGENPLOT_WARNING_FLAGS /W4 /Zc:preprocessor)
endif()

target_compile_options(SkigenPlot PRIVATE ${SKIGENPLOT_WARNING_FLAGS})

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(SKIGENPLOT_BUILD_TESTS    "Build unit tests"      ON)
option(SKIGENPLOT_BUILD_EXAMPLES "Build example programs" OFF)

# ---------------------------------------------------------------------------
# Sub-directories
# ---------------------------------------------------------------------------
if(SKIGENPLOT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(SKIGENPLOT_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(DIRECTORY include/skigen
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT Devel
)

# CMake package export — only when dependencies are exportable
# (source-tree Eigen3_source and qt_add_shaders resource targets are not)
if(Eigen3_FOUND)
    # Include the auto-generated shader resource target in the export set
    install(TARGETS SkigenPlot SkigenPlot_resources_1
        EXPORT SkigenPlotTargets
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    set(CMAKEPACKAGE_INSTALL_DIR
        "${CMAKE_INSTALL_DATADIR}/SkigenPlot/cmake"
        CACHE PATH "CMake package config install directory")

    if(IS_ABSOLUTE "${CMAKEPACKAGE_INSTALL_DIR}")
        file(RELATIVE_PATH CMAKEPACKAGE_INSTALL_DIR
             "${CMAKE_INSTALL_PREFIX}" "${CMAKEPACKAGE_INSTALL_DIR}")
    endif()

    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/SkigenPlotConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/SkigenPlotConfig.cmake"
        INSTALL_DESTINATION ${CMAKEPACKAGE_INSTALL_DIR}
    )

    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/SkigenPlotConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )

    install(EXPORT SkigenPlotTargets
        FILE SkigenPlotTargets.cmake
        NAMESPACE SkigenPlot::
        DESTINATION ${CMAKEPACKAGE_INSTALL_DIR}
    )

    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/SkigenPlotConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/SkigenPlotConfigVersion.cmake"
        DESTINATION ${CMAKEPACKAGE_INSTALL_DIR}
    )

    export(EXPORT SkigenPlotTargets
        FILE "${CMAKE_CURRENT_BINARY_DIR}/SkigenPlotTargets.cmake"
        NAMESPACE SkigenPlot::
    )
else()
    install(TARGETS SkigenPlot
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
    message(STATUS "Eigen from source tree — CMake package export disabled. "
                   "Install Eigen3 system-wide for find_package(SkigenPlot) support.")
endif()

message(STATUS "")
message(STATUS "SkigenPlot v${PROJECT_VERSION}")
message(STATUS "  Eigen3: ${Eigen3_DIR}")
message(STATUS "  Qt6:    ${Qt6_DIR}")
message(STATUS "")
