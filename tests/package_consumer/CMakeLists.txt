cmake_minimum_required(VERSION 3.20)

project(SkigenPlotPackageConsumer LANGUAGES CXX)

find_package(SkigenPlot 1.0 REQUIRED CONFIG)

add_executable(skigenplot_package_consumer main.cpp)
target_link_libraries(skigenplot_package_consumer PRIVATE SkigenPlot::SkigenPlot)
