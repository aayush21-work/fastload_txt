cmake_minimum_required(VERSION 3.15)
project(fast_loadtxt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Dependencies ──────────────────────────────────────────────────────────
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

# ── Build the extension module ────────────────────────────────────────────
pybind11_add_module(_fast_loadtxt src/fast_loadtxt.cpp)

# fast_float is vendored as header-only in include/
target_include_directories(_fast_loadtxt PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

# Link OpenMP
target_link_libraries(_fast_loadtxt PRIVATE OpenMP::OpenMP_CXX)

# Optimization flags for release builds
target_compile_options(_fast_loadtxt PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native -DNDEBUG>
)

# Install into the Python package directory
install(TARGETS _fast_loadtxt LIBRARY DESTINATION fast_loadtxt)
