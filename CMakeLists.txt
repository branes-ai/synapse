cmake_minimum_required(VERSION 4.0)

# MSVC debug-info format is governed by CMAKE_MSVC_DEBUG_INFORMATION_FORMAT
# (CMP0141, NEW at our cmake_minimum_required) instead of a hardcoded /Zi.
# CMAKE_POLICY_DEFAULT_CMP0141 carries the NEW default into the FetchContent
# deps' own project() calls (Catch2), so the msvc-ninja /Z7 build doesn't hit
# C1041 (many cl.exe contending on one shared .pdb under Ninja).
set(CMAKE_POLICY_DEFAULT_CMP0141 NEW)

project(branes_synapse LANGUAGES CXX)

# synapse is a header-only C++20 library. C++20 is required (concepts,
# std::span, std::bit_cast); do not lower it.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Organize targets into folders in IDEs (Visual Studio, Xcode).
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# MSVC opt-ins that gcc/clang give by default: the conformant preprocessor
# (Catch2 raw-string macro arguments) and UTF-8 sources.
if(MSVC)
    add_compile_options(/Zc:preprocessor /utf-8)
endif()

# Tests, benchmarks and tools are built by default only when synapse is the
# top-level project, so consumers pulling it in via FetchContent get just the
# header-only branes::synapse target (and MTL5, its math engine).
option(SYNAPSE_BUILD_TESTS "Build the tests/ regression suite" ${PROJECT_IS_TOP_LEVEL})
option(SYNAPSE_BUILD_BENCH "Build the bench/ micro-benchmarks" ${PROJECT_IS_TOP_LEVEL})
option(SYNAPSE_BUILD_TOOLS "Build the tools/ developer tools and reference applications" ${PROJECT_IS_TOP_LEVEL})

if(SYNAPSE_BUILD_TESTS)
    include(CTest)
endif()

# Third-party deps (MTL5 always; Universal and Catch2 only when testing).
# Permissive licenses only.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/deps.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/warnings.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/numerics.cmake)

add_subdirectory(sdk)

if(SYNAPSE_BUILD_BENCH)
    add_subdirectory(bench)
endif()

if(SYNAPSE_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

if(SYNAPSE_BUILD_TESTS AND BUILD_TESTING)
    add_subdirectory(tests)
endif()
