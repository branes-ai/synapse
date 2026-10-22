# cmake/deps.cmake — FetchContent declarations for third-party deps.
#
# Permissive licenses only (MIT / BSL / BSD / Apache-2.0). Pinned versions are
# intentional; bumps go via dedicated PRs so the release-please changelog
# records dependency churn separately from feature work.
#
# MTL5 is synapse's math engine, so unlike in reflex it is part of the
# branes::synapse interface and is always resolved, including for FetchContent
# consumers. Universal and Catch2 are only fetched when tests are built: the
# runtime is generic over the arithmetic type and never names a Universal type
# itself; the tests instantiate it with posits and cfloats to keep it that way.

include(FetchContent)

# Don't re-run `git fetch` on every build.
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "" FORCE)

# ── catchorg/Catch2 v3 — testing framework ──────────────────────────
# SYSTEM so its headers don't trip the warnings-as-errors test build.
if(SYNAPSE_BUILD_TESTS AND BUILD_TESTING)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.15.0
        GIT_SHALLOW    TRUE
        SYSTEM
    )
    FetchContent_MakeAvailable(Catch2)
    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
endif()

# ── stillwater-sc/mtl5 + stillwater-sc/universal ────────────────────
# Header-only Stillwater sister projects, integrated with the pattern from the
# Stillwater mixed-precision repos (mp-iterative, mp-blas), cortex, and reflex:
#
#   1. find_package(<dep> CONFIG QUIET) first — an installed copy wins.
#   2. Otherwise a header-only FetchContent: SOURCE_SUBDIR names a directory
#      with no CMakeLists.txt, so FetchContent_MakeAvailable populates the
#      sources but never add_subdirectory()s the dependency.
#   3. Either path yields the same targets, MTL5::mtl5 and universal::universal.
#
# Co-develop against local sister checkouts (no network) with:
#   -DFETCHCONTENT_SOURCE_DIR_MTL5=/path/to/mtl5
#   -DFETCHCONTENT_SOURCE_DIR_UNIVERSAL=/path/to/universal
#
# Third-party warnings: synapse builds its own targets with strict warnings as
# errors (cmake/warnings.cmake). MTL5 and Universal headers still emit warnings
# under that set (tracked upstream in stillwater-sc/universal#1265); left
# visible, they would fail every build on code we don't own, so the fetched
# targets are SYSTEM by default (find_package's imported targets are SYSTEM
# already). To see them, configure with
# -DSYNAPSE_SHOW_DEP_WARNINGS=ON --compile-no-warning-as-error.
option(SYNAPSE_SHOW_DEP_WARNINGS "Treat MTL5/Universal headers as non-system (show their warnings)" OFF)

set(SYNAPSE_MTL5_VERSION      5.20.0)
set(SYNAPSE_UNIVERSAL_VERSION 5.11.0)

# An installed MTL5 must be at least the pinned version (MTL5's package is
# SameMajorVersion-compatible, so any 5.x >= the pin is accepted). A target a
# parent project provides is checked at compile time instead, by the
# static_assert in kernels/dot.hpp.
if(NOT TARGET MTL5::mtl5)
    find_package(MTL5 ${SYNAPSE_MTL5_VERSION} CONFIG QUIET)
endif()
if(TARGET MTL5::mtl5)
    if(MTL5_FOUND)
        message(STATUS "MTL5: using installed package ${MTL5_VERSION} (${MTL5_DIR})")
    else()
        message(STATUS "MTL5: using the MTL5::mtl5 target provided by the parent project")
    endif()
else()
    if(FETCHCONTENT_SOURCE_DIR_MTL5)
        message(STATUS "MTL5: using local checkout ${FETCHCONTENT_SOURCE_DIR_MTL5} (pin v${SYNAPSE_MTL5_VERSION} bypassed)")
    else()
        message(STATUS "MTL5: fetching v${SYNAPSE_MTL5_VERSION} headers")
    endif()
    FetchContent_Declare(
        mtl5
        GIT_REPOSITORY https://github.com/stillwater-sc/mtl5.git
        GIT_TAG        v${SYNAPSE_MTL5_VERSION}
        GIT_SHALLOW    TRUE
        SOURCE_SUBDIR  _header_only_no_build
    )
    FetchContent_MakeAvailable(mtl5)

    # MTL5's configure step normally generates mtl/version.hpp, which the
    # <mtl/mtl.hpp> umbrella includes. That step is skipped, so generate it
    # from MTL5's own template. (A local checkout still gets the pinned
    # version in these macros.)
    string(REPLACE "." ";" _mtl5_ver "${SYNAPSE_MTL5_VERSION}")
    list(GET _mtl5_ver 0 MTL5_VERSION_MAJOR)
    list(GET _mtl5_ver 1 MTL5_VERSION_MINOR)
    list(GET _mtl5_ver 2 MTL5_VERSION_PATCH)
    set(MTL5_VERSION "${SYNAPSE_MTL5_VERSION}")
    configure_file(
        ${mtl5_SOURCE_DIR}/include/mtl/version.hpp.in
        ${mtl5_BINARY_DIR}/include/mtl/version.hpp
        @ONLY)

    add_library(mtl5 INTERFACE)
    add_library(MTL5::mtl5 ALIAS mtl5)
    target_include_directories(mtl5 INTERFACE
        ${mtl5_SOURCE_DIR}/include
        ${mtl5_BINARY_DIR}/include)
    target_compile_features(mtl5 INTERFACE cxx_std_20)
    if(NOT SYNAPSE_SHOW_DEP_WARNINGS)
        set_target_properties(mtl5 PROPERTIES SYSTEM ON)
    endif()
endif()

# ── google/highway — optional SIMD backend for MTL5 ────────────────
# The fused kernels run their integer inner loops through mtl::simd, which is
# scalar unless MTL5's Highway backend is on. SYNAPSE_WITH_HIGHWAY fetches
# Highway (Apache-2.0 / BSD-3-Clause) at the tag MTL5 pins and enables it on the
# MTL5::mtl5 target. Highway dispatches statically, so it uses the ISA the
# compiler targets: pair it with flags such as -march=native, or
# -march=x86-64-v3 -maes -mpclmul for AVX2 (Highway's AVX2 target also needs
# AES and PCLMUL, which x86-64-v3 lacks; without them it selects SSSE3).
# Without ISA flags it compiles for the baseline (SSE2 on x86-64).
option(SYNAPSE_WITH_HIGHWAY "Enable MTL5's Highway SIMD backend for the kernels" OFF)
if(SYNAPSE_WITH_HIGHWAY)
    if(NOT TARGET mtl5)
        message(WARNING "SYNAPSE_WITH_HIGHWAY: MTL5 comes from an installed package or the parent "
                        "project; enable Highway there (MTL5_WITH_HIGHWAY) instead.")
    else()
        set(HWY_ENABLE_TESTS    OFF CACHE BOOL "" FORCE)
        set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(HWY_ENABLE_CONTRIB  OFF CACHE BOOL "" FORCE)
        set(HWY_ENABLE_INSTALL  OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(
            highway
            GIT_REPOSITORY https://github.com/google/highway.git
            GIT_TAG        1.4.0
            GIT_SHALLOW    TRUE
            SYSTEM
        )
        FetchContent_MakeAvailable(highway)
        target_compile_definitions(mtl5 INTERFACE MTL5_HAS_HIGHWAY)
        target_link_libraries(mtl5 INTERFACE hwy)
        message(STATUS "Highway: enabled for MTL5 (v1.4.0)")
    endif()
endif()

if(SYNAPSE_BUILD_TESTS AND BUILD_TESTING AND NOT TARGET universal::universal)
    find_package(universal CONFIG QUIET)
    if(universal_FOUND)
        message(STATUS "Universal: using installed package ${universal_VERSION} (${universal_DIR})")
    else()
        if(FETCHCONTENT_SOURCE_DIR_UNIVERSAL)
            message(STATUS "Universal: using local checkout ${FETCHCONTENT_SOURCE_DIR_UNIVERSAL} (pin v${SYNAPSE_UNIVERSAL_VERSION} bypassed)")
        else()
            message(STATUS "Universal: fetching v${SYNAPSE_UNIVERSAL_VERSION} headers")
        endif()
        FetchContent_Declare(
            universal
            GIT_REPOSITORY https://github.com/stillwater-sc/universal.git
            GIT_TAG        v${SYNAPSE_UNIVERSAL_VERSION}
            GIT_SHALLOW    TRUE
            SOURCE_SUBDIR  _header_only_no_build
        )
        FetchContent_MakeAvailable(universal)

        # Universal headers use two include conventions:
        #   - external: #include <sw/universal/...>  (needs include/)
        #   - internal: #include <universal/...>     (needs include/sw/)
        add_library(universal INTERFACE)
        add_library(universal::universal ALIAS universal)
        target_include_directories(universal INTERFACE
            ${universal_SOURCE_DIR}/include
            ${universal_SOURCE_DIR}/include/sw)
        target_compile_features(universal INTERFACE cxx_std_20)
        if(NOT SYNAPSE_SHOW_DEP_WARNINGS)
            set_target_properties(universal PROPERTIES SYSTEM ON)
        endif()
    endif()
endif()
