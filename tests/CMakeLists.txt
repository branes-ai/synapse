# Regression tests. Each .cpp under a layer subdirectory becomes a Catch2 test
# executable automatically — see cmake/compile_all.cmake. To add a test, drop a
# `<name>.cpp` into the matching tests/<layer>/ subdirectory; no edit here is
# needed. File stems must be globally unique (they name the targets).

include(${PROJECT_SOURCE_DIR}/cmake/compile_all.cmake)

# ── SDK layer ──────────────────────────────────────────────────────────────
file(GLOB SDK_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/sdk/*.cpp)
compile_all(FOLDER "Tests/sdk" LIBS branes::synapse SOURCES ${SDK_SRC})

# ── Examples ───────────────────────────────────────────────────────────────
# End-to-end examples that run synapse-style kernels through MTL5 and
# Universal number types, so CI exercises both dependencies.
file(GLOB EXAMPLES_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cpp)
compile_all(FOLDER "Tests/examples" LIBS branes::synapse universal::universal SOURCES ${EXAMPLES_SRC})
