# sdk/ — synapse tensor graph runtime (header-only).
#
# `branes::synapse` is an INTERFACE target carrying the include path, the
# C++20 requirement, and MTL5, the math engine the operator kernels dispatch
# to. Tensors and kernels are templated on the element type, so the same graph
# runs in float, double, or custom arithmetic (e.g. Universal posits).

add_library(branes_synapse INTERFACE)
add_library(branes::synapse ALIAS branes_synapse)

target_include_directories(branes_synapse INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(branes_synapse INTERFACE cxx_std_20)
target_link_libraries(branes_synapse INTERFACE MTL5::mtl5)
