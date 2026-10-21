# bench/ — micro-benchmarks for the runtime kernels.
#
# Each .cpp here is a standalone executable that reports kernel throughput or
# latency. They are built by default (SYNAPSE_BUILD_BENCH) so they keep
# compiling, but are not registered with CTest: timings are machine-dependent
# and do not gate CI.

file(GLOB BENCH_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
foreach(source ${BENCH_SRC})
    get_filename_component(name ${source} NAME_WE)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE branes::synapse)
    target_compile_features(${name} PRIVATE cxx_std_20)
    synapse_set_warnings(${name})
    set_target_properties(${name} PROPERTIES FOLDER "Bench")
endforeach()
