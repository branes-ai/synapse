# tools/ — host-side developer tools and reference applications.
#
# Each .cpp here is a standalone executable. They are built by default
# (SYNAPSE_BUILD_TOOLS) so they keep compiling; they need a model file to run,
# so they are not registered with CTest.

file(GLOB TOOL_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
foreach(source ${TOOL_SRC})
    get_filename_component(name ${source} NAME_WE)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE branes::synapse)
    target_compile_features(${name} PRIVATE cxx_std_20)
    synapse_set_warnings(${name})
    synapse_set_strict_fp(${name})
    set_target_properties(${name} PROPERTIES FOLDER "Tools")
endforeach()
