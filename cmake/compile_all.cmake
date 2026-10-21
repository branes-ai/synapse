# compile_all.cmake — auto-generate one Catch2 test (or example) executable per
# source file in a directory.
#
# Adapted from the compile_all() pattern in MTL5 and Universal (via cortex), with
# two specializations: targets register every Catch2 TEST_CASE with CTest via
# catch_discover_tests (so `ctest` lists individual cases, not just one entry
# per binary), and an optional TEST_PREFIX namespaces those case names for
# suites whose acceptance commands filter on a prefix (e.g. `ctest -R gguf`).
#
# The point is productivity: drop a `<name>.cpp` into the right tests/<layer>/
# subdirectory and it is built, linked, IDE-grouped, and registered with no edit
# to any CMakeLists. Each target is named after its file stem (which must be
# globally unique) and placed under an IDE FOLDER so Visual Studio's Solution
# Explorer mirrors the layout.
#
#   compile_all(
#     FOLDER       <ide-folder>          # e.g. "Tests/sdk" (Solution Explorer group)
#     LIBS         <lib> [<lib> ...]     # libraries to link beyond Catch2
#     SOURCES      <file.cpp> [...]      # typically a file(GLOB ...) result
#     [TEST_PREFIX <prefix>]             # optional ctest name prefix, e.g. "gguf."
#   )
#
# Catch2::Catch2WithMain is linked into every target, C++20 is required, and the
# strict synapse warning set (cmake/warnings.cmake) is applied, so callers only
# list the layer libraries they need.
include_guard(GLOBAL)

include(Catch)
include(${CMAKE_CURRENT_LIST_DIR}/warnings.cmake)

function(compile_all)
    cmake_parse_arguments(CA "" "FOLDER;TEST_PREFIX" "LIBS;SOURCES" ${ARGN})
    if(NOT CA_SOURCES)
        message(FATAL_ERROR "compile_all: SOURCES is required (FOLDER='${CA_FOLDER}')")
    endif()

    foreach(source ${CA_SOURCES})
        get_filename_component(name ${source} NAME_WE)
        add_executable(${name} ${source})
        target_link_libraries(${name} PRIVATE
            Catch2::Catch2WithMain
            ${CA_LIBS}
        )
        target_compile_features(${name} PRIVATE cxx_std_20)
        synapse_set_warnings(${name})
        if(CA_FOLDER)
            set_target_properties(${name} PROPERTIES FOLDER "${CA_FOLDER}")
        endif()
        if(CA_TEST_PREFIX)
            catch_discover_tests(${name} TEST_PREFIX "${CA_TEST_PREFIX}")
        else()
            catch_discover_tests(${name})
        endif()
    endforeach()
endfunction()
