# cmake/warnings.cmake — strict warnings for synapse's own targets.
#
# Applied to tests and benchmarks (the library is header-only, so its headers
# are compiled — and held to these warnings — through them). Warnings are
# errors: a runtime that compiles with warnings is a runtime with a latent bug,
# and silent narrowing in a quantization kernel is exactly that kind of bug.
include_guard(GLOBAL)

function(synapse_set_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive-)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Wconversion)
    endif()
    set_target_properties(${target} PROPERTIES COMPILE_WARNING_AS_ERROR ON)
endfunction()
