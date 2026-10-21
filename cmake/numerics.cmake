# cmake/numerics.cmake — floating-point code generation for synapse's own targets.
#
# GCC (in GNU mode) and Clang contract a * b + c into a fused multiply-add on
# FMA targets, and may do so differently at different inlined call sites. Two
# guarantees depend on there being no contraction:
#   - G1 (kernels/gemm.hpp): each gemm row equals gemv bit for bit, so a
#     batched prefill reproduces token-by-token decoding exactly;
#   - the golden tests, which compare bit for bit against references computed
#     without fused multiply-adds (numpy, ggml's scalar paths).
# MSVC does not contract under its default /fp:precise.
include_guard(GLOBAL)

function(synapse_set_strict_fp target)
    target_compile_options(${target} PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-ffp-contract=off>)
endfunction()
