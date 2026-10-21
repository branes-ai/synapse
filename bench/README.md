# bench/

Micro-benchmarks for the runtime kernels. Each `<name>.cpp` dropped here is
built as a standalone executable (see `CMakeLists.txt`); none are registered
with CTest, since timings are machine-dependent and do not gate CI.
