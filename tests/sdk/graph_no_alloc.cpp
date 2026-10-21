// session::run() must not allocate (invariant I4 of the step-4 design: the
// arena is allocated once, at construction). This test replaces the global
// allocation functions for its own executable, counts calls while armed, and
// runs a full decode-step graph with the counter armed.

#include <branes/synapse/graph/session.hpp>

#include "../support/random_weights.hpp"

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <new>
#include <random>

namespace {

std::atomic<bool> armed{false};
std::atomic<long> allocations{0};

void note() noexcept {
    if (armed.load(std::memory_order_relaxed)) {
        allocations.fetch_add(1, std::memory_order_relaxed);
    }
}

void* aligned_allocate(std::size_t n, std::size_t align) {
    n = n == 0 ? 1 : n;
#if defined(_WIN32)
    void* p = _aligned_malloc(n, align);
#else
    void* p = std::aligned_alloc(align, (n + align - 1) / align * align);
#endif
    if (p == nullptr) {
        throw std::bad_alloc();
    }
    return p;
}

void aligned_free(void* p) noexcept {
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

}  // namespace

// Replacements for the global allocation functions. The array and nothrow
// forms default to these, so counting here covers them too.
void* operator new(std::size_t n) {
    note();
    if (void* p = std::malloc(n == 0 ? 1 : n)) {
        return p;
    }
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept {
    std::free(p);
}
void operator delete(void* p, std::size_t) noexcept {
    std::free(p);
}
void* operator new(std::size_t n, std::align_val_t a) {
    note();
    return aligned_allocate(n, static_cast<std::size_t>(a));
}
void operator delete(void* p, std::align_val_t) noexcept {
    aligned_free(p);
}
void operator delete(void* p, std::size_t, std::align_val_t) noexcept {
    aligned_free(p);
}
void* operator new[](std::size_t n, std::align_val_t a) {
    note();
    return aligned_allocate(n, static_cast<std::size_t>(a));
}
void operator delete[](void* p, std::align_val_t) noexcept {
    aligned_free(p);
}
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept {
    aligned_free(p);
}

using namespace branes::synapse::graph;
using synapse_test::random_matrix;
using synapse_test::tensor_type;

TEST_CASE("graph_no_alloc: session::run performs no allocation", "[graph][session]") {
    std::mt19937 rng(11);
    const random_matrix<tensor_type::f32> gamma(1, 256, rng, 0.5f, 1.5f);
    const random_matrix<tensor_type::q4_k> w1(512, 256, rng, 1e-4f, 1e-3f);
    const random_matrix<tensor_type::q4_0> w2(256, 512, rng, 1e-3f, 5e-3f);
    const random_matrix<tensor_type::f16> w3(64, 256, rng, 1e-3f, 5e-3f);

    graph_builder b;
    const tensor_id x = b.input(256);
    const tensor_id h = b.rms_norm(x, gamma.view(), 1e-5f);
    const tensor_id y = b.matmul(w2.view(), b.silu(b.matmul(w1.view(), h)));
    const tensor_id z = b.softmax(b.matmul(w3.view(), b.add(x, y)));
    b.output(z);
    auto g = std::move(b).finalize();
    REQUIRE(g);

    session s{*g};  // the one allocation: the arena
    for (float& v : s.tensor(x)) {
        v = 0.25f;
    }

    // Control: the counter does see allocations when they happen. A direct
    // call to ::operator new, because the compiler may elide a paired
    // new-expression and delete-expression (and does, at -O3).
    armed = true;
    ::operator delete(::operator new(16));
    armed = false;
    REQUIRE(allocations == 1);
    allocations = 0;

    armed = true;
    s.run();
    s.run();
    armed = false;
    CHECK(allocations == 0);
}
