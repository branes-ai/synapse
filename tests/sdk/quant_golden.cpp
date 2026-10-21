// Golden tests for the dequantizers: tests/data/quant_golden.gguf (written by
// scripts/gen_quant_fixture.py) pairs each encoded tensor `<fmt>.q` with the
// float32 tensor `<fmt>.expected` that llama.cpp's gguf-py decodes it to. The
// fixture is loaded through synapse's own mapped_file and GGUF parser, so this
// is also an end-to-end test of the loader.
//
// The comparison is bit-exact: the dequantizers follow ggml's order of float
// operations. A platform that fuses multiply-adds (-ffp-contract=fast with FMA
// hardware) could differ in the last bit of the K-quant results.

#include <branes/synapse/gguf/mapped_file.hpp>
#include <branes/synapse/gguf/reader.hpp>
#include <branes/synapse/gguf/tensor_view.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace branes::synapse;

namespace {

const std::string fixture_path = std::string(SYNAPSE_TEST_DATA_DIR) + "/quant_golden.gguf";

// The fixture stays mapped for the whole test binary.
const gguf::gguf_file& fixture() {
    static const gguf::mapped_file file = [] {
        auto f = gguf::mapped_file::open(fixture_path);
        REQUIRE(f);
        return std::move(*f);
    }();
    static const gguf::gguf_file parsed = [] {
        auto p = gguf::parse(file.bytes());
        REQUIRE(p);
        return std::move(*p);
    }();
    return parsed;
}

// Row r of a float32 tensor, read from its raw bytes.
std::vector<float> expected_row(const gguf::tensor_info& t, std::size_t r) {
    REQUIRE(t.type == gguf::tensor_type::f32);
    const auto cols = static_cast<std::size_t>(t.ne[0]);
    std::vector<float> row(cols);
    std::memcpy(row.data(), fixture().tensor_data(t).data() + r * cols * sizeof(float), cols * sizeof(float));
    return row;
}

// Number of elements whose bits differ.
std::size_t bit_mismatches(const std::vector<float>& a, const std::vector<float>& b) {
    std::size_t n = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        n += std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(b[i]) ? 1 : 0;
    }
    return n;
}

}  // namespace

TEST_CASE("quant_golden: every format decodes bit-exactly to the gguf-py reference", "[quant][golden]") {
    const std::string name = GENERATE("f16", "bf16", "q4_0", "q8_0", "q8_0.3d", "q4_k", "q6_k");
    INFO("tensor " << name);
    const auto& file = fixture();
    const auto* q = file.find_tensor(name + ".q");
    const auto* expected = file.find_tensor(name + ".expected");
    REQUIRE(q != nullptr);
    REQUIRE(expected != nullptr);

    const bool visited = gguf::visit_matrix(file, *q, [&](const auto& view) {
        REQUIRE(view.rows() * view.cols() == expected->n_elements());
        const auto expected_view = gguf::view_as<quant::scheme<gguf::tensor_type::f32>>(file, *expected);
        REQUIRE(expected_view);
        REQUIRE(expected_view->rows() == view.rows());
        std::vector<float> row(view.cols());
        for (std::size_t r = 0; r < view.rows(); ++r) {
            view.dequantize_row(r, row);
            CHECK(bit_mismatches(row, expected_row(*expected, r)) == 0);
        }
    });
    CHECK(visited);
}

TEST_CASE("quant_golden: ne[1..3] collapse into rows", "[quant][golden]") {
    const auto* t = fixture().find_tensor("q8_0.3d.q");
    REQUIRE(t != nullptr);
    CHECK(t->n_dims == 3);
    const auto view = gguf::view_as<quant::scheme<gguf::tensor_type::q8_0>>(fixture(), *t);
    REQUIRE(view);
    CHECK(view->rows() == 4);
    CHECK(view->cols() == 64);
}

TEST_CASE("quant_golden: to_dense agrees with dequantize_row", "[quant][golden]") {
    const auto* t = fixture().find_tensor("q4_k.q");
    REQUIRE(t != nullptr);
    const auto view = gguf::view_as<quant::scheme<gguf::tensor_type::q4_k>>(fixture(), *t);
    REQUIRE(view);
    const auto dense = view->to_dense();
    REQUIRE(dense.num_rows() == view->rows());
    REQUIRE(dense.num_cols() == view->cols());
    std::vector<float> row(view->cols());
    for (std::size_t r = 0; r < view->rows(); ++r) {
        view->dequantize_row(r, row);
        for (std::size_t c = 0; c < view->cols(); c += 37) {
            CHECK(dense(r, c) == row[c]);
            CHECK(view->at(r, c) == row[c]);
        }
    }
}

TEST_CASE("quant_golden: visit_matrix rejects types without a scheme", "[quant][golden]") {
    const auto* t = fixture().find_tensor("i32.unsupported");
    REQUIRE(t != nullptr);
    bool called = false;
    CHECK_FALSE(gguf::visit_matrix(fixture(), *t, [&](const auto&) { called = true; }));
    CHECK_FALSE(called);
}

TEST_CASE("quant_golden: view_as rejects a mismatched scheme", "[quant][golden]") {
    const auto* t = fixture().find_tensor("q4_k.q");
    REQUIRE(t != nullptr);
    CHECK_FALSE(gguf::view_as<quant::scheme<gguf::tensor_type::q6_k>>(fixture(), *t));
}
