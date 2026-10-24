// Unit tests for the block layouts, dequantizers, and MTL5's block_matrix_view
// over synapse's schemes,
// on hand-built blocks whose decoded values follow directly from the format
// definitions in quant/blocks.hpp.

#include <branes/synapse/quant/blocks.hpp>
#include <branes/synapse/quant/scheme.hpp>

#include <catch2/catch_test_macros.hpp>
#include <mtl/mat/block_matrix_view.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

using namespace branes::synapse;
using quant::tensor_type;

namespace {

constexpr std::uint16_t fp16_one = 0x3c00;
constexpr std::uint16_t fp16_half = 0x3800;

template <typename Block>
std::vector<std::byte> bytes_of(const std::vector<Block>& blocks) {
    std::vector<std::byte> out(blocks.size() * sizeof(Block));
    std::memcpy(out.data(), blocks.data(), out.size());
    return out;
}

// Q4_0 block with scale d whose weight j decodes to d * ((j % 16) - 8).
quant::block_q4_0 ramp_q4_0(std::uint16_t d) {
    quant::block_q4_0 b{};
    b.d = d;
    for (std::size_t j = 0; j < 16; ++j) {
        b.qs[j] = static_cast<std::uint8_t>(j | (j << 4));
    }
    return b;
}

}  // namespace

TEST_CASE("quantized_view: Q4_0 decodes low nibbles then high nibbles", "[quant]") {
    const auto b = ramp_q4_0(fp16_half);
    std::array<float, 32> y{};
    quant::dequantize(b, y.data());
    for (std::size_t j = 0; j < 16; ++j) {
        CHECK(y[j] == 0.5f * (static_cast<float>(j) - 8.0f));
        CHECK(y[j + 16] == 0.5f * (static_cast<float>(j) - 8.0f));
    }
}

TEST_CASE("quantized_view: Q5_0 takes weight j's 5th bit from bit j of qh", "[quant]") {
    // Low nibbles ramp 0..15 in both halves; qh sets the 5th bit of the odd
    // weights in the low half and of the even weights in the high half, so a
    // misplaced bit shows as a wrong weight.
    quant::block_q5_0 b{};
    b.d = fp16_half;
    for (std::size_t j = 0; j < 16; ++j) {
        b.qs[j] = static_cast<std::uint8_t>(j | (j << 4));
    }
    const std::uint32_t qh = 0x5555AAAAu;
    std::memcpy(b.qh, &qh, sizeof qh);  // little-endian, as ggml stores it
    REQUIRE(quant::q5_0_high_bits(b) == qh);
    std::array<float, 32> y{};
    quant::dequantize(b, y.data());
    for (std::size_t j = 0; j < 32; ++j) {
        const auto q = static_cast<int>((j % 16) | (((qh >> j) & 1) << 4));
        CHECK(y[j] == 0.5f * static_cast<float>(q - 16));
    }
    CHECK(y[0] == -8.0f);   // q = 0
    CHECK(y[1] == 0.5f);    // q = 1 | 16 (bit 1 set)
    CHECK(y[16] == 0.0f);   // q = 0 | 16 (bit 16 set)
    CHECK(y[17] == -7.5f);  // q = 1 (bit 17 clear)
}

TEST_CASE("quantized_view: Q8_0 is scale times signed quant", "[quant]") {
    quant::block_q8_0 b{};
    b.d = fp16_half;
    for (std::size_t j = 0; j < 32; ++j) {
        b.qs[j] = static_cast<std::int8_t>(static_cast<int>(j) * 8 - 128);
    }
    std::array<float, 32> y{};
    quant::dequantize(b, y.data());
    CHECK(y[0] == -64.0f);
    CHECK(y[31] == 0.5f * (31.0f * 8.0f - 128.0f));
}

TEST_CASE("quantized_view: Q4_K unpacks 6-bit scales and mins", "[quant]") {
    quant::block_q4_k b{};
    b.d = fp16_one;
    b.dmin = fp16_one;
    // Sub-blocks 0..3: scale j+1, min j. Sub-blocks 4..7: scale 40+j (needs the
    // high bits stored in bytes 0-3), min 50+j (high bits in bytes 4-7).
    for (std::size_t j = 0; j < 4; ++j) {
        const auto sc_hi = static_cast<std::uint8_t>(40 + j);
        const auto m_hi = static_cast<std::uint8_t>(50 + j);
        b.scales[j] = static_cast<std::uint8_t>((j + 1) | static_cast<std::size_t>((sc_hi >> 4) << 6));
        b.scales[j + 4] = static_cast<std::uint8_t>(j | static_cast<std::size_t>((m_hi >> 4) << 6));
        b.scales[j + 8] = static_cast<std::uint8_t>((sc_hi & 0x0f) | ((m_hi & 0x0f) << 4));
    }
    for (std::size_t j = 0; j < 8; ++j) {
        std::uint8_t sc = 0;
        std::uint8_t m = 0;
        quant::q4_k_scale_min(j, b.scales, sc, m);
        CHECK(sc == (j < 4 ? j + 1 : 40 + (j - 4)));
        CHECK(m == (j < 4 ? j : 50 + (j - 4)));
    }
    // All quants 3: weight in sub-block j is sc[j] * 3 - m[j].
    std::memset(b.qs, 0x33, sizeof(b.qs));
    std::array<float, 256> y{};
    quant::dequantize(b, y.data());
    CHECK(y[0] == 1.0f * 3 - 0);      // sub-block 0
    CHECK(y[32] == 2.0f * 3 - 1);     // sub-block 1
    CHECK(y[255] == 43.0f * 3 - 53);  // sub-block 7
}

TEST_CASE("quantized_view: Q6_K recombines low and high bits", "[quant]") {
    quant::block_q6_k b{};
    b.d = fp16_one;
    for (auto& s : b.scales) {
        s = 1;
    }
    // Low bits 0xF, high bits 0b11: every quant is 63, so every weight is 63 - 32.
    std::memset(b.ql, 0xff, sizeof(b.ql));
    std::memset(b.qh, 0xff, sizeof(b.qh));
    std::array<float, 256> y{};
    quant::dequantize(b, y.data());
    for (float v : y) {
        CHECK(v == 31.0f);
    }
    // A zero quant decodes to -32 * scale.
    std::memset(b.ql, 0, sizeof(b.ql));
    std::memset(b.qh, 0, sizeof(b.qh));
    b.scales[15] = -2;
    quant::dequantize(b, y.data());
    CHECK(y[0] == -32.0f);
    CHECK(y[255] == 64.0f);  // last sub-block: -2 * -32
}

TEST_CASE("quantized_view: geometry and row addressing", "[quant]") {
    // 2 rows x 64 cols of Q4_0: 2 blocks per row; row r block b has scale (r*2+b+1)/2.
    const std::uint16_t scales[] = {fp16_half, fp16_one, 0x3e00 /* 1.5 */, 0x4000 /* 2 */};
    std::vector<quant::block_q4_0> blocks;
    for (auto s : scales) {
        blocks.push_back(ramp_q4_0(s));
    }
    const auto data = bytes_of(blocks);
    const mtl::mat::block_matrix_view<quant::scheme<tensor_type::q4_0>> view(2, 64, data);

    CHECK(view.num_rows() == 2);
    CHECK(view.num_cols() == 64);
    CHECK(view.blocks_per_row() == 2);
    CHECK(view.row(1).size() == 36);
    CHECK(view.row(1).data() == data.data() + 36);
    CHECK(quant::detail::load_block<quant::block_q4_0>(view.block(1, 1).data()).d == 0x4000);

    std::vector<float> row(64);
    view.decode_row(1, row.data());
    CHECK(row[0] == 1.5f * -8.0f);  // row 1, block 0
    CHECK(row[63] == 2.0f * 7.0f);  // row 1, block 1, last weight
    CHECK(view(1, 63) == row[63]);
    CHECK(view(0, 32) == 1.0f * -8.0f);

    const auto dense = mtl::convert<float>(view);
    CHECK(dense(1, 0) == row[0]);
    CHECK(dense(0, 33) == view(0, 33));
}

TEST_CASE("quantized_view: plain formats are one-element blocks", "[quant]") {
    const std::vector<std::uint16_t> halves = {fp16_one, fp16_half, 0xc000, 0x0000};
    const auto data = bytes_of(halves);
    const mtl::mat::block_matrix_view<quant::scheme<tensor_type::f16>> view(2, 2, data);
    CHECK(view.blocks_per_row() == 2);
    CHECK(view(0, 1) == 0.5f);
    CHECK(view(1, 0) == -2.0f);
    CHECK(quant::detail::load_block<std::uint16_t>(view.block(0, 0).data()) == fp16_one);

    const mtl::mat::block_matrix_view<quant::scheme<tensor_type::f32>> empty(0, 0, {});
    CHECK(empty.num_rows() == 0);
    CHECK(empty.size() == 0);
}

TEST_CASE("quantized_view: block_matrix_view rejects a shape its bytes do not hold", "[quant]") {
    using view_t = mtl::mat::block_matrix_view<quant::scheme<tensor_type::q4_0>>;
    const std::vector<std::byte> data(2 * sizeof(quant::block_q4_0));
    CHECK_NOTHROW(view_t(1, 64, data));
    CHECK_THROWS_AS(view_t(1, 48, data), std::invalid_argument);  // not whole blocks
    CHECK_THROWS_AS(view_t(2, 64, data), std::invalid_argument);  // too few bytes
}

TEST_CASE("quantized_view: scheme registry", "[quant]") {
    STATIC_REQUIRE(quant::has_scheme<tensor_type::q4_k>);
    STATIC_REQUIRE(quant::has_scheme<tensor_type::bf16>);
    STATIC_REQUIRE(quant::has_scheme<tensor_type::q5_0>);
    STATIC_REQUIRE_FALSE(quant::has_scheme<tensor_type::q5_1>);
    STATIC_REQUIRE_FALSE(quant::has_scheme<tensor_type::q5_k>);
    STATIC_REQUIRE_FALSE(quant::has_scheme<tensor_type::i32>);
    STATIC_REQUIRE(quant::Scheme<quant::scheme<tensor_type::q6_k>>);
    STATIC_REQUIRE(mtl::BlockScheme<quant::scheme<tensor_type::q4_0>>);
    STATIC_REQUIRE(mtl::BlockScheme<quant::scheme<tensor_type::f16>>);
    STATIC_REQUIRE(quant::scheme<tensor_type::q4_k>::block_bytes == quant::scheme<tensor_type::q4_k>::type_size);
}
