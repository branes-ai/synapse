// Mixed-precision matrix-vector product — an end-to-end example that runs the
// core transformer kernel through MTL5 and Universal, so CI exercises both
// dependencies.
//
// Every linear layer in a transformer is y = W x. This example computes a
// small projection with MTL5 dense containers in double (the reference),
// float, posit<32,2>, and two 16-bit formats an inference runtime stores
// activations in, IEEE binary16 (cfloat<16,5>) and posit<16,1>, and checks
// each against the double reference. The same kernel code instantiates for
// every type: that genericity is what the synapse runtime builds on.

#include <catch2/catch_test_macros.hpp>
#include <mtl/mtl.hpp>
#include <universal/number/cfloat/cfloat.hpp>
#include <universal/number/posit/posit.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

using posit32 = sw::universal::posit<32, 2>;
using posit16 = sw::universal::posit<16, 1>;
using binary16 = sw::universal::cfloat<16, 5, std::uint16_t, true, false, false>;  // IEEE-754 binary16 layout

constexpr std::size_t kRows = 16;
constexpr std::size_t kCols = 64;

// Deterministic weights and activations in [-1, 1), the range a normalized
// layer sees.
double weight(std::size_t r, std::size_t c) {
    return std::sin(0.37 * static_cast<double>(r) + 0.11 * static_cast<double>(c));
}

double activation(std::size_t c) {
    return std::cos(0.23 * static_cast<double>(c));
}

template <typename T>
std::vector<double> matvec() {
    mtl::mat::dense2D<T> w(kRows, kCols);
    mtl::vec::dense_vector<T> x(kCols);
    for (std::size_t r = 0; r < kRows; ++r) {
        for (std::size_t c = 0; c < kCols; ++c) {
            w(r, c) = T(weight(r, c));
        }
    }
    for (std::size_t c = 0; c < kCols; ++c) {
        x[c] = T(activation(c));
    }
    mtl::vec::dense_vector<T> y(kRows);
    y = w * x;

    std::vector<double> out(kRows);
    for (std::size_t r = 0; r < kRows; ++r) {
        out[r] = static_cast<double>(y[r]);
    }
    return out;
}

double max_abs_error(const std::vector<double>& a, const std::vector<double>& b) {
    double err = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        err = std::fmax(err, std::fabs(a[i] - b[i]));
    }
    return err;
}

}  // namespace

TEST_CASE("matvec: double reference matches a hand-rolled loop", "[examples][matvec]") {
    const auto y = matvec<double>();
    for (std::size_t r = 0; r < kRows; ++r) {
        double expected = 0.0;
        for (std::size_t c = 0; c < kCols; ++c) {
            expected += weight(r, c) * activation(c);
        }
        CHECK(std::fabs(y[r] - expected) < 1e-12);
    }
}

TEST_CASE("matvec: 32-bit formats track the double reference", "[examples][matvec]") {
    const auto ref = matvec<double>();
    CHECK(max_abs_error(matvec<float>(), ref) < 1e-5);
    CHECK(max_abs_error(matvec<posit32>(), ref) < 1e-5);
}

TEST_CASE("matvec: 16-bit formats track the double reference", "[examples][matvec]") {
    // A 64-term dot product accumulated in 16 bits: error grows with the
    // accumulation length, which is why production kernels accumulate wider.
    const auto ref = matvec<double>();
    CHECK(max_abs_error(matvec<binary16>(), ref) < 0.1);
    CHECK(max_abs_error(matvec<posit16>(), ref) < 0.1);
}
