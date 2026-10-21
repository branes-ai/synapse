// Decodes a golden file's tokens through synapse and compares the logits
// after each token with llama.cpp's (see scripts/llama_oracle/oracle.cpp).
#pragma once

#include <branes/synapse/gguf/mapped_file.hpp>
#include <branes/synapse/gguf/reader.hpp>
#include <branes/synapse/graph/session.hpp>
#include <branes/synapse/models/llama.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <vector>

namespace synapse_test {

/// How far synapse's logits are from llama.cpp's at one position.
struct logit_match {
    double rel_max_err = 0.0;  ///< max |ours - ref| / max |ref|
    double rel_rms_err = 0.0;  ///< rms(ours - ref) / rms(ref)
    bool same_argmax = false;
    std::size_t ref_rank = 0;  ///< rank of llama.cpp's top token among ours (0 = also our top)
};

struct golden_run {
    std::vector<logit_match> positions;
};

inline golden_run decode_against_golden(const std::filesystem::path& model_path,
                                        const std::filesystem::path& golden_path) {
    using namespace branes::synapse;
    auto model_file = gguf::mapped_file::open(model_path);
    REQUIRE(model_file);
    auto golden_file = gguf::mapped_file::open(golden_path);
    REQUIRE(golden_file);
    auto mf = gguf::parse(model_file->bytes());
    REQUIRE(mf);
    auto gf = gguf::parse(golden_file->bytes());
    REQUIRE(gf);

    const gguf::array_value* tokens = gf->get_array("synapse.golden.tokens");
    REQUIRE(tokens != nullptr);
    const gguf::tensor_info* logits_info = gf->find_tensor("logits");
    REQUIRE(logits_info != nullptr);
    REQUIRE(logits_info->type == gguf::tensor_type::f32);
    const auto n_tokens = static_cast<std::size_t>(tokens->size());
    REQUIRE(logits_info->ne[1] == n_tokens);

    auto model = models::load_llama(*mf, n_tokens);
    if (!model) {
        FAIL(to_string(model.error().code) << ": " << model.error().name);
    }
    const std::size_t n_vocab = model->config.n_vocab;
    REQUIRE(logits_info->ne[0] == n_vocab);
    std::vector<float> ref(n_vocab * n_tokens);
    std::memcpy(ref.data(), gf->tensor_data(*logits_info).data(), ref.size() * sizeof(float));

    graph::session s{model->graph};
    golden_run run;
    for (std::size_t i = 0; i < n_tokens; ++i) {
        const auto token = tokens->at<std::uint32_t>(i);
        REQUIRE(token);
        REQUIRE(s.run({*token, static_cast<std::uint32_t>(i)}));
        const auto ours = s.tensor(model->logits);
        const float* r = ref.data() + i * n_vocab;
        double max_err = 0.0, max_ref = 0.0, sq_err = 0.0, sq_ref = 0.0;
        for (std::size_t v = 0; v < n_vocab; ++v) {
            const double e = static_cast<double>(ours[v]) - r[v];
            max_err = std::max(max_err, std::fabs(e));
            max_ref = std::max(max_ref, std::fabs(static_cast<double>(r[v])));
            sq_err += e * e;
            sq_ref += static_cast<double>(r[v]) * r[v];
        }
        logit_match m;
        m.rel_max_err = max_err / max_ref;
        m.rel_rms_err = std::sqrt(sq_err / sq_ref);
        const auto ref_top = static_cast<std::size_t>(std::max_element(r, r + n_vocab) - r);
        m.ref_rank = static_cast<std::size_t>(
            std::count_if(ours.begin(), ours.end(), [&](float v) { return v > ours[ref_top]; }));
        m.same_argmax = m.ref_rank == 0;
        run.positions.push_back(m);
    }
    return run;
}

}  // namespace synapse_test
