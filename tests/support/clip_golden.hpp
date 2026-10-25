// Encodes a golden file's image through synapse's mmproj encoder and compares
// the embeddings with llama.cpp's (see scripts/clip_oracle/oracle.cpp).
#pragma once

#include <branes/synapse/gguf/mapped_file.hpp>
#include <branes/synapse/gguf/reader.hpp>
#include <branes/synapse/graph/session.hpp>
#include <branes/synapse/models/clip.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <vector>

namespace synapse_test {

/// How far synapse's embedding of one token is from llama.cpp's.
struct embedding_match {
    double rel_max_err = 0.0;  ///< max |ours - ref| / max |ref|
    double rel_rms_err = 0.0;  ///< rms(ours - ref) / rms(ref)
    double cosine = 0.0;       ///< cosine similarity of ours and ref
};

struct clip_run {
    std::vector<embedding_match> tokens;
    std::vector<float> embeddings;  ///< ours, n_tokens rows
};

/// A float32 tensor of a golden file, copied out.
inline std::vector<float> golden_floats(const branes::synapse::gguf::gguf_file& f, const char* name) {
    const auto* t = f.find_tensor(name);
    REQUIRE(t != nullptr);
    REQUIRE(t->type == branes::synapse::gguf::tensor_type::f32);
    std::vector<float> v(static_cast<std::size_t>(t->n_elements()));
    std::memcpy(v.data(), f.tensor_data(*t).data(), v.size() * sizeof(float));
    return v;
}

inline clip_run encode_against_golden(const std::filesystem::path& model_path,
                                      const std::filesystem::path& golden_path,
                                      unsigned threads = 1) {
    using namespace branes::synapse;
    auto model_file = gguf::mapped_file::open(model_path);
    REQUIRE(model_file);
    auto golden_file = gguf::mapped_file::open(golden_path);
    REQUIRE(golden_file);
    auto mf = gguf::parse(model_file->bytes());
    REQUIRE(mf);
    auto gf = gguf::parse(golden_file->bytes());
    REQUIRE(gf);

    auto model = models::load_clip(*mf);
    if (!model) {
        FAIL(to_string(model.error().code) << ": " << model.error().name);
    }
    const models::clip_config& c = model->config;
    const auto image = golden_floats(*gf, "image");
    const auto ref = golden_floats(*gf, "embeddings");
    REQUIRE(image.size() == 3 * c.image_size * c.image_size);
    REQUIRE(ref.size() == c.n_tokens() * c.n_embd_text);

    graph::session s{model->graph, threads};
    clip_run run;
    run.embeddings.resize(ref.size());
    REQUIRE(models::encode(s, *model, image, run.embeddings));
    for (std::size_t t = 0; t < c.n_tokens(); ++t) {
        const float* ours = run.embeddings.data() + t * c.n_embd_text;
        const float* r = ref.data() + t * c.n_embd_text;
        double max_err = 0.0, max_ref = 0.0, sq_err = 0.0, sq_ref = 0.0, sq_ours = 0.0, dot = 0.0;
        for (std::size_t i = 0; i < c.n_embd_text; ++i) {
            const double e = static_cast<double>(ours[i]) - r[i];
            max_err = std::max(max_err, std::fabs(e));
            max_ref = std::max(max_ref, std::fabs(static_cast<double>(r[i])));
            sq_err += e * e;
            sq_ref += static_cast<double>(r[i]) * r[i];
            sq_ours += static_cast<double>(ours[i]) * ours[i];
            dot += static_cast<double>(ours[i]) * r[i];
        }
        run.tokens.push_back({max_err / max_ref, std::sqrt(sq_err / sq_ref), dot / std::sqrt(sq_ours * sq_ref)});
    }
    return run;
}

}  // namespace synapse_test
