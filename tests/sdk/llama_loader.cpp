// Unit tests for models::load_llama: a small valid Llama file assembled in
// memory, then one targeted defect per rejection rule. The file is untrusted
// input, so every defect must come back as an error naming the key or
// tensor, never as a crash or a silently wrong graph.

#include <branes/synapse/gguf/reader.hpp>
#include <branes/synapse/graph/session.hpp>
#include <branes/synapse/models/llama.hpp>

#include "../support/gguf_writer.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

using namespace branes::synapse;
using namespace synapse_test::gguf_writer;
using models::model_errc;

namespace {

constexpr std::uint32_t f32_tag = 0;
constexpr std::uint32_t q5_0_tag = 6;  // a format synapse has no scheme for

constexpr std::uint32_t n_embd = 32, n_ff = 64, n_head = 4, n_head_kv = 2, head_dim = 8, n_vocab = 16;

struct tensor {
    std::vector<std::uint64_t> ne;
    std::uint32_t type = f32_tag;
    float fill = 0.05f;
};

// A one-layer Llama model: keys and tensors by name, so a test can drop or
// replace any of them before building the image.
struct model_spec {
    std::map<std::string, kv_fn> keys;
    std::map<std::string, tensor> tensors;

    model_spec() {
        keys["general.architecture"] = kv_string("general.architecture", "llama");
        keys["llama.block_count"] = kv("llama.block_count", value_type::uint32, std::uint32_t{1});
        keys["llama.context_length"] = kv("llama.context_length", value_type::uint32, std::uint32_t{64});
        keys["llama.embedding_length"] = kv("llama.embedding_length", value_type::uint32, n_embd);
        keys["llama.feed_forward_length"] = kv("llama.feed_forward_length", value_type::uint32, n_ff);
        keys["llama.attention.head_count"] = kv("llama.attention.head_count", value_type::uint32, n_head);
        keys["llama.attention.head_count_kv"] = kv("llama.attention.head_count_kv", value_type::uint32, n_head_kv);
        keys["llama.attention.layer_norm_rms_epsilon"] =
            kv("llama.attention.layer_norm_rms_epsilon", value_type::float32, 1e-5f);
        keys["llama.rope.freq_base"] = kv("llama.rope.freq_base", value_type::float32, 10000.0f);

        tensors["token_embd.weight"] = {{n_embd, n_vocab}};
        tensors["output_norm.weight"] = {{n_embd}, f32_tag, 1.0f};
        tensors["blk.0.attn_norm.weight"] = {{n_embd}, f32_tag, 1.0f};
        tensors["blk.0.attn_q.weight"] = {{n_embd, n_head * head_dim}};
        tensors["blk.0.attn_k.weight"] = {{n_embd, n_head_kv * head_dim}};
        tensors["blk.0.attn_v.weight"] = {{n_embd, n_head_kv * head_dim}};
        tensors["blk.0.attn_output.weight"] = {{n_head * head_dim, n_embd}};
        tensors["blk.0.ffn_norm.weight"] = {{n_embd}, f32_tag, 1.0f};
        tensors["blk.0.ffn_gate.weight"] = {{n_embd, n_ff}};
        tensors["blk.0.ffn_up.weight"] = {{n_embd, n_ff}};
        tensors["blk.0.ffn_down.weight"] = {{n_ff, n_embd}};
    }

    std::vector<std::byte> image() const {
        constexpr std::size_t alignment = 32;
        const auto nbytes = [](const tensor& t) {
            std::uint64_t n = 1;
            for (auto d : t.ne) {
                n *= d;
            }
            return static_cast<std::size_t>(t.type == q5_0_tag ? n / 32 * 22 : n * sizeof(float));
        };
        writer w;
        w.put<std::uint32_t>(0x46554747).put<std::uint32_t>(3);
        w.put<std::uint64_t>(tensors.size()).put<std::uint64_t>(keys.size());
        for (const auto& [name, f] : keys) {
            f(w);
        }
        std::size_t offset = 0;
        for (const auto& [name, t] : tensors) {
            w.put_string(name).put<std::uint32_t>(static_cast<std::uint32_t>(t.ne.size()));
            for (auto d : t.ne) {
                w.put(d);
            }
            w.put(t.type).put<std::uint64_t>(offset);
            offset += (nbytes(t) + alignment - 1) / alignment * alignment;
        }
        w.pad_to(alignment);
        for (const auto& [name, t] : tensors) {
            if (t.type == f32_tag) {
                for (std::size_t i = 0; i < nbytes(t) / sizeof(float); ++i) {
                    // Matrices alternate signs so matmuls do not wash out;
                    // vectors (norms, frequency factors) hold `fill`.
                    w.put(t.ne.size() > 1 && i % 2 != 0 ? -t.fill * 0.5f : t.fill);
                }
            } else {
                for (std::size_t i = 0; i < nbytes(t); ++i) {
                    w.put(std::uint8_t{0});
                }
            }
            w.pad_to(alignment);
        }
        return w.bytes();
    }
};

struct loaded {
    std::vector<std::byte> bytes;
    std::optional<gguf::gguf_file> file;
};

result<models::llama_model, models::model_error> load(const model_spec& spec, loaded& keep, std::size_t max_ctx = 8) {
    keep.bytes = spec.image();
    auto f = gguf::parse(keep.bytes);
    REQUIRE(f);
    keep.file.emplace(std::move(*f));
    return models::load_llama(*keep.file, max_ctx);
}

void require_error(const model_spec& spec, model_errc code, const std::string& name, std::size_t max_ctx = 8) {
    loaded keep;
    const auto m = load(spec, keep, max_ctx);
    REQUIRE_FALSE(m);
    CHECK(m.error().code == code);
    CHECK(m.error().name == name);
}

}  // namespace

TEST_CASE("llama_loader: builds the decode graph of a valid model", "[llama][loader]") {
    model_spec spec;
    loaded keep;
    auto m = load(spec, keep, 8);
    if (!m) {
        FAIL(to_string(m.error().code) << ": " << m.error().name);
    }
    const auto& c = m->config;
    CHECK(c.n_layer == 1);
    CHECK(c.n_embd == n_embd);
    CHECK(c.n_head == n_head);
    CHECK(c.n_head_kv == n_head_kv);
    CHECK(c.head_dim == head_dim);
    CHECK(c.n_vocab == n_vocab);  // from token_embd's shape
    CHECK(m->max_ctx == 8);

    const auto& g = m->graph;
    CHECK(g.inputs().empty());
    REQUIRE(g.outputs().size() == 1);
    CHECK(g.outputs()[0] == m->logits);
    CHECK(g.tensors()[m->logits].size == n_vocab);
    CHECK(g.states().size() == 1);  // one KV cache per layer
    CHECK(g.token_limit() == n_vocab);
    CHECK(g.position_limit() == 8);
    const auto count = [&](auto pred) { return std::count_if(g.nodes().begin(), g.nodes().end(), pred); };
    CHECK(count([](const graph::node& n) { return std::holds_alternative<graph::op_attention>(n); }) == 1);
    CHECK(count([](const graph::node& n) { return std::holds_alternative<graph::op_rope>(n); }) == 2);
    CHECK(count([](const graph::node& n) { return std::holds_alternative<graph::op_matmul>(n); }) == 8);

    // It runs, and its logits are finite.
    graph::session s{g};
    for (std::uint32_t pos = 0; pos < 8; ++pos) {
        REQUIRE(s.run({pos, pos}));
    }
    for (float v : s.tensor(m->logits)) {
        CHECK(std::isfinite(v));
    }
    CHECK_FALSE(s.run({n_vocab, 0}));  // token out of range
    CHECK_FALSE(s.run({0, 8}));        // position beyond the KV cache
}

TEST_CASE("llama_loader: optional keys take llama.cpp's defaults", "[llama][loader]") {
    model_spec spec;
    spec.keys.erase("llama.attention.head_count_kv");
    spec.keys.erase("llama.rope.freq_base");
    spec.tensors["blk.0.attn_k.weight"] = {{n_embd, n_head * head_dim}};
    spec.tensors["blk.0.attn_v.weight"] = {{n_embd, n_head * head_dim}};
    loaded keep;
    auto m = load(spec, keep);
    REQUIRE(m);
    CHECK(m->config.n_head_kv == n_head);
    CHECK(m->config.rope_base == 10000.0f);
}

TEST_CASE("llama_loader: uses output.weight when present, else the tied embedding", "[llama][loader]") {
    const auto logits_weight = [](const models::llama_model& m) {
        const auto& last = std::get<graph::op_matmul>(m.graph.nodes().back());
        return std::visit([](const auto& v) { return v.data().data(); }, last.w);
    };
    model_spec tied;
    loaded keep_tied;
    auto a = load(tied, keep_tied);
    REQUIRE(a);
    CHECK(logits_weight(*a) == keep_tied.file->tensor_data(*keep_tied.file->find_tensor("token_embd.weight")).data());

    model_spec untied;
    untied.tensors["output.weight"] = {{n_embd, n_vocab}};
    loaded keep_untied;
    auto b = load(untied, keep_untied);
    REQUIRE(b);
    CHECK(logits_weight(*b) == keep_untied.file->tensor_data(*keep_untied.file->find_tensor("output.weight")).data());
}

TEST_CASE("llama_loader: reads llama3 rope frequency factors", "[llama][loader]") {
    model_spec spec;
    spec.tensors["rope_freqs.weight"] = {{head_dim / 2}, f32_tag, 2.0f};
    loaded keep;
    auto m = load(spec, keep);
    REQUIRE(m);
    for (const auto& n : m->graph.nodes()) {
        if (const auto* r = std::get_if<graph::op_rope>(&n)) {
            REQUIRE(r->freq_factors.size() == head_dim / 2);
            CHECK(r->freq_factors[0] == 2.0f);
        }
    }
}

TEST_CASE("llama_loader: rejects files that are not llama models", "[llama][loader]") {
    model_spec spec;
    spec.keys["general.architecture"] = kv_string("general.architecture", "gpt2");
    require_error(spec, model_errc::wrong_architecture, "general.architecture");
}

TEST_CASE("llama_loader: rejects missing and invalid metadata", "[llama][loader]") {
    {
        model_spec spec;
        spec.keys.erase("llama.block_count");
        require_error(spec, model_errc::missing_key, "llama.block_count");
    }
    {
        model_spec spec;
        spec.keys.erase("llama.attention.layer_norm_rms_epsilon");
        require_error(spec, model_errc::missing_key, "llama.attention.layer_norm_rms_epsilon");
    }
    {
        model_spec spec;  // a count stored as a string
        spec.keys["llama.embedding_length"] = kv_string("llama.embedding_length", "32");
        require_error(spec, model_errc::bad_value, "llama.embedding_length");
    }
    {
        model_spec spec;
        spec.keys["llama.block_count"] = kv("llama.block_count", value_type::uint32, std::uint32_t{0});
        require_error(spec, model_errc::bad_value, "llama.block_count");
    }
    {
        model_spec spec;  // negative counts are not counts
        spec.keys["llama.attention.head_count"] = kv("llama.attention.head_count", value_type::int32, std::int32_t{-4});
        require_error(spec, model_errc::bad_value, "llama.attention.head_count");
    }
    {
        model_spec spec;  // 4 query heads cannot share 3 KV heads
        spec.keys["llama.attention.head_count_kv"] =
            kv("llama.attention.head_count_kv", value_type::uint32, std::uint32_t{3});
        require_error(spec, model_errc::bad_value, "llama.attention.head_count_kv");
    }
    {
        model_spec spec;
        spec.keys["llama.rope.freq_base"] = kv("llama.rope.freq_base", value_type::float32, -1.0f);
        require_error(spec, model_errc::bad_value, "llama.rope.freq_base");
    }
    {
        model_spec spec;
        spec.keys["llama.attention.layer_norm_rms_epsilon"] =
            kv("llama.attention.layer_norm_rms_epsilon", value_type::float32, std::nanf(""));
        require_error(spec, model_errc::bad_value, "llama.attention.layer_norm_rms_epsilon");
    }
    {
        model_spec spec;
        require_error(spec, model_errc::bad_value, "max_ctx", 0);
    }
}

TEST_CASE("llama_loader: rejects features it does not implement", "[llama][loader]") {
    {
        model_spec spec;
        spec.tensors["blk.0.attn_q.bias"] = {{n_head * head_dim}};
        require_error(spec, model_errc::unsupported, "blk.0.attn_q.bias");
    }
    {
        model_spec spec;
        spec.keys["llama.rope.scaling.type"] = kv_string("llama.rope.scaling.type", "linear");
        require_error(spec, model_errc::unsupported, "llama.rope.scaling.type");
    }
    {
        model_spec spec;
        spec.keys["llama.expert_count"] = kv("llama.expert_count", value_type::uint32, std::uint32_t{8});
        require_error(spec, model_errc::unsupported, "llama.expert_count");
    }
    {
        model_spec spec;
        spec.keys["llama.rope.dimension_count"] =
            kv("llama.rope.dimension_count", value_type::uint32, std::uint32_t{head_dim / 2});
        require_error(spec, model_errc::unsupported, "llama.rope.dimension_count");
    }
    {
        model_spec spec;
        spec.keys["llama.attention.value_length"] =
            kv("llama.attention.value_length", value_type::uint32, std::uint32_t{2 * head_dim});
        require_error(spec, model_errc::unsupported, "llama.attention.value_length");
    }
}

TEST_CASE("llama_loader: rejects missing, misshapen, and uncomputable tensors", "[llama][loader]") {
    {
        model_spec spec;
        spec.tensors.erase("blk.0.ffn_up.weight");
        require_error(spec, model_errc::missing_tensor, "blk.0.ffn_up.weight");
    }
    {
        model_spec spec;
        spec.tensors.erase("token_embd.weight");
        require_error(spec, model_errc::missing_tensor, "token_embd.weight");
    }
    {
        model_spec spec;  // Wq with too few rows for n_head heads
        spec.tensors["blk.0.attn_q.weight"] = {{n_embd, n_head_kv * head_dim}};
        require_error(spec, model_errc::bad_tensor, "blk.0.attn_q.weight");
    }
    {
        model_spec spec;  // a norm weight that is a matrix
        spec.tensors["blk.0.ffn_norm.weight"] = {{n_embd, 2}};
        require_error(spec, model_errc::bad_tensor, "blk.0.ffn_norm.weight");
    }
    {
        model_spec spec;
        spec.tensors["blk.0.ffn_up.weight"] = {{n_embd, n_ff}, q5_0_tag};
        require_error(spec, model_errc::unsupported_type, "blk.0.ffn_up.weight");
    }
    {
        model_spec spec;
        spec.tensors["rope_freqs.weight"] = {{head_dim}};
        require_error(spec, model_errc::bad_tensor, "rope_freqs.weight");
    }
    {
        model_spec spec;
        spec.tensors["rope_freqs.weight"] = {{32}, q5_0_tag};
        require_error(spec, model_errc::unsupported_type, "rope_freqs.weight");
    }
}

TEST_CASE("llama_loader: reports graph errors from invalid weights", "[llama][loader]") {
    model_spec spec;  // a zero frequency factor would divide theta by zero
    spec.tensors["rope_freqs.weight"] = {{head_dim / 2}, f32_tag, 0.0f};
    loaded keep;
    const auto m = load(spec, keep);
    REQUIRE_FALSE(m);
    CHECK(m.error().code == model_errc::graph);
    CHECK(m.error().graph.code == graph::graph_errc::bad_parameter);
}

TEST_CASE("llama_loader: a block count beyond the file's tensors fails fast", "[llama][loader]") {
    model_spec spec;  // claims 4 billion layers; blk.1 is the first one missing
    spec.keys["llama.block_count"] = kv("llama.block_count", value_type::uint32, std::uint32_t{4'000'000'000});
    require_error(spec, model_errc::missing_tensor, "blk.1.attn_norm.weight");
}
