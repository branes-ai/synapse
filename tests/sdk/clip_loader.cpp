// Unit tests for load_clip: a well-formed idefics3 mmproj builds, and every
// key and tensor it depends on is checked, with the failing name reported.

#include <branes/synapse/gguf/reader.hpp>
#include <branes/synapse/models/clip.hpp>

#include "../support/gguf_writer.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

using namespace branes::synapse;
using namespace synapse_test::gguf_writer;
using models::model_errc;

namespace {

constexpr std::uint32_t f32_tag = 0;
constexpr std::uint32_t q5_1_tag = 7;  // a format synapse has no scheme for

// A 16 x 16 image of 4 x 4 patches (a 4 x 4 grid), width 8, 2 heads, one
// layer, merged 2 x 2 into 4 tokens of 6 floats.
constexpr std::uint32_t image = 16, patch = 4, n_embd = 8, n_ff = 16, n_head = 2, scale = 2, n_text = 6;
constexpr std::uint64_t n_patches = (image / patch) * (image / patch);

kv_fn kv_bool(const std::string& key, bool v) {
    return [key, v](writer& w) {
        w.put_string(key).put(static_cast<std::uint32_t>(value_type::boolean)).put(static_cast<std::uint8_t>(v));
    };
}

kv_fn kv_floats(const std::string& key, std::vector<float> v) {
    return [key, v](writer& w) {
        w.put_string(key).put(static_cast<std::uint32_t>(value_type::array));
        w.put(static_cast<std::uint32_t>(value_type::float32)).put<std::uint64_t>(v.size());
        for (float x : v) {
            w.put(x);
        }
    };
}

struct tensor {
    std::vector<std::uint64_t> ne;
    std::uint32_t type = f32_tag;
};

struct mmproj_spec {
    std::map<std::string, kv_fn> keys;
    std::map<std::string, tensor> tensors;

    mmproj_spec() {
        const auto u32 = [&](const std::string& key, std::uint32_t v) { keys[key] = kv(key, value_type::uint32, v); };
        keys["clip.projector_type"] = kv_string("clip.projector_type", "idefics3");
        keys["clip.has_vision_encoder"] = kv_bool("clip.has_vision_encoder", true);
        keys["clip.use_gelu"] = kv_bool("clip.use_gelu", true);
        u32("clip.vision.image_size", image);
        u32("clip.vision.patch_size", patch);
        u32("clip.vision.embedding_length", n_embd);
        u32("clip.vision.feed_forward_length", n_ff);
        u32("clip.vision.block_count", 1);
        u32("clip.vision.attention.head_count", n_head);
        u32("clip.vision.projector.scale_factor", scale);
        u32("clip.vision.projection_dim", n_text);
        keys["clip.vision.attention.layer_norm_epsilon"] =
            kv("clip.vision.attention.layer_norm_epsilon", value_type::float32, 1e-6f);
        keys["clip.vision.image_mean"] = kv_floats("clip.vision.image_mean", {0.5f, 0.5f, 0.5f});
        keys["clip.vision.image_std"] = kv_floats("clip.vision.image_std", {0.5f, 0.5f, 0.5f});

        tensors["v.patch_embd.weight"] = {{patch, patch, 3, n_embd}};
        tensors["v.patch_embd.bias"] = {{n_embd}};
        tensors["v.position_embd.weight"] = {{n_embd, n_patches}};
        for (const char* n : {"ln1", "ln2"}) {
            tensors[std::string("v.blk.0.") + n + ".weight"] = {{n_embd}};
            tensors[std::string("v.blk.0.") + n + ".bias"] = {{n_embd}};
        }
        for (const char* m : {"attn_q", "attn_k", "attn_v", "attn_out"}) {
            tensors[std::string("v.blk.0.") + m + ".weight"] = {{n_embd, n_embd}};
            tensors[std::string("v.blk.0.") + m + ".bias"] = {{n_embd}};
        }
        tensors["v.blk.0.ffn_up.weight"] = {{n_embd, n_ff}};
        tensors["v.blk.0.ffn_up.bias"] = {{n_ff}};
        tensors["v.blk.0.ffn_down.weight"] = {{n_ff, n_embd}};
        tensors["v.blk.0.ffn_down.bias"] = {{n_embd}};
        tensors["v.post_ln.weight"] = {{n_embd}};
        tensors["v.post_ln.bias"] = {{n_embd}};
        tensors["mm.model.fc.weight"] = {{n_embd * scale * scale, n_text}};
    }

    std::vector<std::byte> bytes() const {
        constexpr std::size_t alignment = 32;
        const auto nbytes = [](const tensor& t) {
            std::uint64_t n = 1;
            for (auto d : t.ne) {
                n *= d;
            }
            return static_cast<std::size_t>(t.type == q5_1_tag ? n / 32 * 24 : n * sizeof(float));
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
            for (std::size_t i = 0; i < nbytes(t); ++i) {
                w.put(std::uint8_t{0});
            }
            w.pad_to(alignment);
        }
        return std::move(w.bytes());
    }
};

// Keeps a spec's image alive with the parsed file over it.
struct loaded {
    std::vector<std::byte> bytes;
    std::optional<gguf::gguf_file> file;
};

result<models::clip_model, models::model_error> load(const mmproj_spec& spec, loaded& keep) {
    keep.bytes = spec.bytes();
    auto f = gguf::parse(keep.bytes);
    REQUIRE(f);
    keep.file.emplace(std::move(*f));
    return models::load_clip(*keep.file);
}

void require_error(const mmproj_spec& spec, model_errc code, const std::string& name) {
    loaded keep;
    const auto m = load(spec, keep);
    REQUIRE_FALSE(m);
    INFO(to_string(m.error().code) << ": " << m.error().name);
    CHECK(m.error().code == code);
    CHECK(m.error().name == name);
}

}  // namespace

TEST_CASE("clip_loader: a well-formed idefics3 mmproj builds its encoder", "[clip][loader]") {
    loaded keep;
    const auto m = load(mmproj_spec{}, keep);
    if (!m) {
        FAIL(to_string(m.error().code) << ": " << m.error().name);
    }
    const auto& c = m->config;
    CHECK(c.grid() == 4);
    CHECK(c.n_patches() == 16);
    CHECK(c.n_tokens() == 4);
    CHECK(c.patch_floats() == 48);
    CHECK(c.image_mean[2] == 0.5f);
    CHECK(m->graph.full_batch_only());
    CHECK(m->graph.max_batch() == 16);
    CHECK(m->graph.tensors()[m->patches].size == 48);
    CHECK(m->graph.tensors()[m->embeddings].size == n_text);
    CHECK(m->graph.rows(m->embeddings) == 4);

    graph::session s{m->graph};
    std::vector<float> img(3 * image * image, 0.25f);
    std::vector<float> out(4 * n_text, -1.0f);
    CHECK(models::encode(s, *m, img, out));
    CHECK(out == std::vector<float>(4 * n_text, 0.0f));  // all-zero weights

    // SmolVLM's files store the FFN the other way round; both load.
    mmproj_spec swapped;
    swapped.tensors["v.blk.0.ffn_down.weight"] = {{n_embd, n_ff}};
    swapped.tensors["v.blk.0.ffn_down.bias"] = {{n_ff}};
    swapped.tensors["v.blk.0.ffn_up.weight"] = {{n_ff, n_embd}};
    swapped.tensors["v.blk.0.ffn_up.bias"] = {{n_embd}};
    loaded keep2;
    CHECK(load(swapped, keep2));
}

TEST_CASE("clip_loader: rejects keys it cannot honour", "[clip][loader]") {
    {
        mmproj_spec spec;
        spec.keys["clip.projector_type"] = kv_string("clip.projector_type", "gemma3");
        require_error(spec, model_errc::wrong_architecture, "clip.projector_type");
    }
    {
        mmproj_spec spec;
        spec.keys["clip.use_gelu"] = kv_bool("clip.use_gelu", false);
        require_error(spec, model_errc::unsupported, "clip.use_gelu");
    }
    {
        mmproj_spec spec;
        spec.keys["clip.has_audio_encoder"] = kv_bool("clip.has_audio_encoder", true);
        require_error(spec, model_errc::unsupported, "clip.has_audio_encoder");
    }
    {
        mmproj_spec spec;
        spec.keys.erase("clip.vision.image_size");
        require_error(spec, model_errc::missing_key, "clip.vision.image_size");
    }
    {
        mmproj_spec spec;
        spec.keys["clip.vision.patch_size"] = kv("clip.vision.patch_size", value_type::uint32, std::uint32_t{5});
        require_error(spec, model_errc::bad_value, "clip.vision.patch_size");
    }
    {
        mmproj_spec spec;
        spec.keys["clip.vision.attention.head_count"] =
            kv("clip.vision.attention.head_count", value_type::uint32, std::uint32_t{3});
        require_error(spec, model_errc::bad_value, "clip.vision.attention.head_count");
    }
    {
        mmproj_spec spec;  // 3 does not divide the 4 x 4 grid (llama.cpp would pad)
        spec.keys["clip.vision.projector.scale_factor"] =
            kv("clip.vision.projector.scale_factor", value_type::uint32, std::uint32_t{3});
        require_error(spec, model_errc::unsupported, "clip.vision.projector.scale_factor");
    }
    {
        mmproj_spec spec;
        spec.keys["clip.vision.image_mean"] = kv_floats("clip.vision.image_mean", {0.5f, 0.5f});
        require_error(spec, model_errc::bad_value, "clip.vision.image_mean");
    }
    {
        mmproj_spec spec;
        spec.keys.erase("clip.vision.attention.layer_norm_epsilon");
        require_error(spec, model_errc::missing_key, "clip.vision.attention.layer_norm_epsilon");
    }
}

TEST_CASE("clip_loader: rejects missing, misshapen, unsupported and unknown tensors", "[clip][loader]") {
    {
        mmproj_spec spec;
        spec.tensors.erase("v.blk.0.ln1.weight");
        require_error(spec, model_errc::missing_tensor, "v.blk.0.ln1.weight");
    }
    {
        mmproj_spec spec;
        spec.tensors["v.patch_embd.weight"] = {{patch, patch, 1, n_embd}};
        require_error(spec, model_errc::bad_tensor, "v.patch_embd.weight");
    }
    {
        mmproj_spec spec;
        spec.tensors["v.position_embd.weight"] = {{n_embd, n_patches - 1}};
        require_error(spec, model_errc::bad_tensor, "v.position_embd.weight");
    }
    {
        mmproj_spec spec;
        spec.tensors["mm.model.fc.weight"] = {{n_embd, n_text}};
        require_error(spec, model_errc::bad_tensor, "mm.model.fc.weight");
    }
    {
        mmproj_spec spec;
        spec.tensors["v.blk.0.attn_q.weight"] = {{n_embd * 4, n_embd / 4}, q5_1_tag};  // 32 x 2, whole blocks
        require_error(spec, model_errc::bad_tensor, "v.blk.0.attn_q.weight");
    }
    {
        mmproj_spec spec;  // 32 columns: whole Q5_1 blocks, but no scheme
        spec.tensors["mm.model.fc.weight"] = {{n_embd * scale * scale, n_text}, q5_1_tag};
        require_error(spec, model_errc::unsupported_type, "mm.model.fc.weight");
    }
    for (const char* extra : {"v.pre_ln.weight",
                              "v.class_embd",
                              "mm.model.fc.bias",
                              "v.blk.0.attn_qkv.weight",
                              "v.blk.0.ls1.weight",
                              "v.blk.0.ffn_gate.weight"}) {
        mmproj_spec spec;
        spec.tensors[extra] = {{n_embd}};
        require_error(spec, model_errc::unsupported, extra);
    }
}
