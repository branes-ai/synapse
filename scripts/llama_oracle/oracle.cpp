// llama_oracle — golden logits from llama.cpp itself.
//
// Built and run by scripts/gen_llama_fixture.py against a local llama.cpp
// checkout. Two modes:
//
//   oracle tiny <model.gguf> <golden.gguf>
//       Writes a small random-weight Llama model (2 layers, grouped-query
//       attention, llama3-style rope_freqs, tied output embedding, a mix of
//       Q4_K/Q6_K/Q8_0/Q4_0/F32 weights quantized by ggml), then decodes
//       a fixed token sequence through it with libllama.
//
//   oracle tiny-f32 <model.gguf> <golden.gguf>
//       The same with every weight F32 and smaller widths: no activation is
//       quantized on either side, so the logits must agree to float rounding
//       (this checks the graph's semantics apart from quantization noise).
//
//   oracle logits <model.gguf> <golden.gguf> <token>...
//       Decodes the given tokens through an existing model.
//
//   oracle tokenize <vocab.gguf> <cases.txt> <golden.gguf>
//       Tokenizes each case of cases.txt (separated by lines holding
//       __ggml_vocab_test__, llama.cpp's own test-vector format) with
//       llama_tokenize, BOS added, once without and once with special-token
//       parsing; and detokenizes the latter with llama_token_to_piece. The
//       golden file holds, per case i, synapse.tok.<i>.text, .plain and
//       .special (token arrays) and .detok, plus synapse.tok.count.
//
// Tokens are decoded one per llama_decode call (the decode path synapse
// implements), with flash attention off and the default F16 KV cache. The
// golden file holds the token ids (metadata array synapse.golden.tokens) and
// the logits after each token (tensor "logits", n_vocab x n_tokens, F32).

#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr uint32_t n_layer = 2;
constexpr uint32_t n_head = 4;
constexpr uint32_t n_head_kv = 2;
constexpr uint32_t n_ctx_train = 128;
constexpr int n_tokens = 16;

struct layer_types {
    ggml_type q, k, v, o, gate, up, down;
};

struct tiny_config {
    uint32_t n_embd, n_ff, n_vocab;
    ggml_type embd;
    layer_types layers[n_layer];

    uint32_t head_dim() const {
        return n_embd / n_head;
    }
};

// K-quant rows need 256 columns.
constexpr tiny_config quantized{
    256,
    512,
    512,
    GGML_TYPE_Q6_K,
    {{GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K},
     {GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_F32, GGML_TYPE_Q4_K}}};

constexpr layer_types all_f32{
    GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32};
constexpr tiny_config plain{64, 128, 128, GGML_TYPE_F32, {all_f32, all_f32}};

std::mt19937 rng(20261021);

// A rows x cols weight of `type` with N(0, scale) entries, quantized by ggml.
void add_weight(ggml_context* ctx,
                gguf_context* out,
                const std::string& name,
                ggml_type type,
                int64_t rows,
                int64_t cols,
                float mean,
                float scale) {
    std::normal_distribution<float> normal(mean, scale);
    std::vector<float> w(static_cast<size_t>(rows * cols));
    for (auto& v : w) {
        v = normal(rng);
    }
    ggml_tensor* t = rows == 1 ? ggml_new_tensor_1d(ctx, type, cols) : ggml_new_tensor_2d(ctx, type, cols, rows);
    ggml_set_name(t, name.c_str());
    ggml_quantize_chunk(type, w.data(), t->data, 0, rows, cols, nullptr);
    gguf_add_tensor(out, t);
}

void write_tiny_model(const tiny_config& cfg, const char* path) {
    const uint32_t n_embd = cfg.n_embd;
    const uint32_t n_ff = cfg.n_ff;
    const uint32_t n_vocab = cfg.n_vocab;
    const uint32_t head_dim = cfg.head_dim();
    ggml_init_params params{};
    params.mem_size = 64 * 1024 * 1024;
    ggml_context* ctx = ggml_init(params);
    gguf_context* out = gguf_init_empty();

    gguf_set_val_str(out, "general.architecture", "llama");
    gguf_set_val_str(out, "general.name", "synapse tiny llama (random weights)");
    gguf_set_val_u32(out, "llama.block_count", n_layer);
    gguf_set_val_u32(out, "llama.context_length", n_ctx_train);
    gguf_set_val_u32(out, "llama.embedding_length", n_embd);
    gguf_set_val_u32(out, "llama.feed_forward_length", n_ff);
    gguf_set_val_u32(out, "llama.attention.head_count", n_head);
    gguf_set_val_u32(out, "llama.attention.head_count_kv", n_head_kv);
    gguf_set_val_f32(out, "llama.rope.freq_base", 10000.0f);
    gguf_set_val_f32(out, "llama.attention.layer_norm_rms_epsilon", 1e-5f);
    gguf_set_val_u32(out, "llama.rope.dimension_count", head_dim);
    gguf_set_val_u32(out, "llama.vocab_size", n_vocab);
    gguf_set_val_str(out, "tokenizer.ggml.model", "none");

    const float wscale = 1.0f / std::sqrt(static_cast<float>(n_embd));
    add_weight(ctx, out, "token_embd.weight", cfg.embd, n_vocab, n_embd, 0.0f, 0.5f);
    add_weight(ctx, out, "output_norm.weight", GGML_TYPE_F32, 1, n_embd, 1.0f, 0.1f);

    // llama3-style frequency factors: 1 for high frequencies, up to 8 for low.
    ggml_tensor* rf = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim / 2);
    ggml_set_name(rf, "rope_freqs.weight");
    for (uint32_t i = 0; i < head_dim / 2; ++i) {
        static_cast<float*>(rf->data)[i] = 1.0f + 7.0f * static_cast<float>(i) / static_cast<float>(head_dim / 2 - 1);
    }
    gguf_add_tensor(out, rf);

    for (uint32_t l = 0; l < n_layer; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        const layer_types& t = cfg.layers[l];
        add_weight(ctx, out, p + "attn_norm.weight", GGML_TYPE_F32, 1, n_embd, 1.0f, 0.1f);
        add_weight(ctx, out, p + "attn_q.weight", t.q, n_head * head_dim, n_embd, 0.0f, wscale);
        add_weight(ctx, out, p + "attn_k.weight", t.k, n_head_kv * head_dim, n_embd, 0.0f, wscale);
        add_weight(ctx, out, p + "attn_v.weight", t.v, n_head_kv * head_dim, n_embd, 0.0f, wscale);
        add_weight(ctx, out, p + "attn_output.weight", t.o, n_embd, n_head * head_dim, 0.0f, wscale);
        add_weight(ctx, out, p + "ffn_norm.weight", GGML_TYPE_F32, 1, n_embd, 1.0f, 0.1f);
        add_weight(ctx, out, p + "ffn_gate.weight", t.gate, n_ff, n_embd, 0.0f, wscale);
        add_weight(ctx, out, p + "ffn_up.weight", t.up, n_ff, n_embd, 0.0f, wscale);
        add_weight(
            ctx, out, p + "ffn_down.weight", t.down, n_embd, n_ff, 0.0f, 1.0f / std::sqrt(static_cast<float>(n_ff)));
    }
    if (!gguf_write_to_file(out, path, false)) {
        std::fprintf(stderr, "cannot write %s\n", path);
        std::exit(1);
    }
    gguf_free(out);
    ggml_free(ctx);
}

void quiet_log(ggml_log_level level, const char* text, void*) {
    if (level >= GGML_LOG_LEVEL_WARN) {
        std::fputs(text, stderr);
    }
}

void write_golden(const char* model_path, const char* golden_path, const std::vector<llama_token>& tokens) {
    llama_log_set(quiet_log, nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    llama_model* model = llama_model_load_from_file(model_path, mp);
    if (model == nullptr) {
        std::fprintf(stderr, "cannot load %s\n", model_path);
        std::exit(1);
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 256;
    cp.n_batch = 1;
    cp.n_ubatch = 1;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.no_perf = true;
    llama_context* lctx = llama_init_from_model(model, cp);
    if (lctx == nullptr) {
        std::fprintf(stderr, "cannot create a context\n");
        std::exit(1);
    }
    const int32_t vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    ggml_init_params params{};
    params.mem_size = static_cast<size_t>(vocab) * tokens.size() * sizeof(float) + 1024 * 1024;
    ggml_context* ctx = ggml_init(params);
    ggml_tensor* logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, vocab, static_cast<int64_t>(tokens.size()));
    ggml_set_name(logits, "logits");
    for (size_t i = 0; i < tokens.size(); ++i) {
        llama_token tok = tokens[i];
        if (llama_decode(lctx, llama_batch_get_one(&tok, 1)) != 0) {
            std::fprintf(stderr, "llama_decode failed at token %zu\n", i);
            std::exit(1);
        }
        std::memcpy(static_cast<float*>(logits->data) + i * static_cast<size_t>(vocab),
                    llama_get_logits_ith(lctx, -1),
                    static_cast<size_t>(vocab) * sizeof(float));
    }

    gguf_context* out = gguf_init_empty();
    gguf_set_val_str(out, "general.architecture", "synapse-test");
    gguf_set_val_str(out, "synapse.fixture.generator", "scripts/gen_llama_fixture.py");
    std::vector<uint32_t> ids(tokens.begin(), tokens.end());
    gguf_set_arr_data(out, "synapse.golden.tokens", GGUF_TYPE_UINT32, ids.data(), ids.size());
    gguf_add_tensor(out, logits);
    if (!gguf_write_to_file(out, golden_path, false)) {
        std::fprintf(stderr, "cannot write %s\n", golden_path);
        std::exit(1);
    }
    gguf_free(out);
    ggml_free(ctx);
    llama_free(lctx);
    llama_model_free(model);
    llama_backend_free();
}

std::vector<llama_token> tokenize(const llama_vocab* vocab, const std::string& text, bool parse_special) {
    std::vector<llama_token> out(text.size() + 8);
    int32_t n = llama_tokenize(vocab,
                               text.data(),
                               static_cast<int32_t>(text.size()),
                               out.data(),
                               static_cast<int32_t>(out.size()),
                               true,
                               parse_special);
    if (n < 0) {
        out.resize(static_cast<size_t>(-n));
        n = llama_tokenize(vocab,
                           text.data(),
                           static_cast<int32_t>(text.size()),
                           out.data(),
                           static_cast<int32_t>(out.size()),
                           true,
                           parse_special);
    }
    out.resize(static_cast<size_t>(n));
    return out;
}

void write_tokenizer_golden(const char* vocab_path, const char* cases_path, const char* golden_path) {
    std::FILE* in = std::fopen(cases_path, "rb");
    if (in == nullptr) {
        std::fprintf(stderr, "cannot read %s\n", cases_path);
        std::exit(1);
    }
    std::string all;
    char buf[4096];
    for (size_t n; (n = std::fread(buf, 1, sizeof buf, in)) > 0;) {
        all.append(buf, n);
    }
    std::fclose(in);
    std::vector<std::string> cases;
    const std::string sep = "\n__ggml_vocab_test__\n";
    for (size_t at = 0;;) {
        const size_t m = all.find(sep, at);
        if (m == std::string::npos) {
            break;
        }
        cases.push_back(all.substr(at, m - at));
        at = m + sep.size();
    }

    llama_log_set(quiet_log, nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model* model = llama_model_load_from_file(vocab_path, mp);
    if (model == nullptr) {
        std::fprintf(stderr, "cannot load %s\n", vocab_path);
        std::exit(1);
    }
    const llama_vocab* vocab = llama_model_get_vocab(model);

    gguf_context* out = gguf_init_empty();
    gguf_set_val_str(out, "general.architecture", "synapse-test");
    gguf_set_val_str(out, "synapse.fixture.generator", "scripts/gen_tokenizer_fixture.py");
    gguf_set_val_u32(out, "synapse.tok.count", static_cast<uint32_t>(cases.size()));
    for (size_t i = 0; i < cases.size(); ++i) {
        const std::string key = "synapse.tok." + std::to_string(i) + ".";
        gguf_set_val_str(out, (key + "text").c_str(), cases[i].c_str());
        for (bool special : {false, true}) {
            const auto toks = tokenize(vocab, cases[i], special);
            std::vector<uint32_t> ids(toks.begin(), toks.end());
            gguf_set_arr_data(
                out, (key + (special ? "special" : "plain")).c_str(), GGUF_TYPE_UINT32, ids.data(), ids.size());
            if (special) {
                std::string detok;
                for (llama_token t : toks) {
                    char piece[512];
                    const int32_t n = llama_token_to_piece(vocab, t, piece, sizeof piece, 0, true);
                    detok.append(piece, static_cast<size_t>(n > 0 ? n : 0));
                }
                gguf_set_val_str(out, (key + "detok").c_str(), detok.c_str());
            }
        }
    }
    if (!gguf_write_to_file(out, golden_path, false)) {
        std::fprintf(stderr, "cannot write %s\n", golden_path);
        std::exit(1);
    }
    gguf_free(out);
    llama_model_free(model);
    llama_backend_free();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 4 && (std::strcmp(argv[1], "tiny") == 0 || std::strcmp(argv[1], "tiny-f32") == 0)) {
        const tiny_config& cfg = std::strcmp(argv[1], "tiny") == 0 ? quantized : plain;
        write_tiny_model(cfg, argv[2]);
        std::uniform_int_distribution<llama_token> pick(0, static_cast<llama_token>(cfg.n_vocab) - 1);
        std::vector<llama_token> tokens(n_tokens);
        for (auto& t : tokens) {
            t = pick(rng);
        }
        write_golden(argv[2], argv[3], tokens);
        return 0;
    }
    if (argc == 5 && std::strcmp(argv[1], "tokenize") == 0) {
        write_tokenizer_golden(argv[2], argv[3], argv[4]);
        return 0;
    }
    if (argc >= 5 && std::strcmp(argv[1], "logits") == 0) {
        std::vector<llama_token> tokens;
        for (int i = 4; i < argc; ++i) {
            tokens.push_back(static_cast<llama_token>(std::strtol(argv[i], nullptr, 10)));
        }
        write_golden(argv[2], argv[3], tokens);
        return 0;
    }
    std::fprintf(stderr,
                 "usage: %s tiny|tiny-f32 <model.gguf> <golden.gguf>\n"
                 "       %s logits <model.gguf> <golden.gguf> <token>...\n"
                 "       %s tokenize <vocab.gguf> <cases.txt> <golden.gguf>\n",
                 argv[0],
                 argv[0],
                 argv[0]);
    return 2;
}
