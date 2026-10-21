// llama_decode — greedy decoding of a Llama-architecture GGUF model.
//
//   llama_decode <model.gguf> [-n <new tokens>] [-c <context>] [<token id>...]
//
// Feeds the prompt token ids one per step (prefill as repeated decode), then
// generates -n more by taking the most likely next token each step. Prints
// the tokens as text when the file carries a vocabulary, and the decode rate.
// Without prompt tokens it starts from the model's BOS token.
//
// synapse has no tokenizer yet, so the prompt is given as token ids; the
// vocabulary is only used to show the output.

#include <branes/synapse/gguf/mapped_file.hpp>
#include <branes/synapse/gguf/reader.hpp>
#include <branes/synapse/graph/session.hpp>
#include <branes/synapse/models/llama.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

using namespace branes::synapse;

namespace {

// UTF-8 code points of s.
std::vector<std::uint32_t> code_points(std::string_view s) {
    std::vector<std::uint32_t> cps;
    for (std::size_t i = 0; i < s.size();) {
        const auto c = static_cast<unsigned char>(s[i]);
        const std::size_t n = c < 0x80 ? 1 : c < 0xe0 ? 2 : c < 0xf0 ? 3 : 4;
        std::uint32_t cp = n == 1 ? c : c & (0x7fu >> n);
        for (std::size_t j = 1; j < n && i + j < s.size(); ++j) {
            cp = (cp << 6) | (static_cast<unsigned char>(s[i + j]) & 0x3fu);
        }
        cps.push_back(cp);
        i += n;
    }
    return cps;
}

// A vocabulary entry as text. GPT-2-style BPE vocabularies (Llama 3) store
// each byte as a printable code point; SentencePiece ones (Llama 2) mark
// spaces with U+2581 and store raw bytes as <0xNN>.
class detokenizer {
public:
    explicit detokenizer(const gguf::gguf_file& f) : tokens_(f.get_array("tokenizer.ggml.tokens")) {
        gpt2_ = f.get<std::string_view>("tokenizer.ggml.model") == std::string_view{"gpt2"};
        // GPT-2's bytes_to_unicode: printable bytes stand for themselves, the
        // rest are shifted to code points 256 and up, in byte order.
        int shifted = 0;
        for (int b = 0; b < 256; ++b) {
            const bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
            byte_of_[static_cast<std::size_t>(printable ? b : 256 + shifted++)] = static_cast<char>(b);
        }
    }

    std::string text(std::uint32_t token) const {
        if (tokens_ == nullptr || token >= tokens_->strings().size()) {
            return "[" + std::to_string(token) + "]";
        }
        const std::string_view piece = tokens_->strings()[token];
        std::string out;
        if (gpt2_) {
            for (std::uint32_t cp : code_points(piece)) {
                if (cp < byte_of_.size()) {
                    out += byte_of_[cp];
                }
            }
        } else if (piece.size() == 6 && piece.starts_with("<0x") && piece.ends_with(">")) {
            out += static_cast<char>(std::strtol(std::string(piece.substr(3, 2)).c_str(), nullptr, 16));
        } else {
            for (std::size_t i = 0; i < piece.size(); ++i) {
                if (piece.compare(i, 3, "\xe2\x96\x81") == 0) {  // U+2581
                    out += ' ';
                    i += 2;
                } else {
                    out += piece[i];
                }
            }
        }
        return out;
    }

private:
    const gguf::array_value* tokens_;
    bool gpt2_ = false;
    std::vector<char> byte_of_ = std::vector<char>(512, '?');
};

int usage(const char* argv0) {
    std::fprintf(stderr, "usage: %s <model.gguf> [-n <new tokens>] [-c <context>] [<token id>...]\n", argv0);
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        return usage(argv[0]);
    }
    std::size_t n_new = 32;
    std::size_t n_ctx = 0;
    std::vector<std::uint32_t> prompt;
    for (int i = 2; i < argc; ++i) {
        const std::string_view a = argv[i];
        if ((a == "-n" || a == "-c") && i + 1 < argc) {
            (a == "-n" ? n_new : n_ctx) = std::strtoull(argv[++i], nullptr, 10);
        } else {
            char* end = nullptr;
            const unsigned long long t = std::strtoull(argv[i], &end, 10);
            if (end == argv[i] || *end != '\0') {
                return usage(argv[0]);
            }
            prompt.push_back(static_cast<std::uint32_t>(t));
        }
    }

    auto file = gguf::mapped_file::open(argv[1]);
    if (!file) {
        std::fprintf(stderr, "cannot open %s: %s\n", argv[1], std::string(to_string(file.error().code)).c_str());
        return 1;
    }
    auto parsed = gguf::parse(file->bytes());
    if (!parsed) {
        std::fprintf(stderr,
                     "cannot parse %s: %s at byte %llu\n",
                     argv[1],
                     std::string(to_string(parsed.error().code)).c_str(),
                     static_cast<unsigned long long>(parsed.error().offset));
        return 1;
    }
    if (prompt.empty()) {
        if (const auto bos = parsed->get<std::uint32_t>("tokenizer.ggml.bos_token_id")) {
            prompt.push_back(*bos);
        } else {
            std::fprintf(stderr, "no prompt tokens and no BOS token in the model\n");
            return 1;
        }
    }
    if (n_ctx == 0) {
        n_ctx = prompt.size() + n_new;
    }
    auto model = models::load_llama(*parsed, n_ctx);
    if (!model) {
        const auto& e = model.error();
        std::fprintf(stderr, "cannot load the model: %s %s", std::string(to_string(e.code)).c_str(), e.name.c_str());
        if (e.code == models::model_errc::graph) {
            std::fprintf(stderr, " (%s at node %zu)", std::string(to_string(e.graph.code)).c_str(), e.graph.node);
        }
        std::fprintf(stderr, "\n");
        return 1;
    }
    const auto& c = model->config;
    graph::session s{model->graph};
    std::fprintf(stderr,
                 "llama: %zu layers, d=%zu, ff=%zu, %zu heads (%zu KV), vocab %zu\n"
                 "memory: arena %.1f KiB, KV cache %.1f MiB for %zu positions\n",
                 c.n_layer,
                 c.n_embd,
                 c.n_ff,
                 c.n_head,
                 c.n_head_kv,
                 c.n_vocab,
                 static_cast<double>(s.memory().arena_bytes) / 1024.0,
                 static_cast<double>(s.state_bytes()) / (1024.0 * 1024.0),
                 n_ctx);

    const detokenizer vocab(*parsed);
    using clock = std::chrono::steady_clock;
    std::uint32_t pos = 0;
    std::uint32_t next = 0;
    const auto step = [&](std::uint32_t token) {
        if (!s.run({token, pos++})) {
            std::fprintf(stderr, "\ntoken %u or position %u out of range\n", token, pos - 1);
            std::exit(1);
        }
        const auto logits = s.tensor(model->logits);
        next = static_cast<std::uint32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
    };

    const auto t0 = clock::now();
    for (std::uint32_t t : prompt) {
        std::printf("%s", vocab.text(t).c_str());
        step(t);
    }
    std::fflush(stdout);
    const auto t1 = clock::now();
    for (std::size_t i = 0; i < n_new && pos < n_ctx; ++i) {
        const std::uint32_t t = next;
        std::printf("%s", vocab.text(t).c_str());
        std::fflush(stdout);
        step(t);
    }
    const auto t2 = clock::now();
    std::printf("\n");
    std::fflush(stdout);

    const auto secs = [](auto d) { return std::chrono::duration<double>(d).count(); };
    std::fprintf(stderr,
                 "prefill: %zu tokens, %.2f tok/s\ndecode:  %zu tokens, %.2f tok/s\n",
                 prompt.size(),
                 static_cast<double>(prompt.size()) / secs(t1 - t0),
                 static_cast<std::size_t>(pos) - prompt.size(),
                 static_cast<double>(pos - prompt.size()) / secs(t2 - t1));
    return 0;
}
