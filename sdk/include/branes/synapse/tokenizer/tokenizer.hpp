#pragma once
/// @file tokenizer.hpp
/// @brief The tokenizer a GGUF file asks for, behind one interface.
///
/// tokenizer.ggml.model selects the algorithm:
///
///   "gpt2"   byte-level BPE (bpe.hpp; Llama 3 and Qwen2 pre-tokenizers)
///   "llama"  SentencePiece (spm.hpp; Llama 2, TinyLlama, Mistral v1)
///
/// Anything else is unsupported_model. Both report errors instead of
/// throwing for a malformed (untrusted) vocabulary.

#include <branes/synapse/gguf/reader.hpp>
#include <branes/synapse/result.hpp>
#include <branes/synapse/tokenizer/bpe.hpp>
#include <branes/synapse/tokenizer/spm.hpp>
#include <branes/synapse/tokenizer/vocab.hpp>

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace branes::synapse::tokenizer {

class any_tokenizer {
public:
    static result<any_tokenizer, tokenizer_error> from_gguf(const gguf::gguf_file& f) {
        const auto model = f.get<std::string_view>("tokenizer.ggml.model");
        if (!model) {
            return tokenizer_error{tokenizer_errc::missing_key, "tokenizer.ggml.model"};
        }
        if (*model == "gpt2") {
            auto t = bpe_tokenizer::from_gguf(f);
            if (!t) {
                return t.error();
            }
            return any_tokenizer(std::move(*t));
        }
        if (*model == "llama") {
            auto t = spm_tokenizer::from_gguf(f);
            if (!t) {
                return t.error();
            }
            return any_tokenizer(std::move(*t));
        }
        return tokenizer_error{tokenizer_errc::unsupported_model, "tokenizer.ggml.model"};
    }

    /// The algorithm in use: "bpe" or "spm".
    std::string_view kind() const noexcept {
        return std::holds_alternative<bpe_tokenizer>(t_) ? "bpe" : "spm";
    }
    std::size_t size() const noexcept {
        return std::visit([](const auto& t) { return t.size(); }, t_);
    }
    std::uint32_t bos() const noexcept {
        return std::visit([](const auto& t) { return t.bos(); }, t_);
    }
    std::uint32_t eos() const noexcept {
        return std::visit([](const auto& t) { return t.eos(); }, t_);
    }
    token_type type(std::uint32_t id) const noexcept {
        return std::visit([id](const auto& t) { return t.type(id); }, t_);
    }
    std::vector<std::uint32_t> encode(std::string_view text, encode_options opt = {}) const {
        return std::visit([&](const auto& t) { return t.encode(text, opt); }, t_);
    }
    std::string decode(std::uint32_t id, bool special = false) const {
        return std::visit([&](const auto& t) { return t.decode(id, special); }, t_);
    }
    std::string decode(std::span<const std::uint32_t> ids, bool special = false) const {
        return std::visit([&](const auto& t) { return t.decode(ids, special); }, t_);
    }

private:
    explicit any_tokenizer(std::variant<bpe_tokenizer, spm_tokenizer> t) : t_(std::move(t)) {}

    std::variant<bpe_tokenizer, spm_tokenizer> t_;
};

}  // namespace branes::synapse::tokenizer
