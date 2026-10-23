// A minimal GGUF writer for tests: each test states exactly which bytes it
// feeds the parser or the model loader.
#pragma once

#include <branes/synapse/gguf/metadata.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace synapse_test::gguf_writer {

using branes::synapse::gguf::value_type;

class writer {
public:
    template <typename T>
    writer& put(T v) {
        append(&v, sizeof(T));
        return *this;
    }
    writer& put_string(std::string_view s) {
        put<std::uint64_t>(s.size());
        append(s.data(), s.size());
        return *this;
    }
    writer& pad_to(std::size_t alignment) {
        while (bytes_.size() % alignment != 0) {
            bytes_.push_back(std::byte{0});
        }
        return *this;
    }
    std::vector<std::byte>& bytes() {
        return bytes_;
    }

private:
    // resize + memcpy rather than vector::insert: GCC 12/13 at -O2 reports a
    // false -Wstringop-overflow on small inserted ranges.
    void append(const void* p, std::size_t n) {
        const std::size_t old = bytes_.size();
        bytes_.resize(old + n);
        if (n != 0) {
            std::memcpy(bytes_.data() + old, p, n);
        }
    }

    std::vector<std::byte> bytes_;
};

using kv_fn = std::function<void(writer&)>;

template <typename T>
kv_fn kv(std::string_view key, value_type type, T v) {
    // Owned copies: a key built at run time must outlive the call.
    return [key = std::string(key), type, v](writer& w) {
        w.put_string(key).put(static_cast<std::uint32_t>(type)).put(v);
    };
}

inline kv_fn kv_string(std::string_view key, std::string_view v) {
    return [key = std::string(key), v = std::string(v)](writer& w) {
        w.put_string(key).put(static_cast<std::uint32_t>(value_type::string)).put_string(v);
    };
}

struct tensor_spec {
    std::string_view name;
    std::vector<std::uint64_t> ne;
    std::uint32_t type;
    std::uint64_t offset;
};

inline std::vector<std::byte> build(std::uint32_t version,
                                    const std::vector<kv_fn>& kvs,
                                    const std::vector<tensor_spec>& tensors,
                                    std::size_t data_bytes,
                                    std::size_t alignment = 32) {
    writer w;
    w.put<std::uint32_t>(0x46554747);  // "GGUF"
    w.put(version).put<std::uint64_t>(tensors.size()).put<std::uint64_t>(kvs.size());
    for (const auto& f : kvs) {
        f(w);
    }
    for (const auto& t : tensors) {
        w.put_string(t.name).put<std::uint32_t>(static_cast<std::uint32_t>(t.ne.size()));
        for (auto d : t.ne) {
            w.put(d);
        }
        w.put(t.type).put(t.offset);
    }
    w.pad_to(alignment);
    for (std::size_t i = 0; i < data_bytes; ++i) {
        w.bytes().push_back(static_cast<std::byte>(i & 0xff));
    }
    return w.bytes();
}

}  // namespace synapse_test::gguf_writer
