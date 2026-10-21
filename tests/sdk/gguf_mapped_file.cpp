// Unit tests for branes::synapse::gguf::mapped_file: map a real file, parse it
// through the mapping, and check the RAII and failure contracts.

#include <branes/synapse/gguf/mapped_file.hpp>
#include <branes/synapse/gguf/reader.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

using namespace branes::synapse::gguf;

namespace {

// A temporary file removed on scope exit.
class temp_file {
public:
    temp_file(const std::string& name, const std::vector<char>& contents)
        : path_(std::filesystem::temp_directory_path() / name) {
        std::ofstream out(path_, std::ios::binary);
        out.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    }
    ~temp_file() {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }
    temp_file(const temp_file&) = delete;
    temp_file& operator=(const temp_file&) = delete;

    const std::filesystem::path& path() const {
        return path_;
    }

private:
    std::filesystem::path path_;
};

// resize + memcpy rather than vector::insert: GCC 12/13 at -O2 reports a false
// -Wstringop-overflow on small inserted ranges.
void append(std::vector<char>& out, const void* p, std::size_t n) {
    const std::size_t old = out.size();
    out.resize(old + n);
    std::memcpy(out.data() + old, p, n);
}

template <typename T>
void put(std::vector<char>& out, T v) {
    append(out, &v, sizeof(T));
}

// A GGUF v3 file with one key, general.name = "mapped", and no tensors.
std::vector<char> tiny_gguf() {
    std::vector<char> out;
    put<std::uint32_t>(out, 0x46554747);
    put<std::uint32_t>(out, 3);
    put<std::uint64_t>(out, 0);
    put<std::uint64_t>(out, 1);
    const std::string key = "general.name";
    const std::string value = "mapped";
    put<std::uint64_t>(out, key.size());
    append(out, key.data(), key.size());
    put<std::uint32_t>(out, 8);  // string
    put<std::uint64_t>(out, value.size());
    append(out, value.data(), value.size());
    return out;
}

}  // namespace

TEST_CASE("mapped_file: maps a file and parses it in place", "[gguf][mapped_file]") {
    const auto contents = tiny_gguf();
    const temp_file file{"synapse_mapped_file_test.gguf", contents};

    auto mapped = mapped_file::open(file.path());
    REQUIRE(mapped);
    CHECK(mapped->is_open());
    CHECK(mapped->size() == contents.size());

    const auto r = parse(mapped->bytes());
    REQUIRE(r);
    CHECK(r->get<std::string_view>("general.name") == "mapped");
}

TEST_CASE("mapped_file: moving transfers the mapping", "[gguf][mapped_file]") {
    const temp_file file{"synapse_mapped_file_move.gguf", tiny_gguf()};
    auto mapped = mapped_file::open(file.path());
    REQUIRE(mapped);

    mapped_file a = std::move(*mapped);
    CHECK(a.is_open());
    CHECK_FALSE(mapped->is_open());  // NOLINT(bugprone-use-after-move): moved-from is empty by contract
    CHECK(mapped->size() == 0);

    mapped_file b;
    CHECK_FALSE(b.is_open());
    b = std::move(a);
    CHECK(b.is_open());
    CHECK_FALSE(a.is_open());  // NOLINT(bugprone-use-after-move)
    CHECK(b.bytes().size() == tiny_gguf().size());
}

TEST_CASE("mapped_file: reports missing and empty files", "[gguf][mapped_file]") {
    const auto missing = mapped_file::open(std::filesystem::temp_directory_path() / "synapse_no_such_file.gguf");
    REQUIRE_FALSE(missing);
    CHECK(missing.error().code == errc::io_error);
    CHECK(missing.error().os_code != 0);

    const temp_file empty{"synapse_mapped_file_empty.gguf", {}};
    const auto r = mapped_file::open(empty.path());
    REQUIRE_FALSE(r);
    CHECK(r.error().code == errc::truncated);
}
