// Portable access to environment variables for tests (MSVC deprecates
// std::getenv and has no setenv).
#pragma once

#include <cstdlib>
#include <optional>
#include <string>

namespace synapse_test {

/// The environment variable `name`, if set.
inline std::optional<std::string> get_env(const char* name) {
#if defined(_MSC_VER)
    char* value = nullptr;
    std::size_t len = 0;
    if (_dupenv_s(&value, &len, name) != 0 || value == nullptr) {
        return std::nullopt;
    }
    std::string s(value);
    std::free(value);
    return s;
#else
    const char* value = std::getenv(name);
    return value == nullptr ? std::nullopt : std::optional<std::string>(value);
#endif
}

/// Set the environment variable `name` to `value` (overwriting).
inline void set_env(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

}  // namespace synapse_test
