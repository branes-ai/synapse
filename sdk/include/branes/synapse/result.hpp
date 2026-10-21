#pragma once
/// @file result.hpp
/// @brief result<T, E>: a value or an error, for the no-throw entry points.
///
/// Every synapse entry point that can fail on untrusted input (loading a
/// model, building a graph from model metadata) returns a result instead of
/// throwing, because those paths often run with exceptions disabled.

#include <cassert>
#include <type_traits>
#include <utility>
#include <variant>

namespace branes::synapse {

/// Either a T or an E. A minimal stand-in for C++23 std::expected.
///
/// T and E must be distinct types, so the converting constructors are
/// unambiguous.
template <typename T, typename E>
class result {
    static_assert(!std::is_same_v<T, E>, "result<T, E> needs distinct T and E");

public:
    result(T value) : v_(std::move(value)) {}
    result(E err) : v_(std::move(err)) {}

    bool has_value() const noexcept {
        return v_.index() == 0;
    }
    explicit operator bool() const noexcept {
        return has_value();
    }

    /// Pre: has_value().
    T& value() & {
        assert(has_value());
        return *std::get_if<0>(&v_);
    }
    const T& value() const& {
        assert(has_value());
        return *std::get_if<0>(&v_);
    }
    T&& value() && {
        assert(has_value());
        return std::move(*std::get_if<0>(&v_));
    }
    T& operator*() & {
        return value();
    }
    const T& operator*() const& {
        return value();
    }
    T* operator->() {
        return &value();
    }
    const T* operator->() const {
        return &value();
    }

    /// Pre: !has_value().
    const E& error() const {
        assert(!has_value());
        return *std::get_if<1>(&v_);
    }

private:
    std::variant<T, E> v_;
};

}  // namespace branes::synapse
