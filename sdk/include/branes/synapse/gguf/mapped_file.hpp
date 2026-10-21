#pragma once
/// @file mapped_file.hpp
/// @brief Read-only memory mapping of a model file (POSIX mmap / Windows file mapping).
///
/// Mapping instead of reading means a multi-gigabyte model costs no heap
/// memory and no copy: the OS pages weights in on first touch, and tensor
/// views from the GGUF parser point straight into the mapping.
///
/// Model
/// -----
///   mapped_file
///     data_ : const byte* [owned mapping, read-only] — start of the mapping
///     size_ : size_t — mapped length (the file size)
///
/// Invariants
/// ----------
///   I1: data_ != nullptr iff the object owns a mapping, and then size_ > 0;
///       otherwise size_ == 0                           — constrains data_, size_
///
/// Move-only. A moved-from mapped_file is empty. The file and mapping handles
/// are closed as soon as the view exists; only the view is kept, and it is
/// released by the destructor.

#include <branes/synapse/gguf/error.hpp>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <span>
#include <utility>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>

#include <cerrno>

#include <fcntl.h>
#include <unistd.h>
#endif

namespace branes::synapse::gguf {

class mapped_file {
public:
    /// Map `path` read-only.
    ///
    ///   Pre:  none.
    ///   Post: on success, size() is the file size and bytes() stays valid
    ///         until this object is destroyed or moved from. An empty file
    ///         fails with errc::truncated (no GGUF image is empty); OS failures
    ///         give errc::io_error with errno / GetLastError() in os_code.
    static result<mapped_file> open(const std::filesystem::path& path) noexcept {
#if defined(_WIN32)
        HANDLE file = ::CreateFileW(
            path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file == INVALID_HANDLE_VALUE) {
            return os_failure(static_cast<int>(::GetLastError()));
        }
        LARGE_INTEGER file_size{};
        if (!::GetFileSizeEx(file, &file_size)) {
            const auto err = static_cast<int>(::GetLastError());
            ::CloseHandle(file);
            return os_failure(err);
        }
        if (file_size.QuadPart <= 0) {
            ::CloseHandle(file);
            return error_info{errc::truncated, 0, 0};
        }
        if (static_cast<std::uint64_t>(file_size.QuadPart) > std::numeric_limits<std::size_t>::max()) {
            ::CloseHandle(file);
            return os_failure(ERROR_FILE_TOO_LARGE);
        }
        HANDLE mapping = ::CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
        const auto mapping_err = static_cast<int>(::GetLastError());
        ::CloseHandle(file);  // the mapping keeps the file open
        if (mapping == nullptr) {
            return os_failure(mapping_err);
        }
        void* view = ::MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
        const auto view_err = static_cast<int>(::GetLastError());
        ::CloseHandle(mapping);  // the view keeps the mapping alive
        if (view == nullptr) {
            return os_failure(view_err);
        }
        return mapped_file{static_cast<const std::byte*>(view), static_cast<std::size_t>(file_size.QuadPart)};
#else
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            return os_failure(errno);
        }
        struct stat st = {};
        if (::fstat(fd, &st) != 0) {
            const int err = errno;
            ::close(fd);
            return os_failure(err);
        }
        if (st.st_size <= 0) {
            ::close(fd);
            return error_info{errc::truncated, 0, 0};
        }
        if (static_cast<std::uintmax_t>(st.st_size) > std::numeric_limits<std::size_t>::max()) {
            ::close(fd);
            return os_failure(EFBIG);
        }
        const auto size = static_cast<std::size_t>(st.st_size);
        void* view = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        const int err = errno;
        ::close(fd);  // the mapping keeps the file open
        if (view == MAP_FAILED) {
            return os_failure(err);
        }
        return mapped_file{static_cast<const std::byte*>(view), size};
#endif
    }

    /// An empty object that owns no mapping.
    mapped_file() noexcept = default;

    mapped_file(const mapped_file&) = delete;
    mapped_file& operator=(const mapped_file&) = delete;

    mapped_file(mapped_file&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}

    mapped_file& operator=(mapped_file&& other) noexcept {
        if (this != &other) {
            release();
            data_ = std::exchange(other.data_, nullptr);
            size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }

    ~mapped_file() {
        release();
    }

    bool is_open() const noexcept {
        return data_ != nullptr;
    }
    std::size_t size() const noexcept {
        return size_;
    }
    std::span<const std::byte> bytes() const noexcept {
        return {data_, size_};
    }

private:
    mapped_file(const std::byte* data, std::size_t size) noexcept : data_(data), size_(size) {
        assert(data_ != nullptr && size_ > 0);  // I1
    }

    static error_info os_failure(int os_code) noexcept {
        return error_info{errc::io_error, 0, os_code};
    }

    void release() noexcept {
        if (data_ != nullptr) {
#if defined(_WIN32)
            ::UnmapViewOfFile(data_);
#else
            ::munmap(const_cast<std::byte*>(data_), size_);
#endif
        }
        data_ = nullptr;  // I1
        size_ = 0;
    }

    const std::byte* data_ = nullptr;
    std::size_t size_ = 0;
};

}  // namespace branes::synapse::gguf
