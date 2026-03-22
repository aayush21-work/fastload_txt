/**
 * file_mapper.h — Memory-mapped file wrapper (POSIX)
 *
 * Maps a file into memory using mmap. The OS kernel manages paging,
 * so we never allocate heap memory for the file contents. Under memory
 * pressure the kernel simply evicts pages (they're file-backed, no swap).
 *
 * Usage:
 *     FileMapper fm("data.dat");
 *     const char* begin = fm.data();
 *     const char* end   = fm.data() + fm.size();
 */

#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

#include <fcntl.h>     // open
#include <sys/mman.h>  // mmap, munmap, madvise
#include <sys/stat.h>  // fstat
#include <unistd.h>    // close

class FileMapper {
public:
    /**
     * Map the entire file into memory.
     * Throws std::runtime_error if anything goes wrong.
     */
    explicit FileMapper(const std::string& path) {
        // Open file (read-only)
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            throw std::runtime_error("Cannot open file: " + path);
        }

        // Get file size
        struct stat st{};
        if (::fstat(fd_, &st) != 0) {
            ::close(fd_);
            throw std::runtime_error("Cannot stat file: " + path);
        }
        size_ = static_cast<size_t>(st.st_size);

        if (size_ == 0) {
            ::close(fd_);
            throw std::runtime_error("File is empty: " + path);
        }

        // Map file into memory (read-only, private — no writes to disk)
        data_ = static_cast<const char*>(
            ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0));

        if (data_ == MAP_FAILED) {
            ::close(fd_);
            throw std::runtime_error("mmap failed for: " + path);
        }

        // Hint: we'll read sequentially (helps kernel readahead)
        ::madvise(const_cast<char*>(data_), size_, MADV_SEQUENTIAL);
    }

    // No copies — this owns the mapping
    FileMapper(const FileMapper&) = delete;
    FileMapper& operator=(const FileMapper&) = delete;

    // Move is fine
    FileMapper(FileMapper&& other) noexcept
        : data_(other.data_), size_(other.size_), fd_(other.fd_) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.fd_ = -1;
    }

    ~FileMapper() {
        if (data_ && data_ != MAP_FAILED) {
            ::munmap(const_cast<char*>(data_), size_);
        }
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }

    // Accessors
    const char* data() const { return data_; }
    size_t size() const { return size_; }

    /**
     * Switch madvise hint for the parallel parse pass.
     * MADV_WILLNEED tells the kernel to prefetch aggressively.
     */
    void hint_random_access() {
        ::madvise(const_cast<char*>(data_), size_, MADV_WILLNEED);
    }

private:
    const char* data_ = nullptr;
    size_t size_ = 0;
    int fd_ = -1;
};
