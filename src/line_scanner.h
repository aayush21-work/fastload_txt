/**
 * line_scanner.h — Pass 1: scan file to count rows and detect column count.
 *
 * This runs over the mmap'd bytes looking for newlines.
 * No heap allocations. No string copies.
 * Just pointer arithmetic and a counter.
 *
 * Also parses the first data line to figure out how many columns there are
 * (needed to pre-allocate the numpy array before pass 2).
 */

#pragma once

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "fast_float/fast_float.h"

struct FileScanResult {
    size_t num_rows;      // total data rows (excluding comments/blanks)
    size_t num_cols;      // columns detected from first data line
    size_t skip_bytes;    // bytes to skip at start (header rows)

    // Byte offset of the start of each data line.
    // This is the key output: pass 2 uses these offsets so threads
    // know exactly where their chunk starts — no scanning needed.
    std::vector<size_t> line_offsets;
};

/**
 * Scan the file and build a line offset table.
 *
 * @param data        Pointer to mmap'd file bytes
 * @param file_size   Total file size in bytes
 * @param comment     Comment character (lines starting with this are skipped)
 * @param skip_rows   Number of non-comment data rows to skip (headers)
 * @return FileScanResult with row count, column count, and line offsets
 */
inline FileScanResult scan_file(const char* data,
                                size_t file_size,
                                char comment = '#',
                                int skip_rows = 0) {
    FileScanResult result{};
    result.line_offsets.reserve(file_size / 40);  // rough guess: 40 bytes/row

    size_t pos = 0;
    int skipped = 0;
    bool first_data_line_found = false;

    while (pos < file_size) {
        // Find end of this line
        size_t line_start = pos;
        while (pos < file_size && data[pos] != '\n') {
            pos++;
        }
        size_t line_end = pos;
        if (pos < file_size) pos++;  // skip the '\n'

        // Skip blank lines
        size_t first_char = line_start;
        while (first_char < line_end &&
               (data[first_char] == ' ' || data[first_char] == '\t')) {
            first_char++;
        }
        if (first_char == line_end) continue;  // blank line

        // Skip comment lines
        if (data[first_char] == comment) continue;

        // Skip header rows
        if (skipped < skip_rows) {
            skipped++;
            continue;
        }

        // Detect column count from first data line
        if (!first_data_line_found) {
            first_data_line_found = true;
            result.skip_bytes = line_start;

            // Count columns by parsing the first line
            const char* p = data + line_start;
            const char* end = data + line_end;
            size_t cols = 0;

            while (p < end) {
                // Skip whitespace and delimiters
                while (p < end && (*p == ' ' || *p == '\t' ||
                                   *p == ',' || *p == '\r')) {
                    p++;
                }
                if (p >= end) break;

                // Try to parse a number
                double val;
                auto [ptr, ec] = fast_float::from_chars(p, end, val);
                if (ec == std::errc()) {
                    cols++;
                    p = ptr;
                } else {
                    // Not a number — skip this token
                    while (p < end && *p != ' ' && *p != '\t' &&
                           *p != ',' && *p != '\n') {
                        p++;
                    }
                }
            }
            result.num_cols = cols;
        }

        // Record this line's byte offset
        result.line_offsets.push_back(line_start);
        result.num_rows++;
    }

    if (result.num_rows == 0) {
        throw std::runtime_error("No data rows found in file");
    }
    if (result.num_cols == 0) {
        throw std::runtime_error("Could not detect any numeric columns");
    }

    return result;
}
