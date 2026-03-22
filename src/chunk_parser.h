/**
 * chunk_parser.h — Pass 2: parallel parse using OpenMP + fast_float.
 *
 * Each thread gets a range of rows (from the line offset table)
 * and parses them directly into the pre-allocated numpy buffer.
 *
 * Memory overhead per thread: zero heap allocations.
 * Each thread just reads from mmap'd bytes and writes into the output array.
 *
 * The chunk_size parameter controls OpenMP scheduling granularity.
 * Default 500 rows per chunk — tune based on row width and core count.
 */

#pragma once

#include <cstddef>
#include <vector>

#include <omp.h>

#include "fast_float/fast_float.h"

/**
 * Parse all data rows in parallel, writing into a pre-allocated buffer.
 *
 * @param file_data     Pointer to the mmap'd file bytes
 * @param file_size     Total file size (used as end boundary for last line)
 * @param line_offsets  Byte offset of each data line (from pass 1)
 * @param num_rows      Number of data rows
 * @param num_cols      Number of columns per row
 * @param output        Pre-allocated output buffer (row-major, num_rows × num_cols)
 * @param chunk_size    Rows per OpenMP scheduling chunk (default: 500)
 * @param num_threads   OpenMP thread count (0 = let OpenMP decide)
 */
inline void parse_rows_parallel(const char* file_data,
                                size_t file_size,
                                const std::vector<size_t>& line_offsets,
                                size_t num_rows,
                                size_t num_cols,
                                double* output,
                                int chunk_size = 500,
                                int num_threads = 0) {

    // Set thread count if requested
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }

    // Each iteration is independent: thread i writes to output[i * num_cols].
    // No locks, no shared mutable state, no false sharing (rows are large).
    #pragma omp parallel for schedule(dynamic, chunk_size)
    for (size_t row = 0; row < num_rows; row++) {

        // Where does this line start and end in the file?
        const char* line_begin = file_data + line_offsets[row];
        const char* line_end;

        if (row + 1 < num_rows) {
            line_end = file_data + line_offsets[row + 1];
        } else {
            line_end = file_data + file_size;
        }

        // Where does this row live in the output buffer?
        double* row_output = output + (row * num_cols);

        // Parse each column from the line
        const char* p = line_begin;
        size_t col = 0;

        while (p < line_end && col < num_cols) {
            // Skip whitespace and delimiters
            while (p < line_end && (*p == ' ' || *p == '\t' ||
                                    *p == ',' || *p == '\r')) {
                p++;
            }
            if (p >= line_end) break;

            // Parse one number with fast_float
            double val = 0.0;
            auto [ptr, ec] = fast_float::from_chars(p, line_end, val);

            if (ec == std::errc()) {
                row_output[col] = val;
                col++;
                p = ptr;
            } else {
                // Skip non-numeric token (shouldn't happen in clean data)
                while (p < line_end && *p != ' ' && *p != '\t' &&
                       *p != ',' && *p != '\n') {
                    p++;
                }
            }
        }

        // Fill remaining columns with 0.0 (ragged row protection)
        while (col < num_cols) {
            row_output[col] = 0.0;
            col++;
        }
    }
}
