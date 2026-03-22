/**
 * fast_loadtxt.cpp — pybind11 module
 *
 * Exposes a single function: fast_loadtxt(filepath, ...)
 * that returns a 2D numpy ndarray of float64.
 *
 * Pipeline:
 *   1. mmap the file         (zero-copy, OS manages pages)
 *   2. scan for line offsets  (sequential byte scan, one heap alloc for offsets)
 *   3. allocate numpy array   (the only large allocation)
 *   4. parse in parallel      (OpenMP + fast_float, writes into numpy buffer)
 *   5. munmap                 (OS reclaims pages)
 *
 * Peak memory overhead = numpy array + line offset table.
 * The offset table is ~8 bytes per row (a size_t per line).
 * For 1M rows that's ~8 MB — negligible compared to the output array.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <string>

#include "file_mapper.h"
#include "line_scanner.h"
#include "chunk_parser.h"

namespace py = pybind11;


py::array_t<double> fast_loadtxt(const std::string& filepath,
                                  char comment = '#',
                                  int skip_rows = 0,
                                  int chunk_size = 500,
                                  int num_threads = 0) {
    // ── Step 1: Memory-map the file ──────────────────────────────────────
    //   No data is copied into our address space yet.
    //   The kernel pages in data on demand as we touch it.
    FileMapper file(filepath);

    // ── Step 2: Sequential scan ──────────────────────────────────────────
    //   Counts rows, detects columns, builds the line offset table.
    //   This is the only pass that touches every byte sequentially.
    //   The kernel readahead (MADV_SEQUENTIAL) makes this very fast.
    FileScanResult scan = scan_file(
        file.data(), file.size(), comment, skip_rows);

    // ── Step 3: Allocate the output numpy array ──────────────────────────
    //   This is the only large heap allocation in the entire pipeline.
    //   We get a pointer to the raw buffer so C++ can write into it directly.
    auto result = py::array_t<double>({scan.num_rows, scan.num_cols});
    double* output_ptr = result.mutable_data();

    // ── Step 4: Parallel parse ───────────────────────────────────────────
    //   Switch madvise hint: threads will access random-ish pages.
    file.hint_random_access();

    //   OpenMP threads parse their chunks directly into the numpy buffer.
    //   Each thread: read mmap'd bytes → fast_float → write to output_ptr.
    //   Zero intermediate allocations.
    parse_rows_parallel(
        file.data(),
        file.size(),
        scan.line_offsets,
        scan.num_rows,
        scan.num_cols,
        output_ptr,
        chunk_size,
        num_threads);

    // ── Step 5: Cleanup ──────────────────────────────────────────────────
    //   FileMapper destructor calls munmap + close.
    //   scan.line_offsets is freed (vector destructor).
    //   Only the numpy array survives — exactly what Python needs.

    return result;
}


// ── Module definition ────────────────────────────────────────────────────

PYBIND11_MODULE(_fast_loadtxt, m) {
    m.doc() = R"pbdoc(
        fast_loadtxt — High-performance text file loader
        
        Uses mmap + OpenMP + fast_float to load numeric text files
        into NumPy arrays with minimal memory overhead.
    )pbdoc";

    m.def("fast_loadtxt", &fast_loadtxt,
          py::arg("filepath"),
          py::arg("comment") = '#',
          py::arg("skip_rows") = 0,
          py::arg("chunk_size") = 500,
          py::arg("num_threads") = 0,
          R"pbdoc(
Load a numeric text file into a 2D NumPy array.

This is a drop-in replacement for numpy.loadtxt, optimized for speed
and memory efficiency on large files.

Parameters
----------
filepath : str
    Path to the data file.
comment : str, optional
    Comment character. Lines starting with this are skipped. Default: '#'.
skip_rows : int, optional
    Number of data rows to skip after comments. Default: 0.
chunk_size : int, optional
    Number of rows per OpenMP scheduling chunk. Tune this for your
    hardware — larger chunks reduce scheduling overhead, smaller chunks
    improve load balancing. Default: 500.
num_threads : int, optional
    Number of OpenMP threads. 0 = use OMP_NUM_THREADS or system default.

Returns
-------
numpy.ndarray
    2D array of float64, shape (num_rows, num_cols).

Examples
--------
>>> import fast_loadtxt as fl
>>> data = fl.loadtxt("simulation_output.dat")
>>> data.shape
(100000, 5)

>>> # Use 4 threads, 1000 rows per chunk
>>> data = fl.loadtxt("big_file.csv", chunk_size=1000, num_threads=4)
)pbdoc");
}
