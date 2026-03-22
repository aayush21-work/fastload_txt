# fast_loadtxt

A high-performance drop-in replacement for `numpy.loadtxt`, built with **mmap + OpenMP + fast_float**.

Designed for scientific computing workflows where loading large text data files is a bottleneck — simulation outputs, observational datasets, parameter chains, etc.

## Why?

`numpy.loadtxt` reads the entire file into memory, parses it single-threaded, and creates intermediate Python objects. For large files this is slow and memory-hungry.

`fast_loadtxt` takes a different approach:

| | numpy.loadtxt | fast_loadtxt |
|---|---|---|
| File access | `read()` into heap buffer | `mmap` (zero-copy, OS-managed) |
| Parsing | Single-threaded, `strtod` | OpenMP parallel, `fast_float` |
| Intermediate allocs | Many (Python objects, lists) | One (the output array) |
| Memory overhead | ~3× file size | ~8 bytes per row (line offsets) |

## Architecture

```
Pass 1 (serial):   mmap file → scan bytes for '\n' → build line offset table
                    Detect column count from first data line
                    Allocate numpy array (nrows × ncols)

Pass 2 (parallel): OpenMP threads parse chunks of rows
                    Each thread reads mmap'd bytes → fast_float → writes into numpy buffer
                    Zero intermediate allocations

Cleanup:           munmap (OS reclaims pages), free offset table
                    Only the numpy array survives
```

## Installation

Requirements: C++17 compiler with OpenMP support, Python ≥ 3.8, NumPy.

```bash
pip install .
```

Or for development:
```bash
pip install -e .
```

## Usage

```python
import fast_loadtxt as fl

# Basic — just like numpy.loadtxt
data = fl.loadtxt("simulation_output.dat")

# CSV with header
data = fl.loadtxt("results.csv", skip_rows=1)

# Tune for your hardware
data = fl.loadtxt("huge_file.dat", chunk_size=2000, num_threads=8)
```

### Parameters

- `filepath` — Path to the data file (space/tab/comma delimited)
- `comment` — Comment character, default `'#'`
- `skip_rows` — Header rows to skip (after comments), default `0`
- `chunk_size` — Rows per OpenMP chunk, default `500`
- `num_threads` — Thread count, default `0` (auto)

## Benchmarks

Run the benchmark suite:
```bash
python benchmarks/bench.py
```

## How It Works (for contributors)

The code is split into four small, focused files:

1. **`file_mapper.h`** — POSIX mmap wrapper. Maps the file read-only, calls `madvise` for kernel readahead hints. RAII cleanup.

2. **`line_scanner.h`** — Sequential byte scan. Counts data rows, detects columns, builds a `vector<size_t>` of byte offsets for each data line. This is the only heap allocation beyond the output array.

3. **`chunk_parser.h`** — OpenMP parallel loop. Each thread gets a range of row indices, looks up their byte offsets, parses with `fast_float`, writes directly into the pre-allocated numpy buffer. Zero allocations in the hot loop.

4. **`fast_loadtxt.cpp`** — pybind11 glue. Calls the above three in sequence, returns the numpy array.

## License

MIT
