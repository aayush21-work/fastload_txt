# fast_loadtxt

[![PyPI version](https://img.shields.io/pypi/v/fast_loadtxt.svg)](https://pypi.org/project/fast_loadtxt/)
[![Build](https://github.com/aayush21-work/fastload_txt/actions/workflows/build_wheels.yml/badge.svg)](https://github.com/aayush21-work/fastload_txt/actions)

A high-performance drop-in replacement for `numpy.loadtxt`.

Built with **memory-mapped I/O**, **OpenMP** parallel parsing, and **fast_float** — designed for scientific computing workflows where loading large text data files is a bottleneck.

## Installation

```bash
pip install fast_loadtxt
```

Pre-built wheels are available for Linux (x86_64) and macOS (Apple Silicon). On other platforms, a C++17 compiler with OpenMP support is required for building from source.

## Quick Start

```python
import fast_loadtxt as fl

# Drop-in replacement for numpy.loadtxt
data = fl.loadtxt("simulation_output.dat")

# CSV with a header row
data = fl.loadtxt("results.csv", skip_rows=1)

# Tune parallelism for your hardware
data = fl.loadtxt("huge_file.dat", chunk_size=2000, num_threads=8)
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `filepath` | — | Path to the data file (space, tab, or comma delimited) |
| `comment` | `'#'` | Lines starting with this character are skipped |
| `skip_rows` | `0` | Number of non-comment rows to skip at the top |
| `chunk_size` | `500` | Rows per OpenMP scheduling chunk |
| `num_threads` | `0` | Thread count for parsing (`0` = auto-detect) |

**Returns:** 2D `numpy.ndarray` of `float64`, shape `(nrows, ncols)`.

## Benchmarks

Tested on an 8-thread laptop (AMD/Intel, Linux):

| File size | numpy.loadtxt | fast_loadtxt | Speedup |
|---|---|---|---|
| 1K rows × 5 cols | 2.5 ms | 0.3 ms | **8.9×** |
| 10K rows × 5 cols | 23.7 ms | 4.0 ms | **5.9×** |
| 100K rows × 5 cols | 215 ms | 8.5 ms | **25.2×** |
| 100K rows × 20 cols | 405 ms | 21.4 ms | **18.9×** |
| 1M rows × 5 cols | 1,033 ms | 74.3 ms | **13.9×** |

Run the benchmarks on your own hardware:

```bash
git clone https://github.com/aayush21-work/fastload_txt.git
cd fastload_txt
pip install .
python benchmarks/bench.py
```

## Why is it faster?

`numpy.loadtxt` reads the entire file into a heap buffer, parses every number single-threaded with `strtod`, and creates intermediate Python objects along the way. Peak memory usage is roughly 3× the file size.

`fast_loadtxt` takes a fundamentally different approach:

| | numpy.loadtxt | fast_loadtxt |
|---|---|---|
| File access | `read()` into heap buffer | `mmap` — zero-copy, OS-managed pages |
| Float parsing | `strtod`, single-threaded | `fast_float`, 3–5× faster per number |
| Parallelism | None | OpenMP across all cores |
| Intermediate allocations | Lists, Python floats, copies | **One** — the output array |
| Memory overhead | ~3× file size | ~8 bytes per row (line offset table) |

Under memory pressure, the OS can evict the mmap'd file pages for free — they're file-backed, so no swap is needed. The only heap allocation that survives is the numpy array itself.

## Architecture

```
                    ┌──────────────────────────────────┐
                    │          fast_loadtxt()           │
                    └──────────────┬───────────────────┘
                                   │
                    ┌──────────────▼───────────────────┐
  Pass 1 (serial)  │         FileMapper (mmap)         │
                    │  Map file into address space      │
                    │  Zero heap allocation              │
                    └──────────────┬───────────────────┘
                                   │
                    ┌──────────────▼───────────────────┐
  Pass 1 (serial)  │    LineScanner (byte scan)        │
                    │  Count rows, detect columns       │
                    │  Build line offset table           │
                    └──────────────┬───────────────────┘
                                   │
                    ┌──────────────▼───────────────────┐
                    │   Allocate numpy array (N × M)    │
                    │   The only large heap allocation   │
                    └──────────────┬───────────────────┘
                                   │
                    ┌──────────────▼───────────────────┐
  Pass 2 (parallel)│   ChunkParser (OpenMP threads)    │
                    │  Each thread:                     │
                    │    read mmap'd bytes              │
                    │    → fast_float parse             │
                    │    → write into numpy buffer      │
                    │  Zero allocations in hot loop     │
                    └──────────────┬───────────────────┘
                                   │
                    ┌──────────────▼───────────────────┐
  Cleanup          │  munmap (OS reclaims pages)       │
                    │  Free offset table                │
                    │  Return numpy array to Python     │
                    └──────────────────────────────────┘
```

For small files (<5000 rows), OpenMP thread spawning is automatically skipped to avoid overhead — the fast_float single-threaded path alone is still 5–9× faster than numpy.

## Source Structure

The C++ core is four small, focused files — each under 110 lines:

```
src/
├── file_mapper.h       # POSIX mmap wrapper, RAII cleanup, madvise hints
├── line_scanner.h      # Sequential byte scan → row count, column detection, offset table
├── chunk_parser.h      # OpenMP parallel parse loop, fast_float, direct-to-buffer writes
├── fast_loadtxt.cpp    # pybind11 module — ties the pipeline together
└── fast_loadtxt/
    └── __init__.py     # Python API wrapper with input validation
```

## Memory Model

For a 500K × 5 column file (~60 MB of text):

```
numpy.loadtxt:
  File read buffer          ~60 MB  (read() into heap)
  Intermediate Python lists ~40 MB  (list of lists of floats)
  Final numpy array         ~20 MB
  Peak total:              ~120 MB

fast_loadtxt:
  File data (mmap)           0 MB   (kernel page cache, not heap)
  Line offset table          ~4 MB  (8 bytes × 500K rows, freed after parse)
  Final numpy array         ~20 MB
  Peak total:               ~24 MB  (5× less)
```

## Use Cases

`fast_loadtxt` is particularly useful for:

- **Simulation output** — loading large `.dat` files from numerical solvers
- **MCMC chains** — reading parameter chains from samplers like Cobaya, emcee, or CosmoMC
- **Observational data** — catalogs, light curves, spectra in text format
- **Any workflow** where `numpy.loadtxt` is a bottleneck

## Supported Formats

- Space-delimited (`.dat`, `.txt`)
- Tab-delimited (`.tsv`)
- Comma-delimited (`.csv`)
- Comment lines (`#` by default, configurable)
- Mixed whitespace
- Ragged rows (padded with `0.0`)

## Limitations

- **POSIX only** — uses `mmap` and `unistd.h`. Linux and macOS are supported. Windows support is planned.
- **float64 only** — all values are parsed as double-precision floats.
- **Numeric data only** — string columns are not supported. Use `pandas.read_csv` for mixed-type data.
- **No dtype selection** — unlike `numpy.loadtxt`, you cannot specify output dtype. Output is always `float64`.

## Contributing

Contributions are welcome. The codebase is intentionally small and readable — the entire C++ core is ~400 lines across four files.

```bash
git clone https://github.com/aayush21-work/fastload_txt.git
cd fastload_txt
pip install -e .
python benchmarks/bench.py
```

Areas where contributions would be valuable:

- **Windows support** — adding `CreateFileMapping`/`MapViewOfFile` fallback in `file_mapper.h`
- **Integer dtype support** — `int32`/`int64` output arrays
- **Column selection** — `usecols` parameter like numpy
- **Streaming mode** — iterator-based loading for files larger than RAM

## License

GNU GPL v3

## Acknowledgments

- [fast_float](https://github.com/fastfloat/fast_float) — Daniel Lemire et al. Header-only, high-performance float parser.
- [pybind11](https://github.com/pybind/pybind11) — Seamless C++/Python interoperability.
