#!/usr/bin/env python3
"""
Benchmark: fast_loadtxt vs numpy.loadtxt

Generates test files of various sizes and compares load times.
Run after installing:  pip install .
"""

import time
import tempfile
import os

import numpy as np


def generate_test_file(path: str, nrows: int, ncols: int):
    """Write a random numeric text file."""
    data = np.random.randn(nrows, ncols)
    header = "# " + "  ".join(f"col_{i}" for i in range(ncols))
    np.savetxt(path, data, header=header, fmt="%.15e")
    return data


def bench_numpy(path: str, repeats: int = 3):
    """Time numpy.loadtxt."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        data = np.loadtxt(path)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.median(times), data


def bench_fast(path: str, repeats: int = 3, **kwargs):
    """Time fast_loadtxt.loadtxt."""
    import fast_loadtxt as fl
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        data = fl.loadtxt(path, **kwargs)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.median(times), data


def run_benchmark():
    test_cases = [
        (1_000,    5,  "1K rows × 5 cols"),
        (10_000,   5,  "10K rows × 5 cols"),
        (100_000,  5,  "100K rows × 5 cols"),
        (100_000,  20, "100K rows × 20 cols"),
        (1_000_000, 5, "1M rows × 5 cols"),
    ]

    print("=" * 70)
    print(f"{'Test case':<25} {'numpy (s)':>10} {'fast (s)':>10} {'speedup':>10}")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        for nrows, ncols, label in test_cases:
            path = os.path.join(tmpdir, f"test_{nrows}_{ncols}.dat")

            # Generate test file
            ref_data = generate_test_file(path, nrows, ncols)
            file_mb = os.path.getsize(path) / (1024 * 1024)

            # Benchmark numpy
            t_np, data_np = bench_numpy(path)

            # Benchmark fast_loadtxt
            t_fl, data_fl = bench_fast(path)

            # Verify correctness
            if not np.allclose(data_np, data_fl, atol=1e-12):
                print(f"  WARNING: results differ for {label}!")

            speedup = t_np / t_fl
            print(f"{label:<25} {t_np:>10.4f} {t_fl:>10.4f} {speedup:>9.1f}×")

    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
