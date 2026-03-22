"""
fast_loadtxt — Drop-in replacement for numpy.loadtxt.

Uses mmap + OpenMP + fast_float under the hood for:
  - Near-zero memory overhead (only the output array is heap-allocated)
  - Parallel parsing across all CPU cores
  - 3–5x faster float parsing via fast_float

Basic usage:
    import fast_loadtxt as fl
    data = fl.loadtxt("simulation_output.dat")

With options:
    data = fl.loadtxt("big.csv", comment='%', skip_rows=1,
                       chunk_size=1000, num_threads=4)
"""

from ._fast_loadtxt import fast_loadtxt as _fast_loadtxt

__version__ = "0.1.0"
__all__ = ["loadtxt"]


def loadtxt(filepath: str,
            comment: str = "#",
            skip_rows: int = 0,
            chunk_size: int = 500,
            num_threads: int = 0):
    """
    Load a numeric text file into a 2D NumPy array (float64).

    Parameters
    ----------
    filepath : str
        Path to the data file. Supports space, tab, and comma delimiters.
    comment : str, optional
        Lines starting with this character are skipped. Default: '#'.
    skip_rows : int, optional
        Number of non-comment rows to skip at the top. Default: 0.
    chunk_size : int, optional
        Rows per OpenMP scheduling chunk. Default: 500.
        - Larger values → less scheduling overhead
        - Smaller values → better load balancing for uneven rows
    num_threads : int, optional
        Number of threads for parallel parsing. 0 = auto. Default: 0.

    Returns
    -------
    numpy.ndarray
        2D array of shape (nrows, ncols), dtype float64.

    Notes
    -----
    Memory usage is approximately:
        output array  +  8 bytes per row (line offset table, freed after parse)
    The file itself is memory-mapped, not read into heap memory.

    Examples
    --------
    >>> import fast_loadtxt as fl

    >>> # Simple usage
    >>> data = fl.loadtxt("output.dat")

    >>> # CSV with a header row
    >>> data = fl.loadtxt("results.csv", skip_rows=1)

    >>> # Tune for a large file on an 8-core machine
    >>> data = fl.loadtxt("huge.dat", chunk_size=2000, num_threads=8)
    """
    # Validate inputs before passing to C++
    if not isinstance(filepath, str):
        raise TypeError(f"filepath must be a string, got {type(filepath).__name__}")
    if len(comment) != 1:
        raise ValueError("comment must be a single character")
    if skip_rows < 0:
        raise ValueError("skip_rows must be non-negative")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if num_threads < 0:
        raise ValueError("num_threads must be non-negative (0 = auto)")

    return _fast_loadtxt(filepath,
                         comment=comment,
                         skip_rows=skip_rows,
                         chunk_size=chunk_size,
                         num_threads=num_threads)
