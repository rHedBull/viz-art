"""
Timing utilities for performance monitoring.

T029: Timer context manager for millisecond-precision timing
"""

import time
from typing import Optional
from contextlib import contextmanager


class Timer:
    """
    Context manager for millisecond-precision timing.

    T029: Timer with ms precision

    Example:
        with Timer() as timer:
            # code to time
            pass
        print(f"Execution time: {timer.elapsed_ms}ms")
    """

    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        if self.start_time is not None:
            self.elapsed_ms = (self.end_time - self.start_time) * 1000.0
        return False  # Don't suppress exceptions


@contextmanager
def measure_time():
    """
    Simple context manager that yields elapsed time in milliseconds.

    Yields:
        dict: Dictionary with 'elapsed_ms' key updated on exit

    Example:
        timing = {}
        with measure_time() as timing:
            # code to time
            pass
        print(f"Took {timing['elapsed_ms']}ms")
    """
    start = time.time()
    timing = {"elapsed_ms": 0.0}
    yield timing
    timing["elapsed_ms"] = (time.time() - start) * 1000.0
