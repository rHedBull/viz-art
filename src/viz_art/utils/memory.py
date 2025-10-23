"""
Memory tracking utilities for performance monitoring.

T027-T028: CPU and GPU memory tracking functions
"""

import psutil
from typing import Optional


def get_cpu_memory_mb() -> float:
    """
    Get current process CPU memory usage in megabytes (RSS).

    T027: CPU memory tracking using psutil

    Returns:
        float: Memory usage in MB
    """
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # Convert bytes to MB


def get_gpu_memory_mb(device_index: int = 0) -> Optional[float]:
    """
    Get GPU memory usage in megabytes.

    T028: GPU memory tracking with pynvml integration (graceful degradation)

    Args:
        device_index: GPU device index (default: 0)

    Returns:
        float: GPU memory usage in MB, or None if GPU unavailable
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return info.used / 1024 / 1024  # Convert bytes to MB
    except (ImportError, Exception):
        # Gracefully degrade if pynvml not available or GPU not present
        return None
