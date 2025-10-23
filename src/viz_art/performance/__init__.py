"""
Performance monitoring public API.

T051-T054: Factory functions and exports
"""

from pathlib import Path
from .profiler import Profiler
from .metrics_storage import MetricsStorage
from .dashboard import PerformanceDashboard


def create_profiler(storage, enabled: bool = True):
    """
    Create a performance profiler instance.

    T051: Profiler factory function

    Args:
        storage: Where to write metrics
        enabled: Whether profiling is active (False = no-op)

    Returns:
        Profiler instance

    Example:
        storage = create_metrics_storage(Path("output/metrics"))
        profiler = create_profiler(storage)

        @profiler
        def my_stage(inputs):
            return process(inputs)
    """
    return Profiler(storage=storage, enabled=enabled)


def create_metrics_storage(output_dir: Path, retention_days: int = 365):
    """
    Create a metrics storage instance.

    T052: MetricsStorage factory function

    Args:
        output_dir: Directory for Parquet files (e.g., output/metrics/)
        retention_days: Auto-delete metrics older than this (0 = never delete)

    Returns:
        MetricsStorage instance

    Raises:
        ValueError: If output_dir doesn't exist or isn't writable

    Example:
        storage = create_metrics_storage(Path("output/metrics"), retention_days=90)
    """
    return MetricsStorage(output_dir=output_dir, retention_days=retention_days)


def create_dashboard(storage):
    """
    Create a performance dashboard instance.

    T053: PerformanceDashboard factory function

    Args:
        storage: Metrics source

    Returns:
        PerformanceDashboard instance

    Example:
        storage = create_metrics_storage(Path("output/metrics"))
        dashboard = create_dashboard(storage)
        html = dashboard.render_timing_chart("abc-123")
    """
    return PerformanceDashboard(storage=storage)


# T054: Public exports
__all__ = [
    "Profiler",
    "MetricsStorage",
    "PerformanceDashboard",
    "create_profiler",
    "create_metrics_storage",
    "create_dashboard",
]
