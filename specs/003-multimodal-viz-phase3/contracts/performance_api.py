"""
Performance Profiling API Contract

This file defines the public API for performance profiling functionality.
All functions and classes marked as public (no leading underscore) are part
of the stable API contract and must maintain backward compatibility.
"""

from typing import Dict, List, Optional, Protocol
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager


class Profiler(Protocol):
    """
    Performance profiler for pipeline stages.

    Automatically tracks execution time and memory usage when used as decorator
    or context manager.
    """

    def __call__(self, func):
        """
        Decorator usage: @profile

        Args:
            func: Function to profile (typically Stage.execute)

        Returns:
            Wrapped function that records metrics

        Example:
            @profile
            def execute(self, inputs):
                return process(inputs)
        """
        ...

    @contextmanager
    def measure(self, stage_name: str):
        """
        Context manager usage for explicit profiling.

        Args:
            stage_name: Human-readable stage identifier

        Yields:
            None

        Example:
            with profiler.measure("preprocessing"):
                result = expensive_operation()
        """
        ...


class MetricsStorage(Protocol):
    """
    Storage interface for performance metrics.

    Handles writing and querying metrics in Parquet format.
    """

    def write_metrics(
        self,
        run_id: str,
        stage_name: str,
        execution_time_ms: float,
        cpu_memory_mb: float,
        gpu_memory_mb: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Write performance metrics for a stage execution.

        Args:
            run_id: Unique run identifier (UUID v4)
            stage_name: Stage identifier
            execution_time_ms: Stage duration in milliseconds
            cpu_memory_mb: Peak CPU memory in MB
            gpu_memory_mb: Peak GPU memory in MB (None if unavailable)
            timestamp: Metric timestamp (defaults to now if None)

        Raises:
            ValueError: If run_id is not valid UUID v4
            IOError: If unable to write to storage

        Example:
            storage.write_metrics(
                run_id="abc-123",
                stage_name="detection",
                execution_time_ms=150.5,
                cpu_memory_mb=512.0,
                gpu_memory_mb=2048.0,
            )
        """
        ...

    def query_metrics(
        self,
        stage_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        run_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Query performance metrics with filters.

        Args:
            stage_name: Filter by stage (None = all stages)
            start_date: Filter >= this date (None = no lower bound)
            end_date: Filter <= this date (None = no upper bound)
            run_ids: Filter by specific run IDs (None = all runs)

        Returns:
            List of metric dictionaries with keys:
                - run_id: str
                - timestamp: datetime
                - stage_name: str
                - execution_time_ms: float
                - cpu_memory_mb: float
                - gpu_memory_mb: Optional[float]

        Example:
            metrics = storage.query_metrics(
                stage_name="detection",
                start_date=datetime(2025, 10, 1),
                end_date=datetime(2025, 10, 23),
            )
        """
        ...

    def get_aggregate_stats(
        self,
        stage_name: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """
        Calculate aggregate statistics for a stage.

        Args:
            stage_name: Stage to analyze
            start_date: Filter >= this date (None = all time)
            end_date: Filter <= this date (None = all time)

        Returns:
            Dictionary with keys:
                - median: float
                - mean: float
                - std: float
                - min: float
                - max: float
                - p50: float
                - p95: float
                - p99: float

        Example:
            stats = storage.get_aggregate_stats("detection")
            print(f"Median: {stats['median']}ms")
        """
        ...


class PerformanceDashboard(Protocol):
    """
    Visualization interface for performance data.

    Generates interactive Plotly charts for performance analysis.
    """

    def render_timing_chart(
        self,
        run_id: str,
        output_path: Optional[Path] = None,
    ) -> str:
        """
        Generate per-stage timing chart for a run.

        Args:
            run_id: Run to visualize
            output_path: Save HTML to this path (None = return HTML string)

        Returns:
            HTML string with embedded Plotly chart

        Raises:
            ValueError: If run_id not found

        Example:
            html = dashboard.render_timing_chart("abc-123")
            # Or save to file:
            dashboard.render_timing_chart("abc-123", Path("report.html"))
        """
        ...

    def render_memory_chart(
        self,
        run_id: str,
        output_path: Optional[Path] = None,
    ) -> str:
        """
        Generate memory usage chart (CPU and GPU) for a run.

        Args:
            run_id: Run to visualize
            output_path: Save HTML to this path (None = return HTML string)

        Returns:
            HTML string with embedded Plotly chart
        """
        ...

    def render_trend_chart(
        self,
        stage_name: str,
        start_date: datetime,
        end_date: datetime,
        output_path: Optional[Path] = None,
    ) -> str:
        """
        Generate historical trend chart for a stage.

        Args:
            stage_name: Stage to analyze
            start_date: Start of date range
            end_date: End of date range
            output_path: Save HTML to this path (None = return HTML string)

        Returns:
            HTML string with line chart showing execution time over time

        Example:
            dashboard.render_trend_chart(
                "detection",
                datetime(2025, 10, 1),
                datetime(2025, 10, 23),
                Path("trends.html"),
            )
        """
        ...


# Public factory functions

def create_profiler(
    storage: MetricsStorage,
    enabled: bool = True,
) -> Profiler:
    """
    Create a performance profiler instance.

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
    ...


def create_metrics_storage(
    output_dir: Path,
    retention_days: int = 365,
) -> MetricsStorage:
    """
    Create a metrics storage instance.

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
    ...


def create_dashboard(
    storage: MetricsStorage,
) -> PerformanceDashboard:
    """
    Create a performance dashboard instance.

    Args:
        storage: Metrics source

    Returns:
        PerformanceDashboard instance

    Example:
        storage = create_metrics_storage(Path("output/metrics"))
        dashboard = create_dashboard(storage)
        html = dashboard.render_timing_chart("abc-123")
    """
    ...
