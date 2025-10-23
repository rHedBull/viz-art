"""
Performance profiler for pipeline stages.

T031-T035: Profiler implementation with decorator and context manager
"""

from typing import Callable, Optional, Any
from functools import wraps
from contextlib import contextmanager
from datetime import datetime

from ..utils.timing import Timer
from ..utils.memory import get_cpu_memory_mb, get_gpu_memory_mb
from ..utils.identifiers import generate_run_id


class Profiler:
    """
    Performance profiler for pipeline stages.

    Automatically tracks execution time and memory usage when used as decorator
    or context manager.

    T031-T035: Complete profiler implementation
    """

    def __init__(self, storage: Optional[Any] = None, enabled: bool = True):
        """
        Initialize profiler.

        T031: Profiler constructor

        Args:
            storage: MetricsStorage instance for persisting metrics
            enabled: Whether profiling is active (False = no-op)
        """
        self.storage = storage
        self.enabled = enabled
        self._current_run_id: Optional[str] = None

    def __call__(self, func: Callable) -> Callable:
        """
        Decorator usage: @profiler

        T032: Decorator method to wrap stage execute() functions

        Args:
            func: Function to profile (typically Stage.execute)

        Returns:
            Wrapped function that records metrics
        """
        if not self.enabled:
            return func

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract stage name from function or class
            stage_name = getattr(func, '__name__', 'unknown')
            if args and hasattr(args[0], '__class__'):
                stage_name = args[0].__class__.__name__

            with self.measure(stage_name):
                result = func(*args, **kwargs)
            return result

        return wrapper

    @contextmanager
    def measure(self, stage_name: str):
        """
        Context manager usage for explicit profiling.

        T033: Context manager for profiling blocks

        Args:
            stage_name: Human-readable stage identifier

        Yields:
            None

        Example:
            with profiler.measure("preprocessing"):
                result = expensive_operation()
        """
        if not self.enabled:
            yield
            return

        # Capture initial memory
        start_cpu_mb = get_cpu_memory_mb()
        start_gpu_mb = get_gpu_memory_mb()

        # Time the execution
        with Timer() as timer:
            yield

        # Capture final memory (peak during execution)
        end_cpu_mb = get_cpu_memory_mb()
        end_gpu_mb = get_gpu_memory_mb()

        # Use max memory as peak (conservative estimate)
        cpu_memory_mb = max(start_cpu_mb, end_cpu_mb)
        gpu_memory_mb = end_gpu_mb if end_gpu_mb is not None else None

        # Record metrics
        self._record_metrics(
            stage_name=stage_name,
            execution_time_ms=timer.elapsed_ms,
            cpu_memory_mb=cpu_memory_mb,
            gpu_memory_mb=gpu_memory_mb,
        )

    def _record_metrics(
        self,
        stage_name: str,
        execution_time_ms: float,
        cpu_memory_mb: float,
        gpu_memory_mb: Optional[float] = None,
    ) -> None:
        """
        Record performance metrics to storage.

        T034-T035: Private method to capture and record metrics

        Args:
            stage_name: Stage identifier
            execution_time_ms: Stage duration in milliseconds
            cpu_memory_mb: Peak CPU memory in MB
            gpu_memory_mb: Peak GPU memory in MB (None if unavailable)
        """
        if not self.storage:
            return

        # Use existing run ID or generate new one
        run_id = self._current_run_id or generate_run_id()

        # T035: Integrate Timer and memory tracking utilities
        self.storage.write_metrics(
            run_id=run_id,
            stage_name=stage_name,
            execution_time_ms=execution_time_ms,
            cpu_memory_mb=cpu_memory_mb,
            gpu_memory_mb=gpu_memory_mb,
            timestamp=datetime.utcnow(),
        )

    def set_run_id(self, run_id: str) -> None:
        """
        Set the run ID for subsequent profiling calls.

        Args:
            run_id: Run identifier to use
        """
        self._current_run_id = run_id
