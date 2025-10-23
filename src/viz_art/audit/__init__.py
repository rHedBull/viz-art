"""
Audit logging public API.

T089-T092: Factory functions and exports
"""

from pathlib import Path
from .logger import AuditLogger
from .run_tracker import RunTracker
from .query import AuditQuery
from ..types import LogLevel


def create_logger(
    run_id: str,
    output_dir: Path,
    rotation: str = "100 MB",
    retention: str = "30 days",
):
    """
    Create an audit logger instance.

    T089: Logger factory function

    Args:
        run_id: Run identifier to bind to all logs
        output_dir: Directory for log files (e.g., output/logs/)
        rotation: When to rotate logs (size or time, e.g., "100 MB", "1 day")
        retention: How long to keep logs (e.g., "30 days", "1 week")

    Returns:
        AuditLogger instance with run_id bound

    Example:
        logger = create_logger("abc-123", Path("output/logs"))
        logger.info("Pipeline started", input_count=10)
    """
    return AuditLogger(
        run_id=run_id,
        output_dir=output_dir,
        rotation=rotation,
        retention=retention,
    )


def create_run_tracker(output_dir: Path):
    """
    Create a run tracker instance.

    T090: RunTracker factory function

    Args:
        output_dir: Directory for run metadata (e.g., output/runs/)

    Returns:
        RunTracker instance

    Example:
        tracker = create_run_tracker(Path("output/runs"))
        with tracker.track(config, inputs, output_dir) as run_id:
            process_pipeline(run_id)
    """
    return RunTracker(output_dir=output_dir)


def create_query(log_dir: Path = Path("output/logs")):
    """
    Create an audit query builder.

    T091: AuditQuery factory function

    Args:
        log_dir: Directory containing log files

    Returns:
        AuditQuery instance

    Example:
        query = create_query()
        logs = query.after(datetime(2025, 10, 20)).stage("detection").fetch()
    """
    return AuditQuery(log_dir=log_dir)


# T092: Public exports
__all__ = [
    "AuditLogger",
    "RunTracker",
    "AuditQuery",
    "LogLevel",
    "create_logger",
    "create_run_tracker",
    "create_query",
]
