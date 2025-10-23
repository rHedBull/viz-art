"""
Audit Logging API Contract

This file defines the public API for audit logging and query functionality.
"""

from typing import Dict, List, Optional, Any, Protocol
from datetime import datetime
from pathlib import Path
from enum import Enum
from contextlib import contextmanager


class LogLevel(str, Enum):
    """Log severity levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AuditLogger(Protocol):
    """
    Structured logger for pipeline audit trails.

    Automatically includes run_id context in all log entries.
    """

    def debug(self, message: str, **metadata) -> None:
        """Log debug-level message with optional metadata"""
        ...

    def info(self, message: str, **metadata) -> None:
        """Log info-level message with optional metadata"""
        ...

    def warning(self, message: str, **metadata) -> None:
        """Log warning-level message with optional metadata"""
        ...

    def error(self, message: str, **metadata) -> None:
        """Log error-level message with optional metadata"""
        ...

    def critical(self, message: str, **metadata) -> None:
        """Log critical-level message with optional metadata"""
        ...

    def bind(self, **context) -> "AuditLogger":
        """
        Create logger with bound context.

        Args:
            **context: Key-value pairs to include in all logs

        Returns:
            New logger with context bound

        Example:
            stage_logger = logger.bind(stage_name="detection")
            stage_logger.info("Starting detection")
            # Logs: {"message": "Starting detection", "stage_name": "detection", ...}
        """
        ...


class RunTracker(Protocol):
    """
    Run ID generator and context manager.

    Manages run lifecycle: creates ID, tracks start/end, updates status.
    """

    @contextmanager
    def track(
        self,
        config_snapshot: Dict[str, Any],
        input_files: List[Path],
        output_dir: Path,
    ):
        """
        Context manager for tracking a pipeline run.

        Args:
            config_snapshot: Pipeline configuration at run time
            input_files: Input images/point clouds being processed
            output_dir: Where outputs will be saved

        Yields:
            run_id: str (UUID v4)

        Raises:
            Exception: Propagates any exception from pipeline, marks run as FAILED

        Example:
            tracker = create_run_tracker(Path("output/runs"))
            with tracker.track(config, inputs, output_dir) as run_id:
                logger = create_logger(run_id)
                logger.info("Starting pipeline")
                results = pipeline.run(inputs)
                logger.info("Pipeline completed", output_count=len(results))
        """
        ...

    def get_run_metadata(self, run_id: str) -> Dict[str, Any]:
        """
        Retrieve metadata for a completed run.

        Args:
            run_id: Run to query

        Returns:
            Dictionary with keys:
                - run_id: str
                - timestamp: datetime
                - status: str (RUNNING, COMPLETED, FAILED)
                - config_snapshot: Dict
                - input_files: List[str]
                - total_duration_ms: float
                - error: Optional[str]

        Raises:
            KeyError: If run_id not found
        """
        ...


class AuditQuery(Protocol):
    """
    Fluent interface for querying audit logs.

    Uses builder pattern for composable filters.
    """

    def run_id(self, run_id: str) -> "AuditQuery":
        """Filter by run ID"""
        ...

    def stage(self, stage_name: str) -> "AuditQuery":
        """Filter by stage name"""
        ...

    def level(self, level: LogLevel) -> "AuditQuery":
        """Filter by log level"""
        ...

    def after(self, timestamp: datetime) -> "AuditQuery":
        """Filter logs >= timestamp"""
        ...

    def before(self, timestamp: datetime) -> "AuditQuery":
        """Filter logs <= timestamp"""
        ...

    def failed(self) -> "AuditQuery":
        """Filter to ERROR and CRITICAL levels only"""
        ...

    def limit(self, count: int) -> "AuditQuery":
        """Limit results to first N entries"""
        ...

    def fetch(self) -> List[Dict[str, Any]]:
        """
        Execute query and return results.

        Returns:
            List of log entry dictionaries with keys:
                - run_id: str
                - timestamp: datetime
                - level: str
                - stage_name: Optional[str]
                - message: str
                - metadata: Dict

        Example:
            logs = (
                AuditQuery()
                .after(datetime(2025, 10, 20))
                .stage("detection")
                .failed()
                .limit(10)
                .fetch()
            )
        """
        ...

    def export_json(self, output_path: Path) -> int:
        """
        Export filtered logs to JSON file.

        Args:
            output_path: Where to write JSON

        Returns:
            Number of log entries exported

        Example:
            count = (
                AuditQuery()
                .after(datetime(2025, 10, 20))
                .export_json(Path("filtered_logs.json"))
            )
            print(f"Exported {count} log entries")
        """
        ...


# Public factory functions

def create_logger(
    run_id: str,
    output_dir: Path,
    rotation: str = "100 MB",
    retention: str = "30 days",
) -> AuditLogger:
    """
    Create an audit logger instance.

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
    ...


def create_run_tracker(
    output_dir: Path,
) -> RunTracker:
    """
    Create a run tracker instance.

    Args:
        output_dir: Directory for run metadata (e.g., output/runs/)

    Returns:
        RunTracker instance

    Example:
        tracker = create_run_tracker(Path("output/runs"))
        with tracker.track(config, inputs, output_dir) as run_id:
            process_pipeline(run_id)
    """
    ...


def create_query() -> AuditQuery:
    """
    Create an audit query builder.

    Returns:
        AuditQuery instance

    Example:
        query = create_query()
        logs = query.after(datetime(2025, 10, 20)).stage("detection").fetch()
    """
    ...


# CLI integration

def query_logs_cli(
    after: Optional[str] = None,
    before: Optional[str] = None,
    run_id: Optional[str] = None,
    stage: Optional[str] = None,
    level: Optional[str] = None,
    failed: bool = False,
    limit: Optional[int] = None,
    output: Optional[str] = None,
) -> int:
    """
    CLI interface for querying audit logs.

    Args:
        after: ISO 8601 date string (e.g., "2025-10-20")
        before: ISO 8601 date string
        run_id: Filter by run ID
        stage: Filter by stage name
        level: Filter by log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        failed: If True, show only ERROR and CRITICAL
        limit: Max number of results
        output: Path to export JSON (None = print to stdout)

    Returns:
        Number of log entries found

    Example (as CLI):
        viz-art logs --after 2025-10-20 --stage detection --failed --limit 10
    """
    ...
