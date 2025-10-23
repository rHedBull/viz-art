"""
Audit logger with Loguru integration.

T058-T066: AuditLogger implementation with structured logging
"""

from pathlib import Path
from typing import Optional, Dict, Any

try:
    from loguru import logger as loguru_logger
    LOGURU_AVAILABLE = True
except ImportError:
    LOGURU_AVAILABLE = False

from ..types import LogLevel


class AuditLogger:
    """
    Structured logger for pipeline audit trails.

    Automatically includes run_id context in all log entries.
    T058-T066: Complete logger implementation
    """

    def __init__(
        self,
        run_id: str,
        output_dir: Path,
        rotation: str = "100 MB",
        retention: str = "30 days",
    ):
        """
        Initialize audit logger.

        T058-T062: Constructor with Loguru configuration

        Args:
            run_id: Run identifier to bind to all logs
            output_dir: Directory for log files (e.g., output/logs/)
            rotation: When to rotate logs (size or time)
            retention: How long to keep logs
        """
        if not LOGURU_AVAILABLE:
            raise ImportError(
                "Loguru is required for audit logging. "
                "Install with: pip install loguru>=0.7"
            )

        self.run_id = run_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # T059: Configure Loguru with JSON serialization
        # T060: Date-based file naming pattern
        from datetime import datetime
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        log_file = self.output_dir / f"{date_str}.jsonl"

        # Remove default handler
        loguru_logger.remove()

        # T061-T062: Add rotation and retention configuration
        # Add JSON handler with rotation
        loguru_logger.add(
            log_file,
            rotation=rotation,
            retention=retention,
            format="{message}",  # We'll format as JSON ourselves
            serialize=True,  # Enable JSON serialization
            enqueue=True,  # Thread-safe
        )

        # T064: Bind run_id to all logs
        self._logger = loguru_logger.bind(run_id=run_id)

    def debug(self, message: str, **metadata) -> None:
        """
        Log debug-level message with optional metadata.

        T063: Debug log method
        """
        self._log(LogLevel.DEBUG, message, metadata)

    def info(self, message: str, **metadata) -> None:
        """
        Log info-level message with optional metadata.

        T063: Info log method
        """
        self._log(LogLevel.INFO, message, metadata)

    def warning(self, message: str, **metadata) -> None:
        """
        Log warning-level message with optional metadata.

        T063: Warning log method
        """
        self._log(LogLevel.WARNING, message, metadata)

    def error(self, message: str, **metadata) -> None:
        """
        Log error-level message with optional metadata.

        T063: Error log method
        """
        self._log(LogLevel.ERROR, message, metadata)

    def critical(self, message: str, **metadata) -> None:
        """
        Log critical-level message with optional metadata.

        T063: Critical log method
        """
        self._log(LogLevel.CRITICAL, message, metadata)

    def bind(self, **context) -> "AuditLogger":
        """
        Create logger with bound context.

        T065: Bind method for additional context

        Args:
            **context: Key-value pairs to include in all logs

        Returns:
            New logger with context bound

        Example:
            stage_logger = logger.bind(stage_name="detection")
            stage_logger.info("Starting detection")
        """
        new_logger = AuditLogger.__new__(AuditLogger)
        new_logger.run_id = self.run_id
        new_logger.output_dir = self.output_dir
        new_logger._logger = self._logger.bind(**context)
        return new_logger

    def _log(self, level: LogLevel, message: str, metadata: Dict[str, Any]) -> None:
        """
        Internal log method.

        T066: Structured metadata support via **kwargs
        """
        # Map our LogLevel to loguru levels
        level_map = {
            LogLevel.DEBUG: "DEBUG",
            LogLevel.INFO: "INFO",
            LogLevel.WARNING: "WARNING",
            LogLevel.ERROR: "ERROR",
            LogLevel.CRITICAL: "CRITICAL",
        }

        loguru_level = level_map.get(level, "INFO")

        # T066: Add structured metadata
        self._logger.bind(**metadata).log(loguru_level, message)
