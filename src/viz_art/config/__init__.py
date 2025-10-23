"""Configuration management for viz-art pipeline.

Provides configuration options for monitoring and debugging features.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class MonitoringConfig:
    """Configuration for Phase 3 monitoring features.

    Attributes:
        enable_profiling: Enable performance profiling (timing and memory tracking)
        enable_audit_logging: Enable structured audit logging for debugging
        enable_validation: Enable ground truth validation
        output_dir: Base directory for monitoring outputs
        metrics_retention_days: How many days to retain metrics (0 = forever)
        log_retention_days: How many days to retain logs (0 = forever)
        ground_truth_dataset: Optional ground truth dataset for validation
    """

    enable_profiling: bool = False
    enable_audit_logging: bool = False
    enable_validation: bool = False
    output_dir: Path = field(default_factory=lambda: Path("output"))
    metrics_retention_days: int = 365
    log_retention_days: int = 30
    ground_truth_dataset: Optional[str] = None

    def __post_init__(self):
        """Convert output_dir to Path if it's a string."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)


# Public exports
__all__ = ["MonitoringConfig"]
