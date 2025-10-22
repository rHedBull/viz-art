"""Result data structures for pipeline execution.

This module defines immutable dataclasses for tracking pipeline execution results.
All result objects are frozen to ensure they cannot be modified after creation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional
from uuid import uuid4


class StageStatus(Enum):
    """Status of a single stage execution."""

    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class RunStatus(Enum):
    """Status of a complete pipeline run."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class StageResult:
    """Result from a single stage execution.

    This is an immutable record of one stage's execution within a pipeline run.
    All fields are set at creation and cannot be modified.

    Attributes:
        stage_name: Name of the executed stage
        status: Execution status (SUCCESS, FAILED, SKIPPED)
        started_at: Timestamp when stage execution began (UTC)
        duration_ms: Execution time in milliseconds
        outputs: Dictionary of stage outputs (empty if failed/skipped)
        error: Error message if status is FAILED, None otherwise
    """

    stage_name: str
    status: StageStatus
    started_at: datetime
    duration_ms: float
    outputs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def __post_init__(self):
        """Validate result consistency after initialization."""
        if self.duration_ms < 0:
            raise ValueError(f"duration_ms must be >= 0, got {self.duration_ms}")
        if self.status == StageStatus.FAILED and self.error is None:
            raise ValueError("error must be set when status is FAILED")


@dataclass(frozen=True)
class PipelineRun:
    """Result from a complete pipeline execution.

    This is an immutable record of one pipeline execution with all stage results.

    Attributes:
        run_id: Unique identifier for this run (UUID)
        pipeline_name: Name of the executed pipeline
        started_at: Timestamp when pipeline execution began (UTC)
        completed_at: Timestamp when pipeline execution finished (UTC)
        status: Overall execution status
        inputs: Dictionary of input data provided to run()
        outputs: Dictionary of all stage outputs (stage_name -> outputs)
        stage_results: List of individual stage execution results
        error: Error message if status is FAILED, None otherwise
    """

    run_id: str = field(default_factory=lambda: str(uuid4()))
    pipeline_name: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.utcnow())
    completed_at: Optional[datetime] = None
    status: RunStatus = RunStatus.RUNNING
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    stage_results: List[StageResult] = field(default_factory=list)
    error: Optional[str] = None

    def __post_init__(self):
        """Validate run consistency after initialization."""
        if self.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            if self.completed_at is None:
                raise ValueError("completed_at must be set when run is finished")


@dataclass(frozen=True)
class BatchResult:
    """Result from batch processing execution.

    This is an immutable record of processing multiple images through a pipeline.

    Attributes:
        batch_id: Unique identifier for this batch execution (UUID)
        total_files: Total number of images discovered and attempted
        successful: Number of successfully processed images
        failed: Number of failed images
        run_results: List of individual pipeline runs (one per image)
        started_at: Timestamp when batch processing began (UTC)
        completed_at: Timestamp when batch processing finished (UTC)
        report_path: Path to generated HTML report
    """

    batch_id: str = field(default_factory=lambda: str(uuid4()))
    total_files: int = 0
    successful: int = 0
    failed: int = 0
    run_results: List[PipelineRun] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.utcnow())
    completed_at: datetime = field(default_factory=lambda: datetime.utcnow())
    report_path: str = ""

    def __post_init__(self):
        """Validate batch result consistency."""
        if self.total_files != (self.successful + self.failed):
            raise ValueError(
                f"total_files ({self.total_files}) must equal "
                f"successful ({self.successful}) + failed ({self.failed})"
            )
        if len(self.run_results) != self.total_files:
            raise ValueError(
                f"run_results length ({len(self.run_results)}) must match "
                f"total_files ({self.total_files})"
            )
