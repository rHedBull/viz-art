"""Type definitions for performance monitoring and debugging (Phase 3)."""

from enum import Enum
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
import uuid


# Enums (T011, T018, T021)

class RunStatus(str, Enum):
    """Pipeline run status."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class LogLevel(str, Enum):
    """Log severity levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AnnotationFormat(str, Enum):
    """Supported ground truth annotation formats."""
    COCO = "coco"
    YOLO = "yolo"
    PNG_MASKS = "png_masks"
    PLY_LABELS = "ply_labels"
    CUSTOM = "custom"


# Core Models

class StageExecution(BaseModel):
    """
    Records details of a single stage execution within a pipeline run.

    T015-T017: StageExecution model with validation
    """
    run_id: str
    stage_name: str
    stage_index: int = Field(ge=0)
    start_time: datetime
    end_time: datetime
    execution_time_ms: float = Field(gt=0)
    cpu_memory_mb: float = Field(gt=0)
    gpu_memory_mb: Optional[float] = Field(default=None, gt=0)
    input_paths: List[Path]
    output_paths: List[Path]
    success: bool = True
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None

    @validator("end_time")
    def validate_time_order(cls, v, values):
        """T016: Validate end_time > start_time"""
        if "start_time" in values and v <= values["start_time"]:
            raise ValueError("end_time must be after start_time")
        return v

    @validator("execution_time_ms")
    def validate_duration(cls, v, values):
        """T017: Validate execution_time_ms matches time delta ±1ms"""
        if "start_time" in values and "end_time" in values:
            expected = (values["end_time"] - values["start_time"]).total_seconds() * 1000
            if abs(v - expected) > 1.0:  # 1ms tolerance
                raise ValueError(
                    f"execution_time_ms {v} doesn't match time delta {expected:.2f}"
                )
        return v

    class Config:
        json_encoders = {
            Path: str,
            datetime: lambda dt: dt.isoformat(),
        }


class Run(BaseModel):
    """
    Represents a single pipeline execution with unique identifier and metadata.

    T012-T014: Run model with UUID and status validators
    """
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    config_snapshot: Dict[str, Any]
    input_files: List[Path]
    status: RunStatus = RunStatus.RUNNING
    error: Optional[str] = None
    stage_executions: List[StageExecution] = []
    total_duration_ms: float = 0.0
    output_dir: Path

    @validator("run_id")
    def validate_uuid(cls, v):
        """T013: Validate run_id is valid UUID v4"""
        try:
            uuid.UUID(v)
        except ValueError as e:
            raise ValueError(f"Invalid UUID: {v}") from e
        return v

    @validator("status")
    def validate_status_with_error(cls, v, values):
        """T014: Validate error message set when status=FAILED"""
        if v == RunStatus.FAILED and not values.get("error"):
            raise ValueError("error must be set when status=FAILED")
        return v

    class Config:
        json_encoders = {
            Path: str,
            datetime: lambda dt: dt.isoformat(),
        }


class AuditLog(BaseModel):
    """
    Structured log entry stored in JSON Lines format.

    T019-T020: AuditLog model with JSON Lines serialization
    """
    run_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    level: LogLevel
    stage_name: Optional[str] = None
    message: str
    metadata: Dict[str, Any] = {}

    def to_jsonl(self) -> str:
        """T020: Serialize to JSON Lines format"""
        return self.json() + "\n"

    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
        }


class GroundTruthDataset(BaseModel):
    """
    Collection of labeled samples for validation.

    T022-T023: GroundTruthDataset model with file validation
    """
    dataset_id: str
    name: str
    description: Optional[str] = None
    root_path: Path
    annotation_format: AnnotationFormat
    annotation_files: List[Path]
    sample_count: int = Field(gt=0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = {}

    @validator("annotation_files")
    def validate_files_exist(cls, v, values):
        """T023: Validate annotation files exist under root_path"""
        root = values.get("root_path")
        if root:
            for file in v:
                full_path = root / file
                if not full_path.exists():
                    raise ValueError(f"Annotation file not found: {full_path}")
        return v

    class Config:
        json_encoders = {
            Path: str,
            datetime: lambda dt: dt.isoformat(),
        }


class AccuracyMetrics(BaseModel):
    """
    Per-stage performance measurements calculated from ground truth comparison.

    T024-T026: AccuracyMetrics model with precision/recall validators
    """
    run_id: str
    stage_name: str
    dataset_id: str
    true_positives: int = Field(ge=0)
    false_positives: int = Field(ge=0)
    false_negatives: int = Field(ge=0)
    true_negatives: int = Field(ge=0, default=0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1_score: float = Field(ge=0.0, le=1.0)
    mean_average_precision: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    iou_mean: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    sample_count: int = Field(gt=0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @validator("precision")
    def validate_precision(cls, v, values):
        """T025: Validate precision = TP/(TP+FP)"""
        tp = values.get("true_positives", 0)
        fp = values.get("false_positives", 0)
        if tp + fp == 0:
            if v != 0.0:
                raise ValueError("precision must be 0 when TP+FP=0")
        else:
            expected = tp / (tp + fp)
            if abs(v - expected) > 1e-6:
                raise ValueError(
                    f"precision {v} doesn't match expected {expected:.6f}"
                )
        return v

    @validator("recall")
    def validate_recall(cls, v, values):
        """T026: Validate recall = TP/(TP+FN)"""
        tp = values.get("true_positives", 0)
        fn = values.get("false_negatives", 0)
        if tp + fn == 0:
            if v != 0.0:
                raise ValueError("recall must be 0 when TP+FN=0")
        else:
            expected = tp / (tp + fn)
            if abs(v - expected) > 1e-6:
                raise ValueError(
                    f"recall {v} doesn't match expected {expected:.6f}"
                )
        return v

    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
        }
