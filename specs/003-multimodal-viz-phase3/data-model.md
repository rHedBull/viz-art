# Data Model: Performance Monitoring & Debugging System

**Feature**: Performance Monitoring & Debugging System
**Branch**: `003-multimodal-viz-phase3`
**Date**: 2025-10-23

## Overview

This document defines the data entities and their relationships for Phase 3. All entities use Pydantic for validation and serialization. The model supports three primary concerns: performance profiling, audit logging, and ground truth validation.

## Core Entities

### 1. Run

Represents a single pipeline execution with unique identifier and metadata.

**Fields**:
- `run_id: str` - UUID v4 unique identifier (e.g., "a3b5c7d9-1234-5678-90ab-cdef12345678")
- `timestamp: datetime` - Pipeline start time (UTC, ISO 8601 format)
- `config_snapshot: Dict[str, Any]` - Pipeline configuration at execution time (serialized OmegaConf)
- `input_files: List[Path]` - Absolute paths to input images/point clouds
- `status: RunStatus` - Enum: RUNNING, COMPLETED, FAILED
- `error: Optional[str]` - Error message if status=FAILED (None otherwise)
- `stage_executions: List[StageExecution]` - Ordered list of stage runs (populated as pipeline progresses)
- `total_duration_ms: float` - End-to-end execution time (sum of stage times + overhead)
- `output_dir: Path` - Root directory for all outputs (e.g., `output/runs/{run_id}/`)

**Relationships**:
- One-to-many with `StageExecution` (one run contains multiple stage executions)
- References `GroundTruthDataset` if validation mode enabled

**Validation Rules**:
- `run_id` must be valid UUID v4 format
- `timestamp` must be in UTC timezone
- `status` transitions: RUNNING → {COMPLETED, FAILED} (no backward transitions)
- `input_files` must exist at time of Run creation
- `total_duration_ms` must equal sum of `stage_executions[*].execution_time_ms` ± 5% (accounts for overhead)

**State Transitions**:
```
[Created] → RUNNING → COMPLETED (if all stages succeed)
                   → FAILED (if any stage fails)
```

**Pydantic Model**:
```python
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import uuid

class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class Run(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    config_snapshot: Dict[str, Any]
    input_files: List[Path]
    status: RunStatus = RunStatus.RUNNING
    error: Optional[str] = None
    stage_executions: List["StageExecution"] = []
    total_duration_ms: float = 0.0
    output_dir: Path

    @validator("run_id")
    def validate_uuid(cls, v):
        uuid.UUID(v)  # Raises ValueError if invalid
        return v

    @validator("status")
    def validate_status_with_error(cls, v, values):
        if v == RunStatus.FAILED and not values.get("error"):
            raise ValueError("error must be set when status=FAILED")
        return v

    class Config:
        json_encoders = {
            Path: str,
            datetime: lambda dt: dt.isoformat(),
        }
```

---

### 2. StageExecution

Records details of a single stage execution within a pipeline run.

**Fields**:
- `run_id: str` - Foreign key to parent Run (UUID v4)
- `stage_name: str` - Human-readable stage identifier (e.g., "preprocessing", "detection")
- `stage_index: int` - Execution order within pipeline (0-based)
- `start_time: datetime` - Stage start timestamp (UTC)
- `end_time: datetime` - Stage completion timestamp (UTC)
- `execution_time_ms: float` - Computed duration (end_time - start_time in milliseconds)
- `cpu_memory_mb: float` - Peak CPU memory usage (RSS in MB)
- `gpu_memory_mb: Optional[float]` - Peak GPU memory usage (MB, None if no GPU)
- `input_paths: List[Path]` - Stage inputs (outputs from previous stage or initial inputs)
- `output_paths: List[Path]` - Stage outputs saved by OutputSaver
- `success: bool` - True if stage completed without errors
- `error_message: Optional[str]` - Exception message if success=False
- `error_traceback: Optional[str]` - Full stack trace for debugging

**Relationships**:
- Many-to-one with `Run` (many stage executions per run)
- Generates `PerformanceMetrics` entries for storage

**Validation Rules**:
- `end_time` must be after `start_time`
- `execution_time_ms` must equal `(end_time - start_time).total_seconds() * 1000` ± 1ms (tolerance)
- `cpu_memory_mb` must be positive
- `gpu_memory_mb` nullable (None if GPU unavailable)
- If `success=False`, `error_message` must be set

**Pydantic Model**:
```python
class StageExecution(BaseModel):
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
        if v <= values.get("start_time"):
            raise ValueError("end_time must be after start_time")
        return v

    @validator("execution_time_ms")
    def validate_duration(cls, v, values):
        if "start_time" in values and "end_time" in values:
            expected = (values["end_time"] - values["start_time"]).total_seconds() * 1000
            if abs(v - expected) > 1.0:  # 1ms tolerance
                raise ValueError(f"execution_time_ms {v} doesn't match time delta {expected}")
        return v
```

---

### 3. PerformanceMetrics

Aggregated performance data stored in Parquet format for historical analysis.

**Fields** (Parquet Schema):
- `run_id: string` - Links to Run
- `timestamp: timestamp[us, tz=UTC]` - When metric was recorded
- `stage_name: string` - Stage identifier
- `stage_index: int32` - Execution order
- `execution_time_ms: float64` - Stage duration
- `cpu_memory_mb: float64` - CPU memory usage
- `gpu_memory_mb: float64 (nullable)` - GPU memory (null if unavailable)
- `input_count: int32` - Number of inputs processed
- `output_count: int32` - Number of outputs generated
- `success: bool` - Stage success/failure

**Storage Format**:
- **Location**: `output/metrics/{stage_name}.parquet` (one file per stage)
- **Partitioning**: By stage name for efficient filtering
- **Compression**: Snappy (good balance of speed and compression ratio)
- **Append Mode**: New runs append to existing Parquet files

**Query Patterns**:
```python
import pyarrow.parquet as pq

# Filter by date range and stage
table = pq.read_table(
    "output/metrics/detection.parquet",
    filters=[
        ("timestamp", ">=", start_date),
        ("timestamp", "<=", end_date),
    ]
)

# Aggregate statistics
import pandas as pd
df = table.to_pandas()
stats = df.groupby("stage_name")["execution_time_ms"].agg(["median", "mean", "std", "min", "max"])
```

**Schema Evolution**:
- New columns can be added with null defaults (backward compatible)
- Existing columns cannot be removed (breaking change)
- Type changes require migration script

---

### 4. AuditLog

Structured log entry stored in JSON Lines format.

**Fields**:
- `run_id: str` - Links to Run (UUID v4)
- `timestamp: datetime` - Log entry creation time (UTC)
- `level: LogLevel` - Enum: DEBUG, INFO, WARNING, ERROR, CRITICAL
- `stage_name: Optional[str]` - Stage context (None for pipeline-level logs)
- `message: str` - Human-readable log message
- `metadata: Dict[str, Any]` - Arbitrary context (input paths, outputs, errors, etc.)

**Relationships**:
- Many-to-one with `Run` (many log entries per run)
- References `StageExecution` via `stage_name` and `run_id`

**Storage Format**:
- **Location**: `output/logs/{YYYY-MM-DD}.jsonl` (one file per date for fast date filtering)
- **Format**: JSON Lines (one JSON object per line)
- **Rotation**: Daily at midnight UTC
- **Compression**: Gzip after 7 days (`.jsonl.gz`)

**Example Entry**:
```json
{
  "run_id": "a3b5c7d9-1234-5678-90ab-cdef12345678",
  "timestamp": "2025-10-23T14:32:10.123456Z",
  "level": "ERROR",
  "stage_name": "detection",
  "message": "Stage failed: Invalid input shape",
  "metadata": {
    "input_shape": [3, 224, 224],
    "expected_shape": [3, 512, 512],
    "error_type": "ValueError",
    "traceback": "..."
  }
}
```

**Pydantic Model**:
```python
class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class AuditLog(BaseModel):
    run_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    level: LogLevel
    stage_name: Optional[str] = None
    message: str
    metadata: Dict[str, Any] = {}

    def to_jsonl(self) -> str:
        """Serialize to JSON Lines format"""
        return self.json() + "\n"
```

---

### 5. GroundTruthDataset

Collection of labeled samples for validation.

**Fields**:
- `dataset_id: str` - Unique identifier (user-defined, e.g., "coco_val_2017")
- `name: str` - Human-readable name
- `description: Optional[str]` - Dataset description
- `root_path: Path` - Base directory containing images/point clouds and annotations
- `annotation_format: AnnotationFormat` - Enum: COCO, YOLO, PNG_MASKS, PLY_LABELS, CUSTOM
- `annotation_files: List[Path]` - Paths to annotation files (relative to root_path)
- `sample_count: int` - Total number of labeled samples
- `created_at: datetime` - Dataset creation timestamp
- `metadata: Dict[str, Any]` - Arbitrary metadata (annotation source, version, class mapping, etc.)

**Relationships**:
- One-to-many with `Annotation` (one dataset contains many annotations)
- Referenced by `Run` when validation enabled

**Validation Rules**:
- `root_path` must exist and be readable
- `annotation_files` must exist under `root_path`
- `sample_count` must match actual annotation count (validated on load)
- `annotation_format` must have registered loader

**Pydantic Model**:
```python
class AnnotationFormat(str, Enum):
    COCO = "coco"
    YOLO = "yolo"
    PNG_MASKS = "png_masks"
    PLY_LABELS = "ply_labels"
    CUSTOM = "custom"

class GroundTruthDataset(BaseModel):
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
        root = values.get("root_path")
        if root:
            for file in v:
                full_path = root / file
                if not full_path.exists():
                    raise ValueError(f"Annotation file not found: {full_path}")
        return v
```

---

### 6. AccuracyMetrics

Per-stage performance measurements calculated from ground truth comparison.

**Fields**:
- `run_id: str` - Links to Run
- `stage_name: str` - Stage being evaluated
- `dataset_id: str` - Links to GroundTruthDataset
- `true_positives: int` - Correctly predicted positive samples
- `false_positives: int` - Incorrectly predicted as positive
- `false_negatives: int` - Incorrectly predicted as negative
- `true_negatives: int` - Correctly predicted negative samples (optional, often N/A for detection)
- `precision: float` - TP / (TP + FP), range [0, 1]
- `recall: float` - TP / (TP + FN), range [0, 1]
- `f1_score: float` - 2 * (precision * recall) / (precision + recall), range [0, 1]
- `mean_average_precision: Optional[float]` - mAP for detection tasks (None for others)
- `iou_mean: Optional[float]` - Mean IoU for segmentation tasks (None for others)
- `sample_count: int` - Number of samples evaluated
- `timestamp: datetime` - When metrics were computed

**Relationships**:
- Many-to-one with `Run` (multiple stages evaluated per run)
- Many-to-one with `GroundTruthDataset` (metrics computed against specific dataset)

**Validation Rules**:
- `true_positives`, `false_positives`, `false_negatives`, `true_negatives` must be non-negative
- `precision`, `recall`, `f1_score` must be in range [0, 1]
- If `true_positives + false_positives = 0`, then `precision = 0.0` (by convention)
- If `true_positives + false_negatives = 0`, then `recall = 0.0` (by convention)
- `sample_count` must equal sum of confusion matrix entries (for binary classification)

**Pydantic Model**:
```python
class AccuracyMetrics(BaseModel):
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
        tp = values.get("true_positives", 0)
        fp = values.get("false_positives", 0)
        if tp + fp == 0:
            assert v == 0.0, "precision must be 0 when TP+FP=0"
        else:
            expected = tp / (tp + fp)
            assert abs(v - expected) < 1e-6, f"precision {v} doesn't match {expected}"
        return v

    @validator("recall")
    def validate_recall(cls, v, values):
        tp = values.get("true_positives", 0)
        fn = values.get("false_negatives", 0)
        if tp + fn == 0:
            assert v == 0.0, "recall must be 0 when TP+FN=0"
        else:
            expected = tp / (tp + fn)
            assert abs(v - expected) < 1e-6, f"recall {v} doesn't match {expected}"
        return v
```

---

## Entity Relationships Diagram

```
┌─────────────────┐
│      Run        │
│  - run_id (PK)  │
│  - timestamp    │
│  - status       │
│  - config       │
└────────┬────────┘
         │ 1
         │
         │ N
┌────────┴────────────┐
│  StageExecution     │
│  - run_id (FK)      │◄────┐
│  - stage_name       │     │
│  - execution_time   │     │
│  - memory_usage     │     │
└─────────────────────┘     │
         │                  │
         │ Generates        │
         ▼                  │
┌─────────────────────┐     │
│ PerformanceMetrics  │     │
│  - run_id (FK)      │     │
│  - stage_name       │     │
│  - metrics (Parquet)│     │
└─────────────────────┘     │
                            │
┌─────────────────────┐     │
│    AuditLog         │     │
│  - run_id (FK)      │────►│
│  - stage_name       │     │
│  - message          │     │
│  - metadata (JSONL) │     │
└─────────────────────┘     │
                            │
┌─────────────────────┐     │
│ GroundTruthDataset  │     │
│  - dataset_id (PK)  │     │
│  - annotations      │     │
└────────┬────────────┘     │
         │ 1                │
         │                  │
         │ N                │
┌────────┴────────────┐     │
│  AccuracyMetrics    │     │
│  - run_id (FK)      │────►│
│  - dataset_id (FK)  │     │
│  - stage_name       │     │
│  - precision/recall │     │
└─────────────────────┘
```

## Storage Layout

```
output/
├── runs/
│   └── {run_id}/
│       ├── stages/
│       │   ├── preprocessing/
│       │   │   ├── image_001.png
│       │   │   └── image_002.png
│       │   └── detection/
│       │       ├── result_001.json
│       │       └── result_002.json
│       └── run_metadata.json  # Serialized Run entity
├── logs/
│   ├── 2025-10-23.jsonl       # Today's logs (AuditLog entries)
│   ├── 2025-10-22.jsonl.gz    # Yesterday's logs (compressed)
│   └── ...
├── metrics/
│   ├── preprocessing.parquet  # PerformanceMetrics for preprocessing stage
│   ├── detection.parquet
│   └── postprocessing.parquet
└── ground_truth/
    ├── coco_val_2017/
    │   ├── annotations/
    │   │   └── instances_val2017.json
    │   └── images/
    │       └── ...
    └── dataset_registry.json  # List of GroundTruthDataset entities
```

## Data Migration Strategy

Since this is Phase 3 (new feature), no migration needed from prior phases. However, establish patterns for future:

1. **Schema Versioning**: Include `schema_version: str` in all Pydantic models
2. **Backward Compatibility**: New fields must have defaults
3. **Migration Scripts**: Store in `migrations/` with version tags
4. **Data Retention**: Configure retention policies in config (default: 90 days for logs, 365 days for metrics)

## Performance Considerations

- **Parquet Partitioning**: By stage name reduces scan size (10x faster for single-stage queries)
- **JSONL Rotation**: Daily rotation prevents large file scans (100x faster for date-filtered queries)
- **Lazy Loading**: Query interfaces only load needed files (memory efficient)
- **Caching**: LRU cache for repeated queries (e.g., dashboard loads same date range)
- **Index Files**: Optional `.index.json` for fast run_id lookups without scanning all logs

## Validation Testing

All Pydantic models include comprehensive validators. Example test coverage:

```python
def test_run_status_transitions():
    run = Run(config_snapshot={}, input_files=[], output_dir=Path("/tmp"))
    assert run.status == RunStatus.RUNNING

    run.status = RunStatus.COMPLETED
    assert run.status == RunStatus.COMPLETED

    # Cannot transition back
    with pytest.raises(ValidationError):
        run.status = RunStatus.RUNNING

def test_stage_execution_timing_validation():
    start = datetime.utcnow()
    end = start + timedelta(milliseconds=100)

    # Valid
    stage = StageExecution(
        run_id="test",
        stage_name="test",
        stage_index=0,
        start_time=start,
        end_time=end,
        execution_time_ms=100.0,
        cpu_memory_mb=100.0,
        input_paths=[],
        output_paths=[],
    )
    assert stage.execution_time_ms == 100.0

    # Invalid: mismatched duration
    with pytest.raises(ValidationError):
        StageExecution(
            ...,
            execution_time_ms=200.0,  # Doesn't match 100ms delta
        )
```

## Next Steps

Proceed to Phase 1: Generate API contracts (OpenAPI schemas for Python APIs if exposing REST interface, or Python type stubs for library usage).
