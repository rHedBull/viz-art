# Data Model: Accuracy Tracking & Analysis System

**Branch**: `004-accuracy-tracking-analysis` | **Date**: 2025-10-25

This document defines the key entities, their attributes, relationships, and validation rules for the accuracy tracking and analysis system.

---

## Entity Overview

```text
┌─────────────────────┐
│ Ground Truth        │
│ Dataset             │
└──────┬──────────────┘
       │ contains
       ▼
┌─────────────────────┐      validates      ┌─────────────────────┐
│ Ground Truth        ├────────────────────>│ Prediction          │
│ Sample              │                     │ (from pipeline)     │
└─────────────────────┘                     └──────┬──────────────┘
                                                   │ generates
                                                   ▼
                                            ┌─────────────────────┐
                                            │ Accuracy Metrics    │
                                            └──────┬──────────────┘
                                                   │ produces
                                                   ▼
                                            ┌─────────────────────┐
                                            │ Error Case          │
                                            │ (when wrong)        │
                                            └──────┬──────────────┘
                                                   │ clusters into
                                                   ▼
                                            ┌─────────────────────┐
                                            │ Error Pattern       │
                                            └─────────────────────┘

Performance Trend collects metrics across multiple runs ─────────────┐
                                                                      │
                                                                      ▼
                                               ┌──────────────────────────────┐
                                               │ Historical Performance       │
                                               │ (time-series data)           │
                                               └──────────────────────────────┘
```

---

## Entity Definitions

### 1. Ground Truth Dataset

**Purpose**: Collection of labeled samples with annotations for validating pipeline outputs.

**Attributes**:

| Attribute | Type | Required | Description | Validation |
|-----------|------|----------|-------------|------------|
| `dataset_id` | `str` | Yes | Unique identifier for dataset | UUID v4 format |
| `name` | `str` | Yes | Human-readable dataset name | 1-100 chars, non-empty |
| `description` | `str` | No | Dataset purpose and contents | Max 500 chars |
| `base_path` | `Path` | Yes | Root directory containing data | Must exist, readable |
| `annotation_path` | `Path` | Yes | Directory with ground truth labels | Must exist, readable |
| `annotation_format` | `AnnotationFormat` | Yes | Format of annotations | Enum: COCO, PASCAL_VOC, PCD_LABELS |
| `num_samples` | `int` | Yes | Total number of labeled samples | > 0 |
| `sample_ids` | `List[str]` | Yes | List of sample identifiers | Non-empty, unique IDs |
| `metadata` | `Dict[str, Any]` | No | Additional dataset properties | JSON-serializable |
| `created_at` | `datetime` | Yes | Dataset creation timestamp | ISO 8601 format |
| `updated_at` | `datetime` | Yes | Last update timestamp | >= created_at |

**Relationships**:
- Contains many `Ground Truth Samples` (1:N)
- Used by `Accuracy Metrics` for validation (1:N)

**Validation Rules**:
- `annotation_path` files must match `annotation_format` schema
- `sample_ids` must have corresponding files in `base_path` and `annotation_path`
- `num_samples` must equal `len(sample_ids)`

**Example**:
```python
from pathlib import Path
from enum import Enum

class AnnotationFormat(Enum):
    COCO = "coco"              # COCO JSON format for detection
    PASCAL_VOC = "pascal_voc"  # PASCAL VOC XML for segmentation
    PCD_LABELS = "pcd_labels"  # Point cloud annotation files

dataset = GroundTruthDataset(
    dataset_id="550e8400-e29b-41d4-a716-446655440000",
    name="validation_set_2024",
    description="Indoor object detection validation set",
    base_path=Path("/data/validation/images"),
    annotation_path=Path("/data/validation/annotations"),
    annotation_format=AnnotationFormat.COCO,
    num_samples=1000,
    sample_ids=["sample_0001", "sample_0002", ...],
    metadata={"scene_type": "indoor", "camera": "realsense"},
    created_at=datetime(2024, 10, 1),
    updated_at=datetime(2024, 10, 25),
)
```

---

### 2. Ground Truth Sample

**Purpose**: Individual labeled sample with stage-specific annotations.

**Attributes**:

| Attribute | Type | Required | Description | Validation |
|-----------|------|----------|-------------|------------|
| `sample_id` | `str` | Yes | Unique sample identifier | Matches dataset sample_ids |
| `dataset_id` | `str` | Yes | Parent dataset ID | Valid dataset reference |
| `stage_labels` | `Dict[str, Any]` | Yes | Labels for each pipeline stage | Stage names must match pipeline |
| `final_label` | `Any` | Yes | Ground truth for final output | Format matches task type |
| `annotation_format` | `AnnotationFormat` | Yes | Format of label data | Same as parent dataset |
| `confidence_scores` | `Dict[str, float]` | No | Confidence for each label | 0.0-1.0 range |
| `metadata` | `Dict[str, Any]` | No | Sample-specific metadata | JSON-serializable |
| `image_path` | `Optional[Path]` | No | Path to image file | Must exist if provided |
| `pointcloud_path` | `Optional[Path]` | No | Path to point cloud file | Must exist if provided |

**Relationships**:
- Belongs to one `Ground Truth Dataset` (N:1)
- Compared with `Prediction` to generate `Error Cases` (1:1 comparison)

**Validation Rules**:
- At least one of `image_path` or `pointcloud_path` must be provided
- `stage_labels` keys must match pipeline stage names
- Label format must match `annotation_format` schema
- Confidence scores must be in [0.0, 1.0] range

**Example**:
```python
sample = GroundTruthSample(
    sample_id="sample_0001",
    dataset_id="550e8400-e29b-41d4-a716-446655440000",
    stage_labels={
        "detection": {
            "boxes": [[100, 100, 200, 200], [300, 150, 400, 250]],
            "labels": ["person", "chair"],
        },
        "classification": ["person", "chair"],
    },
    final_label={
        "boxes": [[100, 100, 200, 200]],
        "labels": ["person"],
        "scores": [0.95],
    },
    annotation_format=AnnotationFormat.COCO,
    confidence_scores={"detection": 1.0, "classification": 1.0},
    image_path=Path("/data/validation/images/sample_0001.jpg"),
)
```

---

### 3. Accuracy Metrics

**Purpose**: Calculated performance measures for pipeline stages.

**Attributes**:

| Attribute | Type | Required | Description | Validation |
|-----------|------|----------|-------------|------------|
| `metrics_id` | `str` | Yes | Unique identifier | UUID v4 format |
| `run_id` | `str` | Yes | Associated pipeline run | Valid run ID from audit trail |
| `stage_name` | `str` | Yes | Pipeline stage name | Non-empty, matches pipeline |
| `metric_type` | `MetricType` | Yes | Type of metric | Enum: PRECISION, RECALL, F1, MAP, IOU |
| `value` | `float` | Yes | Calculated metric value | 0.0-1.0 range (or mAP: 0-100) |
| `per_class_values` | `Dict[str, float]` | No | Per-class breakdown | Class names → values |
| `confusion_matrix` | `Optional[np.ndarray]` | No | Confusion matrix | Square matrix, int values |
| `counts` | `AccuracyCounts` | Yes | Correct/wrong/invalid/unlabeled | See AccuracyCounts below |
| `timestamp` | `datetime` | Yes | Metric calculation time | ISO 8601 format |
| `ground_truth_ref` | `str` | Yes | Dataset ID used | Valid dataset reference |
| `metadata` | `Dict[str, Any]` | No | Additional metric info | JSON-serializable |

**Relationships**:
- Belongs to one pipeline run (N:1, via `run_id`)
- References one `Ground Truth Dataset` (N:1)
- Aggregated into `Performance Trend` (N:1 per run)

**Sub-entity: AccuracyCounts**:
```python
@dataclass
class AccuracyCounts:
    correct: int        # Predictions matching ground truth
    wrong: int          # Predictions not matching ground truth
    invalid: int        # Malformed predictions or ground truth
    unlabeled: int      # Samples without ground truth

    @property
    def total(self) -> int:
        return self.correct + self.wrong + self.invalid + self.unlabeled

    @property
    def accuracy(self) -> float:
        """Accuracy = correct / (correct + wrong)."""
        denominator = self.correct + self.wrong
        return self.correct / denominator if denominator > 0 else 0.0
```

**Validation Rules**:
- `value` must be in [0.0, 1.0] for precision/recall/F1/IoU, or [0, 100] for mAP
- All count fields (`correct`, `wrong`, etc.) must be >= 0
- `confusion_matrix` must be square with dimensions matching number of classes
- `per_class_values` keys must match class labels in ground truth

**Example**:
```python
from enum import Enum

class MetricType(Enum):
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1_score"
    MAP = "mean_average_precision"
    IOU = "intersection_over_union"

metrics = AccuracyMetrics(
    metrics_id="650e8400-e29b-41d4-a716-446655440001",
    run_id="run_2024_10_25_001",
    stage_name="detection",
    metric_type=MetricType.MAP,
    value=0.78,  # 78% mAP
    per_class_values={"person": 0.85, "chair": 0.71},
    counts=AccuracyCounts(correct=850, wrong=120, invalid=10, unlabeled=20),
    timestamp=datetime.now(),
    ground_truth_ref="550e8400-e29b-41d4-a716-446655440000",
)
```

---

### 4. Error Case

**Purpose**: Failed prediction instance with saved outputs and metadata.

**Attributes**:

| Attribute | Type | Required | Description | Validation |
|-----------|------|----------|-------------|------------|
| `error_id` | `str` | Yes | Unique error identifier | UUID v4 format |
| `run_id` | `str` | Yes | Pipeline run that produced error | Valid run ID |
| `stage_name` | `str` | Yes | Stage where error occurred | Non-empty, matches pipeline |
| `sample_id` | `str` | Yes | Sample that failed | Must exist in ground truth |
| `error_type` | `ErrorType` | Yes | Type of failure | Enum: FALSE_POSITIVE, FALSE_NEGATIVE, MISCLASSIFICATION, LOW_IOU |
| `severity` | `ErrorSeverity` | Yes | Impact level | Enum: CRITICAL, HIGH, MEDIUM, LOW |
| `prediction` | `Any` | Yes | Model prediction output | Format matches stage output |
| `ground_truth` | `Any` | Yes | Expected output | Format matches annotation format |
| `iou` | `Optional[float]` | No | IoU score (if applicable) | 0.0-1.0 range |
| `confidence` | `Optional[float]` | No | Prediction confidence | 0.0-1.0 range |
| `saved_artifacts` | `Dict[str, Path]` | Yes | Paths to saved outputs | Paths must exist |
| `timestamp` | `datetime` | Yes | When error occurred | ISO 8601 format |
| `metadata` | `Dict[str, Any]` | No | Additional error context | JSON-serializable |

**Relationships**:
- Belongs to one pipeline run (N:1)
- Belongs to one stage (N:1)
- Grouped into `Error Patterns` (N:1)
- References `Ground Truth Sample` (N:1)

**Sub-entities**:
```python
class ErrorType(Enum):
    FALSE_POSITIVE = "false_positive"    # Detected when not present
    FALSE_NEGATIVE = "false_negative"    # Missed detection
    MISCLASSIFICATION = "misclassification"  # Wrong class label
    LOW_IOU = "low_iou"                 # Poor localization

class ErrorSeverity(Enum):
    CRITICAL = "critical"  # Pipeline fails entirely
    HIGH = "high"         # Wrong output (FP/FN with IoU < 0.3)
    MEDIUM = "medium"     # Degraded quality (misclass or 0.3 <= IoU < 0.5)
    LOW = "low"           # Minor deviation (IoU >= 0.5)
```

**Validation Rules**:
- `iou` required for detection/segmentation errors, must be in [0.0, 1.0]
- `confidence` must be in [0.0, 1.0] if provided
- `saved_artifacts` must include at least: `prediction_path`, `ground_truth_path`
- Severity must match spec assumptions (see research.md)

**Example**:
```python
error = ErrorCase(
    error_id="750e8400-e29b-41d4-a716-446655440002",
    run_id="run_2024_10_25_001",
    stage_name="detection",
    sample_id="sample_0042",
    error_type=ErrorType.FALSE_POSITIVE,
    severity=ErrorSeverity.HIGH,
    prediction={"boxes": [[150, 150, 250, 250]], "labels": ["chair"], "scores": [0.72]},
    ground_truth={"boxes": [], "labels": []},  # No objects present
    iou=0.0,
    confidence=0.72,
    saved_artifacts={
        "prediction_image": Path("/output/runs/run_001/errors/sample_0042_pred.jpg"),
        "ground_truth_image": Path("/output/runs/run_001/errors/sample_0042_gt.jpg"),
        "diff_visualization": Path("/output/runs/run_001/errors/sample_0042_diff.jpg"),
    },
    timestamp=datetime.now(),
)
```

---

### 5. Error Pattern

**Purpose**: Clustered group of similar failures for analysis.

**Attributes**:

| Attribute | Type | Required | Description | Validation |
|-----------|------|----------|-------------|------------|
| `pattern_id` | `str` | Yes | Unique pattern identifier | Composite key: `{stage}_{error_type}` |
| `run_id` | `str` | Yes | Pipeline run | Valid run ID |
| `stage_name` | `str` | Yes | Stage with errors | Non-empty, matches pipeline |
| `error_type` | `ErrorType` | Yes | Type of errors in cluster | Enum value |
| `severity` | `ErrorSeverity` | Yes | Aggregate severity | Most severe in cluster |
| `error_count` | `int` | Yes | Number of errors in pattern | > 0 |
| `affected_samples` | `List[str]` | Yes | Sample IDs with this pattern | Non-empty, unique |
| `statistics` | `PatternStatistics` | Yes | Aggregate stats | See PatternStatistics below |
| `suggested_cause` | `str` | No | Inferred root cause | Max 200 chars |
| `timestamp` | `datetime` | Yes | Pattern detection time | ISO 8601 format |

**Relationships**:
- Contains many `Error Cases` (1:N)
- Belongs to one pipeline run (N:1)

**Sub-entity: PatternStatistics**:
```python
@dataclass
class PatternStatistics:
    avg_iou: float              # Average IoU across errors (if applicable)
    avg_confidence: float       # Average prediction confidence
    min_iou: float              # Worst IoU in pattern
    max_iou: float              # Best IoU in pattern
    sample_preview: List[str]   # First 5 sample IDs for preview
```

**Validation Rules**:
- `error_count` must equal `len(affected_samples)`
- `pattern_id` format: `{stage_name}_{error_type.value}` (e.g., "detection_false_positive")
- `statistics.avg_iou` and confidence must be in [0.0, 1.0]
- All `affected_samples` must exist in the run's error cases

**Example**:
```python
pattern = ErrorPattern(
    pattern_id="detection_false_positive",
    run_id="run_2024_10_25_001",
    stage_name="detection",
    error_type=ErrorType.FALSE_POSITIVE,
    severity=ErrorSeverity.HIGH,
    error_count=45,
    affected_samples=["sample_0042", "sample_0103", ...],
    statistics=PatternStatistics(
        avg_iou=0.12,
        avg_confidence=0.68,
        min_iou=0.0,
        max_iou=0.28,
        sample_preview=["sample_0042", "sample_0103", "sample_0221", "sample_0305", "sample_0412"]
    ),
    suggested_cause="Model generating spurious detections - check confidence threshold",
    timestamp=datetime.now(),
)
```

---

### 6. Performance Trend

**Purpose**: Historical accuracy data across multiple runs.

**Attributes**:

| Attribute | Type | Required | Description | Validation |
|-----------|------|----------|-------------|------------|
| `trend_id` | `str` | Yes | Unique identifier | UUID v4 format |
| `dataset_id` | `str` | Yes | Ground truth dataset used | Valid dataset reference |
| `run_ids` | `List[str]` | Yes | Pipeline runs included | Non-empty, chronological order |
| `timestamps` | `List[datetime]` | Yes | Run execution times | Same length as run_ids |
| `per_stage_metrics` | `Dict[str, List[MetricSnapshot]]` | Yes | Metrics per stage over time | Stage → time-series |
| `pipeline_version` | `Optional[str]` | No | Pipeline version tag | Semantic version format |
| `configuration_params` | `Dict[str, Any]` | No | Config used for runs | JSON-serializable |
| `aggregate_statistics` | `TrendStatistics` | Yes | Summary stats | See TrendStatistics below |
| `created_at` | `datetime` | Yes | Trend record creation | ISO 8601 format |
| `updated_at` | `datetime` | Yes | Last update | >= created_at |

**Relationships**:
- References multiple pipeline runs (1:N)
- Uses one `Ground Truth Dataset` (N:1)
- Contains multiple `MetricSnapshot` entries (1:N)

**Sub-entity: MetricSnapshot**:
```python
@dataclass
class MetricSnapshot:
    run_id: str
    timestamp: datetime
    metric_type: MetricType
    value: float
    counts: AccuracyCounts
```

**Sub-entity: TrendStatistics**:
```python
@dataclass
class TrendStatistics:
    best_run_id: str            # Run with highest metric
    best_value: float           # Best metric value
    worst_run_id: str           # Run with lowest metric
    worst_value: float          # Worst metric value
    mean_value: float           # Average across runs
    std_dev: float              # Standard deviation
    regression_detected: bool   # Accuracy drop > threshold
    regression_details: Optional[str]  # Description if detected
```

**Validation Rules**:
- `run_ids` and `timestamps` must have same length
- `timestamps` must be in ascending order
- `per_stage_metrics` must have consistent stage names across runs
- `aggregate_statistics` values must match computed stats from `per_stage_metrics`

**Example**:
```python
trend = PerformanceTrend(
    trend_id="850e8400-e29b-41d4-a716-446655440003",
    dataset_id="550e8400-e29b-41d4-a716-446655440000",
    run_ids=["run_001", "run_002", "run_003"],
    timestamps=[datetime(2024, 10, 1), datetime(2024, 10, 15), datetime(2024, 10, 25)],
    per_stage_metrics={
        "detection": [
            MetricSnapshot("run_001", datetime(2024, 10, 1), MetricType.MAP, 0.75, ...),
            MetricSnapshot("run_002", datetime(2024, 10, 15), MetricType.MAP, 0.78, ...),
            MetricSnapshot("run_003", datetime(2024, 10, 25), MetricType.MAP, 0.71, ...),  # Regression!
        ]
    },
    pipeline_version="v1.2.3",
    aggregate_statistics=TrendStatistics(
        best_run_id="run_002",
        best_value=0.78,
        worst_run_id="run_003",
        worst_value=0.71,
        mean_value=0.747,
        std_dev=0.029,
        regression_detected=True,  # 0.78 → 0.71 is 9% drop
        regression_details="Detection mAP dropped 9% from run_002 to run_003",
    ),
    created_at=datetime(2024, 10, 1),
    updated_at=datetime(2024, 10, 25),
)
```

---

## Storage Schema

### File-Based Storage Structure

```text
output/
├── ground_truth/
│   ├── datasets/
│   │   └── {dataset_id}.json              # GroundTruthDataset metadata
│   └── samples/
│       └── {dataset_id}/
│           └── {sample_id}.json           # GroundTruthSample data
├── metrics/
│   ├── {run_id}_metrics.parquet           # AccuracyMetrics (time-series)
│   └── {run_id}_metadata.json             # Metric metadata
├── errors/
│   ├── {run_id}_errors.json               # ErrorCase records
│   ├── {run_id}_patterns.json             # ErrorPattern summaries
│   └── {run_id}/
│       └── {error_id}/                    # Saved artifacts per error
│           ├── prediction.jpg
│           ├── ground_truth.jpg
│           └── diff_visualization.jpg
└── trends/
    └── {dataset_id}_trends.parquet        # PerformanceTrend (historical)
```

### Parquet Schema for Metrics

```python
# AccuracyMetrics Parquet schema
pyarrow.schema([
    ("metrics_id", pa.string()),
    ("run_id", pa.string()),
    ("stage_name", pa.string()),
    ("metric_type", pa.string()),
    ("value", pa.float64()),
    ("correct", pa.int64()),
    ("wrong", pa.int64()),
    ("invalid", pa.int64()),
    ("unlabeled", pa.int64()),
    ("timestamp", pa.timestamp('us')),
    ("ground_truth_ref", pa.string()),
])
```

---

## State Transitions

### Error Case Lifecycle

```text
[Prediction Made] → [Comparison with Ground Truth] → [Error Detected]
                                                            │
                                                            ▼
                                                    [Error Case Created]
                                                            │
                                                            ▼
                                                    [Artifacts Saved]
                                                            │
                                                            ▼
                                                    [Pattern Clustering]
                                                            │
                                                            ▼
                                                    [Added to Pattern]
                                                            │
                                                            ▼
                                                    [Available in Browser]
```

### Performance Trend Lifecycle

```text
[Run Completes] → [Metrics Calculated] → [Metrics Stored]
                                              │
                                              ▼
                                      [Added to Trend]
                                              │
                                              ▼
                                      [Regression Check]
                                              │
                                              ├─> [Regression Detected] → [Alert Generated]
                                              │
                                              └─> [No Regression] → [Normal Update]
```

---

## Validation Summary

### Cross-Entity Constraints

1. **Run ID Consistency**: All entities with `run_id` must reference valid audit trail entries
2. **Dataset ID Consistency**: All `dataset_id` references must point to existing `GroundTruthDataset` records
3. **Sample ID Consistency**: `Error Cases` must reference `sample_ids` that exist in the associated `GroundTruthDataset`
4. **Metric Ranges**: All metric values must be in valid ranges (0-1 or 0-100 for mAP)
5. **Path Existence**: All file paths in `saved_artifacts` must exist and be readable
6. **Timestamp Ordering**: `updated_at` >= `created_at` for all entities with both fields

### Data Integrity Checks

```python
# Example validation functions
def validate_accuracy_metrics(metrics: AccuracyMetrics) -> None:
    """Validate metric consistency."""
    assert 0 <= metrics.value <= 1 or 0 <= metrics.value <= 100
    assert metrics.counts.total > 0
    assert metrics.counts.correct + metrics.counts.wrong > 0  # For accuracy calc

def validate_error_case(error: ErrorCase) -> None:
    """Validate error case data."""
    if error.iou is not None:
        assert 0 <= error.iou <= 1
    if error.confidence is not None:
        assert 0 <= error.confidence <= 1
    assert all(path.exists() for path in error.saved_artifacts.values())

def validate_performance_trend(trend: PerformanceTrend) -> None:
    """Validate trend consistency."""
    assert len(trend.run_ids) == len(trend.timestamps)
    assert trend.timestamps == sorted(trend.timestamps)
    assert trend.updated_at >= trend.created_at
```

---

## Migration Notes

### Integration with Existing Entities

This data model extends the following existing viz-art entities:

1. **Audit Trail** (`src/viz_art/audit/run_tracker.py`):
   - `run_id` references come from existing audit log entries
   - Accuracy metrics link to audit trail via `run_id`

2. **OutputSaver** (Phase 3):
   - `saved_artifacts` paths follow existing OutputSaver directory structure
   - Error case artifacts stored using existing OutputSaver validation mode

3. **Performance Metrics** (`src/viz_art/performance/metrics_storage.py`):
   - `AccuracyMetrics` extends existing metrics storage using same Parquet format
   - Historical trends use existing PyArrow infrastructure

---

## References

- **Spec**: `spec.md` (FR-001 to FR-021, Key Entities section)
- **Research**: `research.md` (error severity definitions, metric library choices)
- **Existing Code**: `src/viz_art/validation/metrics.py` (MetricsCalculator patterns)
