# API Contracts: Performance Monitoring & Debugging System

**Feature**: `003-multimodal-viz-phase3`
**Purpose**: Define stable public APIs for Phase 3 functionality

## Overview

This directory contains API contract definitions for the Performance Monitoring & Debugging System. These contracts use Python Protocol classes to define interfaces that implementations must satisfy. All public APIs documented here must maintain backward compatibility across minor version releases.

## Contract Files

### 1. `performance_api.py`
Performance profiling and metrics storage interfaces.

**Key Interfaces**:
- `Profiler`: Decorator/context manager for automatic performance tracking
- `MetricsStorage`: Parquet-based metrics persistence and querying
- `PerformanceDashboard`: Plotly visualization generation

**Usage Example**:
```python
from viz_art.performance import create_profiler, create_metrics_storage

storage = create_metrics_storage(Path("output/metrics"))
profiler = create_profiler(storage)

@profiler
def my_stage(inputs):
    return process(inputs)
```

---

### 2. `audit_api.py`
Audit logging and query interfaces.

**Key Interfaces**:
- `AuditLogger`: Structured logging with run ID context
- `RunTracker`: Run lifecycle management and metadata
- `AuditQuery`: Fluent query builder for log filtering

**Usage Example**:
```python
from viz_art.audit import create_run_tracker, create_logger, create_query

tracker = create_run_tracker(Path("output/runs"))
with tracker.track(config, inputs, output_dir) as run_id:
    logger = create_logger(run_id, Path("output/logs"))
    logger.info("Processing started", input_count=len(inputs))
    # ... pipeline execution ...

# Query logs
logs = create_query().after(datetime(2025, 10, 20)).stage("detection").fetch()
```

---

### 3. `validation_api.py`
Ground truth validation and error analysis interfaces.

**Key Interfaces**:
- `GroundTruthDataset`: Dataset loading and annotation access
- `MetricsCalculator`: Accuracy metric computation
- `ErrorAnalyzer`: Failure detection and visualization
- `FormatLoader`: Plugin interface for custom annotation formats

**Usage Example**:
```python
from viz_art.validation import (
    create_dataset,
    create_metrics_calculator,
    validate_pipeline,
    AnnotationFormat,
)

dataset = create_dataset(
    dataset_id="coco_val",
    root_path=Path("data/validation"),
    annotation_format=AnnotationFormat.COCO,
    annotation_files=[Path("annotations/instances.json")],
)

results = validate_pipeline(
    run_id="abc-123",
    dataset=dataset,
    stage_outputs={"detection": predictions},
)
print(f"mAP: {results['detection']['mean_average_precision']}")
```

---

## Design Principles

### 1. Protocol-Based Contracts
All interfaces use `typing.Protocol` to define contracts without forcing inheritance. This allows:
- Duck typing compatibility
- Easier testing with mocks
- Flexibility in implementation

### 2. Factory Functions
Public API uses factory functions instead of exposing classes directly:
- `create_profiler()` instead of `Profiler()`
- `create_logger()` instead of `AuditLogger()`
- `create_dataset()` instead of `GroundTruthDataset()`

**Benefits**:
- Hide implementation details
- Allow future optimizations without breaking API
- Simpler imports

### 3. Fluent Interfaces
Query builders use method chaining for readability:
```python
logs = (
    AuditQuery()
    .after(datetime(2025, 10, 20))
    .stage("detection")
    .failed()
    .limit(10)
    .fetch()
)
```

### 4. Context Managers
Lifecycle management uses context managers for automatic cleanup:
```python
with RunTracker().track(config, inputs, output_dir) as run_id:
    # Automatic run status updates
    results = pipeline.run(inputs)
    # Automatic COMPLETED status on success
# Automatic FAILED status on exception
```

---

## Backward Compatibility Guarantees

### Stable (MUST NOT break)
- Function signatures (arguments, return types)
- Protocol method names and signatures
- Enum values
- Exception types raised

### Extensible (MAY add)
- New optional parameters with defaults
- New methods on Protocol interfaces
- New Enum values
- New factory functions

### Unstable (MAY change)
- Implementation details (private methods, classes)
- Performance characteristics
- File formats (with migration path)

---

## Versioning

API contracts follow semantic versioning:
- **MAJOR**: Breaking changes to public APIs
- **MINOR**: Backward-compatible additions
- **PATCH**: Bug fixes without API changes

Current version: `0.1.0` (Phase 3 initial release)

---

## Testing Contracts

All implementations must pass contract tests:

```python
import pytest
from viz_art.performance import create_profiler, create_metrics_storage

def test_profiler_contract():
    """Verify Profiler implementation satisfies Protocol"""
    storage = create_metrics_storage(Path("/tmp/metrics"))
    profiler = create_profiler(storage)

    # Test decorator usage
    @profiler
    def sample_stage(x):
        return x * 2

    result = sample_stage(5)
    assert result == 10

    # Test context manager usage
    with profiler.measure("test_stage"):
        pass

    # Verify metrics were written
    metrics = storage.query_metrics(stage_name="test_stage")
    assert len(metrics) > 0
```

---

## CLI Integration

All contracts include CLI integration functions for command-line usage:

```bash
# Query logs
viz-art logs --after 2025-10-20 --stage detection --failed

# Validate against ground truth
viz-art validate --run-id abc-123 --dataset coco_val --output report.html

# View performance metrics
viz-art metrics --stage detection --start 2025-10-01 --end 2025-10-23
```

CLI functions are defined in each contract file (e.g., `query_logs_cli`, `validate_cli`).

---

## Extension Points

### Custom Annotation Formats
Register new format loaders:
```python
from viz_art.validation import register_format_loader

class MyFormatLoader:
    @property
    def format_name(self):
        return "my_format"

    def load(self, path):
        return parse_my_format(path)

    def validate(self, data):
        return "required_field" in data

register_format_loader(MyFormatLoader())
```

### Custom Metrics
Extend metrics calculator:
```python
from viz_art.validation import create_metrics_calculator

calculator = create_metrics_calculator()
# Custom metric calculation
custom_score = my_custom_metric(predictions, ground_truth)
```

---

## Migration Guide

When upgrading from older versions:

### 0.1.0 → 0.2.0 (hypothetical)
No breaking changes. New features:
- Added `AuditQuery.count()` method
- Added `MetricsStorage.delete_old_metrics(before_date)` method

### 0.2.0 → 1.0.0 (hypothetical - breaking)
Breaking changes:
- `create_profiler(storage)` → `create_profiler(storage, enabled=True)`
  - **Migration**: Add `enabled=True` to existing calls
- `LogLevel.WARNING` renamed to `LogLevel.WARN`
  - **Migration**: Update all references

---

## References

- **Pydantic Models**: See `data-model.md` for entity schemas
- **Implementation Examples**: See `quickstart.md`
- **Research Decisions**: See `research.md` for technical rationale
