# Quickstart Guide: Performance Monitoring & Debugging

**Feature**: Performance Monitoring & Debugging System
**Branch**: `003-multimodal-viz-phase3`
**Prerequisites**: Phase 2 multi-modal visualization complete

## Installation

Add Phase 3 dependencies to your environment:

```bash
pip install loguru pyarrow psutil pynvml scikit-learn
# Optional for advanced metrics:
pip install torchmetrics
```

## 5-Minute Quickstart

### 1. Basic Performance Profiling

Add automatic timing and memory tracking to your pipeline:

```python
from pathlib import Path
from viz_art.pipeline import Pipeline
from viz_art.performance import create_profiler, create_metrics_storage

# Setup
storage = create_metrics_storage(Path("output/metrics"))
profiler = create_profiler(storage, enabled=True)

# Decorate your stage
class DetectionStage:
    @profiler  # <-- Add this decorator
    def execute(self, inputs):
        # Your existing stage logic
        return detect_objects(inputs)

# Run pipeline
pipeline = Pipeline(stages=[DetectionStage()])
results = pipeline.run(image=my_image)

# View metrics
metrics = storage.query_metrics(stage_name="DetectionStage")
print(f"Execution time: {metrics[0]['execution_time_ms']}ms")
print(f"Memory used: {metrics[0]['cpu_memory_mb']}MB")
```

**Result**: Automatic performance tracking with zero changes to stage logic.

---

### 2. Audit Logging

Track pipeline execution with structured logs:

```python
from viz_art.audit import create_run_tracker, create_logger
from datetime import datetime

# Setup run tracking
tracker = create_run_tracker(Path("output/runs"))

# Execute with logging
with tracker.track(config={...}, input_files=[Path("data/img.jpg")], output_dir=Path("output")) as run_id:
    logger = create_logger(run_id, Path("output/logs"))

    logger.info("Pipeline started", input_count=1)
    try:
        results = pipeline.run(image=my_image)
        logger.info("Pipeline completed", output_count=len(results))
    except Exception as e:
        logger.error("Pipeline failed", error=str(e))
        raise

# Query logs
from viz_art.audit import create_query

failed_runs = (
    create_query()
    .after(datetime(2025, 10, 20))
    .failed()
    .fetch()
)
print(f"Found {len(failed_runs)} failures")
```

**Result**: Complete audit trail with searchable logs by run ID, date, stage, and status.

---

### 3. Ground Truth Validation

Validate pipeline accuracy against labeled data:

```python
from viz_art.validation import (
    create_dataset,
    validate_pipeline,
    AnnotationFormat,
)

# Load ground truth dataset
dataset = create_dataset(
    dataset_id="coco_val_2017",
    root_path=Path("data/validation"),
    annotation_format=AnnotationFormat.COCO,
    annotation_files=[Path("annotations/instances_val2017.json")],
)

# Run validation
stage_outputs = {
    "detection": [
        {"bbox": [10, 20, 100, 200], "class": "car", "score": 0.95},
        # ... more predictions
    ]
}

results = validate_pipeline(
    run_id=run_id,
    dataset=dataset,
    stage_outputs=stage_outputs,
)

# View accuracy
print(f"Precision: {results['detection']['precision']:.2%}")
print(f"Recall: {results['detection']['recall']:.2%}")
print(f"mAP: {results['detection']['mean_average_precision']:.2%}")
```

**Result**: Automatic accuracy metrics (precision, recall, F1, mAP) against ground truth.

---

## Common Use Cases

### Use Case 1: Identify Performance Bottlenecks

**Scenario**: Your pipeline is slow, but you don't know which stage is the problem.

**Solution**:
```python
from viz_art.performance import create_dashboard

# Generate performance dashboard
dashboard = create_dashboard(storage)
html = dashboard.render_timing_chart(run_id)

# Save report
with open("performance_report.html", "w") as f:
    f.write(html)

# Opens in browser showing per-stage timing
```

**Outcome**: Visual breakdown showing which stage(s) take the most time.

---

### Use Case 2: Debug Failed Runs

**Scenario**: Pipeline failed on sample #47, need to understand why.

**Solution**:
```python
# Query logs for specific run
logs = (
    create_query()
    .run_id("abc-123-def-456")  # Run that failed
    .stage("detection")
    .level(LogLevel.ERROR)
    .fetch()
)

# Examine error
for log in logs:
    print(f"Error at {log['timestamp']}: {log['message']}")
    print(f"Metadata: {log['metadata']}")
    # Metadata includes: input_paths, error_type, traceback

# Load saved outputs for inspection
from viz_art.batch import load_output
failed_output = load_output(f"output/runs/{run_id}/stages/detection/sample_047.jpg")
```

**Outcome**: Exact error message, stack trace, and saved artifacts for the failed sample.

---

### Use Case 3: Track Performance Over Time

**Scenario**: Want to ensure recent changes didn't slow down the pipeline.

**Solution**:
```python
# Query historical metrics
from datetime import datetime, timedelta

last_week = datetime.utcnow() - timedelta(days=7)
stats = storage.get_aggregate_stats(
    stage_name="detection",
    start_date=last_week,
)

print(f"Median time: {stats['median']}ms")
print(f"95th percentile: {stats['p95']}ms")

# Generate trend chart
dashboard.render_trend_chart(
    stage_name="detection",
    start_date=last_week,
    end_date=datetime.utcnow(),
    output_path=Path("trends.html"),
)
```

**Outcome**: Historical performance trends showing if recent changes caused regressions.

---

### Use Case 4: Error Analysis

**Scenario**: 10% of samples are failing detection, need to understand patterns.

**Solution**:
```python
from viz_art.validation import create_error_analyzer

analyzer = create_error_analyzer(Path("output/error_analysis"))

# Find all failures
failures = analyzer.find_failures(
    run_id=run_id,
    dataset=dataset,
    threshold=0.5,  # IoU threshold
)

# Categorize errors
categories = analyzer.categorize_errors(failures)
print(f"False positives: {len(categories['false_positive'])}")
print(f"False negatives: {len(categories['false_negative'])}")
print(f"Misclassifications: {len(categories['misclassification'])}")

# Visualize specific failure
analyzer.visualize_comparison(
    sample_id="image_047.jpg",
    prediction=failures[0]["prediction"],
    ground_truth=failures[0]["ground_truth"],
    output_path=Path("comparison_047.png"),
)
```

**Outcome**: Categorized errors with side-by-side visualizations for debugging.

---

## Command-Line Interface

Phase 3 adds CLI commands for common tasks:

### Query Logs
```bash
# All logs from yesterday
viz-art logs --after 2025-10-22

# Failed runs in detection stage
viz-art logs --stage detection --failed --limit 10

# Export to file
viz-art logs --after 2025-10-20 --output filtered.json
```

### View Metrics
```bash
# Performance stats for a stage
viz-art metrics --stage detection --start 2025-10-01

# Generate dashboard
viz-art dashboard --run-id abc-123 --output report.html
```

### Validate Against Ground Truth
```bash
# Run validation
viz-art validate --run-id abc-123 --dataset coco_val

# Generate error analysis report
viz-art validate --run-id abc-123 --dataset coco_val --output errors.html
```

---

## Integration Patterns

### Pattern 1: Full Observability Stack

Combine all Phase 3 features for complete observability:

```python
from viz_art.pipeline import Pipeline
from viz_art.performance import create_profiler, create_metrics_storage, create_dashboard
from viz_art.audit import create_run_tracker, create_logger
from viz_art.validation import create_dataset, validate_pipeline

# Setup
output_root = Path("output")
metrics_storage = create_metrics_storage(output_root / "metrics")
profiler = create_profiler(metrics_storage)
tracker = create_run_tracker(output_root / "runs")
dataset = create_dataset(...)  # Ground truth

# Decorate stages
class MyStage:
    @profiler
    def execute(self, inputs):
        return process(inputs)

# Execute with full tracking
pipeline = Pipeline(stages=[MyStage()])

with tracker.track(config, inputs, output_root) as run_id:
    logger = create_logger(run_id, output_root / "logs")
    logger.info("Starting pipeline")

    # Run pipeline
    results = pipeline.run(image=my_image)

    # Validate accuracy
    validation_results = validate_pipeline(
        run_id=run_id,
        dataset=dataset,
        stage_outputs={"MyStage": results},
    )

    logger.info("Completed",
                execution_time=results.metadata["execution_time"],
                accuracy=validation_results["MyStage"]["f1_score"])

# Generate comprehensive report
dashboard = create_dashboard(metrics_storage)
html = dashboard.render_timing_chart(run_id)
# Includes: timing, memory, accuracy, logs
```

---

### Pattern 2: Opt-In Monitoring

Make profiling optional for production:

```python
from omegaconf import DictConfig

def create_pipeline(config: DictConfig):
    stages = [PreprocessStage(), DetectionStage()]

    # Only enable profiling if configured
    if config.get("enable_profiling", False):
        storage = create_metrics_storage(Path(config.output_dir) / "metrics")
        profiler = create_profiler(storage, enabled=True)

        # Wrap stages
        for stage in stages:
            stage.execute = profiler(stage.execute)

    return Pipeline(stages=stages)

# In config.yaml:
# enable_profiling: true  # Development
# enable_profiling: false  # Production
```

---

### Pattern 3: Custom Annotation Format

Support your custom ground truth format:

```python
from viz_art.validation import register_format_loader, FormatLoader
from pathlib import Path
import json

class MyCustomLoader:
    @property
    def format_name(self):
        return "my_custom_format"

    def load(self, annotation_file: Path):
        with open(annotation_file) as f:
            data = json.load(f)

        # Transform to standard format
        annotations = {}
        for item in data["labels"]:
            annotations[item["image_id"]] = {
                "bboxes": item["boxes"],
                "classes": item["categories"],
            }
        return annotations

    def validate(self, annotations: dict):
        # Basic validation
        return all("bboxes" in v for v in annotations.values())

# Register loader
register_format_loader(MyCustomLoader())

# Use with custom format
dataset = create_dataset(
    dataset_id="my_dataset",
    root_path=Path("data"),
    annotation_format="my_custom_format",  # String, not enum
    annotation_files=[Path("labels.json")],
)
```

---

## Performance Tips

### 1. Minimize Overhead
```python
# Only profile specific stages
@profiler  # Add only where needed
def expensive_stage(inputs):
    pass

def cheap_stage(inputs):  # No profiler
    pass
```

### 2. Batch Queries
```python
# Efficient: Single query with filters
metrics = storage.query_metrics(
    stage_name="detection",
    start_date=start,
    end_date=end,
)

# Inefficient: Multiple queries
# Don't do this:
for date in date_range:
    metrics = storage.query_metrics(start_date=date, end_date=date)
```

### 3. Configure Retention
```python
# Limit log retention to save disk space
logger = create_logger(
    run_id,
    output_dir,
    retention="7 days",  # Auto-delete after 7 days
)

# Limit metrics retention
storage = create_metrics_storage(
    output_dir,
    retention_days=90,  # Keep 90 days
)
```

---

## Troubleshooting

### Problem: "No metrics found"
**Cause**: Profiler not decorating stages
**Solution**: Ensure `@profiler` is on `execute()` method

### Problem: "GPU memory is always None"
**Cause**: No NVIDIA GPU or pynvml not installed
**Solution**: GPU metrics are optional, gracefully degraded to CPU-only

### Problem: "Parquet files too large"
**Cause**: High-frequency metrics collection
**Solution**: Configure retention or reduce profiling scope

### Problem: "Query is slow"
**Cause**: Scanning all Parquet files
**Solution**: Use date filters to limit scan range

---

## Next Steps

- **Implementation**: See `tasks.md` for development tasks (generated by `/speckit.tasks`)
- **API Reference**: See `contracts/` for detailed API documentation
- **Data Models**: See `data-model.md` for entity schemas
- **Research**: See `research.md` for technical decisions and alternatives

## Examples Repository

Full working examples available in `examples/phase3/`:
- `example_profiling.py`: Basic performance tracking
- `example_logging.py`: Audit logging and queries
- `example_validation.py`: Ground truth validation
- `example_full_stack.py`: All features combined
