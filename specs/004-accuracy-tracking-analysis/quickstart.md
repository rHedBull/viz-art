# Quickstart: Accuracy Tracking & Analysis

**Branch**: `004-accuracy-tracking-analysis` | **Date**: 2025-10-25

This guide provides a quick overview of implementing and using the accuracy tracking and analysis system.

---

## Overview

The accuracy tracking system adds three main capabilities to the viz-art pipeline:

1. **Per-Stage Accuracy Metrics**: Calculate precision, recall, F1, mAP, IoU against ground truth
2. **Error Analysis**: Browse, visualize, and debug failure cases with side-by-side comparisons
3. **Performance Trends**: Track accuracy over time and detect regressions

---

## 5-Minute Setup

### 1. Prepare Ground Truth Dataset

Create a directory structure for your labeled validation data:

```bash
mkdir -p /data/validation/
mkdir -p /data/validation/images/
mkdir -p /data/validation/annotations/
```

For COCO format (object detection):
```json
{
  "images": [{"id": 1, "file_name": "sample_0001.jpg", ...}],
  "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [x, y, w, h], ...}],
  "categories": [{"id": 1, "name": "person"}, ...]
}
```

### 2. Configure Pipeline for Validation

Update your pipeline config to enable accuracy tracking:

```yaml
# config/validation.yaml
pipeline:
  name: "detection_pipeline"
  stages:
    - name: "detection"
      type: "object_detection"
    - name: "classification"
      type: "classification"

# NEW: Ground truth configuration
ground_truth:
  dataset_id: "validation_set_2024"
  annotation_path: "/data/validation/annotations/coco_format.json"
  annotation_format: "coco"

# NEW: Accuracy tracking configuration
accuracy_tracking:
  enabled: true
  metrics:
    detection: ["map", "precision", "recall"]
    classification: ["precision", "recall", "f1"]
  thresholds:
    detection_map: 0.75  # Warn if mAP < 75%
    classification_f1: 0.80  # Warn if F1 < 80%

# NEW: Error analysis configuration
error_analysis:
  enabled: true
  save_artifacts: true  # Save error visualizations
  clustering:
    grouping_rules: ["stage_name", "error_type"]
```

### 3. Run Validation

```python
from viz_art import Pipeline
from viz_art.accuracy import AccuracyTracker

# Load pipeline and ground truth
pipeline = Pipeline.from_config("config/validation.yaml")
ground_truth_dataset = pipeline.load_ground_truth()

# Run validation with accuracy tracking
tracker = AccuracyTracker(pipeline, ground_truth_dataset)
results = tracker.run_validation(
    data_dir="/data/validation/images/",
    output_dir="/output/runs/validation_001/"
)

# View results
print(f"Overall accuracy: {results.overall_accuracy:.2%}")
print(f"Errors detected: {len(results.errors)}")
print(f"Report saved to: {results.report_path}")
```

**Output**:
```
Overall accuracy: 87.3%
Errors detected: 127
Report saved to: /output/runs/validation_001/report.html
```

### 4. Browse Errors

Open the generated report in a browser:

```bash
open /output/runs/validation_001/report.html
```

The report shows:
- ✅ Per-stage accuracy counts (correct: 873, wrong: 120, invalid: 7)
- 📊 Detailed metrics (mAP: 78%, precision: 0.85, recall: 0.81)
- 🔗 Link to error browser
- 📈 Link to historical trends

Click "Browse Errors" to see failures grouped by pattern:
- **Detection False Positives** (45 errors) - "Model generating spurious detections"
- **Classification Misclassifications** (38 errors) - "Confusion between chair and stool classes"
- **Detection False Negatives** (32 errors) - "Missing small objects"

### 5. View Historical Trends

```python
from viz_art.performance import TrendAnalyzer

analyzer = TrendAnalyzer(dataset_id="validation_set_2024")
trend = analyzer.get_trend(stage_name="detection", metric_type="map")

# Check for regressions
regression = analyzer.detect_regression(
    current_run="validation_003",
    baseline_run="validation_001"
)

if regression["regression_detected"]:
    print(f"⚠️  Regression detected in {regression['affected_stages']}")
    print(f"   Detection mAP dropped from 0.78 to 0.71 (-9%)")
```

---

## Key Workflows

### Workflow 1: Validate New Model

**Use case**: You trained a new model and want to check if it improved accuracy.

```python
# 1. Run validation with new model
tracker = AccuracyTracker(pipeline, ground_truth_dataset)
results = tracker.run_validation(
    data_dir="/data/validation/",
    run_id="model_v2_validation"
)

# 2. Compare with baseline
comparison = tracker.compare_with_baseline(
    current_run="model_v2_validation",
    baseline_run="model_v1_validation"
)

# 3. View results
for stage, deltas in comparison.items():
    print(f"{stage}: mAP {deltas['map_delta']:+.2%}")
```

**Expected output**:
```
detection: mAP +3.5%  ✅ Improvement
classification: mAP -1.2%  ⚠️ Slight regression
```

### Workflow 2: Debug Failure Patterns

**Use case**: Your pipeline has 15% error rate - find out what's failing.

```python
from viz_art.error_analysis import ErrorBrowser

# 1. Load errors from run
browser = ErrorBrowser(run_id="validation_001")
errors = browser.load_errors(filters={"severity": "high"})

# 2. Get error patterns
patterns = browser.get_patterns()
top_pattern = patterns[0]  # Most common pattern

print(f"Top issue: {top_pattern['pattern_id']}")
print(f"Affects {top_pattern['error_count']} samples")
print(f"Suggested cause: {top_pattern['suggested_cause']}")

# 3. Visualize specific error
error = browser.get_error_by_id(top_pattern['sample_preview'][0])
browser.visualize_error(error, show_diff=True)
```

**Expected output**:
```
Top issue: detection_false_positive
Affects 45 samples
Suggested cause: Model generating spurious detections - check confidence threshold

[Opens visualization showing side-by-side prediction vs ground truth]
```

### Workflow 3: Track Performance Over Time

**Use case**: You've run 10 validation runs over the past month. Check for trends and regressions.

```python
from viz_art.performance import ReportGenerator

# 1. Generate historical report
generator = ReportGenerator()
report_path = generator.generate_historical_report(
    dataset_id="validation_set_2024",
    output_path="/output/trends/historical_report.html"
)

print(f"Historical report: {report_path}")
```

**Report shows**:
- 📈 Accuracy trends over time (line chart for each stage)
- 📊 Confusion matrices across runs
- ⚠️ Regression alerts (if accuracy dropped > 5%)
- 🔍 Per-class performance breakdown

---

## API Quick Reference

### Ground Truth Loading

```python
from viz_art.accuracy.ground_truth import GroundTruthLoader, AnnotationFormat

loader = GroundTruthLoader()
dataset = loader.load_dataset(
    dataset_path=Path("/data/validation/"),
    annotation_format=AnnotationFormat.COCO
)

sample = loader.load_sample(dataset, sample_id="sample_0001")
```

### Accuracy Metrics

```python
from viz_art.accuracy.metrics import MetricsCalculator

calculator = MetricsCalculator()

# Classification metrics
metrics = calculator.calculate_precision_recall_f1(
    predictions=[0, 1, 0, 1],
    ground_truths=[0, 1, 1, 1],
    stage_name="classification"
)

# Object detection mAP
map_metrics = calculator.calculate_map(
    predictions=[{"boxes": [...], "labels": [...], "scores": [...]}],
    ground_truths=[{"boxes": [...], "labels": [...]}],
    stage_name="detection",
    iou_threshold=0.5
)

# Segmentation IoU
iou_metrics = calculator.calculate_iou(
    predictions=[pred_mask],
    ground_truths=[gt_mask],
    stage_name="segmentation"
)

# Point cloud Chamfer distance
chamfer = calculator.calculate_chamfer_distance(
    predictions=[pred_pcd],
    ground_truths=[gt_pcd],
    stage_name="pointcloud_processing"
)
```

### Error Analysis

```python
from viz_art.error_analysis import ErrorBrowser, ErrorVisualizer

# Browse errors
browser = ErrorBrowser(run_id="validation_001")
errors = browser.load_errors(filters={
    "stage_name": "detection",
    "error_type": "false_positive",
    "min_confidence": 0.5
})

# Visualize error
visualizer = ErrorVisualizer()
viz_path = visualizer.create_side_by_side_visualization(
    error_case=errors[0],
    output_path="/output/errors/error_001_viz.jpg"
)

# Create diff visualization
diff_path = visualizer.create_diff_visualization(
    error_case=errors[0],
    output_path="/output/errors/error_001_diff.jpg",
    diff_type="image_diff"  # or "pointcloud_heatmap"
)
```

### Performance Trends

```python
from viz_art.performance import PerformanceTracker, RegressionDetector

# Track metrics
tracker = PerformanceTracker()
tracker.record_metrics(
    run_id="validation_001",
    metrics=accuracy_metrics,
    dataset_id="validation_set_2024",
    pipeline_version="v1.2.0"
)

# Get trend
trend = tracker.get_trend(
    dataset_id="validation_set_2024",
    stage_name="detection",
    metric_type="map"
)

# Detect regression
detector = RegressionDetector()
regression = detector.detect_regression(
    current_metrics=current_run_metrics,
    baseline_metrics=baseline_run_metrics,
    threshold=0.05  # 5% drop threshold
)
```

---

## Configuration Reference

### Ground Truth Configuration

```yaml
ground_truth:
  dataset_id: "validation_set_2024"
  annotation_path: "/data/validation/annotations/coco_format.json"
  annotation_format: "coco"  # Options: coco, pascal_voc, pcd_labels
  validation:
    check_completeness: true
    required_stages: ["detection", "classification"]
```

### Accuracy Tracking Configuration

```yaml
accuracy_tracking:
  enabled: true
  metrics:
    detection: ["map", "precision", "recall", "iou"]
    classification: ["precision", "recall", "f1"]
    segmentation: ["iou"]
    pointcloud_processing: ["chamfer_distance"]
  thresholds:
    detection_map: 0.75
    classification_f1: 0.80
    segmentation_iou: 0.65
  report:
    output_dir: "/output/reports/"
    format: "html"  # Options: html, json, markdown
```

### Error Analysis Configuration

```yaml
error_analysis:
  enabled: true
  save_artifacts: true
  output_dir: "/output/errors/"
  clustering:
    grouping_rules: ["stage_name", "error_type", "severity"]
  visualization:
    side_by_side: true
    diff_mode: "auto"  # Options: auto, image_diff, pointcloud_heatmap
    colormap: "viridis"  # For point cloud heatmaps
  filters:
    min_error_count: 5  # Don't create patterns with < 5 errors
    severity_levels: ["high", "medium"]  # Ignore low severity
```

### Performance Trends Configuration

```yaml
performance_trends:
  enabled: true
  storage:
    format: "parquet"
    retention_days: 90
  regression_detection:
    enabled: true
    threshold: 0.05  # 5% drop
    baseline_strategy: "best"  # Options: best, latest, mean, median
  reports:
    historical_report_path: "/output/trends/historical.html"
    update_frequency: "per_run"  # Options: per_run, daily, weekly
```

---

## File Structure

After running validation, your output directory will look like:

```text
/output/
├── runs/
│   └── validation_001/
│       ├── report.html                    # Per-run accuracy report
│       ├── metrics.parquet                # Accuracy metrics (time-series)
│       ├── errors.json                    # Error case metadata
│       ├── patterns.json                  # Error pattern summaries
│       └── errors/
│           ├── error_001/
│           │   ├── prediction.jpg
│           │   ├── ground_truth.jpg
│           │   └── diff.jpg
│           └── error_002/
│               └── ...
├── trends/
│   └── validation_set_2024_trends.parquet  # Historical performance
└── ground_truth/
    ├── datasets/
    │   └── validation_set_2024.json
    └── samples/
        └── validation_set_2024/
            └── sample_0001.json
```

---

## Performance Guidelines

Based on spec success criteria:

| Operation | Performance Target | Tips |
|-----------|-------------------|------|
| Report loading | < 1 second | Use static HTML with embedded charts |
| Error browsing | < 100ms filtering | Pre-compute clusters, indexed lookups |
| Image visualization | < 2 seconds | Lazy load thumbnails, full res on demand |
| Point cloud visualization | < 5 seconds | Downsample for preview, full on click |
| Batch report generation | < 30 seconds for 10k samples | Parallel processing, incremental updates |

**Optimization tips**:
- Use `voxel_down_sample()` for point cloud previews
- Pre-compute error clusters during validation (not on browser load)
- Store Parquet metrics with indexed columns (run_id, stage_name, timestamp)
- Generate thumbnails asynchronously after validation completes

---

## Testing Your Implementation

### Unit Tests

```python
# tests/accuracy/test_metrics.py
def test_map_calculation():
    calculator = MetricsCalculator()
    predictions = [{"boxes": [[100, 100, 200, 200]], "labels": [1], "scores": [0.9]}]
    ground_truths = [{"boxes": [[105, 105, 195, 195]], "labels": [1]}]

    metrics = calculator.calculate_map(predictions, ground_truths, "detection")
    assert 0 <= metrics.value <= 100
    assert metrics.counts.correct + metrics.counts.wrong > 0

# tests/error_analysis/test_clustering.py
def test_error_clustering():
    detector = ErrorPatternDetector()
    errors = [
        ErrorCase(stage_name="detection", error_type="false_positive", ...),
        ErrorCase(stage_name="detection", error_type="false_positive", ...),
        ErrorCase(stage_name="detection", error_type="false_negative", ...),
    ]

    patterns = detector.cluster_errors(errors)
    assert "detection_false_positive" in patterns
    assert patterns["detection_false_positive"].error_count == 2
```

### Integration Tests

```python
# tests/integration/test_accuracy_workflow.py
def test_end_to_end_validation():
    # 1. Setup ground truth
    dataset = create_test_ground_truth_dataset()

    # 2. Run pipeline
    pipeline = Pipeline.from_config("test_config.yaml")
    tracker = AccuracyTracker(pipeline, dataset)
    results = tracker.run_validation(data_dir="tests/fixtures/validation/")

    # 3. Verify results
    assert results.overall_accuracy > 0
    assert len(results.errors) >= 0
    assert results.report_path.exists()

    # 4. Check error browser
    browser = ErrorBrowser(run_id=results.run_id)
    errors = browser.load_errors()
    assert len(errors) == len(results.errors)
```

---

## Next Steps

1. **Read the full spec**: See `spec.md` for detailed requirements
2. **Review data model**: See `data-model.md` for entity definitions
3. **Check API contracts**: See `contracts/` for interface specifications
4. **Implement Phase 4.1**: Start with metrics implementation (see `tasks.md` when generated)

---

## Troubleshooting

**Problem**: Ground truth loader fails with "Invalid annotation format"

**Solution**: Verify your annotation file matches the declared format:
```python
from viz_art.accuracy.ground_truth import GroundTruthValidator

validator = GroundTruthValidator()
errors = validator.validate_dataset(dataset)
for error in errors:
    print(f"❌ {error}")
```

**Problem**: Error browser is slow with 1000+ errors

**Solution**: Enable indexed filtering and pre-compute clusters:
```yaml
error_analysis:
  clustering:
    precompute: true  # Cluster during validation, not on browser load
  browser:
    lazy_loading: true
    pagination: 50  # Load 50 errors per page
```

**Problem**: Point cloud diff visualization is very slow

**Solution**: Use downsampling for interactive preview:
```python
visualizer = ErrorVisualizer()
diff_pcd = visualizer.create_pointcloud_diff(
    pred_pcd.voxel_down_sample(voxel_size=0.01),  # Downsample
    gt_pcd.voxel_down_sample(voxel_size=0.01),
    use_icp=False  # Skip ICP for speed
)
```

---

## References

- **Feature Spec**: `spec.md`
- **Data Model**: `data-model.md`
- **API Contracts**: `contracts/*.py`
- **Research Decisions**: `research.md`
- **Implementation Plan**: `plan.md`
