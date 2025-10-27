# Accuracy Tracking Module

Complete accuracy tracking system for pipeline validation with per-stage metrics, error analysis, and performance trends.

## Features

### 1. Per-Stage Accuracy Metrics
- **Classification**: Precision, recall, F1 score, confusion matrix
- **Object Detection**: mAP, per-class AP, IoU
- **Segmentation**: Mean IoU, per-class IoU, pixel accuracy
- **Point Clouds**: Chamfer distance, alignment metrics

### 2. Error Analysis
- Automatic error detection and categorization
- Error severity classification (critical/high/medium/low)
- Rule-based error clustering by pattern
- Side-by-side visualizations with diff views
- Interactive Streamlit browser with filtering
- Export to JSON/CSV/Parquet

### 3. Performance Trends
- Historical accuracy tracking over time
- Regression detection with configurable thresholds
- Baseline selection strategies (best/latest/mean/median)
- Time-series trend charts
- Automated alerts for regressions

## Quick Start

### Basic Usage

```python
from pathlib import Path
from viz_art.accuracy import AccuracyTracker, GroundTruthDataset, AnnotationFormat

# 1. Load ground truth dataset
dataset = GroundTruthDataset(
    dataset_id="validation_set_2024",
    name="Validation Set",
    description="Labeled validation data",
    base_path=Path("/data/validation/images"),
    annotation_path=Path("/data/validation/annotations/coco.json"),
    annotation_format=AnnotationFormat.COCO,
    num_samples=100,
    sample_ids=[f"sample_{i:04d}" for i in range(100)]
)

# 2. Create accuracy tracker
tracker = AccuracyTracker(dataset)

# 3. Run validation
predictions = {
    "detection": [
        {"boxes": [[100, 100, 50, 75]], "labels": ["person"], "scores": [0.95]},
        # ... more predictions
    ],
    "classification": [0, 1, 2, ...]  # Predicted labels
}

results = tracker.run_validation(
    predictions=predictions,
    run_id="validation_001",
    output_dir=Path("output/validation"),
    stage_task_types={
        "detection": "detection",
        "classification": "classification"
    }
)

# 4. View results
print(f"Overall accuracy: {results['overall_accuracy']:.2%}")
print(f"Correct: {results['correct']}, Wrong: {results['wrong']}")
print(f"Report: {results['report_path']}")
print(f"Errors detected: {len(results['errors'])}")
```

### Browse Errors

```python
from viz_art.error_analysis.browser import ErrorBrowser

browser = ErrorBrowser(run_id="validation_001", error_dir=Path("output/errors"))

# Filter errors
errors = browser.load_errors(filters={
    "stage_name": "detection",
    "error_type": "false_positive",
    "severity": "high"
})

print(f"Found {len(errors)} high-severity false positives")

# Visualize specific error
error = browser.get_error_by_id(errors[0].error_id)
vis_path = browser.visualize_error(error, show_diff=True)
print(f"Visualization saved to: {vis_path}")
```

### Track Performance Trends

```python
from viz_art.performance.trends import PerformanceTracker, RegressionDetector

tracker = PerformanceTracker(Path("output/trends"))

# Get historical trend
trend = tracker.get_trend(
    dataset_id="validation_set_2024",
    stage_name="detection",
    metric_type="mean_average_precision"
)

print(f"Trend data: {len(trend)} runs")

# Detect regression
detector = RegressionDetector(threshold=0.05)  # 5% threshold

regression = detector.detect_regression(
    current_metrics=current_run_metrics,
    baseline_metrics=baseline_run_metrics
)

if regression['regression_detected']:
    print(f"⚠️ Regression detected in: {regression['affected_stages']}")

    # Create alert
    alert = detector.create_alert(
        regression,
        run_id="validation_003",
        baseline_run_id="validation_001"
    )
    print(alert['message'])
```

## Architecture

```
AccuracyTracker (orchestrator)
├── GroundTruthLoader
│   ├── Load COCO annotations
│   ├── Load PASCAL VOC annotations
│   └── Load point cloud labels
├── MetricsCalculator
│   ├── calculate_precision_recall_f1()
│   ├── calculate_mean_average_precision()
│   ├── calculate_iou()
│   └── calculate_chamfer_distance()
├── ComparisonEngine
│   ├── compare_single() - Detection, classification, segmentation, pointcloud
│   ├── compare_batch()
│   └── aggregate_results()
├── ErrorDetector
│   ├── categorize_error() - False positive/negative, misclassification, low IoU
│   └── detect_errors()
├── ErrorPatternDetector
│   ├── cluster_errors() - Rule-based clustering
│   └── summarize_patterns()
├── ReportGenerator
│   ├── generate_per_run_report()
│   └── generate_historical_report()
└── PerformanceTracker
    ├── record_metrics()
    ├── get_trend()
    └── compare_runs()
```

## File Structure

```
output/
├── validation/
│   └── validation_001/
│       ├── validation_001_accuracy_report.html  # Per-run report
│       ├── validation_001_metrics.parquet       # Metrics storage
│       └── errors/
│           ├── validation_001_errors.json       # Error metadata
│           ├── validation_001_patterns.json     # Error patterns
│           └── validation_001/
│               ├── error_001/
│               │   ├── prediction.json
│               │   ├── ground_truth.json
│               │   └── diff.jpg
│               └── error_002/
│                   └── ...
├── trends/
│   └── trends_validation_set_2024.parquet      # Historical trends
└── historical_report.html                       # Historical trends report
```

## Configuration

### Ground Truth Configuration

```yaml
ground_truth:
  dataset_id: "validation_set_2024"
  annotation_path: "/data/validation/annotations/coco_format.json"
  annotation_format: "coco"  # Options: coco, pascal_voc, pcd_labels
```

### Accuracy Tracking Configuration

```yaml
accuracy_tracking:
  enabled: true
  metrics:
    detection: ["map", "precision", "recall"]
    classification: ["precision", "recall", "f1"]
    segmentation: ["iou"]
    pointcloud_processing: ["chamfer_distance"]
  thresholds:
    detection_map: 0.75
    classification_f1: 0.80
    segmentation_iou: 0.65
```

### Error Analysis Configuration

```yaml
error_analysis:
  enabled: true
  save_artifacts: true
  clustering:
    grouping_rules: ["stage_name", "error_type"]
  visualization:
    side_by_side: true
    diff_mode: "auto"  # Options: auto, image_diff, pointcloud_heatmap
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
```

## API Reference

### Core Classes

#### AccuracyTracker
```python
tracker = AccuracyTracker(ground_truth_dataset)
results = tracker.run_validation(predictions, run_id, output_dir)
```

#### GroundTruthLoader
```python
loader = GroundTruthLoader()
dataset = loader.load_dataset(dataset_path, AnnotationFormat.COCO)
sample = loader.load_sample(dataset, sample_id)
```

#### MetricsCalculator
```python
calculator = MetricsCalculator()
metrics = calculator.calculate_precision_recall_f1(predictions, ground_truths, stage_name, run_id, dataset_id)
```

#### ErrorBrowser
```python
browser = ErrorBrowser(run_id, error_dir)
errors = browser.load_errors(filters={"stage_name": "detection"})
error = browser.get_error_by_id(error_id)
```

#### PerformanceTracker
```python
tracker = PerformanceTracker(storage_dir)
tracker.record_metrics(run_id, metrics, dataset_id)
trend = tracker.get_trend(dataset_id, stage_name, metric_type)
```

#### RegressionDetector
```python
detector = RegressionDetector(threshold=0.05)
regression = detector.detect_regression(current_metrics, baseline_metrics)
alert = detector.create_alert(regression, run_id, baseline_run_id)
```

## Performance

| Operation | Target | Implementation |
|-----------|--------|---------------|
| Report loading | < 1 second | Static HTML with embedded charts |
| Error browsing | < 100ms | Indexed filtering with Parquet |
| Error clustering | < 15ms for 1000 errors | Rule-based O(n) algorithm |
| Image visualization | < 2 seconds | OpenCV with lazy loading |
| Point cloud viz | < 5 seconds | Open3D with downsampling |

## Testing

```bash
# Unit tests
pytest tests/error_analysis/test_patterns.py -v
pytest tests/error_analysis/test_visualizer.py -v
pytest tests/performance/test_regression.py -v

# Integration tests
pytest tests/integration/test_accuracy_workflow.py -v
pytest tests/integration/test_trends.py -v

# All tests
pytest tests/ -v
```

## Troubleshooting

### Issue: Ground truth loader fails
**Solution**: Verify annotation format matches declared format:
```python
from viz_art.accuracy.ground_truth import GroundTruthValidator

validator = GroundTruthValidator()
errors = validator.validate_dataset(dataset)
```

### Issue: Error browser is slow
**Solution**: Enable indexed filtering and pre-compute clusters:
```yaml
error_analysis:
  clustering:
    precompute: true
  browser:
    lazy_loading: true
    pagination: 50
```

### Issue: Point cloud diff is slow
**Solution**: Use downsampling:
```python
visualizer.create_pointcloud_diff(
    pred_pcd.voxel_down_sample(voxel_size=0.01),
    gt_pcd.voxel_down_sample(voxel_size=0.01),
    use_icp=False
)
```

## Dependencies

- Python 3.8+
- scikit-learn (metrics calculation)
- Open3D >= 0.18 (point cloud operations)
- Plotly (interactive charts)
- Jinja2 (HTML templating)
- PyArrow (Parquet storage)
- pandas (data manipulation)
- opencv-python (image processing)
- streamlit (optional, for UI)

## License

See project LICENSE file.
