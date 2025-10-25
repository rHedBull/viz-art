# viz-art Development Guidelines

Auto-generated from all feature plans. Last updated: 2025-10-22

## Active Technologies
- Python 3.8+ (matches existing codebase) (002-multimodal-viz)
- File-based (.pcd, .ply, .xyz for point clouds; YAML for calibration) (002-multimodal-viz)
- Python 3.8+ (existing project requirement) + Loguru (structured logging), PyArrow (metrics storage), psutil (memory tracking), torchmetrics or scikit-learn (accuracy metrics) (003-multimodal-viz-phase3)
- File-based: JSON for audit logs, Parquet for performance metrics, existing OutputSaver for stage artifacts (003-multimodal-viz-phase3)
- Python 3.8+ (existing project requirement) + torchmetrics or scikit-learn (metrics), Open3D (point cloud diff), Plotly (visualizations), Jinja2 (HTML reports), existing Streamlit UI (004-accuracy-tracking-analysis)
- File-based - Parquet for metrics storage (existing PyArrow infrastructure), JSON for error metadata, saved artifacts via existing OutputSaver (004-accuracy-tracking-analysis)

- Python 3.8+ + pydantic (data validation), omegaconf (configuration), pytest (testing), PIL/opencv-python (image processing), jinja2 (HTML templating) (001-base-pipeline-arch)

## Project Structure

```text
src/
tests/
```

## Commands

cd src [ONLY COMMANDS FOR ACTIVE TECHNOLOGIES][ONLY COMMANDS FOR ACTIVE TECHNOLOGIES] pytest [ONLY COMMANDS FOR ACTIVE TECHNOLOGIES][ONLY COMMANDS FOR ACTIVE TECHNOLOGIES] ruff check .

## Code Style

Python 3.8+: Follow standard conventions

## Recent Changes
- 004-accuracy-tracking-analysis: Added Python 3.8+ (existing project requirement) + torchmetrics or scikit-learn (metrics), Open3D (point cloud diff), Plotly (visualizations), Jinja2 (HTML reports), existing Streamlit UI
- 003-multimodal-viz-phase3: Added Python 3.8+ (existing project requirement) + Loguru (structured logging), PyArrow (metrics storage), psutil (memory tracking), torchmetrics or scikit-learn (accuracy metrics)
- 002-multimodal-viz: Added Python 3.8+ (matches existing codebase)


<!-- MANUAL ADDITIONS START -->

## Accuracy Tracking & Analysis System

### Overview
Complete accuracy tracking system with per-stage metrics, error analysis, and performance trends.

### Key Modules
- **src/viz_art/accuracy/** - Core accuracy tracking (metrics, comparison, reporting)
- **src/viz_art/error_analysis/** - Error detection, clustering, visualization, browser
- **src/viz_art/performance/trends.py** - Historical tracking and regression detection

### Usage Example
```python
from viz_art.accuracy import AccuracyTracker, GroundTruthDataset

# Load ground truth
dataset = GroundTruthDataset(...)

# Run validation
tracker = AccuracyTracker(dataset)
results = tracker.run_validation(
    predictions=stage_predictions,
    run_id="validation_001",
    output_dir=Path("output/validation")
)

# View results
print(f"Overall accuracy: {results['overall_accuracy']:.2%}")
print(f"Report: {results['report_path']}")
print(f"Errors detected: {len(results['errors'])}")
```

### Features
- ✅ Multi-task accuracy metrics (classification, detection, segmentation, point clouds)
- ✅ Automatic error detection and severity classification
- ✅ Rule-based error clustering (< 15ms for 1000 errors)
- ✅ Interactive Streamlit error browser with filtering
- ✅ Side-by-side visualizations (images, point clouds, masks)
- ✅ Historical trend tracking with regression detection
- ✅ Beautiful HTML reports with Plotly charts
- ✅ Export to JSON/CSV/Parquet

### Testing
```bash
# Run accuracy tracking tests
pytest tests/integration/test_accuracy_workflow.py -v

# Run error analysis tests
pytest tests/error_analysis/ -v

# Run trend tracking tests
pytest tests/performance/test_regression.py -v
pytest tests/integration/test_trends.py -v
```

### Performance
- Report loading: < 1 second
- Error browsing: < 100ms filtering
- Error clustering: < 15ms for 1000 errors
- Image visualization: < 2 seconds
- Point cloud visualization: < 5 seconds

<!-- MANUAL ADDITIONS END -->
