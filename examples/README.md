# viz-art Examples

Clean, focused examples demonstrating the viz-art pipeline features.

## 🚀 Quick Start

### 1. Create Sample Data

Generate sample images and point clouds:

```bash
python examples/data/create_sample_data.py
```

Creates:
- `data/sample_000.jpg`, `sample_001.jpg`, `sample_002.jpg` - Sample images (480×640)
- `data/cloud_000.npy`, `cloud_001.npy`, `cloud_002.npy` - Sample point clouds (1000-2000 points)

### 2. Run the Demo

**Option A: Simple Demo** (Quick start)
```bash
uv run python examples/demo_simple.py
```
- Runtime: < 1 second
- Features: Performance profiling, audit logging, basic charts
- Best for: Understanding the basics

**Option B: Comprehensive Demo** ⭐ **RECOMMENDED**
```bash
uv run python examples/demo_comprehensive.py
```
- Runtime: ~1 second
- Features: **ALL Phase 3 capabilities**
  - ✅ Performance profiling (timing + memory charts)
  - ✅ Audit logging with structured logs
  - ✅ Ground truth validation (precision, recall, F1)
  - ✅ Error analysis (false positives, false negatives)
  - ✅ Comprehensive HTML report (25KB with embedded charts)
- Best for: Seeing the complete Phase 3 system

## Output Structure

All demo scripts create outputs in this structure:

```
examples/output/<demo_name>/
├── metrics/                     # Performance metrics (Parquet)
│   ├── image_processing.parquet
│   └── pointcloud_processing.parquet
├── logs/                        # Audit logs (JSONL)
│   └── YYYY-MM-DD.jsonl
└── runs/                        # Run metadata (JSON)
    └── <run-id>/
        └── run_metadata.json
```

## Phase 3 Observability Features

### Performance Profiling

Track execution time and memory usage:

```python
from viz_art.performance import create_profiler, create_metrics_storage

# Setup
metrics_storage = create_metrics_storage(
    output_dir=Path("output/metrics"),
    retention_days=90
)
profiler = create_profiler(storage=metrics_storage, enabled=True)

# Measure performance
with profiler.measure("my_stage"):
    result = expensive_operation()
```

Metrics are stored in Parquet format with:
- Stage name
- Execution time (ms)
- CPU memory usage (MB)
- GPU memory usage (MB) if available
- Timestamp

### Audit Logging

Track all pipeline operations with structured logs:

```python
from viz_art.audit import create_logger, create_run_tracker

# Setup run tracking
tracker = create_run_tracker(Path("output/runs"))

# Track a pipeline run
with tracker.track(config_snapshot, input_files, output_dir) as run_id:
    logger = create_logger(run_id, Path("output/logs"), retention="30 days")

    # Log with structured metadata
    logger.info("Processing started", count=10, mode="batch")
    logger.warning("Low memory", available_mb=512)
    logger.error("Processing failed", error="details", file="input.jpg")
```

Logs are stored in JSONL format with:
- Timestamp
- Log level (INFO, WARNING, ERROR)
- Message
- Structured metadata (as key-value pairs)
- Run ID binding
- File/line information

### CLI Tools

Query logs and metrics from the command line:

```bash
# View logs for a specific run
viz-art-logs --run-id <run-id>

# View failed runs
viz-art-logs --failed

# View logs from date range
viz-art-logs --after 2025-10-20 --before 2025-10-25

# View logs for specific stage
viz-art-logs --stage detection

# View metrics for a stage
viz-art-metrics --stage image_processing

# Export logs to JSON
viz-art-logs --run-id <run-id> --output filtered_logs.json
```

## Available Examples

### `demo_comprehensive.py` ⭐ RECOMMENDED
**Purpose:** Complete Phase 3 observability system
**Time:** ~1 second
**Features:**
- Ground truth validation with accuracy metrics
- Error analysis (FP, FN categorization)
- Performance profiling with charts
- Audit logging with structured logs
- 25KB HTML report with ALL features embedded
**Best for:** Production-ready observability example

### `demo_simple.py`
**Purpose:** Minimal Phase 3 example
**Time:** < 1 second
**Best for:** Quick start and understanding basics

### `test_validation.py`
**Purpose:** Point cloud data quality validation
**Features:** Tests 6 validation scenarios (NaN, Inf, range checks, empty data)
**Best for:** Understanding validation rules

### `test_visualization.py`
**Purpose:** Interactive 3D point cloud visualization
**Features:** Plotly WebGL visualization with color modes
**Best for:** Visualizing point cloud data

## Stage Library

The `stages/` directory contains reusable pipeline stage implementations:

### Point Cloud Stages
- `pointcloud_loader.py` - Load .pcd, .ply, .xyz files
- `pointcloud_validation.py` - Data quality validation
- `pointcloud_visualization.py` - Interactive 3D visualization
- `pointcloud_projection.py` - Project 3D points onto 2D images
- `pointcloud_thumbnail.py` - Generate static thumbnails

### Image Stages
- `image_loader.py` - Load and preprocess images
- `image_resizer.py` - Resize images
- `grayscale_stage.py` - Convert to grayscale
- `simple_filter.py` - Apply image filters

### Multi-Modal Stages
- `multimodal_loader.py` - Load synchronized image + point cloud pairs

**Usage example:**
```python
from examples.stages.image_loader import ImageLoader
from examples.stages.grayscale_stage import GrayscaleStage

# Use stages in your pipeline
loader = ImageLoader(name="loader", resize=(640, 480))
grayscale = GrayscaleStage(name="gray")

result = loader.run({"image_path": "input.jpg"})
result = grayscale.run(result)
```

## Configuration Examples

The `configs/` directory contains YAML pipeline configurations you can customize.

## Creating Custom Pipelines

### 1. Setup observability

```python
from pathlib import Path
from viz_art.performance import create_profiler, create_metrics_storage
from viz_art.audit import create_logger, create_run_tracker

output_dir = Path("output")

# Performance profiling
metrics_storage = create_metrics_storage(output_dir / "metrics", retention_days=90)
profiler = create_profiler(storage=metrics_storage)

# Run tracking
tracker = create_run_tracker(output_dir / "runs")
```

### 2. Track your pipeline run

```python
config_snapshot = {
    "pipeline": "my_pipeline",
    "version": "1.0",
    "param": "value"
}

with tracker.track(config_snapshot, input_files, output_dir) as run_id:
    logger = create_logger(run_id, output_dir / "logs")

    logger.info("Pipeline started")

    # Your pipeline code here
    with profiler.measure("stage_name"):
        result = process_data()

    logger.info("Pipeline completed", items_processed=len(result))
```

### 3. Query results

```bash
# View all logs for this run
viz-art-logs --run-id <run-id>

# View performance metrics
viz-art-metrics --stage stage_name
```

## Performance Targets

Based on Phase 2 specifications:

- **Loading**: 100K points in < 5s (SC-001)
- **Visualization**: 30+ FPS for 500K points (SC-002)
- **Thumbnails**: < 3s per cloud (SC-006)
- **Projection**: < 2px accuracy (SC-004)

## Troubleshooting

### "No sample data found"
Run the data creation script:
```bash
python examples/data/create_sample_data.py
```

### "Module not found" errors
Always use `uv run`:
```bash
uv run python examples/demo_simple.py
```

### Open3D installation issues
```bash
pip install --upgrade pip
pip install open3d --no-cache-dir
```

### Headless rendering (servers without display)
```bash
# Install xvfb (Linux)
apt-get install xvfb

# Run with virtual display
xvfb-run python examples/test_visualization.py
```

## Directory Structure

```
examples/
├── data/                        # Sample data and generation scripts
│   ├── create_sample_data.py
│   ├── sample_*.jpg             # Generated sample images
│   └── cloud_*.npy              # Generated sample point clouds
├── stages/                      # Reusable pipeline stage implementations
│   ├── image_*.py
│   ├── pointcloud_*.py
│   └── multimodal_loader.py
├── configs/                     # YAML pipeline configurations
├── calibration/                 # Camera calibration files
├── output/                      # Generated outputs (gitignored)
│   └── <demo_name>/
│       ├── metrics/
│       ├── logs/
│       └── runs/
├── demo_simple.py               # ⭐ Main demo
├── test_validation.py           # Validation testing
├── test_visualization.py        # Visualization testing
└── README.md                    # This file
```

## Next Steps

1. ✅ Run `demo_simple.py` to see Phase 3 features
2. 📁 Explore outputs in `examples/output/simple_demo/`
3. 🔍 Use CLI tools (`viz-art-logs`, `viz-art-metrics`)
4. 🛠️ Create custom pipelines using stage library
5. 📖 Read detailed specs:
   - [Phase 3: Observability](../specs/003-multimodal-viz-phase3/)
   - [Phase 2: Multi-modal](../specs/002-multimodal-viz/)
   - [Phase 1: Base Architecture](../specs/001-base-pipeline-arch/)
