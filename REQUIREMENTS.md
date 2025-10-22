# Vision Pipeline MVP Library - Unified Requirements Document

## Overview
A lightweight library for building multi-modal vision pipelines with integrated UI, performance tracking, and debugging capabilities. Designed for rapid prototyping while maintaining production-ready architecture.

## Core Requirements

### 1. Pipeline System Architecture
- **Pipeline class** that orchestrates execution of connected stages in sequence
- **Stage class** with standard interface: pre_process → predict → post_process methods
- **Connection system** to map outputs from one stage to inputs of another
- **Support for multiple data types**: images and point clouds as first-class citizens
- **Simple execution**: pipeline.run(image=..., pointcloud=...) returns all stage outputs

### 2. Data Validation & Quality Control
- **Input validators** that check for corrupted images, empty point clouds, reasonable value ranges
- **Output validators** that catch obvious errors (e.g., 500 detections in one frame)
- **Configurable thresholds** for what constitutes valid data
- **Graceful handling** of validation failures with clear error messages

### 3. Calibration & Coordinate Systems
- **Calibration storage** for camera intrinsics and sensor extrinsics
- **Built-in transformations** between coordinate systems (image ↔ world ↔ LiDAR)
- **Simple calibration loader** from standard formats (YAML, JSON)
- **Projection utilities** for overlaying 3D points on 2D images

### 4. Performance Tracking
- **Automatic timing** of each stage and sub-stage
- **Memory usage tracking** (CPU and GPU if available)
- **Simple metrics storage** without external dependencies
- **Per-stage performance visibility** in the UI

### 5. Ground Truth & Accuracy Tracking
- **Golden dataset support** with labels for each stage and final output
- **Label format flexibility** to handle different annotation types (bboxes, segmentation, 3D points)
- **Real-time accuracy display**: correct/wrong/invalid/unlabeled counts
- **Per-stage metrics**: precision, recall, F1, mAP, IoU as appropriate
- **Error analysis tools** to browse and filter failure cases
- **Performance regression testing** with configurable thresholds

### 6. Configuration Management
- **Single config file** (YAML/JSON) defining all pipeline parameters
- **Per-stage configuration** sections
- **Environment overrides** (dev/test/prod settings)
- **Runtime parameter updates** through UI without code changes

### 7. Model & Pipeline Versioning
- **Version tags** for both individual models and complete pipelines
- **Simple registry** mapping version names to file paths
- **Track which version** processed which data
- **Performance comparison** between versions

### 8. Data Lineage & Auditing
- **Run ID** for each pipeline execution
- **Basic audit log**: timestamp, stage name, input/output paths, success/failure
- **Performance metrics** included in audit trail
- **Simple file-based storage** (JSON logs) for debugging
- **Query by run ID** to trace what happened to specific data

### 9. Integrated UI
- **Auto-generated interface** based on pipeline definition
- **File upload widgets** for image and point cloud inputs
- **Stage-by-stage visualization** showing inputs, outputs, timing, and accuracy
- **Live progress tracking** with performance metrics
- **Results viewer** with appropriate visualizations and ground truth comparison
- **Config editor** to modify parameters without restarting
- **Run history browser** with performance trends
- **Error case browser** for debugging failures

### 10. Design Principles
- **Convention over configuration**: sensible defaults for everything
- **Single-file stages**: each stage definable in one Python file
- **Minimal dependencies**: core library uses only standard packages
- **Progressive enhancement**: basic features work immediately
- **Clear error messages**: user knows exactly what failed and where

---

## Implementation Strategy: Build vs Use External Libraries

### Components to Build Yourself

1. **Core Pipeline Framework**
   - The Pipeline and Stage base classes
   - Connection/data flow management
   - Simple, tailored to your exact needs

2. **Integration Layer**
   - UI auto-generation logic
   - Unified API across all components
   - Custom validation rules

### Components to Use External Libraries For

#### UI Framework
**Use: Streamlit or Gradio**
- Handles all web server complexity
- Provides ready-made widgets
- Free hosting options
- **Specifically**: File uploaders, progress bars, metrics display, dataframe viewing

#### Data Validation
**Use: Pydantic**
```python
from pydantic import BaseModel, validator
# Automatic validation, serialization, and clear error messages
```

#### Configuration Management
**Use: Hydra or OmegaConf**
- Hierarchical configs
- Environment overrides
- Runtime parameter updates
- Config validation

#### Performance Metrics
**Use: torchmetrics or scikit-learn**
- Pre-built metric calculations
- Handles edge cases correctly
- Optimized implementations

#### Visualization
**Use: Multiple specialized libraries**
- **Plotly**: Interactive plots and 3D point cloud visualization
- **OpenCV**: Image annotation and overlay
- **Open3D**: Point cloud processing and visualization
- **Seaborn**: Performance trend charts

#### Logging & Monitoring
**Use: Loguru + Rich**
- **Loguru**: Better logging with automatic rotation
- **Rich**: Beautiful console output and progress bars

#### Model Registry
**Use: MLflow (lightweight mode)**
- Just for model versioning and metric tracking
- Can run without server infrastructure
- Provides comparison UI out-of-the-box

#### Data Storage
**Use: Apache Arrow/Parquet**
- Efficient storage for audit logs
- Fast querying without database
- Good compression for sensor data

#### Testing
**Use: pytest + pytest-benchmark**
- Handles test discovery and fixtures
- Built-in performance regression testing

### Minimal External Dependencies Setup

```python
# requirements.txt for MVP
streamlit>=1.28.0          # UI framework
pydantic>=2.0             # Data validation
omegaconf>=2.3            # Configuration
plotly>=5.17              # Visualization
opencv-python>=4.8        # Image processing
open3d>=0.17              # Point clouds
loguru>=0.7               # Logging
scikit-learn>=1.3         # Metrics
pyarrow>=13.0            # Efficient storage
pytest>=7.4              # Testing

# Optional but recommended
mlflow>=2.8              # Experiment tracking
rich>=13.6               # Better console output
pillow>=10.0             # Image I/O
numpy>=1.24              # Numerical operations
pandas>=2.0              # Data manipulation
```

### What This Approach Gives You

1. **Fast Development**: 80% of functionality from mature libraries
2. **Customization**: Your pipeline logic remains fully custom
3. **Reliability**: Battle-tested components for critical parts
4. **Maintainability**: Less code to maintain yourself
5. **Community**: Leverage existing documentation and examples

### Estimated Development Time

- **Core pipeline framework**: 2-3 days
- **UI integration**: 1-2 days (thanks to Streamlit)
- **Validation & metrics**: 1 day (thanks to Pydantic/sklearn)
- **Visualization**: 1-2 days
- **Testing & polish**: 2-3 days

**Total: ~1-2 weeks for a solid MVP** vs months if building everything from scratch

The key is using libraries for well-solved problems (UI, metrics, visualization) while keeping your unique pipeline logic custom and simple.
