# viz-art Examples

This directory contains comprehensive examples demonstrating all features of the viz-art pipeline library.

## Quick Start

All examples can be run directly from the `examples/` directory:

```bash
cd examples
python test_visualization.py
python test_projection.py
python test_validation.py
```

## Example Scripts

### Point Cloud Processing

#### `test_visualization.py`
Interactive 3D point cloud visualization using Plotly WebGL.

**Features:**
- Load and display point clouds interactively
- Color coding by height, intensity, or RGB
- Rotation, zoom, pan controls
- Auto-downsampling for large clouds (>500K points)
- Export to HTML for sharing

**Usage:**
```bash
python test_visualization.py
```

#### `test_projection.py`
Project 3D point clouds onto 2D images using camera calibration.

**Features:**
- Load camera calibration from YAML
- Project 3D points to 2D image plane
- Depth-based color coding
- Adjustable transparency blending
- Generates synthetic test data

**Usage:**
```bash
python test_projection.py
```

#### `test_validation.py`
Comprehensive data quality validation test suite.

**Features:**
- Tests 6 validation scenarios (valid, NaN, Inf, range, empty, etc.)
- Configurable validation rules
- Clear error reporting
- Performance benchmarking

**Usage:**
```bash
python test_validation.py
```

**Expected output:**
- All 6 tests should pass
- Each test shows detected issues
- Success rate: 100%

#### `test_stage_outputs.py`
Multi-modal pipeline with stage-by-stage output inspection.

**Features:**
- Process both images and point clouds
- View intermediate outputs from each stage
- Automatic thumbnail generation
- HTML report with 3D viewers

**Usage:**
```bash
python test_stage_outputs.py
```

#### `test_batch_thumbnails.py`
Batch processing with thumbnail generation and caching.

**Features:**
- Create synthetic test point clouds
- Generate thumbnails with caching
- Performance benchmarking (first vs cached)
- Verify SC-006 target (< 3s per cloud)

**Usage:**
```bash
python test_batch_thumbnails.py
```

**Expected output:**
- 5 test clouds created
- First pass: 1-2s per thumbnail
- Second pass (cached): 0.01-0.05s per thumbnail
- Speedup: ~10-100x

## Example Stages

The `stages/` directory contains production-ready pipeline stages:

### Point Cloud Stages

#### `pointcloud_loader.py`
Load point clouds from .pcd, .ply, .xyz files with preprocessing.

**Parameters:**
- `downsample_voxel_size`: Voxel size for downsampling (meters)
- `remove_outliers`: Enable statistical outlier removal
- `outlier_neighbors`: Number of neighbors for outlier detection
- `outlier_std_ratio`: Standard deviation ratio threshold

**Example:**
```python
loader = PointCloudLoader(
    name="loader",
    downsample_voxel_size=0.05,  # 5cm voxels
    remove_outliers=True
)
result = loader.run({"pointcloud_path": "cloud.pcd"})
```

#### `pointcloud_validation.py`
Validate point cloud data quality.

**Parameters:**
- `rules`: PointCloudValidationRules object
  - `min_points`: Minimum acceptable point count
  - `max_points`: Maximum acceptable point count
  - `check_nan`: Detect NaN values
  - `check_inf`: Detect Inf values
  - `coord_range_min/max`: Coordinate bounds
  - `fail_fast`: Stop on first error
- `raise_on_invalid`: Raise exception on validation failure

**Example:**
```python
rules = PointCloudValidationRules(
    min_points=1000,
    check_nan=True,
    check_inf=True
)
validator = PointCloudValidationStage(name="validator", rules=rules)
result = validator.run({"points": points_array})
# result contains: is_valid, validation_errors, metrics
```

#### `pointcloud_visualization.py`
Generate interactive 3D Plotly visualizations.

**Parameters:**
- `color_mode`: "height", "intensity", "rgb", "class"
- `point_size`: Point size in pixels
- `opacity`: Transparency [0,1]
- `colorscale`: Plotly colorscale name
- `max_render_points`: Auto-downsample threshold
- `output_html`: Save HTML file
- `output_json`: Save JSON file

**Example:**
```python
visualizer = PointCloudVisualizationStage(
    name="viz",
    color_mode="height",
    output_html=True
)
result = visualizer.run({"points": points, "colors": colors})
```

#### `pointcloud_projection.py`
Project 3D point clouds onto 2D images.

**Parameters:**
- `calibration_path`: Path to YAML calibration file
- `color_mode`: "depth", "rgb", "intensity"
- `point_radius`: Rendered point radius (pixels)
- `opacity`: Blending opacity [0,1]
- `z_threshold`: Minimum Z distance (meters)

**Example:**
```python
projection = PointCloudProjectionStage(
    name="projection",
    calibration_path="calibration/camera.yaml",
    color_mode="depth"
)
result = projection.run({"image": image, "points": points_3d})
```

#### `pointcloud_thumbnail.py`
Generate static thumbnail images from multiple viewpoints.

**Parameters:**
- `viewpoints`: List of viewpoints ("front", "top", "side", "diagonal")
- `width`: Thumbnail width (pixels)
- `height`: Thumbnail height (pixels)
- `point_size`: Point size (pixels)
- `background_color`: RGB background color

**Example:**
```python
thumbnail_gen = PointCloudThumbnailStage(
    name="thumbnail",
    viewpoints=["diagonal", "front", "top"]
)
result = thumbnail_gen.run({"points": points, "colors": colors})
```

#### `multimodal_loader.py`
Load synchronized image and point cloud pairs.

**Parameters:**
- `sync_tolerance_ms`: Maximum time delta for synchronization
- `require_both`: Enforce both modalities present
- `load_colors`: Load point cloud colors
- `load_normals`: Load point cloud normals

**Example:**
```python
loader = MultiModalLoaderStage(
    name="multimodal",
    sync_tolerance_ms=100.0,
    require_both=True
)
result = loader.run({
    "image_path": "image.jpg",
    "pointcloud_path": "cloud.pcd",
    "image_timestamp": "2025-10-22T10:30:00.000",
    "pointcloud_timestamp": "2025-10-22T10:30:00.050"
})
# result contains: is_synchronized, time_delta_ms, metadata
```

## Example Configurations

The `configs/` directory contains YAML pipeline configurations:

### `pointcloud_simple.yaml`
Basic point cloud loading and validation pipeline.

### `multimodal_stages.yaml`
Multi-modal pipeline with image and point cloud stages.

### `validation_strict.yaml`
Strict validation rules for quality checks.

### `batch_pointclouds.yaml`
Batch processing with thumbnail generation.

**Usage:**
```bash
# Run with config file
viz-art-cli run --config configs/pointcloud_simple.yaml
```

## Calibration Files

The `calibration/` directory contains example camera calibration files:

### `camera.yaml`
Sample calibration with intrinsic and extrinsic parameters.

**Format:**
```yaml
camera_name: "front_camera"
intrinsics:
  fx: 525.0
  fy: 525.0
  cx: 319.5
  cy: 239.5
  width: 640
  height: 480
  distortion_coeffs: [0.0, 0.0, 0.0, 0.0, 0.0]
extrinsics:
  rotation_matrix:
    - [1.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0]
    - [0.0, 0.0, 1.0]
  translation_vector: [0.0, 0.0, 0.0]
```

## Performance Notes

- **Loading**: 100K points in < 5s (SC-001)
- **Visualization**: 30+ FPS for 500K points (SC-002)
- **Thumbnails**: < 3s per cloud (SC-006)
- **Caching**: 10-100x speedup on repeated renders
- **Projection**: < 2px accuracy (SC-004)

## Troubleshooting

### Open3D Installation Issues

```bash
# Try upgrading pip first
pip install --upgrade pip

# Install Open3D
pip install open3d --no-cache-dir

# Alternative: use conda
conda install -c open3d-admin open3d
```

### WebGL Performance Issues

If interactive viewers are slow (< 30 FPS):
- Reduce `max_render_points` to 100K-200K
- Enable `auto_downsample` in VisualizationConfig
- Check browser supports WebGL 2.0

### Headless Rendering

For thumbnail generation on servers without display:

```bash
# Install xvfb (Linux)
apt-get install xvfb

# Run with xvfb
xvfb-run python examples/test_batch_thumbnails.py
```

## Next Steps

1. Try the example scripts to understand the API
2. Modify example configs for your use case
3. Implement custom stages by extending `PipelineStage`
4. See [specs/002-multimodal-viz/](../specs/002-multimodal-viz/) for detailed documentation

## Support

- Documentation: [specs/002-multimodal-viz/quickstart.md](../specs/002-multimodal-viz/quickstart.md)
- API Reference: [specs/002-multimodal-viz/contracts/](../specs/002-multimodal-viz/contracts/)
- Issues: GitHub Issues
