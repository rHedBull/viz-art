# Data Model: Multi-Modal Visualization

**Feature**: 002-multimodal-viz
**Date**: 2025-10-22
**Status**: Draft

## Overview

This document defines the core data entities for point cloud processing and multi-modal visualization in the viz-art pipeline library.

## Entity Definitions

### 1. PointCloud

**Purpose**: Represents 3D spatial data from LiDAR/depth sensors

**Attributes**:
- `points`: Nx3 NumPy array of XYZ coordinates (float32)
- `colors`: Optional Nx3 NumPy array of RGB values in [0,1] (float32)
- `normals`: Optional Nx3 NumPy array of surface normals (float32)
- `intensity`: Optional N array of reflectivity values (float32)
- `num_points`: Integer count of points
- `coordinate_system`: String identifier ("camera", "lidar", "world")
- `timestamp`: ISO 8601 timestamp string
- `sensor_id`: String identifier for source sensor

**Relationships**:
- Consumed by: PointCloudProcessingStage, VisualizationStage, ProjectionStage
- Produced by: PointCloudLoaderStage, DownsamplingStage, FilteringStage

**Validation Rules**:
- `points` MUST be Nx3 shape with N > 0
- `colors`, if present, MUST match points length
- `colors` MUST be in [0,1] range
- No NaN or Inf values in coordinate arrays
- `coordinate_system` MUST be one of: "camera", "lidar", "world"

**State Transitions**: Immutable after creation (frozen dataclass)

**Example**:
```python
from dataclasses import dataclass
from numpydantic import NDArray, Shape
import numpy as np

@dataclass(frozen=True)
class PointCloud:
    points: NDArray[Shape["* n_points, 3"], np.float32]
    colors: NDArray[Shape["* n_points, 3"], np.float32] | None = None
    normals: NDArray[Shape["* n_points, 3"], np.float32] | None = None
    intensity: NDArray[Shape["* n_points"], np.float32] | None = None
    num_points: int = 0
    coordinate_system: str = "lidar"
    timestamp: str = ""
    sensor_id: str = "default"

    def __post_init__(self):
        if self.num_points == 0:
            object.__setattr__(self, 'num_points', len(self.points))
```

---

### 2. MultiModalInput

**Purpose**: Represents synchronized image and point cloud data pairs with calibration

**Attributes**:
- `image`: Optional NumPy array HxWx3 (uint8)
- `pointcloud`: Optional PointCloud object
- `timestamp`: ISO 8601 timestamp string
- `sync_tolerance_ms`: Float, maximum time delta for sync (milliseconds)
- `calibration_ref`: String reference to calibration data
- `metadata`: Dictionary of additional key-value pairs

**Relationships**:
- Consumed by: MultiModalProcessingStage, ProjectionStage, OverlayVisualizationStage
- Produced by: MultiModalLoaderStage

**Validation Rules**:
- At least ONE of `image` or `pointcloud` MUST be present
- If both present and `sync_tolerance_ms` set, timestamps MUST be within tolerance
- `calibration_ref` REQUIRED if projection operations planned

**State Transitions**: Immutable after creation

**Example**:
```python
@dataclass(frozen=True)
class MultiModalInput:
    image: np.ndarray | None = None
    pointcloud: PointCloud | None = None
    timestamp: str = ""
    sync_tolerance_ms: float = 100.0  # 100ms default
    calibration_ref: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.image is None and self.pointcloud is None:
            raise ValueError("At least one modality required")
```

---

### 3. Calibration

**Purpose**: Camera intrinsic and extrinsic parameters for projection

**Attributes**:

**Intrinsics**:
- `fx`: Focal length X (pixels, float)
- `fy`: Focal length Y (pixels, float)
- `cx`: Principal point X (pixels, float)
- `cy`: Principal point Y (pixels, float)
- `width`: Image width (pixels, int)
- `height`: Image height (pixels, int)
- `distortion_coeffs`: List of 5 floats [k1, k2, p1, p2, k3]

**Extrinsics**:
- `rotation_matrix`: 3x3 NumPy array (float64)
- `translation_vector`: 3x1 NumPy array (float64)

**Metadata**:
- `camera_name`: String identifier
- `calibration_date`: ISO 8601 date string
- `calibration_method`: String (e.g., "opencv_checkerboard")

**Relationships**:
- Consumed by: ProjectionStage, OverlayVisualizationStage
- Loaded from: YAML/JSON calibration files

**Validation Rules**:
- `fx`, `fy` MUST be positive
- `width`, `height` MUST be positive integers
- `rotation_matrix` MUST be 3x3 orthogonal matrix (det ≈ 1)
- `distortion_coeffs` MUST have exactly 5 elements

**Methods**:
- `to_camera_matrix()` → 3x3 NumPy array
- `to_rodrigues_vector()` → 3x1 rotation vector (for OpenCV)
- `project_points(points_3d)` → points_2d, valid_mask

**Example**:
```python
@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion_coeffs: list[float] = field(default_factory=lambda: [0.0]*5)

    def to_matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0, 0, 1]], dtype=np.float64)

@dataclass
class CameraExtrinsics:
    rotation_matrix: np.ndarray = field(default_factory=lambda: np.eye(3))
    translation_vector: np.ndarray = field(default_factory=lambda: np.zeros((3,1)))

    def to_rodrigues_vector(self) -> np.ndarray:
        import cv2
        rvec, _ = cv2.Rodrigues(self.rotation_matrix)
        return rvec

@dataclass
class Calibration:
    camera_name: str
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics
    calibration_date: str = ""
    calibration_method: str = "manual"
```

---

### 4. PointCloudValidationRules

**Purpose**: Configuration for validating point cloud data quality

**Attributes**:
- `min_points`: Minimum acceptable point count (int, default 10)
- `max_points`: Maximum acceptable point count (int | None, default None)
- `check_nan`: Boolean, reject NaN values (default True)
- `check_inf`: Boolean, reject Inf values (default True)
- `coord_range_min`: Optional 3-element list for XYZ min bounds
- `coord_range_max`: Optional 3-element list for XYZ max bounds
- `fail_fast`: Boolean, stop on first error (default False)
- `log_level`: String, "error" | "warning" | "info" (default "error")

**Relationships**:
- Consumed by: PointCloudValidationStage, PointCloudLoaderStage
- Configured via: YAML config files, Pipeline configuration

**Validation Rules**:
- `min_points` MUST be >= 1
- `max_points`, if set, MUST be > `min_points`
- `coord_range_min` and `coord_range_max` MUST both be set or both None

**Example**:
```python
@dataclass
class PointCloudValidationRules:
    min_points: int = 10
    max_points: int | None = None
    check_nan: bool = True
    check_inf: bool = True
    coord_range_min: tuple[float, float, float] | None = None
    coord_range_max: tuple[float, float, float] | None = None
    fail_fast: bool = False
    log_level: str = "error"

    def validate_pointcloud(self, pcd: PointCloud) -> tuple[bool, list[str]]:
        """Validate point cloud against rules. Returns (is_valid, errors)."""
        errors = []

        # Check point count
        if pcd.num_points < self.min_points:
            errors.append(f"Too few points: {pcd.num_points} < {self.min_points}")
            if self.fail_fast:
                return False, errors

        if self.max_points and pcd.num_points > self.max_points:
            errors.append(f"Too many points: {pcd.num_points} > {self.max_points}")
            if self.fail_fast:
                return False, errors

        # Check NaN/Inf
        if self.check_nan and np.isnan(pcd.points).any():
            errors.append("NaN values detected in points")
            if self.fail_fast:
                return False, errors

        if self.check_inf and np.isinf(pcd.points).any():
            errors.append("Inf values detected in points")
            if self.fail_fast:
                return False, errors

        # Check coordinate ranges
        if self.coord_range_min and self.coord_range_max:
            mins = pcd.points.min(axis=0)
            maxs = pcd.points.max(axis=0)

            for i, (mn, mx, coord) in enumerate(zip(self.coord_range_min, self.coord_range_max, ['X', 'Y', 'Z'])):
                if mins[i] < mn or maxs[i] > mx:
                    errors.append(f"{coord} range [{mins[i]:.2f}, {maxs[i]:.2f}] outside [{mn}, {mx}]")
                    if self.fail_fast:
                        return False, errors

        return len(errors) == 0, errors
```

---

### 5. VisualizationConfig

**Purpose**: Configuration for visualization display and rendering

**Attributes**:

**3D Viewer Settings**:
- `point_size`: Float, size of points in pixels (default 2.0)
- `opacity`: Float in [0,1], point transparency (default 0.8)
- `color_mode`: String, "height" | "intensity" | "rgb" | "class" (default "height")
- `colorscale`: String, Plotly colorscale name (default "Viridis")
- `background_color`: RGBA tuple (default (1, 1, 1, 1))
- `camera_projection`: String, "perspective" | "orthographic" (default "perspective")

**2D Viewer Settings**:
- `enable_zoom`: Boolean (default True)
- `enable_pan`: Boolean (default True)
- `annotation_thickness`: Int, pixels (default 2)
- `annotation_color`: RGB tuple (default (255, 0, 0))

**Thumbnail Settings**:
- `thumbnail_width`: Int, pixels (default 800)
- `thumbnail_height`: Int, pixels (default 600)
- `thumbnail_viewpoint`: String, "front" | "top" | "side" | "diagonal" (default "diagonal")
- `thumbnail_quality`: String, "low" | "medium" | "high" (default "medium")

**Performance Settings**:
- `auto_downsample`: Boolean, enable automatic downsampling (default True)
- `max_render_points`: Int, maximum points before downsampling (default 500000)
- `render_mode`: String, "webgl" | "svg" (default "webgl")

**Relationships**:
- Consumed by: All visualization stages
- Configured via: Pipeline YAML configs

**Validation Rules**:
- `point_size` MUST be > 0
- `opacity` MUST be in [0, 1]
- `color_mode` MUST be valid option
- `thumbnail_quality` affects rendering time vs quality tradeoff

**Example**:
```python
@dataclass
class VisualizationConfig:
    # 3D viewer
    point_size: float = 2.0
    opacity: float = 0.8
    color_mode: str = "height"
    colorscale: str = "Viridis"
    background_color: tuple[float, float, float, float] = (1, 1, 1, 1)
    camera_projection: str = "perspective"

    # 2D viewer
    enable_zoom: bool = True
    enable_pan: bool = True
    annotation_thickness: int = 2
    annotation_color: tuple[int, int, int] = (255, 0, 0)

    # Thumbnails
    thumbnail_width: int = 800
    thumbnail_height: int = 600
    thumbnail_viewpoint: str = "diagonal"
    thumbnail_quality: str = "medium"

    # Performance
    auto_downsample: bool = True
    max_render_points: int = 500000
    render_mode: str = "webgl"

    def should_downsample(self, num_points: int) -> bool:
        """Determine if downsampling needed."""
        return self.auto_downsample and num_points > self.max_render_points
```

---

## Entity Relationships Diagram

```
┌─────────────────┐
│  MultiModalInput │
└────────┬─────────┘
         │
         ├───► Image (NumPy array)
         │
         ├───► PointCloud
         │      │
         │      ├─► points (Nx3)
         │      ├─► colors (Nx3)
         │      ├─► normals (Nx3)
         │      └─► metadata
         │
         └───► Calibration
                 │
                 ├─► CameraIntrinsics
                 │    └─► to_matrix()
                 │
                 └─► CameraExtrinsics
                      └─► to_rodrigues_vector()

┌──────────────────────────┐
│  PointCloudValidationRules│
└────────────────┬─────────┘
                 │
                 ├───► Validates PointCloud
                 │
                 └───► Returns (is_valid, errors)

┌──────────────────────┐
│  VisualizationConfig  │
└────────────┬─────────┘
             │
             ├───► 3D Viewer Settings
             ├───► 2D Viewer Settings
             ├───► Thumbnail Settings
             └───► Performance Settings
```

## Data Flow Through Pipeline

```
Input Files
(.pcd, .ply, .xyz, .jpg)
         │
         ▼
    ┌────────┐
    │ Loader │ ──► PointCloud + Image
    │ Stages │
    └────┬───┘
         │
         ▼
  ┌──────────────┐
  │ Processing   │ ──► Transformed PointCloud/Image
  │ Stages       │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │ Validation   │ ──► ValidationResult
  │ (optional)   │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │ Visualization│ ──► HTML/Interactive Viewer
  │ /Thumbnail   │      + Thumbnail Images
  └──────────────┘
```

## Storage Considerations

### Point Cloud Files
- **Format Priority**: .ply (full features) > .pcd (Open3D native) > .xyz (minimal)
- **Typical Size**: 100K points ≈ 5-10MB (.ply with colors)
- **Memory Usage**: 100K points ≈ 12MB in RAM (float32)

### Calibration Files
- **Format**: YAML (preferred) or JSON
- **Size**: < 1KB per calibration
- **Versioning**: Include calibration_date in metadata

### Visualization Cache
- **Thumbnail Images**: 800x600 PNG ≈ 0.5-2MB
- **Downsampled Point Clouds**: Cached at 50K-100K points for repeated viewing
- **Location**: `output/cache/` directory

## Configuration Examples

### YAML Configuration for Point Cloud Loading

```yaml
stages:
  - name: pc_loader
    stage_type: PointCloudLoader
    config:
      downsample_voxel_size: 0.05  # 5cm voxels
      remove_outliers: true
      outlier_neighbors: 20
      outlier_std_ratio: 2.0
      validation_rules:
        min_points: 1000
        max_points: 2000000
        check_nan: true
        check_inf: true
        fail_fast: false
```

### YAML Calibration File

```yaml
camera_name: front_camera
calibration_date: "2025-10-15"
calibration_method: opencv_checkerboard

intrinsics:
  fx: 525.0
  fy: 525.0
  cx: 319.5
  cy: 239.5
  width: 640
  height: 480
  distortion_coeffs: [0.1, -0.2, 0.001, -0.002, 0.0]

extrinsics:
  rotation_matrix:
    - [0.999, 0.001, -0.01]
    - [-0.001, 1.0, 0.002]
    - [0.01, -0.002, 0.999]
  translation_vector: [0.0, 0.0, 0.0]
```

## Next Steps

1. Create contracts for point cloud stage APIs
2. Define quickstart guide with example configurations
3. Implement validation logic in stage base classes
