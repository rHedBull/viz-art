# Python API Contracts: Multi-Modal Visualization

**Feature**: 002-multimodal-viz
**Date**: 2025-10-22
**Type**: Library API (not REST/GraphQL)

## Overview

This document defines the Python API contracts for point cloud processing stages and multi-modal visualization components.

## Stage API Contracts

### 1. PointCloudLoaderStage

**Purpose**: Load point cloud files and perform initial preprocessing

**Class Signature**:
```python
class PointCloudLoader(PipelineStage):
    def __init__(
        self,
        name: str = "pointcloud_loader",
        downsample_voxel_size: float | None = None,
        remove_outliers: bool = True,
        outlier_neighbors: int = 20,
        outlier_std_ratio: float = 2.0,
        validation_rules: PointCloudValidationRules | None = None,
    ) -> None:
        ...
```

**Input Contract**:
```python
{
    "pointcloud_path": str  # Path to .pcd, .ply, or .xyz file (REQUIRED)
}
```

**Output Contract**:
```python
{
    "pointcloud": o3d.geometry.PointCloud,  # Open3D PointCloud object
    "points": np.ndarray,                    # Nx3 float32 array
    "colors": np.ndarray | None,             # Nx3 float32 array or None
    "metadata": {
        "num_points": int,
        "has_colors": bool,
        "has_normals": bool,
        "coordinate_system": str,            # "lidar", "camera", "world"
        "file_format": str,                   # ".pcd", ".ply", ".xyz"
    }
}
```

**Error Conditions**:
- `ValueError`: File not found, unsupported format, or invalid path
- `RuntimeError`: Empty point cloud after loading
- `ValidationError`: Point cloud fails validation rules

**Example Usage**:
```python
from viz_art.pipeline import Pipeline
from examples.stages.pointcloud_loader import PointCloudLoader

loader = PointCloudLoader(
    name="pc_loader",
    downsample_voxel_size=0.05,  # 5cm voxels
    remove_outliers=True
)

result = loader.run({"pointcloud_path": "scan.pcd"})
print(f"Loaded {result['metadata']['num_points']} points")
```

---

### 2. PointCloudProjectionStage

**Purpose**: Project 3D point cloud onto 2D image plane using camera calibration

**Class Signature**:
```python
class PointCloudProjectionStage(PipelineStage):
    def __init__(
        self,
        name: str = "projection",
        calibration_path: str | None = None,
        calibration: Calibration | None = None,
        point_size: int = 2,
        color_by_depth: bool = False,
        transparency: float = 1.0,
    ) -> None:
        ...
```

**Input Contract**:
```python
{
    "image": np.ndarray,          # HxWx3 uint8 array (REQUIRED)
    "points": np.ndarray,         # Nx3 float32 array (REQUIRED)
    "colors": np.ndarray | None   # Nx3 float32 array (OPTIONAL)
}
```

**Output Contract**:
```python
{
    "projected_image": np.ndarray,      # HxWx3 uint8 with overlaid points
    "points_2d": np.ndarray,            # Nx2 float32 projected coordinates
    "projection_mask": np.ndarray,      # N boolean mask of visible points
    "metadata": {
        "num_projected": int,            # Count of points within image
        "projection_rate": float,        # Fraction of points visible
    }
}
```

**Error Conditions**:
- `ValueError`: Missing calibration, invalid image/points dimensions
- `FileNotFoundError`: Calibration file not found
- `RuntimeError`: Calibration loading failed

**Example Usage**:
```python
projection = PointCloudProjectionStage(
    name="overlay",
    calibration_path="calibration/camera.yaml",
    color_by_depth=True
)

result = projection.run({
    "image": image_array,
    "points": points_3d,
    "colors": point_colors
})

cv2.imshow("Overlay", result["projected_image"])
```

---

### 3. PointCloudThumbnailStage

**Purpose**: Generate static thumbnail images of point clouds from fixed viewpoints

**Class Signature**:
```python
class PointCloudThumbnailStage(PipelineStage):
    def __init__(
        self,
        name: str = "thumbnail_generator",
        width: int = 800,
        height: int = 600,
        viewpoints: list[str] = None,  # ["front", "top", "diagonal"]
        background_color: tuple[float, float, float, float] = (1, 1, 1, 1),
        point_size: float = 3.0,
    ) -> None:
        ...
```

**Input Contract**:
```python
{
    "pointcloud": o3d.geometry.PointCloud  # Open3D PointCloud (REQUIRED)
}
```

**Output Contract**:
```python
{
    "thumbnails": {
        "front": np.ndarray,     # HxWx3 uint8 array
        "top": np.ndarray,       # HxWx3 uint8 array
        "diagonal": np.ndarray,  # HxWx3 uint8 array
        # ... other viewpoints
    },
    "thumbnail_paths": {          # Populated by OutputSaver
        "front": str | None,
        "top": str | None,
        "diagonal": str | None,
    },
    "metadata": {
        "render_time_ms": float,
        "viewpoints_generated": list[str],
    }
}
```

**Error Conditions**:
- `ValueError`: Empty point cloud, invalid dimensions
- `RuntimeError`: Rendering failed (GPU/display issues)

**Example Usage**:
```python
thumbnail_gen = PointCloudThumbnailStage(
    name="thumbs",
    width=400,
    height=300,
    viewpoints=["diagonal"]
)

result = thumbnail_gen.run({"pointcloud": pcd})
Image.fromarray(result["thumbnails"]["diagonal"]).save("thumb.png")
```

---

### 4. PointCloudVisualizationStage

**Purpose**: Generate interactive 3D Plotly visualization

**Class Signature**:
```python
class PointCloudVisualizationStage(PipelineStage):
    def __init__(
        self,
        name: str = "visualizer",
        vis_config: VisualizationConfig | None = None,
        output_html: bool = True,
        output_json: bool = False,
    ) -> None:
        ...
```

**Input Contract**:
```python
{
    "points": np.ndarray,                  # Nx3 float32 array (REQUIRED)
    "colors": np.ndarray | None,           # Nx3 float32 array (OPTIONAL)
    "labels": np.ndarray | None,           # N int array for classification (OPTIONAL)
    "intensity": np.ndarray | None,        # N float32 array (OPTIONAL)
}
```

**Output Contract**:
```python
{
    "plotly_figure": go.Figure,            # Plotly Figure object
    "html_content": str | None,            # HTML string if output_html=True
    "json_data": dict | None,              # JSON dict if output_json=True
    "metadata": {
        "num_points_rendered": int,
        "render_mode": str,                # "webgl" or "svg"
        "downsampled": bool,
    }
}
```

**Error Conditions**:
- `ValueError`: Empty points array, invalid dimensions
- `MemoryError`: Point cloud too large for rendering

**Example Usage**:
```python
visualizer = PointCloudVisualizationStage(
    name="3d_viewer",
    output_html=True
)

result = visualizer.run({
    "points": points,
    "colors": colors
})

fig = result["plotly_figure"]
fig.show()  # Opens in browser
```

---

### 5. PointCloudValidationStage

**Purpose**: Validate point cloud data quality against configurable rules

**Class Signature**:
```python
class PointCloudValidationStage(PipelineStage):
    def __init__(
        self,
        name: str = "validator",
        rules: PointCloudValidationRules,
        raise_on_invalid: bool = True,
    ) -> None:
        ...
```

**Input Contract**:
```python
{
    "pointcloud": o3d.geometry.PointCloud | None,  # (OPTIONAL)
    "points": np.ndarray | None                     # Nx3 array (OPTIONAL)
    # At least ONE must be provided
}
```

**Output Contract**:
```python
{
    "is_valid": bool,
    "validation_errors": list[str],        # List of error messages
    "validation_warnings": list[str],      # List of warnings
    "metrics": {
        "num_points": int,
        "has_nan": bool,
        "has_inf": bool,
        "coord_ranges": {
            "x": tuple[float, float],
            "y": tuple[float, float],
            "z": tuple[float, float],
        }
    }
}
```

**Error Conditions**:
- `ValueError`: No input provided (both pointcloud and points are None)
- `ValidationError`: Point cloud invalid and `raise_on_invalid=True`

**Example Usage**:
```python
rules = PointCloudValidationRules(
    min_points=1000,
    max_points=2000000,
    check_nan=True,
    check_inf=True
)

validator = PointCloudValidationStage(
    name="validate",
    rules=rules,
    raise_on_invalid=True
)

result = validator.run({"pointcloud": pcd})
if not result["is_valid"]:
    print("Validation errors:", result["validation_errors"])
```

---

### 6. MultiModalLoaderStage

**Purpose**: Load synchronized image and point cloud pairs

**Class Signature**:
```python
class MultiModalLoaderStage(PipelineStage):
    def __init__(
        self,
        name: str = "multimodal_loader",
        sync_tolerance_ms: float = 100.0,
        require_both: bool = True,
    ) -> None:
        ...
```

**Input Contract**:
```python
{
    "image_path": str | None,              # Path to image file (OPTIONAL)
    "pointcloud_path": str | None,         # Path to point cloud file (OPTIONAL)
    "calibration_path": str | None,        # Path to calibration file (OPTIONAL)
    # At least ONE of image_path or pointcloud_path REQUIRED
}
```

**Output Contract**:
```python
{
    "image": np.ndarray | None,            # HxWx3 uint8 array
    "pointcloud": o3d.geometry.PointCloud | None,
    "points": np.ndarray | None,           # Nx3 float32 array
    "colors": np.ndarray | None,           # Nx3 float32 array
    "calibration": Calibration | None,
    "metadata": {
        "has_image": bool,
        "has_pointcloud": bool,
        "has_calibration": bool,
        "time_synchronized": bool,
        "sync_delta_ms": float | None,
    }
}
```

**Error Conditions**:
- `ValueError`: No valid inputs provided
- `RuntimeError`: Synchronization check failed and `require_both=True`
- `FileNotFoundError`: Input file(s) not found

**Example Usage**:
```python
loader = MultiModalLoaderStage(
    name="mm_loader",
    sync_tolerance_ms=50.0,
    require_both=False
)

result = loader.run({
    "image_path": "frame_001.jpg",
    "pointcloud_path": "scan_001.pcd",
    "calibration_path": "camera.yaml"
})

if result["metadata"]["time_synchronized"]:
    print("Data is synchronized")
```

---

## Configuration API

### PipelineConfig Extension

**Addition to existing config schema**:

```python
@dataclass
class PointCloudStageConfig:
    """Configuration for point cloud processing stages."""
    name: str
    stage_type: str  # "PointCloudLoader", "PointCloudProjectionStage", etc.
    config: dict = field(default_factory=dict)
    enabled: bool = True
    input_data_types: dict[str, str] = field(default_factory=dict)
    output_data_types: dict[str, str] = field(default_factory=dict)

# Example YAML
"""
stages:
  - name: pc_loader
    stage_type: PointCloudLoader
    config:
      downsample_voxel_size: 0.05
      remove_outliers: true
    enabled: true
    input_data_types:
      pointcloud_path: str
    output_data_types:
      pointcloud: PointCloud
      points: ndarray
"""
```

---

## Helper Functions API

### Calibration Loading

```python
def load_calibration(
    path: str | Path,
    format: str = "auto"  # "auto", "yaml", "json", "opencv"
) -> Calibration:
    """
    Load camera calibration from file.

    Args:
        path: Path to calibration file
        format: File format, auto-detected if "auto"

    Returns:
        Calibration object

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If format invalid or parsing failed
    """
```

### Point Cloud Conversion

```python
def numpy_to_pointcloud(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    normals: np.ndarray | None = None
) -> o3d.geometry.PointCloud:
    """
    Convert NumPy arrays to Open3D PointCloud.

    Args:
        points: Nx3 array of XYZ coordinates
        colors: Nx3 array of RGB colors in [0,1] (optional)
        normals: Nx3 array of surface normals (optional)

    Returns:
        Open3D PointCloud object

    Raises:
        ValueError: If array shapes invalid
    """

def pointcloud_to_numpy(
    pcd: o3d.geometry.PointCloud
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Convert Open3D PointCloud to NumPy arrays.

    Args:
        pcd: Open3D PointCloud object

    Returns:
        Tuple of (points, colors, normals)
        colors and normals are None if not present
    """
```

### Projection Utilities

```python
def project_points(
    points_3d: np.ndarray,
    calibration: Calibration
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points onto 2D image plane.

    Args:
        points_3d: Nx3 array of 3D points
        calibration: Camera calibration

    Returns:
        Tuple of (points_2d, valid_mask)
        - points_2d: Nx2 array of image coordinates
        - valid_mask: N boolean array of points within image bounds

    Raises:
        ValueError: If points array invalid
    """
```

---

## Type Hints Module

**Location**: `src/viz_art/types/pointcloud.py`

```python
"""Type definitions for point cloud processing."""

from typing import TypeAlias, Literal
import numpy as np
import numpy.typing as npt
import open3d as o3d

# NumPy array type aliases
PointArray: TypeAlias = npt.NDArray[np.float32]  # Nx3
ColorArray: TypeAlias = npt.NDArray[np.float32]  # Nx3, [0,1]
NormalArray: TypeAlias = npt.NDArray[np.float32]  # Nx3
IntensityArray: TypeAlias = npt.NDArray[np.float32]  # N
ImageArray: TypeAlias = npt.NDArray[np.uint8]  # HxWx3

# Open3D type alias
PointCloudObject: TypeAlias = o3d.geometry.PointCloud

# Coordinate system types
CoordinateSystem: TypeAlias = Literal["camera", "lidar", "world"]

# Visualization types
ColorMode: TypeAlias = Literal["height", "intensity", "rgb", "class"]
Viewpoint: TypeAlias = Literal["front", "top", "side", "diagonal"]
RenderMode: TypeAlias = Literal["webgl", "svg"]
```

---

## Version Compatibility

**Python**: 3.8+
**Open3D**: 0.18+ (0.19 recommended)
**Plotly**: 5.0+
**NumPy**: 1.24+ (2.0 compatible)
**Pydantic**: 2.0+

---

## Error Handling Guidelines

### Standard Error Hierarchy

```
Exception
├── ValueError
│   ├── Invalid input dimensions
│   ├── Missing required parameters
│   └── Invalid parameter values
│
├── RuntimeError
│   ├── Empty point cloud after processing
│   ├── Rendering failures
│   └── Calibration loading errors
│
├── FileNotFoundError
│   ├── Point cloud file not found
│   └── Calibration file not found
│
└── ValidationError (custom)
    ├── Point cloud validation failure
    └── Type mismatch in pipeline
```

### Error Message Format

```python
# Good error messages
raise ValueError(
    f"Points array must be Nx3, got shape {points.shape}. "
    f"Expected format: (num_points, 3) with XYZ coordinates."
)

# Bad error messages
raise ValueError("Invalid points")  # Too vague
```

---

## Testing Contracts

### Unit Test Requirements

Each stage MUST have unit tests covering:

1. **Happy path**: Valid inputs → expected outputs
2. **Edge cases**: Empty inputs, boundary values
3. **Error cases**: Invalid inputs → appropriate exceptions
4. **Type validation**: Wrong types → ValueError
5. **Performance**: Large inputs within time limits (SC-001, SC-002)

### Integration Test Requirements

Multi-stage pipelines MUST test:

1. **Data flow**: Output of stage N → input of stage N+1
2. **Type compatibility**: Pipeline validation catches mismatches
3. **Error propagation**: Errors handled gracefully in pipeline
4. **End-to-end**: Full pipeline execution with real data

---

## Backward Compatibility

### Breaking Changes

None - this is a new feature (Phase 2) that extends existing API without modifying it.

### Deprecations

None planned.

### Migration Path

For users upgrading from Phase 1 (image-only):

1. Existing image pipelines continue to work without changes
2. Add `open3d`, `plotly`, `numpydantic` to dependencies
3. Optionally add point cloud stages to pipelines
4. Update OutputSaver config to save point clouds if desired

No code changes required for existing functionality.

---

## Documentation Requirements

Each public API function/class MUST have:

1. **Docstring** with Google-style format
2. **Args** section with types and descriptions
3. **Returns** section with type and description
4. **Raises** section listing all possible exceptions
5. **Example** usage code snippet

Example:

```python
def load_pointcloud(
    path: str,
    downsample: float | None = None
) -> o3d.geometry.PointCloud:
    """
    Load point cloud from file with optional downsampling.

    Supports .pcd, .ply, and .xyz formats. Automatically detects
    format from file extension.

    Args:
        path: Path to point cloud file
        downsample: Optional voxel size for downsampling (meters)

    Returns:
        Open3D PointCloud object with loaded data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format unsupported
        RuntimeError: If point cloud is empty after loading

    Example:
        >>> pcd = load_pointcloud("scan.pcd", downsample=0.05)
        >>> print(f"Loaded {len(pcd.points)} points")
        Loaded 12450 points
    """
```
