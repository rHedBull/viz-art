# Quickstart: Multi-Modal Visualization

**Feature**: 002-multimodal-viz
**Date**: 2025-10-22
**Audience**: Developers implementing point cloud support

## Prerequisites

- Completed Phase 1 (base pipeline architecture)
- Python 3.8+
- Understanding of 3D coordinate systems

## Installation

Add new dependencies to `pyproject.toml`:

```toml
[project]
dependencies = [
    # Phase 1 dependencies (existing)
    "pydantic>=2.0",
    "omegaconf>=2.3",
    "pillow>=10.0",
    "opencv-python>=4.8",
    "jinja2>=3.1",
    "numpy>=1.24",

    # Phase 2 additions
    "open3d>=0.18",        # Point cloud processing
    "plotly>=5.0",         # Interactive visualization
    "numpydantic>=1.6",    # NumPy validation
]
```

Install dependencies:
```bash
pip install -e ".[dev]"
```

## Quick Start: Load and Visualize Point Cloud

### 1. Create a Point Cloud Loader Stage

Create `examples/stages/pointcloud_loader.py`:

```python
from typing import Dict, Any, List
from pathlib import Path
import numpy as np
import open3d as o3d
from viz_art.pipeline import PipelineStage

class PointCloudLoader(PipelineStage):
    """Load point clouds from .pcd/.ply/.xyz files."""

    def __init__(
        self,
        name: str = "pointcloud_loader",
        downsample_voxel_size: float | None = 0.05,
        remove_outliers: bool = True,
    ):
        self._name = name
        self.downsample_voxel_size = downsample_voxel_size
        self.remove_outliers = remove_outliers

    @property
    def name(self) -> str:
        return self._name

    @property
    def input_keys(self) -> List[str]:
        return ["pointcloud_path"]

    @property
    def output_keys(self) -> List[str]:
        return ["pointcloud", "points", "colors", "metadata"]

    def pre_process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        path = Path(inputs["pointcloud_path"])
        if not path.exists():
            raise FileNotFoundError(f"Point cloud not found: {path}")
        return inputs

    def predict(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        # Load point cloud
        pcd = o3d.io.read_point_cloud(preprocessed["pointcloud_path"])

        if len(pcd.points) == 0:
            raise RuntimeError("Point cloud is empty")

        # Downsample if requested
        if self.downsample_voxel_size:
            pcd = pcd.voxel_down_sample(self.downsample_voxel_size)

        # Remove outliers
        if self.remove_outliers:
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        # Convert to numpy
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None

        return {
            "pcd_object": pcd,
            "points_array": points,
            "colors_array": colors,
            "num_points": len(points),
        }

    def post_process(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "pointcloud": predictions["pcd_object"],
            "points": predictions["points_array"],
            "colors": predictions["colors_array"],
            "metadata": {
                "num_points": predictions["num_points"],
                "has_colors": predictions["colors_array"] is not None,
            }
        }
```

### 2. Test the Stage

Create `examples/test_pointcloud.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stages.pointcloud_loader import PointCloudLoader
import open3d as o3d

# Load a point cloud
loader = PointCloudLoader(
    name="loader",
    downsample_voxel_size=0.05  # 5cm voxels
)

result = loader.run({"pointcloud_path": "path/to/your/pointcloud.pcd"})

print(f"Loaded {result['metadata']['num_points']} points")
print(f"Has colors: {result['metadata']['has_colors']}")

# Visualize with Open3D
o3d.visualization.draw_geometries([result['pointcloud']])
```

Run:
```bash
python examples/test_pointcloud.py
```

## Example: Multi-Modal Pipeline

### Configuration File

Create `config/multimodal_pipeline.yaml`:

```yaml
pipeline_name: "multimodal-demo"
output_dir: "./output/multimodal"

stages:
  - name: image_loader
    stage_type: ImageLoader
    config:
      resize: [640, 480]
      color_mode: RGB

  - name: pc_loader
    stage_type: PointCloudLoader
    config:
      downsample_voxel_size: 0.05
      remove_outliers: true

connections:
  []  # Parallel processing, no connections

batch_config:
  input_dir: "./data/multimodal"
  output_dir: "./output/multimodal"
  file_patterns:
    - "*.jpg"
    - "*.pcd"
    - "*.ply"
  recursive: true
  continue_on_error: true
  report_output: "multimodal_report.html"

  output_mode: "sample"
  save_outputs:
    enabled: true
    stages: ["all"]
    max_samples: 10
    format: "png"  # For point clouds, will use .ply
```

### Run Pipeline

```python
from viz_art.config.loader import load_config
from viz_art.pipeline.base import Pipeline
from viz_art.batch.processor import BatchProcessor
from viz_art.batch.reporter import HTMLReporter

# Import stages
from stages.image_loader import ImageLoader
from stages.pointcloud_loader import PointCloudLoader

# Create registry
STAGE_REGISTRY = {
    "ImageLoader": ImageLoader,
    "PointCloudLoader": PointCloudLoader,
}

# Load config
config = load_config("config/multimodal_pipeline.yaml")

# Create pipeline
pipeline = Pipeline.from_config(config, stage_registry=STAGE_REGISTRY)

# Run batch processing
processor = BatchProcessor(pipeline, config.batch_config)
batch_result = processor.run()

# Generate report
reporter = HTMLReporter()
report_path = reporter.generate(
    batch_result,
    config.batch_config.output_dir + "/" + config.batch_config.report_output,
    pipeline_name=config.pipeline_name
)

print(f"Report: {report_path}")
```

## Example: Point Cloud Projection

### 1. Create Calibration File

Create `calibration/camera.yaml`:

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
  rotation:
    - [1.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0]
    - [0.0, 0.0, 1.0]
  translation: [0.0, 0.0, 0.0]
```

### 2. Create Projection Stage

See `contracts/python_api.md` for `PointCloudProjectionStage` implementation.

### 3. Test Projection

```python
from stages.projection import PointCloudProjectionStage
import cv2

projection = PointCloudProjectionStage(
    name="overlay",
    calibration_path="calibration/camera.yaml"
)

result = projection.run({
    "image": image_array,
    "points": points_3d,
    "colors": point_colors
})

cv2.imshow("Overlay", result["projected_image"])
cv2.waitKey(0)
```

## Testing

### Unit Tests

Create `tests/unit/test_pointcloud_stages.py`:

```python
import pytest
import numpy as np
import open3d as o3d
from examples.stages.pointcloud_loader import PointCloudLoader

def test_pointcloud_loader():
    # Create test point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))
    test_path = "/tmp/test.pcd"
    o3d.io.write_point_cloud(test_path, pcd)

    # Test stage
    stage = PointCloudLoader(name="test")
    result = stage.run({"pointcloud_path": test_path})

    assert "points" in result
    assert result["points"].shape == (100, 3)
    assert result["metadata"]["num_points"] == 100

def test_empty_pointcloud_error():
    stage = PointCloudLoader()

    with pytest.raises(FileNotFoundError):
        stage.run({"pointcloud_path": "nonexistent.pcd"})
```

Run tests:
```bash
pytest tests/unit/test_pointcloud_stages.py -v
```

## Common Issues

### Issue 1: Open3D Installation Fails

**Solution**: Use pre-built wheels
```bash
pip install --upgrade pip
pip install open3d --no-cache-dir
```

Or use conda:
```bash
conda install -c open3d-admin open3d
```

### Issue 2: WebGL Rendering Slow

**Problem**: Plotly struggles with >500K points

**Solution**: Enable automatic downsampling
```python
# In visualization config
vis_config = VisualizationConfig(
    auto_downsample=True,
    max_render_points=100000
)
```

### Issue 3: Headless Thumbnail Rendering

**Problem**: `OffscreenRenderer` needs display server

**Solution**: Use xvfb on Linux
```bash
apt-get install xvfb
xvfb-run python generate_thumbnails.py
```

## Next Steps

1. **Add validation**: Implement `PointCloudValidationStage`
2. **Add visualization**: Create interactive Plotly viewer
3. **Add thumbnails**: Implement thumbnail generation for reports
4. **Extend OutputSaver**: Support .pcd/.ply file saving

## Resources

- [Open3D Documentation](https://www.open3d.org/docs/latest/)
- [Plotly 3D Scatter](https://plotly.com/python/3d-scatter-plots/)
- [Phase 2 Research](./research.md)
- [Data Model](./data-model.md)
- [API Contracts](./contracts/python_api.md)
