"""Generate sample point cloud test data.

This script creates test point clouds in various formats for unit testing.
Run once during setup or when test data needs to be regenerated.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    print("Warning: Open3D not installed. Skipping test data generation.")
    print("Install with: pip install open3d>=0.18")
    OPEN3D_AVAILABLE = False
    sys.exit(1)


def generate_small_cube(output_dir: Path):
    """Generate a small 100-point cube."""
    np.random.seed(42)
    points = np.random.rand(100, 3).astype(np.float32)
    colors = np.random.rand(100, 3).astype(np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Save in multiple formats
    o3d.io.write_point_cloud(str(output_dir / "small_100pts.pcd"), pcd)
    o3d.io.write_point_cloud(str(output_dir / "small_100pts.ply"), pcd)

    # .xyz format (no colors)
    xyz_path = output_dir / "small_100pts.xyz"
    np.savetxt(xyz_path, points, fmt='%.6f')

    print(f"✓ Generated small_100pts (100 points)")


def generate_medium_cloud(output_dir: Path):
    """Generate a medium 10K-point sphere."""
    np.random.seed(123)

    # Generate points on a sphere
    n_points = 10000
    phi = np.random.uniform(0, 2 * np.pi, n_points)
    costheta = np.random.uniform(-1, 1, n_points)
    theta = np.arccos(costheta)
    radius = 1.0

    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    points = np.column_stack([x, y, z]).astype(np.float32)

    # Color by height (z-coordinate)
    colors = np.zeros((n_points, 3), dtype=np.float32)
    colors[:, 0] = (z - z.min()) / (z.max() - z.min())  # Red channel
    colors[:, 2] = 1.0 - colors[:, 0]  # Blue channel (inverse)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )

    o3d.io.write_point_cloud(str(output_dir / "medium_10k.pcd"), pcd)
    o3d.io.write_point_cloud(str(output_dir / "medium_10k.ply"), pcd)

    print(f"✓ Generated medium_10k (10,000 points with normals)")


def generate_large_cloud(output_dir: Path):
    """Generate a large 100K-point random cloud."""
    np.random.seed(456)

    n_points = 100000
    points = np.random.randn(n_points, 3).astype(np.float32) * 10.0  # Scale up

    # Color by distance from origin
    distances = np.linalg.norm(points, axis=1)
    colors = np.zeros((n_points, 3), dtype=np.float32)
    colors[:, 1] = (distances - distances.min()) / (distances.max() - distances.min())

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(str(output_dir / "large_100k.pcd"), pcd)

    print(f"✓ Generated large_100k (100,000 points)")


def generate_empty_cloud(output_dir: Path):
    """Generate an empty point cloud for error testing."""
    pcd = o3d.geometry.PointCloud()

    o3d.io.write_point_cloud(str(output_dir / "empty.pcd"), pcd)

    print(f"✓ Generated empty.pcd (0 points - for error testing)")


def generate_corrupted_data(output_dir: Path):
    """Generate corrupted test data (NaN, Inf) for validation testing."""
    np.random.seed(789)

    # Cloud with NaN values
    points_nan = np.random.rand(100, 3).astype(np.float32)
    points_nan[10:15, 0] = np.nan  # Inject NaN
    colors_nan = np.random.rand(100, 3).astype(np.float32)

    pcd_nan = o3d.geometry.PointCloud()
    pcd_nan.points = o3d.utility.Vector3dVector(points_nan)
    pcd_nan.colors = o3d.utility.Vector3dVector(colors_nan)

    o3d.io.write_point_cloud(str(output_dir / "corrupted_nan.pcd"), pcd_nan)

    # Cloud with Inf values
    points_inf = np.random.rand(100, 3).astype(np.float32)
    points_inf[20:25, 1] = np.inf  # Inject Inf
    colors_inf = np.random.rand(100, 3).astype(np.float32)

    pcd_inf = o3d.geometry.PointCloud()
    pcd_inf.points = o3d.utility.Vector3dVector(points_inf)
    pcd_inf.colors = o3d.utility.Vector3dVector(colors_inf)

    o3d.io.write_point_cloud(str(output_dir / "corrupted_inf.pcd"), pcd_inf)

    print(f"✓ Generated corrupted test data (NaN and Inf)")


def generate_readme(output_dir: Path):
    """Generate README documenting test fixtures."""
    readme_content = """# Sample Point Cloud Test Fixtures

This directory contains point cloud test data for unit and integration testing.

## Files

### Valid Test Data

- `small_100pts.{pcd,ply,xyz}` - Small 100-point random cube
  - Format: All supported formats
  - Features: Points + colors
  - Use: Fast unit tests

- `medium_10k.{pcd,ply}` - Medium 10K-point sphere
  - Format: PCD and PLY
  - Features: Points + colors + normals
  - Use: Performance tests, normal estimation

- `large_100k.pcd` - Large 100K-point random cloud
  - Format: PCD only (file size)
  - Features: Points + colors
  - Use: Performance benchmarks, downsampling tests

### Invalid Test Data (for validation testing)

- `empty.pcd` - Empty point cloud (0 points)
  - Use: Error handling tests

- `corrupted_nan.pcd` - Contains NaN values
  - Use: Validation framework tests

- `corrupted_inf.pcd` - Contains Inf values
  - Use: Validation framework tests

## Regenerating Test Data

If test data is corrupted or needs updating:

```bash
cd tests/fixtures/sample_pointclouds/
python generate_test_data.py
```

Requires: `open3d>=0.18`

## Usage in Tests

```python
import pytest
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "sample_pointclouds"

def test_load_pointcloud():
    pcd_path = FIXTURES_DIR / "small_100pts.pcd"
    # ... test code
```

Generated automatically - do not edit manually.
"""

    readme_path = output_dir / "README.md"
    readme_path.write_text(readme_content)

    print(f"✓ Generated README.md")


def main():
    """Generate all test fixtures."""
    if not OPEN3D_AVAILABLE:
        return

    output_dir = Path(__file__).parent
    print(f"Generating test point clouds in: {output_dir}\n")

    generate_small_cube(output_dir)
    generate_medium_cloud(output_dir)
    generate_large_cloud(output_dir)
    generate_empty_cloud(output_dir)
    generate_corrupted_data(output_dir)
    generate_readme(output_dir)

    print("\n✅ All test fixtures generated successfully!")
    print("Total files:", len(list(output_dir.glob("*.p*"))))


if __name__ == "__main__":
    main()
