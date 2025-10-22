"""Test script for interactive 3D point cloud visualization.

This script demonstrates:
1. Loading a point cloud
2. Creating an interactive 3D viewer with Plotly
3. Saving the viewer as HTML
4. Testing different color modes

Usage:
    python examples/test_visualization.py
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from examples.stages.pointcloud_loader import PointCloudLoader
from examples.stages.pointcloud_visualization import PointCloudVisualizationStage
from viz_art.types.pointcloud import VisualizationConfig


def test_basic_visualization():
    """Test basic point cloud visualization."""
    print("\n=== Test 1: Basic Visualization ===")

    # Use sample point cloud from fixtures
    fixtures_dir = Path(__file__).parent.parent / "tests" / "fixtures" / "sample_pointclouds"

    # Find first available point cloud file
    pcd_files = list(fixtures_dir.glob("*.pcd")) + list(fixtures_dir.glob("*.ply"))

    if not pcd_files:
        print("ERROR: No sample point cloud files found in fixtures directory")
        print(f"Please add .pcd or .ply files to: {fixtures_dir}")
        return False

    pcd_path = pcd_files[0]
    print(f"Loading point cloud: {pcd_path.name}")

    # Load point cloud
    loader = PointCloudLoader(
        name="loader",
        downsample_voxel_size=0.05,
        remove_outliers=True,
    )

    try:
        result = loader.run({"pointcloud_path": str(pcd_path)})
        print(f"✓ Loaded {result['metadata']['num_points']} points")
    except Exception as e:
        print(f"✗ Failed to load point cloud: {e}")
        return False

    # Create visualization
    output_dir = Path(__file__).parent.parent / "output" / "visualization"
    output_dir.mkdir(parents=True, exist_ok=True)

    viz_stage = PointCloudVisualizationStage(
        name="viewer",
        config=VisualizationConfig(
            point_size=3.0,
            opacity=0.8,
            color_mode="height",
            colorscale="Viridis",
            auto_downsample=True,
            max_render_points=100000,
        ),
        output_html=str(output_dir / "viewer_basic.html"),
    )

    try:
        viz_result = viz_stage.run({
            "points": result["points"],
            "colors": result.get("colors"),
        })

        print(f"✓ Created visualization with {viz_result['num_rendered_points']} points")

        if viz_result['was_downsampled']:
            print(f"  (Auto-downsampled from {result['metadata']['num_points']} points)")

        if viz_result['html_path']:
            print(f"✓ Saved HTML viewer: {viz_result['html_path']}")

        return True

    except Exception as e:
        print(f"✗ Failed to create visualization: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_color_modes():
    """Test different color modes."""
    print("\n=== Test 2: Different Color Modes ===")

    # Create synthetic point cloud
    n_points = 10000
    theta = np.linspace(0, 4 * np.pi, n_points)
    r = np.linspace(0, 2, n_points)

    # Spiral shape
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = theta / (2 * np.pi)

    points = np.column_stack([x, y, z]).astype(np.float32)

    # Generate colors (RGB gradient)
    colors = np.zeros((n_points, 3), dtype=np.float32)
    colors[:, 0] = np.linspace(0, 1, n_points)  # Red
    colors[:, 1] = np.linspace(1, 0, n_points)  # Green
    colors[:, 2] = 0.5  # Blue constant

    # Generate intensity (random)
    intensity = np.random.rand(n_points).astype(np.float32)

    output_dir = Path(__file__).parent.parent / "output" / "visualization"

    # Test height coloring
    print("Testing height coloring...")
    viz_height = PointCloudVisualizationStage(
        name="viz_height",
        color_mode="height",
        output_html=str(output_dir / "viewer_height.html"),
    )

    result_height = viz_height.run({"points": points, "colors": colors, "intensity": intensity})
    print(f"✓ Height mode: {result_height['html_path']}")

    # Test intensity coloring
    print("Testing intensity coloring...")
    viz_intensity = PointCloudVisualizationStage(
        name="viz_intensity",
        color_mode="intensity",
        output_html=str(output_dir / "viewer_intensity.html"),
    )

    result_intensity = viz_intensity.run({"points": points, "colors": colors, "intensity": intensity})
    print(f"✓ Intensity mode: {result_intensity['html_path']}")

    # Test RGB coloring
    print("Testing RGB coloring...")
    viz_rgb = PointCloudVisualizationStage(
        name="viz_rgb",
        color_mode="rgb",
        output_html=str(output_dir / "viewer_rgb.html"),
    )

    result_rgb = viz_rgb.run({"points": points, "colors": colors, "intensity": intensity})
    print(f"✓ RGB mode: {result_rgb['html_path']}")

    return True


def test_performance():
    """Test visualization performance with large point clouds."""
    print("\n=== Test 3: Performance (Large Point Cloud) ===")

    # Generate large synthetic point cloud
    n_points = 500000
    print(f"Generating {n_points:,} random points...")

    points = np.random.randn(n_points, 3).astype(np.float32) * 10

    output_dir = Path(__file__).parent.parent / "output" / "visualization"

    # Test with auto-downsampling
    viz_large = PointCloudVisualizationStage(
        name="viz_large",
        config=VisualizationConfig(
            auto_downsample=True,
            max_render_points=100000,
            point_size=2.0,
        ),
        output_html=str(output_dir / "viewer_large.html"),
    )

    import time
    start = time.time()

    result = viz_large.run({"points": points})

    elapsed = time.time() - start

    print(f"✓ Rendered {result['num_rendered_points']:,} points in {elapsed:.2f}s")
    print(f"  (Downsampled from {n_points:,} points)")
    print(f"  HTML saved: {result['html_path']}")

    if elapsed > 5.0:
        print(f"⚠ WARNING: Rendering took {elapsed:.2f}s (target < 5s)")
        return False

    return True


def main():
    """Run all visualization tests."""
    print("=" * 60)
    print("Point Cloud Visualization Tests")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Basic Visualization", test_basic_visualization()))
    results.append(("Color Modes", test_color_modes()))
    results.append(("Performance", test_performance()))

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n✓ All tests passed!")
        print("\nOpen the HTML files in a web browser to view the interactive visualizations:")
        output_dir = Path(__file__).parent.parent / "output" / "visualization"
        print(f"  {output_dir}/")
    else:
        print("\n✗ Some tests failed")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
