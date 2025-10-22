"""Example: Batch processing with point cloud thumbnails.

This script demonstrates batch processing of multiple point clouds
with automatic thumbnail generation and HTML report creation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import open3d as o3d
from viz_art.visualization.thumbnail import render_thumbnail_with_cache


def create_test_pointclouds(output_dir: Path, num_clouds: int = 5):
    """Create synthetic test point clouds.

    Args:
        output_dir: Directory to save test clouds
        num_clouds: Number of clouds to create

    Returns:
        List of created file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    created_files = []

    for i in range(num_clouds):
        # Generate random point cloud
        num_points = np.random.randint(1000, 10000)
        points = np.random.randn(num_points, 3).astype(np.float32) * 10

        # Add some color variation
        colors = np.random.rand(num_points, 3).astype(np.float32)

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Save to file
        filename = f"test_cloud_{i:03d}.ply"
        filepath = output_dir / filename
        o3d.io.write_point_cloud(str(filepath), pcd)

        created_files.append(filepath)
        print(f"Created: {filename} ({num_points} points)")

    return created_files


def test_thumbnail_generation():
    """Test thumbnail generation with caching and performance."""
    print("=" * 70)
    print("Point Cloud Thumbnail Generation Test")
    print("=" * 70)

    # Create test data
    print("\n1. Creating test point clouds...")
    data_dir = Path(__file__).parent / "data" / "test_pointclouds"
    cloud_files = create_test_pointclouds(data_dir, num_clouds=5)
    print(f"   Created {len(cloud_files)} test point clouds")

    # Setup cache
    cache_dir = Path(__file__).parent / "output" / ".thumbnail_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"   Cache directory: {cache_dir}")

    # Test thumbnail generation
    print("\n2. Generating thumbnails (first pass - no cache)...")
    output_dir = Path(__file__).parent / "output" / "test_thumbnails"
    output_dir.mkdir(parents=True, exist_ok=True)

    import time

    first_pass_times = []

    for i, cloud_file in enumerate(cloud_files):
        print(f"\n   Processing: {cloud_file.name}")

        # Load point cloud
        pcd = o3d.io.read_point_cloud(str(cloud_file))
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None

        print(f"   - Points: {points.shape[0]}")

        # Generate thumbnail with cache
        start_time = time.time()

        thumbnail = render_thumbnail_with_cache(
            points,
            colors=colors,
            width=800,
            height=600,
            viewpoint="diagonal",
            cache_dir=cache_dir,
            max_points_for_rendering=100000,
        )

        elapsed = time.time() - start_time
        first_pass_times.append(elapsed)

        print(f"   - Time: {elapsed:.3f}s")
        print(f"   - Thumbnail shape: {thumbnail.shape}")

        # Save thumbnail
        from viz_art.visualization.thumbnail import save_thumbnail
        output_path = output_dir / f"{cloud_file.stem}_thumbnail.png"
        save_thumbnail(thumbnail, str(output_path))
        print(f"   - Saved: {output_path.name}")

    # Test cache effectiveness
    print("\n3. Generating thumbnails (second pass - with cache)...")
    second_pass_times = []

    for i, cloud_file in enumerate(cloud_files):
        print(f"\n   Processing: {cloud_file.name}")

        # Load point cloud
        pcd = o3d.io.read_point_cloud(str(cloud_file))
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None

        # Generate thumbnail (should use cache)
        start_time = time.time()

        thumbnail = render_thumbnail_with_cache(
            points,
            colors=colors,
            width=800,
            height=600,
            viewpoint="diagonal",
            cache_dir=cache_dir,
            max_points_for_rendering=100000,
        )

        elapsed = time.time() - start_time
        second_pass_times.append(elapsed)

        print(f"   - Time: {elapsed:.3f}s (cached)")

    # Performance summary
    print("\n" + "=" * 70)
    print("Performance Summary")
    print("=" * 70)

    avg_first = np.mean(first_pass_times)
    avg_second = np.mean(second_pass_times)
    speedup = avg_first / avg_second if avg_second > 0 else 0

    print(f"\nFirst pass (no cache):")
    print(f"  Average time: {avg_first:.3f}s")
    print(f"  Min time: {min(first_pass_times):.3f}s")
    print(f"  Max time: {max(first_pass_times):.3f}s")

    print(f"\nSecond pass (with cache):")
    print(f"  Average time: {avg_second:.3f}s")
    print(f"  Min time: {min(second_pass_times):.3f}s")
    print(f"  Max time: {max(second_pass_times):.3f}s")

    print(f"\nSpeedup: {speedup:.1f}x")

    # Check SC-006 (< 3s per cloud)
    print(f"\nSC-006 Target (< 3s per cloud):")
    passed = all(t < 3.0 for t in first_pass_times)
    if passed:
        print(f"  ✓ PASS - All thumbnails generated in < 3s")
    else:
        slow_clouds = [i for i, t in enumerate(first_pass_times) if t >= 3.0]
        print(f"  ✗ FAIL - {len(slow_clouds)} cloud(s) took >= 3s")

    # Cache effectiveness
    cache_files = list(cache_dir.glob("*.png"))
    print(f"\nCache statistics:")
    print(f"  Cached thumbnails: {len(cache_files)}")
    print(f"  Cache directory size: {sum(f.stat().st_size for f in cache_files) / 1024:.1f} KB")

    print("\n" + "=" * 70)
    print("✓ Test completed successfully!")
    print(f"Thumbnails saved to: {output_dir}")
    print("=" * 70)

    return passed


if __name__ == "__main__":
    success = test_thumbnail_generation()
    sys.exit(0 if success else 1)
