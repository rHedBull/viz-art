"""Test script for stage-by-stage output inspection with multimodal data.

This script demonstrates:
1. Running a pipeline with both image and point cloud stages
2. Inspecting intermediate outputs from each stage
3. Viewing both 2D images and 3D point clouds
4. Generating thumbnails for batch reports

Usage:
    python examples/test_stage_outputs.py
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from viz_art.pipeline.base import Pipeline
from viz_art.pipeline.output_saver import OutputSaver

# Import example stages
from examples.stages.pointcloud_loader import PointCloudLoader
from examples.stages.pointcloud_validation import PointCloudValidationStage
from examples.stages.pointcloud_visualization import PointCloudVisualizationStage
from examples.stages.pointcloud_thumbnail import PointCloudThumbnailStage


def create_sample_pipeline():
    """Create a sample multi-stage pipeline for point clouds."""
    # Stage 1: Load point cloud
    loader = PointCloudLoader(
        name="loader",
        downsample_voxel_size=0.05,
        remove_outliers=True,
    )

    # Stage 2: Validate point cloud
    validator = PointCloudValidationStage(
        name="validator",
        min_points=100,
        check_nan=True,
        check_inf=True,
    )

    # Stage 3: Generate visualization
    visualizer = PointCloudVisualizationStage(
        name="visualizer",
        color_mode="height",
    )

    # Stage 4: Generate thumbnails
    thumbnailer = PointCloudThumbnailStage(
        name="thumbnailer",
        viewpoints=("front", "top", "diagonal"),
        width=400,
        height=300,
    )

    return [loader, validator, visualizer, thumbnailer]


def test_stage_by_stage_execution():
    """Test executing stages and inspecting outputs."""
    print("\n=== Test: Stage-by-Stage Execution ===")

    # Find sample point cloud
    fixtures_dir = Path(__file__).parent.parent / "tests" / "fixtures" / "sample_pointclouds"
    pcd_files = list(fixtures_dir.glob("*.pcd")) + list(fixtures_dir.glob("*.ply"))

    if not pcd_files:
        print("ERROR: No sample point cloud files found")
        return False

    pcd_path = pcd_files[0]
    print(f"Input: {pcd_path.name}")

    # Create pipeline stages
    stages = create_sample_pipeline()

    # Create output saver
    output_dir = Path(__file__).parent.parent / "output" / "stage_outputs"
    saver = OutputSaver(
        output_mode="sample",
        save_config={
            "enabled": True,
            "stages": ["all"],
            "max_samples": 10,
        },
        output_dir=str(output_dir),
    )

    # Execute stages one by one
    print("\n" + "=" * 60)
    current_data = {"pointcloud_path": str(pcd_path)}

    for i, stage in enumerate(stages):
        print(f"\nStage {i+1}/{len(stages)}: {stage.name}")
        print("-" * 60)

        try:
            # Run stage
            result = stage.run(current_data)

            # Print output summary
            print(f"✓ Stage completed successfully")
            print(f"  Output keys: {list(result.keys())}")

            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    print(f"  - {key}: NumPy array {value.shape} ({value.dtype})")
                elif isinstance(value, dict):
                    print(f"  - {key}: Dictionary with {len(value)} keys")
                elif isinstance(value, (int, float)):
                    print(f"  - {key}: {value}")
                else:
                    print(f"  - {key}: {type(value).__name__}")

            # Save outputs
            saved_paths = saver.save_stage_output(
                run_id="test_run",
                stage_name=stage.name,
                filename=pcd_path.name,
                outputs=result,
                image_index=0,
            )

            if saved_paths:
                print(f"  Saved outputs:")
                for key, path in saved_paths.items():
                    print(f"    - {key}: {path}")

            # Pass relevant outputs to next stage
            current_data = result

        except Exception as e:
            print(f"✗ Stage failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    print("\n" + "=" * 60)
    print(f"✓ All stages completed successfully")
    print(f"\nOutputs saved to: {output_dir}")

    return True


def test_thumbnail_generation():
    """Test thumbnail generation for multiple viewpoints."""
    print("\n=== Test: Thumbnail Generation ===")

    # Create synthetic point cloud
    n_points = 5000
    t = np.linspace(0, 4 * np.pi, n_points)

    # Helix shape
    x = np.cos(t)
    y = np.sin(t)
    z = t / (2 * np.pi)

    points = np.column_stack([x, y, z]).astype(np.float32)

    # Generate colors (gradient)
    colors = np.zeros((n_points, 3), dtype=np.float32)
    colors[:, 0] = np.linspace(0, 1, n_points)  # Red gradient
    colors[:, 2] = np.linspace(1, 0, n_points)  # Blue gradient

    # Create thumbnail stage
    output_dir = Path(__file__).parent.parent / "output" / "thumbnails"

    thumbnailer = PointCloudThumbnailStage(
        name="thumbnailer",
        viewpoints=("front", "top", "side", "diagonal"),
        width=400,
        height=300,
        point_size=2.0,
        output_dir=str(output_dir),
    )

    # Generate thumbnails
    print("Generating thumbnails from multiple viewpoints...")

    try:
        result = thumbnailer.run({"points": points, "colors": colors})

        print(f"✓ Generated {len(result['thumbnails'])} thumbnails")

        for viewpoint in result['viewpoints']:
            print(f"  - {viewpoint}: {result['thumbnail_paths'].get(viewpoint, 'not saved')}")

        return True

    except Exception as e:
        print(f"✗ Thumbnail generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_output_saver_thumbnail():
    """Test OutputSaver thumbnail generation method."""
    print("\n=== Test: OutputSaver Thumbnail Generation ===")

    # Create synthetic point cloud
    n_points = 1000
    points = np.random.randn(n_points, 3).astype(np.float32) * 2

    # Create colors
    colors = np.random.rand(n_points, 3).astype(np.float32)

    # Create output saver
    output_dir = Path(__file__).parent.parent / "output" / "saver_thumbnails"
    saver = OutputSaver(
        output_mode="sample",
        output_dir=str(output_dir),
    )

    # Generate thumbnail
    thumbnail_path = output_dir / "test_thumbnail.png"

    print(f"Generating thumbnail to: {thumbnail_path}")

    try:
        img = saver.generate_thumbnail(
            points,
            colors=colors,
            output_path=thumbnail_path,
            width=400,
            height=300,
            viewpoint="diagonal",
        )

        if img is not None:
            print(f"✓ Generated thumbnail: {img.shape}")
            print(f"  Saved to: {thumbnail_path}")
            return True
        else:
            print("✗ Thumbnail generation returned None")
            return False

    except Exception as e:
        print(f"✗ Thumbnail generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all stage output tests."""
    print("=" * 60)
    print("Stage-by-Stage Output Inspection Tests")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Stage-by-Stage Execution", test_stage_by_stage_execution()))
    results.append(("Thumbnail Generation", test_thumbnail_generation()))
    results.append(("OutputSaver Thumbnail", test_output_saver_thumbnail()))

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
    else:
        print("\n✗ Some tests failed")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
