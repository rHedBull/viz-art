"""Test point cloud batch processing with HTML report generation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from viz_art.batch.runner import BatchRunner
from viz_art.pipeline.pipeline import Pipeline
from stages.pointcloud_loader import PointCloudLoader
from stages.pointcloud_validation import PointCloudValidationStage
from stages.pointcloud_thumbnail import PointCloudThumbnailStage
from stages.pointcloud_visualization import PointCloudVisualizationStage

def main():
    print("=" * 70)
    print("Point Cloud Batch Processing Test")
    print("=" * 70)

    # Setup paths
    test_data_dir = Path(__file__).parent.parent / "tests" / "fixtures" / "sample_pointclouds"
    output_dir = Path(__file__).parent / "output" / "pointcloud_batch"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nInput directory: {test_data_dir}")
    print(f"Output directory: {output_dir}")

    # Create pipeline
    print("\n1. Creating pipeline...")
    pipeline = Pipeline(name="pointcloud-batch")

    # Add stages
    loader = PointCloudLoader(
        name="loader",
        downsample_voxel_size=0.05,
        remove_outliers=True,
        outlier_neighbors=20,
        outlier_std_ratio=2.0,
    )

    validator = PointCloudValidationStage(
        name="validator",
        rules={
            "min_points": 10,
            "max_points": 10000000,
            "check_nan": True,
            "check_inf": True,
            "fail_fast": False,
        },
        raise_on_invalid=False,
    )

    thumbnail = PointCloudThumbnailStage(
        name="thumbnail",
        viewpoints=["diagonal"],
        width=400,
        height=300,
        point_size=2.0,
        background_color=[1.0, 1.0, 1.0],
    )

    viz = PointCloudVisualizationStage(
        name="visualization",
        color_mode="height",
        point_size=2.0,
        opacity=0.8,
        colorscale="Viridis",
        max_render_points=100000,
        output_html=False,  # Don't save individual HTML files
        output_json=False,
    )

    pipeline.add_stage(loader)
    pipeline.add_stage(validator)
    pipeline.add_stage(thumbnail)
    pipeline.add_stage(viz)

    # Add connections
    pipeline.add_connection(loader, validator, {"points": "points", "colors": "colors"})
    pipeline.add_connection(validator, thumbnail, {"points": "points", "colors": "colors"})
    pipeline.add_connection(validator, viz, {"points": "points", "colors": "colors"})

    print("   ✓ Pipeline created with 4 stages")

    # Find point cloud files
    print("\n2. Finding point cloud files...")
    pc_files = []
    for pattern in ["*.pcd", "*.ply"]:
        pc_files.extend(list(test_data_dir.glob(pattern)))

    # Filter out corrupted/empty files for cleaner report
    pc_files = [f for f in pc_files if not any(x in f.name for x in ["corrupted", "empty"])]

    print(f"   Found {len(pc_files)} point clouds:")
    for f in pc_files:
        print(f"   - {f.name}")

    # Create batch runner
    print("\n3. Running batch processing...")
    runner = BatchRunner(
        pipeline=pipeline,
        output_dir=output_dir,
        continue_on_error=True,
    )

    # Run batch
    report = runner.run_batch(
        inputs=[{"input_path": str(f)} for f in pc_files],
        save_outputs=True,
        max_samples=100,
    )

    print(f"\n   ✓ Batch processing complete!")
    print(f"   Total: {report.total_samples}")
    print(f"   Successful: {report.successful_samples}")
    print(f"   Failed: {report.failed_samples}")
    print(f"   Duration: {report.duration:.2f}s")

    # Generate HTML report
    print("\n4. Generating HTML report...")
    report_path = output_dir / "pointcloud_report.html"
    report.save_html_report(report_path)

    print(f"   ✓ Report saved to: {report_path}")

    print("\n" + "=" * 70)
    print("✓ Test completed successfully!")
    print(f"Open report: file://{report_path.absolute()}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
