#!/usr/bin/env python3
"""Test point cloud batch processing with HTML report generation.

This script demonstrates batch processing of point clouds:
1. Load configuration
2. Create pipeline from config
3. Process batch of point clouds
4. Generate HTML report
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "examples"))

from viz_art.config.loader import load_config
from viz_art.pipeline.base import Pipeline
from viz_art.batch.processor import BatchProcessor
from viz_art.batch.reporter import HTMLReporter

# Import example stages
from stages.pointcloud_loader import PointCloudLoader
from stages.pointcloud_validation import PointCloudValidationStage
from stages.pointcloud_thumbnail import PointCloudThumbnailStage

# Create stage registry
STAGE_REGISTRY = {
    "PointCloudLoader": PointCloudLoader,
    "PointCloudValidationStage": PointCloudValidationStage,
    "PointCloudThumbnailStage": PointCloudThumbnailStage,
}


def main():
    print("=" * 70)
    print("VIZ-ART POINT CLOUD BATCH PROCESSING TEST")
    print("=" * 70)

    # Load configuration
    config_path = Path(__file__).parent / "pointcloud_batch_config.yaml"
    print(f"\n1. Loading configuration from: {config_path}")
    config = load_config(config_path)
    print(f"   ✓ Pipeline: {config.pipeline_name}")
    print(f"   ✓ Stages: {len(config.stages)}")

    # Create pipeline
    print("\n2. Creating pipeline from configuration...")
    pipeline = Pipeline.from_config(config, stage_registry=STAGE_REGISTRY)
    print(f"   ✓ Pipeline created with {len(pipeline._stages)} stages")

    # Setup batch processor
    print("\n3. Setting up batch processor...")
    processor = BatchProcessor(pipeline, config.batch_config)
    print(f"   ✓ Input directory: {config.batch_config.input_dir}")
    print(f"   ✓ Output directory: {config.batch_config.output_dir}")

    # Run batch processing
    print("\n4. Starting batch processing...")
    print("   (This may take a few moments...)")
    batch_result = processor.run()

    # Generate HTML report
    print("\n5. Generating HTML report...")
    reporter = HTMLReporter()
    output_dir = Path(config.batch_config.output_dir)
    report_path = output_dir / config.batch_config.report_output

    report_path = reporter.generate(
        batch_result,
        report_path,
        pipeline_name=config.pipeline_name,
    )

    # Print summary
    print("\n" + "=" * 70)
    print("BATCH PROCESSING COMPLETE!")
    print("=" * 70)
    print(f"Total Files:    {batch_result.total_files}")
    print(f"Successful:     {batch_result.successful}")
    print(f"Failed:         {batch_result.failed}")

    duration = (batch_result.completed_at - batch_result.started_at).total_seconds()
    print(f"Duration:       {duration:.2f} seconds")

    if batch_result.total_files > 0:
        success_rate = (batch_result.successful / batch_result.total_files) * 100
        print(f"Success Rate:   {success_rate:.1f}%")

    print(f"\nHTML Report:    {report_path}")
    print(f"\nTo view the report, open: file://{report_path.absolute()}")
    print("=" * 70)

    return 0 if batch_result.failed == 0 else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
