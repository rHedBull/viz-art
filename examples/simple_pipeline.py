"""Simple 2-stage pipeline example.

This script demonstrates basic pipeline usage:
1. Load an image
2. Apply a Gaussian blur filter
3. Display results

Usage:
    python examples/simple_pipeline.py <image_path>
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from viz_art.pipeline.base import Pipeline
from stages.image_loader import ImageLoader
from stages.simple_filter import SimpleFilter


def main():
    """Run the simple pipeline example."""
    if len(sys.argv) < 2:
        print("Usage: python examples/simple_pipeline.py <image_path>")
        print("\nExample:")
        print("  python examples/simple_pipeline.py tests/fixtures/sample_images/test.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    print("=" * 60)
    print("Simple 2-Stage Pipeline Example")
    print("=" * 60)
    print(f"\nInput image: {image_path}\n")

    # Create pipeline
    pipeline = Pipeline(name="simple-pipeline")
    print("✓ Created pipeline: 'simple-pipeline'")

    # Add stages
    loader = ImageLoader(name="loader", resize=None, color_mode="RGB")
    filter_stage = SimpleFilter(name="filter", kernel_size=7, sigma=1.0)

    pipeline.add_stage(loader)
    print("✓ Added stage: 'loader' (ImageLoader)")

    pipeline.add_stage(filter_stage)
    print("✓ Added stage: 'filter' (SimpleFilter with kernel=7)")

    # Connect stages
    pipeline.connect(
        source_stage="loader",
        target_stage="filter",
        output_key="image",
        input_key="image",
    )
    print("✓ Connected: loader.image → filter.image\n")

    # Execute pipeline
    print("Running pipeline...")
    print("-" * 60)

    try:
        results = pipeline.run(image_path=image_path)

        # Check if pipeline actually succeeded
        if results['_status'] != 'completed':
            print(f"\n✗ Pipeline failed!")
            print(f"  Status: {results['_status']}")
            print(f"  Error: {results.get('_error', 'Unknown error')}")
            sys.exit(1)

        print(f"\n✓ Pipeline completed successfully!")
        print(f"  Run ID: {results['_run_id']}")
        print(f"  Status: {results['_status']}")
        print(f"  Pipeline: {results['_pipeline_name']}")
        print(f"  Started: {results['_started_at']}")
        print(f"  Completed: {results['_completed_at']}")

        # Display stage outputs
        print("\nStage Outputs:")
        print("-" * 60)

        if "loader" in results:
            loader_out = results["loader"]
            print(f"\n1. loader:")
            print(f"   - image shape: {loader_out['metadata']['shape']}")
            print(f"   - image dtype: {loader_out['metadata']['dtype']}")
            print(f"   - source path: {loader_out['metadata']['path']}")

        if "filter" in results:
            filter_out = results["filter"]
            print(f"\n2. filter:")
            print(f"   - filtered_image shape: {filter_out['filtered_image'].shape}")
            print(f"   - filtered_image dtype: {filter_out['filtered_image'].dtype}")

        print("\n" + "=" * 60)
        print("Pipeline execution complete!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Pipeline failed: {str(e)}")
        if "_error" in results:
            print(f"  Error: {results['_error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
