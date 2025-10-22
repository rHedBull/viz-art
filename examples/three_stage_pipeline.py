"""Three-stage pipeline example to test data flow.

Usage:
    python examples/three_stage_pipeline.py <image_path>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from viz_art.pipeline.base import Pipeline
from stages.image_loader import ImageLoader
from stages.simple_filter import SimpleFilter
from stages.image_resizer import ImageResizer


def main():
    """Run 3-stage pipeline test."""
    if len(sys.argv) < 2:
        print("Usage: python examples/three_stage_pipeline.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    print("=" * 60)
    print("3-Stage Pipeline Test: Data Flow Validation")
    print("=" * 60)
    print(f"\nInput: {image_path}\n")

    # Create pipeline
    pipeline = Pipeline(name="three-stage-test")

    # Add 3 stages
    loader = ImageLoader(name="loader")
    filter_stage = SimpleFilter(name="filter", kernel_size=5)
    resizer = ImageResizer(name="resizer", width=320, height=240)

    pipeline.add_stage(loader)
    pipeline.add_stage(filter_stage)
    pipeline.add_stage(resizer)

    # Connect: loader → filter → resizer
    pipeline.connect("loader", "filter", "image", "image")
    pipeline.connect("filter", "resizer", "filtered_image", "filtered_image")

    print("✓ Pipeline: loader → filter → resizer")
    print("  Connection 1: loader.image → filter.image")
    print("  Connection 2: filter.filtered_image → resizer.filtered_image\n")

    # Execute
    results = pipeline.run(image_path=image_path)

    if results["_status"] == "completed":
        print("✓ Pipeline completed successfully!\n")
        print("Data Flow Verification:")
        print(f"  Stage 1 (loader): {results['loader']['metadata']['shape']}")
        print(f"  Stage 2 (filter): {results['filter']['filtered_image'].shape}")
        print(f"  Stage 3 (resizer): {results['resizer']['resized_image'].shape}")
        print("\n✓ Data flowed correctly: stage 1 → stage 2 → stage 3")
    else:
        print(f"✗ Pipeline failed: {results.get('_error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
