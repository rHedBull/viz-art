"""Single-stage pipeline example.

Usage:
    python examples/single_stage_pipeline.py <image_path>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from viz_art.pipeline.base import Pipeline
from stages.image_loader import ImageLoader


def main():
    """Run single-stage pipeline test."""
    if len(sys.argv) < 2:
        print("Usage: python examples/single_stage_pipeline.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    print("Single-Stage Pipeline Test")
    print("=" * 60)

    # Create pipeline with single stage
    pipeline = Pipeline(name="single-stage")
    loader = ImageLoader(name="loader")
    pipeline.add_stage(loader)

    print("✓ Pipeline with single stage: loader")

    # Execute
    results = pipeline.run(image_path=image_path)

    if results["_status"] == "completed":
        print(f"✓ Pipeline completed: {results['loader']['metadata']['shape']}")
    else:
        print(f"✗ Failed: {results.get('_error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
