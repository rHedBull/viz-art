"""Pipeline execution from YAML configuration.

This script demonstrates loading a pipeline from YAML config.

Usage:
    python examples/config_pipeline.py <config_file> <image_path>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from viz_art.config.loader import load_config
from viz_art.pipeline.base import Pipeline
from stages.image_loader import ImageLoader
from stages.simple_filter import SimpleFilter
from stages.image_resizer import ImageResizer


def main():
    """Run pipeline from YAML configuration."""
    if len(sys.argv) < 3:
        print("Usage: python examples/config_pipeline.py <config_file> <image_path>")
        print("\nExample:")
        print("  python examples/config_pipeline.py examples/configs/simple_pipeline.yaml tests/fixtures/sample_images/test_640x480.jpg")
        sys.exit(1)

    config_file = sys.argv[1]
    image_path = sys.argv[2]

    print("=" * 60)
    print("Config-Based Pipeline Execution")
    print("=" * 60)
    print(f"\nConfig: {config_file}")
    print(f"Image: {image_path}\n")

    # Load configuration
    print("Loading configuration...")
    try:
        config = load_config(config_file)
        print(f"✓ Loaded config: '{config.pipeline_name}'")
        print(f"  Stages: {len(config.stages)}")
        print(f"  Connections: {len(config.connections)}")
    except Exception as e:
        print(f"✗ Failed to load config: {str(e)}")
        sys.exit(1)

    # Create pipeline from config
    print("\nBuilding pipeline from config...")
    try:
        # Build stage registry
        stage_registry = {
            "ImageLoader": ImageLoader,
            "SimpleFilter": SimpleFilter,
            "ImageResizer": ImageResizer,
        }

        pipeline = Pipeline.from_config(config, stage_registry=stage_registry)
        print(f"✓ Created pipeline with {len(pipeline.stages)} stages")
        for stage in pipeline.stages:
            print(f"  - {stage.name}")
    except Exception as e:
        print(f"✗ Failed to build pipeline: {str(e)}")
        sys.exit(1)

    # Execute pipeline
    print("\nExecuting pipeline...")
    print("-" * 60)

    try:
        results = pipeline.run(image_path=image_path)

        if results["_status"] == "completed":
            print(f"\n✓ Pipeline completed successfully!")
            print(f"  Run ID: {results['_run_id']}")

            # Display outputs
            print("\nStage Outputs:")
            for i, stage in enumerate(pipeline.stages, 1):
                if stage.name in results:
                    stage_output = results[stage.name]
                    print(f"  {i}. {stage.name}:")
                    for key, value in stage_output.items():
                        if hasattr(value, "shape"):
                            print(f"     - {key}: shape {value.shape}")
                        else:
                            print(f"     - {key}: {value}")
        else:
            print(f"\n✗ Pipeline failed: {results.get('_error')}")
            sys.exit(1)

        print("\n" + "=" * 60)
        print("Execution complete!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Execution failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
