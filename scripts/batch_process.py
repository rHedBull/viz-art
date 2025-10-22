#!/usr/bin/env python3
"""Batch processing CLI script for viz-art.

This script processes multiple images from a directory through a configured
pipeline and generates an HTML report of the results.

Usage:
    python scripts/batch_process.py --config configs/my_config.yaml

    python scripts/batch_process.py \\
        --config configs/my_config.yaml \\
        --input-dir ./data/images \\
        --output-dir ./output/results

Examples:
    # Use config file settings
    python scripts/batch_process.py --config configs/batch_config.yaml

    # Override input/output directories
    python scripts/batch_process.py \\
        --config configs/pipeline.yaml \\
        --input-dir ./test_images \\
        --output-dir ./test_output
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

from viz_art.config.loader import load_config
from viz_art.config.schema import BatchConfigItem
from viz_art.pipeline.base import Pipeline
from viz_art.batch.processor import BatchProcessor
from viz_art.batch.reporter import HTMLReporter


def setup_logging(verbose: bool = False):
    """Setup logging configuration.

    Args:
        verbose: Enable debug logging if True
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Batch process images through a vision pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to pipeline configuration YAML file",
    )

    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        help="Input directory containing images (overrides config)",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        help="Output directory for results (overrides config)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose debug logging",
    )

    return parser.parse_args()


def validate_config_has_batch(config) -> BatchConfigItem:
    """Validate that config has batch_config section.

    Args:
        config: Loaded PipelineConfig

    Returns:
        BatchConfigItem from config

    Raises:
        ValueError: If batch_config is missing
    """
    if config.batch_config is None:
        raise ValueError(
            "Configuration file missing 'batch_config' section. "
            "Please add batch processing configuration to your YAML file."
        )
    return config.batch_config


def override_batch_config(
    batch_config: BatchConfigItem,
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> BatchConfigItem:
    """Override batch config with CLI arguments.

    Args:
        batch_config: Original batch config from file
        input_dir: Override input directory
        output_dir: Override output directory

    Returns:
        New BatchConfigItem with overrides applied
    """
    # Convert to dict, apply overrides, recreate model
    config_dict = batch_config.model_dump()

    if input_dir is not None:
        config_dict["input_dir"] = input_dir
        logging.info(f"Overriding input_dir: {input_dir}")

    if output_dir is not None:
        config_dict["output_dir"] = output_dir
        logging.info(f"Overriding output_dir: {output_dir}")

    return BatchConfigItem(**config_dict)


def print_summary(batch_result):
    """Print batch processing summary to console.

    Args:
        batch_result: BatchResult from processor
    """
    print("\n" + "=" * 60)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Batch ID:       {batch_result.batch_id}")
    print(f"Total Images:   {batch_result.total_files}")
    print(f"Successful:     {batch_result.successful}")
    print(f"Failed:         {batch_result.failed}")

    duration = (batch_result.completed_at - batch_result.started_at).total_seconds()
    print(f"Duration:       {duration:.2f} seconds")

    if batch_result.total_files > 0:
        success_rate = (batch_result.successful / batch_result.total_files) * 100
        print(f"Success Rate:   {success_rate:.1f}%")

    print(f"\nHTML Report:    {batch_result.report_path}")
    print("=" * 60)


def main():
    """Main entry point for batch processing script."""
    args = parse_arguments()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)

    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config_path = Path(args.config)

        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            sys.exit(1)

        config = load_config(config_path)
        logger.info(f"Configuration loaded: {config.pipeline_name}")

        # Validate batch config exists
        batch_config = validate_config_has_batch(config)

        # Override with CLI arguments if provided
        batch_config = override_batch_config(
            batch_config,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
        )

        # Validate input directory
        input_path = Path(batch_config.input_dir)
        if not input_path.exists():
            logger.error(f"Input directory does not exist: {input_path}")
            sys.exit(1)

        # Create pipeline from config
        logger.info("Creating pipeline from configuration...")
        pipeline = Pipeline.from_config(config)
        logger.info(f"Pipeline created with {len(pipeline._stages)} stages")

        # Create batch processor
        logger.info("Initializing batch processor...")
        processor = BatchProcessor(pipeline, batch_config)

        # Execute batch processing
        logger.info("Starting batch processing...")
        print(f"\nProcessing images from: {batch_config.input_dir}")
        print(f"Output directory: {batch_config.output_dir}\n")

        batch_result = processor.run()

        # Generate HTML report
        logger.info("Generating HTML report...")
        reporter = HTMLReporter()

        report_path = Path(batch_config.output_dir) / batch_config.report_output
        report_path = reporter.generate(
            batch_result,
            report_path,
            pipeline_name=config.pipeline_name,
        )

        # Update batch result with report path
        # Note: BatchResult is frozen, so we create a new one
        from dataclasses import replace

        batch_result = replace(batch_result, report_path=str(report_path))

        # Print summary
        print_summary(batch_result)

        # Exit with appropriate code
        if batch_result.failed > 0:
            logger.warning(f"{batch_result.failed} images failed to process")
            sys.exit(2)  # Partial success
        else:
            logger.info("All images processed successfully")
            sys.exit(0)

    except KeyboardInterrupt:
        logger.warning("Batch processing interrupted by user")
        sys.exit(130)

    except Exception as e:
        logger.error(f"Batch processing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
