"""CLI tool for ground truth validation.

Usage:
    viz-art-validate --run-id abc-123 --dataset coco_val
"""

import argparse
import sys
from pathlib import Path


def validate_cli():
    """Run validation from command line."""
    parser = argparse.ArgumentParser(
        description="Validate viz-art pipeline outputs against ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate a run against COCO dataset
  viz-art-validate --run-id abc-123 --dataset coco_val

  # Generate HTML error analysis report
  viz-art-validate --run-id abc-123 --dataset coco_val --output errors.html
        """,
    )

    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Run ID to validate",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Ground truth dataset ID",
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("output/ground_truth"),
        help="Root directory for ground truth datasets (default: output/ground_truth)",
    )

    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("output/runs"),
        help="Directory containing run outputs (default: output/runs)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Generate HTML error report to this file",
    )

    args = parser.parse_args()

    # Import here to avoid slow startup
    from viz_art.validation import create_dataset, validate_pipeline, create_error_analyzer
    from viz_art.types.monitoring import AnnotationFormat

    # Load ground truth dataset
    # Note: In a real implementation, you'd have a dataset registry
    # For now, we assume COCO format in dataset_root/dataset_id/
    dataset_path = args.dataset_root / args.dataset
    annotation_file = dataset_path / "annotations.json"

    if not dataset_path.exists():
        print(f"Error: Dataset directory not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    if not annotation_file.exists():
        print(f"Error: Annotation file not found: {annotation_file}", file=sys.stderr)
        print(f"Expected: {annotation_file}", file=sys.stderr)
        sys.exit(1)

    try:
        dataset = create_dataset(
            dataset_id=args.dataset,
            root_path=dataset_path,
            annotation_format=AnnotationFormat.COCO,
            annotation_files=[Path("annotations.json")],
        )
        print(f"Loaded dataset: {dataset.name} ({dataset.sample_count} samples)")
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        sys.exit(1)

    # Load run outputs
    run_path = args.run_dir / args.run_id
    if not run_path.exists():
        print(f"Error: Run directory not found: {run_path}", file=sys.stderr)
        sys.exit(1)

    # Load stage outputs from run directory
    # Note: This is a simplified implementation
    # In production, you'd load from OutputSaver structure
    stage_outputs = {}

    # Try to find output files
    for stage_dir in run_path.glob("stages/*"):
        if stage_dir.is_dir():
            stage_name = stage_dir.name
            # Load predictions from stage directory
            # This is very simplified - in reality you'd parse the actual output format
            stage_outputs[stage_name] = {}

    if not stage_outputs:
        print(f"Warning: No stage outputs found in {run_path}/stages/", file=sys.stderr)
        print("Validation may not work correctly", file=sys.stderr)

    # Run validation
    try:
        results = validate_pipeline(
            run_id=args.run_id,
            dataset=dataset,
            stage_outputs=stage_outputs,
            output_dir=run_path / "validation",
        )

        print(f"\nValidation Results for Run: {args.run_id}")
        print("=" * 60)

        for stage_name, metrics in results.items():
            print(f"\nStage: {stage_name}")
            print(f"  Precision:    {metrics.get('precision', 0):.2%}")
            print(f"  Recall:       {metrics.get('recall', 0):.2%}")
            print(f"  F1 Score:     {metrics.get('f1_score', 0):.2%}")

            if metrics.get("mean_average_precision"):
                print(f"  mAP:          {metrics['mean_average_precision']:.2%}")

            if metrics.get("iou_mean"):
                print(f"  Mean IoU:     {metrics['iou_mean']:.2%}")

            print(f"  Sample Count: {metrics.get('sample_count', 0)}")

    except Exception as e:
        print(f"Error during validation: {e}", file=sys.stderr)
        sys.exit(1)

    # Generate error analysis report if requested
    if args.output:
        try:
            analyzer = create_error_analyzer(run_path / "error_analysis")

            # Get ground truth data
            ground_truth_data = {}
            for sample_id, annotation in dataset.iter_samples():
                ground_truth_data[sample_id] = annotation

            # Find failures
            failures = []
            for stage_name, predictions in stage_outputs.items():
                stage_failures = analyzer.find_failures(
                    run_id=args.run_id,
                    predictions=predictions,
                    ground_truth=ground_truth_data,
                )
                failures.extend(stage_failures)

            # Generate report
            analyzer.generate_error_report(failures, args.output)
            print(f"\nError analysis report saved to: {args.output}")

        except Exception as e:
            print(f"\nWarning: Could not generate error report: {e}", file=sys.stderr)


if __name__ == "__main__":
    validate_cli()
