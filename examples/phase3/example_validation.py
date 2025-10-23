"""Example: Ground Truth Validation

This example demonstrates:
1. Loading a COCO format ground truth dataset
2. Running validation against pipeline predictions
3. Calculating accuracy metrics (precision, recall, F1, mAP)
4. Performing error analysis with visualizations
5. Generating HTML error reports
"""

from pathlib import Path

from viz_art.validation import (
    AnnotationFormat,
    create_dataset,
    create_error_analyzer,
    create_metrics_calculator,
    validate_pipeline,
)


def main():
    """Run ground truth validation example."""
    print("=" * 80)
    print("Ground Truth Validation Example")
    print("=" * 80)

    # Setup paths
    output_dir = Path("output/validation_example")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load COCO format ground truth dataset
    print("\n1. Loading COCO format ground truth dataset...")

    # For this example, we'll use test fixtures
    dataset_root = Path("tests/fixtures/golden_datasets/bboxes")

    try:
        dataset = create_dataset(
            dataset_id="coco_sample",
            root_path=dataset_root,
            annotation_format=AnnotationFormat.COCO,
            annotation_files=[Path("coco_sample.json")],
            name="COCO Sample Dataset",
            metadata={"description": "Sample COCO annotations for testing"},
        )

        print(f"   ✓ Loaded dataset: {dataset.name}")
        print(f"   ✓ Sample count: {dataset.sample_count}")
        print(f"   ✓ Format: {dataset.annotation_format.value}")

    except FileNotFoundError as e:
        print(f"   ✗ Error loading dataset: {e}")
        print("   Note: Run this example after creating test fixtures")
        return

    # Step 2: Create sample predictions (simulating pipeline output)
    print("\n2. Creating sample predictions...")

    # Sample predictions in detection format
    sample_predictions = {
        "image_001.jpg": [
            {
                "bbox": [100, 100, 200, 200],  # [x, y, width, height]
                "class": "car",
                "score": 0.95,
            },
            {
                "bbox": [300, 150, 150, 180],
                "class": "person",
                "score": 0.87,
            },
        ],
        "image_002.jpg": [
            {
                "bbox": [50, 60, 180, 220],
                "class": "bicycle",
                "score": 0.82,
            },
        ],
        "image_003.jpg": [
            {
                "bbox": [200, 200, 100, 100],
                "class": "car",
                "score": 0.91,
            },
        ],
    }

    print(f"   ✓ Created predictions for {len(sample_predictions)} images")

    # Step 3: Validate pipeline against ground truth
    print("\n3. Running validation...")

    stage_outputs = {"detection": sample_predictions}

    try:
        validation_results = validate_pipeline(
            run_id="example_run_001",
            dataset=dataset,
            stage_outputs=stage_outputs,
            output_dir=output_dir,
        )

        print("   ✓ Validation complete")

        # Display metrics
        for stage_name, metrics in validation_results.items():
            print(f"\n   Stage: {stage_name}")
            print(f"   - Precision: {metrics['precision']:.2%}")
            print(f"   - Recall: {metrics['recall']:.2%}")
            print(f"   - F1 Score: {metrics['f1_score']:.2%}")

            if "mean_average_precision" in metrics and metrics["mean_average_precision"]:
                print(f"   - mAP: {metrics['mean_average_precision']:.2%}")

            print(f"   - Sample count: {metrics['sample_count']}")

    except Exception as e:
        print(f"   ✗ Validation error: {e}")
        print("   Note: Ground truth dataset may be empty or invalid")
        validation_results = {}

    # Step 4: Error analysis
    print("\n4. Performing error analysis...")

    analyzer = create_error_analyzer(output_dir / "error_analysis")

    # Get ground truth for comparison
    ground_truth_data = {}
    for sample_id, annotation in dataset.iter_samples():
        ground_truth_data[sample_id] = annotation

    # Find failures
    failures = analyzer.find_failures(
        run_id="example_run_001",
        predictions=sample_predictions,
        ground_truth=ground_truth_data,
        iou_threshold=0.5,
        confidence_threshold=0.3,
    )

    print(f"   ✓ Found {len(failures)} potential errors")

    # Categorize errors
    if failures:
        categories = analyzer.categorize_errors(failures)

        print(f"\n   Error breakdown:")
        print(f"   - False Positives: {len(categories['false_positive'])}")
        print(f"   - False Negatives: {len(categories['false_negative'])}")
        print(f"   - Misclassifications: {len(categories['misclassification'])}")

        # Visualize first failure (if any)
        if failures:
            first_failure = failures[0]
            sample_id = first_failure["sample_id"]

            print(f"\n   Generating visualization for: {sample_id}")

            viz_path = output_dir / "error_analysis" / f"{sample_id}_comparison.png"

            # Note: In a real scenario, you'd provide the actual image path
            analyzer.visualize_comparison(
                sample_id=sample_id,
                prediction=first_failure["prediction"],
                ground_truth=first_failure["ground_truth"],
                output_path=viz_path,
                image_path=None,  # Would be actual image path
            )

            print(f"   ✓ Saved visualization to: {viz_path}")

        # Generate HTML error report
        report_path = output_dir / "error_analysis_report.html"
        analyzer.generate_error_report(failures, report_path)

        print(f"   ✓ Generated error report: {report_path}")

    else:
        print("   ✓ No errors found (all predictions match ground truth)")

    # Step 5: Calculate detailed metrics with MetricsCalculator
    print("\n5. Calculating detailed metrics...")

    calculator = create_metrics_calculator(metric_type="detection")

    # Flatten predictions and ground truth for mAP calculation
    all_predictions = []
    all_ground_truths = []

    for sample_id in sample_predictions:
        if sample_id in ground_truth_data:
            all_predictions.extend(sample_predictions[sample_id])
            all_ground_truths.extend(ground_truth_data[sample_id])

    if all_predictions and all_ground_truths:
        try:
            detailed_metrics = calculator.calculate_mean_average_precision(
                predictions=all_predictions,
                ground_truth=all_ground_truths,
                iou_threshold=0.5,
            )

            print(f"   ✓ Detailed mAP: {detailed_metrics['mean_average_precision']:.2%}")

            if "per_class_ap" in detailed_metrics:
                print("\n   Per-class Average Precision:")
                for cls, ap in detailed_metrics["per_class_ap"].items():
                    print(f"   - {cls}: {ap:.2%}")

        except Exception as e:
            print(f"   ✗ Error calculating detailed metrics: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"✓ Dataset: {dataset.name} ({dataset.sample_count} samples)")
    print(f"✓ Predictions: {len(sample_predictions)} images")
    print(f"✓ Validation results saved to: {output_dir}")
    print(f"✓ Error analysis completed: {len(failures)} errors found")
    print("\nNext steps:")
    print("  - Review error analysis report in browser")
    print("  - Examine failure visualizations")
    print("  - Adjust detection thresholds if needed")
    print("  - Retrain model on difficult cases")


if __name__ == "__main__":
    main()
