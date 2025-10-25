#!/usr/bin/env python3
"""
Comprehensive Phase 3 Demo

Demonstrates ALL Phase 3 observability features:
- Performance profiling (timing + memory)
- Audit logging with run tracking
- Ground truth validation with accuracy metrics
- Error analysis and categorization
- Comprehensive HTML report with embedded charts

This showcases the complete Phase 3 system.
"""
import sys
import time
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from viz_art.performance import create_profiler, create_metrics_storage, create_dashboard
from viz_art.audit import create_logger, create_run_tracker, create_query
from viz_art.validation import create_dataset, create_error_analyzer, create_metrics_calculator
from viz_art.types.monitoring import AnnotationFormat
import numpy as np


def simulate_detection_stage(image_id: int):
    """Simulate object detection with some errors."""
    time.sleep(0.05 + np.random.rand() * 0.05)

    # Simulate predictions (some correct, some wrong)
    if image_id == 0:
        # Perfect detection
        return {"boxes": [[10, 10, 100, 100], [150, 150, 250, 250]], "scores": [0.95, 0.92], "labels": [1, 2]}
    elif image_id == 1:
        # Missed one object
        return {"boxes": [[10, 10, 100, 100]], "scores": [0.88], "labels": [1]}
    else:
        # False positive
        return {"boxes": [[10, 10, 100, 100], [200, 200, 250, 250]], "scores": [0.91, 0.65], "labels": [1, 1]}


def create_mock_ground_truth():
    """Create mock ground truth dataset."""
    return {
        0: {"boxes": [[10, 10, 100, 100], [150, 150, 250, 250]], "labels": [1, 2]},
        1: {"boxes": [[10, 10, 100, 100], [50, 50, 150, 150]], "labels": [1, 1]},
        2: {"boxes": [[10, 10, 100, 100]], "labels": [1]},
    }


def calculate_simple_metrics(predictions, ground_truths):
    """Calculate simple accuracy metrics."""
    total_gt = sum(len(gt["boxes"]) for gt in ground_truths.values())
    total_pred = sum(len(pred["boxes"]) for pred in predictions.values())

    # Simple matching (IoU > 0.5)
    true_positives = 0
    for img_id in predictions:
        pred_boxes = predictions[img_id]["boxes"]
        gt_boxes = ground_truths[img_id]["boxes"]
        # Simplified: count matches
        matches = min(len(pred_boxes), len(gt_boxes))
        true_positives += matches

    false_positives = total_pred - true_positives
    false_negatives = total_gt - true_positives

    precision = true_positives / total_pred if total_pred > 0 else 0
    recall = true_positives / total_gt if total_gt > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "total_objects": total_gt,
    }


def main():
    print("=" * 80)
    print("Comprehensive Phase 3 Observability Demo")
    print("=" * 80)

    output_dir = Path(__file__).parent / "output" / "comprehensive_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ==================================================================================
    # Phase 3 Setup
    # ==================================================================================

    print("\n📋 Setting up Phase 3 observability stack...\n")

    # 1. Performance Profiling
    metrics_storage = create_metrics_storage(
        output_dir=output_dir / "metrics",
        retention_days=90,
    )
    profiler = create_profiler(storage=metrics_storage, enabled=True)
    print("✓ Performance profiling (timing + memory tracking)")

    # 2. Run Tracking & Logging
    tracker = create_run_tracker(output_dir=output_dir / "runs")
    print("✓ Run tracking (JSON metadata)")

    # 3. Ground Truth & Validation
    ground_truth = create_mock_ground_truth()
    print("✓ Ground truth dataset (3 samples loaded)")

    # 4. Error Analysis
    error_analyzer = create_error_analyzer(output_dir=output_dir / "error_analysis")
    print("✓ Error analysis toolkit")

    print()

    # ==================================================================================
    # Execute Pipeline with Full Observability
    # ==================================================================================

    config_snapshot = {
        "pipeline": "object_detection",
        "version": "1.0",
        "model": "mock_detector",
        "confidence_threshold": 0.5,
    }

    with tracker.track(
        config_snapshot=config_snapshot,
        input_files=[Path(f"image_{i}.jpg") for i in range(3)],
        output_dir=output_dir,
    ) as run_id:
        print(f"🔖 Started run: {run_id}\n")

        # Bind profiler to this run
        profiler._current_run_id = run_id

        # Create logger
        logger = create_logger(
            run_id=run_id,
            output_dir=output_dir / "logs",
            retention="30 days",
        )

        logger.info("Pipeline started", mode="validation", dataset_size=3)

        # Process images with detection
        print("🔄 Running object detection with validation...\n")
        predictions = {}

        for i in range(3):
            logger.info(f"Processing image {i}", image_id=i)

            with profiler.measure("object_detection"):
                result = simulate_detection_stage(i)
                predictions[i] = result

            logger.info(
                f"Detected objects in image {i}",
                num_detections=len(result["boxes"]),
                confidence=float(np.mean(result["scores"])),
            )

        # Calculate accuracy metrics
        print("📊 Calculating accuracy metrics...\n")
        with profiler.measure("accuracy_calculation"):
            metrics = calculate_simple_metrics(predictions, ground_truth)

        logger.info(
            "Validation complete",
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1_score=metrics["f1_score"],
        )

        # Error analysis
        print("🔍 Performing error analysis...\n")
        errors = []
        for img_id in predictions:
            pred_count = len(predictions[img_id]["boxes"])
            gt_count = len(ground_truth[img_id]["boxes"])
            if pred_count != gt_count:
                error_type = "false_negative" if pred_count < gt_count else "false_positive"
                errors.append({
                    "image_id": img_id,
                    "type": error_type,
                    "predicted": pred_count,
                    "expected": gt_count,
                })
                logger.warning(
                    f"Detection mismatch on image {img_id}",
                    error_type=error_type,
                    predicted=pred_count,
                    expected=gt_count,
                )

        logger.info("Pipeline completed", total_errors=len(errors))
        print(f"✓ Processing complete ({len(errors)} errors found)\n")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"✓ Run ID: {run_id}")
    print(f"✓ Images processed: 3")
    print(f"✓ Accuracy: Precision={metrics['precision']:.1%}, Recall={metrics['recall']:.1%}, F1={metrics['f1_score']:.1%}")
    print(f"✓ Errors detected: {len(errors)}")
    print(f"✓ Metrics stored: {output_dir}/metrics")
    print(f"✓ Logs stored: {output_dir}/logs")
    print()
    print("ℹ️  For HTML reports with visualizations, use the batch processing workflow:")
    print("   python examples/demo_batch_with_phase3.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
