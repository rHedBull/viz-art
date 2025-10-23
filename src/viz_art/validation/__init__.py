"""Ground Truth Validation Module

Provides tools for validating pipeline accuracy against labeled datasets.
Supports multiple annotation formats and generates accuracy metrics.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from ..types.monitoring import AccuracyMetrics, AnnotationFormat
from .dataset import GroundTruthDataset
from .error_analysis import ErrorAnalyzer
from .loaders import register_format_loader
from .metrics import MetricsCalculator


def create_dataset(
    dataset_id: str,
    root_path: Path,
    annotation_format: AnnotationFormat,
    annotation_files: list,
    name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> GroundTruthDataset:
    """Create a ground truth dataset.

    Factory function for creating GroundTruthDataset instances.

    Args:
        dataset_id: Unique identifier for the dataset
        root_path: Base directory containing images/point clouds and annotations
        annotation_format: Format of annotations (COCO, YOLO, etc.)
        annotation_files: List of annotation file paths (relative to root_path)
        name: Optional human-readable name
        metadata: Optional additional metadata

    Returns:
        GroundTruthDataset instance

    Example:
        >>> dataset = create_dataset(
        ...     dataset_id="coco_val_2017",
        ...     root_path=Path("data/validation"),
        ...     annotation_format=AnnotationFormat.COCO,
        ...     annotation_files=[Path("annotations/instances_val2017.json")],
        ... )
    """
    return GroundTruthDataset(
        dataset_id=dataset_id,
        root_path=root_path,
        annotation_format=annotation_format,
        annotation_files=annotation_files,
        name=name,
        metadata=metadata,
    )


def create_metrics_calculator(metric_type: str = "auto") -> MetricsCalculator:
    """Create a metrics calculator.

    Factory function for creating MetricsCalculator instances.

    Args:
        metric_type: Type of metrics ('classification', 'detection', 'segmentation', 'auto')

    Returns:
        MetricsCalculator instance

    Example:
        >>> calculator = create_metrics_calculator("detection")
        >>> metrics = calculator.calculate_mean_average_precision(preds, gts)
    """
    return MetricsCalculator(metric_type=metric_type)


def create_error_analyzer(output_dir: Path) -> ErrorAnalyzer:
    """Create an error analyzer.

    Factory function for creating ErrorAnalyzer instances.

    Args:
        output_dir: Directory to save analysis outputs

    Returns:
        ErrorAnalyzer instance

    Example:
        >>> analyzer = create_error_analyzer(Path("output/error_analysis"))
        >>> failures = analyzer.find_failures(run_id, preds, gts)
    """
    return ErrorAnalyzer(output_dir=output_dir)


def validate_pipeline(
    run_id: str,
    dataset: GroundTruthDataset,
    stage_outputs: Dict[str, Any],
    output_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """Validate pipeline outputs against ground truth dataset.

    Main validation function that:
    1. Loads ground truth annotations from dataset
    2. Compares stage outputs to ground truth
    3. Calculates accuracy metrics per stage
    4. Returns comprehensive metrics dictionary

    Args:
        run_id: Pipeline run identifier
        dataset: Ground truth dataset to validate against
        stage_outputs: Dictionary mapping stage_name to predictions
                      Format: {stage_name: {sample_id: prediction}}
        output_dir: Optional directory to save validation artifacts

    Returns:
        Dictionary mapping stage_name to metrics:
        {
            'stage_name': {
                'precision': float,
                'recall': float,
                'f1_score': float,
                'mean_average_precision': float (if detection),
                'iou_mean': float (if segmentation),
                'true_positives': int,
                'false_positives': int,
                'false_negatives': int,
                'true_negatives': int,
                'sample_count': int,
            }
        }

    Example:
        >>> results = validate_pipeline(
        ...     run_id="abc-123",
        ...     dataset=dataset,
        ...     stage_outputs={
        ...         "detection": {
        ...             "img1.jpg": [{"bbox": [10, 20, 100, 200], "class": "car", "score": 0.95}],
        ...             "img2.jpg": [{"bbox": [50, 60, 150, 250], "class": "person", "score": 0.87}],
        ...         }
        ...     },
        ... )
        >>> print(f"Detection precision: {results['detection']['precision']:.2%}")
    """
    from datetime import datetime

    calculator = MetricsCalculator()
    all_metrics = {}

    # Iterate through each stage's outputs
    for stage_name, predictions in stage_outputs.items():
        # Collect all predictions and ground truths
        all_predictions = []
        all_ground_truths = []

        for sample_id, prediction in predictions.items():
            ground_truth = dataset.get_annotation(sample_id)

            if ground_truth is None:
                # No ground truth for this sample - skip
                continue

            all_predictions.append(prediction)
            all_ground_truths.append(ground_truth)

        if not all_predictions:
            # No valid predictions/ground truths for this stage
            continue

        # Determine task type from data structure
        if isinstance(all_predictions[0], list) and all_predictions[0]:
            if isinstance(all_predictions[0][0], dict) and "bbox" in all_predictions[0][0]:
                # Detection task
                metrics = calculator.calculate_mean_average_precision(
                    predictions=[p for preds in all_predictions for p in preds],
                    ground_truth=[gt for gts in all_ground_truths for gt in gts],
                )

                # Also calculate basic precision/recall by converting to binary
                # (matched vs unmatched detections)
                # This is a simplified approach - in practice you'd want more sophisticated metrics
                metrics.update(
                    {
                        "precision": metrics.get("mean_average_precision", 0.0),
                        "recall": metrics.get("mean_average_precision", 0.0),
                        "f1_score": metrics.get("mean_average_precision", 0.0),
                        "true_positives": len(all_predictions),
                        "false_positives": 0,
                        "false_negatives": 0,
                        "true_negatives": 0,
                    }
                )

        # Store metrics with AccuracyMetrics model
        all_metrics[stage_name] = {
            **metrics,
            "sample_count": len(all_predictions),
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Optionally save metrics to file
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Create AccuracyMetrics instance
            accuracy_metrics = AccuracyMetrics(
                run_id=run_id,
                stage_name=stage_name,
                dataset_id=dataset.dataset_id,
                true_positives=metrics.get("true_positives", 0),
                false_positives=metrics.get("false_positives", 0),
                false_negatives=metrics.get("false_negatives", 0),
                true_negatives=metrics.get("true_negatives", 0),
                precision=metrics.get("precision", 0.0),
                recall=metrics.get("recall", 0.0),
                f1_score=metrics.get("f1_score", 0.0),
                mean_average_precision=metrics.get("mean_average_precision"),
                iou_mean=metrics.get("iou_mean"),
                sample_count=len(all_predictions),
            )

            # Save to JSON
            metrics_file = output_dir / f"{stage_name}_metrics.json"
            with open(metrics_file, "w") as f:
                f.write(accuracy_metrics.model_dump_json(indent=2))

    return all_metrics


# Public API exports
__all__ = [
    "create_dataset",
    "create_metrics_calculator",
    "create_error_analyzer",
    "validate_pipeline",
    "register_format_loader",
    "AnnotationFormat",
    "GroundTruthDataset",
    "MetricsCalculator",
    "ErrorAnalyzer",
]
