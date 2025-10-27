"""Accuracy metrics calculation and storage.

This module provides data models and calculation functions for per-stage
accuracy metrics (precision, recall, F1, mAP, IoU).
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
import numpy as np


class MetricType(Enum):
    """Types of accuracy metrics."""

    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1_score"
    MAP = "mean_average_precision"
    IOU = "intersection_over_union"
    CHAMFER_DISTANCE = "chamfer_distance"


@dataclass
class AccuracyCounts:
    """Counts of correct, wrong, invalid, and unlabeled predictions.

    Attributes:
        correct: Predictions matching ground truth
        wrong: Predictions not matching ground truth
        invalid: Malformed predictions or ground truth
        unlabeled: Samples without ground truth
    """

    correct: int
    wrong: int
    invalid: int
    unlabeled: int

    def __post_init__(self):
        """Validate counts are non-negative."""
        if any(count < 0 for count in [self.correct, self.wrong, self.invalid, self.unlabeled]):
            raise ValueError("All counts must be >= 0")

    @property
    def total(self) -> int:
        """Total number of samples."""
        return self.correct + self.wrong + self.invalid + self.unlabeled

    @property
    def accuracy(self) -> float:
        """Accuracy = correct / (correct + wrong)."""
        denominator = self.correct + self.wrong
        return self.correct / denominator if denominator > 0 else 0.0


@dataclass
class AccuracyMetrics:
    """Calculated performance measures for pipeline stages.

    Attributes:
        metrics_id: Unique identifier (UUID v4 format)
        run_id: Associated pipeline run (from audit trail)
        stage_name: Pipeline stage name
        metric_type: Type of metric
        value: Calculated metric value (0.0-1.0 or 0-100 for mAP)
        per_class_values: Per-class breakdown (class name → value)
        confusion_matrix: Confusion matrix (optional)
        counts: Correct/wrong/invalid/unlabeled counts
        timestamp: Metric calculation time
        ground_truth_ref: Dataset ID used
        metadata: Additional metric info
    """

    metrics_id: str
    run_id: str
    stage_name: str
    metric_type: MetricType
    value: float
    counts: AccuracyCounts
    timestamp: datetime
    ground_truth_ref: str
    per_class_values: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate metric attributes."""
        # Validate value range
        if self.metric_type == MetricType.MAP:
            if not 0 <= self.value <= 100:
                raise ValueError(f"mAP value must be in [0, 100]: {self.value}")
        else:
            if not 0.0 <= self.value <= 1.0:
                raise ValueError(
                    f"{self.metric_type.value} must be in [0, 1]: {self.value}"
                )

        # Validate per_class_values (allow 0-1 for most metrics, 0-100 for mAP)
        max_value = 100.0 if self.metric_type == MetricType.MAP else 1.0
        for class_name, class_value in self.per_class_values.items():
            if not 0.0 <= class_value <= max_value:
                raise ValueError(
                    f"Per-class value for {class_name} must be in [0, {max_value}]: {class_value}"
                )

        # Validate confusion matrix is square
        if self.confusion_matrix is not None:
            if len(self.confusion_matrix.shape) != 2:
                raise ValueError("Confusion matrix must be 2D")
            if self.confusion_matrix.shape[0] != self.confusion_matrix.shape[1]:
                raise ValueError("Confusion matrix must be square")


class MetricsCalculator:
    """Calculate accuracy metrics from predictions and ground truth.

    This class provides methods for calculating various accuracy metrics
    for different task types (classification, detection, segmentation, point clouds).
    """

    def calculate_precision_recall_f1(
        self, predictions, ground_truths, stage_name: str, run_id: str, ground_truth_ref: str
    ) -> AccuracyMetrics:
        """Calculate precision, recall, and F1 score for classification.

        Args:
            predictions: List of predicted labels
            ground_truths: List of ground truth labels
            stage_name: Pipeline stage name
            run_id: Pipeline run identifier
            ground_truth_ref: Dataset ID used for validation

        Returns:
            AccuracyMetrics instance with precision/recall/F1

        Implementation: Uses scikit-learn.metrics.precision_recall_fscore_support
        """
        from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
        import uuid

        # Convert to numpy arrays
        y_pred = np.array(predictions)
        y_true = np.array(ground_truths)

        # Calculate precision, recall, F1 for each class, then average
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0.0
        )

        # Calculate per-class metrics
        per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0.0
        )

        # Build per-class values dict (using class index as key)
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        per_class_values = {}
        for i, cls in enumerate(unique_classes):
            if i < len(per_class_f1):
                per_class_values[str(cls)] = float(per_class_f1[i])

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Calculate accuracy counts
        correct = int(np.sum(y_pred == y_true))
        wrong = int(np.sum(y_pred != y_true))
        invalid = 0  # No invalid samples for well-formed classification
        unlabeled = 0  # All samples have labels

        counts = AccuracyCounts(
            correct=correct,
            wrong=wrong,
            invalid=invalid,
            unlabeled=unlabeled
        )

        return AccuracyMetrics(
            metrics_id=str(uuid.uuid4()),
            run_id=run_id,
            stage_name=stage_name,
            metric_type=MetricType.F1,
            value=float(f1),
            per_class_values=per_class_values,
            confusion_matrix=cm,
            counts=counts,
            timestamp=datetime.now(),
            ground_truth_ref=ground_truth_ref,
            metadata={
                "precision": float(precision),
                "recall": float(recall),
                "average": "weighted"
            }
        )

    def calculate_mean_average_precision(
        self, predictions, ground_truths, stage_name: str, run_id: str,
        ground_truth_ref: str, iou_threshold: float = 0.5
    ) -> AccuracyMetrics:
        """Calculate mean average precision for object detection.

        Args:
            predictions: List of detection dicts with boxes, labels, scores
            ground_truths: List of ground truth dicts with boxes, labels
            stage_name: Pipeline stage name
            run_id: Pipeline run identifier
            ground_truth_ref: Dataset ID used for validation
            iou_threshold: IoU threshold for matching detections

        Returns:
            AccuracyMetrics instance with mAP value

        Implementation: Extends existing MetricsCalculator.calculate_mean_average_precision
        from src/viz_art/validation/metrics.py
        """
        from viz_art.validation.metrics import MetricsCalculator as ValidationMetricsCalculator
        import uuid

        # Convert predictions and ground_truths to format expected by validation metrics
        # (needs 'class' key instead of 'labels')
        def convert_format(items):
            converted = []
            for item in items:
                if isinstance(item, dict):
                    new_item = item.copy()
                    # Convert 'labels' to 'class' if present
                    if 'labels' in new_item and 'class' not in new_item:
                        labels = new_item.pop('labels')
                        # If labels is a list, use first element
                        new_item['class'] = labels[0] if isinstance(labels, list) and labels else labels
                    # Convert 'boxes' to 'bbox' if present
                    if 'boxes' in new_item and 'bbox' not in new_item:
                        boxes = new_item.pop('boxes')
                        # If boxes is a list of boxes, use first box
                        new_item['bbox'] = boxes[0] if isinstance(boxes, list) and boxes else boxes
                    # Convert 'scores' to 'score' if present
                    if 'scores' in new_item and 'score' not in new_item:
                        scores = new_item.pop('scores')
                        # If scores is a list, use first element
                        new_item['score'] = scores[0] if isinstance(scores, list) and scores else scores
                    converted.append(new_item)
                else:
                    converted.append(item)
            return converted

        predictions_converted = convert_format(predictions)
        ground_truths_converted = convert_format(ground_truths)

        # Use existing mAP implementation
        val_calculator = ValidationMetricsCalculator()
        mAP_results = val_calculator.calculate_mean_average_precision(
            predictions_converted, ground_truths_converted, iou_threshold
        )

        # Calculate accuracy counts
        # Count how many ground truth boxes were matched
        total_gt = sum(len(gt.get("boxes", [])) for gt in ground_truths if "boxes" in gt)
        total_gt = max(total_gt, sum(len(gt.get("bbox", [])) if isinstance(gt.get("bbox"), list) and isinstance(gt.get("bbox")[0], list) else 1 for gt in ground_truths if "bbox" in gt))

        total_pred = sum(len(pred.get("boxes", [])) for pred in predictions if "boxes" in pred)
        total_pred = max(total_pred, sum(len(pred.get("bbox", [])) if isinstance(pred.get("bbox"), list) and isinstance(pred.get("bbox")[0], list) else 1 for pred in predictions if "bbox" in pred))

        # Approximate correct/wrong counts from mAP calculation
        # This is a simplified heuristic: mAP * total_gt gives approximate matches
        map_value = mAP_results["mean_average_precision"]
        correct = int(map_value * total_gt)
        wrong = max(0, total_pred - correct)  # False positives
        wrong += max(0, total_gt - correct)  # False negatives
        invalid = 0
        unlabeled = 0

        counts = AccuracyCounts(
            correct=correct,
            wrong=wrong,
            invalid=invalid,
            unlabeled=unlabeled
        )

        return AccuracyMetrics(
            metrics_id=str(uuid.uuid4()),
            run_id=run_id,
            stage_name=stage_name,
            metric_type=MetricType.MAP,
            value=float(map_value * 100),  # mAP is stored as 0-100
            per_class_values={k: float(v * 100) for k, v in mAP_results["per_class_ap"].items()},
            counts=counts,
            timestamp=datetime.now(),
            ground_truth_ref=ground_truth_ref,
            metadata={"iou_threshold": iou_threshold}
        )

    def calculate_iou(
        self, predictions, ground_truths, stage_name: str, run_id: str, ground_truth_ref: str
    ) -> AccuracyMetrics:
        """Calculate IoU for segmentation masks.

        Args:
            predictions: List of predicted segmentation masks (numpy arrays)
            ground_truths: List of ground truth masks (numpy arrays)
            stage_name: Pipeline stage name
            run_id: Pipeline run identifier
            ground_truth_ref: Dataset ID used for validation

        Returns:
            AccuracyMetrics instance with IoU value

        Implementation: Extends existing MetricsCalculator.calculate_iou
        from src/viz_art/validation/metrics.py
        """
        from viz_art.validation.metrics import MetricsCalculator as ValidationMetricsCalculator
        import uuid

        # Use existing IoU implementation
        val_calculator = ValidationMetricsCalculator()

        # Calculate IoU for each prediction/ground truth pair, then average
        iou_scores = []
        per_class_iou_totals = {}
        per_class_counts = {}

        for pred_mask, gt_mask in zip(predictions, ground_truths):
            iou_result = val_calculator.calculate_iou(pred_mask, gt_mask)
            iou_scores.append(iou_result["iou_mean"])

            # Accumulate per-class IoU
            for cls, iou_val in iou_result["per_class_iou"].items():
                if not np.isnan(iou_val):
                    cls_str = str(cls)
                    if cls_str not in per_class_iou_totals:
                        per_class_iou_totals[cls_str] = 0.0
                        per_class_counts[cls_str] = 0
                    per_class_iou_totals[cls_str] += iou_val
                    per_class_counts[cls_str] += 1

        # Average per-class IoU across all samples
        per_class_values = {
            cls: per_class_iou_totals[cls] / per_class_counts[cls]
            for cls in per_class_iou_totals
        }

        # Calculate overall mean IoU
        mean_iou = np.mean(iou_scores) if iou_scores else 0.0

        # Calculate accuracy counts
        # For segmentation, count pixels that match
        correct_pixels = 0
        wrong_pixels = 0
        total_pixels = 0

        for pred_mask, gt_mask in zip(predictions, ground_truths):
            matches = (pred_mask == gt_mask)
            correct_pixels += int(np.sum(matches))
            wrong_pixels += int(np.sum(~matches))
            total_pixels += pred_mask.size

        counts = AccuracyCounts(
            correct=correct_pixels,
            wrong=wrong_pixels,
            invalid=0,
            unlabeled=0
        )

        return AccuracyMetrics(
            metrics_id=str(uuid.uuid4()),
            run_id=run_id,
            stage_name=stage_name,
            metric_type=MetricType.IOU,
            value=float(mean_iou),
            per_class_values=per_class_values,
            counts=counts,
            timestamp=datetime.now(),
            ground_truth_ref=ground_truth_ref,
            metadata={"total_samples": len(predictions), "total_pixels": total_pixels}
        )

    def calculate_chamfer_distance(
        self, predictions, ground_truths, stage_name: str, run_id: str, ground_truth_ref: str
    ) -> AccuracyMetrics:
        """Calculate Chamfer distance for point clouds.

        Args:
            predictions: List of predicted point clouds (Open3D PointCloud objects)
            ground_truths: List of ground truth point clouds
            stage_name: Pipeline stage name
            run_id: Pipeline run identifier
            ground_truth_ref: Dataset ID used for validation

        Returns:
            AccuracyMetrics instance with Chamfer distance

        Implementation: Uses Open3D.geometry.PointCloud.compute_point_cloud_distance
        """
        import uuid
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError("Open3D is required for point cloud metrics. Install with: pip install open3d>=0.18")

        # Calculate Chamfer distance for each pair
        chamfer_distances = []

        for pred_pcd, gt_pcd in zip(predictions, ground_truths):
            # Compute distances from pred to gt
            pred_to_gt = np.asarray(pred_pcd.compute_point_cloud_distance(gt_pcd))

            # Compute distances from gt to pred
            gt_to_pred = np.asarray(gt_pcd.compute_point_cloud_distance(pred_pcd))

            # Chamfer distance is the average of both directions
            chamfer = (np.mean(pred_to_gt) + np.mean(gt_to_pred)) / 2.0
            chamfer_distances.append(float(chamfer))

        # Calculate mean Chamfer distance
        mean_chamfer = float(np.mean(chamfer_distances)) if chamfer_distances else 0.0

        # Calculate accuracy counts based on distance threshold
        # Use 5cm threshold (0.05 units) as "correct" match
        distance_threshold = 0.05
        correct = 0
        wrong = 0
        total_points = 0

        for pred_pcd, gt_pcd in zip(predictions, ground_truths):
            pred_to_gt = np.asarray(pred_pcd.compute_point_cloud_distance(gt_pcd))
            correct += int(np.sum(pred_to_gt < distance_threshold))
            wrong += int(np.sum(pred_to_gt >= distance_threshold))
            total_points += len(pred_to_gt)

        counts = AccuracyCounts(
            correct=correct,
            wrong=wrong,
            invalid=0,
            unlabeled=0
        )

        return AccuracyMetrics(
            metrics_id=str(uuid.uuid4()),
            run_id=run_id,
            stage_name=stage_name,
            metric_type=MetricType.CHAMFER_DISTANCE,
            value=mean_chamfer,
            counts=counts,
            timestamp=datetime.now(),
            ground_truth_ref=ground_truth_ref,
            metadata={
                "total_samples": len(predictions),
                "total_points": total_points,
                "distance_threshold": distance_threshold,
                "per_sample_distances": chamfer_distances
            }
        )
