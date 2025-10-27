"""Prediction vs ground truth comparison engine.

This module provides functionality for comparing pipeline predictions
against ground truth labels for different task types.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from .metrics import AccuracyMetrics, AccuracyCounts


class ComparisonEngine:
    """Compare predictions against ground truth for different task types."""

    def __init__(self):
        """Initialize comparison engine."""
        self._task_comparators = {
            "detection": self._compare_detection,
            "classification": self._compare_classification,
            "segmentation": self._compare_segmentation,
            "pointcloud": self._compare_pointcloud
        }

    def compare_single(
        self, prediction: Any, ground_truth: Any, task_type: str, **kwargs
    ) -> Dict[str, Any]:
        """Compare single prediction against ground truth.

        Args:
            prediction: Model prediction
            ground_truth: Ground truth label
            task_type: Type of task (detection, classification, segmentation, pointcloud)
            **kwargs: Task-specific parameters (e.g., iou_threshold)

        Returns:
            Comparison result dict with match status, IoU, etc.

        Implementation: T024-T027 for different task types
        """
        if task_type not in self._task_comparators:
            raise ValueError(f"Unsupported task type: {task_type}")

        return self._task_comparators[task_type](prediction, ground_truth, **kwargs)

    def _compare_detection(
        self, prediction: Dict, ground_truth: Dict, iou_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Compare detection prediction with ground truth (T024).

        Args:
            prediction: Dict with 'boxes', 'labels', 'scores'
            ground_truth: Dict with 'boxes', 'labels'
            iou_threshold: IoU threshold for matching

        Returns:
            Dict with: matched, iou, is_correct, error_type
        """
        from viz_art.validation.metrics import MetricsCalculator

        calc = MetricsCalculator()

        pred_boxes = prediction.get("boxes", prediction.get("bbox", []))
        pred_labels = prediction.get("labels", prediction.get("class", []))
        pred_scores = prediction.get("scores", prediction.get("score", []))

        gt_boxes = ground_truth.get("boxes", ground_truth.get("bbox", []))
        gt_labels = ground_truth.get("labels", ground_truth.get("class", []))

        # Handle edge cases
        if len(pred_boxes) == 0 and len(gt_boxes) == 0:
            return {"matched": True, "iou": 1.0, "is_correct": True, "error_type": None}

        if len(pred_boxes) == 0:
            return {"matched": False, "iou": 0.0, "is_correct": False, "error_type": "false_negative"}

        if len(gt_boxes) == 0:
            return {"matched": False, "iou": 0.0, "is_correct": False, "error_type": "false_positive"}

        # Match predictions to ground truth boxes
        best_iou = 0.0
        best_match = None

        for i, pred_box in enumerate(pred_boxes):
            for j, gt_box in enumerate(gt_boxes):
                iou = calc._calculate_iou(pred_box, gt_box, format='xywh')

                if iou > best_iou:
                    best_iou = iou
                    pred_label = pred_labels[i] if i < len(pred_labels) else None
                    gt_label = gt_labels[j] if j < len(gt_labels) else None
                    best_match = {
                        "pred_idx": i,
                        "gt_idx": j,
                        "pred_label": pred_label,
                        "gt_label": gt_label
                    }

        # Determine correctness
        if best_iou >= iou_threshold and best_match:
            # Check label match
            labels_match = best_match["pred_label"] == best_match["gt_label"]
            is_correct = labels_match
            error_type = "misclassification" if not labels_match else None
        elif best_iou > 0.0:
            is_correct = False
            error_type = "low_iou"
        else:
            is_correct = False
            error_type = "false_positive" if len(pred_boxes) > len(gt_boxes) else "false_negative"

        return {
            "matched": best_iou >= iou_threshold,
            "iou": best_iou,
            "is_correct": is_correct,
            "error_type": error_type,
            "match_info": best_match
        }

    def _compare_classification(
        self, prediction: Any, ground_truth: Any, **kwargs
    ) -> Dict[str, Any]:
        """Compare classification prediction with ground truth (T025).

        Args:
            prediction: Predicted label (int or str)
            ground_truth: Ground truth label

        Returns:
            Dict with: matched, is_correct, error_type
        """
        is_correct = prediction == ground_truth
        error_type = "misclassification" if not is_correct else None

        return {
            "matched": is_correct,
            "is_correct": is_correct,
            "error_type": error_type,
            "prediction": prediction,
            "ground_truth": ground_truth
        }

    def _compare_segmentation(
        self, prediction: np.ndarray, ground_truth: np.ndarray, **kwargs
    ) -> Dict[str, Any]:
        """Compare segmentation prediction with ground truth (T026).

        Args:
            prediction: Predicted segmentation mask (H x W)
            ground_truth: Ground truth mask (H x W)

        Returns:
            Dict with: matched, iou, is_correct, error_type, pixel_accuracy
        """
        from viz_art.validation.metrics import MetricsCalculator

        calc = MetricsCalculator()
        iou_result = calc.calculate_iou(prediction, ground_truth)

        mean_iou = iou_result["iou_mean"]
        pixel_accuracy = np.mean(prediction == ground_truth)

        # Define correctness threshold
        iou_threshold = kwargs.get("iou_threshold", 0.5)
        is_correct = mean_iou >= iou_threshold

        error_type = None
        if not is_correct:
            if mean_iou < 0.3:
                error_type = "low_iou"
            else:
                error_type = "partial_match"

        return {
            "matched": is_correct,
            "iou": mean_iou,
            "is_correct": is_correct,
            "error_type": error_type,
            "pixel_accuracy": pixel_accuracy,
            "per_class_iou": iou_result["per_class_iou"]
        }

    def _compare_pointcloud(
        self, prediction, ground_truth, distance_threshold: float = 0.05, **kwargs
    ) -> Dict[str, Any]:
        """Compare point cloud prediction with ground truth (T027).

        Args:
            prediction: Predicted point cloud (Open3D PointCloud)
            ground_truth: Ground truth point cloud
            distance_threshold: Distance threshold for matching (default: 5cm)

        Returns:
            Dict with: matched, chamfer_distance, is_correct, error_type
        """
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError("Open3D required for point cloud comparison")

        # Compute distances
        pred_to_gt = np.asarray(prediction.compute_point_cloud_distance(ground_truth))
        gt_to_pred = np.asarray(ground_truth.compute_point_cloud_distance(prediction))

        # Chamfer distance
        chamfer = (np.mean(pred_to_gt) + np.mean(gt_to_pred)) / 2.0

        # Calculate match percentage
        match_ratio = np.mean(pred_to_gt < distance_threshold)
        is_correct = match_ratio >= 0.7  # 70% of points within threshold

        error_type = None
        if not is_correct:
            if chamfer > 0.1:  # 10cm
                error_type = "large_deviation"
            else:
                error_type = "partial_alignment"

        return {
            "matched": is_correct,
            "chamfer_distance": float(chamfer),
            "is_correct": is_correct,
            "error_type": error_type,
            "match_ratio": float(match_ratio),
            "mean_distance": float(np.mean(pred_to_gt))
        }

    def compare_batch(
        self, predictions: List[Any], ground_truths: List[Any], task_type: str, **kwargs
    ) -> List[Dict[str, Any]]:
        """Compare batch of predictions against ground truths (T028).

        Args:
            predictions: List of predictions
            ground_truths: List of ground truths
            task_type: Type of task
            **kwargs: Task-specific parameters

        Returns:
            List of comparison result dicts
        """
        if len(predictions) != len(ground_truths):
            raise ValueError(
                f"Predictions and ground truths must have same length: "
                f"{len(predictions)} vs {len(ground_truths)}"
            )

        results = []
        for pred, gt in zip(predictions, ground_truths):
            try:
                result = self.compare_single(pred, gt, task_type, **kwargs)
                result["valid"] = True
            except Exception as e:
                # Handle invalid comparisons
                result = {
                    "matched": False,
                    "is_correct": False,
                    "error_type": "invalid",
                    "valid": False,
                    "error_message": str(e)
                }
            results.append(result)

        return results

    def aggregate_results(
        self, comparison_results: List[Dict[str, Any]]
    ) -> AccuracyCounts:
        """Aggregate comparison results into counts (T029).

        Args:
            comparison_results: List of comparison result dicts

        Returns:
            AccuracyCounts with correct/wrong/invalid/unlabeled counts
        """
        correct = 0
        wrong = 0
        invalid = 0
        unlabeled = 0

        for result in comparison_results:
            if not result.get("valid", True):
                invalid += 1
            elif result.get("is_correct", False):
                correct += 1
            else:
                wrong += 1

        return AccuracyCounts(
            correct=correct,
            wrong=wrong,
            invalid=invalid,
            unlabeled=unlabeled
        )
