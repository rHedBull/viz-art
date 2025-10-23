"""Metrics Calculation for Ground Truth Validation

Provides utilities for calculating accuracy metrics (precision, recall, F1, mAP, IoU)
for comparing pipeline outputs against ground truth annotations.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


class MetricsCalculator:
    """Calculator for accuracy metrics.

    Supports different metric types based on task:
    - classification: precision, recall, F1
    - detection: mAP (mean Average Precision), IoU
    - segmentation: per-class IoU, mean IoU
    """

    def __init__(self, metric_type: str = "auto"):
        """Initialize metrics calculator.

        Args:
            metric_type: Type of metrics to calculate
                        ('classification', 'detection', 'segmentation', 'auto')
                        'auto' attempts to detect from data structure
        """
        self.metric_type = metric_type

    def calculate_precision_recall(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate precision, recall, and F1 scores.

        Args:
            y_true: Ground truth labels (shape: [n_samples])
            y_pred: Predicted labels (shape: [n_samples])

        Returns:
            Dictionary with metrics:
            {
                'precision': float,
                'recall': float,
                'f1_score': float,
                'true_positives': int,
                'false_positives': int,
                'false_negatives': int,
                'true_negatives': int,
            }
        """
        # Calculate precision, recall, F1
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0.0
        )

        # Calculate confusion matrix for TP/FP/FN/TN
        cm = confusion_matrix(y_true, y_pred)

        # For binary classification
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Multi-class: calculate as one-vs-rest for class 1 (positive)
            tp = cm[1, 1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
            fp = cm[0, 1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
            fn = cm[1, 0] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
            tn = cm[0, 0] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
        }

    def calculate_mean_average_precision(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
        iou_threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Calculate mean Average Precision for object detection.

        Args:
            predictions: List of predicted bboxes with format:
                        [{'bbox': [x, y, w, h], 'score': float, 'class': str}, ...]
            ground_truth: List of ground truth bboxes with format:
                         [{'bbox': [x, y, w, h], 'class': str}, ...]
            iou_threshold: IoU threshold for positive match (default: 0.5)

        Returns:
            Dictionary with mAP metrics:
            {
                'mean_average_precision': float,
                'per_class_ap': Dict[str, float],
            }
        """
        # Group predictions and ground truth by class
        pred_by_class: Dict[str, List] = {}
        gt_by_class: Dict[str, List] = {}

        for pred in predictions:
            cls = pred["class"]
            if cls not in pred_by_class:
                pred_by_class[cls] = []
            pred_by_class[cls].append(pred)

        for gt in ground_truth:
            cls = gt["class"]
            if cls not in gt_by_class:
                gt_by_class[cls] = []
            gt_by_class[cls].append(gt)

        # Calculate AP for each class
        per_class_ap = {}
        all_classes = set(pred_by_class.keys()) | set(gt_by_class.keys())

        for cls in all_classes:
            class_preds = pred_by_class.get(cls, [])
            class_gts = gt_by_class.get(cls, [])

            if not class_gts:
                # No ground truth for this class
                per_class_ap[cls] = 0.0
                continue

            if not class_preds:
                # No predictions for this class
                per_class_ap[cls] = 0.0
                continue

            # Sort predictions by confidence score (descending)
            class_preds = sorted(class_preds, key=lambda x: x["score"], reverse=True)

            # Match predictions to ground truth
            tp = []
            fp = []
            matched_gts = set()

            for pred in class_preds:
                pred_bbox = pred["bbox"]
                best_iou = 0.0
                best_gt_idx = -1

                for gt_idx, gt in enumerate(class_gts):
                    if gt_idx in matched_gts:
                        continue

                    iou = self._calculate_iou(pred_bbox, gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    tp.append(1)
                    fp.append(0)
                    matched_gts.add(best_gt_idx)
                else:
                    tp.append(0)
                    fp.append(1)

            # Calculate precision-recall curve
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)

            recalls = tp_cumsum / len(class_gts)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)

            # Calculate AP using 11-point interpolation
            ap = 0.0
            for t in np.linspace(0, 1, 11):
                precisions_above = precisions[recalls >= t]
                if len(precisions_above) > 0:
                    ap += np.max(precisions_above) / 11.0

            per_class_ap[cls] = float(ap)

        # Calculate mAP
        mean_ap = np.mean(list(per_class_ap.values())) if per_class_ap else 0.0

        return {
            "mean_average_precision": float(mean_ap),
            "per_class_ap": per_class_ap,
        }

    def _calculate_iou(
        self, bbox1: List[float], bbox2: List[float], format: str = "xywh"
    ) -> float:
        """Calculate Intersection over Union for bounding boxes.

        Args:
            bbox1: First bounding box [x, y, w, h] or [x1, y1, x2, y2]
            bbox2: Second bounding box in same format
            format: 'xywh' (x, y, width, height) or 'xyxy' (x1, y1, x2, y2)

        Returns:
            IoU score (0-1)
        """
        if format == "xywh":
            # Convert [x, y, w, h] to [x1, y1, x2, y2]
            x1_1, y1_1 = bbox1[0], bbox1[1]
            x2_1, y2_1 = bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]
            x1_2, y1_2 = bbox2[0], bbox2[1]
            x2_2, y2_2 = bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]
        else:
            x1_1, y1_1, x2_1, y2_1 = bbox1
            x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate intersection
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)

        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        iou = intersection / (union + 1e-10)
        return float(iou)

    def calculate_iou(
        self, pred_mask: np.ndarray, gt_mask: np.ndarray, num_classes: Optional[int] = None
    ) -> Dict[str, float]:
        """Calculate IoU for segmentation masks.

        Args:
            pred_mask: Predicted segmentation mask (shape: [H, W])
            gt_mask: Ground truth mask (shape: [H, W])
            num_classes: Number of classes (auto-detected if None)

        Returns:
            Dictionary with IoU metrics:
            {
                'iou_mean': float,
                'per_class_iou': Dict[int, float],
            }
        """
        if num_classes is None:
            num_classes = max(pred_mask.max(), gt_mask.max()) + 1

        per_class_iou = {}

        for cls in range(num_classes):
            pred_cls = pred_mask == cls
            gt_cls = gt_mask == cls

            intersection = np.logical_and(pred_cls, gt_cls).sum()
            union = np.logical_or(pred_cls, gt_cls).sum()

            if union == 0:
                # Class not present in either prediction or ground truth
                iou = float("nan")
            else:
                iou = intersection / union

            per_class_iou[cls] = float(iou)

        # Calculate mean IoU (excluding NaN for absent classes)
        valid_ious = [iou for iou in per_class_iou.values() if not np.isnan(iou)]
        iou_mean = np.mean(valid_ious) if valid_ious else 0.0

        return {
            "iou_mean": float(iou_mean),
            "per_class_iou": per_class_iou,
        }
