"""
Accuracy Metrics Calculation API Contract

Defines interfaces for calculating per-stage accuracy metrics and comparing
predictions against ground truth.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class MetricType(Enum):
    """Supported metric types."""
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1_score"
    MAP = "mean_average_precision"
    IOU = "intersection_over_union"


@dataclass
class AccuracyCounts:
    """Accuracy counts for a stage."""
    correct: int  # Predictions matching ground truth
    wrong: int  # Predictions not matching ground truth
    invalid: int  # Malformed predictions or ground truth
    unlabeled: int  # Samples without ground truth

    @property
    def total(self) -> int:
        return self.correct + self.wrong + self.invalid + self.unlabeled

    @property
    def accuracy(self) -> float:
        """Accuracy = correct / (correct + wrong)."""
        denominator = self.correct + self.wrong
        return self.correct / denominator if denominator > 0 else 0.0


@dataclass
class AccuracyMetrics:
    """Calculated accuracy metrics for a stage."""
    metrics_id: str
    run_id: str
    stage_name: str
    metric_type: MetricType
    value: float  # 0.0-1.0 for most metrics, 0-100 for mAP
    per_class_values: Optional[Dict[str, float]]
    confusion_matrix: Optional[np.ndarray]
    counts: AccuracyCounts
    timestamp: datetime
    ground_truth_ref: str  # Dataset ID
    metadata: Optional[Dict[str, Any]]


class MetricsCalculator(ABC):
    """Interface for calculating accuracy metrics."""

    @abstractmethod
    def calculate_precision_recall_f1(
        self,
        predictions: List[Any],
        ground_truths: List[Any],
        stage_name: str,
        **kwargs
    ) -> Dict[str, AccuracyMetrics]:
        """
        Calculate precision, recall, and F1 score.

        Args:
            predictions: List of predicted outputs
            ground_truths: List of ground truth labels
            stage_name: Pipeline stage name
            **kwargs: Additional parameters (e.g., threshold, averaging)

        Returns:
            Dictionary of metrics:
            {
                "precision": AccuracyMetrics(...),
                "recall": AccuracyMetrics(...),
                "f1": AccuracyMetrics(...)
            }

        Raises:
            ValueError: If predictions and ground_truths have different lengths
        """
        pass

    @abstractmethod
    def calculate_map(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
        stage_name: str,
        iou_threshold: float = 0.5
    ) -> AccuracyMetrics:
        """
        Calculate mean Average Precision (mAP) for object detection.

        Args:
            predictions: List of detection predictions
                Each item: {"boxes": [[x, y, w, h]], "labels": [...], "scores": [...]}
            ground_truths: List of ground truth detections
                Each item: {"boxes": [[x, y, w, h]], "labels": [...]}
            stage_name: Pipeline stage name
            iou_threshold: IoU threshold for matching (default: 0.5)

        Returns:
            AccuracyMetrics with mAP value and per-class AP breakdown

        Raises:
            ValueError: If prediction/ground_truth formats are invalid
        """
        pass

    @abstractmethod
    def calculate_iou(
        self,
        predictions: List[np.ndarray],
        ground_truths: List[np.ndarray],
        stage_name: str,
        num_classes: Optional[int] = None
    ) -> AccuracyMetrics:
        """
        Calculate Intersection over Union (IoU) for segmentation.

        Args:
            predictions: List of predicted segmentation masks
            ground_truths: List of ground truth masks
            stage_name: Pipeline stage name
            num_classes: Number of classes (auto-detect if None)

        Returns:
            AccuracyMetrics with mean IoU and per-class IoU breakdown

        Raises:
            ValueError: If mask shapes don't match
        """
        pass

    @abstractmethod
    def calculate_chamfer_distance(
        self,
        predictions: List[Any],  # Point clouds
        ground_truths: List[Any],  # Point clouds
        stage_name: str
    ) -> AccuracyMetrics:
        """
        Calculate Chamfer distance for point cloud comparison.

        Args:
            predictions: List of predicted point clouds (Open3D or numpy arrays)
            ground_truths: List of ground truth point clouds
            stage_name: Pipeline stage name

        Returns:
            AccuracyMetrics with mean Chamfer distance

        Raises:
            ValueError: If point clouds are empty or invalid
        """
        pass


class ComparisonEngine(ABC):
    """Interface for comparing predictions with ground truth."""

    @abstractmethod
    def compare_single(
        self,
        prediction: Any,
        ground_truth: Any,
        comparison_type: str
    ) -> Dict[str, Any]:
        """
        Compare a single prediction with ground truth.

        Args:
            prediction: Model prediction
            ground_truth: Ground truth label
            comparison_type: Type of comparison ("detection", "classification", "segmentation", "pointcloud")

        Returns:
            Comparison result:
            {
                "is_correct": bool,
                "error_type": Optional[str],  # "false_positive", "false_negative", etc.
                "iou": Optional[float],  # For detection/segmentation
                "distance": Optional[float],  # For point clouds
                "confidence": Optional[float],
                "metadata": Dict[str, Any]
            }

        Raises:
            ValueError: If comparison_type is unsupported or data formats are invalid
        """
        pass

    @abstractmethod
    def compare_batch(
        self,
        predictions: List[Any],
        ground_truths: List[Any],
        comparison_type: str
    ) -> List[Dict[str, Any]]:
        """
        Compare a batch of predictions with ground truth.

        Args:
            predictions: List of predictions
            ground_truths: List of ground truth labels
            comparison_type: Type of comparison

        Returns:
            List of comparison results (one per sample)

        Raises:
            ValueError: If lengths don't match or formats are invalid
        """
        pass

    @abstractmethod
    def aggregate_results(
        self,
        comparison_results: List[Dict[str, Any]]
    ) -> AccuracyCounts:
        """
        Aggregate comparison results into counts.

        Args:
            comparison_results: List of comparison results from compare_batch()

        Returns:
            AccuracyCounts with correct/wrong/invalid/unlabeled tallies
        """
        pass
