"""
Error Analysis API Contract

Defines interfaces for browsing, visualizing, and analyzing error cases.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class ErrorType(Enum):
    """Types of errors."""
    FALSE_POSITIVE = "false_positive"
    FALSE_NEGATIVE = "false_negative"
    MISCLASSIFICATION = "misclassification"
    LOW_IOU = "low_iou"


class ErrorSeverity(Enum):
    """Error severity levels."""
    CRITICAL = "critical"  # Pipeline fails entirely
    HIGH = "high"  # Wrong output (FP/FN with IoU < 0.3)
    MEDIUM = "medium"  # Degraded quality (misclass or 0.3 <= IoU < 0.5)
    LOW = "low"  # Minor deviation (IoU >= 0.5)


@dataclass
class ErrorCase:
    """Individual error case."""
    error_id: str
    run_id: str
    stage_name: str
    sample_id: str
    error_type: ErrorType
    severity: ErrorSeverity
    prediction: Any
    ground_truth: Any
    iou: Optional[float]
    confidence: Optional[float]
    saved_artifacts: Dict[str, Path]  # artifact_name -> path
    timestamp: datetime
    metadata: Optional[Dict[str, Any]]


@dataclass
class ErrorPattern:
    """Clustered group of similar errors."""
    pattern_id: str  # Composite key: "{stage}_{error_type}"
    run_id: str
    stage_name: str
    error_type: ErrorType
    severity: ErrorSeverity
    error_count: int
    affected_samples: List[str]
    statistics: Dict[str, Any]  # avg_iou, avg_confidence, etc.
    suggested_cause: Optional[str]
    timestamp: datetime


class ErrorDetector(ABC):
    """Interface for detecting and categorizing errors."""

    @abstractmethod
    def detect_errors(
        self,
        comparison_results: List[Dict[str, Any]],
        stage_name: str,
        run_id: str
    ) -> List[ErrorCase]:
        """
        Detect errors from comparison results.

        Args:
            comparison_results: Results from ComparisonEngine.compare_batch()
            stage_name: Pipeline stage name
            run_id: Pipeline run identifier

        Returns:
            List of ErrorCase instances for failures

        Side effects:
            - Saves error artifacts via OutputSaver
            - Creates error metadata JSON files
        """
        pass

    @abstractmethod
    def categorize_error(
        self,
        prediction: Any,
        ground_truth: Any,
        comparison_result: Dict[str, Any]
    ) -> tuple[ErrorType, ErrorSeverity]:
        """
        Categorize an error by type and severity.

        Args:
            prediction: Model prediction
            ground_truth: Ground truth label
            comparison_result: Result from ComparisonEngine

        Returns:
            Tuple of (ErrorType, ErrorSeverity)

        Logic:
        - ErrorType determined by comparison (FP, FN, misclass, low IoU)
        - ErrorSeverity based on spec assumptions (IoU thresholds)
        """
        pass


class ErrorPatternDetector(ABC):
    """Interface for detecting error patterns via clustering."""

    @abstractmethod
    def cluster_errors(
        self,
        failures: List[ErrorCase],
        grouping_rules: List[str] = ["stage_name", "error_type"]
    ) -> Dict[str, ErrorPattern]:
        """
        Cluster errors using rule-based grouping.

        Args:
            failures: List of error cases to cluster
            grouping_rules: Ordered list of features to group by
                          Options: ["stage_name", "error_type", "severity", "iou_bucket"]

        Returns:
            Dictionary mapping pattern_id to ErrorPattern:
            {
                "detection_false_positive": ErrorPattern(...),
                "classification_misclassification": ErrorPattern(...),
                ...
            }

        Performance:
        - O(n) complexity for rule-based clustering
        - Must complete in <15ms for 1000 errors (spec SC-010)
        """
        pass

    @abstractmethod
    def summarize_patterns(
        self,
        patterns: Dict[str, ErrorPattern]
    ) -> List[Dict[str, Any]]:
        """
        Generate summary statistics for error patterns.

        Args:
            patterns: Dictionary of error patterns

        Returns:
            List of pattern summaries sorted by error_count (descending):
            [
                {
                    "pattern_id": "detection_false_positive",
                    "error_count": 45,
                    "percentage": 0.32,
                    "avg_iou": 0.12,
                    "suggested_cause": "Model generating spurious detections...",
                    "sample_preview": ["sample_001", "sample_002", ...]
                },
                ...
            ]
        """
        pass


class ErrorVisualizer(ABC):
    """Interface for generating error visualizations."""

    @abstractmethod
    def create_side_by_side_visualization(
        self,
        error_case: ErrorCase,
        output_path: Path
    ) -> Path:
        """
        Create side-by-side prediction vs ground truth visualization.

        Args:
            error_case: Error case to visualize
            output_path: Where to save visualization

        Returns:
            Path to saved visualization image

        Supports:
        - Images: Bounding boxes overlaid on images
        - Point clouds: 3D visualization with color coding
        """
        pass

    @abstractmethod
    def create_diff_visualization(
        self,
        error_case: ErrorCase,
        output_path: Path,
        diff_type: str = "auto"
    ) -> Path:
        """
        Create difference visualization.

        Args:
            error_case: Error case to visualize
            output_path: Where to save diff visualization
            diff_type: Type of diff ("image_diff", "pointcloud_heatmap", "auto")

        Returns:
            Path to saved diff visualization

        Implementations:
        - image_diff: Color-coded difference regions (FR-008)
        - pointcloud_heatmap: Distance error heatmap (FR-009)
        - auto: Choose based on data type
        """
        pass

    @abstractmethod
    def create_image_diff(
        self,
        pred_image: np.ndarray,
        gt_image: np.ndarray
    ) -> np.ndarray:
        """
        Create color-coded image difference visualization.

        Args:
            pred_image: Predicted image (H x W x C)
            gt_image: Ground truth image (same shape)

        Returns:
            Difference image with color coding:
            - Green: Correct regions
            - Red: False positives
            - Blue: False negatives
            - Yellow: Misalignments
        """
        pass

    @abstractmethod
    def create_pointcloud_diff(
        self,
        pred_pcd: Any,  # Open3D PointCloud
        gt_pcd: Any,  # Open3D PointCloud
        use_icp: bool = False
    ) -> Any:  # Open3D PointCloud with heatmap colors
        """
        Create point cloud distance error heatmap.

        Args:
            pred_pcd: Predicted point cloud
            gt_pcd: Ground truth point cloud
            use_icp: Whether to use ICP alignment (default: nearest-neighbor only)

        Returns:
            Colored point cloud with distance heatmap:
            - Blue: Low error (< 1cm)
            - Green: Medium error (1-5cm)
            - Yellow: High error (5-10cm)
            - Red: Very high error (> 10cm)

        Implementation (from research.md):
        - Primary: Open3D compute_point_cloud_distance() (nearest-neighbor)
        - Fallback: ICP alignment if median distance > 5cm threshold
        """
        pass


class ErrorBrowser(ABC):
    """Interface for browsing and filtering error cases."""

    @abstractmethod
    def load_errors(
        self,
        run_id: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[ErrorCase]:
        """
        Load error cases from a run with optional filtering.

        Args:
            run_id: Pipeline run identifier
            filters: Optional filters:
                {
                    "stage_name": str or List[str],
                    "error_type": ErrorType or List[ErrorType],
                    "severity": ErrorSeverity or List[ErrorSeverity],
                    "min_confidence": float,
                    "max_iou": float,
                    "sample_ids": List[str]
                }

        Returns:
            List of ErrorCase instances matching filters

        Performance:
        - Must support 1000+ errors with <100ms filtering (spec SC-010)
        - Use indexed lookups for filtering
        """
        pass

    @abstractmethod
    def get_error_by_id(self, error_id: str) -> ErrorCase:
        """
        Retrieve a specific error case by ID.

        Args:
            error_id: Unique error identifier

        Returns:
            ErrorCase instance

        Raises:
            KeyError: If error_id not found
        """
        pass

    @abstractmethod
    def export_errors(
        self,
        errors: List[ErrorCase],
        output_path: Path,
        format: str = "json"
    ) -> Path:
        """
        Export error cases for offline review.

        Args:
            errors: List of errors to export
            output_path: Where to save export
            format: Export format ("json", "csv", "parquet")

        Returns:
            Path to exported file

        Exports include:
        - Error metadata (JSON/CSV)
        - Saved artifacts (copied to export directory)
        - Visualizations (if available)
        """
        pass
