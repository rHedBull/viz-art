"""
Performance Trends API Contract

Defines interfaces for tracking historical performance and detecting regressions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .accuracy_metrics_api import AccuracyMetrics, MetricType


@dataclass
class MetricSnapshot:
    """Metric value at a specific point in time."""
    run_id: str
    timestamp: datetime
    metric_type: MetricType
    value: float
    counts: Dict[str, int]  # correct, wrong, invalid, unlabeled


@dataclass
class TrendStatistics:
    """Aggregate statistics for a performance trend."""
    best_run_id: str
    best_value: float
    worst_run_id: str
    worst_value: float
    mean_value: float
    std_dev: float
    regression_detected: bool
    regression_details: Optional[str]


@dataclass
class PerformanceTrend:
    """Historical performance data across multiple runs."""
    trend_id: str
    dataset_id: str  # Ground truth dataset used
    run_ids: List[str]
    timestamps: List[datetime]
    per_stage_metrics: Dict[str, List[MetricSnapshot]]  # stage_name -> time-series
    pipeline_version: Optional[str]
    configuration_params: Optional[Dict[str, Any]]
    aggregate_statistics: TrendStatistics
    created_at: datetime
    updated_at: datetime


class PerformanceTracker(ABC):
    """Interface for tracking performance metrics over time."""

    @abstractmethod
    def record_metrics(
        self,
        run_id: str,
        metrics: List[AccuracyMetrics],
        dataset_id: str,
        pipeline_version: Optional[str] = None
    ) -> None:
        """
        Record metrics from a pipeline run.

        Args:
            run_id: Pipeline run identifier
            metrics: List of accuracy metrics from run
            dataset_id: Ground truth dataset used
            pipeline_version: Optional version tag

        Side effects:
            - Stores metrics in Parquet format (existing PyArrow infrastructure)
            - Updates performance trend records
            - Triggers regression detection
        """
        pass

    @abstractmethod
    def get_trend(
        self,
        dataset_id: str,
        stage_name: Optional[str] = None,
        metric_type: Optional[MetricType] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> PerformanceTrend:
        """
        Retrieve performance trend data.

        Args:
            dataset_id: Ground truth dataset to filter by
            stage_name: Optional stage filter
            metric_type: Optional metric type filter
            start_date: Optional start date for time range
            end_date: Optional end date for time range

        Returns:
            PerformanceTrend instance with filtered data

        Performance:
        - Must support 100+ runs without degradation (spec SC-006)
        - Use indexed queries on Parquet data
        """
        pass

    @abstractmethod
    def compare_runs(
        self,
        run_id_1: str,
        run_id_2: str,
        dataset_id: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare metrics between two runs.

        Args:
            run_id_1: First run identifier
            run_id_2: Second run identifier
            dataset_id: Ground truth dataset (must be same for both runs)

        Returns:
            Per-stage metric deltas:
            {
                "detection": {
                    "map_delta": -0.05,  # 5% decrease
                    "precision_delta": 0.02,  # 2% increase
                    ...
                },
                "classification": {...},
                ...
            }

        Raises:
            ValueError: If runs used different datasets
        """
        pass


class RegressionDetector(ABC):
    """Interface for detecting performance regressions."""

    @abstractmethod
    def detect_regression(
        self,
        current_metrics: List[AccuracyMetrics],
        baseline_metrics: List[AccuracyMetrics],
        threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        Detect performance regression vs baseline.

        Args:
            current_metrics: Metrics from current run
            baseline_metrics: Metrics from baseline run
            threshold: Regression threshold (default: 5% drop)

        Returns:
            Regression detection result:
            {
                "regression_detected": bool,
                "affected_stages": List[str],
                "details": [
                    {
                        "stage_name": "detection",
                        "metric_type": "map",
                        "baseline_value": 0.78,
                        "current_value": 0.71,
                        "delta": -0.07,
                        "delta_percentage": -0.09  # 9% drop
                    },
                    ...
                ],
                "severity": "high" if > 10% drop, "medium" if > 5%, "low" otherwise
            }

        Logic (from spec FR-018):
        - Compare current vs baseline for each metric
        - Flag regression if drop > threshold
        - Use simple threshold-based approach (not statistical testing)
        """
        pass

    @abstractmethod
    def get_baseline(
        self,
        dataset_id: str,
        stage_name: str,
        metric_type: MetricType,
        strategy: str = "best"
    ) -> MetricSnapshot:
        """
        Get baseline metric for comparison.

        Args:
            dataset_id: Ground truth dataset
            stage_name: Pipeline stage
            metric_type: Metric to retrieve
            strategy: Baseline selection strategy:
                - "best": Best historical value
                - "latest": Most recent run
                - "mean": Average of last N runs
                - "median": Median of last N runs

        Returns:
            MetricSnapshot for baseline
        """
        pass


class ReportGenerator(ABC):
    """Interface for generating accuracy reports."""

    @abstractmethod
    def generate_per_run_report(
        self,
        run_id: str,
        metrics: List[AccuracyMetrics],
        error_patterns: List[Dict[str, Any]],
        output_path: Path
    ) -> Path:
        """
        Generate HTML report for a single run.

        Args:
            run_id: Pipeline run identifier
            metrics: Accuracy metrics from run
            error_patterns: Error pattern summaries
            output_path: Where to save HTML report

        Returns:
            Path to generated HTML report

        Report contents (spec FR-004):
        - Per-stage accuracy counts (correct/wrong/invalid/unlabeled)
        - Detailed metrics (precision, recall, F1, mAP, IoU)
        - Accuracy warnings if below thresholds
        - Link to error browser for this run
        - Link to historical comparison report

        Performance (spec SC-001):
        - Report must load in <1 second after pipeline completion
        - Use static HTML with embedded Plotly charts
        """
        pass

    @abstractmethod
    def generate_historical_report(
        self,
        trend: PerformanceTrend,
        output_path: Path
    ) -> Path:
        """
        Generate historical comparison HTML report.

        Args:
            trend: Performance trend data
            output_path: Where to save HTML report

        Returns:
            Path to generated HTML report

        Report contents (spec FR-015):
        - Accuracy trend charts over time for each stage
        - Side-by-side performance metrics with delta calculations
        - Confusion matrices and per-class breakdowns
        - Regression highlights (if detected)

        Performance (spec SC-006):
        - Must support 100+ runs without performance degradation
        - Use interactive Plotly charts for zooming/filtering
        """
        pass

    @abstractmethod
    def generate_confusion_matrix(
        self,
        metrics: AccuracyMetrics
    ) -> np.ndarray:
        """
        Generate confusion matrix for classification metrics.

        Args:
            metrics: Classification metrics with per-class values

        Returns:
            Confusion matrix as numpy array (N x N for N classes)

        Used in historical reports for per-class performance breakdown.
        """
        pass
