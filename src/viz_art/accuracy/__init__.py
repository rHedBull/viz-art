"""Accuracy tracking and metrics calculation module.

This module provides functionality for calculating per-stage accuracy metrics,
comparing pipeline predictions against ground truth labels, and generating
accuracy reports.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import uuid

from .metrics import AccuracyCounts, MetricType, AccuracyMetrics, MetricsCalculator
from .ground_truth import (
    GroundTruthDataset,
    GroundTruthSample,
    AnnotationFormat,
    GroundTruthLoader
)
from .comparison import ComparisonEngine
from .reporter import ReportGenerator
from ..error_analysis.patterns import ErrorDetector, ErrorPatternDetector
from ..performance.trends import PerformanceTracker


class AccuracyTracker:
    """Orchestrator for accuracy tracking workflow (T034).

    This class coordinates ground truth loading, metrics calculation,
    comparison, and report generation for pipeline validation.
    """

    def __init__(self, ground_truth_dataset: GroundTruthDataset):
        """Initialize accuracy tracker.

        Args:
            ground_truth_dataset: Ground truth dataset for validation
        """
        self.dataset = ground_truth_dataset
        self.gt_loader = GroundTruthLoader()
        self.metrics_calculator = MetricsCalculator()
        self.comparison_engine = ComparisonEngine()
        self.report_generator = ReportGenerator()
        self.error_detector = ErrorDetector()  # T058
        self.pattern_detector = ErrorPatternDetector()
        self.performance_tracker = PerformanceTracker(Path("output/trends"))  # T080

    def run_validation(
        self,
        predictions: Dict[str, List[Any]],
        run_id: str,
        output_dir: Path,
        stage_task_types: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Run validation workflow (T035).

        Args:
            predictions: Dict mapping stage name to list of predictions
            run_id: Pipeline run identifier
            output_dir: Output directory for reports and metrics
            stage_task_types: Dict mapping stage name to task type
                             (detection, classification, segmentation, pointcloud)

        Returns:
            Dict with validation results:
            {
                'run_id': str,
                'overall_accuracy': float,
                'stage_metrics': Dict[str, List[AccuracyMetrics]],
                'report_path': Path,
                'metrics_path': Path
            }
        """
        if stage_task_types is None:
            # Auto-detect task types from predictions
            stage_task_types = self._infer_task_types(predictions)

        # Load ground truth samples
        sample_ids = list(predictions.values())[0]  # Assumes all stages have same samples
        if isinstance(sample_ids, list) and len(sample_ids) > 0:
            if isinstance(sample_ids[0], dict) and 'sample_id' in sample_ids[0]:
                sample_ids = [s['sample_id'] for s in sample_ids]
            else:
                # Generate sample IDs if not provided
                sample_ids = [f"sample_{i:04d}" for i in range(len(sample_ids))]

        # Calculate metrics for each stage
        stage_metrics = {}
        all_errors = []  # T058: Collect errors across all stages

        for stage_name, stage_predictions in predictions.items():
            task_type = stage_task_types.get(stage_name, "classification")

            # Load ground truth for this stage
            ground_truths = []
            for sample_id in sample_ids[:len(stage_predictions)]:
                gt_sample = self.gt_loader.load_sample(self.dataset, sample_id)
                # Extract stage-specific ground truth
                stage_gt = gt_sample.stage_labels.get(stage_name, gt_sample.final_label)
                ground_truths.append(stage_gt)

            # Run comparison engine to get comparison results
            comparison_results = self.comparison_engine.compare_batch(
                predictions=stage_predictions,
                ground_truths=ground_truths,
                task_type=task_type
            )

            # Detect errors from comparison results (T058)
            stage_errors = self.error_detector.detect_errors(
                comparison_results=comparison_results,
                run_id=run_id,
                stage_name=stage_name,
                sample_ids=sample_ids[:len(stage_predictions)],
                predictions=stage_predictions,
                ground_truths=ground_truths,
                output_dir=output_dir
            )
            all_errors.extend(stage_errors)

            # Calculate metrics based on task type
            metrics = self._calculate_stage_metrics(
                stage_name=stage_name,
                task_type=task_type,
                predictions=stage_predictions,
                ground_truths=ground_truths,
                run_id=run_id
            )

            stage_metrics[stage_name] = metrics

        # Calculate overall accuracy
        all_metrics = [m for metrics in stage_metrics.values() for m in metrics]
        correct_total = sum(m.counts.correct for m in all_metrics)
        wrong_total = sum(m.counts.wrong for m in all_metrics)
        overall_accuracy = correct_total / (correct_total + wrong_total) if (correct_total + wrong_total) > 0 else 0.0

        # Generate HTML report
        report_path = output_dir / f"{run_id}_accuracy_report.html"
        self.report_generator.generate_per_run_report(
            run_id=run_id,
            dataset_name=self.dataset.name,
            stage_metrics=stage_metrics,
            output_path=report_path,
            error_browser_link=f"{run_id}_errors.html",
            historical_report_link="historical_report.html"
        )

        # Store metrics to Parquet (T037)
        metrics_path = self._store_metrics(run_id, stage_metrics, output_dir)

        # Cluster errors into patterns (T058)
        error_patterns = []
        if all_errors:
            clusters = self.pattern_detector.cluster_errors(all_errors)
            error_patterns = self.pattern_detector.summarize_patterns(clusters)

            # Store error metadata to JSON (T060)
            self._store_errors(run_id, all_errors, error_patterns, output_dir)

        # Record metrics for trend tracking (T080)
        self.performance_tracker.record_metrics(
            run_id=run_id,
            metrics=stage_metrics,
            dataset_id=self.dataset.dataset_id
        )

        return {
            'run_id': run_id,
            'overall_accuracy': overall_accuracy,
            'stage_metrics': stage_metrics,
            'report_path': report_path,
            'metrics_path': metrics_path,
            'correct': correct_total,
            'wrong': wrong_total,
            'errors': all_errors,
            'error_patterns': error_patterns
        }

    def _infer_task_types(self, predictions: Dict[str, List[Any]]) -> Dict[str, str]:
        """Infer task types from prediction structure.

        Args:
            predictions: Dict mapping stage name to predictions

        Returns:
            Dict mapping stage name to task type
        """
        task_types = {}

        for stage_name, preds in predictions.items():
            if not preds or len(preds) == 0:
                task_types[stage_name] = "classification"
                continue

            sample = preds[0]

            # Check prediction structure
            if isinstance(sample, dict):
                if 'boxes' in sample or 'bbox' in sample:
                    task_types[stage_name] = "detection"
                elif 'mask' in sample or 'segmentation' in sample:
                    task_types[stage_name] = "segmentation"
                else:
                    task_types[stage_name] = "classification"
            elif hasattr(sample, 'points'):  # Open3D PointCloud
                task_types[stage_name] = "pointcloud"
            else:
                task_types[stage_name] = "classification"

        return task_types

    def _calculate_stage_metrics(
        self,
        stage_name: str,
        task_type: str,
        predictions: List[Any],
        ground_truths: List[Any],
        run_id: str
    ) -> List[AccuracyMetrics]:
        """Calculate metrics for a single stage.

        Args:
            stage_name: Stage name
            task_type: Task type (detection, classification, segmentation, pointcloud)
            predictions: List of predictions
            ground_truths: List of ground truths
            run_id: Run identifier

        Returns:
            List of AccuracyMetrics
        """
        metrics_list = []

        if task_type == "classification":
            metrics = self.metrics_calculator.calculate_precision_recall_f1(
                predictions=predictions,
                ground_truths=ground_truths,
                stage_name=stage_name,
                run_id=run_id,
                ground_truth_ref=self.dataset.dataset_id
            )
            metrics_list.append(metrics)

        elif task_type == "detection":
            metrics = self.metrics_calculator.calculate_mean_average_precision(
                predictions=predictions,
                ground_truths=ground_truths,
                stage_name=stage_name,
                run_id=run_id,
                ground_truth_ref=self.dataset.dataset_id
            )
            metrics_list.append(metrics)

        elif task_type == "segmentation":
            metrics = self.metrics_calculator.calculate_iou(
                predictions=predictions,
                ground_truths=ground_truths,
                stage_name=stage_name,
                run_id=run_id,
                ground_truth_ref=self.dataset.dataset_id
            )
            metrics_list.append(metrics)

        elif task_type == "pointcloud":
            metrics = self.metrics_calculator.calculate_chamfer_distance(
                predictions=predictions,
                ground_truths=ground_truths,
                stage_name=stage_name,
                run_id=run_id,
                ground_truth_ref=self.dataset.dataset_id
            )
            metrics_list.append(metrics)

        return metrics_list

    def _store_metrics(
        self,
        run_id: str,
        stage_metrics: Dict[str, List[AccuracyMetrics]],
        output_dir: Path
    ) -> Path:
        """Store metrics to Parquet using PyArrow (T037).

        Args:
            run_id: Run identifier
            stage_metrics: Dict mapping stage to metrics list
            output_dir: Output directory

        Returns:
            Path to stored metrics file
        """
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            # Fall back to JSON if PyArrow not available
            import json
            metrics_path = output_dir / f"{run_id}_metrics.json"
            metrics_data = []
            for stage_name, metrics_list in stage_metrics.items():
                for m in metrics_list:
                    metrics_data.append({
                        "metrics_id": m.metrics_id,
                        "run_id": m.run_id,
                        "stage_name": m.stage_name,
                        "metric_type": m.metric_type.value,
                        "value": m.value,
                        "correct": m.counts.correct,
                        "wrong": m.counts.wrong,
                        "invalid": m.counts.invalid,
                        "unlabeled": m.counts.unlabeled,
                        "timestamp": m.timestamp.isoformat(),
                        "ground_truth_ref": m.ground_truth_ref
                    })
            metrics_path.write_text(json.dumps(metrics_data, indent=2))
            return metrics_path

        # Build Parquet data
        metrics_data = {
            "metrics_id": [],
            "run_id": [],
            "stage_name": [],
            "metric_type": [],
            "value": [],
            "correct": [],
            "wrong": [],
            "invalid": [],
            "unlabeled": [],
            "timestamp": [],
            "ground_truth_ref": []
        }

        for stage_name, metrics_list in stage_metrics.items():
            for m in metrics_list:
                metrics_data["metrics_id"].append(m.metrics_id)
                metrics_data["run_id"].append(m.run_id)
                metrics_data["stage_name"].append(m.stage_name)
                metrics_data["metric_type"].append(m.metric_type.value)
                metrics_data["value"].append(m.value)
                metrics_data["correct"].append(m.counts.correct)
                metrics_data["wrong"].append(m.counts.wrong)
                metrics_data["invalid"].append(m.counts.invalid)
                metrics_data["unlabeled"].append(m.counts.unlabeled)
                metrics_data["timestamp"].append(m.timestamp)
                metrics_data["ground_truth_ref"].append(m.ground_truth_ref)

        # Create PyArrow table
        table = pa.table(metrics_data)

        # Write to Parquet
        metrics_path = output_dir / f"{run_id}_metrics.parquet"
        output_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, str(metrics_path))

        return metrics_path

    def _store_errors(
        self,
        run_id: str,
        errors: List,
        patterns: List,
        output_dir: Path
    ) -> None:
        """Store error metadata to JSON files (T060).

        Args:
            run_id: Run identifier
            errors: List of ErrorCase instances
            patterns: List of ErrorPattern instances
            output_dir: Output directory
        """
        import json

        errors_dir = output_dir / "errors"
        errors_dir.mkdir(parents=True, exist_ok=True)

        # Store errors
        errors_file = errors_dir / f"{run_id}_errors.json"
        errors_data = []
        for error in errors:
            data = {
                'error_id': error.error_id,
                'run_id': error.run_id,
                'stage_name': error.stage_name,
                'sample_id': error.sample_id,
                'error_type': error.error_type.value,
                'severity': error.severity.value,
                'prediction': str(error.prediction),
                'ground_truth': str(error.ground_truth),
                'iou': error.iou,
                'confidence': error.confidence,
                'saved_artifacts': {k: str(v) for k, v in error.saved_artifacts.items()},
                'timestamp': error.timestamp.isoformat(),
                'metadata': error.metadata
            }
            errors_data.append(data)

        errors_file.write_text(json.dumps(errors_data, indent=2))

        # Store patterns
        patterns_file = errors_dir / f"{run_id}_patterns.json"
        patterns_data = []
        for pattern in patterns:
            data = {
                'pattern_id': pattern.pattern_id,
                'run_id': pattern.run_id,
                'stage_name': pattern.stage_name,
                'error_type': pattern.error_type.value,
                'severity': pattern.severity.value,
                'error_count': pattern.error_count,
                'affected_samples': pattern.affected_samples,
                'statistics': {
                    'avg_iou': pattern.statistics.avg_iou,
                    'avg_confidence': pattern.statistics.avg_confidence,
                    'min_iou': pattern.statistics.min_iou,
                    'max_iou': pattern.statistics.max_iou,
                    'sample_preview': pattern.statistics.sample_preview
                },
                'suggested_cause': pattern.suggested_cause,
                'timestamp': pattern.timestamp.isoformat()
            }
            patterns_data.append(data)

        patterns_file.write_text(json.dumps(patterns_data, indent=2))


__all__ = [
    "AccuracyCounts",
    "MetricType",
    "AccuracyMetrics",
    "GroundTruthDataset",
    "GroundTruthSample",
    "AnnotationFormat",
    "AccuracyTracker",
]
