"""Performance trends and regression detection for accuracy tracking.

This module provides functionality for tracking accuracy metrics over time,
detecting regressions, and generating historical trend reports.
"""

from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import json

try:
    import pandas as pd
    import pyarrow.parquet as pq
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class PerformanceTracker:
    """Track accuracy metrics over time (T065-T066)."""

    def __init__(self, storage_dir: Path):
        """Initialize performance tracker.

        Args:
            storage_dir: Directory for storing trend data (Parquet files)
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        if not PANDAS_AVAILABLE:
            raise ImportError("Pandas required for trends. Install with: pip install pandas pyarrow")

    def record_metrics(
        self,
        run_id: str,
        metrics: Dict[str, Any],
        dataset_id: str,
        pipeline_version: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Record accuracy metrics for a run (T065).

        Args:
            run_id: Pipeline run identifier
            metrics: Dict of stage_name -> AccuracyMetrics
            dataset_id: Ground truth dataset identifier
            pipeline_version: Pipeline version (optional)
            timestamp: Run timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Flatten metrics for storage
        records = []

        from viz_art.accuracy import AccuracyMetrics
        for stage_name, metrics_list in metrics.items():
            if not isinstance(metrics_list, list):
                metrics_list = [metrics_list]

            for metric in metrics_list:
                if isinstance(metric, AccuracyMetrics):
                    record = {
                        'run_id': run_id,
                        'timestamp': timestamp,
                        'dataset_id': dataset_id,
                        'pipeline_version': pipeline_version or 'unknown',
                        'stage_name': stage_name,
                        'metric_type': metric.metric_type.value,
                        'value': metric.value,
                        'correct': metric.counts.correct,
                        'wrong': metric.counts.wrong,
                        'invalid': metric.counts.invalid,
                        'unlabeled': metric.counts.unlabeled,
                        'accuracy': metric.counts.accuracy
                    }
                    records.append(record)

        if not records:
            return

        # Append to Parquet file (partitioned by dataset_id)
        dataset_file = self.storage_dir / f"trends_{dataset_id}.parquet"

        df = pd.DataFrame(records)

        if dataset_file.exists():
            # Append to existing
            existing_df = pd.read_parquet(dataset_file)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.to_parquet(dataset_file, index=False, compression='snappy')
        else:
            # Create new
            df.to_parquet(dataset_file, index=False, compression='snappy')

    def get_trend(
        self,
        dataset_id: str,
        stage_name: str,
        metric_type: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get historical trend for a metric (T066).

        Args:
            dataset_id: Dataset identifier
            stage_name: Stage name
            metric_type: Metric type (e.g., 'f1_score', 'mean_average_precision')
            start_date: Filter >= this date
            end_date: Filter <= this date

        Returns:
            DataFrame with columns: timestamp, run_id, value
        """
        dataset_file = self.storage_dir / f"trends_{dataset_id}.parquet"

        if not dataset_file.exists():
            return pd.DataFrame(columns=['timestamp', 'run_id', 'value'])

        # Read with filters
        df = pd.read_parquet(dataset_file)

        # Filter by stage and metric type
        df = df[
            (df['stage_name'] == stage_name) &
            (df['metric_type'] == metric_type)
        ]

        # Apply date filters
        if start_date:
            df = df[df['timestamp'] >= start_date]
        if end_date:
            df = df[df['timestamp'] <= end_date]

        # Sort by timestamp
        df = df.sort_values('timestamp')

        return df[['timestamp', 'run_id', 'value', 'accuracy']]

    def get_all_runs(
        self,
        dataset_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[str]:
        """Get all run IDs for a dataset (T067).

        Args:
            dataset_id: Dataset identifier
            start_date: Filter >= this date
            end_date: Filter <= this date

        Returns:
            List of run IDs
        """
        dataset_file = self.storage_dir / f"trends_{dataset_id}.parquet"

        if not dataset_file.exists():
            return []

        df = pd.read_parquet(dataset_file)

        # Apply date filters
        if start_date:
            df = df[df['timestamp'] >= start_date]
        if end_date:
            df = df[df['timestamp'] <= end_date]

        return sorted(df['run_id'].unique().tolist())

    def compare_runs(
        self,
        dataset_id: str,
        run_id_1: str,
        run_id_2: str
    ) -> Dict[str, Dict[str, float]]:
        """Compare metrics between two runs (T068).

        Args:
            dataset_id: Dataset identifier
            run_id_1: First run ID
            run_id_2: Second run ID

        Returns:
            Dict mapping stage_name to metric deltas:
            {
                'stage_name': {
                    'metric_type': delta_value,
                    ...
                },
                ...
            }
        """
        dataset_file = self.storage_dir / f"trends_{dataset_id}.parquet"

        if not dataset_file.exists():
            return {}

        df = pd.read_parquet(dataset_file)

        # Get metrics for both runs
        run1_df = df[df['run_id'] == run_id_1]
        run2_df = df[df['run_id'] == run_id_2]

        # Compare by stage and metric type
        comparison = {}

        for stage in run1_df['stage_name'].unique():
            stage_run1 = run1_df[run1_df['stage_name'] == stage]
            stage_run2 = run2_df[run2_df['stage_name'] == stage]

            if stage not in comparison:
                comparison[stage] = {}

            for metric_type in stage_run1['metric_type'].unique():
                metric_run1 = stage_run1[stage_run1['metric_type'] == metric_type]
                metric_run2 = stage_run2[stage_run2['metric_type'] == metric_type]

                if len(metric_run1) > 0 and len(metric_run2) > 0:
                    value1 = metric_run1['value'].iloc[0]
                    value2 = metric_run2['value'].iloc[0]
                    delta = value2 - value1

                    comparison[stage][metric_type] = {
                        'value_1': float(value1),
                        'value_2': float(value2),
                        'delta': float(delta),
                        'delta_percent': float((delta / value1 * 100) if value1 != 0 else 0)
                    }

        return comparison


class RegressionDetector:
    """Detect accuracy regressions (T070-T072)."""

    def __init__(self, threshold: float = 0.05):
        """Initialize regression detector.

        Args:
            threshold: Regression threshold (e.g., 0.05 = 5% drop)
        """
        self.threshold = threshold

    def detect_regression(
        self,
        current_metrics: Dict[str, Any],
        baseline_metrics: Dict[str, Any],
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Detect regression between current and baseline (T070).

        Args:
            current_metrics: Current run metrics (stage -> AccuracyMetrics)
            baseline_metrics: Baseline run metrics
            threshold: Custom threshold (overrides default)

        Returns:
            Dict with regression info:
            {
                'regression_detected': bool,
                'affected_stages': List[str],
                'details': {stage: {metric: delta, ...}, ...}
            }
        """
        if threshold is None:
            threshold = self.threshold

        regression_detected = False
        affected_stages = []
        details = {}

        from viz_art.accuracy import AccuracyMetrics

        # Compare each stage
        for stage_name in current_metrics.keys():
            if stage_name not in baseline_metrics:
                continue

            current_list = current_metrics[stage_name]
            baseline_list = baseline_metrics[stage_name]

            if not isinstance(current_list, list):
                current_list = [current_list]
            if not isinstance(baseline_list, list):
                baseline_list = [baseline_list]

            stage_details = {}

            for current, baseline in zip(current_list, baseline_list):
                if isinstance(current, AccuracyMetrics) and isinstance(baseline, AccuracyMetrics):
                    # Compare values
                    delta = current.value - baseline.value
                    delta_percent = (delta / baseline.value) if baseline.value != 0 else 0

                    # Check for regression (negative delta exceeding threshold)
                    if delta_percent < -threshold:
                        regression_detected = True
                        if stage_name not in affected_stages:
                            affected_stages.append(stage_name)

                        stage_details[current.metric_type.value] = {
                            'current': current.value,
                            'baseline': baseline.value,
                            'delta': delta,
                            'delta_percent': delta_percent
                        }

            if stage_details:
                details[stage_name] = stage_details

        return {
            'regression_detected': regression_detected,
            'affected_stages': affected_stages,
            'details': details,
            'threshold': threshold
        }

    def get_baseline_strategy(
        self,
        dataset_id: str,
        tracker: PerformanceTracker,
        strategy: str = "best"
    ) -> Optional[str]:
        """Get baseline run ID using specified strategy (T071).

        Args:
            dataset_id: Dataset identifier
            tracker: PerformanceTracker instance
            strategy: Baseline strategy ("best", "latest", "mean", "median")

        Returns:
            Baseline run ID, or None if no runs available
        """
        dataset_file = tracker.storage_dir / f"trends_{dataset_id}.parquet"

        if not dataset_file.exists():
            return None

        df = pd.read_parquet(dataset_file)

        if len(df) == 0:
            return None

        if strategy == "best":
            # Find run with highest overall accuracy
            accuracy_by_run = df.groupby('run_id')['accuracy'].mean()
            return accuracy_by_run.idxmax()

        elif strategy == "latest":
            # Most recent run
            latest_timestamp = df['timestamp'].max()
            latest_run = df[df['timestamp'] == latest_timestamp]['run_id'].iloc[0]
            return latest_run

        elif strategy == "mean":
            # Run closest to mean accuracy
            accuracy_by_run = df.groupby('run_id')['accuracy'].mean()
            mean_accuracy = accuracy_by_run.mean()
            closest_run = (accuracy_by_run - mean_accuracy).abs().idxmin()
            return closest_run

        elif strategy == "median":
            # Run closest to median accuracy
            accuracy_by_run = df.groupby('run_id')['accuracy'].mean()
            median_accuracy = accuracy_by_run.median()
            closest_run = (accuracy_by_run - median_accuracy).abs().idxmin()
            return closest_run

        return None

    def create_alert(
        self,
        regression_result: Dict[str, Any],
        run_id: str,
        baseline_run_id: str
    ) -> Dict[str, Any]:
        """Create regression alert (T072).

        Args:
            regression_result: Result from detect_regression()
            run_id: Current run ID
            baseline_run_id: Baseline run ID

        Returns:
            Alert dict with formatted message
        """
        if not regression_result['regression_detected']:
            return {
                'alert': False,
                'message': f"No regression detected in run {run_id}"
            }

        affected = regression_result['affected_stages']
        details = regression_result['details']

        # Build alert message
        message_lines = [
            f"⚠️ Regression Alert: Run {run_id}",
            f"Baseline: {baseline_run_id}",
            f"Affected stages: {', '.join(affected)}",
            ""
        ]

        for stage, metrics in details.items():
            message_lines.append(f"**{stage}**:")
            for metric_name, metric_data in metrics.items():
                delta_pct = metric_data['delta_percent'] * 100
                message_lines.append(
                    f"  - {metric_name}: {metric_data['baseline']:.3f} → {metric_data['current']:.3f} "
                    f"({delta_pct:+.1f}%)"
                )
            message_lines.append("")

        return {
            'alert': True,
            'severity': 'high' if len(affected) > 1 else 'medium',
            'message': '\n'.join(message_lines),
            'affected_stages': affected,
            'details': details
        }
