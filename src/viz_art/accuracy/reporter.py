"""HTML report generation for accuracy tracking.

This module provides functionality for generating HTML reports with
per-stage accuracy metrics and Plotly visualizations.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

from jinja2 import Environment, FileSystemLoader
import plotly.graph_objects as go
import plotly.io as pio

from .metrics import AccuracyMetrics, MetricType


class ReportGenerator:
    """Generate HTML reports for accuracy tracking results."""

    def __init__(self):
        """Initialize report generator."""
        # Setup Jinja2 environment
        template_dir = Path(__file__).parent / "templates"
        self.env = Environment(loader=FileSystemLoader(str(template_dir)))

    def generate_per_run_report(
        self,
        run_id: str,
        dataset_name: str,
        stage_metrics: Dict[str, List[AccuracyMetrics]],
        output_path: Path,
        error_browser_link: Optional[str] = None,
        historical_report_link: Optional[str] = None,
    ) -> Path:
        """Generate per-run HTML report with accuracy metrics (T031).

        Args:
            run_id: Pipeline run identifier
            dataset_name: Ground truth dataset name
            stage_metrics: Dict mapping stage name to list of AccuracyMetrics
            output_path: Path to save HTML report
            error_browser_link: Link to error browser (optional)
            historical_report_link: Link to historical trends (optional)

        Returns:
            Path to generated HTML report
        """
        # Calculate overall statistics
        all_metrics = [m for metrics in stage_metrics.values() for m in metrics]

        correct_total = sum(m.counts.correct for m in all_metrics)
        wrong_total = sum(m.counts.wrong for m in all_metrics)
        invalid_total = sum(m.counts.invalid for m in all_metrics)
        unlabeled_total = sum(m.counts.unlabeled for m in all_metrics)
        total_samples = correct_total + wrong_total + invalid_total + unlabeled_total

        overall_accuracy = correct_total / (correct_total + wrong_total) if (correct_total + wrong_total) > 0 else 0.0

        # Prepare stage data
        stages = []
        for stage_name, metrics_list in stage_metrics.items():
            stage_data = self._prepare_stage_data(stage_name, metrics_list)
            stages.append(stage_data)

        # Render template
        template = self.env.get_template("per_run_report.html")
        html_content = template.render(
            run_id=run_id,
            dataset_name=dataset_name,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_samples=total_samples,
            overall_accuracy=overall_accuracy,
            correct_total=correct_total,
            wrong_total=wrong_total,
            invalid_total=invalid_total,
            stages=stages,
            error_browser_link=error_browser_link or "#",
            historical_report_link=historical_report_link or "#",
        )

        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html_content, encoding='utf-8')

        return output_path

    def _prepare_stage_data(
        self, stage_name: str, metrics_list: List[AccuracyMetrics]
    ) -> Dict[str, Any]:
        """Prepare stage data for template rendering.

        Args:
            stage_name: Name of pipeline stage
            metrics_list: List of AccuracyMetrics for this stage

        Returns:
            Dict with stage data for template
        """
        # Aggregate counts across all metrics for this stage
        total_counts = {
            "correct": sum(m.counts.correct for m in metrics_list),
            "wrong": sum(m.counts.wrong for m in metrics_list),
            "invalid": sum(m.counts.invalid for m in metrics_list),
            "unlabeled": sum(m.counts.unlabeled for m in metrics_list),
        }
        total = sum(total_counts.values())
        accuracy = total_counts["correct"] / (total_counts["correct"] + total_counts["wrong"]) if (total_counts["correct"] + total_counts["wrong"]) > 0 else 0.0

        # Build metrics list
        metrics = []
        for m in metrics_list:
            metric_info = {
                "name": m.metric_type.value.replace("_", " ").title(),
                "value": m.value,
                "type": self._get_metric_display_type(m.metric_type)
            }
            metrics.append(metric_info)

        # Generate Plotly chart for per-class metrics
        chart_data = None
        chart_title = None
        chart_xlabel = None
        chart_ylabel = None

        if metrics_list and metrics_list[0].per_class_values:
            chart_data, chart_title, chart_xlabel, chart_ylabel = self._create_per_class_chart(
                stage_name, metrics_list[0]
            )

        # Determine status and warnings
        status, warnings = self._check_stage_status(stage_name, metrics_list, accuracy)

        return {
            "name": stage_name,
            "counts": {
                "correct": total_counts["correct"],
                "wrong": total_counts["wrong"],
                "invalid": total_counts["invalid"],
                "unlabeled": total_counts["unlabeled"],
                "total": total,
                "accuracy": accuracy
            },
            "metrics": metrics,
            "chart_data": chart_data,
            "chart_title": chart_title,
            "chart_xlabel": chart_xlabel,
            "chart_ylabel": chart_ylabel,
            "status": status,
            "warnings": warnings
        }

    def _get_metric_display_type(self, metric_type: MetricType) -> str:
        """Get display type for metric value formatting."""
        if metric_type == MetricType.MAP:
            return "map"  # 0-100 scale
        elif metric_type == MetricType.CHAMFER_DISTANCE:
            return "distance"  # Raw value
        else:
            return "percentage"  # 0-1 scale shown as percentage

    def _create_per_class_chart(
        self, stage_name: str, metrics: AccuracyMetrics
    ) -> tuple:
        """Create Plotly chart for per-class metrics (T032).

        Args:
            stage_name: Stage name
            metrics: AccuracyMetrics with per_class_values

        Returns:
            Tuple of (chart_data_json, title, xlabel, ylabel)
        """
        per_class = metrics.per_class_values

        if not per_class:
            return None, None, None, None

        # Sort by class name
        classes = sorted(per_class.keys())
        values = [per_class[cls] for cls in classes]

        # Create bar chart
        trace = go.Bar(
            x=classes,
            y=values,
            marker=dict(
                color=values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=metrics.metric_type.value)
            ),
            text=[f"{v:.2f}" for v in values],
            textposition='auto',
        )

        # Convert to JSON for embedding in HTML
        # Create a Figure object, not just a list
        fig = go.Figure(data=[trace])

        return (
            json.loads(pio.to_json(fig)),
            f"Per-Class {metrics.metric_type.value.replace('_', ' ').title()}",
            "Class",
            metrics.metric_type.value.replace('_', ' ').title()
        )

    def _check_stage_status(
        self, stage_name: str, metrics_list: List[AccuracyMetrics], accuracy: float
    ) -> tuple:
        """Check stage status and generate warnings (T033).

        Args:
            stage_name: Stage name
            metrics_list: List of metrics for stage
            accuracy: Overall accuracy for stage

        Returns:
            Tuple of (status, warnings_list)
        """
        warnings = []

        # Define thresholds (can be configured)
        accuracy_threshold = 0.75  # 75%
        map_threshold = 60.0  # 60 mAP

        # Check accuracy threshold
        if accuracy < accuracy_threshold:
            warnings.append({
                "title": "Low Accuracy",
                "message": f"Stage accuracy ({accuracy*100:.1f}%) is below threshold ({accuracy_threshold*100:.0f}%)"
            })

        # Check mAP threshold for detection stages
        for m in metrics_list:
            if m.metric_type == MetricType.MAP and m.value < map_threshold:
                warnings.append({
                    "title": "Low mAP",
                    "message": f"Mean Average Precision ({m.value:.1f}) is below threshold ({map_threshold:.0f})"
                })

        # Determine overall status
        if len(warnings) == 0:
            status = "pass"
        elif accuracy < 0.5:
            status = "fail"
        else:
            status = "warn"

        return status, warnings

    def generate_historical_report(
        self,
        dataset_name: str,
        trend_data: Dict[str, Any],
        output_path: Path,
        regressions: Optional[List[Dict[str, Any]]] = None
    ) -> Path:
        """Generate historical trends HTML report (T074).

        Args:
            dataset_name: Dataset name
            trend_data: Historical trend data by stage
            output_path: Path to save HTML report
            regressions: List of regression alerts (optional)

        Returns:
            Path to generated HTML report
        """
        # Prepare stage data with charts
        stages = []

        for stage_name, stage_info in trend_data.items():
            # Create trend chart data (T075)
            trend_chart = self._create_trend_chart(stage_name, stage_info)

            # Create confusion matrix if available (T076)
            confusion_chart = None
            if 'confusion_matrix' in stage_info:
                confusion_chart = self._create_confusion_matrix_chart(
                    stage_info['confusion_matrix']
                )

            # Create per-class chart if available (T077)
            per_class_chart = None
            if 'per_class_values' in stage_info:
                per_class_chart = self._create_per_class_chart_historical(
                    stage_name, stage_info['per_class_values']
                )

            stages.append({
                'name': stage_name,
                'trend_data': trend_chart,
                'confusion_matrix_chart': confusion_chart,
                'per_class_chart': per_class_chart
            })

        # Extract metadata
        all_runs = []
        pipeline_versions = set()
        for stage_info in trend_data.values():
            if 'runs' in stage_info:
                all_runs.extend(stage_info['runs'])
            if 'pipeline_versions' in stage_info:
                pipeline_versions.update(stage_info['pipeline_versions'])

        total_runs = len(set(r['run_id'] for r in all_runs)) if all_runs else 0

        # Date range
        if all_runs:
            dates = [r['timestamp'] for r in all_runs if 'timestamp' in r]
            if dates:
                date_range = f"{min(dates).strftime('%Y-%m-%d')} to {max(dates).strftime('%Y-%m-%d')}"
            else:
                date_range = "N/A"
        else:
            date_range = "N/A"

        # Render template
        template = self.env.get_template("historical_report.html")
        html_content = template.render(
            dataset_name=dataset_name,
            total_runs=total_runs,
            date_range=date_range,
            pipeline_versions=sorted(pipeline_versions) if pipeline_versions else ['unknown'],
            stages=stages,
            regressions=regressions or [],
            run_comparison=[]  # Can be populated if needed
        )

        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html_content, encoding='utf-8')

        return output_path

    def _create_trend_chart(
        self, stage_name: str, stage_info: Dict[str, Any]
    ) -> list:
        """Create Plotly time-series chart for accuracy trends (T075).

        Args:
            stage_name: Stage name
            stage_info: Stage trend data

        Returns:
            Plotly chart data as JSON-compatible list
        """
        runs = stage_info.get('runs', [])

        if not runs:
            return []

        # Extract timestamps and values
        timestamps = [r['timestamp'] for r in runs]
        values = [r['value'] for r in runs]

        trace = go.Scatter(
            x=timestamps,
            y=values,
            mode='lines+markers',
            name=stage_name,
            line=dict(width=2),
            marker=dict(size=8)
        )

        fig = go.Figure(data=[trace])
        return json.loads(pio.to_json(fig))

    def _create_confusion_matrix_chart(
        self, confusion_matrix: Any
    ) -> list:
        """Create confusion matrix heatmap (T076).

        Args:
            confusion_matrix: Confusion matrix array

        Returns:
            Plotly heatmap data
        """
        import numpy as np

        if isinstance(confusion_matrix, list):
            confusion_matrix = np.array(confusion_matrix)

        trace = go.Heatmap(
            z=confusion_matrix,
            colorscale='Blues',
            showscale=True
        )

        fig = go.Figure(data=[trace])
        return json.loads(pio.to_json(fig))

    def _create_per_class_chart_historical(
        self, stage_name: str, per_class_values: Dict[str, float]
    ) -> list:
        """Create per-class performance breakdown chart for historical report.

        Args:
            stage_name: Stage name
            per_class_values: Dict mapping class name to metric value

        Returns:
            Plotly chart data
        """
        if not per_class_values:
            return []

        # Extract class names and values
        class_names = list(per_class_values.keys())
        values = list(per_class_values.values())

        trace = go.Bar(
            x=class_names,
            y=values,
            name=stage_name
        )

        fig = go.Figure(data=[trace])
        return json.loads(pio.to_json(fig))
