"""
Performance visualization dashboard using Plotly.

T043-T050: PerformanceDashboard implementation
"""

from pathlib import Path
from typing import Optional
from datetime import datetime

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class PerformanceDashboard:
    """
    Visualization interface for performance data using Plotly charts.

    T043-T050: Complete dashboard implementation
    """

    def __init__(self, storage):
        """
        Initialize performance dashboard.

        T043: Constructor

        Args:
            storage: MetricsStorage instance
        """
        self.storage = storage

        if not PLOTLY_AVAILABLE:
            raise ImportError(
                "Plotly is required for dashboards. "
                "Install with: pip install plotly>=5.0"
            )

    def render_timing_chart(
        self,
        run_id: str,
        output_path: Optional[Path] = None,
    ) -> str:
        """
        Generate per-stage timing chart for a run.

        T044-T045: Timing chart with horizontal bars

        Args:
            run_id: Run to visualize
            output_path: Save HTML to this path (None = return HTML string)

        Returns:
            HTML string with embedded Plotly chart

        Raises:
            ValueError: If run_id not found
        """
        # T045: Load per-stage timing data
        metrics = self.storage.query_metrics(run_ids=[run_id])

        if not metrics:
            raise ValueError(f"No metrics found for run_id: {run_id}")

        # Group by stage and sum execution times
        stage_times = {}
        for metric in metrics:
            stage = metric['stage_name']
            time_ms = metric['execution_time_ms']
            stage_times[stage] = stage_times.get(stage, 0) + time_ms

        # T044: Create horizontal bar chart
        stages = list(stage_times.keys())
        times = list(stage_times.values())

        fig = go.Figure(data=[
            go.Bar(
                x=times,
                y=stages,
                orientation='h',
                marker=dict(color='steelblue'),
                text=[f"{t:.2f}ms" for t in times],
                textposition='auto',
            )
        ])

        fig.update_layout(
            title=f"Stage Execution Times - Run {run_id[:8]}",
            xaxis_title="Execution Time (ms)",
            yaxis_title="Stage",
            height=max(400, len(stages) * 60),
            template="plotly_white",
        )

        # T050: HTML export functionality
        html = pio.to_html(fig, include_plotlyjs='cdn', full_html=True)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(html)

        return html

    def render_memory_chart(
        self,
        run_id: str,
        output_path: Optional[Path] = None,
    ) -> str:
        """
        Generate memory usage chart (CPU and GPU) for a run.

        T046-T047: Memory chart with stacked area

        Args:
            run_id: Run to visualize
            output_path: Save HTML to this path (None = return HTML string)

        Returns:
            HTML string with embedded Plotly chart
        """
        # T047: Load memory data with null GPU handling
        metrics = self.storage.query_metrics(run_ids=[run_id])

        if not metrics:
            raise ValueError(f"No metrics found for run_id: {run_id}")

        stages = [m['stage_name'] for m in metrics]
        cpu_memory = [m['cpu_memory_mb'] for m in metrics]
        gpu_memory = [m.get('gpu_memory_mb', 0) or 0 for m in metrics]

        # T046: Create stacked area chart
        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='CPU Memory',
            x=stages,
            y=cpu_memory,
            marker=dict(color='lightblue'),
        ))

        if any(gpu_memory):
            fig.add_trace(go.Bar(
                name='GPU Memory',
                x=stages,
                y=gpu_memory,
                marker=dict(color='coral'),
            ))

        fig.update_layout(
            title=f"Memory Usage - Run {run_id[:8]}",
            xaxis_title="Stage",
            yaxis_title="Memory (MB)",
            barmode='group',
            height=500,
            template="plotly_white",
        )

        html = pio.to_html(fig, include_plotlyjs='cdn', full_html=True)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(html)

        return html

    def render_trend_chart(
        self,
        stage_name: str,
        start_date: datetime,
        end_date: datetime,
        output_path: Optional[Path] = None,
        aggregate_by: str = "none",
    ) -> str:
        """
        Generate historical trend chart for a stage.

        T048-T049: Trend chart with line plot
        T159-T162: Enhanced with aggregation and regression detection

        Args:
            stage_name: Stage to analyze
            start_date: Start of date range
            end_date: End of date range
            output_path: Save HTML to this path (None = return HTML string)
            aggregate_by: Aggregation period ('none', 'daily', 'weekly', 'monthly')

        Returns:
            HTML string with line chart showing execution time over time
        """
        # T048: Load date range and generate trend
        metrics = self.storage.query_metrics(
            stage_name=stage_name,
            start_date=start_date,
            end_date=end_date,
        )

        if not metrics:
            raise ValueError(
                f"No metrics found for stage {stage_name} "
                f"between {start_date} and {end_date}"
            )

        # Sort by timestamp
        import pandas as pd
        import numpy as np

        metrics = sorted(metrics, key=lambda m: m['timestamp'])
        df = pd.DataFrame(metrics)

        # T160: Implement date aggregation
        if aggregate_by != "none":
            if aggregate_by == "daily":
                df['date_group'] = df['timestamp'].dt.date
            elif aggregate_by == "weekly":
                df['date_group'] = df['timestamp'].dt.to_period('W').astype(str)
            elif aggregate_by == "monthly":
                df['date_group'] = df['timestamp'].dt.to_period('M').astype(str)
            else:
                raise ValueError(f"Invalid aggregate_by: {aggregate_by}")

            # Group and calculate statistics
            grouped = df.groupby('date_group')['execution_time_ms']
            timestamps = grouped.mean().index.tolist()
            exec_times = grouped.mean().tolist()
            std_devs = grouped.std().tolist()
        else:
            timestamps = df['timestamp'].tolist()
            exec_times = df['execution_time_ms'].tolist()
            std_devs = [0] * len(exec_times)

        # T049: Add trend line
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=timestamps,
            y=exec_times,
            mode='lines+markers',
            name='Execution Time',
            line=dict(color='steelblue', width=2),
            marker=dict(size=6),
        ))

        # T161: Add confidence interval bands (±1 std dev)
        if aggregate_by != "none" and any(std_devs):
            upper_bound = [t + s for t, s in zip(exec_times, std_devs)]
            lower_bound = [t - s for t, s in zip(exec_times, std_devs)]

            fig.add_trace(go.Scatter(
                x=timestamps + timestamps[::-1],
                y=upper_bound + lower_bound[::-1],
                fill='toself',
                fillcolor='rgba(70, 130, 180, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='±1 Std Dev',
                showlegend=True,
            ))

        # Calculate moving average for trend
        if len(exec_times) >= 3:
            window = min(5, len(exec_times) // 3)
            df_times = pd.DataFrame({'time': exec_times})
            moving_avg = df_times['time'].rolling(window=window, center=True).mean()

            fig.add_trace(go.Scatter(
                x=timestamps,
                y=moving_avg,
                mode='lines',
                name=f'{window}-point Moving Avg',
                line=dict(color='coral', width=2, dash='dash'),
            ))

        # T162: Regression detection highlighting (>10% degradation)
        if len(exec_times) >= 10:
            # Compare first 25% vs last 25%
            split_point = len(exec_times) // 4
            baseline_mean = np.mean(exec_times[:split_point])
            recent_mean = np.mean(exec_times[-split_point:])

            degradation_pct = ((recent_mean - baseline_mean) / baseline_mean) * 100

            if degradation_pct > 10:
                # Add annotation for regression
                fig.add_annotation(
                    x=timestamps[-1],
                    y=recent_mean,
                    text=f"⚠️ Performance degraded by {degradation_pct:.1f}%",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="red",
                    bgcolor="rgba(255, 0, 0, 0.1)",
                    bordercolor="red",
                    borderwidth=2,
                )

        fig.update_layout(
            title=f"Performance Trend - {stage_name}",
            xaxis_title="Time",
            yaxis_title="Execution Time (ms)",
            height=500,
            template="plotly_white",
            hovermode='x unified',
        )

        html = pio.to_html(fig, include_plotlyjs='cdn', full_html=True)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(html)

        return html
