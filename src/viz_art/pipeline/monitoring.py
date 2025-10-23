"""Monitoring integration for Pipeline.

Provides decorators and wrappers to add performance profiling, audit logging,
and validation to pipeline execution without modifying the core Pipeline class.
"""

from pathlib import Path
from typing import Any, Callable, Dict, Optional
from functools import wraps
from datetime import datetime

from ..config import MonitoringConfig


class MonitoredPipeline:
    """Wrapper that adds monitoring to an existing Pipeline.

    This class wraps a Pipeline instance and adds optional monitoring features:
    - Performance profiling (timing and memory)
    - Audit logging (structured logs)
    - Ground truth validation

    Example:
        >>> from viz_art.pipeline import Pipeline
        >>> from viz_art.pipeline.monitoring import MonitoredPipeline
        >>> from viz_art.config import MonitoringConfig
        >>>
        >>> config = MonitoringConfig(
        ...     enable_profiling=True,
        ...     enable_audit_logging=True,
        ...     output_dir=Path("output")
        ... )
        >>>
        >>> pipeline = Pipeline(name="my-pipeline")
        >>> # ... add stages ...
        >>>
        >>> monitored = MonitoredPipeline(pipeline, config)
        >>> results = monitored.run(image_path="test.jpg")
    """

    def __init__(self, pipeline, config: MonitoringConfig):
        """Initialize monitored pipeline wrapper.

        Args:
            pipeline: Existing Pipeline instance to wrap
            config: Monitoring configuration
        """
        self.pipeline = pipeline
        self.config = config

        # Initialize monitoring components if enabled
        self._profiler = None
        self._metrics_storage = None
        self._run_tracker = None
        self._logger = None
        self._dashboard = None
        self._ground_truth_dataset = None

        self._setup_monitoring()

    def _setup_monitoring(self):
        """Setup monitoring components based on configuration."""
        output_dir = self.config.output_dir

        # Setup performance profiling
        if self.config.enable_profiling:
            from ..performance import create_profiler, create_metrics_storage, create_dashboard

            self._metrics_storage = create_metrics_storage(
                output_dir / "metrics",
                retention_days=self.config.metrics_retention_days,
            )
            self._profiler = create_profiler(self._metrics_storage, enabled=True)
            self._dashboard = create_dashboard(self._metrics_storage)

        # Setup audit logging
        if self.config.enable_audit_logging:
            from ..audit import create_run_tracker

            self._run_tracker = create_run_tracker(output_dir / "runs")

        # Setup validation
        if self.config.enable_validation and self.config.ground_truth_dataset:
            from ..validation import create_dataset
            from ..types.monitoring import AnnotationFormat

            # Load ground truth dataset
            # Note: This assumes dataset is already configured somewhere
            # In production, you'd load from a dataset registry
            self._ground_truth_dataset = self.config.ground_truth_dataset

    def run(self, **inputs) -> Dict[str, Any]:
        """Execute pipeline with monitoring.

        Args:
            **inputs: Pipeline inputs

        Returns:
            Dictionary with pipeline outputs and monitoring metadata
        """
        # If no monitoring enabled, just run the pipeline
        if not any([
            self.config.enable_profiling,
            self.config.enable_audit_logging,
            self.config.enable_validation,
        ]):
            return self.pipeline.run(**inputs)

        # Execute with monitoring
        if self._run_tracker:
            # Use run tracker context
            config_snapshot = {
                "pipeline_name": self.pipeline.name,
                "enable_profiling": self.config.enable_profiling,
                "enable_validation": self.config.enable_validation,
            }

            input_files = [
                Path(v) for v in inputs.values()
                if isinstance(v, (str, Path)) and Path(v).exists()
            ]

            with self._run_tracker.track(
                config=config_snapshot,
                input_files=input_files,
                output_dir=self.config.output_dir / "runs",
            ) as run_id:
                # Create logger for this run
                if self.config.enable_audit_logging:
                    from ..audit import create_logger

                    self._logger = create_logger(
                        run_id,
                        self.config.output_dir / "logs",
                        retention=f"{self.config.log_retention_days} days",
                    )
                    self._logger.info("Pipeline started", inputs=list(inputs.keys()))

                # Execute pipeline
                try:
                    results = self._execute_with_profiling(run_id, **inputs)

                    if self._logger:
                        self._logger.info("Pipeline completed successfully")

                    # Generate performance dashboard
                    if self._dashboard and self.config.enable_profiling:
                        dashboard_path = (
                            self.config.output_dir / "runs" / run_id / "performance_dashboard.html"
                        )
                        dashboard_path.parent.mkdir(parents=True, exist_ok=True)

                        try:
                            self._dashboard.render_timing_chart(
                                run_id=run_id,
                                output_path=dashboard_path,
                            )
                        except Exception as e:
                            if self._logger:
                                self._logger.warning(f"Failed to generate dashboard: {e}")

                    # Add monitoring metadata
                    results["_monitoring"] = {
                        "run_id": run_id,
                        "profiling_enabled": self.config.enable_profiling,
                        "logging_enabled": self.config.enable_audit_logging,
                    }

                    return results

                except Exception as e:
                    if self._logger:
                        self._logger.error(f"Pipeline failed: {str(e)}", error_type=type(e).__name__)
                    raise

        else:
            # Run without run tracking (just profiling)
            return self._execute_with_profiling(None, **inputs)

    def _execute_with_profiling(self, run_id: Optional[str], **inputs) -> Dict[str, Any]:
        """Execute pipeline with optional profiling.

        Args:
            run_id: Optional run identifier
            **inputs: Pipeline inputs

        Returns:
            Pipeline outputs
        """
        if not self.config.enable_profiling or not self._profiler:
            # No profiling, just run normally
            return self.pipeline.run(**inputs)

        # Wrap stage methods with profiler
        original_methods = {}

        for stage in self.pipeline._stages:
            # Save original methods
            original_methods[stage.name] = {
                "predict": stage.predict,
            }

            # Wrap with profiler
            stage.predict = self._profiler(stage.predict)

        try:
            # Execute pipeline
            results = self.pipeline.run(**inputs)
            return results

        finally:
            # Restore original methods
            for stage in self.pipeline._stages:
                if stage.name in original_methods:
                    stage.predict = original_methods[stage.name]["predict"]

    # Delegate all other methods to wrapped pipeline
    def __getattr__(self, name):
        """Delegate attribute access to wrapped pipeline."""
        return getattr(self.pipeline, name)


# Convenience function for adding monitoring to a pipeline
def add_monitoring(pipeline, config: Optional[MonitoringConfig] = None) -> MonitoredPipeline:
    """Add monitoring to an existing pipeline.

    Args:
        pipeline: Pipeline instance to monitor
        config: Monitoring configuration (default: all disabled)

    Returns:
        MonitoredPipeline wrapper with monitoring enabled

    Example:
        >>> pipeline = Pipeline(name="my-pipeline")
        >>> # ... add stages ...
        >>> config = MonitoringConfig(enable_profiling=True)
        >>> monitored = add_monitoring(pipeline, config)
        >>> results = monitored.run(image_path="test.jpg")
    """
    if config is None:
        config = MonitoringConfig()

    return MonitoredPipeline(pipeline, config)
