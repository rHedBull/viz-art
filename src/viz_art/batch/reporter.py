"""HTML report generation for batch processing results.

This module provides the HTMLReporter class for generating static HTML reports
from batch processing results using Jinja2 templates.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

from jinja2 import Environment, FileSystemLoader, select_autoescape
from viz_art.pipeline.results import BatchResult

logger = logging.getLogger(__name__)


class HTMLReporter:
    """Generate static HTML reports from batch processing results.

    The HTMLReporter uses Jinja2 templates to create self-contained HTML reports
    that can be viewed offline in any modern browser. Images are referenced via
    relative paths (not base64 encoded) for better performance.

    Example:
        >>> from viz_art.batch.processor import BatchProcessor
        >>> batch_result = processor.run()
        >>> reporter = HTMLReporter()
        >>> report_path = reporter.generate(batch_result, "report.html")
        >>> print(f"Report generated: {report_path}")

    Attributes:
        template_dir: Directory containing Jinja2 templates
        env: Jinja2 Environment for template rendering
    """

    DEFAULT_TEMPLATE = "batch_report.html"

    def __init__(self, template_dir: str | Path | None = None):
        """Initialize HTML reporter.

        Args:
            template_dir: Directory containing Jinja2 templates.
                         If None, uses package default templates.

        Raises:
            ValueError: If template_dir doesn't exist
        """
        if template_dir is None:
            # Use package default templates
            package_dir = Path(__file__).parent
            template_dir = package_dir / "templates"

        self.template_dir = Path(template_dir)

        if not self.template_dir.exists():
            raise ValueError(f"Template directory does not exist: {self.template_dir}")

        # Setup Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=select_autoescape(["html", "xml"]),
        )

        # Register custom filters
        self.env.filters["basename"] = lambda p: Path(p).name

        logger.debug(f"HTMLReporter initialized with templates from {self.template_dir}")

    def generate(
        self,
        batch_result: BatchResult,
        output_path: str | Path,
        pipeline_name: str = "Pipeline",
    ) -> Path:
        """Generate HTML report from batch results.

        Args:
            batch_result: BatchResult from BatchProcessor.run()
            output_path: Path for output HTML file
            pipeline_name: Name of the pipeline for display

        Returns:
            Absolute path to generated HTML file

        Raises:
            ValueError: If batch_result is None
            RuntimeError: If template rendering fails

        Report Structure:
            - Summary statistics (total, successful, failed, duration)
            - Stage-grouped view (all images organized by stage)
            - Per-image complete pipeline view
            - Error section with failure details
        """
        if batch_result is None:
            raise ValueError("batch_result cannot be None")

        output_path = Path(output_path)
        logger.info(f"Generating HTML report: {output_path}")

        try:
            # Load template
            template = self.env.get_template(self.DEFAULT_TEMPLATE)

            # Prepare data for template
            context = self._prepare_context(batch_result, pipeline_name, output_path)

            # Render template
            html_content = template.render(**context)

            # Write to file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(html_content, encoding="utf-8")

            logger.info(f"Report generated successfully: {output_path}")
            return output_path.absolute()

        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            raise RuntimeError(f"Report generation failed: {e}") from e

    def _prepare_context(
        self, batch_result: BatchResult, pipeline_name: str, report_path: Path
    ) -> Dict[str, Any]:
        """Prepare template context from batch result.

        Args:
            batch_result: BatchResult to process
            pipeline_name: Pipeline name for display
            report_path: Path where the report will be written

        Returns:
            Dictionary with template variables
        """
        # Calculate duration
        duration = (batch_result.completed_at - batch_result.started_at).total_seconds()

        # Organize data by stage for stage-grouped view
        stages_data = self._organize_by_stage(batch_result, report_path)

        context = {
            "batch_result": batch_result,
            "pipeline_name": pipeline_name,
            "duration_seconds": duration,
            "stages_data": stages_data,
        }

        return context

    def _organize_by_stage(self, batch_result: BatchResult, report_path: Path) -> Dict[str, List[Dict[str, Any]]]:
        """Organize run results by stage for stage-grouped view.

        Args:
            batch_result: BatchResult to organize
            report_path: Path where the report will be written

        Returns:
            Dictionary mapping stage_name -> list of image data
        """
        stages_data: Dict[str, List[Dict[str, Any]]] = {}

        for run in batch_result.run_results:
            # Extract filename from input path
            image_path = run.inputs.get("image_path", "")
            filename = Path(image_path).name

            # Get saved paths if available
            saved_paths = run.outputs.get('_saved_paths', {})

            # Get all stage outputs
            for stage_name, outputs in run.outputs.items():
                if stage_name.startswith('_'):  # Skip metadata
                    continue

                if stage_name not in stages_data:
                    stages_data[stage_name] = []

                # Get image path from saved outputs
                image_path_rel = None
                if saved_paths and stage_name in saved_paths:
                    stage_saved = saved_paths[stage_name]
                    # Get first image key
                    for key, path in stage_saved.items():
                        if isinstance(path, str) and path.endswith(('.png', '.jpg', '.jpeg')):
                            # Make relative to report location (both in output_dir)
                            try:
                                abs_path = Path(path).resolve()
                                report_dir = report_path.parent.resolve()
                                rel_path = abs_path.relative_to(report_dir)
                                image_path_rel = str(rel_path)
                            except Exception as e:
                                # Fallback to absolute path if relative doesn't work
                                logger.debug(f"Could not create relative path: {e}, using absolute")
                                image_path_rel = str(Path(path).resolve())
                            break

                # Create image data entry
                image_data = {
                    "filename": filename,
                    "success": True,
                    "image_path": image_path_rel,
                    "stage_name": stage_name,
                }

                stages_data[stage_name].append(image_data)

            # Handle failed runs
            if run.error:
                # Add to each stage as failed
                for stage_name in stages_data.keys():
                    # Check if already added
                    existing = [
                        img for img in stages_data[stage_name] if img["filename"] == filename
                    ]
                    if not existing:
                        stages_data[stage_name].append(
                            {
                                "filename": filename,
                                "success": False,
                                "image_path": None,
                                "stage_name": stage_name,
                            }
                        )

        return stages_data
