"""HTML report generation for batch processing results.

This module provides the HTMLReporter class for generating static HTML reports
from batch processing results using Jinja2 templates.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import numpy as np

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
        output_dir = report_path.parent.resolve()

        for run in batch_result.run_results:
            # Extract filename from input path
            image_path = run.inputs.get("image_path", run.inputs.get("pointcloud_path", ""))
            filename = Path(image_path).name

            # Get saved paths if available
            saved_paths = run.outputs.get('_saved_paths', {})

            # Get all stage outputs
            for stage_name, outputs in run.outputs.items():
                if stage_name.startswith('_'):  # Skip metadata
                    continue

                if stage_name not in stages_data:
                    stages_data[stage_name] = []

                # Get image/pointcloud path from saved outputs
                image_path_rel = None
                thumbnail_path_rel = None
                viewer_path_rel = None
                is_pointcloud = False

                if saved_paths and stage_name in saved_paths:
                    stage_saved = saved_paths[stage_name]
                    # Get first image key
                    for key, path in stage_saved.items():
                        if isinstance(path, str) and path.endswith(('.png', '.jpg', '.jpeg')):
                            # Make relative to report location (both in output_dir)
                            try:
                                abs_path = Path(path).resolve()
                                rel_path = abs_path.relative_to(output_dir)
                                image_path_rel = str(rel_path)
                            except Exception as e:
                                # Fallback to absolute path if relative doesn't work
                                logger.debug(f"Could not create relative path: {e}, using absolute")
                                image_path_rel = str(Path(path).resolve())
                            break

                # Check if output contains point cloud data
                if isinstance(outputs, dict):
                    for key, value in outputs.items():
                        if self._is_pointcloud_output(value):
                            is_pointcloud = True
                            # Generate thumbnail and viewer for point cloud
                            filename_prefix = f"{stage_name}_{Path(filename).stem}"

                            thumbnail_rel = self._render_pointcloud_thumbnail(
                                value, output_dir, filename_prefix
                            )
                            if thumbnail_rel:
                                thumbnail_path_rel = thumbnail_rel

                            viewer_rel = self._create_plotly_viewer(
                                value, output_dir, filename_prefix
                            )
                            if viewer_rel:
                                viewer_path_rel = viewer_rel

                            break  # Only process first point cloud in stage

                # Create image data entry
                image_data = {
                    "filename": filename,
                    "success": True,
                    "image_path": image_path_rel,
                    "stage_name": stage_name,
                    "is_pointcloud": is_pointcloud,
                    "thumbnail_path": thumbnail_path_rel,
                    "viewer_path": viewer_path_rel,
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

    def _is_pointcloud_output(self, output_value: Any) -> bool:
        """Check if output value is a point cloud.

        Args:
            output_value: Output value to check

        Returns:
            True if output is a point cloud, False otherwise
        """
        # Check for Open3D PointCloud
        try:
            import open3d as o3d
            if isinstance(output_value, o3d.geometry.PointCloud):
                return True
        except ImportError:
            pass

        # Check for NumPy array that looks like point cloud (Nx3 shape)
        if isinstance(output_value, np.ndarray):
            if output_value.ndim == 2 and output_value.shape[1] == 3:
                # Additional check: should have reasonable number of points
                if 10 <= output_value.shape[0] <= 10_000_000:
                    return True

        return False

    def _render_pointcloud_thumbnail(
        self,
        output_value: Any,
        output_dir: Path,
        filename_prefix: str,
        progress_callback: Optional[callable] = None,
    ) -> Optional[str]:
        """Render thumbnail image for point cloud output.

        Args:
            output_value: Point cloud data (Open3D or NumPy array)
            output_dir: Directory to save thumbnail
            filename_prefix: Prefix for thumbnail filename
            progress_callback: Optional callback for progress updates

        Returns:
            Relative path to thumbnail image, or None if rendering failed
        """
        try:
            from viz_art.visualization.thumbnail import (
                render_thumbnail_with_cache,
                save_thumbnail,
            )
            import open3d as o3d

            # Convert to NumPy array if needed
            points = None
            colors = None

            if isinstance(output_value, o3d.geometry.PointCloud):
                points = np.asarray(output_value.points)
                if output_value.has_colors():
                    colors = np.asarray(output_value.colors)
            elif isinstance(output_value, np.ndarray):
                points = output_value
                colors = None

            if points is None or points.shape[0] == 0:
                logger.warning(f"Empty point cloud, skipping thumbnail for {filename_prefix}")
                return None

            # Progress indication (T079)
            if progress_callback:
                progress_callback(f"Rendering thumbnail for {filename_prefix}...")

            # Setup cache directory
            cache_dir = output_dir / ".thumbnail_cache"

            # Render thumbnail with caching and auto-downsampling (T077, T080, T081)
            thumbnail_img = render_thumbnail_with_cache(
                points,
                colors=colors,
                width=800,
                height=600,
                viewpoint="diagonal",
                point_size=2.0,
                background_color=(1.0, 1.0, 1.0),
                cache_dir=cache_dir,
                max_points_for_rendering=100000,  # Auto-downsample large clouds
            )

            # Save thumbnail
            thumbnail_filename = f"{filename_prefix}_thumbnail.png"
            thumbnail_path = output_dir / "thumbnails" / thumbnail_filename
            thumbnail_path.parent.mkdir(parents=True, exist_ok=True)

            save_thumbnail(thumbnail_img, str(thumbnail_path))

            # Return relative path
            rel_path = Path("thumbnails") / thumbnail_filename
            return str(rel_path)

        except Exception as e:
            logger.warning(f"Failed to render point cloud thumbnail: {e}")
            return None

    def _create_plotly_viewer(
        self,
        output_value: Any,
        output_dir: Path,
        filename_prefix: str,
    ) -> Optional[str]:
        """Create interactive Plotly viewer for point cloud.

        Args:
            output_value: Point cloud data (Open3D or NumPy array)
            output_dir: Directory to save viewer HTML
            filename_prefix: Prefix for viewer filename

        Returns:
            Relative path to viewer HTML file, or None if creation failed
        """
        try:
            from viz_art.visualization.plotly_viewer import create_3d_scatter, save_html
            from viz_art.types.pointcloud import VisualizationConfig
            import open3d as o3d

            # Convert to NumPy array if needed
            points = None
            colors = None

            if isinstance(output_value, o3d.geometry.PointCloud):
                points = np.asarray(output_value.points)
                if output_value.has_colors():
                    colors = np.asarray(output_value.colors)
            elif isinstance(output_value, np.ndarray):
                points = output_value
                colors = None

            if points is None or points.shape[0] == 0:
                logger.warning(f"Empty point cloud, skipping viewer for {filename_prefix}")
                return None

            # Create Plotly figure
            config = VisualizationConfig(
                point_size=2.0,
                opacity=0.8,
                color_mode="height",
                colorscale="Viridis",
            )

            fig = create_3d_scatter(points, colors=colors, config=config)

            # Save as HTML
            viewer_filename = f"{filename_prefix}_viewer.html"
            viewer_path = output_dir / "viewers" / viewer_filename
            viewer_path.parent.mkdir(parents=True, exist_ok=True)

            save_html(fig, str(viewer_path))

            # Return relative path
            rel_path = Path("viewers") / viewer_filename
            return str(rel_path)

        except Exception as e:
            logger.warning(f"Failed to create Plotly viewer: {e}")
            return None

    def embed_plotly_viewer(self, fig_html_path: str) -> str:
        """Generate HTML to embed a Plotly viewer inline.

        Args:
            fig_html_path: Relative path to Plotly HTML file

        Returns:
            HTML string with iframe embed code
        """
        return f'''
        <div class="plotly-viewer-container">
            <iframe src="{fig_html_path}"
                    style="width: 100%; height: 600px; border: 1px solid #e5e7eb; border-radius: 6px;"
                    frameborder="0">
            </iframe>
        </div>
        '''
