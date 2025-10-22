"""Point cloud visualization stage with interactive Plotly 3D viewer."""

from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np
from viz_art.pipeline.stage import PipelineStage
from viz_art.types.pointcloud import VisualizationConfig
from viz_art.visualization.plotly_viewer import (
    create_3d_scatter,
    downsample_for_display,
    save_html,
    save_json,
)


class PointCloudVisualizationStage(PipelineStage):
    """Interactive 3D point cloud visualization with Plotly WebGL rendering.

    This stage creates interactive 3D visualizations of point clouds with
    rotation, zoom, and pan controls. Automatically downsamples large clouds
    for WebGL performance.

    Args:
        name: Stage identifier
        config: Visualization configuration (point size, colors, etc.)
        output_html: Path to save HTML viewer (None = don't save)
        output_json: Path to save JSON data (None = don't save)
        color_mode: Override color mode ("height", "intensity", "rgb")
    """

    def __init__(
        self,
        name: str = "pointcloud_visualization",
        config: Optional[VisualizationConfig] = None,
        output_html: Optional[str] = None,
        output_json: Optional[str] = None,
        color_mode: Optional[str] = None,
    ):
        self._name = name
        self.config = config or VisualizationConfig()
        self.output_html = output_html
        self.output_json = output_json

        # Override color mode if specified
        if color_mode is not None:
            self.config = VisualizationConfig(
                **{**self.config.__dict__, 'color_mode': color_mode}
            )

    @property
    def name(self) -> str:
        return self._name

    @property
    def input_keys(self) -> List[str]:
        return ["points"]

    @property
    def output_keys(self) -> List[str]:
        return ["figure", "html_path", "json_path", "num_rendered_points", "was_downsampled"]

    @property
    def input_data_types(self) -> Dict[str, type]:
        """Expected input types."""
        return {
            "points": np.ndarray,
            "colors": np.ndarray,  # Optional
            "intensity": np.ndarray,  # Optional
        }

    @property
    def output_data_types(self) -> Dict[str, type]:
        """Expected output types."""
        return {
            "figure": object,  # go.Figure
            "html_path": str,
            "json_path": str,
            "num_rendered_points": int,
            "was_downsampled": bool,
        }

    def pre_process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate inputs and prepare for visualization.

        Args:
            inputs: Must contain "points" (Nx3 array)

        Returns:
            Validated inputs

        Raises:
            ValueError: If points are missing or invalid
        """
        if "points" not in inputs:
            raise ValueError("Input must contain 'points' key")

        points = inputs["points"]

        if not isinstance(points, np.ndarray):
            raise ValueError(f"Points must be NumPy array, got {type(points)}")

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Points must be Nx3 array, got shape {points.shape}")

        if points.shape[0] == 0:
            raise ValueError("Cannot visualize empty point cloud")

        return inputs

    def predict(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        """Generate 3D visualization with automatic downsampling.

        Args:
            preprocessed: Validated inputs with points, colors, intensity

        Returns:
            Dictionary with figure and rendering metadata
        """
        points = preprocessed["points"]
        colors = preprocessed.get("colors")
        intensity = preprocessed.get("intensity")

        original_count = points.shape[0]
        was_downsampled = False

        # Auto-downsample for large point clouds
        if self.config.should_downsample(original_count):
            points, colors, intensity = downsample_for_display(
                points,
                colors,
                intensity,
                max_points=self.config.max_render_points
            )
            was_downsampled = True

        # Create 3D scatter plot
        fig = create_3d_scatter(
            points,
            colors=colors,
            intensity=intensity,
            config=self.config,
        )

        return {
            "figure": fig,
            "num_rendered_points": points.shape[0],
            "was_downsampled": was_downsampled,
            "original_count": original_count,
        }

    def post_process(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Save visualization outputs if paths specified.

        Args:
            predictions: Dictionary with figure and metadata

        Returns:
            Dictionary with output paths and metadata
        """
        fig = predictions["figure"]
        html_path = None
        json_path = None

        # Save HTML if requested
        if self.output_html:
            html_path = save_html(fig, self.output_html)

        # Save JSON if requested
        if self.output_json:
            json_path = save_json(fig, self.output_json)

        return {
            "figure": fig,
            "html_path": html_path,
            "json_path": json_path,
            "num_rendered_points": predictions["num_rendered_points"],
            "was_downsampled": predictions["was_downsampled"],
            "visualization_config": self.config,
        }
