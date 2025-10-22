"""Point cloud thumbnail generation stage for batch reports."""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np
from viz_art.pipeline.stage import PipelineStage
from viz_art.types.pointcloud import VisualizationConfig
from viz_art.visualization.thumbnail import (
    render_thumbnail,
    render_multi_view_thumbnail,
    save_thumbnail,
)


class PointCloudThumbnailStage(PipelineStage):
    """Generate static thumbnail images from point clouds for batch reports.

    This stage creates rendered thumbnail images from multiple viewpoints
    using Open3D's OffscreenRenderer. Thumbnails are useful for quickly
    previewing point clouds in HTML reports without requiring interactive viewers.

    Args:
        name: Stage identifier
        viewpoints: Tuple of viewpoint names ("front", "top", "side", "diagonal")
        width: Thumbnail width (pixels)
        height: Thumbnail height (pixels)
        point_size: Point size in pixels
        output_dir: Directory to save thumbnails (None = don't save)
        background_color: RGB background color in [0,1]
    """

    def __init__(
        self,
        name: str = "pointcloud_thumbnail",
        viewpoints: Tuple[str, ...] = ("diagonal",),
        width: int = 800,
        height: int = 600,
        point_size: float = 2.0,
        output_dir: Optional[str] = None,
        background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        self._name = name
        self.viewpoints = viewpoints
        self.width = width
        self.height = height
        self.point_size = point_size
        self.output_dir = output_dir
        self.background_color = background_color

    @property
    def name(self) -> str:
        return self._name

    @property
    def input_keys(self) -> List[str]:
        return ["points"]

    @property
    def output_keys(self) -> List[str]:
        return ["thumbnails", "thumbnail_paths", "viewpoints"]

    @property
    def input_data_types(self) -> Dict[str, type]:
        """Expected input types."""
        return {
            "points": np.ndarray,
            "colors": np.ndarray,  # Optional
        }

    @property
    def output_data_types(self) -> Dict[str, type]:
        """Expected output types."""
        return {
            "thumbnails": dict,  # Dict[str, np.ndarray]
            "thumbnail_paths": dict,  # Dict[str, str]
            "viewpoints": list,  # List[str]
        }

    def pre_process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate inputs.

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
            raise ValueError("Cannot generate thumbnail from empty point cloud")

        return inputs

    def predict(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        """Generate thumbnails from multiple viewpoints.

        Args:
            preprocessed: Validated inputs with points and optional colors

        Returns:
            Dictionary with thumbnails for each viewpoint
        """
        points = preprocessed["points"]
        colors = preprocessed.get("colors")

        # Render thumbnails for all viewpoints
        if len(self.viewpoints) == 1:
            # Single viewpoint - use simpler function
            img = render_thumbnail(
                points,
                colors=colors,
                width=self.width,
                height=self.height,
                viewpoint=self.viewpoints[0],
                point_size=self.point_size,
                background_color=self.background_color,
            )
            thumbnails = {self.viewpoints[0]: img}
        else:
            # Multiple viewpoints
            thumbnails = render_multi_view_thumbnail(
                points,
                colors=colors,
                viewpoints=self.viewpoints,
                width=self.width,
                height=self.height,
                point_size=self.point_size,
                background_color=self.background_color,
            )

        return {
            "thumbnails": thumbnails,
            "num_viewpoints": len(thumbnails),
        }

    def post_process(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Save thumbnails to disk if output directory specified.

        Args:
            predictions: Dictionary with thumbnails

        Returns:
            Dictionary with thumbnails and saved file paths
        """
        thumbnails = predictions["thumbnails"]
        thumbnail_paths = {}

        # Save thumbnails if output directory specified
        if self.output_dir:
            output_dir = Path(self.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            for viewpoint, img in thumbnails.items():
                output_path = output_dir / f"thumbnail_{viewpoint}.png"
                saved_path = save_thumbnail(img, str(output_path), quality=95)
                thumbnail_paths[viewpoint] = saved_path

        return {
            "thumbnails": thumbnails,
            "thumbnail_paths": thumbnail_paths,
            "viewpoints": list(thumbnails.keys()),
            "config": {
                "width": self.width,
                "height": self.height,
                "point_size": self.point_size,
                "background_color": self.background_color,
            },
        }
