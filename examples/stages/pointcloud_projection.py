"""Point cloud projection stage for overlaying 3D data onto 2D images."""

from typing import Dict, Any, List
from pathlib import Path
import numpy as np
import cv2

from viz_art.pipeline import PipelineStage
from viz_art.calibration import (
    load_calibration,
    validate_calibration,
    project_points,
    overlay_points_on_image,
    compute_depth_colormap,
)
from viz_art.types.pointcloud import Calibration


class PointCloudProjectionStage(PipelineStage):
    """Project 3D point cloud onto 2D image using camera calibration.

    This stage takes an image and point cloud as input, projects the 3D points
    onto the 2D image plane using camera calibration, and renders the projected
    points with color coding (depth or original colors).

    Example:
        >>> stage = PointCloudProjectionStage(
        ...     name="projection",
        ...     calibration_path="calibration/camera.yaml",
        ...     color_mode="depth",
        ...     opacity=0.7,
        ... )
        >>> result = stage.run({
        ...     "image": image_array,
        ...     "points": points_3d,
        ... })
        >>> projected_image = result["projected_image"]

    Attributes:
        name: Stage identifier
        calibration_path: Path to calibration YAML/JSON file
        color_mode: Color coding method ("depth", "rgb", "intensity")
        point_radius: Rendered point radius in pixels
        opacity: Blending opacity [0,1]
        z_threshold: Minimum Z distance for valid points (meters)
    """

    def __init__(
        self,
        name: str = "pointcloud_projection",
        calibration_path: str | Path | None = None,
        color_mode: str = "depth",
        point_radius: int = 2,
        opacity: float = 0.7,
        z_threshold: float = 0.1,
    ):
        """Initialize point cloud projection stage.

        Args:
            name: Stage identifier
            calibration_path: Path to calibration file
            color_mode: Color coding method ("depth", "rgb", "intensity")
            point_radius: Point radius in pixels (default 2)
            opacity: Blending opacity 0-1 (default 0.7)
            z_threshold: Minimum Z distance in meters (default 0.1)

        Raises:
            ValueError: If calibration_path is None or parameters invalid
        """
        self._name = name
        self.calibration_path = Path(calibration_path) if calibration_path else None
        self.color_mode = color_mode
        self.point_radius = point_radius
        self.opacity = opacity
        self.z_threshold = z_threshold
        self._calibration: Calibration | None = None

        # Validate parameters
        if self.color_mode not in ["depth", "rgb", "intensity"]:
            raise ValueError(
                f"color_mode must be 'depth', 'rgb', or 'intensity', got '{self.color_mode}'"
            )

        if not (0 <= self.opacity <= 1):
            raise ValueError(f"opacity must be in [0,1], got {self.opacity}")

        if self.point_radius < 1:
            raise ValueError(f"point_radius must be >= 1, got {self.point_radius}")

    @property
    def name(self) -> str:
        """Get stage name."""
        return self._name

    @property
    def input_keys(self) -> List[str]:
        """Define required input keys."""
        return ["image", "points"]

    @property
    def output_keys(self) -> List[str]:
        """Define output keys."""
        return ["projected_image", "points_2d", "valid_mask", "num_visible"]

    @property
    def input_data_types(self) -> Dict[str, type]:
        """Define expected input types."""
        return {
            "image": np.ndarray,  # HxWx3 uint8
            "points": np.ndarray,  # Nx3 float32
        }

    @property
    def output_data_types(self) -> Dict[str, type]:
        """Define output types."""
        return {
            "projected_image": np.ndarray,
            "points_2d": np.ndarray,
            "valid_mask": np.ndarray,
            "num_visible": int,
        }

    def pre_process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load and validate calibration, validate inputs.

        Args:
            inputs: Dictionary with 'image' and 'points' keys

        Returns:
            Validated inputs dictionary

        Raises:
            ValueError: If calibration or inputs are invalid
            FileNotFoundError: If calibration file not found
        """
        # Load calibration if not already loaded
        if self._calibration is None:
            if self.calibration_path is None:
                raise ValueError("calibration_path must be specified")

            self._calibration = load_calibration(self.calibration_path)

            # Validate calibration
            is_valid, warnings = validate_calibration(self._calibration)
            if not is_valid:
                for warning in warnings:
                    print(f"WARNING: {warning}")

        # Validate image
        image = inputs.get("image")
        if image is None:
            raise ValueError("'image' key required in inputs")

        if not isinstance(image, np.ndarray):
            raise ValueError(f"image must be np.ndarray, got {type(image)}")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"image must be HxWx3, got shape {image.shape}")

        # Validate points
        points = inputs.get("points")
        if points is None:
            raise ValueError("'points' key required in inputs")

        if not isinstance(points, np.ndarray):
            raise ValueError(f"points must be np.ndarray, got {type(points)}")

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must be Nx3, got shape {points.shape}")

        # Get optional colors
        colors = inputs.get("colors")
        if colors is not None:
            if colors.shape[0] != points.shape[0]:
                raise ValueError(
                    f"colors length {colors.shape[0]} != points length {points.shape[0]}"
                )

        return inputs

    def predict(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        """Project 3D points to 2D and determine visibility.

        Args:
            preprocessed: Validated inputs with image and points

        Returns:
            Dictionary with projection results
        """
        image = preprocessed["image"]
        points_3d = preprocessed["points"]
        colors = preprocessed.get("colors")
        intensity = preprocessed.get("intensity")

        # Project points to 2D
        points_2d, valid_mask = project_points(points_3d, self._calibration)

        # Filter to visible points only
        visible_points_2d = points_2d[valid_mask]
        num_visible = np.sum(valid_mask)

        # Determine colors for rendering
        if self.color_mode == "depth":
            # Compute depth-based colormap
            visible_points_3d = points_3d[valid_mask]
            point_colors = compute_depth_colormap(visible_points_3d)

        elif self.color_mode == "rgb" and colors is not None:
            # Use original RGB colors
            point_colors = colors[valid_mask]

        elif self.color_mode == "intensity" and intensity is not None:
            # Convert intensity to grayscale colors
            visible_intensity = intensity[valid_mask]
            # Normalize to [0, 1]
            if visible_intensity.max() > visible_intensity.min():
                normalized = (visible_intensity - visible_intensity.min()) / (
                    visible_intensity.max() - visible_intensity.min()
                )
            else:
                normalized = np.ones_like(visible_intensity) * 0.5

            point_colors = np.stack([normalized] * 3, axis=1)

        else:
            # Default to depth if requested mode unavailable
            visible_points_3d = points_3d[valid_mask]
            point_colors = compute_depth_colormap(visible_points_3d)

        return {
            "image": image,
            "visible_points_2d": visible_points_2d,
            "point_colors": point_colors,
            "points_2d": points_2d,
            "valid_mask": valid_mask,
            "num_visible": num_visible,
        }

    def post_process(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Render projected points onto image with color/depth coding.

        Args:
            predictions: Projection results from predict()

        Returns:
            Dictionary with final overlaid image
        """
        image = predictions["image"]
        visible_points_2d = predictions["visible_points_2d"]
        point_colors = predictions["point_colors"]

        # Render points onto image
        projected_image = overlay_points_on_image(
            image,
            visible_points_2d,
            colors=point_colors,
            point_radius=self.point_radius,
            opacity=self.opacity,
        )

        return {
            "projected_image": projected_image,
            "points_2d": predictions["points_2d"],
            "valid_mask": predictions["valid_mask"],
            "num_visible": predictions["num_visible"],
        }
