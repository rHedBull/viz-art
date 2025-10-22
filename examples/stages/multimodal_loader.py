"""Multi-modal loader stage for synchronized image and point cloud data."""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import cv2
from viz_art.pipeline.stage import PipelineStage
from viz_art.types.pointcloud import MultiModalInput

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    o3d = None


class MultiModalLoaderStage(PipelineStage):
    """Load and validate synchronized image and point cloud pairs.

    This stage loads multi-modal data (images + point clouds) and validates
    their synchronization based on timestamps. It can enforce that both
    modalities are present and within a specified time tolerance.

    Example:
        >>> loader = MultiModalLoaderStage(
        ...     name="multimodal_loader",
        ...     sync_tolerance_ms=100.0,
        ...     require_both=True,
        ... )
        >>> result = loader.run({
        ...     "image_path": "data/image_001.jpg",
        ...     "pointcloud_path": "data/cloud_001.pcd",
        ...     "image_timestamp": "2025-10-22T10:30:00.000",
        ...     "pointcloud_timestamp": "2025-10-22T10:30:00.050",
        ... })
        >>> if result["is_synchronized"]:
        ...     image = result["image"]
        ...     points = result["points"]

    Attributes:
        name: Stage identifier
        sync_tolerance_ms: Maximum time difference for synchronization (ms)
        require_both: Whether both image and point cloud must be present
        load_colors: Whether to load point cloud colors
        load_normals: Whether to load point cloud normals
    """

    def __init__(
        self,
        name: str = "multimodal_loader",
        sync_tolerance_ms: float = 100.0,
        require_both: bool = False,
        load_colors: bool = True,
        load_normals: bool = False,
    ):
        """Initialize multi-modal loader stage.

        Args:
            name: Stage identifier
            sync_tolerance_ms: Maximum time delta for synchronization (ms)
            require_both: If True, both modalities must be present
            load_colors: Whether to load point cloud colors
            load_normals: Whether to load point cloud normals

        Raises:
            ValueError: If sync_tolerance_ms is negative
        """
        self._name = name
        self.sync_tolerance_ms = sync_tolerance_ms
        self.require_both = require_both
        self.load_colors = load_colors
        self.load_normals = load_normals

        if self.sync_tolerance_ms < 0:
            raise ValueError(f"sync_tolerance_ms must be >= 0, got {self.sync_tolerance_ms}")

    @property
    def name(self) -> str:
        """Get stage name."""
        return self._name

    @property
    def input_keys(self) -> List[str]:
        """Define input keys (flexible - accepts various combinations)."""
        return []  # Flexible inputs

    @property
    def output_keys(self) -> List[str]:
        """Define output keys."""
        return [
            "image",
            "points",
            "colors",
            "normals",
            "is_synchronized",
            "time_delta_ms",
            "metadata",
        ]

    @property
    def input_data_types(self) -> Optional[Dict[str, type]]:
        """Define expected input types."""
        return {
            "image_path": str,
            "pointcloud_path": str,
        }

    @property
    def output_data_types(self) -> Optional[Dict[str, type]]:
        """Define output types."""
        return {
            "image": (np.ndarray, type(None)),
            "points": (np.ndarray, type(None)),
            "is_synchronized": bool,
            "time_delta_ms": float,
        }

    def pre_process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate inputs and prepare for loading.

        Args:
            inputs: Dictionary with optional image_path, pointcloud_path, timestamps

        Returns:
            Validated inputs

        Raises:
            ValueError: If require_both=True and modalities missing
            FileNotFoundError: If file paths don't exist
        """
        image_path = inputs.get("image_path")
        pointcloud_path = inputs.get("pointcloud_path")

        # Check if at least one modality present
        if image_path is None and pointcloud_path is None:
            raise ValueError(
                "At least one of 'image_path' or 'pointcloud_path' must be provided"
            )

        # Check require_both constraint
        if self.require_both:
            if image_path is None or pointcloud_path is None:
                raise ValueError(
                    f"require_both=True but missing modality: "
                    f"image_path={'present' if image_path else 'missing'}, "
                    f"pointcloud_path={'present' if pointcloud_path else 'missing'}"
                )

        # Validate file existence
        if image_path is not None:
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")

        if pointcloud_path is not None:
            pointcloud_path = Path(pointcloud_path)
            if not pointcloud_path.exists():
                raise FileNotFoundError(f"Point cloud file not found: {pointcloud_path}")

        # Extract timestamps
        image_timestamp = inputs.get("image_timestamp", "")
        pointcloud_timestamp = inputs.get("pointcloud_timestamp", "")

        return {
            "image_path": image_path,
            "pointcloud_path": pointcloud_path,
            "image_timestamp": image_timestamp,
            "pointcloud_timestamp": pointcloud_timestamp,
        }

    def predict(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        """Load image and point cloud data.

        Args:
            preprocessed: Validated inputs from pre_process()

        Returns:
            Loaded data and metadata

        Raises:
            RuntimeError: If loading fails
        """
        image_path = preprocessed["image_path"]
        pointcloud_path = preprocessed["pointcloud_path"]
        image_timestamp = preprocessed["image_timestamp"]
        pointcloud_timestamp = preprocessed["pointcloud_timestamp"]

        # Load image
        image = None
        if image_path is not None:
            try:
                image = cv2.imread(str(image_path))
                if image is None:
                    raise RuntimeError(f"Failed to load image: {image_path}")
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except Exception as e:
                raise RuntimeError(f"Error loading image {image_path}: {e}") from e

        # Load point cloud
        points = None
        colors = None
        normals = None

        if pointcloud_path is not None:
            if not OPEN3D_AVAILABLE:
                raise RuntimeError(
                    "Open3D not available. Install with: pip install open3d>=0.18"
                )

            try:
                pcd = o3d.io.read_point_cloud(str(pointcloud_path))

                if len(pcd.points) == 0:
                    raise RuntimeError(f"Point cloud is empty: {pointcloud_path}")

                points = np.asarray(pcd.points, dtype=np.float32)

                if self.load_colors and pcd.has_colors():
                    colors = np.asarray(pcd.colors, dtype=np.float32)

                if self.load_normals and pcd.has_normals():
                    normals = np.asarray(pcd.normals, dtype=np.float32)

            except Exception as e:
                raise RuntimeError(f"Error loading point cloud {pointcloud_path}: {e}") from e

        # Validate timestamp synchronization
        is_synchronized = True
        time_delta_ms = 0.0

        if image_timestamp and pointcloud_timestamp:
            try:
                time_delta_ms = self._compute_time_delta(
                    image_timestamp, pointcloud_timestamp
                )

                if abs(time_delta_ms) > self.sync_tolerance_ms:
                    is_synchronized = False

            except ValueError as e:
                # If timestamp parsing fails, can't validate sync
                is_synchronized = False
                print(f"Warning: Failed to parse timestamps: {e}")

        return {
            "image": image,
            "points": points,
            "colors": colors,
            "normals": normals,
            "is_synchronized": is_synchronized,
            "time_delta_ms": time_delta_ms,
            "image_path": image_path,
            "pointcloud_path": pointcloud_path,
        }

    def post_process(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Format outputs and add metadata.

        Args:
            predictions: Loaded data from predict()

        Returns:
            Final output dictionary with metadata
        """
        metadata = {
            "num_points": len(predictions["points"]) if predictions["points"] is not None else 0,
            "has_image": predictions["image"] is not None,
            "has_pointcloud": predictions["points"] is not None,
            "has_colors": predictions["colors"] is not None,
            "has_normals": predictions["normals"] is not None,
            "is_synchronized": predictions["is_synchronized"],
            "time_delta_ms": predictions["time_delta_ms"],
        }

        if predictions["image"] is not None:
            metadata["image_shape"] = predictions["image"].shape
            metadata["image_path"] = str(predictions["image_path"])

        if predictions["points"] is not None:
            metadata["pointcloud_path"] = str(predictions["pointcloud_path"])

        return {
            "image": predictions["image"],
            "points": predictions["points"],
            "colors": predictions["colors"],
            "normals": predictions["normals"],
            "is_synchronized": predictions["is_synchronized"],
            "time_delta_ms": predictions["time_delta_ms"],
            "metadata": metadata,
        }

    def _compute_time_delta(self, timestamp1: str, timestamp2: str) -> float:
        """Compute time difference between two ISO 8601 timestamps.

        Args:
            timestamp1: First timestamp string
            timestamp2: Second timestamp string

        Returns:
            Time difference in milliseconds (positive if timestamp1 > timestamp2)

        Raises:
            ValueError: If timestamps cannot be parsed
        """
        try:
            # Parse ISO 8601 timestamps
            dt1 = datetime.fromisoformat(timestamp1.replace('Z', '+00:00'))
            dt2 = datetime.fromisoformat(timestamp2.replace('Z', '+00:00'))

            # Compute delta in milliseconds
            delta = (dt1 - dt2).total_seconds() * 1000.0

            return delta

        except Exception as e:
            raise ValueError(f"Failed to parse timestamps: {e}") from e
