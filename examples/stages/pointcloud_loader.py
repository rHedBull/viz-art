"""Point cloud loader stage for loading and preprocessing point clouds.

This stage loads point clouds from .pcd, .ply, or .xyz files and performs
optional preprocessing (downsampling, outlier removal, validation).
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
from viz_art.pipeline.stage import PipelineStage
from viz_art.types.pointcloud import PointCloudValidationRules, PointCloud

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    o3d = None
    print("Warning: Open3D not installed. PointCloudLoader will not work.")
    print("Install with: pip install open3d>=0.18")


class PointCloudLoader(PipelineStage):
    """Load and preprocess point cloud files.

    Supports .pcd, .ply, and .xyz formats with optional downsampling,
    outlier removal, and validation.

    Example:
        >>> loader = PointCloudLoader(
        ...     name="pc_loader",
        ...     downsample_voxel_size=0.05,
        ...     remove_outliers=True
        ... )
        >>> result = loader.run({"pointcloud_path": "scan.pcd"})
        >>> print(f"Loaded {result['metadata']['num_points']} points")
    """

    def __init__(
        self,
        name: str = "pointcloud_loader",
        downsample_voxel_size: Optional[float] = None,
        remove_outliers: bool = True,
        outlier_neighbors: int = 20,
        outlier_std_ratio: float = 2.0,
        validation_rules: Optional[PointCloudValidationRules] = None,
    ):
        """Initialize PointCloudLoader.

        Args:
            name: Stage name
            downsample_voxel_size: Voxel size for downsampling (meters), None to disable
            remove_outliers: Whether to remove statistical outliers
            outlier_neighbors: Number of neighbors for outlier detection
            outlier_std_ratio: Standard deviation ratio for outlier detection
            validation_rules: Optional validation rules to apply

        Raises:
            ImportError: If Open3D not installed
        """
        if not OPEN3D_AVAILABLE:
            raise ImportError(
                "Open3D is required for PointCloudLoader. "
                "Install with: pip install open3d>=0.18"
            )

        self._name = name
        self.downsample_voxel_size = downsample_voxel_size
        self.remove_outliers_flag = remove_outliers
        self.outlier_neighbors = outlier_neighbors
        self.outlier_std_ratio = outlier_std_ratio
        self.validation_rules = validation_rules

    @property
    def name(self) -> str:
        """Return stage name."""
        return self._name

    @property
    def input_keys(self) -> List[str]:
        """Return required input keys."""
        return ["pointcloud_path"]

    @property
    def output_keys(self) -> List[str]:
        """Return output keys."""
        return ["pointcloud", "points", "colors", "metadata"]

    @property
    def input_data_types(self) -> Optional[Dict[str, type]]:
        """Return input data types for validation."""
        return {"pointcloud_path": str}

    @property
    def output_data_types(self) -> Optional[Dict[str, type]]:
        """Return output data types for validation."""
        return {
            "pointcloud": object,  # o3d.geometry.PointCloud
            "points": np.ndarray,
            "colors": (np.ndarray, type(None)),
            "metadata": dict,
        }

    def pre_process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate inputs before loading.

        Args:
            inputs: Dictionary with "pointcloud_path" key

        Returns:
            Validated inputs

        Raises:
            ValueError: If pointcloud_path missing or invalid
            FileNotFoundError: If file doesn't exist
        """
        if "pointcloud_path" not in inputs:
            raise ValueError("Missing required input: 'pointcloud_path'")

        path = Path(inputs["pointcloud_path"])

        # Validate file exists
        if not path.exists():
            raise FileNotFoundError(f"Point cloud file not found: {path}")

        # Validate file format
        ext = path.suffix.lower()
        supported_formats = {'.pcd', '.ply', '.xyz', '.pts', '.xyzrgb'}

        if ext not in supported_formats:
            raise ValueError(
                f"Unsupported point cloud format: {ext}. "
                f"Supported formats: {', '.join(supported_formats)}"
            )

        # Check file is not empty
        if path.stat().st_size == 0:
            raise ValueError(f"Point cloud file is empty: {path}")

        return inputs

    def predict(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        """Load and preprocess point cloud.

        Args:
            preprocessed: Output from pre_process()

        Returns:
            Dictionary with loaded point cloud data

        Raises:
            RuntimeError: If loading fails or point cloud is empty
        """
        path = preprocessed["pointcloud_path"]
        file_format = Path(path).suffix.lower()

        # Load point cloud
        try:
            pcd = o3d.io.read_point_cloud(path)
        except Exception as e:
            raise RuntimeError(f"Failed to load point cloud from {path}: {e}")

        # Validate not empty
        if not pcd.has_points() or len(pcd.points) == 0:
            raise RuntimeError(f"Loaded point cloud is empty: {path}")

        original_count = len(pcd.points)

        # Apply downsampling if requested
        if self.downsample_voxel_size is not None and self.downsample_voxel_size > 0:
            pcd = pcd.voxel_down_sample(self.downsample_voxel_size)

        # Remove outliers if requested
        if self.remove_outliers_flag:
            pcd, inlier_indices = pcd.remove_statistical_outlier(
                nb_neighbors=self.outlier_neighbors,
                std_ratio=self.outlier_std_ratio
            )
            outliers_removed = original_count - len(pcd.points)
        else:
            outliers_removed = 0

        # Extract data
        points = np.asarray(pcd.points, dtype=np.float32)
        colors = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else None
        normals = np.asarray(pcd.normals, dtype=np.float32) if pcd.has_normals() else None

        # Validate if rules provided
        if self.validation_rules is not None:
            # Create PointCloud for validation
            pc_obj = PointCloud(
                points=points,
                colors=colors,
                normals=normals,
                num_points=len(points),
                coordinate_system="lidar",  # Default assumption
            )

            is_valid, errors = self.validation_rules.validate_pointcloud(pc_obj)

            if not is_valid:
                error_msg = f"Point cloud validation failed: {'; '.join(errors)}"
                raise RuntimeError(error_msg)

        return {
            "pcd_object": pcd,
            "points_array": points,
            "colors_array": colors,
            "normals_array": normals,
            "num_points": len(points),
            "original_count": original_count,
            "outliers_removed": outliers_removed,
            "file_format": file_format,
        }

    def post_process(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Format outputs.

        Args:
            predictions: Output from predict()

        Returns:
            Final output dictionary matching output_keys
        """
        return {
            "pointcloud": predictions["pcd_object"],
            "points": predictions["points_array"],
            "colors": predictions["colors_array"],
            "metadata": {
                "num_points": predictions["num_points"],
                "original_count": predictions["original_count"],
                "outliers_removed": predictions["outliers_removed"],
                "has_colors": predictions["colors_array"] is not None,
                "has_normals": predictions["normals_array"] is not None,
                "file_format": predictions["file_format"],
                "downsampled": self.downsample_voxel_size is not None,
                "voxel_size": self.downsample_voxel_size,
            }
        }

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the full pipeline: pre_process → predict → post_process.

        Args:
            inputs: Dictionary with "pointcloud_path" key

        Returns:
            Dictionary with point cloud data and metadata

        Raises:
            ValueError: If inputs invalid
            FileNotFoundError: If file not found
            RuntimeError: If loading/processing fails
        """
        preprocessed = self.pre_process(inputs)
        predictions = self.predict(preprocessed)
        outputs = self.post_process(predictions)
        return outputs
