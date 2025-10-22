"""Utility functions for point cloud processing and conversion."""

from typing import Optional, Tuple
import numpy as np

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    o3d = None  # type: ignore


def numpy_to_pointcloud(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None
) -> "o3d.geometry.PointCloud":
    """Convert NumPy arrays to Open3D PointCloud.

    Args:
        points: Nx3 array of XYZ coordinates (float32 or float64)
        colors: Optional Nx3 array of RGB colors in [0,1] (float32 or float64)
        normals: Optional Nx3 array of surface normals (float32 or float64)

    Returns:
        Open3D PointCloud object

    Raises:
        ImportError: If Open3D not installed
        ValueError: If array shapes invalid

    Example:
        >>> points = np.random.rand(100, 3).astype(np.float32)
        >>> colors = np.random.rand(100, 3).astype(np.float32)
        >>> pcd = numpy_to_pointcloud(points, colors)
        >>> print(len(pcd.points))
        100
    """
    if not OPEN3D_AVAILABLE or o3d is None:
        raise ImportError(
            "Open3D not installed. Install with: pip install open3d>=0.18"
        )

    # Validate input shapes
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"Points must be Nx3 array, got shape {points.shape}. "
            f"Expected format: (num_points, 3) with XYZ coordinates."
        )

    if colors is not None:
        if colors.ndim != 2 or colors.shape[1] != 3:
            raise ValueError(
                f"Colors must be Nx3 array, got shape {colors.shape}. "
                f"Expected format: (num_points, 3) with RGB values."
            )
        if len(colors) != len(points):
            raise ValueError(
                f"Colors length {len(colors)} != points length {len(points)}"
            )

    if normals is not None:
        if normals.ndim != 2 or normals.shape[1] != 3:
            raise ValueError(
                f"Normals must be Nx3 array, got shape {normals.shape}. "
                f"Expected format: (num_points, 3) with normal vectors."
            )
        if len(normals) != len(points):
            raise ValueError(
                f"Normals length {len(normals)} != points length {len(points)}"
            )

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    if colors is not None:
        # Ensure colors are in [0, 1] range
        colors_float = colors.astype(np.float64)
        if colors_float.max() > 1.0 or colors_float.min() < 0.0:
            raise ValueError(
                f"Colors must be in [0, 1] range, got range [{colors_float.min():.3f}, {colors_float.max():.3f}]. "
                f"If using [0, 255] range, divide by 255 first."
            )
        pcd.colors = o3d.utility.Vector3dVector(colors_float)

    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))

    return pcd


def pointcloud_to_numpy(
    pcd: "o3d.geometry.PointCloud"
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Convert Open3D PointCloud to NumPy arrays.

    Args:
        pcd: Open3D PointCloud object

    Returns:
        Tuple of (points, colors, normals)
        - points: Nx3 float32 array of XYZ coordinates
        - colors: Nx3 float32 array of RGB values in [0,1] or None if not present
        - normals: Nx3 float32 array of surface normals or None if not present

    Raises:
        ImportError: If Open3D not installed
        RuntimeError: If point cloud is empty

    Example:
        >>> pcd = o3d.io.read_point_cloud("scan.pcd")
        >>> points, colors, normals = pointcloud_to_numpy(pcd)
        >>> print(f"Loaded {len(points)} points")
        Loaded 12450 points
    """
    if not OPEN3D_AVAILABLE or o3d is None:
        raise ImportError(
            "Open3D not installed. Install with: pip install open3d>=0.18"
        )

    # Validate point cloud is not empty
    if not pcd.has_points() or len(pcd.points) == 0:
        raise RuntimeError("Point cloud is empty - cannot convert to NumPy")

    # Extract points (always present)
    points = np.asarray(pcd.points, dtype=np.float32)

    # Extract colors if present
    colors = None
    if pcd.has_colors():
        colors = np.asarray(pcd.colors, dtype=np.float32)

    # Extract normals if present
    normals = None
    if pcd.has_normals():
        normals = np.asarray(pcd.normals, dtype=np.float32)

    return points, colors, normals


def validate_pointcloud_file(filepath: str) -> Tuple[bool, str]:
    """Validate if file is a supported point cloud format.

    Args:
        filepath: Path to point cloud file

    Returns:
        Tuple of (is_valid, message)

    Example:
        >>> is_valid, msg = validate_pointcloud_file("scan.pcd")
        >>> if is_valid:
        ...     print(f"Valid: {msg}")
        Valid: Supported format: .pcd
    """
    import os
    from pathlib import Path

    path = Path(filepath)

    # Check file exists
    if not path.exists():
        return False, f"File not found: {filepath}"

    # Check extension
    ext = path.suffix.lower()
    supported_formats = {'.pcd', '.ply', '.xyz', '.pts', '.xyzrgb'}

    if ext not in supported_formats:
        return False, f"Unsupported format: {ext}. Supported: {', '.join(supported_formats)}"

    # Check file is not empty
    if os.path.getsize(filepath) == 0:
        return False, "File is empty"

    return True, f"Supported format: {ext}"


def downsample_pointcloud(
    pcd: "o3d.geometry.PointCloud",
    voxel_size: float
) -> "o3d.geometry.PointCloud":
    """Downsample point cloud using voxel grid.

    Args:
        pcd: Input point cloud
        voxel_size: Size of voxel grid (in meters)

    Returns:
        Downsampled point cloud

    Raises:
        ImportError: If Open3D not installed
        ValueError: If voxel_size <= 0

    Example:
        >>> pcd_original = o3d.io.read_point_cloud("scan.pcd")
        >>> pcd_downsampled = downsample_pointcloud(pcd_original, voxel_size=0.05)
        >>> print(f"Reduced from {len(pcd_original.points)} to {len(pcd_downsampled.points)} points")
        Reduced from 100000 to 12450 points
    """
    if not OPEN3D_AVAILABLE or o3d is None:
        raise ImportError(
            "Open3D not installed. Install with: pip install open3d>=0.18"
        )

    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be > 0, got {voxel_size}")

    return pcd.voxel_down_sample(voxel_size)


def remove_outliers(
    pcd: "o3d.geometry.PointCloud",
    nb_neighbors: int = 20,
    std_ratio: float = 2.0
) -> Tuple["o3d.geometry.PointCloud", np.ndarray]:
    """Remove statistical outliers from point cloud.

    Args:
        pcd: Input point cloud
        nb_neighbors: Number of neighbors to analyze for each point
        std_ratio: Standard deviation ratio threshold

    Returns:
        Tuple of (cleaned_pcd, inlier_indices)

    Raises:
        ImportError: If Open3D not installed

    Example:
        >>> pcd = o3d.io.read_point_cloud("noisy_scan.pcd")
        >>> clean_pcd, inliers = remove_outliers(pcd, nb_neighbors=20, std_ratio=2.0)
        >>> print(f"Removed {len(pcd.points) - len(clean_pcd.points)} outliers")
        Removed 342 outliers
    """
    if not OPEN3D_AVAILABLE or o3d is None:
        raise ImportError(
            "Open3D not installed. Install with: pip install open3d>=0.18"
        )

    clean_pcd, inlier_indices = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )

    return clean_pcd, np.asarray(inlier_indices)


def estimate_normals(
    pcd: "o3d.geometry.PointCloud",
    search_radius: float = 0.1,
    max_nn: int = 30
) -> "o3d.geometry.PointCloud":
    """Estimate surface normals for point cloud.

    Args:
        pcd: Input point cloud
        search_radius: Search radius for normal estimation
        max_nn: Maximum number of nearest neighbors

    Returns:
        Point cloud with estimated normals

    Raises:
        ImportError: If Open3D not installed

    Example:
        >>> pcd = o3d.io.read_point_cloud("scan.pcd")
        >>> pcd_with_normals = estimate_normals(pcd, search_radius=0.1)
        >>> print(f"Normals estimated: {pcd_with_normals.has_normals()}")
        Normals estimated: True
    """
    if not OPEN3D_AVAILABLE or o3d is None:
        raise ImportError(
            "Open3D not installed. Install with: pip install open3d>=0.18"
        )

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=search_radius,
            max_nn=max_nn
        )
    )

    return pcd
