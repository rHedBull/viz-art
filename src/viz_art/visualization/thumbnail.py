"""Open3D-based point cloud thumbnail generation with headless rendering."""

from typing import Tuple, Optional, Dict
from pathlib import Path
import hashlib
import time
import numpy as np


def get_viewpoint_params(
    viewpoint: str = "diagonal",
    distance_scale: float = 2.5,
) -> Dict[str, np.ndarray]:
    """Define camera positions for standard viewpoints.

    Args:
        viewpoint: Viewpoint name ("front", "top", "side", "diagonal")
        distance_scale: Scale factor for camera distance from origin

    Returns:
        Dictionary with 'eye', 'lookat', 'up' vectors

    Raises:
        ValueError: If viewpoint is invalid
    """
    viewpoints = {
        "front": {
            "eye": np.array([0, 0, distance_scale]),
            "lookat": np.array([0, 0, 0]),
            "up": np.array([0, 1, 0]),
        },
        "top": {
            "eye": np.array([0, distance_scale, 0]),
            "lookat": np.array([0, 0, 0]),
            "up": np.array([0, 0, -1]),
        },
        "side": {
            "eye": np.array([distance_scale, 0, 0]),
            "lookat": np.array([0, 0, 0]),
            "up": np.array([0, 1, 0]),
        },
        "diagonal": {
            "eye": np.array([distance_scale, distance_scale, distance_scale]),
            "lookat": np.array([0, 0, 0]),
            "up": np.array([0, 1, 0]),
        },
    }

    if viewpoint not in viewpoints:
        raise ValueError(
            f"Invalid viewpoint '{viewpoint}'. "
            f"Must be one of: {list(viewpoints.keys())}"
        )

    return viewpoints[viewpoint]


def render_thumbnail(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    width: int = 800,
    height: int = 600,
    viewpoint: str = "diagonal",
    point_size: float = 2.0,
    background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Render point cloud thumbnail using Open3D OffscreenRenderer.

    Args:
        points: Nx3 array of XYZ coordinates
        colors: Optional Nx3 array of RGB values in [0,1]
        width: Thumbnail width (pixels)
        height: Thumbnail height (pixels)
        viewpoint: Camera viewpoint ("front", "top", "side", "diagonal")
        point_size: Point size in pixels
        background_color: RGB background color in [0,1]

    Returns:
        HxWx3 NumPy array (uint8) of rendered image

    Raises:
        ValueError: If inputs are invalid
        ImportError: If Open3D not available
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError(
            "Open3D is required for thumbnail rendering. "
            "Install it with: pip install open3d>=0.18"
        )

    if points.shape[0] == 0:
        raise ValueError("Cannot render thumbnail from empty point cloud")

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Points must be Nx3 array, got shape {points.shape}")

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Add colors if provided, otherwise use default gray
    if colors is not None:
        if colors.shape[0] != points.shape[0]:
            raise ValueError(
                f"Colors length {colors.shape[0]} != points length {points.shape[0]}"
            )
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        # Default gray color
        default_color = np.full((points.shape[0], 3), 0.5, dtype=np.float64)
        pcd.colors = o3d.utility.Vector3dVector(default_color)

    # Center the point cloud
    pcd_center = pcd.get_center()
    pcd.translate(-pcd_center)

    # Calculate bounding box for camera positioning
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox_extent = bbox.get_extent()
    max_extent = np.max(bbox_extent)

    # Get viewpoint parameters
    camera_params = get_viewpoint_params(
        viewpoint,
        distance_scale=max_extent * 1.5  # Scale based on point cloud size
    )

    # Create offscreen renderer
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)

    # Set rendering options
    renderer.scene.set_background(
        np.array([*background_color, 1.0], dtype=np.float32)
    )

    # Add point cloud to scene
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.point_size = point_size

    renderer.scene.add_geometry("pointcloud", pcd, material)

    # Set up camera
    renderer.setup_camera(
        60.0,  # Field of view
        pcd_center,  # Center of rotation (already at origin after translation)
        camera_params["eye"],  # Camera position
        camera_params["up"],  # Up vector
    )

    # Render to image
    img = renderer.render_to_image()

    # Convert to NumPy array (uint8)
    img_np = np.asarray(img)

    return img_np


def save_thumbnail(
    img: np.ndarray,
    output_path: str,
    quality: int = 95,
) -> str:
    """Save thumbnail image to file.

    Args:
        img: HxWx3 image array (uint8)
        output_path: Output file path (.png, .jpg)
        quality: JPEG quality (1-100, ignored for PNG)

    Returns:
        Absolute path to saved image file

    Raises:
        ValueError: If image is invalid
        IOError: If file cannot be written
    """
    from PIL import Image

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Image must be HxWx3, got shape {img.shape}")

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert NumPy array to PIL Image
    pil_img = Image.fromarray(img, mode='RGB')

    # Save with appropriate format
    if output_path.suffix.lower() in ['.jpg', '.jpeg']:
        pil_img.save(output_path, 'JPEG', quality=quality)
    elif output_path.suffix.lower() == '.png':
        pil_img.save(output_path, 'PNG')
    else:
        # Default to PNG
        pil_img.save(output_path, 'PNG')

    return str(output_path)


def render_multi_view_thumbnail(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    viewpoints: Tuple[str, ...] = ("front", "top", "diagonal"),
    width: int = 800,
    height: int = 600,
    point_size: float = 2.0,
    background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[str, np.ndarray]:
    """Render thumbnails from multiple viewpoints.

    Args:
        points: Nx3 array of XYZ coordinates
        colors: Optional Nx3 array of RGB values in [0,1]
        viewpoints: Tuple of viewpoint names to render
        width: Thumbnail width (pixels)
        height: Thumbnail height (pixels)
        point_size: Point size in pixels
        background_color: RGB background color in [0,1]

    Returns:
        Dictionary mapping viewpoint names to rendered images

    Raises:
        ValueError: If inputs are invalid
    """
    thumbnails = {}

    for viewpoint in viewpoints:
        img = render_thumbnail(
            points,
            colors=colors,
            width=width,
            height=height,
            viewpoint=viewpoint,
            point_size=point_size,
            background_color=background_color,
        )
        thumbnails[viewpoint] = img

    return thumbnails


def compute_pointcloud_hash(points: np.ndarray) -> str:
    """Compute hash of point cloud for caching.

    Args:
        points: Nx3 array of XYZ coordinates

    Returns:
        SHA256 hash string
    """
    # Use a subset of points for speed if cloud is large
    if points.shape[0] > 10000:
        # Sample 10000 points deterministically
        step = points.shape[0] // 10000
        sample = points[::step]
    else:
        sample = points

    # Compute hash
    point_bytes = sample.tobytes()
    hash_obj = hashlib.sha256(point_bytes)
    return hash_obj.hexdigest()


def get_cached_thumbnail(
    cache_dir: Path,
    points_hash: str,
    viewpoint: str,
    width: int,
    height: int,
) -> Optional[np.ndarray]:
    """Load cached thumbnail if available.

    Args:
        cache_dir: Directory containing cached thumbnails
        points_hash: Hash of point cloud data
        viewpoint: Viewpoint name
        width: Thumbnail width
        height: Thumbnail height

    Returns:
        Cached thumbnail image or None if not found
    """
    cache_path = (
        cache_dir / f"thumb_{points_hash}_{viewpoint}_{width}x{height}.png"
    )

    if cache_path.exists():
        try:
            from PIL import Image
            img = Image.open(cache_path)
            return np.array(img)
        except Exception:
            # Cache read failed, return None
            return None

    return None


def save_cached_thumbnail(
    cache_dir: Path,
    points_hash: str,
    viewpoint: str,
    width: int,
    height: int,
    img: np.ndarray,
) -> None:
    """Save thumbnail to cache.

    Args:
        cache_dir: Directory for cached thumbnails
        points_hash: Hash of point cloud data
        viewpoint: Viewpoint name
        width: Thumbnail width
        height: Thumbnail height
        img: Thumbnail image to cache
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = (
        cache_dir / f"thumb_{points_hash}_{viewpoint}_{width}x{height}.png"
    )

    try:
        save_thumbnail(img, str(cache_path))
    except Exception:
        # Cache write failed, ignore
        pass


def render_thumbnail_with_cache(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    width: int = 800,
    height: int = 600,
    viewpoint: str = "diagonal",
    point_size: float = 2.0,
    background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    cache_dir: Optional[Path] = None,
    max_points_for_rendering: int = 100000,
) -> np.ndarray:
    """Render thumbnail with caching and automatic downsampling.

    This function combines caching and performance optimizations:
    - Checks cache before rendering
    - Auto-downsamples large point clouds
    - Saves to cache after rendering

    Args:
        points: Nx3 array of XYZ coordinates
        colors: Optional Nx3 array of RGB values in [0,1]
        width: Thumbnail width (pixels)
        height: Thumbnail height (pixels)
        viewpoint: Camera viewpoint
        point_size: Point size in pixels
        background_color: RGB background color in [0,1]
        cache_dir: Optional cache directory (None = no caching)
        max_points_for_rendering: Maximum points before downsampling

    Returns:
        HxWx3 NumPy array (uint8) of rendered image
    """
    # Compute hash for caching
    points_hash = None
    if cache_dir is not None:
        points_hash = compute_pointcloud_hash(points)

        # Try to load from cache
        cached_img = get_cached_thumbnail(
            cache_dir, points_hash, viewpoint, width, height
        )
        if cached_img is not None:
            return cached_img

    # Auto-downsample for performance (T080, T081)
    render_points = points
    render_colors = colors

    if points.shape[0] > max_points_for_rendering:
        # Random downsampling
        indices = np.random.choice(
            points.shape[0],
            size=max_points_for_rendering,
            replace=False,
        )
        indices = np.sort(indices)
        render_points = points[indices]

        if colors is not None:
            render_colors = colors[indices]

    # Render thumbnail
    start_time = time.time()

    img = render_thumbnail(
        render_points,
        colors=render_colors,
        width=width,
        height=height,
        viewpoint=viewpoint,
        point_size=point_size,
        background_color=background_color,
    )

    elapsed = time.time() - start_time

    # Warn if rendering took too long (SC-006: < 3s target)
    if elapsed > 3.0:
        print(
            f"Warning: Thumbnail generation took {elapsed:.2f}s "
            f"(target < 3s). Consider reducing max_points_for_rendering."
        )

    # Save to cache
    if cache_dir is not None and points_hash is not None:
        save_cached_thumbnail(
            cache_dir, points_hash, viewpoint, width, height, img
        )

    return img
