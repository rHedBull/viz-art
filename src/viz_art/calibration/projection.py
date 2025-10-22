"""3D to 2D point projection using camera calibration."""

from typing import Tuple
import numpy as np
import cv2

from viz_art.types.pointcloud import Calibration


def project_points(
    points_3d: np.ndarray,
    calibration: Calibration,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3D points to 2D image coordinates using camera calibration.

    Uses OpenCV's projectPoints() with full distortion model.

    Args:
        points_3d: Nx3 array of XYZ coordinates in camera frame
        calibration: Camera calibration parameters

    Returns:
        Tuple of:
            - points_2d: Nx2 array of (u, v) pixel coordinates
            - valid_mask: Boolean array indicating which points are valid

    Raises:
        ValueError: If points_3d is invalid shape
    """
    if points_3d.ndim != 2 or points_3d.shape[1] != 3:
        raise ValueError(f"points_3d must be Nx3, got shape {points_3d.shape}")

    if points_3d.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=bool)

    # Get calibration parameters
    intrinsics = calibration.intrinsics
    extrinsics = calibration.extrinsics

    # Camera matrix (3x3)
    camera_matrix = intrinsics.to_matrix()

    # Distortion coefficients
    dist_coeffs = np.array(intrinsics.distortion_coeffs, dtype=np.float64)

    # Rotation vector (3x1) and translation vector (3x1)
    rvec = extrinsics.to_rodrigues_vector()
    tvec = extrinsics.translation_vector.reshape(3, 1)

    # Project points using OpenCV
    points_2d, _ = cv2.projectPoints(
        points_3d.astype(np.float64),
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )

    # Reshape from (N, 1, 2) to (N, 2)
    points_2d = points_2d.reshape(-1, 2).astype(np.float32)

    # Filter valid points (within image bounds and in front of camera)
    valid_mask = filter_visible_points(
        points_2d,
        points_3d,
        intrinsics.width,
        intrinsics.height,
    )

    return points_2d, valid_mask


def filter_visible_points(
    points_2d: np.ndarray,
    points_3d: np.ndarray,
    image_width: int,
    image_height: int,
    z_threshold: float = 0.1,
) -> np.ndarray:
    """Filter points outside image bounds or behind camera.

    Args:
        points_2d: Nx2 array of (u, v) pixel coordinates
        points_3d: Nx3 array of XYZ coordinates
        image_width: Image width in pixels
        image_height: Image height in pixels
        z_threshold: Minimum Z distance from camera (meters)

    Returns:
        Boolean mask array (N,) where True = visible point

    Raises:
        ValueError: If array shapes don't match
    """
    if points_2d.shape[0] != points_3d.shape[0]:
        raise ValueError(
            f"points_2d ({points_2d.shape[0]}) and points_3d ({points_3d.shape[0]}) "
            "must have same length"
        )

    # Check if points are in front of camera (positive Z)
    in_front = points_3d[:, 2] > z_threshold

    # Check if points are within image bounds
    u_valid = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < image_width)
    v_valid = (points_2d[:, 1] >= 0) & (points_2d[:, 1] < image_height)
    in_bounds = u_valid & v_valid

    # Combine all conditions
    valid_mask = in_front & in_bounds

    return valid_mask


def overlay_points_on_image(
    image: np.ndarray,
    points_2d: np.ndarray,
    colors: np.ndarray | None = None,
    point_radius: int = 2,
    opacity: float = 0.7,
) -> np.ndarray:
    """Render projected points onto image with optional colors.

    Args:
        image: HxWx3 image array (uint8)
        points_2d: Nx2 array of (u, v) pixel coordinates
        colors: Optional Nx3 array of RGB values in [0,1]
        point_radius: Radius of rendered points (pixels)
        opacity: Blending opacity [0,1]

    Returns:
        HxWx3 image array with overlaid points

    Raises:
        ValueError: If image or points_2d are invalid
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Image must be HxWx3, got shape {image.shape}")

    if points_2d.ndim != 2 or points_2d.shape[1] != 2:
        raise ValueError(f"points_2d must be Nx2, got shape {points_2d.shape}")

    if colors is not None and colors.shape[0] != points_2d.shape[0]:
        raise ValueError(
            f"colors ({colors.shape[0]}) must match points_2d ({points_2d.shape[0]})"
        )

    # Create output image (copy to avoid modifying input)
    output = image.copy()

    # Create overlay for transparency
    overlay = image.copy()

    # Draw each point
    for i, (u, v) in enumerate(points_2d):
        # Get point color
        if colors is not None:
            # Convert from [0,1] to [0,255]
            color = tuple(int(c * 255) for c in colors[i])
        else:
            # Default red color
            color = (255, 0, 0)

        # Draw filled circle
        center = (int(u), int(v))
        cv2.circle(overlay, center, point_radius, color, -1)

    # Blend overlay with original image
    cv2.addWeighted(overlay, opacity, output, 1 - opacity, 0, output)

    return output


def compute_depth_colormap(
    points_3d: np.ndarray,
    colormap: int = cv2.COLORMAP_VIRIDIS,
) -> np.ndarray:
    """Compute color-coded depth values for visualization.

    Args:
        points_3d: Nx3 array of XYZ coordinates
        colormap: OpenCV colormap constant

    Returns:
        Nx3 array of RGB colors in [0,1]

    Raises:
        ValueError: If points_3d is invalid
    """
    if points_3d.ndim != 2 or points_3d.shape[1] != 3:
        raise ValueError(f"points_3d must be Nx3, got shape {points_3d.shape}")

    if points_3d.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    # Extract Z coordinates (depth)
    depths = points_3d[:, 2]

    # Normalize to [0, 255]
    min_depth = depths.min()
    max_depth = depths.max()

    if max_depth - min_depth < 1e-6:
        # All points at same depth - use middle color
        normalized = np.full_like(depths, 128, dtype=np.uint8)
    else:
        normalized = ((depths - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)

    # Apply colormap
    colors_bgr = cv2.applyColorMap(normalized.reshape(-1, 1), colormap)

    # Convert BGR to RGB and normalize to [0,1]
    colors_rgb = colors_bgr[:, 0, [2, 1, 0]].astype(np.float32) / 255.0

    return colors_rgb
