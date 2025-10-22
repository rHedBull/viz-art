"""Camera calibration utilities for multi-modal data processing.

This module provides tools for loading and working with camera calibration data,
enabling projection of 3D point clouds onto 2D images.
"""

from viz_art.calibration.loader import load_calibration, validate_calibration
from viz_art.calibration.projection import (
    project_points,
    filter_visible_points,
    overlay_points_on_image,
    compute_depth_colormap,
)

__all__ = [
    "load_calibration",
    "validate_calibration",
    "project_points",
    "filter_visible_points",
    "overlay_points_on_image",
    "compute_depth_colormap",
]
