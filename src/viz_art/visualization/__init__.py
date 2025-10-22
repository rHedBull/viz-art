"""3D point cloud visualization tools for interactive and static rendering.

This module provides comprehensive visualization capabilities for point cloud data:

- **Interactive 3D Viewers**: WebGL-based Plotly visualizations with rotation,
  zoom, pan controls and color coding options
- **Static Thumbnails**: Headless Open3D rendering for batch reports and previews
- **Performance Optimizations**: Auto-downsampling, caching, and quality settings
  for handling large point clouds (>500K points)

Key Components:
    - plotly_viewer: Interactive 3D scatter plots with WebGL rendering
    - thumbnail: Static thumbnail generation with multiple viewpoints
    - Automatic downsampling for large clouds
    - Caching system to avoid regenerating identical thumbnails

Examples:
    Create an interactive 3D viewer:
        >>> from viz_art.visualization.plotly_viewer import create_3d_scatter
        >>> from viz_art.types.pointcloud import VisualizationConfig
        >>> config = VisualizationConfig(color_mode="height")
        >>> fig = create_3d_scatter(points, colors=colors, config=config)
        >>> fig.show()

    Generate static thumbnails:
        >>> from viz_art.visualization.thumbnail import render_thumbnail
        >>> img = render_thumbnail(points, viewpoint="diagonal")

Performance Notes:
    - Interactive viewer maintains 30+ FPS for 500K points (SC-002)
    - Thumbnail generation targets < 3s per cloud (SC-006)
    - Caching reduces repeated rendering time by ~10x
    - Auto-downsampling prevents memory issues with large clouds

See Also:
    - viz_art.types.pointcloud.VisualizationConfig: Configuration options
    - examples/test_visualization.py: Complete usage examples
"""

from viz_art.visualization.plotly_viewer import (
    create_3d_scatter,
    color_by_height,
    color_by_intensity,
    downsample_for_display,
    save_html,
    save_json,
)

from viz_art.visualization.thumbnail import (
    render_thumbnail,
    render_multi_view_thumbnail,
    render_thumbnail_with_cache,
    save_thumbnail,
    get_viewpoint_params,
    compute_pointcloud_hash,
)

__all__ = [
    # Plotly interactive viewers
    "create_3d_scatter",
    "color_by_height",
    "color_by_intensity",
    "downsample_for_display",
    "save_html",
    "save_json",
    # Static thumbnails
    "render_thumbnail",
    "render_multi_view_thumbnail",
    "render_thumbnail_with_cache",
    "save_thumbnail",
    "get_viewpoint_params",
    "compute_pointcloud_hash",
]
