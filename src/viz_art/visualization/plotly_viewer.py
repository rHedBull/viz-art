"""Plotly-based 3D point cloud visualization with WebGL rendering."""

from typing import Optional, Dict, Any, Tuple
import numpy as np
import plotly.graph_objects as go
from viz_art.types.pointcloud import VisualizationConfig, ColorMode


def create_3d_scatter(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    intensity: Optional[np.ndarray] = None,
    config: Optional[VisualizationConfig] = None,
) -> go.Figure:
    """Generate Plotly 3D scatter plot from point arrays.

    Args:
        points: Nx3 array of XYZ coordinates
        colors: Optional Nx3 array of RGB values in [0,1]
        intensity: Optional N array of intensity values
        config: Visualization configuration (uses defaults if None)

    Returns:
        Plotly Figure object with 3D scatter plot

    Raises:
        ValueError: If points array is invalid or empty
    """
    if config is None:
        config = VisualizationConfig()

    if points.shape[0] == 0:
        raise ValueError("Cannot create scatter plot from empty point cloud")

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Points must be Nx3 array, got shape {points.shape}")

    # Extract coordinates
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Determine color values based on mode
    color_values = None
    colorscale = config.colorscale

    if config.color_mode == "height":
        color_values = color_by_height(points, colorscale=colorscale)
    elif config.color_mode == "intensity" and intensity is not None:
        color_values = color_by_intensity(intensity, colorscale=colorscale)
    elif config.color_mode == "rgb" and colors is not None:
        # Convert RGB [0,1] to hex colors for Plotly
        color_values = [
            f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
            for r, g, b in colors
        ]
    else:
        # Default to height if requested mode unavailable
        color_values = color_by_height(points, colorscale=colorscale)

    # Create scatter plot
    scatter = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=config.point_size,
            color=color_values,
            colorscale=colorscale if not isinstance(color_values, list) else None,
            opacity=config.opacity,
            colorbar=dict(title="Value") if not isinstance(color_values, list) else None,
        ),
        hovertemplate=(
            "X: %{x:.2f}<br>"
            "Y: %{y:.2f}<br>"
            "Z: %{z:.2f}<br>"
            "<extra></extra>"
        ),
    )

    # Create figure
    fig = go.Figure(data=[scatter])

    # Configure layout
    bg_color = config.background_color
    bg_color_str = f"rgba({int(bg_color[0]*255)},{int(bg_color[1]*255)},{int(bg_color[2]*255)},{bg_color[3]})"

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X', backgroundcolor=bg_color_str),
            yaxis=dict(title='Y', backgroundcolor=bg_color_str),
            zaxis=dict(title='Z', backgroundcolor=bg_color_str),
            camera=dict(projection=dict(type=config.camera_projection)),
        ),
        paper_bgcolor=bg_color_str,
        plot_bgcolor=bg_color_str,
        title="3D Point Cloud Viewer",
        hovermode='closest',
    )

    return fig


def color_by_height(
    points: np.ndarray,
    colorscale: str = "Viridis"
) -> np.ndarray:
    """Map Z coordinates to color scale.

    Args:
        points: Nx3 array of XYZ coordinates
        colorscale: Plotly colorscale name (unused, returns Z values)

    Returns:
        N array of Z values for color mapping
    """
    if points.shape[0] == 0:
        return np.array([])

    return points[:, 2]  # Return Z coordinates


def color_by_intensity(
    intensity: np.ndarray,
    colorscale: str = "Viridis"
) -> np.ndarray:
    """Map intensity values to color scale.

    Args:
        intensity: N array of intensity values
        colorscale: Plotly colorscale name (unused, returns intensity values)

    Returns:
        N array of intensity values for color mapping

    Raises:
        ValueError: If intensity array is invalid
    """
    if intensity.ndim != 1:
        raise ValueError(f"Intensity must be 1D array, got shape {intensity.shape}")

    if intensity.shape[0] == 0:
        return np.array([])

    return intensity


def downsample_for_display(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    intensity: Optional[np.ndarray] = None,
    max_points: int = 500000,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Auto-downsample large point clouds for WebGL performance.

    Uses uniform random sampling to reduce point count while preserving
    overall structure and distribution.

    Args:
        points: Nx3 array of XYZ coordinates
        colors: Optional Nx3 array of RGB values
        intensity: Optional N array of intensity values
        max_points: Maximum points to retain

    Returns:
        Tuple of (downsampled_points, downsampled_colors, downsampled_intensity)

    Raises:
        ValueError: If max_points <= 0
    """
    if max_points <= 0:
        raise ValueError(f"max_points must be > 0, got {max_points}")

    num_points = points.shape[0]

    # No downsampling needed
    if num_points <= max_points:
        return points, colors, intensity

    # Uniform random sampling
    rng = np.random.default_rng(seed=42)  # Fixed seed for reproducibility
    indices = rng.choice(num_points, size=max_points, replace=False)
    indices = np.sort(indices)  # Sort for better cache locality

    # Downsample all arrays
    downsampled_points = points[indices]
    downsampled_colors = colors[indices] if colors is not None else None
    downsampled_intensity = intensity[indices] if intensity is not None else None

    return downsampled_points, downsampled_colors, downsampled_intensity


def save_html(fig: go.Figure, output_path: str) -> str:
    """Save Plotly figure as standalone HTML file.

    Args:
        fig: Plotly figure to save
        output_path: Output file path (.html)

    Returns:
        Absolute path to saved HTML file

    Raises:
        IOError: If file cannot be written
    """
    from pathlib import Path

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.write_html(
        str(output_path),
        include_plotlyjs='cdn',  # Use CDN for smaller file size
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'pointcloud_viewer',
                'height': 1080,
                'width': 1920,
                'scale': 1
            }
        }
    )

    return str(output_path)


def save_json(fig: go.Figure, output_path: str) -> str:
    """Save Plotly figure as JSON file.

    Args:
        fig: Plotly figure to save
        output_path: Output file path (.json)

    Returns:
        Absolute path to saved JSON file

    Raises:
        IOError: If file cannot be written
    """
    from pathlib import Path

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.write_json(str(output_path))

    return str(output_path)
