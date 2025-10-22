"""Stage output saving functionality for debugging and analysis.

This module provides the OutputSaver class for saving pipeline stage outputs
to disk in various modes (sample, validation, production).
Supports both images and point clouds.
"""

from pathlib import Path
from typing import Any, Dict, Optional, List
import logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Try to import Open3D for point cloud support
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    o3d = None  # type: ignore


class OutputSaver:
    """Save stage outputs to disk based on configuration.

    Supports three modes:
    - sample: Save all stages for limited samples (debugging)
    - validation: Save failures and final outputs (testing)
    - production: Save final outputs only or nothing (efficiency)

    Example:
        >>> config = {"enabled": True, "stages": ["all"], "max_samples": 10}
        >>> saver = OutputSaver("sample", config, output_dir="./output")
        >>> saver.save_stage_output("run_123", "filter", "img1.jpg",
        ...                         numpy_array, stage_index=1)
    """

    def __init__(
        self,
        output_mode: str = "sample",
        save_config: Optional[Dict[str, Any]] = None,
        output_dir: str = "./output",
    ):
        """Initialize OutputSaver.

        Args:
            output_mode: One of "sample", "validation", "production"
            save_config: Configuration dict with keys:
                - enabled: bool (default True in sample mode)
                - stages: list of stage names or ["all"]
                - max_samples: int (for sample mode)
                - format: "png", "jpg", or "npy"
            output_dir: Base directory for outputs

        Raises:
            ValueError: If output_mode is invalid
        """
        valid_modes = ["sample", "validation", "production"]
        if output_mode not in valid_modes:
            raise ValueError(
                f"output_mode must be one of {valid_modes}, got '{output_mode}'"
            )

        self.output_mode = output_mode
        self.output_dir = Path(output_dir)

        # Default config based on mode
        default_config = self._get_default_config(output_mode)
        self.config = {**default_config, **(save_config or {})}

        self._saved_count = 0

        logger.debug(
            f"OutputSaver initialized: mode={output_mode}, config={self.config}"
        )

    def _get_default_config(self, mode: str) -> Dict[str, Any]:
        """Get default configuration for each mode."""
        defaults = {
            "sample": {
                "enabled": True,
                "stages": ["all"],
                "max_samples": 10,
                "format": "png",
                "pointcloud_format": "ply",  # .ply supports colors/normals
            },
            "validation": {
                "enabled": True,
                "stages": ["all"],  # Save all for error analysis
                "max_samples": None,  # No limit, but only save errors
                "format": "png",
                "pointcloud_format": "ply",
            },
            "production": {
                "enabled": False,  # Disabled by default for efficiency
                "stages": [],
                "max_samples": None,
                "format": "jpg",
                "pointcloud_format": "pcd",  # Open3D native format
            },
        }
        return defaults[mode]

    def should_save(
        self, stage_name: str, image_index: int = 0, is_error: bool = False
    ) -> bool:
        """Determine if output should be saved for this stage.

        Args:
            stage_name: Name of the pipeline stage
            image_index: Index of current image in batch
            is_error: Whether this is an error case

        Returns:
            True if output should be saved
        """
        # Check if saving is enabled
        if not self.config.get("enabled", False):
            return False

        # In validation mode, always save errors
        if self.output_mode == "validation" and is_error:
            return True

        # Check stage filter
        stages = self.config.get("stages", [])
        if stages != ["all"] and stage_name not in stages:
            return False

        # Check max_samples limit (sample mode only)
        max_samples = self.config.get("max_samples")
        if max_samples is not None and image_index >= max_samples:
            return False

        return True

    def save_stage_output(
        self,
        run_id: str,
        stage_name: str,
        filename: str,
        outputs: Dict[str, Any],
        image_index: int = 0,
        is_error: bool = False,
    ) -> Dict[str, str]:
        """Save stage outputs to disk.

        Args:
            run_id: Unique run identifier
            stage_name: Name of the pipeline stage
            filename: Original input filename (for organizing outputs)
            outputs: Stage output dictionary (may contain numpy arrays)
            image_index: Index of current image in batch
            is_error: Whether this is an error case

        Returns:
            Dictionary mapping output keys to saved file paths

        Example:
            >>> outputs = {"image": np.array(...), "metadata": {...}}
            >>> paths = saver.save_stage_output("run_123", "filter", "img1.jpg", outputs)
            >>> # paths = {"image": "./output/runs/run_123/filter/img1.png"}
        """
        if not self.should_save(stage_name, image_index, is_error):
            return {}

        # Create output directory structure
        stage_dir = self._get_stage_dir(run_id, stage_name, is_error)
        stage_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = {}
        base_filename = Path(filename).stem

        # Save each output
        for key, value in outputs.items():
            # Check if it's a point cloud (Open3D object)
            if OPEN3D_AVAILABLE and isinstance(value, o3d.geometry.PointCloud):
                output_path = self._save_pointcloud(
                    value, stage_dir, f"{base_filename}_{key}"
                )
                if output_path:
                    saved_paths[key] = str(output_path)

            # Check if it's a point array (could be Nx3 point cloud coordinates)
            elif isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[1] == 3:
                # Could be point cloud or image - check size to distinguish
                if value.shape[0] > 100:  # Likely point cloud if many rows
                    output_path = self._save_point_array(
                        value, stage_dir, f"{base_filename}_{key}"
                    )
                    if output_path:
                        saved_paths[key] = str(output_path)
                else:
                    # Small Nx3 array, treat as image
                    output_path = self._save_image_array(
                        value, stage_dir, f"{base_filename}_{key}"
                    )
                    if output_path:
                        saved_paths[key] = str(output_path)

            # Check if it's an image (numpy array)
            elif isinstance(value, np.ndarray) and value.ndim in [2, 3]:
                # This looks like an image
                output_path = self._save_image_array(
                    value, stage_dir, f"{base_filename}_{key}"
                )
                if output_path:
                    saved_paths[key] = str(output_path)

        if saved_paths:
            self._saved_count += 1
            logger.debug(
                f"Saved {len(saved_paths)} outputs for {stage_name}/{filename}"
            )

        return saved_paths

    def _get_stage_dir(self, run_id: str, stage_name: str, is_error: bool) -> Path:
        """Get directory path for stage outputs.

        Structure:
        - sample mode: output/runs/{run_id}/stages/{stage_name}/
        - validation mode (errors): output/validation/{run_id}/errors/{stage_name}/
        - validation mode (normal): output/validation/{run_id}/stages/{stage_name}/
        - production mode: output/production/{run_id}/final/
        """
        if self.output_mode == "sample":
            return self.output_dir / "runs" / run_id / "stages" / stage_name
        elif self.output_mode == "validation":
            if is_error:
                return self.output_dir / "validation" / run_id / "errors" / stage_name
            else:
                return self.output_dir / "validation" / run_id / "stages" / stage_name
        else:  # production
            return self.output_dir / "production" / run_id / "final"

    def _save_image_array(
        self, array: np.ndarray, output_dir: Path, base_name: str
    ) -> Optional[Path]:
        """Save numpy array as image file.

        Args:
            array: Numpy array (H, W) or (H, W, C)
            output_dir: Directory to save to
            base_name: Base filename (without extension)

        Returns:
            Path to saved file, or None if save failed
        """
        try:
            img_format = self.config.get("format", "png")
            output_path = output_dir / f"{base_name}.{img_format}"

            # Handle different array shapes
            if array.ndim == 2:
                # Grayscale image
                img = Image.fromarray(array.astype(np.uint8))
            elif array.ndim == 3:
                # Color image (H, W, C)
                if array.shape[2] == 3:
                    # RGB
                    img = Image.fromarray(array.astype(np.uint8), mode="RGB")
                elif array.shape[2] == 4:
                    # RGBA
                    img = Image.fromarray(array.astype(np.uint8), mode="RGBA")
                else:
                    logger.warning(
                        f"Unsupported channel count: {array.shape[2]}, skipping"
                    )
                    return None
            else:
                logger.warning(f"Unsupported array shape: {array.shape}, skipping")
                return None

            # Save with appropriate quality
            if img_format == "jpg":
                img.save(output_path, quality=85, optimize=True)
            else:
                img.save(output_path, optimize=True)

            return output_path

        except Exception as e:
            logger.error(f"Failed to save image array: {e}")
            return None

    def _save_pointcloud(
        self, pcd: "o3d.geometry.PointCloud", output_dir: Path, base_name: str
    ) -> Optional[Path]:
        """Save Open3D point cloud to file.

        Args:
            pcd: Open3D PointCloud object
            output_dir: Directory to save to
            base_name: Base filename (without extension)

        Returns:
            Path to saved file, or None if save failed
        """
        if not OPEN3D_AVAILABLE or o3d is None:
            logger.warning("Open3D not available, cannot save point cloud")
            return None

        try:
            pc_format = self.config.get("pointcloud_format", "ply")
            output_path = output_dir / f"{base_name}.{pc_format}"

            # Save point cloud
            success = o3d.io.write_point_cloud(str(output_path), pcd)

            if success:
                logger.debug(f"Saved point cloud to {output_path}")
                return output_path
            else:
                logger.error(f"Failed to save point cloud to {output_path}")
                return None

        except Exception as e:
            logger.error(f"Failed to save point cloud: {e}")
            return None

    def _save_point_array(
        self, points: np.ndarray, output_dir: Path, base_name: str
    ) -> Optional[Path]:
        """Save point array (Nx3) as point cloud file.

        Args:
            points: Nx3 NumPy array of XYZ coordinates
            output_dir: Directory to save to
            base_name: Base filename (without extension)

        Returns:
            Path to saved file, or None if save failed
        """
        if not OPEN3D_AVAILABLE or o3d is None:
            logger.warning("Open3D not available, saving as .xyz text file instead")
            # Fallback: save as .xyz text format
            try:
                output_path = output_dir / f"{base_name}.xyz"
                np.savetxt(output_path, points, fmt='%.6f')
                logger.debug(f"Saved point array as XYZ to {output_path}")
                return output_path
            except Exception as e:
                logger.error(f"Failed to save point array as XYZ: {e}")
                return None

        try:
            # Convert NumPy array to Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

            # Save using standard method
            return self._save_pointcloud(pcd, output_dir, base_name)

        except Exception as e:
            logger.error(f"Failed to save point array: {e}")
            return None

    def get_saved_count(self) -> int:
        """Get number of outputs saved so far.

        Returns:
            Count of saved outputs
        """
        return self._saved_count

    def get_run_dir(self, run_id: str) -> Path:
        """Get the root directory for a specific run.

        Args:
            run_id: Unique run identifier

        Returns:
            Path to run directory
        """
        if self.output_mode == "sample":
            return self.output_dir / "runs" / run_id
        elif self.output_mode == "validation":
            return self.output_dir / "validation" / run_id
        else:
            return self.output_dir / "production" / run_id

    def generate_thumbnail(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        output_path: Optional[Path] = None,
        width: int = 400,
        height: int = 300,
        viewpoint: str = "diagonal",
    ) -> Optional[np.ndarray]:
        """Generate thumbnail image for point cloud.

        Args:
            points: Nx3 array of XYZ coordinates
            colors: Optional Nx3 array of RGB values in [0,1]
            output_path: Path to save thumbnail (None = don't save)
            width: Thumbnail width (pixels)
            height: Thumbnail height (pixels)
            viewpoint: Camera viewpoint ("front", "top", "side", "diagonal")

        Returns:
            Rendered thumbnail as HxWx3 NumPy array, or None if rendering fails

        Example:
            >>> thumbnail = saver.generate_thumbnail(points, colors, Path("thumb.png"))
        """
        try:
            from viz_art.visualization.thumbnail import render_thumbnail, save_thumbnail

            # Render thumbnail
            img = render_thumbnail(
                points,
                colors=colors,
                width=width,
                height=height,
                viewpoint=viewpoint,
                point_size=2.0,
                background_color=(1.0, 1.0, 1.0),
            )

            # Save if path specified
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                save_thumbnail(img, str(output_path), quality=85)
                logger.debug(f"Generated thumbnail: {output_path}")

            return img

        except ImportError as e:
            logger.warning(f"Cannot generate thumbnail: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to generate thumbnail: {e}")
            return None
