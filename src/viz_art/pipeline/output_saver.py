"""Stage output saving functionality for debugging and analysis.

This module provides the OutputSaver class for saving pipeline stage outputs
to disk in various modes (sample, validation, production).
"""

from pathlib import Path
from typing import Any, Dict, Optional, List
import logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


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
            },
            "validation": {
                "enabled": True,
                "stages": ["all"],  # Save all for error analysis
                "max_samples": None,  # No limit, but only save errors
                "format": "png",
            },
            "production": {
                "enabled": False,  # Disabled by default for efficiency
                "stages": [],
                "max_samples": None,
                "format": "jpg",
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

        # Save each output that is an image (numpy array)
        for key, value in outputs.items():
            if isinstance(value, np.ndarray) and value.ndim in [2, 3]:
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
