"""Example SimpleFilter stage.

This stage applies a Gaussian blur filter to an image using OpenCV.
"""

from typing import Dict, Any, List
from pathlib import Path
import sys

# Add src to path if running examples directly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import cv2
from viz_art.pipeline import PipelineStage


class SimpleFilter(PipelineStage):
    """Apply Gaussian blur filter to images.

    This stage takes a numpy image array as input and applies a Gaussian blur
    filter using OpenCV.

    Attributes:
        name: Unique stage identifier
        kernel_size: Size of the Gaussian kernel (must be odd)
        sigma: Gaussian kernel standard deviation
    """

    def __init__(
        self,
        name: str = "filter",
        kernel_size: int = 5,
        sigma: float = 0.0,
    ):
        """Initialize SimpleFilter stage.

        Args:
            name: Stage identifier
            kernel_size: Size of Gaussian kernel (must be odd, positive integer)
            sigma: Standard deviation for Gaussian kernel (0 = auto-calculate)
        """
        self._name = name
        self.kernel_size = kernel_size
        self.sigma = sigma

        # Validate kernel size
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

    @property
    def name(self) -> str:
        """Return stage name."""
        return self._name

    @property
    def input_keys(self) -> List[str]:
        """Return required input keys."""
        return ["image"]

    @property
    def output_keys(self) -> List[str]:
        """Return output keys produced by this stage."""
        return ["filtered_image"]

    def pre_process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input image.

        Args:
            inputs: Dictionary containing "image" numpy array

        Returns:
            Validated inputs

        Raises:
            ValueError: If image is missing or invalid
        """
        if "image" not in inputs:
            raise ValueError("Missing required input: image")

        image = inputs["image"]
        if not isinstance(image, np.ndarray):
            raise ValueError(f"image must be numpy array, got {type(image)}")

        if image.size == 0:
            raise ValueError("image array is empty")

        return inputs

    def predict(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Gaussian blur filter.

        Args:
            preprocessed: Dictionary containing validated "image" array

        Returns:
            Dictionary with filtered image

        Raises:
            RuntimeError: If filtering fails
        """
        try:
            image = preprocessed["image"]

            # Apply Gaussian blur
            filtered = cv2.GaussianBlur(
                image, (self.kernel_size, self.kernel_size), self.sigma
            )

            return {"filtered": filtered}

        except Exception as e:
            raise RuntimeError(f"Failed to apply filter: {str(e)}")

    def post_process(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Format outputs.

        Args:
            predictions: Dictionary with filtered image

        Returns:
            Dictionary with "filtered_image" key
        """
        return {"filtered_image": predictions["filtered"]}
