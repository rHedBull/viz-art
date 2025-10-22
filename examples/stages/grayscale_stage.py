"""Example Grayscale stage.

This stage converts a color image to grayscale for easy visual verification.
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


class GrayscaleStage(PipelineStage):
    """Convert color images to grayscale.

    This stage takes a numpy image array as input and converts it to grayscale
    while maintaining 3 channels (RGB format with equal values).

    Attributes:
        name: Unique stage identifier
    """

    def __init__(self, name: str = "grayscale"):
        """Initialize GrayscaleStage.

        Args:
            name: Stage identifier
        """
        self._name = name

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
        return ["grayscale_image"]

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
        """Convert image to grayscale.

        Args:
            preprocessed: Dictionary containing validated "image" array

        Returns:
            Dictionary with grayscale image

        Raises:
            RuntimeError: If conversion fails
        """
        try:
            image = preprocessed["image"]

            # Convert to grayscale
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Color image (RGB) - convert to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                # Convert back to 3-channel for consistency
                grayscale_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 2:
                # Already grayscale - convert to 3-channel
                grayscale_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                # Already in correct format or unsupported
                grayscale_image = image

            return {"grayscale": grayscale_image}

        except Exception as e:
            raise RuntimeError(f"Failed to convert to grayscale: {str(e)}")

    def post_process(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Format outputs.

        Args:
            predictions: Dictionary with grayscale image

        Returns:
            Dictionary with "grayscale_image" key
        """
        return {"grayscale_image": predictions["grayscale"]}
