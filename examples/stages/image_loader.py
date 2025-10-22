"""Example ImageLoader stage.

This stage loads an image from a file path and converts it to a numpy array.
"""

from typing import Dict, Any, List
from pathlib import Path
import sys

# Add src to path if running examples directly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
from PIL import Image
from viz_art.pipeline import PipelineStage


class ImageLoader(PipelineStage):
    """Load and preprocess images from file paths.

    This stage takes an image file path as input and produces a numpy array
    containing the image data.

    Attributes:
        name: Unique stage identifier
        resize: Optional tuple (width, height) to resize images
        color_mode: Color mode ("RGB" or "L" for grayscale)
    """

    def __init__(
        self,
        name: str = "loader",
        resize: tuple = None,
        color_mode: str = "RGB",
    ):
        """Initialize ImageLoader stage.

        Args:
            name: Stage identifier
            resize: Optional (width, height) tuple for resizing
            color_mode: PIL color mode ("RGB", "L", "RGBA", etc.)
        """
        self._name = name
        self.resize = resize
        self.color_mode = color_mode

    @property
    def name(self) -> str:
        """Return stage name."""
        return self._name

    @property
    def input_keys(self) -> List[str]:
        """Return required input keys."""
        return ["image_path"]

    @property
    def output_keys(self) -> List[str]:
        """Return output keys produced by this stage."""
        return ["image", "metadata"]

    def pre_process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input and prepare for loading.

        Args:
            inputs: Dictionary containing "image_path"

        Returns:
            Validated inputs

        Raises:
            ValueError: If image_path is missing or invalid
        """
        if "image_path" not in inputs:
            raise ValueError("Missing required input: image_path")

        image_path = inputs["image_path"]
        if not image_path:
            raise ValueError("image_path cannot be empty")

        path = Path(image_path)
        if not path.exists():
            raise ValueError(f"Image file not found: {image_path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {image_path}")

        return inputs

    def predict(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        """Load image from file path.

        Args:
            preprocessed: Dictionary containing validated "image_path"

        Returns:
            Dictionary with loaded image array and metadata

        Raises:
            RuntimeError: If image loading fails
        """
        try:
            image_path = preprocessed["image_path"]
            img = Image.open(image_path)

            # Convert to specified color mode
            if img.mode != self.color_mode:
                img = img.convert(self.color_mode)

            # Resize if requested
            if self.resize:
                img = img.resize(self.resize, Image.Resampling.LANCZOS)

            # Convert to numpy array
            img_array = np.array(img)

            return {
                "image_array": img_array,
                "original_path": str(image_path),
                "shape": img_array.shape,
                "dtype": str(img_array.dtype),
            }

        except Exception as e:
            raise RuntimeError(f"Failed to load image: {str(e)}")

    def post_process(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Format outputs.

        Args:
            predictions: Dictionary with image array and metadata

        Returns:
            Dictionary with "image" and "metadata" keys
        """
        return {
            "image": predictions["image_array"],
            "metadata": {
                "path": predictions["original_path"],
                "shape": predictions["shape"],
                "dtype": predictions["dtype"],
            },
        }
