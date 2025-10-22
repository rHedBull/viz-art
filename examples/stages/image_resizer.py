"""Example ImageResizer stage.

This stage resizes an image to a specified dimension.
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


class ImageResizer(PipelineStage):
    """Resize images to specified dimensions.

    Attributes:
        name: Unique stage identifier
        width: Target width
        height: Target height
    """

    def __init__(self, name: str = "resizer", width: int = 320, height: int = 240):
        """Initialize ImageResizer stage.

        Args:
            name: Stage identifier
            width: Target width in pixels
            height: Target height in pixels
        """
        self._name = name
        self.width = width
        self.height = height

    @property
    def name(self) -> str:
        """Return stage name."""
        return self._name

    @property
    def input_keys(self) -> List[str]:
        """Return required input keys."""
        return ["filtered_image"]

    @property
    def output_keys(self) -> List[str]:
        """Return output keys produced by this stage."""
        return ["resized_image"]

    def pre_process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input."""
        if "filtered_image" not in inputs:
            raise ValueError("Missing required input: filtered_image")
        return inputs

    def predict(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        """Resize image."""
        image = preprocessed["filtered_image"]
        resized = cv2.resize(image, (self.width, self.height))
        return {"resized": resized}

    def post_process(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Format outputs."""
        return {"resized_image": predictions["resized"]}
