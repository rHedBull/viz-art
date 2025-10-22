"""Create a test image for pipeline validation."""

import numpy as np
from PIL import Image
from pathlib import Path


def create_test_image(output_path: str, size: tuple = (640, 480)):
    """Create a simple test image with colored squares.

    Args:
        output_path: Path where to save the image
        size: Image size as (width, height)
    """
    width, height = size

    # Create an RGB image
    img_array = np.zeros((height, width, 3), dtype=np.uint8)

    # Add colored squares
    # Red square (top-left)
    img_array[0:height//2, 0:width//2] = [255, 0, 0]

    # Green square (top-right)
    img_array[0:height//2, width//2:] = [0, 255, 0]

    # Blue square (bottom-left)
    img_array[height//2:, 0:width//2] = [0, 0, 255]

    # Yellow square (bottom-right)
    img_array[height//2:, width//2:] = [255, 255, 0]

    # Convert to PIL Image and save
    img = Image.fromarray(img_array, mode="RGB")
    img.save(output_path)
    print(f"Created test image: {output_path}")


if __name__ == "__main__":
    # Create test images
    fixtures_dir = Path(__file__).parent / "fixtures" / "sample_images"
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    # Create test images of different sizes
    create_test_image(str(fixtures_dir / "test_640x480.jpg"), (640, 480))
    create_test_image(str(fixtures_dir / "test_800x600.jpg"), (800, 600))
    create_test_image(str(fixtures_dir / "test_small.png"), (320, 240))

    print("\nTest images created successfully!")
