#!/usr/bin/env python3
"""Generate sample test images for batch processing demo."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random

def create_test_image(output_path: Path, width: int = 640, height: int = 480, label: str = "Test"):
    """Create a simple test image with colored background and label."""
    # Random pastel color
    r = random.randint(150, 255)
    g = random.randint(150, 255)
    b = random.randint(150, 255)

    # Create image
    img = Image.new('RGB', (width, height), color=(r, g, b))
    draw = ImageDraw.Draw(img)

    # Add some shapes
    for _ in range(5):
        x1 = random.randint(50, width - 100)
        y1 = random.randint(50, height - 100)
        size = random.randint(30, 100)
        x2 = x1 + size
        y2 = y1 + size

        shape_color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )

        draw.ellipse([x1, y1, x2, y2], fill=shape_color)

    # Add text label
    try:
        # Try to use default font
        draw.text((20, 20), label, fill=(0, 0, 0))
    except Exception:
        # If font loading fails, just skip text
        pass

    # Save image
    img.save(output_path, quality=95)
    print(f"Created: {output_path}")

def main():
    """Create a set of test images."""
    input_dir = Path(__file__).parent / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    # Create various test images
    test_images = [
        ("test_landscape_1.jpg", 800, 600, "Landscape 1"),
        ("test_landscape_2.jpg", 800, 600, "Landscape 2"),
        ("test_portrait_1.jpg", 600, 800, "Portrait 1"),
        ("test_square_1.jpg", 512, 512, "Square 1"),
        ("test_square_2.jpg", 512, 512, "Square 2"),
    ]

    for filename, width, height, label in test_images:
        output_path = input_dir / filename
        create_test_image(output_path, width, height, label)

    print(f"\nCreated {len(test_images)} test images in {input_dir}")

if __name__ == "__main__":
    main()
