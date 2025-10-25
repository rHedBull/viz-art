"""Create sample data for examples."""
import numpy as np
from PIL import Image
from pathlib import Path

# Create data directory
data_dir = Path(__file__).parent
data_dir.mkdir(exist_ok=True)

# Create 3 sample images
print("Creating sample images...")
for i in range(3):
    # Create a simple gradient image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:, :, 0] = np.linspace(0, 255, 640, dtype=np.uint8)  # Red gradient
    img[:, :, 1] = np.linspace(0, 255, 480, dtype=np.uint8).reshape(-1, 1)  # Green gradient
    img[:, :, 2] = 128 + i * 40  # Blue varies per image

    Image.fromarray(img).save(data_dir / f"sample_{i:03d}.jpg", quality=95)
    print(f"  Created sample_{i:03d}.jpg")

# Create 3 sample point clouds
print("\nCreating sample point clouds...")
for i in range(3):
    # Create random point cloud (N x 3)
    num_points = 1000 + i * 500
    points = np.random.randn(num_points, 3).astype(np.float32) * 10.0

    # Save as .npy for simplicity (could use .pcd format with open3d)
    np.save(data_dir / f"cloud_{i:03d}.npy", points)
    print(f"  Created cloud_{i:03d}.npy ({num_points} points)")

print("\n✓ Sample data created in examples/data/")
