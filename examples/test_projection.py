"""Example: Overlay point cloud on image using camera calibration.

This script demonstrates how to project 3D point cloud data onto a 2D image
using camera calibration parameters.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import cv2
from stages.pointcloud_projection import PointCloudProjectionStage


def generate_synthetic_data():
    """Generate synthetic test data (image and point cloud).

    Returns:
        Tuple of (image, points_3d)
    """
    # Create a simple test image (640x480, blue gradient)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(480):
        image[i, :, 0] = int(i / 480 * 255)  # Blue channel gradient

    # Create synthetic 3D points (grid in front of camera)
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-1.5, 1.5, 15)
    z = np.full(300, 5.0)  # 5 meters away

    xx, yy = np.meshgrid(x, y)
    points_3d = np.stack([xx.ravel(), yy.ravel(), z], axis=1).astype(np.float32)

    return image, points_3d


def main():
    """Run point cloud projection example."""
    print("=" * 60)
    print("Point Cloud Projection Example")
    print("=" * 60)

    # Generate synthetic test data
    print("\n1. Generating synthetic test data...")
    image, points_3d = generate_synthetic_data()
    print(f"   Image shape: {image.shape}")
    print(f"   Points shape: {points_3d.shape}")
    print(f"   Number of points: {points_3d.shape[0]}")

    # Get calibration path
    calibration_path = Path(__file__).parent / "calibration" / "camera.yaml"

    if not calibration_path.exists():
        print(f"\n❌ Error: Calibration file not found: {calibration_path}")
        print("   Please ensure examples/calibration/camera.yaml exists")
        return 1

    print(f"   Calibration file: {calibration_path}")

    # Create projection stage
    print("\n2. Creating projection stage...")
    stage = PointCloudProjectionStage(
        name="projection",
        calibration_path=calibration_path,
        color_mode="depth",
        point_radius=3,
        opacity=0.8,
    )
    print("   ✓ Stage created")

    # Run projection
    print("\n3. Projecting points onto image...")
    try:
        result = stage.run({
            "image": image,
            "points": points_3d,
        })
        print("   ✓ Projection complete")
    except Exception as e:
        print(f"   ❌ Projection failed: {e}")
        return 1

    # Display results
    print("\n4. Results:")
    print(f"   Total points: {points_3d.shape[0]}")
    print(f"   Visible points: {result['num_visible']}")
    print(f"   Visibility: {result['num_visible'] / points_3d.shape[0] * 100:.1f}%")

    # Save output
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "projected_overlay.png"
    cv2.imwrite(str(output_path), cv2.cvtColor(result["projected_image"], cv2.COLOR_RGB2BGR))
    print(f"\n5. Output saved to: {output_path}")

    # Optionally display (if running in GUI environment)
    try:
        cv2.imshow("Projected Point Cloud Overlay", cv2.cvtColor(result["projected_image"], cv2.COLOR_RGB2BGR))
        print("\nPress any key to close the display window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("\n(Skipping display - no GUI environment detected)")

    print("\n" + "=" * 60)
    print("✓ Example completed successfully!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
