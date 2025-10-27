"""Error visualization with side-by-side comparisons and diff views.

This module provides functionality for creating visual comparisons between
predictions and ground truth for error analysis.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np

from .patterns import ErrorCase


class ErrorVisualizer:
    """Create visualizations for error cases."""

    def create_image_diff(
        self,
        prediction_image: np.ndarray,
        ground_truth_image: np.ndarray,
        pred_boxes: Optional[list] = None,
        gt_boxes: Optional[list] = None,
        output_path: Optional[Path] = None
    ) -> np.ndarray:
        """Create image diff with color-coded regions (T047).

        Args:
            prediction_image: Predicted image (H x W x 3)
            ground_truth_image: Ground truth image (H x W x 3)
            pred_boxes: Predicted bounding boxes [[x, y, w, h], ...]
            gt_boxes: Ground truth bounding boxes
            output_path: Path to save visualization (optional)

        Returns:
            Visualization image with color-coded diff regions
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV required for image visualization. Install with: pip install opencv-python")

        # Ensure images are same size
        if prediction_image.shape != ground_truth_image.shape:
            ground_truth_image = cv2.resize(ground_truth_image,
                                           (prediction_image.shape[1], prediction_image.shape[0]))

        # Create side-by-side visualization
        height, width = prediction_image.shape[:2]
        vis = np.zeros((height, width * 2, 3), dtype=np.uint8)
        vis[:, :width] = prediction_image
        vis[:, width:] = ground_truth_image

        # Draw bounding boxes if provided
        if pred_boxes:
            for box in pred_boxes:
                x, y, w, h = box
                # Red for predictions
                cv2.rectangle(vis, (int(x), int(y)), (int(x+w), int(y+h)), (0, 0, 255), 2)
                cv2.putText(vis, "PRED", (int(x), int(y)-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if gt_boxes:
            for box in gt_boxes:
                x, y, w, h = box
                # Green for ground truth (offset to right side)
                cv2.rectangle(vis, (int(x+width), int(y)), (int(x+w+width), int(y+h)), (0, 255, 0), 2)
                cv2.putText(vis, "GT", (int(x+width), int(y)-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save if output path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), vis)

        return vis

    def create_pointcloud_diff(
        self,
        prediction_pcd,
        ground_truth_pcd,
        use_icp: bool = False,
        output_path: Optional[Path] = None
    ) -> Tuple[Any, np.ndarray]:
        """Create point cloud diff using Open3D nearest-neighbor (T048).

        Args:
            prediction_pcd: Predicted point cloud (Open3D PointCloud)
            ground_truth_pcd: Ground truth point cloud
            use_icp: Use ICP alignment fallback (default: False)
            output_path: Path to save visualization (optional)

        Returns:
            Tuple of (colored_pcd, distances_array)
        """
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError("Open3D required for point cloud visualization. Install with: pip install open3d>=0.18")

        # Compute distances (pred -> gt)
        distances = np.asarray(prediction_pcd.compute_point_cloud_distance(ground_truth_pcd))

        # Check if ICP fallback is needed (T049)
        median_distance = np.median(distances)
        if use_icp and median_distance > 0.05:  # 5cm threshold
            # Run ICP alignment
            reg_result = o3d.pipelines.registration.registration_icp(
                prediction_pcd,
                ground_truth_pcd,
                max_correspondence_distance=0.02,
                init=np.eye(4),
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
            )

            # Check ICP succeeded
            if reg_result.fitness > 0.3:
                prediction_pcd.transform(reg_result.transformation)
                distances = np.asarray(prediction_pcd.compute_point_cloud_distance(ground_truth_pcd))

        # Create color-coded visualization (heatmap)
        # Normalize distances for colormap
        max_dist = np.percentile(distances, 95)  # Use 95th percentile to avoid outliers
        normalized_distances = np.clip(distances / max_dist, 0, 1)

        # Apply colormap (blue = close, red = far)
        colors = self._apply_colormap(normalized_distances, colormap='viridis')
        prediction_pcd.colors = o3d.utility.Vector3dVector(colors)

        # Save if output path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            o3d.io.write_point_cloud(str(output_path), prediction_pcd)

        return prediction_pcd, distances

    def _apply_colormap(
        self, values: np.ndarray, colormap: str = 'viridis'
    ) -> np.ndarray:
        """Apply matplotlib colormap to values.

        Args:
            values: Normalized values (0-1)
            colormap: Colormap name

        Returns:
            RGB colors array (N x 3)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            # Fallback to simple gradient if matplotlib not available
            colors = np.zeros((len(values), 3))
            colors[:, 0] = values  # Red channel
            colors[:, 2] = 1 - values  # Blue channel
            return colors

        cmap = plt.get_cmap(colormap)
        colors = cmap(values)[:, :3]  # Take RGB, ignore alpha
        return colors

    def create_side_by_side_visualization(
        self,
        error_case: ErrorCase,
        output_path: Optional[Path] = None
    ) -> Path:
        """Create side-by-side visualization for error case (T050).

        Args:
            error_case: Error case with prediction and ground truth
            output_path: Path to save visualization (optional)

        Returns:
            Path to saved visualization
        """
        import cv2

        # Determine visualization type based on data
        prediction = error_case.prediction
        ground_truth = error_case.ground_truth

        if output_path is None:
            output_path = error_case.saved_artifacts.get("diff_visualization",
                                                        Path(f"/tmp/{error_case.error_id}_diff.jpg"))

        # Image-based visualization
        if isinstance(prediction, dict) and ('boxes' in prediction or 'bbox' in prediction):
            # Detection task - need to load images
            pred_img_path = error_case.saved_artifacts.get("prediction_path")
            gt_img_path = error_case.saved_artifacts.get("ground_truth_path")

            if pred_img_path and pred_img_path.exists() and pred_img_path.suffix in ['.jpg', '.png']:
                pred_img = cv2.imread(str(pred_img_path))
                gt_img = cv2.imread(str(gt_img_path)) if gt_img_path else pred_img.copy()

                pred_boxes = prediction.get('boxes', prediction.get('bbox', []))
                gt_boxes = ground_truth.get('boxes', ground_truth.get('bbox', []))

                self.create_image_diff(pred_img, gt_img, pred_boxes, gt_boxes, output_path)
            else:
                # Create text-based visualization as fallback
                self._create_text_visualization(error_case, output_path)

        elif isinstance(prediction, np.ndarray) and prediction.ndim == 2:
            # Segmentation mask
            self._create_mask_visualization(prediction, ground_truth, output_path)

        else:
            # Fallback: text-based visualization
            self._create_text_visualization(error_case, output_path)

        return output_path

    def _create_mask_visualization(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        output_path: Path
    ) -> None:
        """Create visualization for segmentation masks.

        Args:
            pred_mask: Predicted mask (H x W)
            gt_mask: Ground truth mask (H x W)
            output_path: Output file path
        """
        import cv2

        # Convert masks to colored images
        pred_colored = self._mask_to_color(pred_mask)
        gt_colored = self._mask_to_color(gt_mask)

        # Create side-by-side
        height, width = pred_mask.shape
        vis = np.zeros((height, width * 2, 3), dtype=np.uint8)
        vis[:, :width] = pred_colored
        vis[:, width:] = gt_colored

        # Add labels
        cv2.putText(vis, "PREDICTION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(vis, "GROUND TRUTH", (width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), vis)

    def _mask_to_color(self, mask: np.ndarray) -> np.ndarray:
        """Convert segmentation mask to colored image.

        Args:
            mask: Segmentation mask (H x W)

        Returns:
            Colored image (H x W x 3)
        """
        # Simple colormap for classes
        num_classes = int(mask.max()) + 1
        colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]  # Background is black

        colored = colors[mask]
        return colored

    def _create_text_visualization(
        self,
        error_case: ErrorCase,
        output_path: Path
    ) -> None:
        """Create text-based visualization as fallback.

        Args:
            error_case: Error case
            output_path: Output file path
        """
        import cv2

        # Create blank image
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255

        # Add text information
        y_offset = 30
        texts = [
            f"Error ID: {error_case.error_id[:8]}",
            f"Sample: {error_case.sample_id}",
            f"Stage: {error_case.stage_name}",
            f"Type: {error_case.error_type.value}",
            f"Severity: {error_case.severity.value}",
            "",
            f"Prediction: {str(error_case.prediction)[:50]}",
            f"Ground Truth: {str(error_case.ground_truth)[:50]}",
        ]

        if error_case.iou is not None:
            texts.append(f"IoU: {error_case.iou:.3f}")
        if error_case.confidence is not None:
            texts.append(f"Confidence: {error_case.confidence:.3f}")

        for text in texts:
            cv2.putText(img, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 0, 0), 1)
            y_offset += 25

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), img)

    def create_diff_visualization(
        self,
        error_case: ErrorCase,
        diff_type: str = "auto",
        output_path: Optional[Path] = None
    ) -> Path:
        """Create diff visualization with auto mode (T051).

        Args:
            error_case: Error case to visualize
            diff_type: Type of diff ("auto", "image_diff", "pointcloud_heatmap")
            output_path: Path to save visualization

        Returns:
            Path to saved visualization
        """
        if output_path is None:
            output_path = Path(f"/tmp/{error_case.error_id}_diff.jpg")

        # Auto-detect diff type
        if diff_type == "auto":
            prediction = error_case.prediction

            if hasattr(prediction, 'points'):  # Open3D PointCloud
                diff_type = "pointcloud_heatmap"
            elif isinstance(prediction, np.ndarray) and prediction.ndim >= 2:
                diff_type = "mask_diff"
            else:
                diff_type = "image_diff"

        # Create appropriate visualization
        if diff_type == "pointcloud_heatmap":
            pred_pcd_path = error_case.saved_artifacts.get("prediction_pcd")
            gt_pcd_path = error_case.saved_artifacts.get("ground_truth_pcd")

            if pred_pcd_path and gt_pcd_path:
                import open3d as o3d
                pred_pcd = o3d.io.read_point_cloud(str(pred_pcd_path))
                gt_pcd = o3d.io.read_point_cloud(str(gt_pcd_path))

                colored_pcd, _ = self.create_pointcloud_diff(pred_pcd, gt_pcd, output_path=output_path)
                return output_path
            else:
                # Fallback
                return self.create_side_by_side_visualization(error_case, output_path)

        elif diff_type == "mask_diff":
            pred_mask = error_case.prediction
            gt_mask = error_case.ground_truth

            if isinstance(pred_mask, np.ndarray) and isinstance(gt_mask, np.ndarray):
                self._create_mask_visualization(pred_mask, gt_mask, output_path)
                return output_path

        # Default to side-by-side
        return self.create_side_by_side_visualization(error_case, output_path)
