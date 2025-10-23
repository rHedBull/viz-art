"""Error Analysis for Ground Truth Validation

Provides tools for analyzing validation failures, categorizing errors,
and generating comparison visualizations.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw


class ErrorAnalyzer:
    """Analyzer for validation errors and failures.

    Provides methods to identify failures, categorize error types,
    and generate visual comparisons between predictions and ground truth.
    """

    def __init__(self, output_dir: Path):
        """Initialize error analyzer.

        Args:
            output_dir: Directory to save analysis outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def find_failures(
        self,
        run_id: str,
        predictions: Dict[str, Any],
        ground_truth: Dict[str, Any],
        iou_threshold: float = 0.5,
        confidence_threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Find prediction failures by comparing against ground truth.

        Args:
            run_id: Pipeline run identifier
            predictions: Predicted outputs {sample_id: prediction}
            ground_truth: Ground truth annotations {sample_id: annotation}
            iou_threshold: IoU threshold for considering match (detection)
            confidence_threshold: Minimum confidence for predictions

        Returns:
            List of failure records with format:
            [
                {
                    'sample_id': str,
                    'prediction': Any,
                    'ground_truth': Any,
                    'error_type': str,  # 'false_positive', 'false_negative', 'misclassification'
                    'iou': float,  # if applicable
                },
                ...
            ]
        """
        failures = []

        # Find all samples in either predictions or ground truth
        all_samples = set(predictions.keys()) | set(ground_truth.keys())

        for sample_id in all_samples:
            pred = predictions.get(sample_id)
            gt = ground_truth.get(sample_id)

            # Case 1: Ground truth exists but no prediction (False Negative)
            if gt and not pred:
                failures.append(
                    {
                        "sample_id": sample_id,
                        "prediction": None,
                        "ground_truth": gt,
                        "error_type": "false_negative",
                        "iou": 0.0,
                    }
                )
                continue

            # Case 2: Prediction exists but no ground truth (False Positive)
            if pred and not gt:
                failures.append(
                    {
                        "sample_id": sample_id,
                        "prediction": pred,
                        "ground_truth": None,
                        "error_type": "false_positive",
                        "iou": 0.0,
                    }
                )
                continue

            # Case 3: Both exist - check for misclassification or low IoU
            if pred and gt:
                # Handle detection case (bboxes)
                if isinstance(pred, list) and isinstance(gt, list):
                    # Check if prediction has sufficient overlap with ground truth
                    from .metrics import MetricsCalculator

                    calc = MetricsCalculator()

                    # Filter predictions by confidence
                    filtered_preds = [
                        p for p in pred if p.get("score", 1.0) >= confidence_threshold
                    ]

                    # Find unmatched ground truths (false negatives)
                    matched_gt_indices = set()
                    for pred_box in filtered_preds:
                        best_iou = 0.0
                        best_gt_idx = -1

                        for gt_idx, gt_box in enumerate(gt):
                            iou = calc._calculate_iou(pred_box["bbox"], gt_box["bbox"])
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = gt_idx

                        if best_iou >= iou_threshold:
                            matched_gt_indices.add(best_gt_idx)

                            # Check for misclassification
                            if pred_box.get("class") != gt[best_gt_idx].get(
                                "category_name"
                            ):
                                failures.append(
                                    {
                                        "sample_id": sample_id,
                                        "prediction": pred_box,
                                        "ground_truth": gt[best_gt_idx],
                                        "error_type": "misclassification",
                                        "iou": best_iou,
                                    }
                                )
                        else:
                            # Low IoU = false positive
                            failures.append(
                                {
                                    "sample_id": sample_id,
                                    "prediction": pred_box,
                                    "ground_truth": None,
                                    "error_type": "false_positive",
                                    "iou": best_iou,
                                }
                            )

                    # Unmatched ground truths = false negatives
                    for gt_idx, gt_box in enumerate(gt):
                        if gt_idx not in matched_gt_indices:
                            failures.append(
                                {
                                    "sample_id": sample_id,
                                    "prediction": None,
                                    "ground_truth": gt_box,
                                    "error_type": "false_negative",
                                    "iou": 0.0,
                                }
                            )

        return failures

    def categorize_errors(
        self, failures: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize failures by error type.

        Args:
            failures: List of failure records from find_failures()

        Returns:
            Dictionary grouping failures by error type:
            {
                'false_positive': [...],
                'false_negative': [...],
                'misclassification': [...]
            }
        """
        categories: Dict[str, List] = {
            "false_positive": [],
            "false_negative": [],
            "misclassification": [],
        }

        for failure in failures:
            error_type = failure["error_type"]
            if error_type in categories:
                categories[error_type].append(failure)

        return categories

    def visualize_comparison(
        self,
        sample_id: str,
        prediction: Any,
        ground_truth: Any,
        output_path: Path,
        image_path: Optional[Path] = None,
    ) -> None:
        """Visualize prediction vs ground truth comparison.

        Args:
            sample_id: Sample identifier
            prediction: Predicted output (bbox, mask, etc.)
            ground_truth: Ground truth annotation
            output_path: Where to save visualization
            image_path: Optional path to original image
        """
        # Load image if provided, otherwise create blank canvas
        if image_path and image_path.exists():
            img = Image.open(image_path).convert("RGB")
        else:
            # Create blank canvas
            img = Image.new("RGB", (640, 480), color=(255, 255, 255))

        draw = ImageDraw.Draw(img)

        # Draw ground truth in green
        if ground_truth and isinstance(ground_truth, dict):
            if "bbox" in ground_truth:
                bbox = ground_truth["bbox"]
                # Convert [x, y, w, h] to [x1, y1, x2, y2]
                x1, y1 = bbox[0], bbox[1]
                x2, y2 = bbox[0] + bbox[2], bbox[1] + bbox[3]
                draw.rectangle([x1, y1, x2, y2], outline="green", width=3)

                # Add label
                label = ground_truth.get("category_name", "GT")
                draw.text((x1, y1 - 15), f"GT: {label}", fill="green")

        # Draw prediction in red
        if prediction and isinstance(prediction, dict):
            if "bbox" in prediction:
                bbox = prediction["bbox"]
                x1, y1 = bbox[0], bbox[1]
                x2, y2 = bbox[0] + bbox[2], bbox[1] + bbox[3]
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

                # Add label
                label = prediction.get("class", "PRED")
                score = prediction.get("score", 0.0)
                draw.text((x1, y1 + 5), f"Pred: {label} ({score:.2f})", fill="red")

        # Save visualization
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)

    def visualize_comparison_pointcloud(
        self,
        sample_id: str,
        prediction_path: Path,
        ground_truth_path: Path,
        output_path: Path,
    ) -> None:
        """Visualize point cloud prediction vs ground truth.

        Args:
            sample_id: Sample identifier
            prediction_path: Path to predicted point cloud
            ground_truth_path: Path to ground truth point cloud
            output_path: Where to save visualization (PNG screenshot)
        """
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError(
                "Open3D is required for point cloud visualization. "
                "Install with: pip install open3d"
            )

        # Load point clouds
        pred_pcd = o3d.io.read_point_cloud(str(prediction_path))
        gt_pcd = o3d.io.read_point_cloud(str(ground_truth_path))

        # Color prediction in red, ground truth in green
        pred_pcd.paint_uniform_color([1, 0, 0])  # Red
        gt_pcd.paint_uniform_color([0, 1, 0])  # Green

        # Create visualization
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(pred_pcd)
        vis.add_geometry(gt_pcd)

        # Render and save
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(str(output_path))
        vis.destroy_window()

    def generate_error_report(
        self, failures: List[Dict[str, Any]], output_path: Path
    ) -> None:
        """Generate HTML error analysis report.

        Args:
            failures: List of failure records
            output_path: Where to save HTML report
        """
        categories = self.categorize_errors(failures)

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Error Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .summary {{ background: #f0f0f0; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .category {{ margin: 20px 0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .error-type {{ font-weight: bold; }}
        .false-positive {{ color: #d32f2f; }}
        .false-negative {{ color: #f57c00; }}
        .misclassification {{ color: #1976d2; }}
    </style>
</head>
<body>
    <h1>Error Analysis Report</h1>

    <div class="summary">
        <h2>Summary</h2>
        <p>Total Failures: {len(failures)}</p>
        <p>False Positives: {len(categories['false_positive'])}</p>
        <p>False Negatives: {len(categories['false_negative'])}</p>
        <p>Misclassifications: {len(categories['misclassification'])}</p>
    </div>

    <div class="category">
        <h2>Detailed Failures</h2>
        <table>
            <tr>
                <th>Sample ID</th>
                <th>Error Type</th>
                <th>IoU</th>
                <th>Prediction</th>
                <th>Ground Truth</th>
            </tr>
"""

        for failure in failures:
            error_type = failure["error_type"]
            html += f"""
            <tr>
                <td>{failure['sample_id']}</td>
                <td class="error-type {error_type}">{error_type}</td>
                <td>{failure['iou']:.3f}</td>
                <td>{str(failure['prediction'])[:100]}</td>
                <td>{str(failure['ground_truth'])[:100]}</td>
            </tr>
"""

        html += """
        </table>
    </div>
</body>
</html>
"""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(html)
