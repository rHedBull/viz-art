"""Unit tests for error visualizations (T062)."""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from viz_art.error_analysis.visualizer import ErrorVisualizer


class TestErrorVisualizer:
    """Test error visualization functions."""

    def test_create_image_diff(self):
        """Test image diff visualization with bounding boxes."""
        visualizer = ErrorVisualizer()

        # Create mock images
        pred_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        gt_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        pred_boxes = [[10, 10, 20, 20]]
        gt_boxes = [[15, 15, 20, 20]]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "diff.jpg"

            result = visualizer.create_image_diff(
                pred_img, gt_img, pred_boxes, gt_boxes, output_path
            )

            # Check output exists
            assert output_path.exists()

            # Check result is correct shape (side-by-side)
            assert result.shape[0] == pred_img.shape[0]
            assert result.shape[1] == pred_img.shape[1] * 2
            assert result.shape[2] == 3

    def test_create_image_diff_no_boxes(self):
        """Test image diff without bounding boxes."""
        visualizer = ErrorVisualizer()

        pred_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        gt_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        result = visualizer.create_image_diff(pred_img, gt_img)

        assert result.shape[0] == pred_img.shape[0]
        assert result.shape[1] == pred_img.shape[1] * 2

    def test_apply_colormap(self):
        """Test colormap application."""
        visualizer = ErrorVisualizer()

        values = np.linspace(0, 1, 100)
        colors = visualizer._apply_colormap(values, colormap='viridis')

        # Check output shape
        assert colors.shape == (100, 3)

        # Check values are in valid range
        assert np.all(colors >= 0)
        assert np.all(colors <= 1)

    def test_mask_to_color(self):
        """Test segmentation mask to color conversion."""
        visualizer = ErrorVisualizer()

        # Create mock segmentation mask
        mask = np.random.randint(0, 5, (100, 100), dtype=np.uint8)

        colored = visualizer._mask_to_color(mask)

        # Check output shape
        assert colored.shape == (100, 100, 3)

        # Background should be black
        background_pixels = colored[mask == 0]
        if len(background_pixels) > 0:
            assert np.all(background_pixels == [0, 0, 0])

    def test_create_mask_visualization(self):
        """Test mask visualization with side-by-side view."""
        visualizer = ErrorVisualizer()

        pred_mask = np.random.randint(0, 3, (100, 100), dtype=np.uint8)
        gt_mask = np.random.randint(0, 3, (100, 100), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "mask_viz.jpg"

            visualizer._create_mask_visualization(pred_mask, gt_mask, output_path)

            assert output_path.exists()

    def test_create_text_visualization(self):
        """Test text-based fallback visualization."""
        from viz_art.error_analysis.patterns import ErrorCase, ErrorType, ErrorSeverity
        from datetime import datetime

        visualizer = ErrorVisualizer()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create error case
            pred_path = tmppath / "pred.json"
            gt_path = tmppath / "gt.json"
            pred_path.write_text("{}")
            gt_path.write_text("{}")

            error_case = ErrorCase(
                error_id="test_error",
                run_id="test_run",
                stage_name="detection",
                sample_id="sample_001",
                error_type=ErrorType.FALSE_POSITIVE,
                severity=ErrorSeverity.HIGH,
                prediction={'boxes': [[10, 10, 20, 20]]},
                ground_truth={'boxes': []},
                iou=0.0,
                confidence=0.85,
                saved_artifacts={
                    "prediction_path": pred_path,
                    "ground_truth_path": gt_path
                },
                timestamp=datetime.now()
            )

            output_path = tmppath / "text_viz.jpg"
            visualizer._create_text_visualization(error_case, output_path)

            assert output_path.exists()

    def test_create_diff_visualization_auto(self):
        """Test auto-detection of visualization type."""
        from viz_art.error_analysis.patterns import ErrorCase, ErrorType, ErrorSeverity
        from datetime import datetime

        visualizer = ErrorVisualizer()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            pred_path = tmppath / "pred.json"
            gt_path = tmppath / "gt.json"
            pred_path.write_text("{}")
            gt_path.write_text("{}")

            error_case = ErrorCase(
                error_id="test_error",
                run_id="test_run",
                stage_name="detection",
                sample_id="sample_001",
                error_type=ErrorType.FALSE_POSITIVE,
                severity=ErrorSeverity.HIGH,
                prediction={'boxes': [[10, 10, 20, 20]]},
                ground_truth={'boxes': []},
                saved_artifacts={
                    "prediction_path": pred_path,
                    "ground_truth_path": gt_path
                },
                timestamp=datetime.now()
            )

            output_path = tmppath / "auto_viz.jpg"
            result_path = visualizer.create_diff_visualization(
                error_case, diff_type="auto", output_path=output_path
            )

            assert result_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
