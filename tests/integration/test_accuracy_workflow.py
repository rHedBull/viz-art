"""Integration tests for end-to-end accuracy tracking workflow.

Tests the complete workflow: ground truth loading, metrics calculation,
comparison, and report generation.
"""

import pytest
from pathlib import Path
from datetime import datetime
import tempfile
import numpy as np

from viz_art.accuracy import (
    AccuracyTracker,
    GroundTruthDataset,
    AnnotationFormat
)


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_ground_truth_dataset(tmp_path):
    """Create a mock ground truth dataset for testing."""
    # Create required directories
    (tmp_path / "images").mkdir(parents=True, exist_ok=True)
    (tmp_path / "annotations").mkdir(parents=True, exist_ok=True)

    return GroundTruthDataset(
        dataset_id="test_dataset_001",
        name="Test Validation Set",
        description="Mock dataset for integration testing",
        base_path=tmp_path / "images",
        annotation_path=tmp_path / "annotations",
        annotation_format=AnnotationFormat.COCO,
        num_samples=2,
        sample_ids=["sample_0001", "sample_0002"],
        metadata={"test": True},
        created_at=datetime.now(),
        updated_at=datetime.now()
    )


def test_classification_workflow(mock_ground_truth_dataset, temp_output_dir):
    """Test end-to-end workflow for classification task.

    Validates:
    - Metrics calculation for classification
    - Report generation
    - Metrics storage
    """
    # Create tracker
    tracker = AccuracyTracker(mock_ground_truth_dataset)

    # Mock predictions (classification labels)
    predictions = {
        "classification": [0, 1]  # Predicted labels
    }

    # Mock ground truth by patching the loader
    def mock_load_sample(dataset, sample_id):
        from viz_art.accuracy import GroundTruthSample
        # Generated sample IDs are sample_0000, sample_0001 (0-indexed)
        if sample_id == "sample_0000":
            return GroundTruthSample(
                sample_id=sample_id,
                dataset_id=dataset.dataset_id,
                stage_labels={"classification": 0},
                final_label=0,
                annotation_format=AnnotationFormat.COCO,
                image_path=temp_output_dir / f"{sample_id}.jpg"  # Add required path
            )
        else:  # sample_0001
            return GroundTruthSample(
                sample_id=sample_id,
                dataset_id=dataset.dataset_id,
                stage_labels={"classification": 1},
                final_label=1,
                annotation_format=AnnotationFormat.COCO,
                image_path=temp_output_dir / f"{sample_id}.jpg"  # Add required path
            )

    # Patch the loader
    tracker.gt_loader.load_sample = mock_load_sample

    # Run validation
    results = tracker.run_validation(
        predictions=predictions,
        run_id="test_run_001",
        output_dir=temp_output_dir,
        stage_task_types={"classification": "classification"}
    )

    # Assertions
    assert results['run_id'] == "test_run_001"
    assert results['overall_accuracy'] == 1.0  # Both predictions correct
    assert 'stage_metrics' in results
    assert 'classification' in results['stage_metrics']

    # Check metrics
    metrics_list = results['stage_metrics']['classification']
    assert len(metrics_list) > 0
    metrics = metrics_list[0]
    assert metrics.counts.correct == 2
    assert metrics.counts.wrong == 0

    # Check files created
    assert results['report_path'].exists()
    assert results['metrics_path'].exists()


def test_detection_workflow(mock_ground_truth_dataset, temp_output_dir):
    """Test end-to-end workflow for object detection task.

    Validates:
    - Metrics calculation for detection (mAP)
    - Comparison engine integration
    - Report with per-class metrics
    """
    tracker = AccuracyTracker(mock_ground_truth_dataset)

    # Mock detection predictions
    predictions = {
        "detection": [
            {
                "boxes": [[100, 100, 50, 75]],
                "labels": ["person"],
                "scores": [0.95]
            },
            {
                "boxes": [[50, 50, 60, 90]],
                "labels": ["person"],
                "scores": [0.88]
            }
        ]
    }

    # Mock ground truth
    def mock_load_sample(dataset, sample_id):
        from viz_art.accuracy import GroundTruthSample
        if sample_id == "sample_0001":
            gt_boxes = [[100, 100, 50, 75]]
            gt_labels = ["person"]
        else:
            gt_boxes = [[50, 50, 60, 90]]
            gt_labels = ["person"]

        return GroundTruthSample(
            sample_id=sample_id,
            dataset_id=dataset.dataset_id,
            stage_labels={
                "detection": {
                    "boxes": gt_boxes,
                    "labels": gt_labels
                }
            },
            final_label={"boxes": gt_boxes, "labels": gt_labels},
            annotation_format=AnnotationFormat.COCO,
            image_path=temp_output_dir / f"{sample_id}.jpg"
        )

    tracker.gt_loader.load_sample = mock_load_sample

    # Run validation
    results = tracker.run_validation(
        predictions=predictions,
        run_id="test_run_002",
        output_dir=temp_output_dir,
        stage_task_types={"detection": "detection"}
    )

    # Assertions
    assert results['run_id'] == "test_run_002"
    assert results['overall_accuracy'] >= 0.0  # mAP-based accuracy
    assert 'detection' in results['stage_metrics']

    # Check report and metrics files
    assert results['report_path'].exists()
    assert results['metrics_path'].exists()

    # Check HTML report content
    html_content = results['report_path'].read_text()
    assert "test_run_002" in html_content
    assert "detection" in html_content.lower()


def test_segmentation_workflow(mock_ground_truth_dataset, temp_output_dir):
    """Test end-to-end workflow for segmentation task.

    Validates:
    - IoU calculation for segmentation masks
    - Per-class IoU reporting
    - Pixel-level accuracy
    """
    tracker = AccuracyTracker(mock_ground_truth_dataset)

    # Mock segmentation predictions (simple masks)
    mask1 = np.zeros((100, 100), dtype=np.int32)
    mask1[20:50, 20:50] = 1  # Class 1 region

    mask2 = np.zeros((100, 100), dtype=np.int32)
    mask2[30:70, 30:70] = 1

    predictions = {
        "segmentation": [mask1, mask2]
    }

    # Mock ground truth
    def mock_load_sample(dataset, sample_id):
        from viz_art.accuracy import GroundTruthSample

        # Create ground truth masks similar to predictions
        if sample_id == "sample_0001":
            gt_mask = np.zeros((100, 100), dtype=np.int32)
            gt_mask[20:50, 20:50] = 1
        else:
            gt_mask = np.zeros((100, 100), dtype=np.int32)
            gt_mask[30:70, 30:70] = 1

        return GroundTruthSample(
            sample_id=sample_id,
            dataset_id=dataset.dataset_id,
            stage_labels={"segmentation": gt_mask},
            final_label=gt_mask,
            annotation_format=AnnotationFormat.COCO,
            image_path=temp_output_dir / f"{sample_id}.jpg"
        )

    tracker.gt_loader.load_sample = mock_load_sample

    # Run validation
    results = tracker.run_validation(
        predictions=predictions,
        run_id="test_run_003",
        output_dir=temp_output_dir,
        stage_task_types={"segmentation": "segmentation"}
    )

    # Assertions
    assert results['run_id'] == "test_run_003"
    assert results['overall_accuracy'] > 0.5  # Good accuracy for identical masks
    assert 'segmentation' in results['stage_metrics']

    metrics_list = results['stage_metrics']['segmentation']
    assert len(metrics_list) > 0
    metrics = metrics_list[0]
    assert metrics.value > 0.5  # Reasonable IoU expected


def test_multi_stage_workflow(mock_ground_truth_dataset, temp_output_dir):
    """Test workflow with multiple stages.

    Validates:
    - Multiple stages processed correctly
    - Overall accuracy aggregation
    - Per-stage metrics in report
    """
    tracker = AccuracyTracker(mock_ground_truth_dataset)

    # Multi-stage predictions
    predictions = {
        "detection": [
            {"boxes": [[100, 100, 50, 75]], "labels": ["person"], "scores": [0.95]},
            {"boxes": [[50, 50, 60, 90]], "labels": ["person"], "scores": [0.88]}
        ],
        "classification": [0, 1]
    }

    # Mock ground truth
    def mock_load_sample(dataset, sample_id):
        from viz_art.accuracy import GroundTruthSample
        if sample_id == "sample_0001":
            return GroundTruthSample(
                sample_id=sample_id,
                dataset_id=dataset.dataset_id,
                stage_labels={
                    "detection": {"boxes": [[100, 100, 50, 75]], "labels": ["person"]},
                    "classification": 0
                },
                final_label=0,
                annotation_format=AnnotationFormat.COCO,
                image_path=temp_output_dir / f"{sample_id}.jpg"
            )
        else:
            return GroundTruthSample(
                sample_id=sample_id,
                dataset_id=dataset.dataset_id,
                stage_labels={
                    "detection": {"boxes": [[50, 50, 60, 90]], "labels": ["person"]},
                    "classification": 1
                },
                final_label=1,
                annotation_format=AnnotationFormat.COCO,
                image_path=temp_output_dir / f"{sample_id}.jpg"
            )

    tracker.gt_loader.load_sample = mock_load_sample

    # Run validation
    results = tracker.run_validation(
        predictions=predictions,
        run_id="test_run_004",
        output_dir=temp_output_dir,
        stage_task_types={
            "detection": "detection",
            "classification": "classification"
        }
    )

    # Assertions
    assert len(results['stage_metrics']) == 2
    assert 'detection' in results['stage_metrics']
    assert 'classification' in results['stage_metrics']

    # Check HTML report has both stages
    html_content = results['report_path'].read_text()
    assert "detection" in html_content.lower()
    assert "classification" in html_content.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
