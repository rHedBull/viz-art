"""Integration test for multi-run trend tracking (T084)."""

import pytest
from pathlib import Path
from datetime import datetime
import tempfile
import time

from viz_art.accuracy import AccuracyTracker, GroundTruthDataset, AnnotationFormat
from viz_art.performance.trends import PerformanceTracker, RegressionDetector


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_ground_truth_dataset(tmp_path):
    """Create a mock ground truth dataset."""
    # Create required directories
    (tmp_path / "images").mkdir(parents=True, exist_ok=True)
    (tmp_path / "annotations").mkdir(parents=True, exist_ok=True)

    return GroundTruthDataset(
        dataset_id="trend_test_dataset",
        name="Trend Test Dataset",
        description="Dataset for trend tracking tests",
        base_path=tmp_path / "images",
        annotation_path=tmp_path / "annotations",
        annotation_format=AnnotationFormat.COCO,
        num_samples=2,
        sample_ids=["sample_0001", "sample_0002"],
        metadata={"test": True},
        created_at=datetime.now(),
        updated_at=datetime.now()
    )


def test_multi_run_trend_tracking(mock_ground_truth_dataset, temp_output_dir):
    """Test tracking trends across multiple validation runs.

    This test validates:
    - Multiple runs are tracked correctly
    - Trends can be queried
    - Regression detection works across runs
    """
    # Create tracker
    tracker = AccuracyTracker(mock_ground_truth_dataset)

    # Override performance tracker to use temp dir
    tracker.performance_tracker = PerformanceTracker(temp_output_dir / "trends")

    # Mock ground truth loader
    def mock_load_sample(dataset, sample_id):
        from viz_art.accuracy import GroundTruthSample
        return GroundTruthSample(
            sample_id=sample_id,
            dataset_id=dataset.dataset_id,
            stage_labels={"classification": 0},
            final_label=0,
            annotation_format=AnnotationFormat.COCO,
            image_path=temp_output_dir / f"{sample_id}.jpg"
        )

    tracker.gt_loader.load_sample = mock_load_sample

    # Run 1: Good accuracy
    predictions_run1 = {
        "classification": [0, 0]  # Both correct
    }

    results_run1 = tracker.run_validation(
        predictions=predictions_run1,
        run_id="trend_run_001",
        output_dir=temp_output_dir,
        stage_task_types={"classification": "classification"}
    )

    assert results_run1['overall_accuracy'] == 1.0

    time.sleep(0.1)  # Ensure different timestamps

    # Run 2: Similar accuracy
    predictions_run2 = {
        "classification": [0, 0]
    }

    results_run2 = tracker.run_validation(
        predictions=predictions_run2,
        run_id="trend_run_002",
        output_dir=temp_output_dir,
        stage_task_types={"classification": "classification"}
    )

    time.sleep(0.1)

    # Run 3: Degraded accuracy (regression)
    predictions_run3 = {
        "classification": [1, 1]  # Both wrong
    }

    results_run3 = tracker.run_validation(
        predictions=predictions_run3,
        run_id="trend_run_003",
        output_dir=temp_output_dir,
        stage_task_types={"classification": "classification"}
    )

    assert results_run3['overall_accuracy'] == 0.0

    # Query trends
    trend = tracker.performance_tracker.get_trend(
        dataset_id="trend_test_dataset",
        stage_name="classification",
        metric_type="f1_score"
    )

    assert len(trend) >= 3  # Should have at least 3 runs

    # Test regression detection
    detector = RegressionDetector(threshold=0.05)

    regression = detector.detect_regression(
        current_metrics=results_run3['stage_metrics'],
        baseline_metrics=results_run1['stage_metrics']
    )

    assert regression['regression_detected'] is True
    assert "classification" in regression['affected_stages']

    # Create alert
    alert = detector.create_alert(
        regression,
        run_id="trend_run_003",
        baseline_run_id="trend_run_001"
    )

    assert alert['alert'] is True
    assert alert['severity'] in ['high', 'medium']


def test_trend_data_persistence(mock_ground_truth_dataset, temp_output_dir):
    """Test that trend data persists across tracker instances."""
    # Create first tracker and record metrics
    tracker1 = AccuracyTracker(mock_ground_truth_dataset)
    tracker1.performance_tracker = PerformanceTracker(temp_output_dir / "trends")

    def mock_load_sample(dataset, sample_id):
        from viz_art.accuracy import GroundTruthSample
        return GroundTruthSample(
            sample_id=sample_id,
            dataset_id=dataset.dataset_id,
            stage_labels={"classification": 0},
            final_label=0,
            annotation_format=AnnotationFormat.COCO,
            image_path=temp_output_dir / f"{sample_id}.jpg"
        )

    tracker1.gt_loader.load_sample = mock_load_sample

    predictions = {
        "classification": [0, 0]
    }

    tracker1.run_validation(
        predictions=predictions,
        run_id="persist_run_001",
        output_dir=temp_output_dir,
        stage_task_types={"classification": "classification"}
    )

    # Create second tracker and verify data is accessible
    tracker2 = AccuracyTracker(mock_ground_truth_dataset)
    tracker2.performance_tracker = PerformanceTracker(temp_output_dir / "trends")

    # Query trend data
    all_runs = tracker2.performance_tracker.get_all_runs("trend_test_dataset")

    assert "persist_run_001" in all_runs


def test_baseline_strategy_selection(mock_ground_truth_dataset, temp_output_dir):
    """Test different baseline selection strategies."""
    tracker = AccuracyTracker(mock_ground_truth_dataset)
    tracker.performance_tracker = PerformanceTracker(temp_output_dir / "trends")

    def mock_load_sample(dataset, sample_id):
        from viz_art.accuracy import GroundTruthSample
        return GroundTruthSample(
            sample_id=sample_id,
            dataset_id=dataset.dataset_id,
            stage_labels={"classification": 0},
            final_label=0,
            annotation_format=AnnotationFormat.COCO,
            image_path=temp_output_dir / f"{sample_id}.jpg"
        )

    tracker.gt_loader.load_sample = mock_load_sample

    # Run with different accuracies
    test_cases = [
        ([0, 0], "run_high"),    # 100% accuracy
        ([0, 1], "run_medium"),  # 50% accuracy
        ([1, 1], "run_low")      # 0% accuracy
    ]

    for predictions, run_id in test_cases:
        tracker.run_validation(
            predictions={"classification": predictions},
            run_id=run_id,
            output_dir=temp_output_dir,
            stage_task_types={"classification": "classification"}
        )
        time.sleep(0.1)

    # Test different strategies
    detector = RegressionDetector()

    # Best strategy should pick run_high
    best_baseline = detector.get_baseline_strategy(
        "trend_test_dataset",
        tracker.performance_tracker,
        strategy="best"
    )
    assert best_baseline == "run_high"

    # Latest strategy should pick run_low (last run)
    latest_baseline = detector.get_baseline_strategy(
        "trend_test_dataset",
        tracker.performance_tracker,
        strategy="latest"
    )
    assert latest_baseline == "run_low"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
