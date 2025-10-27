"""Unit tests for error detection and categorization (T061)."""

import pytest
import numpy as np
from pathlib import Path
from datetime import datetime
import tempfile

from viz_art.error_analysis.patterns import (
    ErrorDetector,
    ErrorPatternDetector,
    ErrorCase,
    ErrorType,
    ErrorSeverity
)


class TestErrorDetector:
    """Test error detection and categorization."""

    def test_categorize_error_false_positive(self):
        """Test categorization of false positive errors."""
        detector = ErrorDetector()

        prediction = {'boxes': [[100, 100, 50, 50]], 'labels': ['person']}
        ground_truth = {'boxes': [], 'labels': []}  # No objects

        error_type, severity = detector.categorize_error(prediction, ground_truth, iou=0.0)

        assert error_type == ErrorType.FALSE_POSITIVE
        assert severity == ErrorSeverity.HIGH  # IoU < 0.3

    def test_categorize_error_false_negative(self):
        """Test categorization of false negative errors."""
        detector = ErrorDetector()

        prediction = {'boxes': [], 'labels': []}
        ground_truth = {'boxes': [[100, 100, 50, 50]], 'labels': ['person']}

        error_type, severity = detector.categorize_error(prediction, ground_truth, iou=0.0)

        assert error_type == ErrorType.FALSE_NEGATIVE
        assert severity == ErrorSeverity.HIGH

    def test_categorize_error_low_iou(self):
        """Test categorization of low IoU errors."""
        detector = ErrorDetector()

        prediction = {'boxes': [[100, 100, 50, 50]], 'labels': ['person']}
        ground_truth = {'boxes': [[110, 110, 50, 50]], 'labels': ['person']}

        error_type, severity = detector.categorize_error(prediction, ground_truth, iou=0.4)

        assert error_type == ErrorType.LOW_IOU
        assert severity == ErrorSeverity.MEDIUM  # 0.3 <= IoU < 0.5

    def test_categorize_error_misclassification(self):
        """Test categorization of misclassification errors."""
        detector = ErrorDetector()

        prediction = 1
        ground_truth = 2

        error_type, severity = detector.categorize_error(prediction, ground_truth)

        assert error_type == ErrorType.MISCLASSIFICATION
        assert severity == ErrorSeverity.MEDIUM

    def test_detect_errors_from_comparison_results(self):
        """Test error detection from comparison results."""
        detector = ErrorDetector()

        comparison_results = [
            {'is_correct': False, 'error_type': 'false_positive', 'iou': 0.0},
            {'is_correct': True, 'error_type': None},
            {'is_correct': False, 'error_type': 'misclassification', 'iou': 0.6}
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            errors = detector.detect_errors(
                comparison_results=comparison_results,
                run_id="test_run",
                stage_name="detection",
                sample_ids=["s1", "s2", "s3"],
                predictions=[{'boxes': [[1, 1, 1, 1]]}, {}, {}],
                ground_truths=[{}, {}, {}],
                output_dir=Path(tmpdir)
            )

        # Should detect 2 errors (skipping the correct one)
        assert len(errors) == 2
        assert errors[0].sample_id == "s1"
        assert errors[1].sample_id == "s3"


class TestErrorPatternDetector:
    """Test error pattern detection and clustering."""

    def create_mock_error(self, error_id, stage, error_type, severity, iou=0.5, confidence=0.7):
        """Helper to create mock ErrorCase."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            pred_path = tmppath / "pred.json"
            gt_path = tmppath / "gt.json"
            pred_path.write_text("{}")
            gt_path.write_text("{}")

            return ErrorCase(
                error_id=error_id,
                run_id="test_run",
                stage_name=stage,
                sample_id=f"sample_{error_id}",
                error_type=error_type,
                severity=severity,
                prediction={},
                ground_truth={},
                iou=iou,
                confidence=confidence,
                saved_artifacts={
                    "prediction_path": pred_path,
                    "ground_truth_path": gt_path
                },
                timestamp=datetime.now()
            )

    def test_cluster_errors_by_stage_and_type(self):
        """Test rule-based clustering by stage and error type."""
        detector = ErrorPatternDetector()

        # Create mock errors with different stages and types
        # Note: We need to keep references to prevent tmpdir cleanup
        errors_data = [
            ("e1", "detection", ErrorType.FALSE_POSITIVE, ErrorSeverity.HIGH),
            ("e2", "detection", ErrorType.FALSE_POSITIVE, ErrorSeverity.HIGH),
            ("e3", "detection", ErrorType.FALSE_NEGATIVE, ErrorSeverity.HIGH),
            ("e4", "classification", ErrorType.MISCLASSIFICATION, ErrorSeverity.MEDIUM),
        ]

        # Create persistent temp directories for each error
        tmpdirs = []
        errors = []
        for error_id, stage, error_type, severity in errors_data:
            tmpdir = tempfile.mkdtemp()
            tmpdirs.append(tmpdir)
            tmppath = Path(tmpdir)
            pred_path = tmppath / "pred.json"
            gt_path = tmppath / "gt.json"
            pred_path.write_text("{}")
            gt_path.write_text("{}")

            error = ErrorCase(
                error_id=error_id,
                run_id="test_run",
                stage_name=stage,
                sample_id=f"sample_{error_id}",
                error_type=error_type,
                severity=severity,
                prediction={},
                ground_truth={},
                iou=0.5,
                confidence=0.7,
                saved_artifacts={
                    "prediction_path": pred_path,
                    "ground_truth_path": gt_path
                },
                timestamp=datetime.now()
            )
            errors.append(error)

        try:
            clusters = detector.cluster_errors(errors)

            # Should create 3 clusters
            assert len(clusters) == 3
            assert "detection_false_positive" in clusters
            assert "detection_false_negative" in clusters
            assert "classification_misclassification" in clusters

            # Check cluster sizes
            assert len(clusters["detection_false_positive"]) == 2
            assert len(clusters["detection_false_negative"]) == 1
            assert len(clusters["classification_misclassification"]) == 1
        finally:
            # Cleanup temp directories
            import shutil
            for tmpdir in tmpdirs:
                shutil.rmtree(tmpdir, ignore_errors=True)

    def test_summarize_patterns(self):
        """Test pattern summarization."""
        detector = ErrorPatternDetector()

        # Create clusters
        tmpdir = tempfile.mkdtemp()
        try:
            tmppath = Path(tmpdir)

            # Create mock errors in cluster
            errors = []
            for i in range(3):
                pred_path = tmppath / f"pred_{i}.json"
                gt_path = tmppath / f"gt_{i}.json"
                pred_path.write_text("{}")
                gt_path.write_text("{}")

                error = ErrorCase(
                    error_id=f"e{i}",
                    run_id="test_run",
                    stage_name="detection",
                    sample_id=f"sample_{i}",
                    error_type=ErrorType.FALSE_POSITIVE,
                    severity=ErrorSeverity.HIGH,
                    prediction={},
                    ground_truth={},
                    iou=0.2 + i * 0.1,  # Varying IoU
                    confidence=0.6 + i * 0.1,  # Varying confidence
                    saved_artifacts={
                        "prediction_path": pred_path,
                        "ground_truth_path": gt_path
                    },
                    timestamp=datetime.now()
                )
                errors.append(error)

            clusters = {"detection_false_positive": errors}
            patterns = detector.summarize_patterns(clusters)

            assert len(patterns) == 1
            pattern = patterns[0]

            assert pattern.pattern_id == "detection_false_positive"
            assert pattern.error_count == 3
            assert pattern.severity == ErrorSeverity.HIGH
            assert len(pattern.affected_samples) == 3
            assert 0.0 <= pattern.statistics.avg_iou <= 1.0
            assert 0.0 <= pattern.statistics.avg_confidence <= 1.0
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_infer_cause(self):
        """Test cause inference for patterns."""
        detector = ErrorPatternDetector()

        tmpdir = tempfile.mkdtemp()
        try:
            tmppath = Path(tmpdir)
            pred_path = tmppath / "pred.json"
            gt_path = tmppath / "gt.json"
            pred_path.write_text("{}")
            gt_path.write_text("{}")

            # False positive with high confidence
            error = ErrorCase(
                error_id="e1",
                run_id="test_run",
                stage_name="detection",
                sample_id="sample_1",
                error_type=ErrorType.FALSE_POSITIVE,
                severity=ErrorSeverity.HIGH,
                prediction={},
                ground_truth={},
                iou=0.0,
                confidence=0.85,  # High confidence
                saved_artifacts={
                    "prediction_path": pred_path,
                    "ground_truth_path": gt_path
                },
                timestamp=datetime.now()
            )

            cause = detector._infer_cause([error])

            assert "false positive" in cause.lower() or "threshold" in cause.lower()
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
