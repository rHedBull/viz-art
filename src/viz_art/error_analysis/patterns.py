"""Error pattern detection and clustering.

This module provides data models and functions for detecting, categorizing,
and clustering error cases into patterns.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np


class ErrorType(Enum):
    """Types of pipeline failures."""

    FALSE_POSITIVE = "false_positive"  # Detected when not present
    FALSE_NEGATIVE = "false_negative"  # Missed detection
    MISCLASSIFICATION = "misclassification"  # Wrong class label
    LOW_IOU = "low_iou"  # Poor localization


class ErrorSeverity(Enum):
    """Impact level of errors."""

    CRITICAL = "critical"  # Pipeline fails entirely
    HIGH = "high"  # Wrong output (FP/FN with IoU < 0.3)
    MEDIUM = "medium"  # Degraded quality (misclass or 0.3 <= IoU < 0.5)
    LOW = "low"  # Minor deviation (IoU >= 0.5)


@dataclass
class ErrorCase:
    """Failed prediction instance with saved outputs and metadata.

    Attributes:
        error_id: Unique error identifier (UUID v4 format)
        run_id: Pipeline run that produced error
        stage_name: Stage where error occurred
        sample_id: Sample that failed
        error_type: Type of failure
        severity: Impact level
        prediction: Model prediction output
        ground_truth: Expected output
        iou: IoU score (if applicable, 0.0-1.0)
        confidence: Prediction confidence (0.0-1.0)
        saved_artifacts: Paths to saved outputs (must exist)
        timestamp: When error occurred
        metadata: Additional error context
    """

    error_id: str
    run_id: str
    stage_name: str
    sample_id: str
    error_type: ErrorType
    severity: ErrorSeverity
    prediction: Any
    ground_truth: Any
    saved_artifacts: Dict[str, Path]
    timestamp: datetime
    iou: Optional[float] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate error case attributes."""
        # Validate IoU range
        if self.iou is not None and not 0.0 <= self.iou <= 1.0:
            raise ValueError(f"IoU must be in [0, 1]: {self.iou}")

        # Validate confidence range
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1]: {self.confidence}")

        # Validate saved artifacts exist
        for artifact_name, artifact_path in self.saved_artifacts.items():
            if not artifact_path.exists():
                raise ValueError(
                    f"Saved artifact '{artifact_name}' does not exist: {artifact_path}"
                )

        # Validate required artifacts
        if "prediction_path" not in self.saved_artifacts:
            raise ValueError("saved_artifacts must include 'prediction_path'")
        if "ground_truth_path" not in self.saved_artifacts:
            raise ValueError("saved_artifacts must include 'ground_truth_path'")


@dataclass
class PatternStatistics:
    """Aggregate statistics for error pattern.

    Attributes:
        avg_iou: Average IoU across errors (if applicable)
        avg_confidence: Average prediction confidence
        min_iou: Worst IoU in pattern
        max_iou: Best IoU in pattern
        sample_preview: First 5 sample IDs for preview
    """

    avg_iou: float
    avg_confidence: float
    min_iou: float
    max_iou: float
    sample_preview: List[str]

    def __post_init__(self):
        """Validate statistics."""
        for value in [self.avg_iou, self.avg_confidence, self.min_iou, self.max_iou]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Value must be in [0, 1]: {value}")

        if len(self.sample_preview) > 5:
            raise ValueError("sample_preview must have at most 5 samples")


@dataclass
class ErrorPattern:
    """Clustered group of similar failures for analysis.

    Attributes:
        pattern_id: Unique pattern identifier (composite: {stage}_{error_type})
        run_id: Pipeline run
        stage_name: Stage with errors
        error_type: Type of errors in cluster
        severity: Aggregate severity (most severe in cluster)
        error_count: Number of errors in pattern
        affected_samples: Sample IDs with this pattern
        statistics: Aggregate stats
        suggested_cause: Inferred root cause (max 200 chars)
        timestamp: Pattern detection time
    """

    pattern_id: str
    run_id: str
    stage_name: str
    error_type: ErrorType
    severity: ErrorSeverity
    error_count: int
    affected_samples: List[str]
    statistics: PatternStatistics
    timestamp: datetime
    suggested_cause: str = ""

    def __post_init__(self):
        """Validate error pattern attributes."""
        # Validate error_count matches affected_samples
        if self.error_count != len(self.affected_samples):
            raise ValueError(
                f"error_count ({self.error_count}) != len(affected_samples) ({len(self.affected_samples)})"
            )

        # Validate pattern_id format
        expected_pattern_id = f"{self.stage_name}_{self.error_type.value}"
        if self.pattern_id != expected_pattern_id:
            raise ValueError(
                f"pattern_id must be '{expected_pattern_id}', got '{self.pattern_id}'"
            )

        # Validate suggested_cause length
        if len(self.suggested_cause) > 200:
            raise ValueError("suggested_cause must be <= 200 chars")

        # Validate error_count > 0
        if self.error_count <= 0:
            raise ValueError(f"error_count must be > 0: {self.error_count}")


class ErrorDetector:
    """Detect and categorize errors from comparison results."""

    def categorize_error(
        self, prediction: Any, ground_truth: Any, iou: Optional[float] = None
    ) -> tuple[ErrorType, ErrorSeverity]:
        """Categorize error type and severity (T041).

        Args:
            prediction: Model prediction
            ground_truth: Expected output
            iou: IoU score (optional, for detection/segmentation)

        Returns:
            Tuple of (ErrorType, ErrorSeverity)

        Implementation: Per spec assumptions in research.md:
        - Critical: Pipeline fails entirely
        - High: Wrong output (FP/FN with IoU < 0.3)
        - Medium: Degraded quality (misclass or 0.3 <= IoU < 0.5)
        - Low: Minor deviation (IoU >= 0.5)
        """
        import numpy as np

        # Determine error type
        if isinstance(prediction, dict) and ('boxes' in prediction or 'bbox' in prediction):
            # Detection task
            pred_boxes = prediction.get('boxes', prediction.get('bbox', []))
            gt_boxes = ground_truth.get('boxes', ground_truth.get('bbox', []))

            if len(pred_boxes) > len(gt_boxes):
                error_type = ErrorType.FALSE_POSITIVE
            elif len(pred_boxes) < len(gt_boxes):
                error_type = ErrorType.FALSE_NEGATIVE
            elif iou is not None and iou < 0.5:
                error_type = ErrorType.LOW_IOU
            else:
                error_type = ErrorType.MISCLASSIFICATION

        elif isinstance(prediction, (int, str)) or isinstance(ground_truth, (int, str)):
            # Classification task
            error_type = ErrorType.MISCLASSIFICATION

        elif isinstance(prediction, np.ndarray) and isinstance(ground_truth, np.ndarray):
            # Segmentation task
            if iou is not None and iou < 0.5:
                error_type = ErrorType.LOW_IOU
            else:
                error_type = ErrorType.MISCLASSIFICATION

        else:
            # Default to misclassification
            error_type = ErrorType.MISCLASSIFICATION

        # Determine severity based on error type and IoU
        if iou is not None:
            if error_type in [ErrorType.FALSE_POSITIVE, ErrorType.FALSE_NEGATIVE] and iou < 0.3:
                severity = ErrorSeverity.HIGH
            elif error_type == ErrorType.MISCLASSIFICATION or (0.3 <= iou < 0.5):
                severity = ErrorSeverity.MEDIUM
            elif iou >= 0.5:
                severity = ErrorSeverity.LOW
            else:
                severity = ErrorSeverity.HIGH
        else:
            # No IoU available - use error type
            if error_type in [ErrorType.FALSE_POSITIVE, ErrorType.FALSE_NEGATIVE]:
                severity = ErrorSeverity.HIGH
            else:
                severity = ErrorSeverity.MEDIUM

        return error_type, severity

    def detect_errors(
        self,
        comparison_results: List[Dict[str, Any]],
        run_id: str,
        stage_name: str,
        sample_ids: List[str],
        predictions: List[Any],
        ground_truths: List[Any],
        output_dir: Path
    ) -> List[ErrorCase]:
        """Detect errors from comparison results (T042).

        Args:
            comparison_results: List of comparison result dicts
            run_id: Pipeline run identifier
            stage_name: Stage name
            sample_ids: List of sample identifiers
            predictions: List of predictions
            ground_truths: List of ground truths
            output_dir: Directory to save error artifacts

        Returns:
            List of ErrorCase instances
        """
        import uuid

        errors = []

        for i, result in enumerate(comparison_results):
            # Skip correct predictions
            if result.get('is_correct', True):
                continue

            sample_id = sample_ids[i] if i < len(sample_ids) else f"sample_{i}"
            prediction = predictions[i] if i < len(predictions) else None
            ground_truth = ground_truths[i] if i < len(ground_truths) else None

            # Categorize error
            iou = result.get('iou', result.get('chamfer_distance', None))
            error_type, severity = self.categorize_error(prediction, ground_truth, iou)

            # Get confidence score
            confidence = None
            if isinstance(prediction, dict) and 'scores' in prediction:
                scores = prediction['scores']
                confidence = float(scores[0]) if len(scores) > 0 else None
            elif isinstance(prediction, dict) and 'score' in prediction:
                confidence = float(prediction['score'])

            # Create error ID
            error_id = str(uuid.uuid4())

            # Save artifacts (placeholder paths for now - actual saving in T059)
            artifacts_dir = output_dir / "errors" / run_id / error_id
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            # Create placeholder artifact files
            pred_path = artifacts_dir / "prediction.json"
            gt_path = artifacts_dir / "ground_truth.json"

            import json
            pred_path.write_text(json.dumps(str(prediction)))
            gt_path.write_text(json.dumps(str(ground_truth)))

            saved_artifacts = {
                "prediction_path": pred_path,
                "ground_truth_path": gt_path
            }

            # Create ErrorCase
            error_case = ErrorCase(
                error_id=error_id,
                run_id=run_id,
                stage_name=stage_name,
                sample_id=sample_id,
                error_type=ErrorType(result.get('error_type', 'misclassification')),
                severity=severity,
                prediction=prediction,
                ground_truth=ground_truth,
                iou=iou if iou is not None else None,
                confidence=confidence,
                saved_artifacts=saved_artifacts,
                timestamp=datetime.now(),
                metadata=result
            )

            errors.append(error_case)

        return errors


class ErrorPatternDetector:
    """Cluster errors into patterns for analysis."""

    def __init__(self, grouping_rules: Optional[List[str]] = None):
        """Initialize pattern detector.

        Args:
            grouping_rules: Rules for grouping (default: ["stage_name", "error_type"])
        """
        self.grouping_rules = grouping_rules or ["stage_name", "error_type"]

    def cluster_errors(
        self, failures: List[ErrorCase], grouping_rules: List[str] = None
    ) -> Dict[str, List[ErrorCase]]:
        """Cluster errors using rule-based grouping (T044).

        Args:
            failures: List of error cases
            grouping_rules: Rules for grouping (default: ["stage_name", "error_type"])

        Returns:
            Dict mapping pattern_id to list of error cases

        Implementation: Rule-based clustering with composite keys
        Complexity: O(n), <15ms for 1000 errors
        """
        from collections import defaultdict

        if grouping_rules is None:
            grouping_rules = self.grouping_rules

        clusters = defaultdict(list)

        for failure in failures:
            # Build composite key based on grouping rules
            key_parts = []

            for rule in grouping_rules:
                if rule == "stage_name":
                    key_parts.append(failure.stage_name)
                elif rule == "error_type":
                    key_parts.append(failure.error_type.value)
                elif rule == "severity":
                    key_parts.append(failure.severity.value)
                else:
                    # Skip unknown rules
                    pass

            # Create pattern_id
            cluster_key = "_".join(key_parts)
            clusters[cluster_key].append(failure)

        return dict(clusters)

    def summarize_patterns(
        self, clusters: Dict[str, List[ErrorCase]]
    ) -> List[ErrorPattern]:
        """Create pattern summaries from clusters (T045).

        Args:
            clusters: Clustered error cases

        Returns:
            List of ErrorPattern instances
        """
        import numpy as np

        patterns = []

        for pattern_id, error_cases in clusters.items():
            if len(error_cases) == 0:
                continue

            # Extract statistics
            ious = [e.iou for e in error_cases if e.iou is not None]
            confidences = [e.confidence for e in error_cases if e.confidence is not None]

            avg_iou = float(np.mean(ious)) if ious else 0.0
            avg_confidence = float(np.mean(confidences)) if confidences else 0.0
            min_iou = float(np.min(ious)) if ious else 0.0
            max_iou = float(np.max(ious)) if ious else 0.0

            # Sample preview (first 5)
            sample_preview = [e.sample_id for e in error_cases[:5]]

            statistics = PatternStatistics(
                avg_iou=avg_iou,
                avg_confidence=avg_confidence,
                min_iou=min_iou,
                max_iou=max_iou,
                sample_preview=sample_preview
            )

            # Determine aggregate severity (most severe in cluster)
            severities = [e.severity for e in error_cases]
            severity_order = {
                ErrorSeverity.CRITICAL: 4,
                ErrorSeverity.HIGH: 3,
                ErrorSeverity.MEDIUM: 2,
                ErrorSeverity.LOW: 1
            }
            aggregate_severity = max(severities, key=lambda s: severity_order[s])

            # Infer suggested cause
            suggested_cause = self._infer_cause(error_cases)

            # Create pattern
            pattern = ErrorPattern(
                pattern_id=pattern_id,
                run_id=error_cases[0].run_id,
                stage_name=error_cases[0].stage_name,
                error_type=error_cases[0].error_type,
                severity=aggregate_severity,
                error_count=len(error_cases),
                affected_samples=[e.sample_id for e in error_cases],
                statistics=statistics,
                suggested_cause=suggested_cause,
                timestamp=datetime.now()
            )

            patterns.append(pattern)

        return patterns

    def _infer_cause(self, error_cases: List[ErrorCase]) -> str:
        """Infer suggested cause for error pattern.

        Args:
            error_cases: List of error cases in pattern

        Returns:
            Suggested cause string (max 200 chars)
        """
        error_type = error_cases[0].error_type
        avg_confidence = np.mean([e.confidence for e in error_cases if e.confidence is not None])

        if error_type == ErrorType.FALSE_POSITIVE:
            if avg_confidence > 0.7:
                return "Model generating confident false positives - check confidence threshold or retrain"
            else:
                return "Model generating spurious detections - review training data quality"

        elif error_type == ErrorType.FALSE_NEGATIVE:
            return "Model missing detections - check recall, increase sensitivity, or add training data"

        elif error_type == ErrorType.MISCLASSIFICATION:
            return "Confusion between similar classes - review class definitions or add discriminative features"

        elif error_type == ErrorType.LOW_IOU:
            return "Poor localization - check anchor boxes, increase resolution, or improve bounding box regression"

        else:
            return "Unknown error pattern - manual review recommended"
