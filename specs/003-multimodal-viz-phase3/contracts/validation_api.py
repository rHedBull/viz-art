"""
Ground Truth Validation API Contract

This file defines the public API for ground truth dataset management
and accuracy validation functionality.
"""

from typing import Dict, List, Optional, Any, Protocol, Callable
from pathlib import Path
from enum import Enum


class AnnotationFormat(str, Enum):
    """Supported annotation formats"""
    COCO = "coco"
    YOLO = "yolo"
    PNG_MASKS = "png_masks"
    PLY_LABELS = "ply_labels"
    CUSTOM = "custom"


class GroundTruthDataset(Protocol):
    """
    Ground truth dataset with labeled samples.

    Handles loading, validation, and access to annotations.
    """

    @property
    def dataset_id(self) -> str:
        """Unique dataset identifier"""
        ...

    @property
    def sample_count(self) -> int:
        """Total number of labeled samples"""
        ...

    def get_annotation(self, sample_id: str) -> Dict[str, Any]:
        """
        Retrieve annotation for a sample.

        Args:
            sample_id: Sample identifier (image filename, point cloud ID, etc.)

        Returns:
            Annotation dictionary (format depends on dataset type)

        Raises:
            KeyError: If sample_id not found in dataset

        Example:
            annotation = dataset.get_annotation("image_001.jpg")
            # For COCO: {"image_id": 1, "annotations": [...bboxes...]}
        """
        ...

    def iter_samples(self):
        """
        Iterate over all samples in dataset.

        Yields:
            Tuples of (sample_id: str, annotation: Dict)

        Example:
            for sample_id, annotation in dataset.iter_samples():
                print(f"{sample_id}: {len(annotation['annotations'])} objects")
        """
        ...


class MetricsCalculator(Protocol):
    """
    Accuracy metrics calculator.

    Compares predictions against ground truth annotations.
    """

    def calculate_precision_recall(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict],
    ) -> Dict[str, float]:
        """
        Calculate precision, recall, F1 score.

        Args:
            predictions: List of prediction dictionaries
            ground_truth: List of ground truth annotation dictionaries

        Returns:
            Dictionary with keys:
                - precision: float [0, 1]
                - recall: float [0, 1]
                - f1_score: float [0, 1]
                - true_positives: int
                - false_positives: int
                - false_negatives: int

        Example:
            metrics = calculator.calculate_precision_recall(
                predictions=[{"bbox": [10, 20, 30, 40], "class": "car"}],
                ground_truth=[{"bbox": [10, 20, 30, 40], "class": "car"}],
            )
            print(f"Precision: {metrics['precision']:.2f}")
        """
        ...

    def calculate_mean_average_precision(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict],
        iou_threshold: float = 0.5,
    ) -> float:
        """
        Calculate mAP for object detection.

        Args:
            predictions: List with keys: bbox, score, class
            ground_truth: List with keys: bbox, class
            iou_threshold: IoU threshold for considering a match

        Returns:
            Mean Average Precision [0, 1]

        Example:
            map_score = calculator.calculate_mean_average_precision(
                predictions=detections,
                ground_truth=annotations,
                iou_threshold=0.5,
            )
        """
        ...

    def calculate_iou(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict],
    ) -> float:
        """
        Calculate mean IoU for segmentation.

        Args:
            predictions: Segmentation masks or point cloud labels
            ground_truth: Ground truth masks or labels

        Returns:
            Mean Intersection over Union [0, 1]
        """
        ...


class ErrorAnalyzer(Protocol):
    """
    Error analysis and visualization tools.

    Helps identify and categorize prediction failures.
    """

    def find_failures(
        self,
        run_id: str,
        dataset: GroundTruthDataset,
        threshold: float = 0.5,
    ) -> List[Dict]:
        """
        Find all failed predictions for a run.

        Args:
            run_id: Run to analyze
            dataset: Ground truth dataset used for validation
            threshold: Confidence/IoU threshold for considering a failure

        Returns:
            List of failure dictionaries with keys:
                - sample_id: str
                - error_type: str (false_positive, false_negative, misclassification)
                - confidence: float
                - ground_truth: Dict
                - prediction: Dict

        Example:
            failures = analyzer.find_failures("abc-123", dataset, threshold=0.5)
            print(f"Found {len(failures)} errors")
        """
        ...

    def categorize_errors(
        self,
        failures: List[Dict],
    ) -> Dict[str, List[Dict]]:
        """
        Group failures by error type.

        Args:
            failures: List from find_failures()

        Returns:
            Dictionary mapping error_type -> list of failures

        Example:
            categories = analyzer.categorize_errors(failures)
            print(f"False positives: {len(categories['false_positive'])}")
        """
        ...

    def visualize_comparison(
        self,
        sample_id: str,
        prediction: Dict,
        ground_truth: Dict,
        output_path: Path,
    ) -> None:
        """
        Generate side-by-side visualization of prediction vs ground truth.

        Args:
            sample_id: Sample identifier
            prediction: Prediction to visualize
            ground_truth: Ground truth annotation
            output_path: Where to save visualization (image or HTML)

        Example:
            analyzer.visualize_comparison(
                "image_001.jpg",
                prediction={"bbox": [10, 20, 30, 40]},
                ground_truth={"bbox": [12, 22, 32, 42]},
                output_path=Path("comparison.png"),
            )
        """
        ...


class FormatLoader(Protocol):
    """
    Plugin interface for custom annotation format loaders.

    Allows users to extend supported formats beyond built-in ones.
    """

    @property
    def format_name(self) -> str:
        """Unique format identifier (e.g., "coco", "yolo")"""
        ...

    def load(self, annotation_file: Path) -> Dict[str, Any]:
        """
        Load and parse annotation file.

        Args:
            annotation_file: Path to annotation file

        Returns:
            Parsed annotations (format-specific structure)

        Raises:
            ValueError: If file format invalid
        """
        ...

    def validate(self, annotations: Dict) -> bool:
        """
        Validate annotation structure.

        Args:
            annotations: Loaded annotations from load()

        Returns:
            True if valid, False otherwise
        """
        ...


# Public factory functions

def create_dataset(
    dataset_id: str,
    root_path: Path,
    annotation_format: AnnotationFormat,
    annotation_files: List[Path],
    name: Optional[str] = None,
) -> GroundTruthDataset:
    """
    Create a ground truth dataset.

    Args:
        dataset_id: Unique identifier (e.g., "coco_val_2017")
        root_path: Base directory containing data
        annotation_format: Format enum
        annotation_files: Paths to annotation files (relative to root_path)
        name: Human-readable name (defaults to dataset_id)

    Returns:
        GroundTruthDataset instance

    Raises:
        ValueError: If annotation_files don't exist or format invalid
        NotImplementedError: If annotation_format not supported

    Example:
        dataset = create_dataset(
            dataset_id="my_validation_set",
            root_path=Path("data/validation"),
            annotation_format=AnnotationFormat.COCO,
            annotation_files=[Path("annotations/instances.json")],
        )
    """
    ...


def create_metrics_calculator(
    metric_type: str = "auto",
) -> MetricsCalculator:
    """
    Create a metrics calculator.

    Args:
        metric_type: Type of metrics to compute
            - "auto": Detect from annotation format
            - "classification": Precision, recall, F1
            - "detection": mAP, IoU
            - "segmentation": IoU, pixel accuracy

    Returns:
        MetricsCalculator instance

    Example:
        calculator = create_metrics_calculator("detection")
        metrics = calculator.calculate_mean_average_precision(preds, gt)
    """
    ...


def create_error_analyzer(
    output_dir: Path,
) -> ErrorAnalyzer:
    """
    Create an error analyzer.

    Args:
        output_dir: Directory for saved visualizations

    Returns:
        ErrorAnalyzer instance

    Example:
        analyzer = create_error_analyzer(Path("output/error_analysis"))
        failures = analyzer.find_failures("abc-123", dataset)
    """
    ...


def register_format_loader(
    loader: FormatLoader,
) -> None:
    """
    Register a custom annotation format loader.

    Args:
        loader: Format loader implementation

    Example:
        class MyCustomLoader:
            @property
            def format_name(self):
                return "my_custom_format"

            def load(self, annotation_file):
                with open(annotation_file) as f:
                    return json.load(f)

            def validate(self, annotations):
                return "version" in annotations

        register_format_loader(MyCustomLoader())
    """
    ...


# Integration with pipeline

def validate_pipeline(
    run_id: str,
    dataset: GroundTruthDataset,
    stage_outputs: Dict[str, List[Dict]],
    metrics_calculator: Optional[MetricsCalculator] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Validate pipeline outputs against ground truth.

    Args:
        run_id: Run identifier
        dataset: Ground truth dataset
        stage_outputs: Dictionary mapping stage_name -> list of predictions
        metrics_calculator: Custom calculator (None = auto-detect)

    Returns:
        Dictionary mapping stage_name -> metrics dict
            Each metrics dict contains: precision, recall, f1, mAP (if applicable)

    Example:
        results = validate_pipeline(
            run_id="abc-123",
            dataset=my_dataset,
            stage_outputs={
                "detection": [{"bbox": [10, 20, 30, 40], "class": "car"}],
                "classification": [{"class": "car", "confidence": 0.95}],
            },
        )
        print(f"Detection mAP: {results['detection']['mean_average_precision']}")
    """
    ...


# CLI integration

def validate_cli(
    run_id: str,
    dataset_id: str,
    output_report: Optional[str] = None,
) -> int:
    """
    CLI interface for validation.

    Args:
        run_id: Run to validate
        dataset_id: Ground truth dataset to use
        output_report: Path to save HTML report (None = print summary)

    Returns:
        Exit code (0 = success, 1 = validation failed)

    Example (as CLI):
        viz-art validate --run-id abc-123 --dataset coco_val --output report.html
    """
    ...
