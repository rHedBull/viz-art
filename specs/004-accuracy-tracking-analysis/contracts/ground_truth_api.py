"""
Ground Truth Management API Contract

Defines interfaces for loading, validating, and managing ground truth datasets.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class AnnotationFormat(Enum):
    """Supported annotation formats."""
    COCO = "coco"  # COCO JSON format for object detection
    PASCAL_VOC = "pascal_voc"  # PASCAL VOC XML for segmentation
    PCD_LABELS = "pcd_labels"  # Point cloud annotation files


@dataclass
class GroundTruthDataset:
    """Ground truth dataset metadata."""
    dataset_id: str
    name: str
    description: Optional[str]
    base_path: Path
    annotation_path: Path
    annotation_format: AnnotationFormat
    num_samples: int
    sample_ids: List[str]
    metadata: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime


@dataclass
class GroundTruthSample:
    """Individual labeled sample."""
    sample_id: str
    dataset_id: str
    stage_labels: Dict[str, Any]  # Labels for each pipeline stage
    final_label: Any  # Ground truth for final output
    annotation_format: AnnotationFormat
    confidence_scores: Optional[Dict[str, float]]
    metadata: Optional[Dict[str, Any]]
    image_path: Optional[Path]
    pointcloud_path: Optional[Path]


class GroundTruthLoader(ABC):
    """Interface for loading ground truth datasets."""

    @abstractmethod
    def load_dataset(self, dataset_path: Path, annotation_format: AnnotationFormat) -> GroundTruthDataset:
        """
        Load ground truth dataset metadata.

        Args:
            dataset_path: Path to dataset directory
            annotation_format: Format of annotations

        Returns:
            GroundTruthDataset instance

        Raises:
            FileNotFoundError: If dataset path doesn't exist
            ValueError: If annotation format is invalid or incompatible
        """
        pass

    @abstractmethod
    def load_sample(self, dataset: GroundTruthDataset, sample_id: str) -> GroundTruthSample:
        """
        Load a single ground truth sample.

        Args:
            dataset: Parent dataset
            sample_id: Unique sample identifier

        Returns:
            GroundTruthSample instance

        Raises:
            FileNotFoundError: If sample files don't exist
            ValueError: If sample data is invalid or corrupted
        """
        pass

    @abstractmethod
    def validate_dataset(self, dataset: GroundTruthDataset) -> List[str]:
        """
        Validate dataset integrity.

        Args:
            dataset: Dataset to validate

        Returns:
            List of validation errors (empty if valid)

        Validation checks:
        - All sample_ids have corresponding files
        - Annotation format matches files
        - No corrupted or missing annotations
        """
        pass


class GroundTruthValidator(ABC):
    """Interface for validating ground truth data quality."""

    @abstractmethod
    def validate_sample(self, sample: GroundTruthSample) -> Dict[str, Any]:
        """
        Validate a single sample's ground truth labels.

        Args:
            sample: Ground truth sample to validate

        Returns:
            Validation result:
            {
                "is_valid": bool,
                "errors": List[str],
                "warnings": List[str],
                "metadata": Dict[str, Any]
            }

        Checks:
        - Label format matches annotation_format
        - Bounding boxes within image bounds
        - Point cloud labels have valid coordinates
        - Confidence scores in [0, 1] range
        """
        pass

    @abstractmethod
    def check_completeness(self, dataset: GroundTruthDataset, pipeline_stages: List[str]) -> Dict[str, float]:
        """
        Check dataset completeness for pipeline stages.

        Args:
            dataset: Dataset to check
            pipeline_stages: List of stage names requiring labels

        Returns:
            Completeness percentage per stage:
            {
                "detection": 0.95,  # 95% of samples have detection labels
                "classification": 1.0,  # 100% labeled
                ...
            }
        """
        pass
