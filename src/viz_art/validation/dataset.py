"""Ground Truth Dataset Management

This module provides classes for managing ground truth datasets used for
pipeline validation. It supports multiple annotation formats and provides
unified access to annotations.
"""

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ..types.monitoring import AnnotationFormat


class GroundTruthDataset:
    """Manages a ground truth dataset for validation.

    Provides access to annotations and samples from a labeled dataset.
    Validates annotation files exist and provides iteration over samples.

    Attributes:
        dataset_id: Unique identifier for this dataset
        root_path: Base directory containing images/point clouds and annotations
        annotation_format: Format of the annotation files (COCO, YOLO, etc.)
        annotation_files: List of annotation file paths relative to root_path
        name: Human-readable dataset name
        metadata: Additional metadata about the dataset
    """

    def __init__(
        self,
        dataset_id: str,
        root_path: Path,
        annotation_format: AnnotationFormat,
        annotation_files: List[Path],
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize ground truth dataset.

        Args:
            dataset_id: Unique identifier for the dataset
            root_path: Base directory for the dataset
            annotation_format: Format of annotations (COCO, YOLO, etc.)
            annotation_files: List of annotation files (relative to root_path)
            name: Optional human-readable name
            metadata: Optional additional metadata

        Raises:
            FileNotFoundError: If root_path doesn't exist
            FileNotFoundError: If any annotation_file doesn't exist
        """
        self.dataset_id = dataset_id
        self.root_path = Path(root_path)
        self.annotation_format = annotation_format
        self.annotation_files = [Path(f) for f in annotation_files]
        self.name = name or dataset_id
        self.metadata = metadata or {}

        # Validate root path exists
        if not self.root_path.exists():
            raise FileNotFoundError(f"Dataset root path not found: {self.root_path}")

        # Validate all annotation files exist
        for ann_file in self.annotation_files:
            full_path = self.root_path / ann_file
            if not full_path.exists():
                raise FileNotFoundError(f"Annotation file not found: {full_path}")

        # Lazy-loaded annotations
        self._annotations: Optional[Dict[str, Any]] = None

    def _load_annotations(self) -> Dict[str, Any]:
        """Load annotations from files using the appropriate loader.

        Returns:
            Dictionary mapping sample_id to annotation data

        Raises:
            ImportError: If the required loader is not available
        """
        from .loaders import get_loader

        loader = get_loader(self.annotation_format)
        annotations = {}

        for ann_file in self.annotation_files:
            full_path = self.root_path / ann_file
            file_annotations = loader.load(full_path)
            annotations.update(file_annotations)

        return annotations

    def get_annotation(self, sample_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve annotation for a specific sample.

        Args:
            sample_id: Identifier for the sample (e.g., image filename)

        Returns:
            Annotation dictionary or None if not found
        """
        if self._annotations is None:
            self._annotations = self._load_annotations()

        return self._annotations.get(sample_id)

    def iter_samples(self) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """Iterate over all samples in the dataset.

        Yields:
            Tuple of (sample_id, annotation)
        """
        if self._annotations is None:
            self._annotations = self._load_annotations()

        for sample_id, annotation in self._annotations.items():
            yield sample_id, annotation

    @property
    def sample_count(self) -> int:
        """Get total number of samples in the dataset.

        Returns:
            Number of annotated samples
        """
        if self._annotations is None:
            self._annotations = self._load_annotations()

        return len(self._annotations)

    def __repr__(self) -> str:
        return (
            f"GroundTruthDataset(dataset_id={self.dataset_id!r}, "
            f"name={self.name!r}, format={self.annotation_format.value}, "
            f"samples={self.sample_count})"
        )
