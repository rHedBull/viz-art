"""Annotation Format Loaders

Provides loaders for different ground truth annotation formats (COCO, YOLO, etc.).
Implements a plugin system for extensibility.
"""

import json
from pathlib import Path
from typing import Any, Dict, Protocol

import numpy as np
from PIL import Image

from ..types.monitoring import AnnotationFormat


class FormatLoader(Protocol):
    """Protocol for annotation format loaders.

    Custom loaders must implement this protocol to be registered.
    """

    @property
    def format_name(self) -> str:
        """Return the format name (e.g., 'coco', 'yolo')."""
        ...

    def load(self, annotation_file: Path) -> Dict[str, Any]:
        """Load annotations from file.

        Args:
            annotation_file: Path to annotation file

        Returns:
            Dictionary mapping sample_id (e.g., image filename) to annotations
        """
        ...

    def validate(self, annotations: Dict[str, Any]) -> bool:
        """Validate loaded annotations.

        Args:
            annotations: Loaded annotation dictionary

        Returns:
            True if valid, False otherwise
        """
        ...


class COCOLoader:
    """Loader for COCO format annotations.

    COCO format: https://cocodataset.org/#format-data
    Expected structure:
    {
        "images": [{"id": 1, "file_name": "img.jpg", ...}],
        "annotations": [{"id": 1, "image_id": 1, "bbox": [...], ...}],
        "categories": [{"id": 1, "name": "car"}]
    }
    """

    @property
    def format_name(self) -> str:
        return "coco"

    def load(self, annotation_file: Path) -> Dict[str, Any]:
        """Load COCO format annotations.

        Args:
            annotation_file: Path to COCO JSON file

        Returns:
            Dictionary mapping image filename to list of annotations
        """
        with open(annotation_file) as f:
            coco_data = json.load(f)

        # Build image_id -> filename mapping
        image_map = {img["id"]: img["file_name"] for img in coco_data["images"]}

        # Build category_id -> name mapping
        category_map = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

        # Group annotations by image
        annotations_by_image: Dict[str, list] = {}
        for ann in coco_data["annotations"]:
            image_id = ann["image_id"]
            filename = image_map[image_id]

            if filename not in annotations_by_image:
                annotations_by_image[filename] = []

            # Convert COCO format to standardized format
            annotations_by_image[filename].append(
                {
                    "bbox": ann["bbox"],  # [x, y, width, height]
                    "category_id": ann["category_id"],
                    "category_name": category_map[ann["category_id"]],
                    "area": ann.get("area"),
                    "iscrowd": ann.get("iscrowd", 0),
                    "segmentation": ann.get("segmentation"),
                }
            )

        return annotations_by_image

    def validate(self, annotations: Dict[str, Any]) -> bool:
        """Validate COCO annotations structure.

        Args:
            annotations: COCO data dictionary

        Returns:
            True if structure is valid
        """
        required_keys = {"images", "annotations", "categories"}
        return all(key in annotations for key in required_keys)


class YOLOLoader:
    """Loader for YOLO format annotations.

    YOLO format: One .txt file per image with lines:
    <class_id> <x_center> <y_center> <width> <height>
    All coordinates normalized to [0, 1].
    """

    @property
    def format_name(self) -> str:
        return "yolo"

    def load(self, annotation_file: Path) -> Dict[str, Any]:
        """Load YOLO format annotations.

        Args:
            annotation_file: Path to directory containing .txt files
                            or path to a single .txt file

        Returns:
            Dictionary mapping image filename to list of bounding boxes
        """
        annotations: Dict[str, list] = {}

        # Handle single file or directory
        if annotation_file.is_file():
            annotation_files = [annotation_file]
        else:
            annotation_files = list(annotation_file.glob("*.txt"))

        for txt_file in annotation_files:
            # Image filename is txt filename with image extension
            # User needs to specify image directory separately
            image_filename = txt_file.stem

            boxes = []
            with open(txt_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    boxes.append(
                        {
                            "class_id": class_id,
                            "bbox_normalized": [x_center, y_center, width, height],
                            "format": "yolo",  # [x_center, y_center, w, h] normalized
                        }
                    )

            annotations[image_filename] = boxes

        return annotations

    def validate(self, annotations: Dict[str, Any]) -> bool:
        """Validate YOLO annotations.

        Args:
            annotations: Loaded YOLO annotations

        Returns:
            True if all coordinates are in valid range [0, 1]
        """
        for image_anns in annotations.values():
            for ann in image_anns:
                bbox = ann.get("bbox_normalized", [])
                if len(bbox) != 4:
                    return False
                # Check all coordinates in [0, 1]
                if not all(0 <= coord <= 1 for coord in bbox):
                    return False
        return True


class PNGMaskLoader:
    """Loader for PNG segmentation masks.

    PNG masks: One PNG file per image where pixel values represent class IDs.
    Typically grayscale or indexed color images.
    """

    @property
    def format_name(self) -> str:
        return "png_masks"

    def load(self, annotation_file: Path) -> Dict[str, Any]:
        """Load PNG segmentation masks.

        Args:
            annotation_file: Path to directory containing mask PNG files

        Returns:
            Dictionary mapping image filename to mask data
        """
        annotations: Dict[str, Any] = {}

        if annotation_file.is_file():
            mask_files = [annotation_file]
        else:
            mask_files = list(annotation_file.glob("*.png"))

        for mask_file in mask_files:
            image_filename = mask_file.stem

            # Load mask as numpy array
            mask = Image.open(mask_file)
            mask_array = np.array(mask)

            # Get unique class IDs (excluding background if 0)
            unique_classes = np.unique(mask_array)

            annotations[image_filename] = {
                "mask_path": str(mask_file),
                "mask_shape": mask_array.shape,
                "classes": unique_classes.tolist(),
                "format": "png_mask",
            }

        return annotations

    def validate(self, annotations: Dict[str, Any]) -> bool:
        """Validate PNG mask annotations.

        Args:
            annotations: Loaded mask annotations

        Returns:
            True if masks have valid dimensions
        """
        for ann in annotations.values():
            mask_shape = ann.get("mask_shape")
            if not mask_shape or len(mask_shape) < 2:
                return False
        return True


class PLYLabelLoader:
    """Loader for PLY/PCD point clouds with label attributes.

    Point cloud labels: Each point has a 'label' or 'class' attribute
    indicating its semantic class.
    """

    @property
    def format_name(self) -> str:
        return "ply_labels"

    def load(self, annotation_file: Path) -> Dict[str, Any]:
        """Load PLY/PCD point cloud labels.

        Args:
            annotation_file: Path to directory containing labeled point clouds

        Returns:
            Dictionary mapping point cloud filename to label data
        """
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError(
                "Open3D is required for point cloud loading. "
                "Install with: pip install open3d"
            )

        annotations: Dict[str, Any] = {}

        if annotation_file.is_file():
            cloud_files = [annotation_file]
        else:
            cloud_files = list(annotation_file.glob("*.ply")) + list(
                annotation_file.glob("*.pcd")
            )

        for cloud_file in cloud_files:
            cloud_filename = cloud_file.stem

            # Load point cloud
            pcd = o3d.io.read_point_cloud(str(cloud_file))

            # Try to extract label attribute
            # Note: Open3D doesn't directly support custom attributes,
            # so this is a placeholder for actual implementation
            # In practice, you'd need to parse the PLY file manually
            # or use a specialized library

            annotations[cloud_filename] = {
                "cloud_path": str(cloud_file),
                "num_points": len(pcd.points),
                "format": "ply_labels",
                # Label extraction would go here
                "has_labels": False,  # Placeholder
            }

        return annotations

    def validate(self, annotations: Dict[str, Any]) -> bool:
        """Validate point cloud annotations.

        Args:
            annotations: Loaded point cloud annotations

        Returns:
            True if point clouds have label field
        """
        for ann in annotations.values():
            # Check if point cloud has labels
            # This is a placeholder - actual validation would check
            # for label attribute in the point cloud
            if "num_points" not in ann:
                return False
        return True


# Loader registry
_LOADER_REGISTRY: Dict[str, FormatLoader] = {}


def register_format_loader(loader: FormatLoader) -> None:
    """Register a custom format loader.

    Args:
        loader: Format loader instance implementing FormatLoader protocol

    Example:
        >>> class MyCustomLoader:
        ...     @property
        ...     def format_name(self):
        ...         return "my_format"
        ...     def load(self, path):
        ...         return {...}
        ...     def validate(self, annotations):
        ...         return True
        >>> register_format_loader(MyCustomLoader())
    """
    _LOADER_REGISTRY[loader.format_name] = loader


def get_loader(annotation_format: AnnotationFormat) -> FormatLoader:
    """Get loader for annotation format.

    Args:
        annotation_format: Format enum value

    Returns:
        Loader instance for the format

    Raises:
        ValueError: If format not supported
    """
    # Map enum to string format name
    format_name = annotation_format.value

    if format_name not in _LOADER_REGISTRY:
        raise ValueError(
            f"No loader registered for format: {format_name}. "
            f"Available formats: {list(_LOADER_REGISTRY.keys())}"
        )

    return _LOADER_REGISTRY[format_name]


# Register built-in loaders
register_format_loader(COCOLoader())
register_format_loader(YOLOLoader())
register_format_loader(PNGMaskLoader())
register_format_loader(PLYLabelLoader())
