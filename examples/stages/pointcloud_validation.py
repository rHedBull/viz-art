"""Point cloud validation stage for quality checks.

This stage validates point cloud data quality against configurable rules,
checking for empty clouds, NaN/Inf values, point count limits, and coordinate ranges.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
from viz_art.pipeline.stage import PipelineStage
from viz_art.types.pointcloud import PointCloudValidationRules, PointCloud

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    o3d = None


class PointCloudValidationStage(PipelineStage):
    """Validate point cloud data quality.

    Checks point clouds against configurable quality rules including:
    - Point count limits (min/max)
    - NaN and Inf value detection
    - Coordinate range validation
    - Empty cloud detection

    Example:
        >>> rules = PointCloudValidationRules(
        ...     min_points=1000,
        ...     max_points=2000000,
        ...     check_nan=True,
        ...     check_inf=True
        ... )
        >>> validator = PointCloudValidationStage(
        ...     name="validate",
        ...     rules=rules,
        ...     raise_on_invalid=True
        ... )
        >>> result = validator.run({"pointcloud": pcd})
        >>> if result["is_valid"]:
        ...     print("Point cloud passed validation")
    """

    def __init__(
        self,
        name: str = "pointcloud_validator",
        rules: Optional[PointCloudValidationRules] = None,
        raise_on_invalid: bool = True,
    ):
        """Initialize PointCloudValidationStage.

        Args:
            name: Stage name
            rules: Validation rules to apply (uses defaults if None)
            raise_on_invalid: Whether to raise exception on validation failure

        Raises:
            ValueError: If rules are invalid
        """
        self._name = name
        self.rules = rules or PointCloudValidationRules()
        self.raise_on_invalid = raise_on_invalid

    @property
    def name(self) -> str:
        """Return stage name."""
        return self._name

    @property
    def input_keys(self) -> List[str]:
        """Return required input keys."""
        return ["points"]  # Accept either pointcloud object or points array

    @property
    def output_keys(self) -> List[str]:
        """Return output keys."""
        return ["is_valid", "validation_errors", "validation_warnings", "metrics"]

    @property
    def input_data_types(self) -> Optional[Dict[str, type]]:
        """Return input data types for validation."""
        return {
            "points": (np.ndarray, object),  # NumPy array or PointCloud object
        }

    @property
    def output_data_types(self) -> Optional[Dict[str, type]]:
        """Return output data types for validation."""
        return {
            "is_valid": bool,
            "validation_errors": list,
            "validation_warnings": list,
            "metrics": dict,
        }

    def pre_process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate inputs and extract point data.

        Args:
            inputs: Dictionary with "points" or "pointcloud" key

        Returns:
            Processed inputs with extracted points

        Raises:
            ValueError: If no valid input provided
        """
        # Accept multiple input formats
        if "points" in inputs and isinstance(inputs["points"], np.ndarray):
            points = inputs["points"]
            colors = inputs.get("colors")
            normals = inputs.get("normals")
        elif "pointcloud" in inputs:
            pcd_input = inputs["pointcloud"]

            # Handle Open3D PointCloud
            if OPEN3D_AVAILABLE and isinstance(pcd_input, o3d.geometry.PointCloud):
                points = np.asarray(pcd_input.points, dtype=np.float32)
                colors = np.asarray(pcd_input.colors, dtype=np.float32) if pcd_input.has_colors() else None
                normals = np.asarray(pcd_input.normals, dtype=np.float32) if pcd_input.has_normals() else None
            # Handle PointCloud dataclass
            elif isinstance(pcd_input, PointCloud):
                points = pcd_input.points
                colors = pcd_input.colors
                normals = pcd_input.normals
            else:
                raise ValueError(
                    f"Unsupported pointcloud type: {type(pcd_input)}. "
                    "Expected Open3D PointCloud or PointCloud dataclass."
                )
        else:
            raise ValueError(
                "No valid input provided. Expected 'points' (NumPy array) "
                "or 'pointcloud' (Open3D PointCloud or PointCloud dataclass)."
            )

        # Validate points array shape
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                f"Points must be Nx3 array, got shape {points.shape}. "
                f"Expected format: (num_points, 3) with XYZ coordinates."
            )

        return {
            "points": points,
            "colors": colors,
            "normals": normals,
            "num_points": len(points),
        }

    def predict(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        """Run validation checks.

        Args:
            preprocessed: Output from pre_process()

        Returns:
            Validation results

        Raises:
            RuntimeError: If validation fails and raise_on_invalid=True
        """
        points = preprocessed["points"]
        num_points = preprocessed["num_points"]

        errors = []
        warnings = []

        # Check point count
        if num_points < self.rules.min_points:
            errors.append(f"Too few points: {num_points} < {self.rules.min_points}")
            if self.rules.fail_fast:
                return self._create_result(False, errors, warnings, preprocessed)

        if self.rules.max_points and num_points > self.rules.max_points:
            errors.append(f"Too many points: {num_points} > {self.rules.max_points}")
            if self.rules.fail_fast:
                return self._create_result(False, errors, warnings, preprocessed)

        # Check for NaN values
        has_nan = False
        if self.rules.check_nan:
            has_nan = np.isnan(points).any()
            if has_nan:
                nan_count = np.isnan(points).sum()
                errors.append(f"NaN values detected: {nan_count} NaN entries in points")
                if self.rules.fail_fast:
                    return self._create_result(False, errors, warnings, preprocessed)

        # Check for Inf values
        has_inf = False
        if self.rules.check_inf:
            has_inf = np.isinf(points).any()
            if has_inf:
                inf_count = np.isinf(points).sum()
                errors.append(f"Inf values detected: {inf_count} Inf entries in points")
                if self.rules.fail_fast:
                    return self._create_result(False, errors, warnings, preprocessed)

        # Check coordinate ranges
        if not has_nan and not has_inf:  # Only check ranges if no NaN/Inf
            mins = points.min(axis=0)
            maxs = points.max(axis=0)

            if self.rules.coord_range_min and self.rules.coord_range_max:
                for i, (mn, mx, coord) in enumerate(
                    zip(self.rules.coord_range_min, self.rules.coord_range_max, ['X', 'Y', 'Z'])
                ):
                    if mins[i] < mn or maxs[i] > mx:
                        errors.append(
                            f"{coord} range [{mins[i]:.2f}, {maxs[i]:.2f}] outside "
                            f"allowed range [{mn}, {mx}]"
                        )
                        if self.rules.fail_fast:
                            return self._create_result(False, errors, warnings, preprocessed)

            # Add range metrics
            preprocessed["coord_ranges"] = {
                "x": (float(mins[0]), float(maxs[0])),
                "y": (float(mins[1]), float(maxs[1])),
                "z": (float(mins[2]), float(maxs[2])),
            }
        else:
            preprocessed["coord_ranges"] = None

        # Warnings (non-critical issues)
        if num_points < 100:
            warnings.append(f"Very few points: {num_points} (consider if this is expected)")

        is_valid = len(errors) == 0

        return self._create_result(is_valid, errors, warnings, preprocessed)

    def _create_result(
        self, is_valid: bool, errors: List[str], warnings: List[str], preprocessed: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create validation result dictionary.

        Args:
            is_valid: Whether validation passed
            errors: List of error messages
            warnings: List of warning messages
            preprocessed: Preprocessed data with metrics

        Returns:
            Result dictionary

        Raises:
            RuntimeError: If validation failed and raise_on_invalid=True
        """
        if not is_valid and self.raise_on_invalid:
            raise RuntimeError(
                f"Point cloud validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )

        return {
            "is_valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "num_points": preprocessed["num_points"],
            "coord_ranges": preprocessed.get("coord_ranges"),
        }

    def post_process(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Format outputs.

        Args:
            predictions: Output from predict()

        Returns:
            Final output dictionary matching output_keys
        """
        return {
            "is_valid": predictions["is_valid"],
            "validation_errors": predictions["errors"],
            "validation_warnings": predictions["warnings"],
            "metrics": {
                "num_points": predictions["num_points"],
                "has_nan": any("NaN" in e for e in predictions["errors"]),
                "has_inf": any("Inf" in e for e in predictions["errors"]),
                "coord_ranges": predictions["coord_ranges"],
            }
        }

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the full validation pipeline.

        Args:
            inputs: Dictionary with point cloud data

        Returns:
            Validation results

        Raises:
            ValueError: If inputs invalid
            RuntimeError: If validation fails and raise_on_invalid=True
        """
        preprocessed = self.pre_process(inputs)
        predictions = self.predict(preprocessed)
        outputs = self.post_process(predictions)
        return outputs
