"""Load camera calibration data from YAML/JSON files."""

from pathlib import Path
from typing import Dict, Any
import numpy as np
from omegaconf import OmegaConf

from viz_art.types.pointcloud import (
    CameraIntrinsics,
    CameraExtrinsics,
    Calibration,
)


def load_calibration(calibration_path: str | Path) -> Calibration:
    """Load calibration from YAML or JSON file.

    Expected file format:
        camera_name: "camera_id"
        calibration_date: "2025-10-22"
        calibration_method: "opencv_checkerboard"
        intrinsics:
          fx: 525.0
          fy: 525.0
          cx: 319.5
          cy: 239.5
          width: 640
          height: 480
          distortion_coeffs: [0.0, 0.0, 0.0, 0.0, 0.0]
        extrinsics:
          rotation_matrix:
            - [1.0, 0.0, 0.0]
            - [0.0, 1.0, 0.0]
            - [0.0, 0.0, 1.0]
          translation_vector: [0.0, 0.0, 0.0]

    Args:
        calibration_path: Path to YAML/JSON calibration file

    Returns:
        Calibration object with parsed parameters

    Raises:
        FileNotFoundError: If calibration file doesn't exist
        ValueError: If calibration format is invalid
        KeyError: If required fields are missing
    """
    calibration_path = Path(calibration_path)

    if not calibration_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calibration_path}")

    # Load configuration using OmegaConf
    try:
        config = OmegaConf.load(calibration_path)
    except Exception as e:
        raise ValueError(f"Failed to parse calibration file: {e}") from e

    # Extract required fields
    try:
        camera_name = config.camera_name
        intrinsics_dict = OmegaConf.to_container(config.intrinsics, resolve=True)
        extrinsics_dict = OmegaConf.to_container(config.extrinsics, resolve=True)
    except AttributeError as e:
        raise KeyError(f"Missing required calibration field: {e}") from e

    # Parse intrinsics
    try:
        intrinsics = CameraIntrinsics(
            fx=float(intrinsics_dict["fx"]),
            fy=float(intrinsics_dict["fy"]),
            cx=float(intrinsics_dict["cx"]),
            cy=float(intrinsics_dict["cy"]),
            width=int(intrinsics_dict["width"]),
            height=int(intrinsics_dict["height"]),
            distortion_coeffs=list(intrinsics_dict.get("distortion_coeffs", [0.0] * 5)),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"Invalid intrinsics format: {e}") from e

    # Parse extrinsics
    try:
        rotation_matrix = np.array(
            extrinsics_dict["rotation_matrix"], dtype=np.float64
        )

        translation_vector = np.array(
            extrinsics_dict["translation_vector"], dtype=np.float64
        ).reshape(3, 1)

        extrinsics = CameraExtrinsics(
            rotation_matrix=rotation_matrix,
            translation_vector=translation_vector,
        )
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"Invalid extrinsics format: {e}") from e

    # Create Calibration object
    calibration = Calibration(
        camera_name=str(camera_name),
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        calibration_date=str(config.get("calibration_date", "")),
        calibration_method=str(config.get("calibration_method", "manual")),
    )

    return calibration


def validate_calibration(calibration: Calibration) -> tuple[bool, list[str]]:
    """Validate calibration parameters.

    Args:
        calibration: Calibration object to validate

    Returns:
        Tuple of (is_valid, list_of_warnings)
    """
    warnings = []

    # Check focal lengths are reasonable
    if calibration.intrinsics.fx < 100 or calibration.intrinsics.fx > 10000:
        warnings.append(
            f"Focal length fx={calibration.intrinsics.fx} seems unusual (expected 100-10000)"
        )

    if calibration.intrinsics.fy < 100 or calibration.intrinsics.fy > 10000:
        warnings.append(
            f"Focal length fy={calibration.intrinsics.fy} seems unusual (expected 100-10000)"
        )

    # Check principal point is roughly centered
    cx_center = calibration.intrinsics.width / 2
    cy_center = calibration.intrinsics.height / 2

    if abs(calibration.intrinsics.cx - cx_center) > calibration.intrinsics.width * 0.3:
        warnings.append(
            f"Principal point cx={calibration.intrinsics.cx} is far from center "
            f"(expected ~{cx_center})"
        )

    if abs(calibration.intrinsics.cy - cy_center) > calibration.intrinsics.height * 0.3:
        warnings.append(
            f"Principal point cy={calibration.intrinsics.cy} is far from center "
            f"(expected ~{cy_center})"
        )

    # Check rotation matrix is orthogonal
    R = calibration.extrinsics.rotation_matrix
    RTR = R.T @ R
    identity = np.eye(3)
    if not np.allclose(RTR, identity, atol=0.01):
        warnings.append(
            "Rotation matrix is not orthogonal (R^T * R != I). "
            "This may cause projection errors."
        )

    return len(warnings) == 0, warnings
