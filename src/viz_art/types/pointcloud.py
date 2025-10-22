"""Type definitions for point cloud processing and multi-modal visualization."""

from dataclasses import dataclass, field
from typing import TypeAlias, Literal, Dict, Tuple, List, Optional
import numpy as np
import numpy.typing as npt

# Type aliases for NumPy arrays
PointArray: TypeAlias = npt.NDArray[np.float32]  # Nx3 XYZ coordinates
ColorArray: TypeAlias = npt.NDArray[np.float32]  # Nx3 RGB values in [0,1]
NormalArray: TypeAlias = npt.NDArray[np.float32]  # Nx3 surface normals
IntensityArray: TypeAlias = npt.NDArray[np.float32]  # N reflectivity values
ImageArray: TypeAlias = npt.NDArray[np.uint8]  # HxWx3 image data

# Deferred import for Open3D to avoid import errors before installation
PointCloudObject: TypeAlias = "open3d.geometry.PointCloud"  # type: ignore

# Literal types for enumerations
CoordinateSystem: TypeAlias = Literal["camera", "lidar", "world"]
ColorMode: TypeAlias = Literal["height", "intensity", "rgb", "class"]
Viewpoint: TypeAlias = Literal["front", "top", "side", "diagonal"]
RenderMode: TypeAlias = Literal["webgl", "svg"]


@dataclass(frozen=True)
class PointCloud:
    """Immutable point cloud data structure.

    Attributes:
        points: Nx3 array of XYZ coordinates (float32)
        colors: Optional Nx3 array of RGB values in [0,1] (float32)
        normals: Optional Nx3 array of surface normals (float32)
        intensity: Optional N array of reflectivity values (float32)
        num_points: Integer count of points (auto-calculated if 0)
        coordinate_system: Coordinate frame identifier
        timestamp: ISO 8601 timestamp string
        sensor_id: Source sensor identifier
    """

    points: PointArray
    colors: Optional[ColorArray] = None
    normals: Optional[NormalArray] = None
    intensity: Optional[IntensityArray] = None
    num_points: int = 0
    coordinate_system: CoordinateSystem = "lidar"
    timestamp: str = ""
    sensor_id: str = "default"

    def __post_init__(self):
        """Validate and auto-calculate num_points."""
        if self.num_points == 0:
            object.__setattr__(self, 'num_points', len(self.points))

        # Validate array shapes
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(f"Points must be Nx3 array, got shape {self.points.shape}")

        if self.colors is not None:
            if len(self.colors) != self.num_points:
                raise ValueError(
                    f"Colors length {len(self.colors)} != points length {self.num_points}"
                )

        if self.normals is not None:
            if len(self.normals) != self.num_points:
                raise ValueError(
                    f"Normals length {len(self.normals)} != points length {self.num_points}"
                )

        if self.intensity is not None:
            if len(self.intensity) != self.num_points:
                raise ValueError(
                    f"Intensity length {len(self.intensity)} != points length {self.num_points}"
                )


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters.

    Attributes:
        fx: Focal length X (pixels)
        fy: Focal length Y (pixels)
        cx: Principal point X (pixels)
        cy: Principal point Y (pixels)
        width: Image width (pixels)
        height: Image height (pixels)
        distortion_coeffs: Distortion coefficients [k1, k2, p1, p2, k3]
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion_coeffs: List[float] = field(default_factory=lambda: [0.0] * 5)

    def __post_init__(self):
        """Validate intrinsic parameters."""
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError(f"Focal lengths must be positive, got fx={self.fx}, fy={self.fy}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Image dimensions must be positive, got {self.width}x{self.height}")
        if len(self.distortion_coeffs) != 5:
            raise ValueError(
                f"Distortion coeffs must have 5 elements, got {len(self.distortion_coeffs)}"
            )

    def to_matrix(self) -> np.ndarray:
        """Convert intrinsics to 3x3 camera matrix.

        Returns:
            3x3 NumPy array in OpenCV format
        """
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)


@dataclass
class CameraExtrinsics:
    """Camera extrinsic parameters (pose).

    Attributes:
        rotation_matrix: 3x3 rotation matrix
        translation_vector: 3x1 translation vector
    """

    rotation_matrix: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    translation_vector: np.ndarray = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))

    def __post_init__(self):
        """Validate extrinsic parameters."""
        if self.rotation_matrix.shape != (3, 3):
            raise ValueError(f"Rotation matrix must be 3x3, got {self.rotation_matrix.shape}")

        # Check if rotation matrix is orthogonal (det ≈ 1)
        det = np.linalg.det(self.rotation_matrix)
        if not (0.9 < det < 1.1):
            raise ValueError(
                f"Rotation matrix determinant should be ~1, got {det:.4f}. "
                "Matrix may not be orthogonal."
            )

        if self.translation_vector.shape not in [(3,), (3, 1)]:
            raise ValueError(
                f"Translation vector must be (3,) or (3,1), got {self.translation_vector.shape}"
            )

    def to_rodrigues_vector(self) -> np.ndarray:
        """Convert rotation matrix to Rodrigues vector.

        Returns:
            3x1 rotation vector for OpenCV

        Raises:
            ImportError: If OpenCV not available
        """
        import cv2
        rvec, _ = cv2.Rodrigues(self.rotation_matrix)
        return rvec


@dataclass
class Calibration:
    """Complete camera calibration (intrinsics + extrinsics).

    Attributes:
        camera_name: Camera identifier
        intrinsics: Camera intrinsic parameters
        extrinsics: Camera extrinsic parameters (pose)
        calibration_date: ISO 8601 date string
        calibration_method: Method used for calibration
    """

    camera_name: str
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics
    calibration_date: str = ""
    calibration_method: str = "manual"


@dataclass
class PointCloudValidationRules:
    """Configuration for point cloud data quality validation.

    Attributes:
        min_points: Minimum acceptable point count
        max_points: Maximum acceptable point count (None = unlimited)
        check_nan: Reject NaN values
        check_inf: Reject Inf values
        coord_range_min: Optional XYZ minimum bounds
        coord_range_max: Optional XYZ maximum bounds
        fail_fast: Stop on first error
        log_level: Logging level for validation messages
    """

    min_points: int = 10
    max_points: Optional[int] = None
    check_nan: bool = True
    check_inf: bool = True
    coord_range_min: Optional[Tuple[float, float, float]] = None
    coord_range_max: Optional[Tuple[float, float, float]] = None
    fail_fast: bool = False
    log_level: str = "error"

    def __post_init__(self):
        """Validate configuration."""
        if self.min_points < 1:
            raise ValueError(f"min_points must be >= 1, got {self.min_points}")

        if self.max_points is not None and self.max_points <= self.min_points:
            raise ValueError(
                f"max_points ({self.max_points}) must be > min_points ({self.min_points})"
            )

        if (self.coord_range_min is None) != (self.coord_range_max is None):
            raise ValueError("coord_range_min and coord_range_max must both be set or both None")

        if self.log_level not in ["error", "warning", "info"]:
            raise ValueError(f"log_level must be error/warning/info, got {self.log_level}")

    def validate_pointcloud(self, pcd: PointCloud) -> Tuple[bool, List[str]]:
        """Validate point cloud against rules.

        Args:
            pcd: Point cloud to validate

        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        errors = []

        # Check point count
        if pcd.num_points < self.min_points:
            errors.append(f"Too few points: {pcd.num_points} < {self.min_points}")
            if self.fail_fast:
                return False, errors

        if self.max_points and pcd.num_points > self.max_points:
            errors.append(f"Too many points: {pcd.num_points} > {self.max_points}")
            if self.fail_fast:
                return False, errors

        # Check NaN/Inf
        if self.check_nan and np.isnan(pcd.points).any():
            errors.append("NaN values detected in points")
            if self.fail_fast:
                return False, errors

        if self.check_inf and np.isinf(pcd.points).any():
            errors.append("Inf values detected in points")
            if self.fail_fast:
                return False, errors

        # Check coordinate ranges
        if self.coord_range_min and self.coord_range_max:
            mins = pcd.points.min(axis=0)
            maxs = pcd.points.max(axis=0)

            for i, (mn, mx, coord) in enumerate(
                zip(self.coord_range_min, self.coord_range_max, ['X', 'Y', 'Z'])
            ):
                if mins[i] < mn or maxs[i] > mx:
                    errors.append(
                        f"{coord} range [{mins[i]:.2f}, {maxs[i]:.2f}] outside [{mn}, {mx}]"
                    )
                    if self.fail_fast:
                        return False, errors

        return len(errors) == 0, errors


@dataclass
class VisualizationConfig:
    """Configuration for visualization display and rendering.

    Attributes:
        point_size: Size of points in pixels
        opacity: Point transparency [0,1]
        color_mode: Coloring method
        colorscale: Plotly colorscale name
        background_color: RGBA background color
        camera_projection: Camera projection type
        enable_zoom: Enable zoom controls
        enable_pan: Enable pan controls
        annotation_thickness: Annotation line thickness (pixels)
        annotation_color: RGB annotation color
        thumbnail_width: Thumbnail width (pixels)
        thumbnail_height: Thumbnail height (pixels)
        thumbnail_viewpoint: Default thumbnail viewpoint
        thumbnail_quality: Thumbnail quality setting
        auto_downsample: Enable automatic downsampling
        max_render_points: Maximum points before downsampling
        render_mode: Rendering backend
    """

    # 3D viewer
    point_size: float = 2.0
    opacity: float = 0.8
    color_mode: ColorMode = "height"
    colorscale: str = "Viridis"
    background_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    camera_projection: str = "perspective"

    # 2D viewer
    enable_zoom: bool = True
    enable_pan: bool = True
    annotation_thickness: int = 2
    annotation_color: Tuple[int, int, int] = (255, 0, 0)

    # Thumbnails
    thumbnail_width: int = 800
    thumbnail_height: int = 600
    thumbnail_viewpoint: Viewpoint = "diagonal"
    thumbnail_quality: str = "medium"

    # Performance
    auto_downsample: bool = True
    max_render_points: int = 500000
    render_mode: RenderMode = "webgl"

    def __post_init__(self):
        """Validate configuration."""
        if self.point_size <= 0:
            raise ValueError(f"point_size must be > 0, got {self.point_size}")

        if not (0 <= self.opacity <= 1):
            raise ValueError(f"opacity must be in [0,1], got {self.opacity}")

        if self.thumbnail_quality not in ["low", "medium", "high"]:
            raise ValueError(
                f"thumbnail_quality must be low/medium/high, got {self.thumbnail_quality}"
            )

    def should_downsample(self, num_points: int) -> bool:
        """Determine if downsampling needed.

        Args:
            num_points: Number of points to render

        Returns:
            True if downsampling recommended
        """
        return self.auto_downsample and num_points > self.max_render_points


@dataclass(frozen=True)
class MultiModalInput:
    """Synchronized image and point cloud data pair.

    Attributes:
        image: Optional HxWx3 image array
        pointcloud: Optional PointCloud object
        timestamp: ISO 8601 timestamp
        sync_tolerance_ms: Maximum time delta for synchronization (ms)
        calibration_ref: Reference to calibration data
        metadata: Additional key-value pairs
    """

    image: Optional[ImageArray] = None
    pointcloud: Optional[PointCloud] = None
    timestamp: str = ""
    sync_tolerance_ms: float = 100.0
    calibration_ref: str = ""
    metadata: Dict[str, any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate multi-modal input."""
        if self.image is None and self.pointcloud is None:
            raise ValueError("At least one modality (image or pointcloud) required")
