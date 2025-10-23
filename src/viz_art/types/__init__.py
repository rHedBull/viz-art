"""Type definitions for viz-art multi-modal processing."""

from .pointcloud import (
    PointArray,
    ColorArray,
    NormalArray,
    IntensityArray,
    ImageArray,
    PointCloudObject,
    CoordinateSystem,
    ColorMode,
    Viewpoint,
    RenderMode,
    PointCloud,
    CameraIntrinsics,
    CameraExtrinsics,
    Calibration,
    PointCloudValidationRules,
    VisualizationConfig,
    MultiModalInput,
)

# Phase 3: Monitoring and debugging types
from .monitoring import (
    RunStatus,
    LogLevel,
    AnnotationFormat,
    Run,
    StageExecution,
    AuditLog,
    GroundTruthDataset,
    AccuracyMetrics,
)

__all__ = [
    # Point cloud types
    "PointArray",
    "ColorArray",
    "NormalArray",
    "IntensityArray",
    "ImageArray",
    "PointCloudObject",
    "CoordinateSystem",
    "ColorMode",
    "Viewpoint",
    "RenderMode",
    "PointCloud",
    "CameraIntrinsics",
    "CameraExtrinsics",
    "Calibration",
    "PointCloudValidationRules",
    "VisualizationConfig",
    "MultiModalInput",
    # Monitoring types
    "RunStatus",
    "LogLevel",
    "AnnotationFormat",
    "Run",
    "StageExecution",
    "AuditLog",
    "GroundTruthDataset",
    "AccuracyMetrics",
]
