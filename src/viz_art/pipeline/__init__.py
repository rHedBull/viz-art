"""Pipeline module for viz-art.

This module provides the core pipeline functionality for orchestrating
multi-stage processing workflows.
"""

from viz_art.pipeline.base import Pipeline
from viz_art.pipeline.stage import PipelineStage
from viz_art.pipeline.connection import StageConnection
from viz_art.pipeline.results import (
    StageStatus,
    RunStatus,
    StageResult,
    PipelineRun,
    BatchResult,
)

__all__ = [
    "Pipeline",
    "PipelineStage",
    "StageConnection",
    "StageStatus",
    "RunStatus",
    "StageResult",
    "PipelineRun",
    "BatchResult",
]
