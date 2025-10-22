"""Pydantic schemas for pipeline configuration validation.

This module defines the configuration data models that validate YAML/JSON
pipeline configurations before pipeline construction.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator


class StageConfigItem(BaseModel):
    """Configuration for a single pipeline stage.

    Attributes:
        name: Unique stage identifier within pipeline
        stage_type: Class name of the stage implementation
        config: Stage-specific parameters as dict
        enabled: Whether to execute this stage (default True)
    """

    name: str = Field(..., description="Unique stage name")
    stage_type: str = Field(..., description="Stage class name")
    config: Dict[str, Any] = Field(
        default_factory=dict, description="Stage-specific parameters"
    )
    enabled: bool = Field(default=True, description="Whether to execute stage")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate stage name is non-empty and alphanumeric."""
        if not v or not v.strip():
            raise ValueError("Stage name cannot be empty")

        # Allow alphanumeric, hyphens, underscores
        cleaned = v.replace("-", "").replace("_", "")
        if not cleaned.isalnum():
            raise ValueError(
                "Stage name must be alphanumeric (hyphens and underscores allowed)"
            )

        return v.strip()

    @field_validator("stage_type")
    @classmethod
    def validate_stage_type(cls, v: str) -> str:
        """Validate stage_type is non-empty."""
        if not v or not v.strip():
            raise ValueError("stage_type cannot be empty")
        return v.strip()


class ConnectionItem(BaseModel):
    """Configuration for a stage connection.

    Attributes:
        source: Source stage name
        target: Target stage name
        output_key: Output key from source stage
        input_key: Input key for target stage
        required: Whether connection must succeed (default True)
    """

    source: str = Field(..., description="Source stage name")
    target: str = Field(..., description="Target stage name")
    output_key: str = Field(..., description="Output key from source")
    input_key: str = Field(..., description="Input key for target")
    required: bool = Field(default=True, description="Connection required")

    @field_validator("source", "target", "output_key", "input_key")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        """Validate field is non-empty."""
        if not v or not v.strip():
            raise ValueError("Connection fields cannot be empty")
        return v.strip()


class OutputSaveConfig(BaseModel):
    """Configuration for saving stage outputs."""

    enabled: bool = Field(default=True, description="Enable saving stage outputs")
    stages: List[str] = Field(
        default=["all"], description="Stage names to save outputs for"
    )
    max_samples: Optional[int] = Field(
        default=10, description="Maximum samples to save (None = unlimited)"
    )
    format: str = Field(default="png", description="Output image format")


class BatchConfigItem(BaseModel):
    """Configuration for batch processing.

    Attributes:
        input_dir: Directory containing input images
        output_dir: Directory for processed outputs
        file_patterns: List of glob patterns for image files
        recursive: Whether to search subdirectories
        continue_on_error: Continue processing if one image fails
        report_output: Filename for HTML report
        output_mode: Output saving mode (sample, validation, production)
        save_outputs: Output saving configuration
    """

    input_dir: str = Field(..., description="Input directory path")
    output_dir: str = Field(default="./output", description="Output directory path")
    file_patterns: List[str] = Field(
        default=["*.png", "*.jpg", "*.jpeg"], description="Image file patterns"
    )
    recursive: bool = Field(default=True, description="Recursive directory search")
    continue_on_error: bool = Field(
        default=True, description="Continue on failure"
    )
    report_output: str = Field(
        default="report.html", description="HTML report filename"
    )
    output_mode: str = Field(
        default="sample", description="Output mode: sample, validation, or production"
    )
    save_outputs: OutputSaveConfig = Field(
        default_factory=OutputSaveConfig, description="Output saving configuration"
    )

    @field_validator("input_dir", "output_dir")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Validate path is non-empty."""
        if not v or not v.strip():
            raise ValueError("Path cannot be empty")
        return v.strip()

    @field_validator("file_patterns")
    @classmethod
    def validate_patterns(cls, v: List[str]) -> List[str]:
        """Validate file patterns list is not empty."""
        if not v:
            raise ValueError("file_patterns cannot be empty")
        return v


class PipelineConfig(BaseModel):
    """Complete pipeline configuration.

    This is the top-level configuration model that encompasses all
    pipeline settings including stages, connections, and batch config.

    Attributes:
        pipeline_name: Unique pipeline identifier
        stages: List of stage configurations
        connections: List of stage connections
        batch_config: Optional batch processing configuration
        output_dir: Default output directory
    """

    pipeline_name: str = Field(..., description="Pipeline identifier")
    stages: List[StageConfigItem] = Field(..., description="Stage definitions")
    connections: List[ConnectionItem] = Field(
        default_factory=list, description="Stage connections"
    )
    batch_config: Optional[BatchConfigItem] = Field(
        default=None, description="Batch processing settings"
    )
    output_dir: str = Field(default="./output", description="Default output directory")

    @field_validator("pipeline_name")
    @classmethod
    def validate_pipeline_name(cls, v: str) -> str:
        """Validate pipeline name is non-empty."""
        if not v or not v.strip():
            raise ValueError("pipeline_name cannot be empty")
        return v.strip()

    @field_validator("stages")
    @classmethod
    def validate_unique_stage_names(cls, v: List[StageConfigItem]) -> List[StageConfigItem]:
        """Validate all stage names are unique."""
        if not v:
            raise ValueError("stages list cannot be empty")

        names = [stage.name for stage in v]
        if len(names) != len(set(names)):
            duplicates = [name for name in names if names.count(name) > 1]
            raise ValueError(
                f"Stage names must be unique. Duplicates found: {set(duplicates)}"
            )

        return v

    @field_validator("connections")
    @classmethod
    def validate_connections_reference_stages(cls, v: List[ConnectionItem], info) -> List[ConnectionItem]:
        """Validate connections reference existing stages."""
        # Get stage names from the stages field
        stages_data = info.data.get("stages", [])
        if not stages_data:
            # If stages not yet validated, skip this check
            return v

        stage_names = {stage.name for stage in stages_data}

        for conn in v:
            if conn.source not in stage_names:
                raise ValueError(
                    f"Connection source '{conn.source}' not found in stages"
                )
            if conn.target not in stage_names:
                raise ValueError(
                    f"Connection target '{conn.target}' not found in stages"
                )

        return v
