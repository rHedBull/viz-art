# Data Model: Base Pipeline Architecture

**Feature**: 001-base-pipeline-arch
**Date**: 2025-10-22
**Status**: Phase 1 Design

## Overview

This document defines the core data entities, their relationships, validation rules, and state transitions for the base pipeline architecture. The model supports sequential stage execution, configuration-driven pipeline construction, and batch processing with reporting.

---

## Core Entities

### 1. Pipeline

**Description**: Orchestrates execution of connected stages in sequence

**Fields**:
- `name: str` - Unique identifier for the pipeline
- `stages: List[PipelineStage]` - Ordered list of stages to execute
- `config: PipelineConfig` - Configuration for pipeline behavior
- `created_at: datetime` - Pipeline creation timestamp

**Validation Rules**:
- `name` must be non-empty string, alphanumeric with hyphens/underscores
- `stages` must contain at least 1 stage
- Stage execution order determined by list position
- All stage names within pipeline must be unique

**State Transitions**:
```
INITIALIZED → RUNNING → (COMPLETED | FAILED)
```

**Relationships**:
- Contains 1+ PipelineStage instances
- Has 1 PipelineConfig
- Produces 1 PipelineRun per execution

---

### 2. PipelineStage

**Description**: Individual processing unit with standard interface methods

**Fields**:
- `name: str` - Unique stage identifier within pipeline
- `stage_type: str` - Stage class name (e.g., "ImageLoader", "SimpleFilter")
- `config: StageConfig` - Stage-specific configuration parameters
- `input_keys: List[str]` - Expected input data keys
- `output_keys: List[str]` - Produced output data keys

**Validation Rules**:
- `name` must be non-empty, unique within pipeline
- `stage_type` must reference a registered stage class
- `input_keys` must be provided by previous stages or pipeline inputs
- `output_keys` must be unique within pipeline outputs
- Stage must implement protocol: `pre_process()`, `predict()`, `post_process()`

**State Transitions**:
```
PENDING → PRE_PROCESSING → PREDICTING → POST_PROCESSING → (COMPLETED | FAILED)
```

**Relationships**:
- Belongs to 1 Pipeline
- Has 1 StageConfig
- Connected to other stages via StageConnection
- Produces StageOutput per execution

---

### 3. StageConnection

**Description**: Defines data flow between stages

**Fields**:
- `source_stage: str` - Name of stage producing output
- `target_stage: str` - Name of stage consuming input
- `output_key: str` - Key in source stage outputs
- `input_key: str` - Key in target stage inputs
- `required: bool = True` - Whether connection must succeed

**Validation Rules**:
- `source_stage` must exist in pipeline
- `target_stage` must exist in pipeline
- `output_key` must be in source stage's `output_keys`
- `input_key` must be in target stage's `input_keys`
- Source stage must execute before target stage
- No circular dependencies allowed

**Relationships**:
- References 2 PipelineStage instances (source and target)
- Enforced by Pipeline during construction

---

### 4. PipelineConfig

**Description**: Configuration specification for pipeline and stages

**Fields**:
- `pipeline_name: str` - Name of pipeline being configured
- `stages: List[StageConfig]` - Configuration for each stage
- `connections: List[Dict[str, str]]` - Stage connection definitions
- `batch_config: Optional[BatchConfig]` - Batch processing settings
- `output_dir: str = "./output"` - Directory for outputs

**Validation Rules**:
- Must be loadable from YAML/JSON
- All referenced stage names must match defined stages
- Connections must reference existing stages
- `output_dir` must be valid path or creatable directory
- Validated against Pydantic schema before pipeline creation

**Source**:
- Loaded from YAML files via OmegaConf
- Converted to native dict and validated with Pydantic

**Example Structure**:
```yaml
pipeline_name: "example-pipeline"
stages:
  - name: "loader"
    stage_type: "ImageLoader"
    config:
      resize: [640, 480]
  - name: "filter"
    stage_type: "SimpleFilter"
    config:
      threshold: 0.5
connections:
  - source: "loader"
    target: "filter"
    output_key: "image"
    input_key: "image"
output_dir: "./output"
```

---

### 5. StageConfig

**Description**: Stage-specific configuration parameters

**Fields**:
- `name: str` - Stage name (matches PipelineStage.name)
- `stage_type: str` - Class name of stage implementation
- `config: Dict[str, Any]` - Stage-specific parameters
- `enabled: bool = True` - Whether to execute this stage

**Validation Rules**:
- `name` must match stage name in pipeline definition
- `stage_type` must be registered/importable
- `config` dict validated against stage's expected schema
- Invalid parameters caught during config loading

---

### 6. PipelineRun

**Description**: A single execution instance of the pipeline

**Fields**:
- `run_id: str` - Unique identifier (UUID)
- `pipeline_name: str` - Name of executed pipeline
- `started_at: datetime` - Execution start time
- `completed_at: Optional[datetime]` - Execution end time
- `status: RunStatus` - Enum: RUNNING, COMPLETED, FAILED
- `inputs: Dict[str, Any]` - Input data provided to run()
- `outputs: Dict[str, Any]` - All stage outputs
- `stage_results: List[StageResult]` - Per-stage execution details
- `error: Optional[str]` - Error message if failed

**Validation Rules**:
- `run_id` must be unique UUID
- `started_at` always set when run created
- `completed_at` only set when status is COMPLETED or FAILED
- `outputs` contains outputs from all successfully executed stages
- `stage_results` length matches number of attempted stages

**State Transitions**:
```
RUNNING → (COMPLETED | FAILED)
```

**Relationships**:
- References 1 Pipeline
- Contains N StageResult instances (one per stage)
- Used by BatchProcessor to track individual image runs

---

### 7. StageResult

**Description**: Execution details for a single stage in a run

**Fields**:
- `stage_name: str` - Name of executed stage
- `status: StageStatus` - Enum: SUCCESS, FAILED, SKIPPED
- `started_at: datetime` - Stage start time
- `duration_ms: float` - Execution time in milliseconds
- `outputs: Dict[str, Any]` - Stage-produced outputs
- `error: Optional[str]` - Error message if failed

**Validation Rules**:
- `duration_ms` must be >= 0
- `error` must be set if status is FAILED
- `outputs` empty if status is FAILED or SKIPPED

**Relationships**:
- Belongs to 1 PipelineRun
- References 1 PipelineStage

---

### 8. BatchConfig

**Description**: Configuration for batch processing

**Fields**:
- `input_dir: str` - Root directory containing images
- `output_dir: str` - Directory for processed outputs
- `file_patterns: List[str] = ["*.png", "*.jpg", "*.jpeg"]` - Image file patterns
- `recursive: bool = True` - Search subdirectories
- `continue_on_error: bool = True` - Keep processing on failure
- `report_output: str = "report.html"` - HTML report filename

**Validation Rules**:
- `input_dir` must be existing directory
- `output_dir` must be writable or creatable
- `file_patterns` must be valid glob patterns
- `report_output` must be valid filename

---

### 9. BatchResult

**Description**: Results from batch processing execution

**Fields**:
- `batch_id: str` - Unique batch execution ID
- `total_files: int` - Total images discovered
- `successful: int` - Successfully processed images
- `failed: int` - Failed image count
- `skipped: int` - Skipped non-image files (not logged)
- `run_results: List[PipelineRun]` - Individual run results
- `started_at: datetime` - Batch start time
- `completed_at: datetime` - Batch end time
- `report_path: str` - Path to generated HTML report

**Validation Rules**:
- `total_files = successful + failed` (skipped not counted)
- `run_results` length equals `total_files`
- `report_path` must exist after batch completion

**Relationships**:
- Contains N PipelineRun instances
- Produces 1 HTMLReport

---

### 10. HTMLReport

**Description**: Generated static HTML visualization of batch results

**Fields**:
- `report_path: str` - Absolute path to HTML file
- `batch_result: BatchResult` - Associated batch execution
- `template_name: str = "batch_report.html"` - Jinja2 template used
- `generated_at: datetime` - Report generation timestamp

**Structure**:
- Stage-grouped view: All images grouped by stage output
- Per-image section: Complete pipeline outputs per image
- Error section: Failed images with error details
- Uses relative file references to images (not base64)

**Relationships**:
- References 1 BatchResult
- Contains references to N output images

---

## Entity Relationship Diagram

```
Pipeline (1) ----< (N) PipelineStage
    |                       |
    |                       |
    | (1)             (1)   |
    |                       |
    v                       v
PipelineConfig         StageConfig
    |
    | (1)
    |
    v
BatchConfig

Pipeline (1) ----< (N) PipelineRun
                        |
                        | (1)
                        |
                        v
                   (N) StageResult

PipelineRun (N) ----< (1) BatchResult
                              |
                              | (1)
                              |
                              v
                         HTMLReport

PipelineStage (source) ----< (1) StageConnection >---- (1) PipelineStage (target)
```

---

## Key Design Decisions

### 1. Immutable Execution Results
- `PipelineRun` and `StageResult` are created once and not modified
- Supports auditing and debugging
- Clear separation between configuration (mutable) and results (immutable)

### 2. Flexible Configuration
- Stage-specific config in nested `config` dict
- Allows any JSON-serializable parameters
- Validated by stage implementation, not framework

### 3. Error Handling Strategy
- `continue_on_error=True` for batch processing
- Each `PipelineRun` independent
- Failed runs tracked but don't stop batch

### 4. No Database Required
- All entities serializable to JSON/YAML
- Results stored as files for simplicity
- Sufficient for MVP use case

---

## Validation Summary

| Entity | Primary Validation | Secondary Validation |
|--------|-------------------|---------------------|
| Pipeline | Name uniqueness, stage presence | Stage order, no cycles |
| PipelineStage | Protocol compliance, name uniqueness | Input/output key validity |
| StageConnection | Stage existence, key matching | No circular dependencies |
| PipelineConfig | Schema validation (Pydantic) | Path validity, stage references |
| PipelineRun | UUID uniqueness, status consistency | Timestamp ordering |
| BatchResult | Count consistency | Report generation success |

---

## Notes

- All datetime fields use UTC timezone
- File paths stored as absolute paths for reliability
- Image data passed by reference (numpy arrays), not copied
- Configuration supports OmegaConf variable interpolation
- Future phases may add: point cloud support, parallel execution, advanced metrics
