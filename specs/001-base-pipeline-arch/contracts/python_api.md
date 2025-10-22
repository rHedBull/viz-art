# Python API Contract: Base Pipeline Architecture

**Feature**: 001-base-pipeline-arch
**Date**: 2025-10-22
**Contract Type**: Python Library API

## Overview

This document defines the public Python API contracts for the viz-art base pipeline architecture. These interfaces form the stable API that users will program against.

---

## 1. PipelineStage Base Class

**Location**: `viz_art.pipeline.stage`

**Purpose**: Abstract base class that all pipeline stages must inherit from

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class PipelineStage(ABC):
    """Abstract base class for all pipeline stages.

    All stages must inherit from this class and implement all abstract methods.
    Stages process data through three phases: pre_process, predict, post_process.
    Each stage declares its input and output keys for data flow validation.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name of this stage within the pipeline."""
        ...

    @property
    @abstractmethod
    def input_keys(self) -> List[str]:
        """List of required input data keys (e.g., ['image', 'metadata'])."""
        ...

    @property
    @abstractmethod
    def output_keys(self) -> List[str]:
        """List of output data keys this stage produces (e.g., ['filtered_image'])."""
        ...

    @abstractmethod
    def pre_process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-processing step before main prediction.

        Args:
            inputs: Dictionary of input data (keys match self.input_keys)

        Returns:
            Dictionary with processed data for predict()

        Raises:
            ValueError: If required inputs missing or invalid
        """
        ...

    @abstractmethod
    def predict(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        """Main processing step (prediction, transformation, etc.).

        Args:
            preprocessed: Output from pre_process()

        Returns:
            Dictionary with prediction results
        """
        ...

    @abstractmethod
    def post_process(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Post-processing step to finalize outputs.

        Args:
            predictions: Output from predict()

        Returns:
            Dictionary with keys matching self.output_keys
        """
        ...
```

**Contract Guarantees**:
- Stages **must inherit** from PipelineStage (enforced by ABC)
- Stages are **stateless** between runs (no state persisted between executions)
- Each method receives dict input and returns dict output
- Output keys MUST match declared `output_keys`
- Exceptions in any method mark the stage as FAILED
- All methods are **synchronous** (no async in Phase 1)
- **Compile-time validation**: Cannot instantiate class without implementing all abstract methods

---

## 2. Pipeline Class

**Location**: `viz_art.pipeline.base`

**Purpose**: Orchestrates execution of connected stages

```python
from typing import List, Dict, Any, Optional
from viz_art.pipeline.stage import PipelineStage
from viz_art.config.schema import PipelineConfig

class Pipeline:
    """Main pipeline orchestrator.

    Example:
        >>> pipeline = Pipeline(name="my-pipeline")
        >>> pipeline.add_stage(ImageLoader(name="loader"))
        >>> pipeline.add_stage(SimpleFilter(name="filter"))
        >>> pipeline.connect("loader", "filter", output_key="image", input_key="image")
        >>> results = pipeline.run(image_path="/path/to/image.jpg")
    """

    def __init__(
        self,
        name: str,
        config: Optional[PipelineConfig] = None
    ):
        """Initialize pipeline.

        Args:
            name: Unique pipeline identifier
            config: Optional configuration object
        """
        ...

    def add_stage(self, stage: PipelineStage) -> None:
        """Add a stage to the pipeline.

        Args:
            stage: PipelineStage instance to add

        Raises:
            ValueError: If stage name conflicts with existing stage
        """
        ...

    def connect(
        self,
        source_stage: str,
        target_stage: str,
        output_key: str,
        input_key: str,
        required: bool = True
    ) -> None:
        """Connect output from source stage to input of target stage.

        Args:
            source_stage: Name of stage producing output
            target_stage: Name of stage consuming input
            output_key: Key in source stage outputs
            input_key: Key in target stage inputs
            required: Whether connection must succeed (default True)

        Raises:
            ValueError: If stages don't exist or keys invalid
            ValueError: If connection creates circular dependency
        """
        ...

    def run(self, **inputs) -> Dict[str, Any]:
        """Execute the pipeline with provided inputs.

        Args:
            **inputs: Keyword arguments providing initial data
                     (e.g., image_path="/path/to/img.jpg")

        Returns:
            Dictionary containing all stage outputs:
            {
                "loader": {"image": <array>},
                "filter": {"filtered_image": <array>},
                "_run_id": "uuid-string",
                "_status": "COMPLETED"
            }

        Raises:
            ValueError: If required inputs missing
            RuntimeError: If pipeline execution fails
        """
        ...

    @classmethod
    def from_config(cls, config: PipelineConfig) -> "Pipeline":
        """Construct pipeline from configuration object.

        Args:
            config: PipelineConfig loaded from YAML

        Returns:
            Configured Pipeline instance with stages and connections

        Raises:
            ValueError: If config invalid or stages can't be instantiated
        """
        ...
```

**Contract Guarantees**:
- Stages execute in **order added**
- Stage outputs accumulated in results dict
- Failed stage stops execution (unless error handling configured)
- Run returns complete execution trace
- Pipeline is **reusable** (can call `run()` multiple times)

---

## 3. Configuration Schema

**Location**: `viz_art.config.schema`

**Purpose**: Pydantic models for configuration validation

```python
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional

class StageConfigItem(BaseModel):
    """Configuration for a single stage."""

    name: str = Field(..., description="Unique stage name")
    stage_type: str = Field(..., description="Stage class name")
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Stage-specific parameters"
    )
    enabled: bool = Field(default=True, description="Whether to execute stage")

    @validator("name")
    def validate_name(cls, v):
        if not v or not v.replace("-", "").replace("_", "").isalnum():
            raise ValueError("Stage name must be non-empty alphanumeric")
        return v

class ConnectionItem(BaseModel):
    """Configuration for stage connection."""

    source: str = Field(..., description="Source stage name")
    target: str = Field(..., description="Target stage name")
    output_key: str = Field(..., description="Output key from source")
    input_key: str = Field(..., description="Input key for target")
    required: bool = Field(default=True, description="Connection required")

class BatchConfigItem(BaseModel):
    """Configuration for batch processing."""

    input_dir: str = Field(..., description="Directory containing images")
    output_dir: str = Field(default="./output", description="Output directory")
    file_patterns: List[str] = Field(
        default=["*.png", "*.jpg", "*.jpeg"],
        description="Image file patterns"
    )
    recursive: bool = Field(default=True, description="Recursive search")
    continue_on_error: bool = Field(
        default=True,
        description="Continue processing on failure"
    )
    report_output: str = Field(
        default="report.html",
        description="HTML report filename"
    )

class PipelineConfig(BaseModel):
    """Complete pipeline configuration."""

    pipeline_name: str = Field(..., description="Pipeline identifier")
    stages: List[StageConfigItem] = Field(..., description="Stage definitions")
    connections: List[ConnectionItem] = Field(
        default_factory=list,
        description="Stage connections"
    )
    batch_config: Optional[BatchConfigItem] = Field(
        default=None,
        description="Batch processing settings"
    )
    output_dir: str = Field(default="./output", description="Output directory")

    @validator("stages")
    def validate_unique_stage_names(cls, v):
        names = [s.name for s in v]
        if len(names) != len(set(names)):
            raise ValueError("Stage names must be unique")
        return v
```

**Contract Guarantees**:
- All validation errors are **Pydantic ValidationError** with detailed messages
- Config loadable from YAML via OmegaConf
- Nested configs support variable interpolation
- Invalid configs caught **before** pipeline creation

---

## 4. Configuration Loader

**Location**: `viz_art.config.loader`

**Purpose**: Load and validate YAML configurations

```python
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from viz_art.config.schema import PipelineConfig

def load_config(config_path: str | Path) -> PipelineConfig:
    """Load and validate pipeline configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Validated PipelineConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If YAML invalid or validation fails

    Example:
        >>> config = load_config("configs/my_pipeline.yaml")
        >>> pipeline = Pipeline.from_config(config)
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load with OmegaConf
    omega_cfg: DictConfig = OmegaConf.load(path)

    # Convert to dict and validate with Pydantic
    cfg_dict = OmegaConf.to_container(omega_cfg, resolve=True)

    return PipelineConfig.model_validate(cfg_dict)
```

**Contract Guarantees**:
- Supports OmegaConf features (interpolation, merging)
- Clear error messages with **line numbers** for YAML syntax errors
- Pydantic validation errors show **field paths**
- Config is **immutable** after loading

---

## 5. Batch Processor

**Location**: `viz_art.batch.processor`

**Purpose**: Process multiple images through pipeline

```python
from pathlib import Path
from typing import Generator, Tuple
from viz_art.pipeline.base import Pipeline
from viz_art.config.schema import BatchConfigItem

class BatchProcessor:
    """Process multiple images through a pipeline.

    Example:
        >>> batch_config = BatchConfigItem(
        ...     input_dir="./images",
        ...     output_dir="./output"
        ... )
        >>> processor = BatchProcessor(pipeline, batch_config)
        >>> result = processor.run()
        >>> print(f"Processed {result.successful}/{result.total_files} images")
    """

    def __init__(
        self,
        pipeline: Pipeline,
        config: BatchConfigItem
    ):
        """Initialize batch processor.

        Args:
            pipeline: Configured Pipeline instance
            config: Batch processing configuration
        """
        ...

    def discover_images(self) -> Generator[Path, None, None]:
        """Discover image files in input directory.

        Yields:
            Path objects for discovered image files

        Note:
            Non-image files are silently skipped (no logging)
        """
        ...

    def run(self) -> "BatchResult":
        """Execute batch processing.

        Returns:
            BatchResult with statistics and per-image results

        Process:
            1. Discover images using configured patterns
            2. Process each image through pipeline
            3. Continue on error if configured
            4. Generate HTML report
            5. Return aggregated results
        """
        ...
```

**Contract Guarantees**:
- **Generator-based** for memory efficiency
- Errors in one image **don't stop batch** (if continue_on_error=True)
- Each image processed **independently**
- HTML report generated **after** all processing complete
- Results include both successes and failures

---

## 6. HTML Report Generator

**Location**: `viz_art.batch.reporter`

**Purpose**: Generate static HTML reports from batch results

```python
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from viz_art.batch.processor import BatchResult

class HTMLReporter:
    """Generate static HTML reports from batch processing results.

    Example:
        >>> reporter = HTMLReporter(template_dir="templates")
        >>> reporter.generate(batch_result, output_path="report.html")
    """

    def __init__(
        self,
        template_dir: str | Path | None = None
    ):
        """Initialize HTML reporter.

        Args:
            template_dir: Directory containing Jinja2 templates
                         (defaults to package templates)
        """
        ...

    def generate(
        self,
        batch_result: "BatchResult",
        output_path: str | Path
    ) -> Path:
        """Generate HTML report from batch results.

        Args:
            batch_result: BatchResult from BatchProcessor.run()
            output_path: Path for output HTML file

        Returns:
            Path to generated HTML file (absolute)

        Report Structure:
            - Summary statistics (total, successful, failed)
            - Stage-grouped view (all images per stage)
            - Per-image complete pipeline view
            - Error section with failure details
        """
        ...
```

**Contract Guarantees**:
- HTML is **self-contained** (inline CSS, relative image paths)
- Images referenced via **relative paths** (not base64)
- Report viewable in **any modern browser** without web server
- Stage-grouped layout as specified in requirements
- Failed images clearly marked with error messages

---

## 7. Result Objects

**Location**: `viz_art.pipeline.results`

**Purpose**: Data classes for execution results

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum

class RunStatus(Enum):
    """Pipeline run status."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class StageStatus(Enum):
    """Stage execution status."""
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass(frozen=True)
class StageResult:
    """Result from a single stage execution."""
    stage_name: str
    status: StageStatus
    started_at: datetime
    duration_ms: float
    outputs: Dict[str, Any]
    error: Optional[str] = None

@dataclass(frozen=True)
class PipelineRun:
    """Result from a complete pipeline execution."""
    run_id: str
    pipeline_name: str
    started_at: datetime
    completed_at: datetime
    status: RunStatus
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    stage_results: List[StageResult]
    error: Optional[str] = None

@dataclass(frozen=True)
class BatchResult:
    """Result from batch processing execution."""
    batch_id: str
    total_files: int
    successful: int
    failed: int
    run_results: List[PipelineRun]
    started_at: datetime
    completed_at: datetime
    report_path: str
```

**Contract Guarantees**:
- All result objects are **immutable** (frozen dataclasses)
- Timestamps always in **UTC**
- Results are **JSON-serializable**
- Duration always in **milliseconds** for consistency

---

## API Usage Examples

### Example 1: Basic Pipeline Execution

```python
from viz_art.pipeline.base import Pipeline
from viz_art.pipeline.stage import PipelineStage
from typing import Dict, Any, List

# Define custom stage (must inherit from PipelineStage)
class ImageLoader(PipelineStage):
    def __init__(self, name: str = "loader"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def input_keys(self) -> List[str]:
        return ["image_path"]

    @property
    def output_keys(self) -> List[str]:
        return ["image"]

    def pre_process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs

    def predict(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        from PIL import Image
        import numpy as np
        img = Image.open(preprocessed["image_path"])
        return {"image_array": np.array(img)}

    def post_process(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        return {"image": predictions["image_array"]}

# Build pipeline
pipeline = Pipeline(name="simple")
pipeline.add_stage(ImageLoader())

# Execute
results = pipeline.run(image_path="test.jpg")
print(results["loader"]["image"].shape)
```

### Example 2: Configuration-Based Pipeline

```python
from viz_art.config.loader import load_config
from viz_art.pipeline.base import Pipeline

# Load config
config = load_config("configs/my_pipeline.yaml")

# Create pipeline from config
pipeline = Pipeline.from_config(config)

# Execute
results = pipeline.run(image_path="input.jpg")
```

### Example 3: Batch Processing

```python
from viz_art.config.loader import load_config
from viz_art.pipeline.base import Pipeline
from viz_art.batch.processor import BatchProcessor

# Setup
config = load_config("configs/batch_config.yaml")
pipeline = Pipeline.from_config(config)

# Batch process
processor = BatchProcessor(pipeline, config.batch_config)
result = processor.run()

print(f"Processed: {result.successful}/{result.total_files}")
print(f"Report: {result.report_path}")
```

---

## Contract Stability

### Guaranteed Stable (Public API)
- `Pipeline` class and methods
- `PipelineStage` protocol
- Configuration schema classes
- `load_config()` function
- `BatchProcessor` class
- Result dataclasses

### Internal (May Change)
- Connection management internals
- Stage registration mechanism
- Template rendering details
- File discovery implementation

---

## Error Handling Contract

### ValueError
- Invalid configuration
- Missing required inputs
- Invalid stage connections
- Circular dependencies

### RuntimeError
- Pipeline execution failure
- Stage processing error
- File I/O errors during batch processing

### Pydantic ValidationError
- Configuration schema validation failures
- Clear field-level error messages

**User Code Should Catch**:
```python
from pydantic import ValidationError

try:
    config = load_config("config.yaml")
    pipeline = Pipeline.from_config(config)
    results = pipeline.run(image_path="test.jpg")
except ValidationError as e:
    print(f"Config validation failed: {e}")
except ValueError as e:
    print(f"Invalid input: {e}")
except RuntimeError as e:
    print(f"Execution failed: {e}")
```

---

## Versioning

**API Version**: 1.0.0 (Phase 1)

**Compatibility Promise**:
- Public API stable within major version
- Deprecation warnings before breaking changes
- Internal APIs may change between minor versions

**Future API Extensions** (Phase 2+):
- Point cloud data type support
- Async stage execution
- GPU acceleration options
- Advanced visualization methods
