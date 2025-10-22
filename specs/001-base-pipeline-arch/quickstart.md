# Quick Start Guide: Base Pipeline Architecture

**Feature**: 001-base-pipeline-arch
**Version**: Phase 1
**Last Updated**: 2025-10-22

## Overview

This guide will get you started with the viz-art base pipeline architecture in under 10 minutes. You'll learn how to:
1. Create a simple 2-stage pipeline programmatically
2. Configure a pipeline using YAML
3. Process a batch of images and generate an HTML report

---

## Prerequisites

**Required**:
- Python 3.8 or higher
- pip or uv package manager

**Recommended**:
- Basic understanding of Python classes
- Familiarity with YAML syntax

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/viz-art.git
cd viz-art

# Install dependencies (using uv)
uv sync

# Or using pip
pip install -e .
```

---

## Quick Start 1: Create Your First Pipeline (5 minutes)

### Step 1: Define a Custom Stage

Create a file `my_stages.py`:

```python
from typing import Dict, Any, List
import numpy as np
from PIL import Image


class ImageLoader:
    """Stage that loads and preprocesses images."""

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
        return ["image", "metadata"]

    def pre_process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Validate input
        if "image_path" not in inputs:
            raise ValueError("Missing required input: image_path")
        return inputs

    def predict(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        # Load image
        img_path = preprocessed["image_path"]
        img = Image.open(img_path).convert("RGB")

        # Convert to numpy array
        img_array = np.array(img)

        return {
            "image_array": img_array,
            "shape": img_array.shape,
            "path": img_path
        }

    def post_process(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "image": predictions["image_array"],
            "metadata": {
                "shape": predictions["shape"],
                "source": predictions["path"]
            }
        }


class SimpleFilter:
    """Stage that applies a simple blur filter."""

    def __init__(self, name: str = "filter", kernel_size: int = 5):
        self._name = name
        self.kernel_size = kernel_size

    @property
    def name(self) -> str:
        return self._name

    @property
    def input_keys(self) -> List[str]:
        return ["image"]

    @property
    def output_keys(self) -> List[str]:
        return ["filtered_image"]

    def pre_process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if "image" not in inputs:
            raise ValueError("Missing required input: image")
        return inputs

    def predict(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        import cv2

        # Apply Gaussian blur
        img = preprocessed["image"]
        blurred = cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), 0)

        return {"filtered": blurred}

    def post_process(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        return {"filtered_image": predictions["filtered"]}
```

### Step 2: Build and Run the Pipeline

Create `run_pipeline.py`:

```python
from viz_art.pipeline.base import Pipeline
from my_stages import ImageLoader, SimpleFilter

# Create pipeline
pipeline = Pipeline(name="my-first-pipeline")

# Add stages
loader = ImageLoader(name="loader")
filter_stage = SimpleFilter(name="filter", kernel_size=7)

pipeline.add_stage(loader)
pipeline.add_stage(filter_stage)

# Connect stages
pipeline.connect(
    source_stage="loader",
    target_stage="filter",
    output_key="image",
    input_key="image"
)

# Run pipeline
results = pipeline.run(image_path="test_image.jpg")

# Access results
print(f"Loaded image shape: {results['loader']['metadata']['shape']}")
print(f"Filtered image shape: {results['filter']['filtered_image'].shape}")
print(f"Run ID: {results['_run_id']}")
print(f"Status: {results['_status']}")
```

### Step 3: Execute

```bash
python run_pipeline.py
```

**Expected Output**:
```
Loaded image shape: (480, 640, 3)
Filtered image shape: (480, 640, 3)
Run ID: 550e8400-e29b-41d4-a716-446655440000
Status: COMPLETED
```

---

## Quick Start 2: Configuration-Based Pipeline (10 minutes)

### Step 1: Create a Configuration File

Create `configs/my_pipeline.yaml`:

```yaml
pipeline_name: "config-based-pipeline"

output_dir: "./output"

stages:
  - name: "loader"
    stage_type: "ImageLoader"
    config:
      resize: [640, 480]
      color_mode: "RGB"

  - name: "filter"
    stage_type: "SimpleFilter"
    config:
      kernel_size: 7

connections:
  - source: "loader"
    target: "filter"
    output_key: "image"
    input_key: "image"
```

### Step 2: Load and Execute

Create `run_from_config.py`:

```python
from viz_art.config.loader import load_config
from viz_art.pipeline.base import Pipeline

# Load configuration
config = load_config("configs/my_pipeline.yaml")

# Create pipeline from config
pipeline = Pipeline.from_config(config)

# Execute
results = pipeline.run(image_path="test_image.jpg")

print(f"Pipeline '{config.pipeline_name}' completed successfully!")
print(f"Stages executed: {list(results.keys())}")
```

### Step 3: Execute

```bash
python run_from_config.py
```

**Benefits of Config-Based Approach**:
- Change parameters without modifying code
- Easy experimentation
- Version control for different configurations
- Environment-specific settings

---

## Quick Start 3: Batch Processing with HTML Report (15 minutes)

### Step 1: Prepare Test Data

Create a directory structure:
```
data/
  images/
    scene1/
      img001.jpg
      img002.jpg
    scene2/
      img001.jpg
```

### Step 2: Create Batch Configuration

Create `configs/batch_config.yaml`:

```yaml
pipeline_name: "batch-processor"

output_dir: "./output/batch_results"

stages:
  - name: "loader"
    stage_type: "ImageLoader"
    config:
      resize: [800, 600]

  - name: "filter"
    stage_type: "SimpleFilter"
    config:
      kernel_size: 5

connections:
  - source: "loader"
    target: "filter"
    output_key: "image"
    input_key: "image"

batch_config:
  input_dir: "./data/images"
  output_dir: "./output/batch_results"
  file_patterns:
    - "*.jpg"
    - "*.png"
  recursive: true
  continue_on_error: true
  report_output: "report.html"
```

### Step 3: Run Batch Processing

Create `run_batch.py`:

```python
from viz_art.config.loader import load_config
from viz_art.pipeline.base import Pipeline
from viz_art.batch.processor import BatchProcessor

# Load config
config = load_config("configs/batch_config.yaml")

# Create pipeline
pipeline = Pipeline.from_config(config)

# Setup batch processor
processor = BatchProcessor(pipeline, config.batch_config)

# Execute batch processing
print("Starting batch processing...")
result = processor.run()

# Print results
print(f"\n{'='*50}")
print(f"Batch Processing Complete!")
print(f"{'='*50}")
print(f"Total images: {result.total_files}")
print(f"Successful: {result.successful}")
print(f"Failed: {result.failed}")
print(f"Duration: {(result.completed_at - result.started_at).total_seconds():.2f}s")
print(f"\nHTML Report: {result.report_path}")
print(f"Open the report in your browser to review results.")
```

### Step 4: Execute and View Report

```bash
python run_batch.py

# Open the generated report
open ./output/batch_results/report.html  # macOS
xdg-open ./output/batch_results/report.html  # Linux
start ./output/batch_results/report.html  # Windows
```

**What You'll See in the Report**:
- Summary statistics (total, successful, failed)
- Stage-grouped view: All images organized by pipeline stage
- Per-image complete results
- Error section for any failures
- All images displayed inline for review

---

## Common Patterns

### Pattern 1: Custom Stage with Parameters

```python
class MyCustomStage:
    def __init__(self, name: str, param1: float = 0.5, param2: str = "default"):
        self._name = name
        self.param1 = param1
        self.param2 = param2

    # ... implement protocol methods ...
```

### Pattern 2: Conditional Stage Execution

In YAML config:
```yaml
stages:
  - name: "optional_stage"
    stage_type: "OptionalStage"
    enabled: false  # Skip this stage
```

### Pattern 3: Error Handling

```python
from pydantic import ValidationError

try:
    config = load_config("config.yaml")
    pipeline = Pipeline.from_config(config)
    results = pipeline.run(image_path="test.jpg")
except ValidationError as e:
    print(f"Configuration error: {e}")
except ValueError as e:
    print(f"Invalid input: {e}")
except RuntimeError as e:
    print(f"Pipeline execution failed: {e}")
```

### Pattern 4: Accessing Intermediate Results

```python
results = pipeline.run(image_path="test.jpg")

# Access specific stage output
loader_output = results["loader"]
filter_output = results["filter"]

# Access specific data
original_image = loader_output["image"]
filtered_image = filter_output["filtered_image"]
```

---

## Testing Your Pipeline

### Unit Test Example

Create `test_my_pipeline.py`:

```python
import pytest
import numpy as np
from my_stages import ImageLoader, SimpleFilter
from viz_art.pipeline.base import Pipeline


def test_image_loader():
    """Test that ImageLoader loads images correctly."""
    loader = ImageLoader(name="test_loader")

    # Mock inputs
    inputs = {"image_path": "test_fixtures/sample.jpg"}

    # Execute stages
    preprocessed = loader.pre_process(inputs)
    predictions = loader.predict(preprocessed)
    outputs = loader.post_process(predictions)

    # Assertions
    assert "image" in outputs
    assert "metadata" in outputs
    assert isinstance(outputs["image"], np.ndarray)
    assert len(outputs["image"].shape) == 3  # Height, Width, Channels


def test_simple_filter():
    """Test that SimpleFilter applies blur correctly."""
    filter_stage = SimpleFilter(name="test_filter", kernel_size=5)

    # Mock input
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    inputs = {"image": test_image}

    # Execute
    preprocessed = filter_stage.pre_process(inputs)
    predictions = filter_stage.predict(preprocessed)
    outputs = filter_stage.post_process(predictions)

    # Assertions
    assert "filtered_image" in outputs
    assert outputs["filtered_image"].shape == test_image.shape


def test_pipeline_execution():
    """Test full pipeline execution."""
    pipeline = Pipeline(name="test_pipeline")
    pipeline.add_stage(ImageLoader(name="loader"))
    pipeline.add_stage(SimpleFilter(name="filter"))
    pipeline.connect("loader", "filter", "image", "image")

    results = pipeline.run(image_path="test_fixtures/sample.jpg")

    assert results["_status"] == "COMPLETED"
    assert "loader" in results
    assert "filter" in results


# Run tests
# pytest test_my_pipeline.py -v
```

---

## Troubleshooting

### Issue: "Config file not found"
**Solution**: Ensure the path to your YAML config is correct and relative to your working directory.

```python
from pathlib import Path
config_path = Path("configs/my_pipeline.yaml")
if not config_path.exists():
    print(f"Config not found at: {config_path.absolute()}")
```

### Issue: "Stage name not unique"
**Solution**: Ensure each stage has a unique name within the pipeline.

```yaml
stages:
  - name: "loader"  # Must be unique
  - name: "filter"  # Must be unique
  - name: "loader"  # ERROR: Duplicate name!
```

### Issue: "Connection creates circular dependency"
**Solution**: Check that your stage connections don't form a cycle.

```yaml
# BAD: Circular dependency
connections:
  - source: "stage1"
    target: "stage2"
  - source: "stage2"
    target: "stage1"  # Creates cycle!
```

### Issue: "Missing required input key"
**Solution**: Ensure connection output_key matches source stage's output_keys.

```python
# Stage defines
output_keys = ["image", "metadata"]

# Connection must use one of these keys
pipeline.connect("loader", "filter", output_key="image", ...)  # OK
pipeline.connect("loader", "filter", output_key="wrong_key", ...)  # ERROR
```

---

## Next Steps

After completing this quickstart, you can:

1. **Explore Advanced Configuration**
   - Use OmegaConf variable interpolation
   - Create environment-specific configs
   - Organize configs hierarchically

2. **Build Custom Stages**
   - Create domain-specific processing stages
   - Add validation and error handling
   - Implement complex multi-step transformations

3. **Optimize Performance**
   - Profile stage execution times
   - Identify bottlenecks
   - Optimize image processing operations

4. **Integrate with CI/CD**
   - Add automated tests
   - Setup GitHub Actions workflow
   - Validate pipeline configs automatically

5. **Extend to Production**
   - Add logging and monitoring
   - Implement data versioning
   - Scale to larger datasets

---

## Additional Resources

- **API Documentation**: See `contracts/python_api.md`
- **Data Model**: See `data-model.md`
- **Research Decisions**: See `research.md`
- **Examples**: Check the `examples/` directory
- **Issue Tracker**: GitHub Issues

---

## Support

If you encounter issues or have questions:
1. Check the troubleshooting section above
2. Review the API documentation
3. Search existing GitHub issues
4. Create a new issue with a minimal reproducible example

**Happy Pipeline Building!** 🚀
