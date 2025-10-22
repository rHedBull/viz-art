# Phase 1 Implementation Research: Base Pipeline Architecture

## Document Overview

This document provides research-backed decisions for the Phase 1 vision pipeline implementation. Each section covers a key architectural decision with practical guidance for implementation.

**Research Date:** 2025-10-22
**Target:** viz-art Phase 1 (Base Pipeline Architecture)

---

## 1. Python Pipeline Architecture Patterns

### Decision: Protocol-Based Composition with Typed Interfaces

> **⚠️ IMPLEMENTATION UPDATE (2025-10-22):**
> After initial implementation, the approach was changed from **Protocol** to **Abstract Base Class (ABC)** for better validation and developer experience. The PipelineStage is now implemented as an ABC with `@abstractmethod` decorators, providing compile-time validation and catching errors at class definition time rather than runtime. See `src/viz_art/pipeline/stage.py` for the final implementation.

**Original Selected Approach:**
- Define a `Pipeline` protocol using `typing.Protocol` for structural subtyping
- Use composition over inheritance for pipeline stages
- Implement stages as independent classes with a common interface
- Pass data between stages using typed dictionaries or dataclasses

**Original Rationale:**

1. **Flexibility:** Protocols provide structural subtyping (duck typing with static type checking), allowing any class that implements the required methods to be used as a pipeline stage without explicit inheritance.

2. **Modularity:** Composition allows stages to be developed and tested independently. Each stage is a self-contained unit with clear inputs and outputs.

3. **Maintainability:** Changes to one stage don't ripple through the codebase. Flat structure is easier to navigate than deep inheritance hierarchies.

4. **Testing:** Stages can be tested in isolation by simply instantiating them and calling their methods, without needing complex test harnesses.

5. **Extension Points:** New stages can be added by implementing the protocol without modifying existing code.

**Alternatives Considered:**

1. **Abstract Base Classes (ABC):** *(Ultimately chosen for implementation)*
   - **Pros:** Explicit contract enforcement, compile-time validation, better IDE support, catches typos immediately
   - **Cons:** Requires explicit inheritance, creates class hierarchies
   - **Verdict:** ~~Too rigid for our use case~~ **Actually chosen** - provides better developer experience and validation

2. **Function-Based Pipeline:**
   - **Pros:** Simple, functional approach, no OOP overhead
   - **Cons:** Harder to maintain state, less clear structure for configuration, difficult to extend
   - **Verdict:** Would work for simple cases but lacks structure for our multi-stage needs

3. **Decorator Pattern:**
   - **Pros:** Dynamic behavior addition, flexible composition
   - **Cons:** More complex to implement, can obscure data flow
   - **Verdict:** Better suited for adding cross-cutting concerns (logging, caching) than core pipeline structure

**Implementation Notes:**

```python
from typing import Protocol, Any, Dict
from pathlib import Path

class PipelineStage(Protocol):
    """Protocol defining the interface for all pipeline stages."""

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and return results.

        Args:
            input_data: Dictionary containing input data and metadata

        Returns:
            Dictionary containing processed results and metadata
        """
        ...

    @property
    def name(self) -> str:
        """Return the stage name for reporting."""
        ...

class Pipeline:
    """Pipeline orchestrator using composition."""

    def __init__(self, stages: list[PipelineStage]):
        self.stages = stages

    def run(self, initial_input: Dict[str, Any]) -> list[Dict[str, Any]]:
        """
        Run all stages sequentially, passing output as input to next stage.

        Returns:
            List of all stage outputs for reporting
        """
        current_data = initial_input
        results = []

        for stage in self.stages:
            try:
                output = stage.process(current_data)
                results.append({
                    'stage': stage.name,
                    'output': output,
                    'success': True
                })
                current_data = output  # Output becomes input for next stage
            except Exception as e:
                results.append({
                    'stage': stage.name,
                    'error': str(e),
                    'success': False
                })
                raise  # Re-raise to stop pipeline on error

        return results
```

**Key Considerations:**

1. **Data Flow:** Use dictionaries for flexibility, but document expected keys. Consider using TypedDict or dataclasses for stricter typing.

2. **Error Handling:** Implement at both stage and pipeline levels. Stages should raise specific exceptions; pipeline should catch and report.

3. **State Management:** Stages should be stateless where possible. If state is needed, make it explicit in the stage's `__init__`.

4. **Configuration:** Each stage receives its configuration in `__init__`, separate from runtime `process()` inputs.

5. **Metadata Propagation:** Pass metadata (image path, original size, etc.) through the pipeline alongside processed data.

---

## 2. OmegaConf + Pydantic Integration

### Decision: Hybrid Workflow - OmegaConf for Loading, Pydantic for Validation

**Selected Approach:**
- Use OmegaConf to load and merge YAML configuration files
- Convert OmegaConf DictConfig to native Python dict using `to_container()`
- Validate using Pydantic's `model_validate()` with comprehensive error reporting
- Define configuration schema using Pydantic BaseModel classes

**Rationale:**

1. **Best of Both Worlds:** OmegaConf handles YAML loading, variable interpolation, and config merging. Pydantic provides robust validation, type coercion, and clear error messages.

2. **Schema-First Design:** Pydantic models serve as living documentation of the configuration structure, with built-in validation and helpful error messages.

3. **Type Safety:** Pydantic ensures configuration values are the correct types and within valid ranges before the pipeline runs.

4. **Developer Experience:** Pydantic's validation errors are detailed and actionable, showing exactly what's wrong and where.

5. **Flexibility:** OmegaConf's interpolation (`${var}` syntax) and merging capabilities are preserved while adding validation.

**Alternatives Considered:**

1. **OmegaConf Structured Configs Only:**
   - **Pros:** Single dependency, native integration
   - **Cons:** Limited validation capabilities, no Union types, less powerful type coercion
   - **Verdict:** Insufficient validation for our needs

2. **Pydantic-Only Configuration:**
   - **Pros:** Single framework, great validation
   - **Cons:** No YAML interpolation, manual config merging, less flexible hierarchical configs
   - **Verdict:** Missing OmegaConf's convenience features

3. **Pydantic Dataclasses as OmegaConf Structured Configs:**
   - **Pros:** Drop-in replacement for stdlib dataclasses
   - **Cons:** Limited to primitives, no Union types, misses some Pydantic features
   - **Verdict:** Too restrictive for complex configurations

**Implementation Notes:**

```python
from omegaconf import OmegaConf, DictConfig
from pydantic import BaseModel, Field, field_validator, ValidationError
from pathlib import Path
from typing import Any

# Define Pydantic schema
class StageConfig(BaseModel):
    """Base configuration for pipeline stages."""
    enabled: bool = True
    timeout: int = Field(default=300, gt=0, description="Stage timeout in seconds")

class LoadConfig(StageConfig):
    """Configuration for image loading stage."""
    input_dir: Path
    recursive: bool = True
    extensions: list[str] = Field(default=['.jpg', '.png', '.jpeg'])

    @field_validator('input_dir')
    @classmethod
    def validate_input_dir(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Input directory does not exist: {v}")
        if not v.is_dir():
            raise ValueError(f"Input path is not a directory: {v}")
        return v

class PipelineConfig(BaseModel):
    """Top-level pipeline configuration."""
    pipeline_name: str
    load: LoadConfig
    # Other stage configs...

    class Config:
        # Allow extra fields for extensibility
        extra = 'forbid'  # Or 'allow' if you want flexibility

# Usage: Load and validate configuration
def load_config(config_path: Path) -> PipelineConfig:
    """
    Load YAML config with OmegaConf and validate with Pydantic.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Validated PipelineConfig instance

    Raises:
        ValidationError: If configuration is invalid
    """
    # Load with OmegaConf (handles YAML, interpolation, merging)
    omega_cfg: DictConfig = OmegaConf.load(config_path)

    # Convert to native Python dict, resolving interpolations
    cfg_dict = OmegaConf.to_container(omega_cfg, resolve=True)

    # Validate with Pydantic
    try:
        validated_config = PipelineConfig.model_validate(cfg_dict)
        return validated_config
    except ValidationError as e:
        # Pydantic provides detailed error messages
        print("Configuration validation failed:")
        print(e.json(indent=2))
        raise

# Alternative: Using kwargs unpacking
def load_config_alternative(config_path: Path) -> PipelineConfig:
    omega_cfg = OmegaConf.load(config_path)
    cfg_dict = OmegaConf.to_container(omega_cfg, resolve=True)
    return PipelineConfig(**cfg_dict)
```

**Example YAML Configuration:**

```yaml
# config.yaml
pipeline_name: "edge_detection_pipeline"

# Use OmegaConf interpolation
directories:
  base: "/data/images"
  output: "/data/output"

load:
  input_dir: ${directories.base}  # Interpolation
  recursive: true
  extensions: ['.jpg', '.png']
  timeout: 300
```

**Key Considerations:**

1. **Error Reporting:** Pydantic aggregates all validation errors into a single exception, providing comprehensive feedback on all issues at once.

2. **Type Coercion:** Pydantic automatically coerces types (e.g., strings to Path objects), but be explicit about expected types in your schema.

3. **Validation Order:** Define validators with clear dependencies. Use `@field_validator` for single-field validation and `@model_validator` for cross-field validation.

4. **Hierarchical Configs:** Nest Pydantic models for stage-specific configurations, mirroring the YAML structure.

5. **Documentation:** Pydantic's Field descriptions become part of the schema and can be used to generate documentation.

6. **Resolve Early:** Always use `resolve=True` in `to_container()` to resolve OmegaConf interpolations before Pydantic validation.

---

## 3. Batch Image Processing Patterns

### Decision: Generator-Based Processing with pathlib.rglob() and Explicit Error Handling

**Selected Approach:**
- Use `pathlib.Path.rglob()` for recursive file discovery
- Return generators to avoid loading all paths into memory
- Implement try-except blocks at the per-file level with "continue on failure" semantics
- Track both successful and failed files for comprehensive reporting
- Use simple progress indication without external dependencies

**Rationale:**

1. **Memory Efficiency:** Generators allow processing one image at a time without loading all file paths or images into memory.

2. **Modern Python:** pathlib provides a cleaner, object-oriented API compared to os.walk() or glob module.

3. **Robustness:** Per-file error handling ensures one corrupted image doesn't stop the entire pipeline.

4. **Transparency:** Tracking failures provides visibility into problems without silent failures.

5. **Simplicity:** No need for external progress bar libraries (tqdm) in Phase 1; simple logging suffices.

**Alternatives Considered:**

1. **os.walk() + glob:**
   - **Pros:** More familiar to some developers, finer control over traversal
   - **Cons:** More verbose, less Pythonic, returns strings instead of Path objects
   - **Verdict:** Outdated approach; pathlib is the modern standard

2. **Parallel Processing (multiprocessing/ThreadPoolExecutor):**
   - **Pros:** Faster for CPU-bound operations
   - **Cons:** More complex, harder to debug, coordination overhead for our OpenCV operations
   - **Verdict:** Premature optimization for Phase 1; add in Phase 2 if needed

3. **Fail-Fast Error Handling:**
   - **Pros:** Simpler to implement, immediate feedback
   - **Cons:** Single bad file stops entire batch, poor user experience
   - **Verdict:** Too fragile for real-world image collections

**Implementation Notes:**

```python
from pathlib import Path
from typing import Iterator, List
import logging

logger = logging.getLogger(__name__)

class ImageDiscovery:
    """Discover image files in directory trees."""

    # Common image extensions
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    @staticmethod
    def find_images(
        root_dir: Path,
        extensions: set[str] | None = None,
        recursive: bool = True
    ) -> Iterator[Path]:
        """
        Generate paths to image files.

        Args:
            root_dir: Root directory to search
            extensions: Set of file extensions to include (with dots)
            recursive: Whether to search subdirectories

        Yields:
            Path objects for discovered image files
        """
        if extensions is None:
            extensions = ImageDiscovery.IMAGE_EXTENSIONS

        # Normalize extensions to lowercase
        extensions = {ext.lower() for ext in extensions}

        if recursive:
            # Use rglob for recursive search
            for ext in extensions:
                pattern = f"*{ext}"
                yield from root_dir.rglob(pattern)
        else:
            # Use glob for single directory
            for ext in extensions:
                pattern = f"*{ext}"
                yield from root_dir.glob(pattern)

class BatchProcessor:
    """Process multiple images with error handling and progress tracking."""

    def __init__(self):
        self.successful: List[Path] = []
        self.failed: List[tuple[Path, Exception]] = []

    def process_batch(
        self,
        image_paths: Iterator[Path],
        process_fn: callable
    ) -> dict:
        """
        Process batch of images with per-file error handling.

        Args:
            image_paths: Iterator of image file paths
            process_fn: Function to process each image (takes Path, returns result)

        Returns:
            Dictionary with processing statistics and results
        """
        results = []

        for idx, image_path in enumerate(image_paths, start=1):
            try:
                # Process single image
                result = process_fn(image_path)
                results.append({
                    'path': image_path,
                    'result': result,
                    'success': True
                })
                self.successful.append(image_path)

                # Simple progress indication
                if idx % 10 == 0:
                    logger.info(f"Processed {idx} images...")

            except Exception as e:
                # Log error and continue processing
                logger.error(f"Failed to process {image_path}: {e}")
                self.failed.append((image_path, e))
                results.append({
                    'path': image_path,
                    'error': str(e),
                    'success': False
                })

        # Summary statistics
        total = len(self.successful) + len(self.failed)
        logger.info(
            f"Batch processing complete: "
            f"{len(self.successful)}/{total} successful, "
            f"{len(self.failed)} failed"
        )

        return {
            'results': results,
            'successful_count': len(self.successful),
            'failed_count': len(self.failed),
            'failed_files': [(str(p), str(e)) for p, e in self.failed]
        }

# Usage example
def process_images(input_dir: Path) -> dict:
    """Process all images in directory."""
    processor = BatchProcessor()
    image_paths = ImageDiscovery.find_images(input_dir, recursive=True)

    def process_single_image(path: Path):
        # Your image processing logic here
        import cv2
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        # ... process image ...
        return {"processed": True}

    return processor.process_batch(image_paths, process_single_image)
```

**Key Considerations:**

1. **File Type Verification:** Don't rely solely on extensions. Consider using `imghdr` or trying to load the image to verify it's actually an image file.

2. **Case Sensitivity:** Always normalize file extensions to lowercase for comparison.

3. **Progress Tracking:** For Phase 1, simple logging every N files is sufficient. Consider tqdm for Phase 2 if needed.

4. **Memory Management:** The generator pattern ensures only one image is in memory at a time. Call `cv2.imread()` lazily, not upfront.

5. **Error Granularity:** Distinguish between different types of errors (file not found, corrupted image, processing failure) for better debugging.

6. **Large Directories:** Be aware that `rglob()` visits every directory in the tree. For very large directory structures (100k+ files), consider chunking or early termination.

7. **Hidden Files:** pathlib includes hidden files (starting with `.`) by default. Filter them if needed.

---

## 4. Static HTML Report Generation

### Decision: Jinja2 Templates with File References for Images

**Selected Approach:**
- Use Jinja2 for HTML template rendering
- Store images as separate files in an output directory structure
- Use relative file paths in HTML (not base64 embedding)
- Organize reports with stage-grouped sections
- Keep templates simple with inline CSS (no external frameworks)

**Rationale:**

1. **Performance:** File references avoid the 33% size increase and parsing slowdowns of base64 encoding. For a pipeline processing many images, this is significant.

2. **Caching:** Browsers can cache image files separately, improving repeat viewing performance.

3. **Debugging:** Separate image files are easier to inspect, compare, and debug than base64 strings in HTML.

4. **Report Size:** HTML files remain small and load quickly, while images load progressively.

5. **Flexibility:** Users can access individual processed images without parsing HTML.

**Alternatives Considered:**

1. **Base64 Embedded Images:**
   - **Pros:** Single self-contained HTML file, no broken links
   - **Cons:** 33% size increase, 10-32x slower parsing, no caching, bloated HTML files
   - **Verdict:** Performance cost too high for multiple images per report
   - **Use Case:** Only consider for small logos/icons (< 5KB)

2. **External CSS Framework (Bootstrap, Tailwind):**
   - **Pros:** Professional appearance, responsive out of the box
   - **Cons:** External dependency, larger download, complexity overhead
   - **Verdict:** Overkill for Phase 1; inline CSS provides adequate styling

3. **Static Site Generator (Sphinx, MkDocs):**
   - **Pros:** Rich features, navigation, search
   - **Cons:** Heavy dependency, learning curve, over-engineered for single reports
   - **Verdict:** Too complex for our needs

**Implementation Notes:**

```python
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
import base64
from typing import List, Dict, Any

class ReportGenerator:
    """Generate HTML reports with Jinja2."""

    def __init__(self, template_dir: Path):
        """
        Initialize report generator.

        Args:
            template_dir: Directory containing Jinja2 templates
        """
        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=True  # Prevent XSS in user-provided data
        )

    def generate_pipeline_report(
        self,
        output_path: Path,
        pipeline_results: List[Dict[str, Any]],
        images_dir: Path
    ) -> Path:
        """
        Generate pipeline execution report.

        Args:
            output_path: Path where HTML report will be written
            pipeline_results: List of stage results from pipeline
            images_dir: Directory containing processed images (relative refs)

        Returns:
            Path to generated report
        """
        template = self.env.get_template('pipeline_report.html')

        # Organize results by stage
        stages = {}
        for result in pipeline_results:
            stage_name = result['stage']
            if stage_name not in stages:
                stages[stage_name] = []
            stages[stage_name].append(result)

        # Render template
        html_content = template.render(
            stages=stages,
            total_images=len(pipeline_results),
            images_dir=images_dir.name  # Use relative path
        )

        # Write report
        output_path.write_text(html_content, encoding='utf-8')
        return output_path

    @staticmethod
    def create_output_structure(base_dir: Path) -> Dict[str, Path]:
        """
        Create standard output directory structure.

        Args:
            base_dir: Base output directory

        Returns:
            Dictionary mapping purpose to Path
        """
        structure = {
            'base': base_dir,
            'images': base_dir / 'images',
            'reports': base_dir / 'reports',
        }

        for path in structure.values():
            path.mkdir(parents=True, exist_ok=True)

        return structure
```

**Example Jinja2 Template:**

```html
<!-- templates/pipeline_report.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pipeline Processing Report</title>
    <style>
        /* Inline CSS for simplicity */
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }

        .header {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .stage {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .stage-header {
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
            margin-bottom: 15px;
        }

        .image-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 15px;
        }

        .image-card {
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
            background: #fafafa;
        }

        .image-card img {
            width: 100%;
            height: auto;
            border-radius: 4px;
        }

        .success { color: #28a745; }
        .error { color: #dc3545; }

        .stats {
            display: flex;
            gap: 20px;
            margin-top: 10px;
        }

        .stat-box {
            padding: 10px;
            background: #f8f9fa;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Pipeline Processing Report</h1>
        <div class="stats">
            <div class="stat-box">
                <strong>Total Images:</strong> {{ total_images }}
            </div>
            <div class="stat-box">
                <strong>Stages:</strong> {{ stages|length }}
            </div>
        </div>
    </div>

    {% for stage_name, results in stages.items() %}
    <div class="stage">
        <div class="stage-header">
            <h2>{{ stage_name }}</h2>
            <p>
                <span class="success">Success: {{ results|selectattr('success')|list|length }}</span> /
                <span class="error">Failed: {{ results|rejectattr('success')|list|length }}</span>
            </p>
        </div>

        <div class="image-grid">
            {% for result in results %}
            <div class="image-card">
                {% if result.success %}
                    <img src="{{ images_dir }}/{{ stage_name }}/{{ result.output.filename }}"
                         alt="{{ result.output.filename }}">
                    <p><strong>{{ result.output.filename }}</strong></p>
                {% else %}
                    <p class="error">Failed: {{ result.error }}</p>
                {% endif %}
            </div>
            {% endfor %}
        </div>
    </div>
    {% endfor %}
</body>
</html>
```

**Directory Structure:**

```
output/
├── reports/
│   └── pipeline_report.html
└── images/
    ├── load_stage/
    │   ├── image1.jpg
    │   └── image2.jpg
    ├── grayscale_stage/
    │   ├── image1_gray.jpg
    │   └── image2_gray.jpg
    └── edge_detection_stage/
        ├── image1_edges.jpg
        └── image2_edges.jpg
```

**Key Considerations:**

1. **Relative Paths:** Use relative paths in HTML to make reports portable. If output dir moves, images still work.

2. **Image Format:** Save processed images in appropriate formats (JPEG for photos, PNG for edges/graphics).

3. **Filename Sanitization:** Ensure filenames are filesystem-safe (no special characters, spaces).

4. **Image Size:** Consider saving thumbnails for the report and linking to full-size images if they're large.

5. **Responsive Design:** Use CSS Grid with `auto-fill` and `minmax()` for responsive layouts without media queries.

6. **Autoescape:** Always enable Jinja2's autoescape to prevent XSS if displaying user-provided data.

7. **Inline CSS vs External:** For Phase 1, inline CSS keeps everything simple. Move to external CSS if templates become complex.

8. **Base64 for Icons:** Small UI icons (< 5KB) can be base64-encoded to reduce HTTP requests, but not processed images.

---

## Summary: Implementation Priorities

### Phase 1 Quick Wins

1. **Start Simple:** Implement basic protocol-based pipeline with 2-3 stages first
2. **Validate Early:** Set up OmegaConf + Pydantic validation before writing pipeline logic
3. **Test with Small Batch:** Use 10-20 images initially to verify error handling
4. **Iterate on Reports:** Start with minimal HTML, add styling incrementally

### Key Success Metrics

- **Pipeline:** Stages are independently testable
- **Configuration:** Invalid configs fail with clear error messages
- **Batch Processing:** One bad image doesn't stop the pipeline
- **Reports:** HTML renders correctly and images load

### Common Pitfalls to Avoid

1. **Over-engineering:** Don't add parallelization, advanced caching, or complex state management in Phase 1
2. **Base64 Trap:** Don't embed large images in HTML for performance reasons
3. **Tight Coupling:** Keep stages independent; they should only depend on input data format
4. **Silent Failures:** Always log errors and track failed files

---

## References

### Python Pipeline Patterns
- Pipeline Pattern in Python (Software Patterns Lexicon)
- Data Pipeline Design Patterns (Start Data Engineering)
- Chain of Responsibility Pattern

### Protocol vs ABC
- PEP 544 - Protocols: Structural subtyping
- Python Protocols vs. ABCs: A Comprehensive Comparison
- Composition Over Inheritance Principle (Python Patterns Guide)

### OmegaConf + Pydantic
- OmegaConf Structured Config Documentation
- Pydantic Validation Documentation
- Configuration Management Best Practices

### Batch Processing
- Python pathlib Essentials (Python Cheatsheet)
- Directory Traversal Tools in Python (GeeksforGeeks)
- Batch Image Processing with Python (Coderholic)

### HTML Reports
- Jinja2 Template Designer Documentation
- Base64 Encoding & Performance (CSS Wizardry)
- Performance Anti-Patterns: Base64 Encoding (Web Performance Calendar)

---

**End of Research Document**
