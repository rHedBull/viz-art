# OutputSaver Integration Guide

## Summary

I've created the OutputSaver foundation. Due to token limits, here's how to complete the integration:

## What's Done ✅

1. **Created `src/viz_art/pipeline/output_saver.py`** - Complete OutputSaver class
2. **Updated `src/viz_art/config/schema.py`** - Added OutputSaveConfig and batch config fields
3. **Updated HTML template** - Professional design with collapsible sections

## What's Needed (30 min)

### Step 1: Update BatchProcessor (15 min)

In `src/viz_art/batch/processor.py`:

```python
# Add import at top
from viz_art.pipeline.output_saver import OutputSaver

# In __init__ method:
def __init__(self, pipeline: Pipeline, config: BatchConfigItem):
    # ... existing code ...

    # Add OutputSaver initialization
    self.output_saver = OutputSaver(
        output_mode=config.output_mode,
        save_config=config.save_outputs.model_dump(),
        output_dir=config.output_dir,
    )

# In run() method, after successful pipeline execution:
for idx, image_path in enumerate(image_paths, start=1):
    try:
        result = self.pipeline.run(image_path=str(image_path))

        # ADD THIS: Save stage outputs
        saved_paths = {}
        for stage_name, stage_output in result.items():
            if not stage_name.startswith('_'):  # Skip metadata
                paths = self.output_saver.save_stage_output(
                    run_id=batch_id,
                    stage_name=stage_name,
                    filename=image_path.name,
                    outputs=stage_output,
                    image_index=idx - 1,
                    is_error=False,
                )
                saved_paths[stage_name] = paths

        # Store saved paths in pipeline_run for reporter
        pipeline_run = self._create_pipeline_run(...)
        pipeline_run.saved_outputs = saved_paths  # Add this field

    except Exception as e:
        # For errors, mark is_error=True
        ...
```

### Step 2: Update HTMLReporter (10 min)

In `src/viz_art/batch/reporter.py`, update `_organize_by_stage()`:

```python
def _organize_by_stage(self, batch_result: BatchResult) -> Dict[str, List[Dict[str, Any]]]:
    stages_data: Dict[str, List[Dict[str, Any]]] = {}

    for run in batch_result.run_results:
        filename = Path(run.inputs.get("image_path", "")).name

        for stage_name, outputs in run.outputs.items():
            if stage_name not in stages_data:
                stages_data[stage_name] = []

            # Get saved image path if available
            image_path = None
            if hasattr(run, 'saved_outputs') and run.saved_outputs:
                stage_saved = run.saved_outputs.get(stage_name, {})
                # Get first saved image (usually 'image' key)
                for key, path in stage_saved.items():
                    if path.endswith(('.png', '.jpg', '.jpeg')):
                        # Make path relative to report location
                        image_path = Path(path).relative_to(self.output_dir)
                        break

            image_data = {
                "filename": filename,
                "success": True,
                "image_path": str(image_path) if image_path else None,
                "stage_name": stage_name,
            }

            stages_data[stage_name].append(image_data)

    return stages_data
```

### Step 3: Update batch_config.yaml (2 min)

Add output settings:

```yaml
batch_config:
  input_dir: "./examples/test_batch/input"
  output_dir: "./examples/test_batch/output"
  file_patterns:
    - "*.jpg"
    - "*.png"
  recursive: true
  continue_on_error: true
  report_output: "batch_report.html"

  # NEW: Output saving configuration
  output_mode: "sample"  # sample | validation | production
  save_outputs:
    enabled: true
    stages: ["all"]  # or ["loader", "filter"]
    max_samples: 10
    format: "png"
```

### Step 4: Test (3 min)

```bash
# Regenerate report
python examples/test_batch/run_batch_test.py

# Check saved outputs
ls examples/test_batch/output/runs/*/stages/

# Open report - should now show images!
xdg-open examples/test_batch/output/batch_report.html
```

## Expected Directory Structure

```
examples/test_batch/output/
├── batch_report.html
└── runs/
    └── batch_20251022_153000/
        └── stages/
            ├── loader/
            │   ├── test_landscape_1_image.png
            │   ├── test_landscape_2_image.png
            │   └── ...
            └── filter/
                ├── test_landscape_1_filtered_image.png
                ├── test_landscape_2_filtered_image.png
                └── ...
```

## HTML Template Changes

The template I created already has:
- ✅ `<img src="{{ image_data.image_path }}">`  placeholders
- ✅ Fallback when no image available
- ✅ Professional design, no gradients
- ✅ Collapsible sections

## Notes

- OutputSaver only saves first 10 images in sample mode (configurable)
- Images are relative paths in HTML for portability
- Production mode can disable saving entirely
- Validation mode saves all errors automatically

Let me know if you want me to make these changes or if you'd like to do it yourself!
