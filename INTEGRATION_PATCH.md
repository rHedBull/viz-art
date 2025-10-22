# OutputSaver Integration - Final Steps

## Status: Almost Complete

✅ **DONE:**
1. OutputSaver class created
2. Config schema updated
3. BatchProcessor.__init__ updated with OutputSaver
4. HTML template updated with collapsible sections

## Remaining Steps (15 min):

### Step 1: Update BatchProcessor.run() - Add Output Saving

In `src/viz_art/batch/processor.py`, find the section around line 150 where `pipeline.run()` is called and update:

```python
# Around line 150-170, in the try block inside the for loop:

# FIND THIS:
try:
    logger.info(f"[{idx}/{total_files}] Processing {image_path.name}")
    result = self.pipeline.run(image_path=str(image_path))

    pipeline_run = self._create_pipeline_run(
        image_path=image_path,
        result=result,
        status=RunStatus.COMPLETED,
        error=None,
    )

    self._successful_runs.append(pipeline_run)

# REPLACE WITH THIS:
try:
    logger.info(f"[{idx}/{total_files}] Processing {image_path.name}")
    result = self.pipeline.run(image_path=str(image_path))

    # Save stage outputs
    saved_paths = {}
    batch_id = f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"  # Get from run() method
    for stage_name, stage_output in result.items():
        if not stage_name.startswith('_'):  # Skip metadata keys
            paths = self.output_saver.save_stage_output(
                run_id=batch_id,
                stage_name=stage_name,
                filename=image_path.name,
                outputs=stage_output,
                image_index=idx - 1,
                is_error=False,
            )
            if paths:
                saved_paths[stage_name] = paths

    pipeline_run = self._create_pipeline_run(
        image_path=image_path,
        result=result,
        status=RunStatus.COMPLETED,
        error=None,
    )

    # Add saved paths to pipeline_run (store as dict in outputs metadata)
    if saved_paths:
        pipeline_run = pipeline_run._replace(
            outputs={**pipeline_run.outputs, '_saved_paths': saved_paths}
        )

    self._successful_runs.append(pipeline_run)
```

**Note:** batch_id should come from the run() method scope (around line 120). It's defined as:
```python
batch_id = f"batch_{started_at.strftime('%Y%m%d_%H%M%S')}"
```

### Step 2: Update HTMLReporter._organize_by_stage()

In `src/viz_art/batch/reporter.py`, around line 110, update the `_organize_by_stage` method:

```python
def _organize_by_stage(self, batch_result: BatchResult) -> Dict[str, List[Dict[str, Any]]]:
    """Organize run results by stage for stage-grouped view."""
    stages_data: Dict[str, List[Dict[str, Any]]] = {}

    for run in batch_result.run_results:
        filename = Path(run.inputs.get("image_path", "")).name

        # Get saved paths if available
        saved_paths = run.outputs.get('_saved_paths', {})

        for stage_name, outputs in run.outputs.items():
            if stage_name.startswith('_'):  # Skip metadata
                continue

            if stage_name not in stages_data:
                stages_data[stage_name] = []

            # Get image path from saved outputs
            image_path = None
            if saved_paths and stage_name in saved_paths:
                stage_saved = saved_paths[stage_name]
                # Get first image key
                for key, path in stage_saved.items():
                    if isinstance(path, str) and path.endswith(('.png', '.jpg', '.jpeg')):
                        # Make relative to report location (both in output_dir)
                        image_path = Path(path).name  # Just filename for now
                        # Better: calculate relative path
                        try:
                            rel_path = Path(path).relative_to(Path(batch_result.report_path).parent)
                            image_path = str(rel_path)
                        except:
                            image_path = path  # Fallback to absolute
                        break

            image_data = {
                "filename": filename,
                "success": True,
                "image_path": image_path,
                "stage_name": stage_name,
            }

            stages_data[stage_name].append(image_data)

    return stages_data
```

### Step 3: Update batch_config.yaml

Add output configuration:

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

  # Output saving configuration
  output_mode: "sample"
  save_outputs:
    enabled: true
    stages: ["all"]
    max_samples: 10
    format: "png"
```

### Step 4: Test!

```bash
# Run batch processing
python examples/test_batch/run_batch_test.py

# Check saved outputs exist
ls -R examples/test_batch/output/runs/

# Open report - images should display!
xdg-open examples/test_batch/output/batch_report.html
```

## Expected Output Structure

```
examples/test_batch/output/
├── batch_report.html
└── runs/
    └── batch_20251022_154500/
        └── stages/
            ├── loader/
            │   ├── test_landscape_1_image.png
            │   ├── test_landscape_1_metadata.png  (if metadata has arrays)
            │   └── ...
            └── filter/
                ├── test_landscape_1_filtered_image.png
                └── ...
```

## If You Get Stuck

The full code is ready in:
- `src/viz_art/pipeline/output_saver.py` - Complete
- `src/viz_art/config/schema.py` - Complete
- Integration needed in 2 files above

Let me know if you need help!
