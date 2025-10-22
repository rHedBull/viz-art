# OutputSaver Feature Roadmap

## Summary

The OutputSaver system has been designed as a foundational component that will evolve across all phases of the viz-art pipeline system. This document outlines what's implemented now (Phase 1) and what will be added in future phases.

---

## Phase 1 (Current - IMPLEMENTED) ✅

### **Core Infrastructure**
- `src/viz_art/pipeline/output_saver.py` - Complete OutputSaver class
- `src/viz_art/config/schema.py` - OutputSaveConfig and batch config integration
- Three operational modes:
  - **Sample mode**: Save all stages for first N images (default: 10)
  - **Validation mode**: Save errors + final outputs for analysis
  - **Production mode**: Minimal or no saving for efficiency

### **Capabilities**
- ✅ Save image outputs (numpy arrays → PNG/JPG)
- ✅ Configurable stage filtering (`["all"]` or specific stages)
- ✅ Max samples limit for debugging
- ✅ Organized output directory structure:
  ```
  output/
  ├── runs/{run_id}/stages/{stage_name}/    # Sample mode
  ├── validation/{run_id}/errors/            # Validation mode errors
  └── production/{run_id}/final/             # Production mode
  ```
- ✅ Integration hooks in BatchProcessor (pending final integration)
- ✅ HTML report template with image display support

### **Configuration**
```yaml
batch_config:
  output_mode: "sample"  # sample | validation | production
  save_outputs:
    enabled: true
    stages: ["all"]      # or specific: ["loader", "filter"]
    max_samples: 10      # limit for debugging
    format: "png"        # png | jpg | npy
```

---

## Phase 2: Multi-Modal Support

### **New Capabilities**
- [ ] Save point clouds (.pcd, .ply, .xyz formats)
- [ ] Multi-modal overlays (image + point cloud fusion)
- [ ] Point cloud thumbnail generation (rendered 2D views for reports)
- [ ] Support for Open3D geometry types

### **OutputSaver Extensions**
```python
# Add to OutputSaver class:
def _save_point_cloud(self, pc: o3d.geometry.PointCloud, output_dir: Path, base_name: str):
    """Save point cloud in multiple formats."""
    output_path = output_dir / f"{base_name}.pcd"
    o3d.io.write_point_cloud(str(output_path), pc)

    # Generate thumbnail for reports
    thumbnail = self._render_point_cloud_thumbnail(pc)
    return output_path, thumbnail

def save_stage_output(self, ..., outputs: Dict[str, Any]):
    # Extend to handle point clouds
    if isinstance(value, o3d.geometry.PointCloud):
        paths = self._save_point_cloud(value, stage_dir, base_filename)
```

---

## Phase 3: Logging & Audit Trail Integration

### **New Capabilities**
- [ ] Link audit logs to saved outputs
- [ ] Save run metadata (JSON format)
- [ ] Performance metrics storage (PyArrow/Parquet)
- [ ] Query interface for run history

### **Audit Log Structure**
```json
{
  "run_id": "run_20251022_153000",
  "timestamp": "2025-10-22T15:30:00Z",
  "pipeline": "detection-pipeline-v2",
  "stages": [
    {
      "name": "loader",
      "status": "completed",
      "duration_ms": 45,
      "outputs_saved": [
        "output/runs/run_20251022_153000/stages/loader/img1_image.png"
      ]
    }
  ],
  "metrics": {
    "total_duration_ms": 123,
    "memory_peak_mb": 456
  }
}
```

### **Integration Points**
- BatchProcessor writes audit log after each run
- Links to OutputSaver paths automatically
- Queryable by run_id, date range, performance thresholds

---

## Phase 4: Error Analysis & Ground Truth

### **New Capabilities**
- [ ] Automatic error output saving (validation mode)
- [ ] Prediction vs ground truth diff visualization
- [ ] Image diff with color-coded differences
- [ ] Point cloud diff with distance heatmaps
- [ ] Error categorization and clustering

### **Error Analysis Workflow**
```python
# In validation mode, automatically save errors
if is_error:
    output_saver.save_stage_output(
        ...,
        is_error=True  # Saves to validation/{run_id}/errors/
    )

# UI loads error artifacts
error_runs = load_validation_errors(run_id)
display_side_by_side(prediction_path, ground_truth_path)
visualize_diff(pred_img, gt_img, mode="color_coded")
```

### **Ground Truth Comparison**
- Save both prediction and ground truth in parallel
- Automatic diff generation
- Per-stage accuracy visualization using saved artifacts

---

## Phase 5: Model Versioning & A/B Testing

### **New Capabilities**
- [ ] Version-tagged output paths
- [ ] Automatic archival of version outputs
- [ ] Side-by-side version comparison
- [ ] A/B testing artifact management

### **Versioned Output Structure**
```
output/
└── runs/
    ├── run_20251022_v1.0/stages/    # Model v1.0 outputs
    ├── run_20251022_v1.1/stages/    # Model v1.1 outputs
    └── run_20251022_v2.0/stages/    # Model v2.0 outputs
```

### **Version Comparison UI**
- Load outputs from different versions
- Display side-by-side for same input
- Automated regression detection
- Performance metric comparison

---

## Phase 6+: Production Features

### **Disk Space Management**
- [ ] Retention policies (auto-delete old runs)
- [ ] Compression for archived outputs
- [ ] Selective cleanup (keep errors, delete successes)
- [ ] Disk usage monitoring and alerts

### **Advanced Features**
- [ ] Streaming output saving (for large batches)
- [ ] Distributed storage support (S3, GCS)
- [ ] Output deduplication
- [ ] Incremental backup

---

## Design Principles

### **Extensibility**
The OutputSaver is designed to be easily extended:
- Add new `_save_*()` methods for new data types
- Configuration-driven behavior
- Pluggable storage backends (future)

### **Performance**
- Lazy saving (only save when needed)
- Configurable limits (max_samples)
- Mode-based behavior (sample vs production)

### **Integration**
- Works seamlessly with existing pipeline
- Minimal changes to stage code
- Automatic path management
- Report generation integration

---

## Migration Path

### **Current State (Phase 1)**
```python
# Basic image saving for debugging
output_saver = OutputSaver("sample", save_config, output_dir)
paths = output_saver.save_stage_output(run_id, stage_name, filename, outputs)
```

### **Phase 2 (Point Clouds)**
```python
# No API changes needed - just works with new types
output_saver.save_stage_output(run_id, stage_name, filename, {
    "image": np_array,
    "pointcloud": o3d_geometry  # Automatically detected and saved
})
```

### **Phase 3 (Audit Integration)**
```python
# Automatic audit log linking
audit_log = create_audit_log(run_id, outputs_saved=output_saver.get_saved_paths())
```

### **Phase 4 (Error Analysis)**
```python
# Validation mode automatically saves errors
error_artifacts = output_saver.load_validation_errors(run_id)
visualize_errors(error_artifacts)
```

---

## Configuration Evolution

### **Phase 1 (Current)**
```yaml
output_mode: "sample"
save_outputs:
  enabled: true
  stages: ["all"]
  max_samples: 10
  format: "png"
```

### **Phase 2+ (Extended)**
```yaml
output_mode: "sample"
save_outputs:
  enabled: true
  stages: ["all"]
  max_samples: 10
  formats:
    images: "png"
    point_clouds: "pcd"
    overlays: "png"
  retention:
    days: 30
    keep_errors: true
  compression: true
  storage_backend: "local"  # or "s3", "gcs"
```

---

## Documentation Updates Complete ✅

- **implementation.md**: Updated all relevant phases with OutputSaver tasks
- **REQUIREMENTS.md**: Integrated OutputSaver into sections 5, 7, 8, 9
- **OUTPUT_SAVER_INTEGRATION.md**: Step-by-step integration guide for Phase 1

---

## Next Steps

1. **Complete Phase 1 Integration** (~30 min)
   - Integrate OutputSaver into BatchProcessor
   - Update HTMLReporter to use saved image paths
   - Test end-to-end with visual report

2. **Phase 2 Planning**
   - Design point cloud saving interface
   - Plan thumbnail generation strategy
   - Define multi-modal overlay format

3. **Long-term**
   - Each phase builds on OutputSaver foundation
   - No breaking changes needed
   - Progressive enhancement approach
