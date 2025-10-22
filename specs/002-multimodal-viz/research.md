# Research: Multi-Modal Visualization with Point Cloud Support

**Feature**: 002-multimodal-viz
**Date**: 2025-10-22
**Status**: Complete

## Overview

This document consolidates research findings for integrating point cloud processing and visualization capabilities into the viz-art Python pipeline library (Python 3.8+).

## Technology Decisions

### 1. Point Cloud Processing Library

**Decision**: Use **Open3D 0.18+**

**Rationale**:
- Native support for .pcd, .ply, .xyz formats (required by FR-001)
- Excellent NumPy interoperability (critical for pipeline integration)
- Built-in operations: downsampling, filtering, outlier removal
- Off-screen rendering for thumbnails (no display server required)
- Active development and Python 3.8+ support

**Alternatives Considered**:
- **PyVista**: Strong visualization but heavier dependency, VTK-based
- **pyntcloud**: Simpler but less feature-complete, slower for large datasets
- **PCL Python bindings**: Powerful but complex setup, less Pythonic

**Version**: 0.18.0+ (NumPy 2.0 compatible), 0.19.0 latest

### 2. Interactive 3D Visualization

**Decision**: Use **Plotly 5.x+ with WebGL rendering**

**Rationale**:
- WebGL achieves 30+ FPS for 500K points (meets SC-002)
- Browser-based, no native UI development required
- Color-coding, rotation, zoom built-in (FR-009, FR-008)
- Integrates with existing Jinja2 HTML reports
- No additional desktop GUI frameworks needed

**Alternatives Considered**:
- **Open3D Visualizer**: Better performance for >1M points but requires desktop windowing
- **PyVista**: Excellent for large datasets but desktop-only, VTK dependency
- **ipyvolume**: Good for Jupyter but limited for standalone applications

**Critical Requirement**: WebGL **MUST** be enabled for >10K points (SVG rendering fails)

### 3. NumPy Validation for Pydantic

**Decision**: Use **numpydantic 1.6+**

**Rationale**:
- Provides type hints for NumPy arrays with shape validation
- Seamless Pydantic integration (existing dependency)
- Validates array dimensions (e.g., Nx3 for points)
- Handles serialization/deserialization

**Alternative**: Custom validators (more code, less type safety)

### 4. Calibration Storage Format

**Decision**: **YAML with OmegaConf structured configs**

**Rationale**:
- Consistent with existing pipeline configuration (FR-016)
- Structured validation with Pydantic dataclasses
- Human-readable for manual editing
- Supports comments and includes

**Alternative**: JSON (less human-friendly, no comments)

### 5. Projection Algorithm

**Decision**: **OpenCV `cv2.projectPoints()`**

**Rationale**:
- Standard computer vision approach
- Handles camera distortion (FR-014)
- Optimized C++ implementation
- Already a project dependency

**No alternatives needed** - industry standard approach

## Integration Patterns

### Pipeline Stage Type System

**Decision**: Add `input_data_types` and `output_data_types` properties to `PipelineStage`

**Pattern**:
```python
class PipelineStage:
    @property
    def input_data_types(self) -> Dict[str, type]:
        """Return expected types for each input key."""
        return {}

    @property
    def output_data_types(self) -> Dict[str, type]:
        """Return expected types for each output key."""
        return {}
```

**Rationale**:
- Enables pipeline validation (FR-029)
- Type safety without runtime overhead
- Backward compatible (optional implementation)

### Multi-Modal Data Handling

**Decision**: Use Union types in `Pipeline.run()` to accept both images and point clouds

**Pattern**:
```python
def run(self,
        image_path: str | None = None,
        pointcloud_path: str | None = None,
        **kwargs) -> Dict[str, Any]:
    """Run pipeline with image and/or point cloud inputs."""
```

**Rationale**:
- Flexible input method (FR-004)
- Maintains backward compatibility
- Clear API for multi-modal pipelines

### OutputSaver Extension

**Decision**: Extend existing `OutputSaver` class with point cloud format detection

**Pattern**:
```python
def save_stage_output(self, ...outputs: Dict[str, Any]...):
    for key, value in outputs.items():
        if isinstance(value, o3d.geometry.PointCloud):
            # Save as .pcd/.ply/.xyz based on config
        elif isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[1] == 3:
            # Could be point cloud array - save as .xyz
        elif isinstance(value, np.ndarray) and value.ndim in [2, 3]:
            # Image array - existing logic
```

**Rationale**:
- Reuses existing batch processing infrastructure (FR-007)
- Type-based format selection
- No changes to batch processor

## Performance Considerations

### Memory Management

**Decision**: Implement voxel downsampling before visualization for >100K points

**Benchmarks** (100K points):
- Voxel downsampling: 50-100ms
- Outlier removal: 200-500ms
- Plotly WebGL render (first): 1-2s
- Open3D thumbnail: 100-500ms

**Target**: Stay under 5s total (SC-001) ✓

### Large Point Cloud Handling

**Decision**: Add optional memory-mapped loading for files >100MB

**Pattern**:
```python
if file_size > 100_000_000:  # 100MB
    # Use memory-mapped numpy arrays
    points_mmap = np.load(cache_path, mmap_mode='r')
```

**Rationale**:
- Prevents memory exhaustion
- Allows processing 1M+ point clouds
- Lazy loading for better startup time

## Common Pitfalls Documented

### 1. Color Range Issues
- **Problem**: Open3D expects [0,1], OpenCV uses [0,255]
- **Solution**: Explicit conversion helpers in utility module

### 2. WebGL Performance
- **Problem**: Plotly SVG rendering fails >10K points
- **Solution**: Force WebGL mode: `render_mode='webgl'`

### 3. Empty Point Clouds
- **Problem**: Open3D crashes on empty geometries
- **Solution**: Always validate `len(pcd.points) > 0`

### 4. Headless Thumbnail Rendering
- **Problem**: `Visualizer` requires display server
- **Solution**: Use `OffscreenRenderer` (no DISPLAY needed)

### 5. Point Cloud File Formats
- **Problem**: .xyz files don't store colors/normals
- **Solution**: Default to .ply for full feature support

## Dependencies to Add

```toml
[project]
dependencies = [
    # Existing
    "pydantic>=2.0",
    "omegaconf>=2.3",
    "pillow>=10.0",
    "opencv-python>=4.8",
    "jinja2>=3.1",
    "numpy>=1.24",
    # New for Phase 2
    "open3d>=0.18",        # Point cloud processing
    "plotly>=5.0",         # Interactive 3D visualization
    "numpydantic>=1.6",    # NumPy array validation
]
```

## Testing Strategy

### Unit Tests
- Point cloud loading (.pcd, .ply, .xyz)
- Projection accuracy (synthetic camera + points)
- Thumbnail generation (headless rendering)
- Validation rules (empty clouds, NaN values)

### Integration Tests
- Multi-modal pipeline execution
- OutputSaver with point clouds
- HTML report generation with thumbnails
- Stage-to-stage point cloud passing

### Performance Tests
- 100K point loading < 5s
- Interactive viewer 30+ FPS
- Projection accuracy within 2px
- Thumbnail generation < 3s per cloud

## Risk Mitigation

### Risk 1: WebGL Browser Compatibility
- **Impact**: Visualization fails in old browsers
- **Mitigation**: Document minimum browser requirements (Chrome 90+, Firefox 88+)
- **Fallback**: Downsample to <50K points automatically

### Risk 2: Open3D Installation Issues
- **Impact**: Complex build requirements on some platforms
- **Mitigation**: Use pre-built wheels (available for most platforms)
- **Fallback**: Document conda installation as alternative

### Risk 3: Memory Constraints
- **Impact**: Large point clouds (>1M points) exceed RAM
- **Mitigation**: Implement automatic downsampling in loader stages
- **Threshold**: Warn users if loading >500K points without downsampling

## Next Steps

Phase 1 artifacts to create:
1. **data-model.md**: Point cloud entity definitions
2. **contracts/**: API schemas for point cloud stages
3. **quickstart.md**: Getting started with multi-modal pipelines

## References

- [Open3D Documentation](https://www.open3d.org/docs/latest/)
- [Plotly 3D Scatter](https://plotly.com/python/3d-scatter-plots/)
- [OpenCV Camera Calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [numpydantic](https://numpydantic.readthedocs.io/)
