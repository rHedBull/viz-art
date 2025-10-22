# Implementation Plan: Multi-Modal Visualization with Point Cloud Support

**Branch**: `002-multimodal-viz` | **Date**: 2025-10-22 | **Spec**: [spec.md](./spec.md)

## Summary

Extend the viz-art pipeline library to support 3D point cloud data alongside existing image processing capabilities. This includes loading .pcd/.ply/.xyz formats, interactive 3D visualization with Plotly, projection onto images using camera calibration, data validation, and thumbnail generation for batch reports. The implementation builds on Phase 1's OutputSaver and batch processing infrastructure.

**Key Technologies**: Open3D (point cloud processing), Plotly (3D visualization), OpenCV (projection), numpydantic (validation)

**Primary Goals**:
1. Load and process point clouds through pipeline stages (FR-001 to FR-007)
2. Interactive 3D visualization with WebGL (FR-008 to FR-013)
3. Multi-modal overlay (image + point cloud) with calibration (FR-014 to FR-017)
4. Data validation framework for point clouds (FR-018 to FR-022)
5. Thumbnail generation for HTML reports (FR-023 to FR-026)

## Technical Context

**Language/Version**: Python 3.8+ (matches existing codebase)
**Primary Dependencies**:
- Existing: Pydantic 2.0+, OmegaConf 2.3+, OpenCV 4.8+, NumPy 1.24+, Jinja2
- New: Open3D 0.18+, Plotly 5.0+, numpydantic 1.6+

**Storage**: File-based (.pcd, .ply, .xyz for point clouds; YAML for calibration)
**Testing**: pytest (existing framework)
**Target Platform**: Linux/macOS/Windows desktop (requires WebGL-capable browsers for visualization)
**Project Type**: Single library project with examples
**Performance Goals**:
- Load 100K points < 5s (SC-001)
- Interactive viewer 30+ FPS for 500K points (SC-002)
- Projection accuracy within 2 pixels (SC-004)
- Thumbnail generation < 3s per cloud (SC-006)

**Constraints**:
- Backward compatible with Phase 1 image-only pipelines
- Memory limit: ~2GB for point cloud visualization (browser WebGL limit)
- CPU-only processing (no GPU requirements for MVP)

**Scale/Scope**:
- Point clouds: 10K - 1M points typical
- Batch processing: 5-100 files
- 6 new stage types, ~30 functional requirements

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Status**: ✅ **PASSED** (No constitution file present - using default principles)

**Default Principles Applied**:
1. **Library-First**: ✅ Extending existing library, not creating new project
2. **Test-First**: ✅ Unit tests for each stage (see [quickstart.md](./quickstart.md))
3. **Backward Compatibility**: ✅ No breaking changes to Phase 1 API
4. **Clear Dependencies**: ✅ All new dependencies justified and documented in [research.md](./research.md)

**No Violations**: Feature builds cleanly on existing architecture without introducing complexity.

## Project Structure

### Documentation (this feature)

```text
specs/002-multimodal-viz/
├── spec.md                  # Feature specification (DONE)
├── plan.md                  # This file (DONE)
├── research.md              # Technical research findings (DONE)
├── data-model.md            # Entity definitions (DONE)
├── quickstart.md            # Getting started guide (DONE)
├── contracts/               # API contracts
│   └── python_api.md        # Stage interfaces (DONE)
└── checklists/
    └── requirements.md      # Spec quality checklist (DONE)
```

### Source Code (repository root)

```text
src/viz_art/
├── pipeline/
│   ├── base.py              # [MODIFY] Add type checking for point clouds
│   ├── stage.py             # [KEEP] No changes needed
│   ├── results.py           # [KEEP] No changes needed
│   ├── output_saver.py      # [MODIFY] Add point cloud format detection
│   └── pointcloud.py        # [NEW] Point cloud utilities
│
├── config/
│   ├── schema.py            # [MODIFY] Add point cloud validation rules
│   └── loader.py            # [KEEP] No changes needed
│
├── batch/
│   ├── processor.py         # [KEEP] No changes (OutputSaver handles new types)
│   ├── reporter.py          # [MODIFY] Add point cloud thumbnail display
│   └── templates/
│       └── batch_report.html  # [MODIFY] Add 3D viewer embeds
│
├── visualization/           # [NEW] Visualization modules
│   ├── __init__.py
│   ├── plotly_viewer.py     # Plotly 3D rendering
│   └── thumbnail.py         # Open3D thumbnail generation
│
├── calibration/             # [NEW] Calibration utilities
│   ├── __init__.py
│   ├── loader.py            # Load YAML/JSON calibration
│   └── projection.py        # 3D→2D projection
│
└── types/                   # [NEW] Type definitions
    ├── __init__.py
    └── pointcloud.py        # Type aliases for point clouds

examples/stages/
├── image_loader.py          # [KEEP] Existing
├── simple_filter.py         # [KEEP] Existing
├── grayscale_stage.py       # [KEEP] Existing
├── pointcloud_loader.py     # [NEW] Load .pcd/.ply/.xyz
├── pointcloud_projection.py # [NEW] Project points onto images
├── pointcloud_validation.py # [NEW] Validate point cloud quality
├── pointcloud_visualization.py  # [NEW] Plotly 3D viewer
└── pointcloud_thumbnail.py  # [NEW] Generate static thumbnails

tests/
├── unit/
│   ├── test_pointcloud_loader.py      # [NEW]
│   ├── test_projection.py             # [NEW]
│   ├── test_validation.py             # [NEW]
│   └── test_visualization.py          # [NEW]
│
├── integration/
│   ├── test_multimodal_pipeline.py    # [NEW]
│   └── test_output_saver_pointclouds.py  # [NEW]
│
└── fixtures/
    └── sample_pointclouds/            # [NEW] Test .pcd/.ply files
        ├── small_100pts.pcd
        └── medium_10k.ply
```

**Structure Decision**: Single project (Option 1) - extends existing library without additional complexity. All new functionality lives in dedicated modules (`visualization/`, `calibration/`, `types/`) to maintain clean separation.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No violations - table not needed.

## Phase 0: Research (COMPLETE)

**Status**: ✅ DONE

**Deliverable**: [research.md](./research.md)

**Key Decisions**:
1. **Point Cloud Library**: Open3D 0.18+ (NumPy 2.0 compatible, off-screen rendering)
2. **3D Visualization**: Plotly 5.x with WebGL (browser-based, 30+ FPS for 500K points)
3. **Validation**: numpydantic 1.6+ (type-safe NumPy arrays with Pydantic)
4. **Calibration Format**: YAML with OmegaConf (consistent with existing config system)
5. **Projection**: OpenCV `projectPoints()` (industry standard, handles distortion)

**Alternatives Considered & Rejected**:
- PyVista (too heavyweight, VTK dependency)
- Open3D Visualizer for web (requires display server, not headless-friendly)
- Custom validators (more code, less type safety than numpydantic)

**Performance Benchmarks** (100K points):
- Load + downsample: 50-150ms ✅
- Plotly WebGL render: 1-2s first load ✅
- Thumbnail generation: 100-500ms ✅
- Total pipeline: < 5s target ✅

## Phase 1: Design & Contracts (COMPLETE)

**Status**: ✅ DONE

**Deliverables**:
1. ✅ [data-model.md](./data-model.md) - 5 core entities defined
2. ✅ [contracts/python_api.md](./contracts/python_api.md) - 6 stage APIs + helpers
3. ✅ [quickstart.md](./quickstart.md) - Getting started guide with examples

**Entity Summary**:
- `PointCloud`: Nx3 points + optional colors/normals/intensity
- `MultiModalInput`: Synchronized image + point cloud pairs
- `Calibration`: Camera intrinsics + extrinsics for projection
- `PointCloudValidationRules`: Configurable quality checks
- `VisualizationConfig`: Rendering and display settings

**Stage APIs Defined**:
1. `PointCloudLoaderStage`: Load .pcd/.ply/.xyz with preprocessing
2. `PointCloudProjectionStage`: 3D→2D projection with calibration
3. `PointCloudThumbnailStage`: Static thumbnail generation
4. `PointCloudVisualizationStage`: Interactive Plotly 3D viewer
5. `PointCloudValidationStage`: Quality checks (NaN, empty, ranges)
6. `MultiModalLoaderStage`: Load synchronized image + point cloud

**Type System**:
- Added `input_data_types` and `output_data_types` properties to `PipelineStage`
- Enables compile-time validation of pipeline connections (FR-029)

## Phase 2: Implementation Tasks (NOT CREATED YET)

**Note**: Implementation tasks will be created using `/speckit.tasks` command after this plan is approved.

**High-Level Task Categories** (to be expanded in tasks.md):

### Category 1: Core Infrastructure (Priority 1)
- Add Open3D, Plotly, numpydantic to dependencies
- Create `src/viz_art/types/pointcloud.py` with type aliases
- Create `src/viz_art/pipeline/pointcloud.py` with conversion utilities
- Extend `OutputSaver` to detect and save point cloud formats
- Add unit tests for core utilities

### Category 2: Point Cloud Stages (Priority 1)
- Implement `PointCloudLoaderStage` with downsampling and outlier removal
- Implement `PointCloudValidationStage` with configurable rules
- Create test fixtures (sample .pcd/.ply files)
- Write unit tests achieving 90%+ coverage

### Category 3: Calibration & Projection (Priority 2)
- Create `src/viz_art/calibration/loader.py` for YAML/JSON loading
- Create `src/viz_art/calibration/projection.py` with OpenCV projection
- Implement `PointCloudProjectionStage` with overlay rendering
- Add calibration validation and error handling
- Write projection accuracy tests (< 2px tolerance)

### Category 4: Visualization (Priority 2)
- Create `src/viz_art/visualization/plotly_viewer.py` for 3D rendering
- Create `src/viz_art/visualization/thumbnail.py` using OffscreenRenderer
- Implement `PointCloudVisualizationStage` with WebGL
- Implement `PointCloudThumbnailStage` with multi-viewpoint support
- Write performance tests (30+ FPS, < 3s thumbnail)

### Category 5: Batch Processing Integration (Priority 2)
- Modify `HTMLReporter` to embed Plotly 3D viewers in reports
- Update `batch_report.html` template with interactive viewers
- Add point cloud thumbnail gallery view
- Test batch processing with mixed image + point cloud inputs

### Category 6: Multi-Modal Support (Priority 3)
- Implement `MultiModalLoaderStage` for synchronized data pairs
- Add timestamp synchronization validation
- Create multi-modal pipeline example
- Write integration tests for end-to-end multi-modal flow

### Category 7: Documentation & Examples (Priority 3)
- Update README.md with point cloud examples
- Create example pipelines in `examples/configs/`
- Add sample calibration files
- Create tutorial notebooks (optional)

**Estimated Implementation Time**: 2 weeks (following Phase 2 roadmap timeline)

## Dependencies

### New Python Packages

Add to `pyproject.toml`:
```toml
[project]
dependencies = [
    # Phase 1 (existing)
    "pydantic>=2.0",
    "omegaconf>=2.3",
    "pillow>=10.0",
    "opencv-python>=4.8",
    "jinja2>=3.1",
    "numpy>=1.24",

    # Phase 2 (new)
    "open3d>=0.18",        # Point cloud processing
    "plotly>=5.0",         # Interactive 3D visualization
    "numpydantic>=1.6",    # NumPy array validation
]
```

### External Dependencies (None)
- No external services required
- No additional system libraries beyond Open3D's dependencies

## Testing Strategy

### Unit Tests (Per-Stage)
- **Coverage Target**: 90%+ for new code
- **Test Cases**: Happy path, edge cases, error conditions
- **Performance Tests**: Verify SC-001 to SC-006 metrics

### Integration Tests (End-to-End)
- Multi-stage point cloud pipeline execution
- Mixed image + point cloud batch processing
- OutputSaver with point cloud formats
- HTML report generation with interactive viewers

### Test Data
- Small: 100 points (fast tests)
- Medium: 10K points (typical use)
- Large: 500K points (performance tests)

**Test Execution Time Target**: < 30 seconds for full suite

## Rollout Plan

### Phase 2.1: Core Support (Week 1, Days 1-2)
- Add dependencies
- Implement `PointCloudLoaderStage`
- Basic visualization with Plotly
- ✅ **Gate**: Can load and display point clouds

### Phase 2.2: Validation & Quality (Week 1, Days 3-4)
- Implement validation framework
- Add outlier removal and downsampling
- ✅ **Gate**: Can validate and preprocess point clouds

### Phase 2.3: Multi-Modal Fusion (Week 2, Days 1-2)
- Calibration loading
- Projection stage
- Overlay visualization
- ✅ **Gate**: Can project point clouds onto images

### Phase 2.4: Batch Integration (Week 2, Days 3-4)
- Thumbnail generation
- HTML report updates
- OutputSaver extensions
- ✅ **Gate**: Batch reports show point cloud thumbnails

### Phase 2.5: Polish & Documentation (Week 2, Day 5)
- Example pipelines
- Documentation updates
- Performance validation
- ✅ **Gate**: All success criteria met, ready for merge

## Success Criteria Validation

| ID | Criterion | Verification Method | Target |
|----|-----------|---------------------|---------|
| SC-001 | Load 100K points | Performance test | < 5s |
| SC-002 | Viewer FPS | Browser profiling | 30+ FPS |
| SC-003 | Display speed | Integration test | < 2s |
| SC-004 | Projection accuracy | Unit test with synthetic data | < 2px |
| SC-005 | Validation accuracy | Unit test with corrupted data | 100% detection |
| SC-006 | Thumbnail speed | Performance test | < 3s |
| SC-007 | Stability | Stress test | 100 runs no crash |
| SC-008 | Usability | Manual testing | 90% success |
| SC-009 | Interaction latency | Browser measurement | < 100ms |
| SC-010 | Thumbnail quality | Visual inspection | Recognizable |

## Risk Mitigation

### Risk 1: Open3D Installation Complexity
- **Probability**: Medium
- **Impact**: High (blocks development)
- **Mitigation**: Document conda installation, provide pre-built wheel links
- **Fallback**: Support Open3D 0.17+ for wider compatibility

### Risk 2: WebGL Browser Compatibility
- **Probability**: Low
- **Impact**: Medium (visualization fails for some users)
- **Mitigation**: Auto-detect and downsample for performance
- **Fallback**: Document minimum browser requirements

### Risk 3: Memory Constraints with Large Point Clouds
- **Probability**: Medium
- **Impact**: Medium (crashes on large files)
- **Mitigation**: Implement automatic downsampling warnings
- **Fallback**: Memory-mapped loading for >100MB files

### Risk 4: Calibration File Format Variability
- **Probability**: Medium
- **Impact**: Low (users provide incompatible calibration)
- **Mitigation**: Support multiple formats, clear error messages
- **Fallback**: Provide calibration conversion utilities

## Next Steps

1. ✅ **Complete**: Research, data model, contracts, quickstart
2. ⏭️ **Next**: Run `/speckit.tasks` to generate detailed implementation tasks
3. ⏭️ **Then**: Begin implementation following rollout plan
4. ⏭️ **Finally**: Validate success criteria and merge to main

## References

- [Feature Specification](./spec.md)
- [Technical Research](./research.md)
- [Data Model](./data-model.md)
- [API Contracts](./contracts/python_api.md)
- [Quickstart Guide](./quickstart.md)
- [Open3D Documentation](https://www.open3d.org/docs/latest/)
- [Plotly 3D Scatter](https://plotly.com/python/3d-scatter-plots/)

---

**Plan Version**: 1.0
**Status**: Ready for `/speckit.tasks`
**Approved**: Pending review
