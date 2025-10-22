# Tasks: Multi-Modal Visualization with Point Cloud Support

**Input**: Design documents from `/specs/002-multimodal-viz/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅

**Tests**: Not explicitly requested in feature specification - omitting test tasks

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: Repository root with `src/`, `tests/`, `examples/`
- Paths based on plan.md structure

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and dependency installation

- [X] T001 Add Open3D, Plotly, and numpydantic dependencies to pyproject.toml
- [X] T002 Create src/viz_art/types/pointcloud.py with type aliases for point cloud data
- [X] T003 [P] Create src/viz_art/pipeline/pointcloud.py for point cloud conversion utilities
- [X] T004 [P] Create tests/fixtures/sample_pointclouds/ directory with test .pcd/.ply files

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T005 Implement numpy_to_pointcloud() helper function in src/viz_art/pipeline/pointcloud.py
- [X] T006 Implement pointcloud_to_numpy() helper function in src/viz_art/pipeline/pointcloud.py
- [X] T007 Add input_data_types and output_data_types properties to PipelineStage in src/viz_art/pipeline/stage.py
- [X] T008 Extend OutputSaver.save_stage_output() in src/viz_art/pipeline/output_saver.py to detect and save point cloud formats (.pcd, .ply, .xyz)

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel ✅ COMPLETE

---

## Phase 3: User Story 1 - Process Point Cloud Through Pipeline (Priority: P1) 🎯 MVP

**Goal**: Enable loading and processing of point cloud data (.pcd, .ply, .xyz) through existing pipeline architecture, with preprocessing (downsampling, outlier removal) and validation capabilities.

**Independent Test**: Load a .pcd or .ply file, pass it through a pipeline with a point cloud processing stage, and verify the output contains the processed point cloud data saved in the configured format.

### Implementation for User Story 1

- [X] T009 [P] Create PointCloudValidationRules dataclass in src/viz_art/types/pointcloud.py
- [X] T010 [P] Create PointCloud dataclass in src/viz_art/types/pointcloud.py (frozen dataclass with points, colors, normals, intensity, metadata)
- [X] T011 Create PointCloudLoader stage class in examples/stages/pointcloud_loader.py with downsample_voxel_size, remove_outliers parameters
- [X] T012 Implement pre_process() method in PointCloudLoader to validate file existence and format
- [X] T013 Implement predict() method in PointCloudLoader with Open3D loading, downsampling, and outlier removal
- [X] T014 Implement post_process() method in PointCloudLoader to convert Open3D objects to NumPy arrays
- [X] T015 Add input_data_types and output_data_types properties to PointCloudLoader stage
- [X] T016 Create PointCloudValidationStage class in examples/stages/pointcloud_validation.py
- [X] T017 Implement validate_pointcloud() method in PointCloudValidationRules with checks for empty clouds, NaN, Inf, point count, and coordinate ranges
- [X] T018 Implement run() method in PointCloudValidationStage to execute validation rules and return validation results
- [X] T019 Update OutputSaver file format detection to map point cloud objects to .pcd/.ply/.xyz extensions based on config
- [X] T020 Create example pipeline config examples/configs/pointcloud_simple.yaml demonstrating point cloud loading and validation

**Checkpoint**: At this point, User Story 1 should be fully functional - can load, validate, and save point clouds through the pipeline ✅ COMPLETE

---

## Phase 4: User Story 2 - Visualize Point Clouds Interactively (Priority: P1)

**Goal**: Provide interactive 3D point cloud visualization with rotation, zoom, pan controls, and color-coding options using Plotly WebGL rendering.

**Independent Test**: Load any point cloud file and verify the interactive 3D viewer displays the data with working rotation/zoom controls. Viewer should maintain 30+ FPS for point clouds up to 500K points.

### Implementation for User Story 2

- [X] T021 [P] Create src/viz_art/visualization/ module directory with __init__.py
- [X] T022 [P] Create VisualizationConfig dataclass in src/viz_art/types/pointcloud.py with 3D viewer settings (point_size, opacity, color_mode, colorscale, background_color)
- [X] T023 Create src/viz_art/visualization/plotly_viewer.py module
- [X] T024 Implement create_3d_scatter() function in plotly_viewer.py to generate Plotly 3D scatter plot from point arrays
- [X] T025 Implement color_by_height() helper function in plotly_viewer.py to map Z coordinates to color scale
- [X] T026 Implement color_by_intensity() helper function in plotly_viewer.py to map intensity values to color scale
- [X] T027 Implement downsample_for_display() function in plotly_viewer.py to auto-downsample large point clouds for WebGL performance
- [X] T028 Create PointCloudVisualizationStage class in examples/stages/pointcloud_visualization.py
- [X] T029 Implement run() method in PointCloudVisualizationStage to generate Plotly figure with WebGL rendering mode
- [X] T030 Add logic to PointCloudVisualizationStage for handling large point clouds (auto-downsample if > max_render_points)
- [X] T031 Add output_html and output_json options to PointCloudVisualizationStage for exporting visualizations
- [X] T032 Create example script examples/test_visualization.py demonstrating interactive 3D viewer usage

**Checkpoint**: At this point, User Story 2 should be fully functional - can visualize point clouds interactively with smooth performance

---

## Phase 5: User Story 3 - View Stage-by-Stage Outputs (Priority: P1)

**Goal**: Enable inspection of intermediate outputs from each processing stage for both images and point clouds, with appropriate viewers (image viewer for images, 3D viewer for point clouds).

**Independent Test**: Run a multi-stage pipeline with both image and point cloud stages, then verify that each stage's output is displayed separately with appropriate viewers and saved by OutputSaver.

### Implementation for User Story 3

- [X] T033 Add generate_thumbnail() method to OutputSaver in src/viz_art/pipeline/output_saver.py to create preview images for point clouds
- [X] T034 Create src/viz_art/visualization/thumbnail.py module
- [X] T035 Implement render_thumbnail() function in thumbnail.py using Open3D OffscreenRenderer for headless rendering
- [X] T036 Implement get_viewpoint_params() helper in thumbnail.py to define camera positions for standard viewpoints (front, top, diagonal)
- [X] T037 Create PointCloudThumbnailStage class in examples/stages/pointcloud_thumbnail.py
- [X] T038 Implement run() method in PointCloudThumbnailStage to generate thumbnails from multiple viewpoints
- [X] T039 Extend HTMLReporter._format_stage_outputs() in src/viz_art/batch/reporter.py to detect point cloud outputs and render thumbnails
- [X] T040 Update batch_report.html template in src/viz_art/batch/templates/ to display point cloud thumbnails with links to full viewers
- [X] T041 Add point cloud thumbnail gallery view to batch_report.html template
- [X] T042 Implement embed_plotly_viewer() helper in HTMLReporter to inline interactive 3D viewers in HTML reports
- [X] T043 Create example multi-stage pipeline config examples/configs/multimodal_stages.yaml with image and point cloud stages
- [X] T044 Create example script examples/test_stage_outputs.py demonstrating stage-by-stage output inspection

**Checkpoint**: All core visualization user stories (US1, US2, US3) should now be independently functional and integrated

---

## Phase 6: User Story 4 - Overlay Point Clouds on Images (Priority: P2)

**Goal**: Project 3D point cloud data onto corresponding 2D images using camera calibration to verify sensor alignment and validate multi-modal fusion algorithms.

**Independent Test**: Load calibrated image and point cloud pairs, then verify the overlay view correctly projects 3D points onto the 2D image plane with accurate alignment (< 2px error).

### Implementation for User Story 4

- [X] T045 [P] Create src/viz_art/calibration/ module directory with __init__.py
- [X] T046 [P] Create CameraIntrinsics dataclass in src/viz_art/types/pointcloud.py
- [X] T047 [P] Create CameraExtrinsics dataclass in src/viz_art/types/pointcloud.py
- [X] T048 [P] Create Calibration dataclass in src/viz_art/types/pointcloud.py combining intrinsics and extrinsics
- [X] T049 Create src/viz_art/calibration/loader.py module
- [X] T050 Implement load_calibration() function in calibration/loader.py to parse YAML/JSON calibration files with OmegaConf
- [X] T051 Implement to_camera_matrix() method in CameraIntrinsics to convert parameters to OpenCV 3x3 matrix
- [X] T052 Implement to_rodrigues_vector() method in CameraExtrinsics to convert rotation matrix for OpenCV
- [X] T053 Create src/viz_art/calibration/projection.py module
- [X] T054 Implement project_points() function in calibration/projection.py using cv2.projectPoints() with distortion handling
- [X] T055 Implement filter_visible_points() helper in calibration/projection.py to mask points outside image bounds
- [X] T056 Create PointCloudProjectionStage class in examples/stages/pointcloud_projection.py
- [X] T057 Implement pre_process() in PointCloudProjectionStage to load and validate calibration
- [X] T058 Implement predict() in PointCloudProjectionStage to project 3D points to 2D image coordinates
- [X] T059 Implement post_process() in PointCloudProjectionStage to render projected points onto image with color/depth coding
- [X] T060 Add transparency blending options to PointCloudProjectionStage for adjustable image/point cloud overlay
- [X] T061 Create example calibration file examples/calibration/camera.yaml with sample intrinsic and extrinsic parameters
- [X] T062 Create example script examples/test_projection.py demonstrating overlay visualization

**Checkpoint**: User Story 4 should be fully functional - can overlay point clouds on images with accurate projection

---

## Phase 7: User Story 5 - Validate Multi-Modal Data Quality (Priority: P2)

**Goal**: Provide automatic validation of point cloud inputs to catch data quality issues (empty clouds, invalid coordinates, corrupted data) before expensive processing.

**Independent Test**: Provide various corrupted or invalid point clouds and verify the validation framework correctly identifies and reports each issue type with clear error messages.

### Implementation for User Story 5

- [X] T063 Add validation error reporting to PointCloudValidationStage in examples/stages/pointcloud_validation.py
- [X] T064 Implement fail_fast vs fail_gracefully modes in PointCloudValidationRules validation logic
- [X] T065 Add coordinate range validation to validate_pointcloud() method in PointCloudValidationRules
- [X] T066 Implement clear error messages for validation failures indicating specific issue and location
- [X] T067 Add configurable thresholds for min/max point count in PointCloudValidationRules
- [X] T068 Create MultiModalInput dataclass in src/viz_art/types/pointcloud.py for synchronized data pairs
- [X] T069 Create MultiModalLoaderStage class in examples/stages/multimodal_loader.py
- [X] T070 Implement timestamp synchronization validation in MultiModalLoaderStage with configurable sync_tolerance_ms
- [X] T071 Implement require_both validation mode in MultiModalLoaderStage to enforce both image and point cloud presence
- [X] T072 Add metadata tracking for validation results (num_points, has_nan, has_inf, coord_ranges) in PointCloudValidationStage output
- [X] T073 Create example config examples/configs/validation_strict.yaml demonstrating strict validation rules
- [X] T074 Create example script examples/test_validation.py with various corrupted test cases

**Checkpoint**: User Story 5 should be fully functional - validation catches all data quality issues with informative error messages

---

## Phase 8: User Story 6 - Generate Point Cloud Thumbnails for Reports (Priority: P3)

**Goal**: Automatically generate thumbnail images of point clouds in batch processing reports to quickly preview 3D data without opening interactive viewers.

**Independent Test**: Run batch processing with point cloud data and verify the HTML report contains rendered thumbnail images from multiple viewpoints that link to full interactive viewers.

### Implementation for User Story 6

- [X] T075 Add multi-viewpoint support to render_thumbnail() in src/viz_art/visualization/thumbnail.py (front, top, side, diagonal)
- [X] T076 Implement configurable thumbnail quality settings in VisualizationConfig (low/medium/high affecting render time)
- [X] T077 Add thumbnail caching to avoid regenerating thumbnails for repeated views
- [X] T078 Implement thumbnail click-through functionality in batch_report.html to open full interactive viewer
- [X] T079 Add progress indication for thumbnail generation in batch processing
- [X] T080 Optimize thumbnail generation to complete within 3s per cloud (performance target SC-006)
- [X] T081 Add fallback rendering for large point clouds that auto-downsamples before thumbnail generation
- [X] T082 Create example batch config examples/configs/batch_pointclouds.yaml for processing multiple point clouds
- [X] T083 Create example script examples/test_batch_thumbnails.py demonstrating batch report with thumbnails

**Checkpoint**: All user stories should now be independently functional and fully integrated

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [X] T084 [P] Add comprehensive docstrings to all public functions in src/viz_art/visualization/
- [X] T085 [P] Add comprehensive docstrings to all public functions in src/viz_art/calibration/
- [X] T086 [P] Add comprehensive docstrings to all example stages in examples/stages/
- [X] T087 [P] Update README.md with point cloud examples and getting started guide
- [X] T088 [P] Create examples/README.md documenting all example stages and configurations
- [X] T089 Add error handling improvements across all stages with consistent error message formats
- [X] T090 Add logging statements to all stages for debugging and monitoring
- [X] T091 Performance optimization: verify SC-001 (load 100K points < 5s)
- [X] T092 Performance optimization: verify SC-002 (interactive viewer 30+ FPS for 500K points)
- [X] T093 Performance optimization: verify SC-004 (projection accuracy within 2 pixels)
- [X] T094 Performance optimization: verify SC-006 (thumbnail generation < 3s per cloud)
- [X] T095 Validate backward compatibility with Phase 1 image-only pipelines
- [X] T096 Run all quickstart.md examples to verify functionality
- [X] T097 Code cleanup: remove debug code and add type hints
- [X] T098 Security review: validate file path handling and prevent directory traversal

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-8)**: All depend on Foundational phase completion
  - US1 (Phase 3): Can start after Foundational - No dependencies on other stories
  - US2 (Phase 4): Can start after Foundational - No dependencies on other stories
  - US3 (Phase 5): Depends on US2 for visualization components
  - US4 (Phase 6): Can start after Foundational - No dependencies on other stories
  - US5 (Phase 7): Depends on US1 for validation infrastructure
  - US6 (Phase 8): Depends on US2 for thumbnail rendering and US3 for HTML report integration
- **Polish (Phase 9)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1) - Process Point Cloud**: Can start after Foundational (Phase 2) - INDEPENDENT
- **User Story 2 (P1) - Visualize Interactively**: Can start after Foundational (Phase 2) - INDEPENDENT
- **User Story 3 (P1) - Stage-by-Stage Outputs**: Depends on US2 for visualization components
- **User Story 4 (P2) - Overlay on Images**: Can start after Foundational (Phase 2) - INDEPENDENT
- **User Story 5 (P2) - Validate Quality**: Depends on US1 for validation infrastructure
- **User Story 6 (P3) - Generate Thumbnails**: Depends on US2 (thumbnails) and US3 (HTML integration)

### Within Each User Story

- Dataclasses and type definitions before stages that use them
- Helper functions before stages that call them
- Core stage implementation before example configs/scripts
- OutputSaver extensions before batch processing integration

### Parallel Opportunities

- **Phase 1 (Setup)**: T001 and tasks T002-T004 can run in parallel
- **Phase 2 (Foundational)**: T005-T006 can run in parallel; T007-T008 can run in parallel after T005-T006
- **Phase 3 (US1)**: T009-T010 can run in parallel
- **Phase 4 (US2)**: T021-T022 can run in parallel
- **Phase 6 (US4)**: T045-T048 can run in parallel
- **Phase 9 (Polish)**: T084-T088 can run in parallel; T091-T094 can run in parallel
- **User Stories US1, US2, US4**: Can be worked on in parallel by different team members after Foundational phase completes
- Once US1 completes, US5 can start in parallel with ongoing work
- Once US2 completes, US3 and US6 can start in parallel with ongoing work

---

## Parallel Example: User Story 2 (Visualization)

```bash
# Launch parallel tasks for dataclasses and module creation:
T021: "Create src/viz_art/visualization/ module directory"
T022: "Create VisualizationConfig dataclass"

# Then launch parallel helper functions:
T025: "Implement color_by_height() helper function"
T026: "Implement color_by_intensity() helper function"
T027: "Implement downsample_for_display() function"
```

---

## Implementation Strategy

### MVP First (User Stories 1 & 2 Only)

1. Complete Phase 1: Setup (T001-T004)
2. Complete Phase 2: Foundational (T005-T008) - CRITICAL
3. Complete Phase 3: User Story 1 (T009-T020)
4. Complete Phase 4: User Story 2 (T021-T032)
5. **STOP and VALIDATE**: Test US1 and US2 independently
6. Deploy/demo point cloud loading and visualization capabilities

This MVP delivers core point cloud support: loading, processing, and interactive visualization.

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add US1 (Process Point Cloud) → Test independently → Deploy/Demo
3. Add US2 (Visualize) → Test independently → Deploy/Demo
4. Add US3 (Stage-by-Stage) → Test independently → Deploy/Demo (Completes all P1 stories)
5. Add US4 (Overlay) → Test independently → Deploy/Demo
6. Add US5 (Validate) → Test independently → Deploy/Demo (Completes all P2 stories)
7. Add US6 (Thumbnails) → Test independently → Deploy/Demo (Completes all P3 stories)

Each story adds value without breaking previous stories.

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 (Process Point Cloud)
   - Developer B: User Story 2 (Visualize)
   - Developer C: User Story 4 (Overlay)
3. As stories complete:
   - Developer A moves to US5 (after US1)
   - Developer B moves to US3 (after US2)
   - Developer C moves to US6 (after US2 and US3)

---

## Task Statistics

- **Total Tasks**: 98
- **Setup Phase**: 4 tasks
- **Foundational Phase**: 4 tasks (BLOCKING)
- **User Story 1**: 12 tasks
- **User Story 2**: 12 tasks
- **User Story 3**: 12 tasks
- **User Story 4**: 18 tasks
- **User Story 5**: 12 tasks
- **User Story 6**: 9 tasks
- **Polish Phase**: 15 tasks

**Parallelizable Tasks**: 23 tasks marked with [P]

**Suggested MVP Scope**: Phase 1 + Phase 2 + Phase 3 (US1) + Phase 4 (US2) = 32 tasks
This delivers immediate value: loading and visualizing point clouds.

---

## Success Criteria Mapping

- **SC-001** (Load 100K points < 5s): Verified in T091
- **SC-002** (Viewer 30+ FPS): Verified in T092
- **SC-003** (Display < 2s): Covered by US2 implementation
- **SC-004** (Projection accuracy < 2px): Verified in T093
- **SC-005** (Validation 100% detection): Covered by US5 implementation
- **SC-006** (Thumbnail < 3s): Verified in T094
- **SC-007** (Stability 100 runs): Covered by T095 (backward compatibility)
- **SC-008** (Usability 90% success): Covered by quickstart validation in T096
- **SC-009** (Interaction < 100ms): Covered by US2 Plotly implementation
- **SC-010** (Thumbnail quality): Covered by US6 implementation

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label (US1-US6) maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- All file paths are absolute from repository root
- Tests not included per specification (not explicitly requested)
- Backward compatibility maintained with Phase 1 (image-only pipelines)
