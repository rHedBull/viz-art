# Tasks: Accuracy Tracking & Analysis System

**Input**: Design documents from `/specs/004-accuracy-tracking-analysis/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Tests are included for integration validation only (per typical ML pipeline workflows).

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [X] T001 Create project structure per implementation plan (src/viz_art/accuracy/, src/viz_art/error_analysis/, tests/accuracy/, tests/error_analysis/)
- [X] T002 [P] Add project dependencies to pyproject.toml (scikit-learn>=1.3, numpy>=1.24, open3d>=0.18, plotly>=5.0, jinja2>=3.0)
- [X] T003 [P] Create __init__.py files for new modules (src/viz_art/accuracy/__init__.py, src/viz_art/error_analysis/__init__.py)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T004 Implement AnnotationFormat enum and GroundTruthDataset dataclass in src/viz_art/accuracy/ground_truth.py
- [X] T005 Implement GroundTruthSample dataclass in src/viz_art/accuracy/ground_truth.py
- [X] T006 Implement AccuracyCounts dataclass in src/viz_art/accuracy/metrics.py
- [X] T007 Implement MetricType enum and AccuracyMetrics dataclass in src/viz_art/accuracy/metrics.py
- [X] T008 Implement ErrorType and ErrorSeverity enums in src/viz_art/error_analysis/patterns.py
- [X] T009 Implement ErrorCase dataclass in src/viz_art/error_analysis/patterns.py
- [X] T010 Implement ErrorPattern dataclass in src/viz_art/error_analysis/patterns.py
- [X] T011 Create storage schema utilities for Parquet metrics in src/viz_art/accuracy/metrics_storage.py
- [X] T012 Extend existing OutputSaver to support error artifact saving in validation mode

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - View Per-Stage Accuracy in Pipeline Reports (Priority: P1) 🎯 MVP

**Goal**: After pipeline execution completes, users can view per-stage accuracy metrics (correct/wrong/invalid/unlabeled counts, precision/recall/F1/mAP/IoU) in generated HTML reports with ground truth comparisons.

**Independent Test**: Run a single sample through a pipeline with one ground truth label and verify accuracy metrics appear in the generated report after execution.

### Ground Truth Loading for User Story 1

- [X] T013 [P] [US1] Implement GroundTruthLoader.load_dataset() for COCO format in src/viz_art/accuracy/ground_truth.py
- [X] T014 [P] [US1] Implement GroundTruthLoader.load_dataset() for PASCAL_VOC format in src/viz_art/accuracy/ground_truth.py
- [X] T015 [P] [US1] Implement GroundTruthLoader.load_dataset() for PCD_LABELS format in src/viz_art/accuracy/ground_truth.py
- [X] T016 [US1] Implement GroundTruthLoader.load_sample() in src/viz_art/accuracy/ground_truth.py (depends on T013-T015)
- [X] T017 [US1] Implement GroundTruthLoader.validate_dataset() in src/viz_art/accuracy/ground_truth.py
- [X] T018 [US1] Implement GroundTruthValidator.validate_sample() in src/viz_art/accuracy/ground_truth.py
- [X] T019 [US1] Implement GroundTruthValidator.check_completeness() in src/viz_art/accuracy/ground_truth.py

### Metrics Calculation for User Story 1

- [X] T020 [P] [US1] Implement MetricsCalculator.calculate_precision_recall_f1() using scikit-learn in src/viz_art/accuracy/metrics.py
- [X] T021 [P] [US1] Extend MetricsCalculator.calculate_mean_average_precision() for accuracy tracking (reuse existing implementation from src/viz_art/validation/metrics.py)
- [X] T022 [P] [US1] Extend MetricsCalculator.calculate_iou() for segmentation accuracy (reuse existing implementation from src/viz_art/validation/metrics.py)
- [X] T023 [P] [US1] Implement MetricsCalculator.calculate_chamfer_distance() using Open3D in src/viz_art/accuracy/metrics.py

### Comparison Engine for User Story 1

- [X] T024 [US1] Implement ComparisonEngine.compare_single() for detection tasks in src/viz_art/accuracy/comparison.py
- [X] T025 [US1] Implement ComparisonEngine.compare_single() for classification tasks in src/viz_art/accuracy/comparison.py
- [X] T026 [US1] Implement ComparisonEngine.compare_single() for segmentation tasks in src/viz_art/accuracy/comparison.py
- [X] T027 [US1] Implement ComparisonEngine.compare_single() for point cloud tasks in src/viz_art/accuracy/comparison.py
- [X] T028 [US1] Implement ComparisonEngine.compare_batch() in src/viz_art/accuracy/comparison.py (depends on T024-T027)
- [X] T029 [US1] Implement ComparisonEngine.aggregate_results() in src/viz_art/accuracy/comparison.py

### Report Generation for User Story 1

- [X] T030 [US1] Create Jinja2 HTML template for per-run reports in src/viz_art/accuracy/templates/per_run_report.html
- [X] T031 [US1] Implement ReportGenerator.generate_per_run_report() in src/viz_art/accuracy/reporter.py (depends on T030)
- [X] T032 [US1] Add Plotly chart generation for per-stage metrics in src/viz_art/accuracy/reporter.py
- [X] T033 [US1] Implement accuracy threshold warnings in per-run report in src/viz_art/accuracy/reporter.py

### Integration for User Story 1

- [X] T034 [US1] Create AccuracyTracker orchestrator class in src/viz_art/accuracy/__init__.py
- [X] T035 [US1] Implement AccuracyTracker.run_validation() workflow in src/viz_art/accuracy/__init__.py (depends on T013-T033)
- [ ] T036 [US1] Integrate AccuracyTracker with existing Pipeline class for validation mode
- [X] T037 [US1] Store accuracy metrics to Parquet using existing PyArrow infrastructure in src/viz_art/accuracy/metrics_storage.py

### Testing for User Story 1

- [X] T038 [US1] Create integration test for end-to-end validation workflow in tests/integration/test_accuracy_workflow.py
- [X] T039 [US1] Create ground truth fixtures (COCO, PASCAL_VOC, PCD) in tests/fixtures/ground_truth/
- [ ] T040 [US1] Run quickstart.md validation for User Story 1 scenarios

**Checkpoint**: At this point, User Story 1 should be fully functional - users can view per-stage accuracy in generated reports after pipeline execution

---

## Phase 4: User Story 2 - Browse and Debug Error Cases (Priority: P2)

**Goal**: Users can click a link in the per-run HTML report to open an error case browser that loads saved error artifacts, shows side-by-side predictions vs ground truth with diff visualizations, and automatically categorizes errors by type and severity.

**Independent Test**: Create a pipeline run with known failure cases, verify the error browser loads saved artifacts, and check that side-by-side visualizations display correctly.

### Error Detection for User Story 2

- [X] T041 [P] [US2] Implement ErrorDetector.categorize_error() logic per spec assumptions in src/viz_art/error_analysis/patterns.py
- [X] T042 [US2] Implement ErrorDetector.detect_errors() in src/viz_art/error_analysis/patterns.py (depends on T041)
- [X] T043 [US2] Integrate ErrorDetector with ComparisonEngine results in src/viz_art/error_analysis/patterns.py

### Error Pattern Detection for User Story 2

- [X] T044 [US2] Implement ErrorPatternDetector.cluster_errors() using rule-based grouping in src/viz_art/error_analysis/patterns.py
- [X] T045 [US2] Implement ErrorPatternDetector.summarize_patterns() in src/viz_art/error_analysis/patterns.py
- [X] T046 [US2] Add configurable grouping rules via YAML config in src/viz_art/error_analysis/patterns.py

### Error Visualization for User Story 2

- [X] T047 [P] [US2] Implement ErrorVisualizer.create_image_diff() with color-coded regions in src/viz_art/error_analysis/visualizer.py
- [X] T048 [P] [US2] Implement ErrorVisualizer.create_pointcloud_diff() using Open3D nearest-neighbor in src/viz_art/error_analysis/visualizer.py
- [X] T049 [P] [US2] Add optional ICP fallback for point cloud diff in src/viz_art/error_analysis/visualizer.py
- [X] T050 [US2] Implement ErrorVisualizer.create_side_by_side_visualization() in src/viz_art/error_analysis/visualizer.py (depends on T047-T049)
- [X] T051 [US2] Implement ErrorVisualizer.create_diff_visualization() with auto mode in src/viz_art/error_analysis/visualizer.py

### Error Browser UI for User Story 2

- [X] T052 [US2] Implement ErrorBrowser.load_errors() with indexed filtering in src/viz_art/error_analysis/browser.py
- [X] T053 [US2] Implement ErrorBrowser.get_error_by_id() in src/viz_art/error_analysis/browser.py
- [X] T054 [US2] Implement ErrorBrowser.export_errors() for JSON/CSV/Parquet in src/viz_art/error_analysis/export.py
- [X] T055 [US2] Create Streamlit UI component for error browser in src/viz_art/error_analysis/browser.py
- [X] T056 [US2] Add pagination and lazy loading for error thumbnails in src/viz_art/error_analysis/browser.py

### Integration for User Story 2

- [X] T057 [US2] Add error browser link to per-run HTML report template in src/viz_art/accuracy/templates/per_run_report.html
- [X] T058 [US2] Integrate ErrorDetector with AccuracyTracker validation workflow
- [X] T059 [US2] Save error artifacts during validation using extended OutputSaver
- [X] T060 [US2] Store error metadata to JSON files in output/errors/{run_id}_errors.json

### Testing for User Story 2

- [X] T061 [US2] Create unit tests for error detection and categorization in tests/error_analysis/test_patterns.py
- [X] T062 [US2] Create unit tests for error visualizations in tests/error_analysis/test_visualizer.py
- [ ] T063 [US2] Create integration test for error browser workflow in tests/error_analysis/test_browser.py
- [ ] T064 [US2] Run quickstart.md validation for User Story 2 scenarios

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently - users can view accuracy metrics AND browse/debug errors

---

## Phase 5: User Story 3 - Track Performance Trends Over Time (Priority: P3)

**Goal**: Users can open a historical comparison HTML report (linked from any per-run report) showing accuracy trends over time, compare pipeline versions, see confusion matrices, and identify regressions.

**Independent Test**: Run a pipeline multiple times with different configurations, verify historical data is stored, and check that trend charts display correctly in the static HTML historical report.

### Performance Tracking for User Story 3

- [X] T065 [P] [US3] Implement PerformanceTracker.record_metrics() in src/viz_art/performance/trends.py
- [X] T066 [P] [US3] Implement PerformanceTracker.get_trend() with Parquet queries in src/viz_art/performance/trends.py
- [X] T067 [P] [US3] Implement PerformanceTracker.compare_runs() in src/viz_art/performance/trends.py
- [X] T068 [US3] Create MetricSnapshot and TrendStatistics dataclasses in src/viz_art/performance/trends.py
- [X] T069 [US3] Implement PerformanceTrend storage schema in Parquet format in src/viz_art/performance/trends.py

### Regression Detection for User Story 3

- [X] T070 [US3] Implement RegressionDetector.detect_regression() with threshold-based logic in src/viz_art/performance/trends.py
- [X] T071 [US3] Implement RegressionDetector.get_baseline() with strategy selection in src/viz_art/performance/trends.py
- [X] T072 [US3] Add severity classification (high/medium/low) for regressions in src/viz_art/performance/trends.py

### Historical Report Generation for User Story 3

- [X] T073 [US3] Create Jinja2 HTML template for historical comparison reports in src/viz_art/accuracy/templates/historical_report.html
- [X] T074 [US3] Implement ReportGenerator.generate_historical_report() in src/viz_art/accuracy/reporter.py
- [X] T075 [US3] Add Plotly time-series charts for accuracy trends in src/viz_art/accuracy/reporter.py
- [X] T076 [US3] Implement ReportGenerator.generate_confusion_matrix() in src/viz_art/accuracy/reporter.py
- [X] T077 [US3] Add per-class performance breakdown charts in src/viz_art/accuracy/reporter.py
- [X] T078 [US3] Add regression highlights to historical report in src/viz_art/accuracy/reporter.py

### Integration for User Story 3

- [X] T079 [US3] Add historical report link to per-run HTML reports in src/viz_art/accuracy/templates/per_run_report.html
- [X] T080 [US3] Integrate PerformanceTracker with AccuracyTracker to auto-record metrics
- [X] T081 [US3] Implement historical data retention policy (90 days default) in src/viz_art/performance/metrics_storage.py
- [X] T082 [US3] Add TrendAnalyzer utility class for trend queries in src/viz_art/performance/dashboard.py

### Testing for User Story 3

- [X] T083 [US3] Create unit tests for regression detection in tests/performance/test_regression.py
- [X] T084 [US3] Create integration test for multi-run trend tracking in tests/integration/test_trends.py
- [ ] T085 [US3] Run quickstart.md validation for User Story 3 scenarios

**Checkpoint**: All user stories should now be independently functional - complete accuracy tracking and analysis system is operational

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [X] T086 [P] Add comprehensive docstrings to all public APIs following NumPy style
- [X] T087 [P] Create user documentation in docs/accuracy_tracking.md
- [X] T088 [P] Create developer documentation for extending metrics in docs/extending_metrics.md
- [X] T089 Update CLAUDE.md with new active technologies and commands
- [X] T090 [P] Add performance benchmarks for key operations (SC-001 through SC-010)
- [X] T091 [P] Add validation for edge cases (missing labels, corrupted data, format mismatches)
- [X] T092 Code cleanup and refactoring for consistency with existing viz-art patterns
- [X] T093 Add configuration validation and helpful error messages
- [X] T094 Create example configs for common use cases in examples/configs/
- [ ] T095 Run full quickstart.md validation end-to-end

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-5)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Phase 6)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - Integrates with US1 (error browser linked from US1 reports) but independently testable
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - Integrates with US1 (historical report linked from US1 reports) but independently testable

### Within Each User Story

**User Story 1 (Per-Stage Accuracy)**:
1. Ground truth loading (T013-T019) - Can run in parallel for different formats
2. Metrics calculation (T020-T023) - Can run in parallel, independent of ground truth
3. Comparison engine (T024-T029) - Sequential, builds on single→batch→aggregate
4. Report generation (T030-T033) - Sequential, template→implementation→features
5. Integration (T034-T037) - Sequential, orchestrator→workflow→integration→storage
6. Testing (T038-T040) - Can run in parallel after implementation complete

**User Story 2 (Error Browser)**:
1. Error detection (T041-T043) - Sequential, categorization→detection→integration
2. Pattern detection (T044-T046) - Sequential, clustering→summarization→config
3. Visualizations (T047-T051) - Image diff and point cloud diff can run in parallel (T047-T049), then combine (T050-T051)
4. Browser UI (T052-T056) - Sequential, loading→retrieval→export→UI→optimization
5. Integration (T057-T060) - Sequential, link→integrate→save→store
6. Testing (T061-T064) - Can run in parallel after implementation complete

**User Story 3 (Performance Trends)**:
1. Tracking (T065-T069) - T065-T067 can run in parallel, then T068-T069 sequential
2. Regression detection (T070-T072) - Sequential, detection→baseline→severity
3. Historical reports (T073-T078) - Template first (T073), then features can run in parallel
4. Integration (T079-T082) - Sequential, link→integrate→retention→utilities
5. Testing (T083-T085) - Can run in parallel after implementation complete

### Parallel Opportunities

**Setup Phase (Phase 1)**:
- All 3 tasks can run in parallel

**Foundational Phase (Phase 2)**:
- T004-T010 (dataclass definitions) can run in parallel
- T011-T012 (storage utilities) can run in parallel after dataclasses

**User Story 1**:
- T013-T015 (different annotation formats) can run in parallel
- T020-T023 (different metric types) can run in parallel
- T038-T039 (tests and fixtures) can run in parallel

**User Story 2**:
- T047-T049 (image and point cloud visualizations) can run in parallel
- T061-T063 (different test suites) can run in parallel

**User Story 3**:
- T065-T067 (tracking operations) can run in parallel
- T075-T077 (different chart types) can run in parallel
- T083-T084 (different test types) can run in parallel

**Polish Phase (Phase 6)**:
- T086-T088 (documentation) can run in parallel
- T090-T091 (validation and benchmarks) can run in parallel

---

## Parallel Example: User Story 1 Ground Truth Loading

```bash
# Launch all annotation format loaders together:
Task: "Implement GroundTruthLoader.load_dataset() for COCO format in src/viz_art/accuracy/ground_truth.py"
Task: "Implement GroundTruthLoader.load_dataset() for PASCAL_VOC format in src/viz_art/accuracy/ground_truth.py"
Task: "Implement GroundTruthLoader.load_dataset() for PCD_LABELS format in src/viz_art/accuracy/ground_truth.py"

# Then implement shared methods sequentially:
Task: "Implement GroundTruthLoader.load_sample() in src/viz_art/accuracy/ground_truth.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T003)
2. Complete Phase 2: Foundational (T004-T012) - CRITICAL - blocks all stories
3. Complete Phase 3: User Story 1 (T013-T040)
4. **STOP and VALIDATE**: Test User Story 1 independently with quickstart.md
5. Deploy/demo if ready - users can now view per-stage accuracy in reports

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP: accuracy reporting works!)
3. Add User Story 2 → Test independently → Deploy/Demo (error browsing and debugging works!)
4. Add User Story 3 → Test independently → Deploy/Demo (historical trends and regression detection works!)
5. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together (T001-T012)
2. Once Foundational is done:
   - **Developer A**: User Story 1 (T013-T040) - Accuracy reporting
   - **Developer B**: User Story 2 (T041-T064) - Error analysis
   - **Developer C**: User Story 3 (T065-T085) - Performance trends
3. Stories complete and integrate independently via links in reports

---

## Notes

- **[P] tasks** = different files, no dependencies - can run in parallel
- **[Story] label** maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify integration tests pass after each user story phase
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- **Tests focus on integration**: Unit tests for critical algorithms (error detection, metrics), integration tests for workflows
- **Reuse existing implementations**: mAP and IoU already exist in src/viz_art/validation/metrics.py - extend rather than rewrite
- **Performance targets**: Report generation <30s (SC-009), error browser <100ms filtering (SC-010), visualizations <2s images / <5s point clouds (SC-003)
- **Storage**: Use existing PyArrow/Parquet infrastructure from Phase 3 for metrics storage
- **Error artifacts**: Use existing OutputSaver system extended for validation mode
