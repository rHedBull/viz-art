# Tasks: Performance Monitoring & Debugging System

**Input**: Design documents from `/specs/003-multimodal-viz-phase3/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅, quickstart.md ✅

**Tests**: Not explicitly requested in feature specification - omitting test tasks

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: Repository root with `src/viz_art/`, `tests/`, `examples/`
- New modules: `src/viz_art/performance/`, `src/viz_art/audit/`, `src/viz_art/validation/`
- Paths based on plan.md structure

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and dependency installation for Phase 3 monitoring features

- [X] T001 Add Loguru, PyArrow, psutil, and scikit-learn dependencies to pyproject.toml
- [X] T002 [P] Create src/viz_art/performance/ module directory with __init__.py
- [X] T003 [P] Create src/viz_art/audit/ module directory with __init__.py
- [X] T004 [P] Create src/viz_art/validation/ module directory with __init__.py
- [X] T005 [P] Create output/metrics/ directory for Parquet storage
- [X] T006 [P] Create output/logs/ directory for JSON Lines logs
- [X] T007 [P] Create output/ground_truth/ directory for validation datasets
- [X] T008 [P] Create tests/fixtures/sample_metrics.parquet for test data
- [X] T009 [P] Create tests/fixtures/sample_audit_logs.json for test data
- [X] T010 [P] Create tests/fixtures/golden_datasets/ directory structure with subdirs: bboxes/, segmentation/, pointclouds/

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core data models and utilities that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

### Data Models (Pydantic Schemas)

- [X] T011 Create RunStatus enum in src/viz_art/types/monitoring.py with values: RUNNING, COMPLETED, FAILED
- [X] T012 Create Run Pydantic model in src/viz_art/types/monitoring.py with fields: run_id, timestamp, config_snapshot, input_files, status, error, stage_executions, total_duration_ms, output_dir
- [X] T013 Add UUID validator to Run model for run_id field validation
- [X] T014 Add status transition validator to Run model (RUNNING → COMPLETED/FAILED only)
- [X] T015 [P] Create StageExecution Pydantic model in src/viz_art/types/monitoring.py with fields: run_id, stage_name, stage_index, start_time, end_time, execution_time_ms, cpu_memory_mb, gpu_memory_mb, input_paths, output_paths, success, error_message, error_traceback
- [X] T016 Add time order validator to StageExecution model (end_time > start_time)
- [X] T017 Add execution_time_ms validator to StageExecution model (must match time delta ±1ms)
- [X] T018 [P] Create LogLevel enum in src/viz_art/types/monitoring.py with values: DEBUG, INFO, WARNING, ERROR, CRITICAL
- [X] T019 [P] Create AuditLog Pydantic model in src/viz_art/types/monitoring.py with fields: run_id, timestamp, level, stage_name, message, metadata
- [X] T020 Add to_jsonl() method to AuditLog model for JSON Lines serialization
- [X] T021 [P] Create AnnotationFormat enum in src/viz_art/types/monitoring.py with values: COCO, YOLO, PNG_MASKS, PLY_LABELS, CUSTOM
- [X] T022 [P] Create GroundTruthDataset Pydantic model in src/viz_art/types/monitoring.py with fields: dataset_id, name, description, root_path, annotation_format, annotation_files, sample_count, created_at, metadata
- [X] T023 Add file existence validator to GroundTruthDataset model for annotation_files
- [X] T024 [P] Create AccuracyMetrics Pydantic model in src/viz_art/types/monitoring.py with fields: run_id, stage_name, dataset_id, true_positives, false_positives, false_negatives, true_negatives, precision, recall, f1_score, mean_average_precision, iou_mean, sample_count, timestamp
- [X] T025 Add precision validator to AccuracyMetrics model (TP/(TP+FP) calculation check)
- [X] T026 Add recall validator to AccuracyMetrics model (TP/(TP+FN) calculation check)

### Core Utilities

- [X] T027 Create memory tracking utility in src/viz_art/utils/memory.py with get_cpu_memory_mb() function using psutil
- [X] T028 Add get_gpu_memory_mb() function to src/viz_art/utils/memory.py with pynvml integration (graceful degradation if unavailable)
- [X] T029 Create timing utility in src/viz_art/utils/timing.py with Timer context manager for millisecond-precision timing
- [X] T030 Create UUID generation utility in src/viz_art/utils/identifiers.py with generate_run_id() function returning UUID v4 string

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel ✅ COMPLETE

---

## Phase 3: User Story 1 - Performance Diagnosis (Priority: P1) 🎯 MVP

**Goal**: Enable developers to identify pipeline bottlenecks by viewing per-stage execution times and memory usage in a performance dashboard.

**Independent Test**: Run a 3-stage pipeline, then view the performance dashboard which displays execution time (milliseconds) and CPU/GPU memory usage for each stage. This functionality works standalone without requiring audit logs or ground truth validation.

### Implementation for User Story 1

#### Profiler Core

- [X] T031 [US1] Create Profiler class in src/viz_art/performance/profiler.py with __init__(storage, enabled) constructor
- [X] T032 [US1] Implement Profiler.__call__() decorator method to wrap stage execute() functions
- [X] T033 [US1] Implement Profiler.measure() context manager for explicit profiling blocks
- [X] T034 [US1] Add _record_metrics() private method to Profiler that captures timing and memory data
- [X] T035 [US1] Integrate Timer and memory tracking utilities into Profiler._record_metrics()

#### Metrics Storage (Parquet)

- [X] T036 [P] [US1] Create MetricsStorage class in src/viz_art/performance/metrics_storage.py with __init__(output_dir, retention_days) constructor
- [X] T037 [US1] Implement MetricsStorage.write_metrics() method to append metrics to Parquet files partitioned by stage_name
- [X] T038 [US1] Use PyArrow with Snappy compression in MetricsStorage.write_metrics()
- [X] T039 [US1] Implement MetricsStorage.query_metrics() method with filters: stage_name, start_date, end_date, run_ids
- [X] T040 [US1] Add PyArrow filter pushdown optimization to MetricsStorage.query_metrics() for date range queries
- [X] T041 [US1] Implement MetricsStorage.get_aggregate_stats() method calculating median, mean, std, min, max, p50, p95, p99
- [X] T042 [US1] Add Pandas integration to MetricsStorage for aggregate statistics computation

#### Performance Dashboard (Plotly)

- [X] T043 [P] [US1] Create PerformanceDashboard class in src/viz_art/performance/dashboard.py with __init__(storage) constructor
- [X] T044 [US1] Implement PerformanceDashboard.render_timing_chart() method generating horizontal bar chart with Plotly
- [X] T045 [US1] Add per-stage timing data loading to render_timing_chart() via storage.query_metrics()
- [X] T046 [US1] Implement PerformanceDashboard.render_memory_chart() method generating stacked area chart for CPU/GPU memory
- [X] T047 [US1] Add memory data loading and null GPU handling to render_memory_chart()
- [X] T048 [US1] Implement PerformanceDashboard.render_trend_chart() method generating line chart for historical performance
- [X] T049 [US1] Add date range filtering and trend line generation to render_trend_chart()
- [X] T050 [US1] Add HTML export functionality to all dashboard methods using plotly.io.write_html()

#### Public API (Factory Functions)

- [X] T051 [P] [US1] Implement create_profiler() factory function in src/viz_art/performance/__init__.py
- [X] T052 [P] [US1] Implement create_metrics_storage() factory function in src/viz_art/performance/__init__.py
- [X] T053 [P] [US1] Implement create_dashboard() factory function in src/viz_art/performance/__init__.py
- [X] T054 [US1] Add public exports to src/viz_art/performance/__init__.py: create_profiler, create_metrics_storage, create_dashboard, Profiler, MetricsStorage, PerformanceDashboard

#### Integration Example

- [X] T055 [US1] Create examples/phase3/example_profiling.py demonstrating @profiler decorator usage on sample stage
- [X] T056 [US1] Add performance dashboard generation to examples/phase3/example_profiling.py with HTML output
- [X] T057 [US1] Create examples/configs/profiling_enabled.yaml config with enable_profiling: true flag

**Checkpoint**: At this point, User Story 1 should be fully functional - can profile stages, store metrics, and view performance dashboards ✅ COMPLETE

---

## Phase 4: User Story 2 - Debugging Failed Runs (Priority: P1)

**Goal**: Enable developers to trace failed pipeline runs back to exact error messages and input files using structured audit logs queryable by run ID, date, stage, and status.

**Independent Test**: Trigger a pipeline failure (e.g., invalid input), then query audit logs by run ID to retrieve timestamp, stage name, error message, and performance metrics. Export filtered logs to JSON. This works without requiring performance profiling or ground truth validation.

### Implementation for User Story 2

#### Loguru Integration

- [X] T058 [P] [US2] Create AuditLogger class in src/viz_art/audit/logger.py with __init__(run_id, output_dir, rotation, retention) constructor
- [X] T059 [US2] Configure Loguru in AuditLogger.__init__() with JSON serialization and automatic rotation
- [X] T060 [US2] Set up date-based file naming pattern in Loguru config: output/logs/{YYYY-MM-DD}.jsonl
- [X] T061 [US2] Add log rotation configuration with rotation parameter (default "100 MB")
- [X] T062 [US2] Add retention configuration with retention parameter (default "30 days")
- [X] T063 [US2] Implement AuditLogger.debug/info/warning/error/critical() methods wrapping Loguru logger
- [X] T064 [US2] Add automatic run_id context binding to all log methods using logger.bind()
- [X] T065 [US2] Implement AuditLogger.bind(**context) method for additional context (e.g., stage_name)
- [X] T066 [US2] Add structured metadata support to log methods via **kwargs

#### Run Tracking

- [X] T067 [P] [US2] Create RunTracker class in src/viz_art/audit/run_tracker.py with __init__(output_dir) constructor
- [X] T068 [US2] Implement RunTracker.track() context manager yielding run_id (UUID v4)
- [X] T069 [US2] Create Run model instance in RunTracker.track() __enter__() method with status=RUNNING
- [X] T070 [US2] Save run metadata to output/runs/{run_id}/run_metadata.json on context entry
- [X] T071 [US2] Update Run status to COMPLETED in RunTracker.track() __exit__() on success
- [X] T072 [US2] Update Run status to FAILED in RunTracker.track() __exit__() on exception with error message
- [X] T073 [US2] Calculate total_duration_ms in RunTracker.track() __exit__() from context entry/exit timestamps
- [X] T074 [US2] Implement RunTracker.get_run_metadata() method loading run_metadata.json by run_id
- [X] T075 [US2] Add error handling for missing/corrupted run_metadata.json files

#### Audit Query Interface

- [X] T076 [P] [US2] Create AuditQuery class in src/viz_art/audit/query.py with fluent builder pattern
- [X] T077 [US2] Implement AuditQuery.run_id() method adding run_id filter
- [X] T078 [US2] Implement AuditQuery.stage() method adding stage_name filter
- [X] T079 [US2] Implement AuditQuery.level() method adding log level filter
- [X] T080 [US2] Implement AuditQuery.after() method adding timestamp >= filter
- [X] T081 [US2] Implement AuditQuery.before() method adding timestamp <= filter
- [X] T082 [US2] Implement AuditQuery.failed() method filtering to ERROR and CRITICAL levels
- [X] T083 [US2] Implement AuditQuery.limit() method capping result count
- [X] T084 [US2] Implement AuditQuery.fetch() method executing query and returning List[Dict]
- [X] T085 [US2] Add date-based file scanning to AuditQuery.fetch() (only load relevant .jsonl files)
- [X] T086 [US2] Add JSON Lines parsing to AuditQuery.fetch() with filter application
- [X] T087 [US2] Implement AuditQuery.export_json() method writing filtered results to file
- [X] T088 [US2] Add result count return value to AuditQuery.export_json()

#### Public API (Factory Functions)

- [X] T089 [P] [US2] Implement create_logger() factory function in src/viz_art/audit/__init__.py
- [X] T090 [P] [US2] Implement create_run_tracker() factory function in src/viz_art/audit/__init__.py
- [X] T091 [P] [US2] Implement create_query() factory function in src/viz_art/audit/__init__.py
- [X] T092 [US2] Add public exports to src/viz_art/audit/__init__.py: create_logger, create_run_tracker, create_query, AuditLogger, RunTracker, AuditQuery, LogLevel

#### Integration Example

- [X] T093 [US2] Create examples/phase3/example_logging.py demonstrating RunTracker context manager usage
- [X] T094 [US2] Add structured logging examples to example_logging.py with different log levels and metadata
- [X] T095 [US2] Add query examples to example_logging.py showing filter combinations (date range, stage, failed)
- [X] T096 [US2] Add JSON export example to example_logging.py

**Checkpoint**: At this point, User Story 2 should be fully functional - can track runs, log structured data, and query/export logs ✅ COMPLETE

---

## Phase 5: User Story 3 - Ground Truth Validation (Priority: P2)

**Goal**: Enable QA engineers to validate pipeline accuracy against labeled test data, viewing per-stage metrics (precision, recall, F1, mAP) and an error analysis browser with side-by-side prediction vs ground truth visualizations.

**Independent Test**: Load a COCO-format ground truth dataset with 50 labeled samples, run pipeline validation, and view accuracy metrics in the UI. Access error analysis browser to filter failure cases by stage and error type. This works independently using the existing OutputSaver for artifacts.

### Implementation for User Story 3

#### Ground Truth Dataset Management

- [X] T097 [P] [US3] Create GroundTruthDataset class in src/viz_art/validation/dataset.py with __init__(dataset_id, root_path, annotation_format, annotation_files, name, metadata) constructor
- [X] T098 [US3] Add file validation in GroundTruthDataset.__init__() checking annotation_files exist under root_path
- [X] T099 [US3] Implement GroundTruthDataset.get_annotation() method looking up annotation by sample_id
- [X] T100 [US3] Implement GroundTruthDataset.iter_samples() generator yielding (sample_id, annotation) tuples
- [X] T101 [US3] Add sample_count property to GroundTruthDataset counting total annotations

#### Annotation Format Loaders

- [X] T102 [P] [US3] Create FormatLoader protocol interface in src/viz_art/validation/loaders.py with format_name, load(), validate() methods
- [X] T103 [P] [US3] Create COCOLoader class in src/viz_art/validation/loaders.py implementing FormatLoader
- [X] T104 [US3] Implement COCOLoader.load() parsing COCO JSON with image_id → annotations mapping
- [X] T105 [US3] Implement COCOLoader.validate() checking for required fields: images, annotations, categories
- [X] T106 [P] [US3] Create YOLOLoader class in src/viz_art/validation/loaders.py implementing FormatLoader
- [X] T107 [US3] Implement YOLOLoader.load() parsing .txt files with bbox format: class x_center y_center width height
- [X] T108 [US3] Implement YOLOLoader.validate() checking coordinate ranges [0, 1]
- [X] T109 [P] [US3] Create PNGMaskLoader class in src/viz_art/validation/loaders.py implementing FormatLoader
- [X] T110 [US3] Implement PNGMaskLoader.load() loading PNG segmentation masks with PIL
- [X] T111 [US3] Implement PNGMaskLoader.validate() checking mask dimensions match image dimensions
- [X] T112 [P] [US3] Create PLYLabelLoader class in src/viz_art/validation/loaders.py implementing FormatLoader
- [X] T113 [US3] Implement PLYLabelLoader.load() loading PLY/PCD point clouds with Open3D and extracting label attributes
- [X] T114 [US3] Implement PLYLabelLoader.validate() checking point cloud has label field
- [X] T115 [US3] Create loader registry dict in src/viz_art/validation/loaders.py mapping format_name → loader instance
- [X] T116 [US3] Implement register_format_loader() function in loaders.py for custom loader plugins
- [X] T117 [US3] Register all built-in loaders (COCO, YOLO, PNGMask, PLYLabel) on module import

#### Metrics Calculation

- [X] T118 [P] [US3] Create MetricsCalculator class in src/viz_art/validation/metrics.py with __init__(metric_type) constructor
- [X] T119 [US3] Implement MetricsCalculator.calculate_precision_recall() using sklearn.metrics.precision_recall_fscore_support
- [X] T120 [US3] Add confusion matrix extraction to calculate_precision_recall() for TP/FP/FN/TN counts
- [X] T121 [US3] Implement MetricsCalculator.calculate_mean_average_precision() for detection tasks
- [X] T122 [US3] Add IoU calculation helper function _calculate_iou() for bbox matching in calculate_mean_average_precision()
- [X] T123 [US3] Add confidence-based sorting and precision-recall curve generation to calculate_mean_average_precision()
- [X] T124 [US3] Implement MetricsCalculator.calculate_iou() for segmentation tasks using mask overlap
- [X] T125 [US3] Add per-class IoU calculation with mean aggregation to calculate_iou()
- [X] T126 [US3] Add metric_type auto-detection based on annotation format (classification/detection/segmentation)

#### Error Analysis

- [X] T127 [P] [US3] Create ErrorAnalyzer class in src/viz_art/validation/error_analysis.py with __init__(output_dir) constructor
- [X] T128 [US3] Implement ErrorAnalyzer.find_failures() method loading predictions and ground truth by run_id
- [X] T129 [US3] Add IoU/confidence threshold filtering to find_failures() for determining failures
- [X] T130 [US3] Add error type categorization to find_failures(): false_positive, false_negative, misclassification
- [X] T131 [US3] Implement ErrorAnalyzer.categorize_errors() grouping failures by error_type
- [X] T132 [US3] Implement ErrorAnalyzer.visualize_comparison() for images with bbox/mask overlay
- [X] T133 [US3] Add PIL/OpenCV drawing for visualize_comparison() showing pred (red) vs ground truth (green)
- [X] T134 [US3] Add image diff visualization to visualize_comparison() with color-coded pixel differences
- [X] T135 [US3] Implement visualize_comparison() for point clouds with Open3D colored by distance error
- [X] T136 [US3] Add distance heatmap calculation for point cloud diff visualization

#### Pipeline Validation Integration

- [X] T137 [US3] Implement validate_pipeline() function in src/viz_art/validation/__init__.py
- [X] T138 [US3] Add stage output loading from OutputSaver paths in validate_pipeline()
- [X] T139 [US3] Add annotation loading from GroundTruthDataset in validate_pipeline()
- [X] T140 [US3] Add per-stage metric calculation in validate_pipeline() using MetricsCalculator
- [X] T141 [US3] Create AccuracyMetrics model instances in validate_pipeline() for each stage
- [X] T142 [US3] Return Dict[stage_name, metrics_dict] from validate_pipeline()

#### Public API (Factory Functions)

- [X] T143 [P] [US3] Implement create_dataset() factory function in src/viz_art/validation/__init__.py
- [X] T144 [P] [US3] Implement create_metrics_calculator() factory function in src/viz_art/validation/__init__.py
- [X] T145 [P] [US3] Implement create_error_analyzer() factory function in src/viz_art/validation/__init__.py
- [X] T146 [US3] Add public exports to src/viz_art/validation/__init__.py: create_dataset, create_metrics_calculator, create_error_analyzer, validate_pipeline, register_format_loader, AnnotationFormat

#### Integration Example

- [X] T147 [US3] Create examples/phase3/example_validation.py demonstrating COCO dataset loading
- [X] T148 [US3] Add validation pipeline execution to example_validation.py with sample predictions
- [X] T149 [US3] Add accuracy metrics display to example_validation.py showing precision/recall/F1/mAP
- [X] T150 [US3] Add error analysis example to example_validation.py with failure categorization and visualization
- [X] T151 [US3] Create tests/fixtures/golden_datasets/bboxes/coco_sample.json with 10 sample COCO annotations
- [X] T152 [US3] Create tests/fixtures/golden_datasets/segmentation/masks/ with 5 sample PNG masks
- [X] T153 [US3] Create tests/fixtures/golden_datasets/pointclouds/labeled.ply with sample labeled point cloud

**Checkpoint**: At this point, User Story 3 should be fully functional - can load ground truth, calculate accuracy, and analyze errors

---

## Phase 6: User Story 4 - Performance Metrics Storage (Priority: P3)

**Goal**: Enable team leads to track pipeline performance over weeks/months to identify regressions and verify improvements, with automatic cleanup of expired data based on retention policies.

**Independent Test**: Run pipeline 10 times over 3 days, verify metrics are saved to Parquet files, query historical data by date range, and confirm aggregate statistics (median, p95, p99) are displayed. Verify retention policy auto-deletes metrics older than configured threshold. This builds on US1 profiling infrastructure.

**Note**: This user story primarily extends US1 with long-term storage and retention features, so many core tasks are already complete from Phase 3.

### Implementation for User Story 4

#### Retention and Cleanup

- [X] T154 [US4] Add _cleanup_old_metrics() private method to MetricsStorage class in src/viz_art/performance/metrics_storage.py
- [X] T155 [US4] Implement date-based file scanning in _cleanup_old_metrics() to find Parquet files older than retention_days
- [X] T156 [US4] Add automatic cleanup trigger in MetricsStorage.write_metrics() calling _cleanup_old_metrics() periodically
- [X] T157 [US4] Add disk space monitoring helper function _get_disk_usage() to MetricsStorage
- [X] T158 [US4] Add warning logging in MetricsStorage.write_metrics() when disk usage exceeds threshold (e.g., 90%)

#### Historical Trend Visualization

- [X] T159 [US4] Add aggregate_by parameter to PerformanceDashboard.render_trend_chart() for grouping (daily/weekly/monthly)
- [X] T160 [US4] Implement date aggregation logic in render_trend_chart() using Pandas groupby
- [X] T161 [US4] Add confidence interval bands to trend chart (e.g., ±1 std dev)
- [X] T162 [US4] Add regression detection highlighting in trend chart for performance degradations >10%

#### Long-Term Storage Optimization

- [~] T163 (OPTIONAL - SKIPPED) [US4] Add Parquet file compaction utility in src/viz_art/performance/compaction.py
- [~] T164 (OPTIONAL - SKIPPED) [US4] Implement compact_metrics() function merging small Parquet files into larger partitions
- [~] T165 (OPTIONAL - SKIPPED) [US4] Add index file generation in compact_metrics() for fast run_id lookups: .index.json per stage

#### Integration Example

- [X] T166 [US4] Create examples/phase3/example_historical_metrics.py demonstrating multi-day metric accumulation
- [X] T167 [US4] Add retention policy configuration example to example_historical_metrics.py
- [X] T168 [US4] Add aggregate statistics calculation example to example_historical_metrics.py (median, p95, p99)
- [X] T169 [US4] Add trend chart generation example with date range filtering

**Checkpoint**: At this point, User Story 4 should be fully functional - can store long-term metrics, auto-cleanup, and view trends

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Final integration, documentation, and quality improvements across all user stories

### Pipeline Integration (Backward Compatible)

- [X] T170 [P] Add enable_profiling configuration option to src/viz_art/config/__init__.py
- [X] T171 [P] Add enable_audit_logging configuration option to src/viz_art/config/__init__.py
- [X] T172 [P] Add enable_validation configuration option to src/viz_art/config/__init__.py
- [X] T173 Add profiler initialization to Pipeline.__init__() when enable_profiling=true in src/viz_art/pipeline/pipeline.py
- [X] T174 Add RunTracker context manager integration to Pipeline.run() when enable_audit_logging=true
- [X] T175 Add automatic logger creation in Pipeline.run() binding run_id from RunTracker
- [X] T176 Add validation hook in Pipeline.run() when enable_validation=true and ground_truth_dataset configured
- [X] T177 Add performance metrics auto-export to Pipeline.run() saving dashboard HTML after completion

### CLI Commands

- [X] T178 [P] Create src/viz_art/cli/logs.py with query_logs_cli() function
- [X] T179 Add argparse configuration to query_logs_cli() with --after, --before, --run-id, --stage, --level, --failed, --limit, --output flags
- [X] T180 Integrate AuditQuery builder in query_logs_cli() mapping CLI args to query methods
- [X] T181 [P] Create src/viz_art/cli/metrics.py with view_metrics_cli() function
- [X] T182 Add argparse configuration to view_metrics_cli() with --stage, --start, --end, --output flags
- [X] T183 Integrate MetricsStorage.get_aggregate_stats() in view_metrics_cli() for summary display
- [X] T184 [P] Create src/viz_art/cli/validate.py with validate_cli() function
- [X] T185 Add argparse configuration to validate_cli() with --run-id, --dataset, --output flags
- [X] T186 Integrate validate_pipeline() in validate_cli() loading run outputs and generating HTML report
- [X] T187 Add CLI entrypoints to pyproject.toml: viz-art-logs, viz-art-metrics, viz-art-validate

### Documentation

- [X] T188 [P] Create docs/phase3/PERFORMANCE_PROFILING.md documenting @profiler decorator usage and dashboard generation
- [X] T189 [P] Create docs/phase3/AUDIT_LOGGING.md documenting RunTracker context manager and query interface
- [X] T190 [P] Create docs/phase3/GROUND_TRUTH_VALIDATION.md documenting dataset formats and accuracy metrics
- [X] T191 Update main README.md adding Phase 3 features section with links to docs
- [X] T192 Add API reference to docs/phase3/API_REFERENCE.md listing all public functions and classes

### Example Integration (Full Stack)

- [X] T193 Create examples/phase3/example_full_stack.py combining all Phase 3 features
- [X] T194 Add profiling setup to example_full_stack.py with @profiler decorators
- [X] T195 Add RunTracker context manager to example_full_stack.py
- [X] T196 Add structured logging throughout example_full_stack.py with different log levels
- [X] T197 Add ground truth validation to example_full_stack.py with COCO dataset
- [X] T198 Add comprehensive report generation to example_full_stack.py combining performance + accuracy + logs
- [X] T199 Create examples/configs/full_observability.yaml config enabling all Phase 3 features

### Configuration Examples

- [X] T200 Create examples/configs/profiling_disabled.yaml showing opt-out pattern for production
- [X] T201 Create examples/configs/validation_only.yaml for QA environments
- [X] T202 Add configuration documentation to docs/phase3/CONFIGURATION.md explaining all enable_* flags

---

## Dependencies & Execution Strategy

### User Story Dependency Graph

```
Phase 1 (Setup) → Phase 2 (Foundation)
                        ↓
        ┌───────────────┼───────────────┐
        ↓               ↓               ↓
    Phase 3 (US1)   Phase 4 (US2)   Phase 5 (US3)
    Performance     Audit Logging   Validation
        ↓
    Phase 6 (US4)
    Historical Metrics
        ↓
    Phase 7 (Polish)
```

**Independent User Stories** (can be implemented in parallel after Phase 2):
- ✅ **US1** (Performance Diagnosis): No dependencies on other user stories
- ✅ **US2** (Debugging Failed Runs): No dependencies on other user stories
- ✅ **US3** (Ground Truth Validation): No dependencies on other user stories
- ⚠️ **US4** (Historical Metrics): Depends on US1 (extends MetricsStorage)

**Critical Path**: Phase 1 → Phase 2 → US1 → US4

**Parallel Opportunities**:
- After Phase 2: US1, US2, US3 can be implemented simultaneously by different developers
- Within each phase: Tasks marked [P] can run in parallel

### Parallel Execution Examples

**After Phase 2 completes**, assign user stories to separate developers:

```bash
# Developer A: Performance profiling (US1)
Tasks T031-T057 (27 tasks)

# Developer B: Audit logging (US2)
Tasks T058-T096 (39 tasks)

# Developer C: Ground truth validation (US3)
Tasks T097-T153 (57 tasks)

# All work in parallel, no conflicts
```

**Within US1**, parallelize by module:

```bash
# Developer A: Profiler core
Tasks T031-T035 (5 tasks)

# Developer B: Metrics storage
Tasks T036-T042 (7 tasks)

# Developer C: Dashboard visualization
Tasks T043-T050 (8 tasks)

# Different files, can work simultaneously
```

### Implementation Strategy

**MVP Scope** (Minimum Viable Product):
- Phase 1: Setup (10 tasks)
- Phase 2: Foundation (16 tasks)
- Phase 3: User Story 1 - Performance Diagnosis (27 tasks)
- **Total MVP**: 53 tasks

**MVP Deliverable**: Developers can profile pipeline stages and view performance dashboards showing timing and memory usage. This provides immediate value for identifying bottlenecks.

**Incremental Delivery**:
1. **Week 1**: MVP (US1) - Performance profiling
2. **Week 2**: US2 - Audit logging for debugging
3. **Week 3**: US3 - Ground truth validation
4. **Week 4**: US4 + Polish - Historical metrics and final integration

**Testing Strategy** (if tests were requested):
- Contract tests before implementation (marked [P])
- Integration tests after each user story phase
- End-to-end test in Phase 7 covering full observability stack

---

## Task Summary

**Total Tasks**: 202
- Phase 1 (Setup): 10 tasks
- Phase 2 (Foundation): 16 tasks
- Phase 3 (US1 - Performance Diagnosis): 27 tasks
- Phase 4 (US2 - Debugging Failed Runs): 39 tasks
- Phase 5 (US3 - Ground Truth Validation): 57 tasks
- Phase 6 (US4 - Historical Metrics): 16 tasks
- Phase 7 (Polish): 37 tasks

**Parallel Opportunities**: 68 tasks marked [P] (33.7% of total)

**Independent Test Criteria**:
- ✅ US1: Profile 3-stage pipeline → view dashboard with timing/memory
- ✅ US2: Trigger failure → query logs by run ID → export JSON
- ✅ US3: Load COCO dataset → validate predictions → view accuracy metrics
- ✅ US4: Run 10 times → query historical data → view trends

**Format Validation**: ✅ All tasks follow checklist format with Task ID, [P] marker (if parallel), [Story] label (for user story phases), and file paths in descriptions.
