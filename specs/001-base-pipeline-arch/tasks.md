# Implementation Tasks: Base Pipeline Architecture

**Feature**: 001-base-pipeline-arch
**Branch**: `001-base-pipeline-arch`
**Generated**: 2025-10-22

## Overview

This document provides a complete, dependency-ordered task list for implementing the base pipeline architecture. Tasks are organized by user story to enable independent implementation and testing of each feature increment.

**Total Tasks**: 54
**Estimated Duration**: 1-2 weeks for MVP (User Story 1 only)

---

## Task Organization

Tasks are structured in phases:
- **Phase 1**: Setup (project initialization)
- **Phase 2**: Foundational (shared infrastructure - MUST complete before user stories)
- **Phase 3**: User Story 1 (P1) - Define and Execute Simple Vision Pipeline
- **Phase 4**: User Story 4 (P2) - Validate Basic Pipeline Testing
- **Phase 5**: User Story 2 (P2) - Configure Pipeline via YAML
- **Phase 6**: User Story 3 (P3) - Batch Process Directory and Generate Review Report
- **Phase 7**: Polish & Documentation

**Legend**:
- `[P]` = Parallelizable task (can run concurrently with other [P] tasks in same phase)
- `[US#]` = User Story label (e.g., US1, US2, US3, US4)

---

## Phase 1: Setup (Project Initialization)

**Goal**: Initialize project structure and development environment

**Tasks**:

- [X] T001 Create project directory structure per plan.md: src/viz_art/, tests/, examples/, scripts/
- [X] T002 [P] Create pyproject.toml with dependencies: pydantic, omegaconf, pytest, pillow, opencv-python, jinja2
- [X] T003 [P] Create .gitignore with Python, IDE, and test artifacts
- [X] T004 [P] Create README.md with project overview and setup instructions
- [X] T005 [P] Initialize src/viz_art/__init__.py with package metadata
- [X] T006 Setup GitHub Actions workflow file .github/workflows/test.yml for CI/CD

---

## Phase 2: Foundational (Shared Infrastructure)

**Goal**: Implement core entities and base classes that all user stories depend on

**Completion Criteria**: Result data structures exist, can be instantiated and serialized

**Tasks**:

- [X] T007 [P] Create src/viz_art/pipeline/__init__.py
- [X] T008 [P] Create src/viz_art/config/__init__.py
- [X] T009 [P] Create src/viz_art/batch/__init__.py
- [X] T010 [P] Create src/viz_art/utils/__init__.py
- [X] T011 Define PipelineStage ABC (Abstract Base Class) in src/viz_art/pipeline/stage.py with name, input_keys, output_keys properties and pre_process(), predict(), post_process() methods
- [X] T012 Define StageStatus and RunStatus enums in src/viz_art/pipeline/results.py
- [X] T013 [P] Create StageResult dataclass in src/viz_art/pipeline/results.py with stage_name, status, started_at, duration_ms, outputs, error fields
- [X] T014 [P] Create PipelineRun dataclass in src/viz_art/pipeline/results.py with run_id, pipeline_name, started_at, completed_at, status, inputs, outputs, stage_results, error fields
- [X] T015 [P] Create BatchResult dataclass in src/viz_art/pipeline/results.py with batch_id, total_files, successful, failed, run_results, started_at, completed_at, report_path fields

---

## Phase 3: User Story 1 (P1) - Define and Execute Simple Vision Pipeline

**User Story**: A developer needs to create a basic vision processing pipeline that takes an image as input, processes it through multiple stages, and produces output results.

**Independent Test Criteria**: Create a 2-stage pipeline (image load → simple transform), execute with a test image, verify outputs from each stage are accessible and correct.

**Completion Criteria**:
- Pipeline class can orchestrate multiple stages
- Stages execute in sequence
- Data flows between stages via connections
- Results accessible after execution

**Tasks**:

### Core Pipeline Implementation

- [X] T016 [US1] Implement Pipeline class in src/viz_art/pipeline/base.py with __init__(name, config=None)
- [X] T017 [US1] Implement Pipeline.add_stage(stage) method to add stages to pipeline
- [X] T018 [US1] Implement Pipeline.connect(source_stage, target_stage, output_key, input_key, required=True) for stage connections
- [X] T019 [US1] Implement stage execution order validation in Pipeline (check for circular dependencies)
- [X] T020 [US1] Implement Pipeline.run(**inputs) method that executes all stages sequentially
- [X] T021 [US1] Implement data flow logic in Pipeline.run() to pass stage outputs as inputs to next stage
- [X] T022 [US1] Implement result aggregation in Pipeline.run() returning dict with all stage outputs, _run_id, _status

### Connection Management

- [X] T023 [US1] Create StageConnection class in src/viz_art/pipeline/connection.py with source_stage, target_stage, output_key, input_key, required fields
- [X] T024 [US1] Implement connection validation in Pipeline to verify output_key exists in source stage's output_keys
- [X] T025 [US1] Implement connection validation in Pipeline to verify input_key exists in target stage's input_keys
- [X] T026 [US1] Implement circular dependency detection algorithm in Pipeline

### Example Stages for Testing

- [X] T027 [P] [US1] Create examples/stages/ directory
- [X] T028 [P] [US1] Implement ImageLoader stage in examples/stages/image_loader.py with PIL-based image loading
- [X] T029 [P] [US1] Implement SimpleFilter stage in examples/stages/simple_filter.py with cv2.GaussianBlur
- [X] T030 [US1] Create examples/simple_pipeline.py demonstrating 2-stage pipeline execution

### Acceptance Validation

- [X] T031 [US1] Manual test: Execute examples/simple_pipeline.py with test image, verify both stage outputs accessible
- [X] T032 [US1] Manual test: Create 3-stage pipeline, verify data flows stage 1 → 2 → 3
- [X] T033 [US1] Manual test: Create single-stage pipeline, verify result returned correctly

---

## Phase 4: User Story 4 (P2) - Validate Basic Pipeline Testing

**User Story**: A developer wants automated tests to verify that the pipeline system works correctly, ensuring stages execute in order and data flows properly.

**Independent Test Criteria**: Run pytest with sample fixtures, verify all test cases pass with assertions on execution order and data flow.

**Completion Criteria**:
- Test fixtures exist for mock stages and sample images
- Unit tests cover pipeline execution flows
- CI/CD runs tests automatically

**Tasks**:

### Test Infrastructure

- [X] T034 [P] [US4] Create tests/fixtures/ directory
- [X] T035 [P] [US4] Create tests/fixtures/sample_images/ with 2-3 small test images (< 1MB each)
- [X] T036 [P] [US4] Create tests/fixtures/configs/ directory for test YAML configs
- [X] T037 [P] [US4] Create tests/unit/ directory
- [X] T038 [US4] Create tests/conftest.py with pytest fixtures for mock stages

### Unit Tests

- [X] T039 [P] [US4] Create tests/unit/test_pipeline.py
- [X] T040 [P] [US4] Write test_pipeline_add_stage() verifying stages added correctly
- [X] T041 [P] [US4] Write test_pipeline_connect_stages() verifying connections created
- [X] T042 [P] [US4] Write test_pipeline_execution_order() verifying stages execute in sequence
- [X] T043 [P] [US4] Write test_pipeline_data_flow() verifying outputs pass to next stage
- [X] T044 [P] [US4] Write test_pipeline_single_stage() verifying single-stage execution
- [X] T045 [P] [US4] Write test_pipeline_circular_dependency_detection() verifying error raised for cycles
- [X] T046 [P] [US4] Write test_pipeline_invalid_connection() verifying error for mismatched keys
- [X] T047 [P] [US4] Create tests/unit/test_stage.py
- [X] T048 [P] [US4] Write test_stage_abc_compliance() verifying stage implements required abstract methods

### CI/CD Integration

- [X] T049 [US4] Update .github/workflows/test.yml to run pytest on push/PR
- [X] T050 [US4] Configure test workflow to install dependencies and run tests
- [X] T051 [US4] Add test coverage reporting to CI/CD workflow (target: 70%+ coverage)

---

## Phase 5: User Story 2 (P2) - Configure Pipeline via YAML

**User Story**: A developer wants to configure pipeline behavior and stage parameters through YAML configuration files.

**Independent Test Criteria**: Create YAML config with stage parameters, load through config system, verify stages receive correct parameter values.

**Completion Criteria**:
- YAML configs loadable via OmegaConf
- Pydantic validates all config parameters
- Pipeline.from_config() constructs configured pipeline
- Invalid configs produce clear error messages

**Tasks**:

### Configuration Schema

- [X] T052 [P] [US2] Create Pydantic models in src/viz_art/config/schema.py: StageConfigItem, ConnectionItem, PipelineConfig
- [X] T053 [P] [US2] Add validation rules to PipelineConfig: unique stage names, non-empty pipeline_name
- [X] T054 [P] [US2] Add BatchConfigItem model to schema.py with input_dir, output_dir, file_patterns, recursive, continue_on_error, report_output fields

### Configuration Loader

- [X] T055 [US2] Implement load_config(config_path) function in src/viz_art/config/loader.py
- [X] T056 [US2] Integrate OmegaConf.load() in loader to read YAML files
- [X] T057 [US2] Convert OmegaConf DictConfig to native dict using OmegaConf.to_container(resolve=True)
- [X] T058 [US2] Validate converted dict with PipelineConfig.model_validate() and handle ValidationError
- [X] T059 [US2] Implement Pipeline.from_config(config) classmethod in src/viz_art/pipeline/base.py
- [X] T060 [US2] In from_config(), instantiate stages based on config.stages list
- [X] T061 [US2] In from_config(), create connections based on config.connections list

### Example Configurations

- [X] T062 [P] [US2] Create examples/configs/simple_pipeline.yaml with 2-stage configuration
- [X] T063 [P] [US2] Update examples/simple_pipeline.py to load from YAML config
- [X] T064 [P] [US2] Create example showing OmegaConf variable interpolation in config

### Configuration Tests

- [X] T065 [P] [US2] Create tests/unit/test_config.py
- [X] T066 [P] [US2] Write test_load_valid_config() verifying YAML loads successfully
- [X] T067 [P] [US2] Write test_load_invalid_yaml() verifying clear error for malformed YAML
- [X] T068 [P] [US2] Write test_validate_missing_required_field() verifying ValidationError with field path
- [X] T069 [P] [US2] Write test_validate_duplicate_stage_names() verifying error for duplicate names
- [X] T070 [P] [US2] Write test_pipeline_from_config() verifying pipeline constructed correctly from config
- [X] T071 [US2] Add test YAML configs to tests/fixtures/configs/

---

## Phase 6: User Story 3 (P3) - Batch Process Directory and Generate Review Report

**User Story**: A user wants to process multiple images from a directory through their pipeline and review results via a static HTML report.

**Independent Test Criteria**: Place test images in directory, execute batch script, verify HTML report generated with all images displayed per-stage.

**Completion Criteria**:
- Batch processor discovers images recursively
- Each image processed independently
- Errors logged but processing continues
- HTML report generated with stage-grouped view
- Report viewable offline in any browser

**Tasks**:

### Batch Processing Core

- [X] T072 [US3] Implement BatchProcessor class in src/viz_art/batch/processor.py with __init__(pipeline, config)
- [X] T073 [US3] Implement BatchProcessor.discover_images() generator using pathlib.rglob() with file_patterns
- [X] T074 [US3] Implement file type filtering in discover_images() to silently skip non-image files
- [X] T075 [US3] Implement BatchProcessor.run() method orchestrating batch execution
- [X] T076 [US3] Implement per-image pipeline execution with try-except for continue-on-error behavior
- [X] T077 [US3] Implement error tracking in BatchProcessor storing failed image paths and error messages
- [X] T078 [US3] Create BatchResult after all images processed with statistics

### HTML Report Generation

- [X] T079 [P] [US3] Create src/viz_art/batch/templates/ directory
- [X] T080 [P] [US3] Create Jinja2 template src/viz_art/batch/templates/batch_report.html
- [X] T081 [US3] Design HTML template structure: summary section, stage-grouped view, per-image view, error section
- [X] T082 [US3] Implement inline CSS in template for offline viewing (no external stylesheets)
- [X] T083 [US3] Implement stage-grouped layout in template (all images grouped by stage output)
- [X] T084 [US3] Implement error indicators in template for failed images
- [X] T085 [US3] Implement HTMLReporter class in src/viz_art/batch/reporter.py with __init__(template_dir=None)
- [X] T086 [US3] Implement HTMLReporter.generate(batch_result, output_path) method
- [X] T087 [US3] Setup Jinja2 Environment in HTMLReporter to load templates
- [X] T088 [US3] Implement template rendering with batch_result data
- [X] T089 [US3] Save rendered HTML to output_path with relative image references

### Batch Processing Script

- [X] T090 [US3] Create scripts/batch_process.py CLI script
- [X] T091 [US3] Implement argparse in batch_process.py: --config, --input-dir, --output-dir arguments
- [X] T092 [US3] Load pipeline config in batch_process.py
- [X] T093 [US3] Create BatchProcessor and execute batch run
- [X] T094 [US3] Print summary statistics after batch completion
- [X] T095 [US3] Generate HTML report using HTMLReporter

### Batch Processing Tests

- [X] T096 [P] [US3] Create tests/unit/test_batch.py
- [X] T097 [P] [US3] Write test_discover_images() verifying recursive image discovery
- [X] T098 [P] [US3] Write test_skip_non_images() verifying non-image files silently skipped
- [X] T099 [P] [US3] Write test_continue_on_error() verifying processing continues after failure
- [X] T100 [P] [US3] Write test_batch_result_statistics() verifying correct success/fail counts
- [X] T101 [P] [US3] Create tests/integration/ directory
- [X] T102 [US3] Create tests/integration/test_end_to_end.py
- [X] T103 [US3] Write test_batch_processing_end_to_end() executing batch script and verifying HTML report exists

---

## Phase 7: Polish & Documentation

**Goal**: Finalize documentation, examples, and cross-cutting concerns

**Tasks**:

### Documentation

- [ ] T104 [P] Update README.md with installation instructions, quickstart, and API overview
- [ ] T105 [P] Create examples/README.md documenting all example scripts
- [ ] T106 [P] Add docstrings to all public classes and methods (Google style)
- [ ] T107 [P] Create CONTRIBUTING.md with development setup and testing guidelines

### Code Quality

- [ ] T108 [P] Run pytest with coverage report, ensure 70%+ coverage
- [ ] T109 [P] Add type hints to all function signatures
- [ ] T110 [P] Run mypy type checking and fix any type errors
- [ ] T111 [P] Format code with black
- [ ] T112 [P] Lint code with ruff/flake8

### Final Validation

- [ ] T113 Execute all acceptance scenarios from spec.md manually
- [ ] T114 Test pipeline on different image sizes (small, medium, large)
- [ ] T115 Test batch processing with empty directory (verify graceful handling)
- [ ] T116 Test HTML report in multiple browsers (Chrome, Firefox, Safari)
- [ ] T117 Verify CI/CD pipeline passes all tests

---

## Dependency Graph (User Story Completion Order)

```
Phase 1: Setup
    ↓
Phase 2: Foundational Infrastructure
    ↓
Phase 3: User Story 1 (P1) ← MVP COMPLETE
    ↓
    ├─→ Phase 4: User Story 4 (P2) [Independent]
    ├─→ Phase 5: User Story 2 (P2) [Independent]
    └─→ Phase 6: User Story 3 (P3) [Depends on US1]
    ↓
Phase 7: Polish & Documentation
```

**Key Dependencies**:
- User Story 1 (P1) is **foundational** - must complete before others
- User Story 4 (P2), User Story 2 (P2) are **independent** - can implement in parallel after US1
- User Story 3 (P3) **depends on** User Story 1 (core pipeline must work)

---

## Parallel Execution Examples

### Phase 1 (Setup) - Parallel Opportunities

Execute T002, T003, T004, T005 in parallel (independent file creation)

```bash
# Terminal 1
task T002  # Create pyproject.toml

# Terminal 2
task T003  # Create .gitignore

# Terminal 3
task T004  # Create README.md

# Terminal 4
task T005  # Initialize __init__.py
```

### Phase 2 (Foundational) - Parallel Opportunities

Execute T007-T010, T013-T015 in parallel (independent module creation)

```bash
# Create all __init__.py files in parallel
parallel-task T007 T008 T009 T010

# Create result dataclasses in parallel
parallel-task T013 T014 T015
```

### Phase 3 (User Story 1) - Parallel Opportunities

Execute T027-T029 in parallel (independent example stages)

```bash
parallel-task T028 T029  # Implement ImageLoader and SimpleFilter simultaneously
```

### Phase 4 (User Story 4) - Parallel Opportunities

Execute T034-T037, T039-T048 in parallel (independent test files)

```bash
# Create test infrastructure in parallel
parallel-task T034 T035 T036 T037

# Write unit tests in parallel (different test functions)
parallel-task T040 T041 T042 T043 T044 T045 T046 T047 T048
```

### Phase 5 (User Story 2) - Parallel Opportunities

Execute T052-T054, T062-T064, T065-T071 in parallel

```bash
# Create schema models in parallel
parallel-task T052 T053 T054

# Create example configs in parallel
parallel-task T062 T063 T064

# Write config tests in parallel
parallel-task T066 T067 T068 T069 T070
```

### Phase 6 (User Story 3) - Parallel Opportunities

Execute T079-T080, T096-T100 in parallel

```bash
# Setup templates in parallel
parallel-task T079 T080

# Write batch tests in parallel
parallel-task T097 T098 T099 T100
```

### Phase 7 (Polish) - Maximum Parallelism

Execute T104-T112 in parallel (independent documentation and quality tasks)

```bash
# All polish tasks can run in parallel
parallel-task T104 T105 T106 T107 T108 T109 T110 T111 T112
```

---

## Implementation Strategy

### Recommended MVP Scope (1 week)

**Focus on User Story 1 only** to get core pipeline working:

1. Complete Phase 1: Setup (T001-T006)
2. Complete Phase 2: Foundational (T007-T015)
3. Complete Phase 3: User Story 1 (T016-T033)

**MVP Deliverable**: Working pipeline that can execute custom stages programmatically

### Full Phase 1 Scope (2 weeks)

Add testing, configuration, and batch processing:

1. MVP (User Story 1)
2. Add Phase 4: User Story 4 - Testing (T034-T051)
3. Add Phase 5: User Story 2 - YAML Config (T052-T071)
4. Add Phase 6: User Story 3 - Batch Processing (T072-T103)
5. Complete Phase 7: Polish (T104-T117)

**Full Deliverable**: Complete base pipeline architecture with all P1-P3 features

### Incremental Delivery

Each phase produces a testable, usable increment:

- **After Phase 3**: Can create and run multi-stage pipelines programmatically
- **After Phase 4**: Can verify pipeline behavior with automated tests
- **After Phase 5**: Can configure pipelines via YAML without code changes
- **After Phase 6**: Can batch process directories and review results in HTML reports
- **After Phase 7**: Production-ready library with documentation

---

## Success Metrics

### Code Coverage
- Target: 70%+ overall coverage
- Core pipeline classes: 80%+ coverage
- Config and batch modules: 60%+ coverage

### Performance
- 2-stage pipeline execution: < 5 seconds for 1920x1080 image
- Batch processing 10 images: < 60 seconds
- Test suite execution: < 2 minutes

### Quality Gates
- All pytest tests pass
- No mypy type errors
- Code formatted with black
- All public APIs documented

---

## Task Execution Notes

### Format Validation
- All tasks follow format: `- [ ] T### [P] [US#] Description with file path`
- Task IDs sequential (T001-T117)
- [P] marker for parallelizable tasks
- [US#] marker for user story tasks only

### File Path Conventions
- Use absolute imports: `from viz_art.pipeline import Pipeline`
- All source code in `src/viz_art/`
- All tests in `tests/`
- Examples in `examples/`
- Scripts in `scripts/`

### Testing Conventions
- Unit tests in `tests/unit/test_*.py`
- Integration tests in `tests/integration/test_*.py`
- Fixtures in `tests/fixtures/`
- Use pytest fixtures from `conftest.py`

---

**Total Tasks**: 117
**Parallelizable Tasks**: 47 (40% can run concurrently)
**User Story Tasks**: 94 (US1: 18, US2: 20, US3: 32, US4: 18, Polish: 23)
