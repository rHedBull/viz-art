# Feature Specification: Base Pipeline Architecture

**Feature Branch**: `001-base-pipeline-arch`
**Created**: 2025-10-22
**Status**: Draft
**Input**: User description: "base-pipeline-arch phase1 from implementation.md with background in REQUIREMENTS.md"

## Clarifications

### Session 2025-10-22

- Q: User Story 3 - Should the UI support live file upload or batch processing from directories? → A: Batch processing from directories with static HTML output for review. No real-time upload needed; script executes once, processes all images in directory structure, generates static HTML report showing pipeline results per stage and in total for reviewing labeling.
- Q: When a pipeline stage fails while processing one image in a batch, how should the system behave? → A: Continue processing remaining images - Log failures, continue with other images, report all errors in final HTML report
- Q: Where should the generated HTML report be saved after batch processing completes? → A: Output directory alongside processed images
- Q: How should per-stage outputs be organized in the HTML report for effective review? → A: Stage-grouped view - Group all images by stage (all stage-1 results together, then all stage-2 results, etc.)
- Q: What should happen when the batch processor encounters non-image files in the input directory? → A: Skip silently - Ignore non-image files and only process valid image formats without logging

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Define and Execute Simple Vision Pipeline (Priority: P1)

A developer needs to create a basic vision processing pipeline that takes an image as input, processes it through multiple stages, and produces output results. They want to define stages programmatically in Python and execute them in sequence without complex orchestration.

**Why this priority**: This is the foundational capability that all other features depend on. Without a working pipeline execution system, no other functionality can be built or tested.

**Independent Test**: Can be fully tested by creating a 2-stage pipeline (e.g., image load → simple transform), executing it with a test image, and verifying that outputs from each stage are accessible and correct.

**Acceptance Scenarios**:

1. **Given** a developer has defined two custom stages (ImageLoader and SimpleFilter), **When** they connect these stages and call pipeline.run() with an image path, **Then** the pipeline executes both stages in sequence and returns outputs from each stage
2. **Given** a pipeline with three connected stages, **When** execution starts, **Then** data flows from stage 1 → stage 2 → stage 3 automatically based on the defined connections
3. **Given** a pipeline is running, **When** a stage completes processing, **Then** its output becomes available as input to the next connected stage
4. **Given** a single-stage pipeline, **When** run() is called with valid image input, **Then** the stage processes the image and the pipeline returns the result

---

### User Story 2 - Configure Pipeline via YAML (Priority: P2)

A developer wants to configure pipeline behavior and stage parameters through YAML configuration files instead of hardcoding values in Python, enabling easy experimentation and environment-specific settings.

**Why this priority**: Configuration management enables rapid iteration and testing without code changes, which is essential for ML experimentation workflows but can be added after basic execution works.

**Independent Test**: Can be fully tested by creating a YAML config with stage parameters, loading it through the config system, and verifying that stages receive the correct parameter values during execution.

**Acceptance Scenarios**:

1. **Given** a YAML config file defining pipeline stages and their parameters, **When** the developer loads the config, **Then** the pipeline is constructed with all stages configured according to the YAML specification
2. **Given** a config file with per-stage parameters (e.g., image resize dimensions), **When** the pipeline runs, **Then** each stage uses its configured parameters
3. **Given** invalid configuration values in YAML, **When** the config is loaded, **Then** validation errors are reported with clear messages indicating which parameters are invalid
4. **Given** a config schema definition, **When** a developer provides a config file, **Then** all required parameters are validated against the schema before pipeline creation

---

### User Story 3 - Batch Process Directory and Generate Review Report (Priority: P3)

A user wants to process multiple images stored in a directory structure through their vision pipeline in batch mode, then review the results (including labeling/annotations) for each stage and the complete pipeline via a generated static HTML report, without needing real-time interaction or manual file uploads.

**Why this priority**: Batch processing with offline review enables efficient evaluation of pipeline performance across datasets and makes results accessible without requiring an active web server, but the core pipeline execution must work first.

**Independent Test**: Can be fully tested by placing test images in a directory structure, executing a batch processing script with the configured pipeline, and opening the generated static HTML report to verify that all images are displayed with their per-stage and final outputs visible for review.

**Acceptance Scenarios**:

1. **Given** a directory containing subdirectories with image files, **When** a user runs the batch processing script with that directory path, **Then** the pipeline discovers and processes all images in the directory structure
2. **Given** images being processed in batch mode, **When** the pipeline executes, **Then** each image is processed through all stages sequentially and outputs are captured
3. **Given** batch processing has completed, **When** the script finishes, **Then** a static HTML report is generated showing all processed images
4. **Given** the generated HTML report, **When** a user opens it in a browser, **Then** they can view results organized by stage with all images' outputs grouped together for each stage (stage-grouped view)
5. **Given** the HTML report is open, **When** reviewing an image, **Then** the user can see the complete pipeline output including any labeling/annotations applied at each stage
6. **Given** a pipeline execution fails for specific images, **When** the report is generated, **Then** failed images are clearly marked with error messages indicating which stage failed and why

---

### User Story 4 - Validate Basic Pipeline Testing (Priority: P2)

A developer wants automated tests to verify that the pipeline system works correctly, ensuring that stages execute in order, data flows properly, and basic error cases are handled.

**Why this priority**: Testing infrastructure is essential for maintaining code quality as the system grows, but can be added after the core pipeline implementation is proven to work manually.

**Independent Test**: Can be fully tested by running pytest with sample test fixtures (mock stages, test images) and verifying that all test cases pass with appropriate assertions on execution order and data flow.

**Acceptance Scenarios**:

1. **Given** a test fixture with mock stages, **When** pytest runs, **Then** tests verify that stages execute in the correct sequence
2. **Given** test data fixtures (sample images), **When** pipeline tests execute, **Then** outputs match expected results
3. **Given** a CI/CD workflow configured with GitHub Actions, **When** code is pushed, **Then** all automated tests run and report pass/fail status
4. **Given** a pipeline with intentionally failing stage, **When** tests run, **Then** error handling is verified and appropriate exceptions are caught

---

### Edge Cases

- Non-image files (PDF, video, text files) in the directory are silently skipped during batch processing without logging or errors
- How does the system handle extremely large images that might cause memory issues during batch processing?
- When a stage fails for one image during batch processing, the system continues processing remaining images and reports all failures in the final HTML report
- How are circular stage connections prevented (e.g., Stage A → Stage B → Stage A)?
- What happens when required stage parameters are missing from the config file?
- How does the system handle stages that produce no output or null results?
- What happens when multiple stages try to write to the same output key?
- What happens when the input directory is empty or contains no valid images?
- How does the system handle deeply nested directory structures during batch processing?
- What happens if HTML report generation fails after successful pipeline processing?
- How are image file path references maintained in the HTML report if source images are moved or deleted?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a Pipeline base class that orchestrates execution of connected PipelineStage instances in sequence
- **FR-002**: System MUST provide a PipelineStage base class with standard interface methods for processing (pre_process, predict, post_process)
- **FR-003**: System MUST support connecting stage outputs to subsequent stage inputs through a connection system
- **FR-004**: System MUST support image data as a first-class input type for pipeline processing
- **FR-005**: Pipeline MUST provide a run() method that accepts input data and executes all connected stages in sequence
- **FR-006**: System MUST return outputs from all executed stages in an accessible structure after pipeline.run() completes
- **FR-007**: System MUST integrate OmegaConf for hierarchical YAML/JSON configuration management
- **FR-008**: System MUST provide Pydantic-based config schema validation with clear error messages for invalid configurations
- **FR-009**: System MUST support loading pipeline definitions and stage parameters from YAML configuration files
- **FR-010**: System MUST validate all configuration parameters against defined schemas before pipeline creation
- **FR-011**: System MUST provide a batch processing script that discovers and processes all images in a specified directory structure
- **FR-012**: Batch processor MUST support common image formats (PNG, JPG, JPEG) found in directory trees and silently skip non-image files without logging or error reporting
- **FR-013**: System MUST generate static HTML reports after batch processing completes and save them to an output directory alongside the processed images
- **FR-014**: HTML reports MUST display per-stage outputs organized by stage, grouping all images' results for each stage together (stage-grouped view)
- **FR-015**: HTML reports MUST display complete pipeline outputs including labeling/annotations in a reviewable format
- **FR-019**: Batch processor MUST continue processing remaining images when a stage fails for one image, logging failures and including error details in the final HTML report
- **FR-020**: System MUST log all processing errors with sufficient detail (image path, stage name, error message) for debugging
- **FR-021**: HTML reports MUST clearly distinguish between successfully processed images and failed images with visible error indicators
- **FR-016**: System MUST provide pytest-based testing framework with test fixtures for sample data
- **FR-017**: System MUST include automated tests for basic pipeline execution flows
- **FR-018**: System MUST support CI/CD integration via GitHub Actions for automated testing on code changes

### Key Entities

- **Pipeline**: Represents the complete processing workflow; contains ordered list of connected stages, manages execution flow, handles data passing between stages
- **PipelineStage**: Individual processing unit in the pipeline; has pre_process/predict/post_process methods, accepts inputs and produces outputs, configurable via parameters
- **StageConnection**: Defines data flow between stages; maps output keys from one stage to input keys of another, enforces typing and validation
- **Config**: Configuration specification for pipeline and stages; defined in YAML/JSON, validated by Pydantic schemas, supports hierarchical parameter organization
- **PipelineRun**: A single execution instance of the pipeline; tracks execution state, stores intermediate outputs, captures timing and status information

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Developers can create a 2-stage pipeline, execute it with sample image input, and retrieve outputs from both stages in under 5 minutes of setup time
- **SC-002**: Pipeline execution completes for a 3-stage workflow processing a 1920x1080 image within reasonable time (under 30 seconds for simple stages on standard hardware)
- **SC-003**: Configuration changes in YAML files take effect immediately on next pipeline run without requiring code modifications or application restart
- **SC-004**: Users can execute batch processing on a directory of 10 images and generate a complete HTML report within a single script execution (one command to process and generate report)
- **SC-005**: Generated HTML reports can be opened in any modern browser without requiring a web server or additional dependencies
- **SC-006**: Automated test suite runs completely in under 2 minutes and achieves at least 70% code coverage of core pipeline classes
- **SC-007**: 95% of configuration errors are caught during config loading with validation messages that clearly identify the problematic parameter and expected format

## Assumptions

- Image inputs will be standard formats supported by common Python libraries (PIL/OpenCV)
- Initial implementation assumes single-threaded sequential execution (parallel processing deferred to later phases)
- Stages are assumed to be CPU-based processing initially (GPU support deferred to later phases)
- Configuration files are assumed to be edited by developers familiar with YAML syntax
- Batch processing operates on local filesystem directories (network/cloud storage deferred to later phases)
- Generated HTML reports are self-contained files viewable offline without web server
- Test data fixtures will include small sample images suitable for fast test execution (< 1MB each)
- GitHub Actions CI/CD assumes standard open-source tier (public repositories)

## Dependencies

- **Python 3.8+**: Required runtime environment
- **External Libraries**: pydantic, omegaconf, pytest, PIL/opencv-python (as specified in REQUIREMENTS.md)
- **HTML Generation**: Standard library or lightweight templating (jinja2 or similar)
- **Development Tools**: pytest for testing, GitHub Actions for CI/CD
- **No External Services**: All components run locally without requiring external APIs or databases
- **No Web Server Required**: HTML reports are static files viewable in any modern browser

## Scope Boundaries

### In Scope for Phase 1:
- Basic pipeline execution with sequential stage processing
- Image-only data support
- YAML-based configuration
- Batch processing from directory structures
- Static HTML report generation for reviewing pipeline outputs
- Per-stage output visualization in HTML reports
- Basic pytest testing framework
- Stage connection and data flow management

### Explicitly Out of Scope (Deferred to Later Phases):
- Point cloud data support (Phase 2)
- Interactive/real-time web UI (deferred - static HTML sufficient for Phase 1)
- Advanced visualizations beyond basic image display (Phase 2)
- Performance profiling and metrics tracking (Phase 3)
- Accuracy tracking and ground truth comparison (Phase 4)
- Model versioning and registry (Phase 5)
- Parallel/distributed batch processing (Phase 5)
- Real-time parameter tuning (Phase 7)
- Multi-GPU or distributed processing (Phase 8)
