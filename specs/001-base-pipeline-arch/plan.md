# Implementation Plan: Base Pipeline Architecture

**Branch**: `001-base-pipeline-arch` | **Date**: 2025-10-22 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-base-pipeline-arch/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

> **⚠️ IMPLEMENTATION UPDATE:** The original plan specified Protocol-based design. The final implementation uses **Abstract Base Class (ABC)** instead for PipelineStage, providing better validation and developer experience. References to "Protocol-based" in this document refer to the original design; actual implementation uses ABC with `@abstractmethod` decorators.

## Summary

Build the foundational vision processing pipeline system that enables sequential stage execution, YAML-based configuration, batch processing from directories, and static HTML report generation for reviewing pipeline outputs. This Phase 1 implementation focuses on image-only data support with a simple, extensible architecture that supports programmatic stage definition in Python.

## Technical Context

**Language/Version**: Python 3.8+
**Primary Dependencies**: pydantic (data validation), omegaconf (configuration), pytest (testing), PIL/opencv-python (image processing), jinja2 (HTML templating)
**Storage**: Filesystem-based (images, YAML configs, static HTML reports)
**Testing**: pytest with fixtures for sample data, GitHub Actions for CI/CD
**Target Platform**: Linux/macOS/Windows (cross-platform Python)
**Project Type**: Single library project
**Performance Goals**: Process 1920x1080 image through 3-stage pipeline in <30 seconds on standard hardware; batch process 10 images and generate report in single execution
**Constraints**: No external services/databases required; offline operation; static HTML reports viewable without web server; single-threaded sequential execution initially
**Scale/Scope**: MVP library for rapid prototyping; support for image-only data in Phase 1; extensible architecture for future point cloud support (Phase 2+)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Note**: No project constitution exists yet (template found at `.specify/memory/constitution.md`). Using standard software engineering principles for gate evaluation.

### Standard Gates Applied

| Gate | Requirement | Status | Notes |
|------|-------------|--------|-------|
| **Testability** | All core components must be unit testable | ✅ PASS | Pipeline, Stage, Config classes are independently testable with mock data |
| **Simplicity** | Use simplest solution that meets requirements | ✅ PASS | Sequential execution, filesystem storage, no external services - minimal complexity |
| **Dependencies** | Justify each external dependency | ✅ PASS | All dependencies serve clear purposes: pydantic (validation), omegaconf (config), pytest (testing), PIL/opencv (images), jinja2 (HTML) |
| **Documentation** | Public APIs must be documented | ✅ PASS | Plan includes quickstart.md generation and docstrings for core classes |
| **Maintainability** | Code must be readable and modular | ✅ PASS | Single-file stages, clear separation between Pipeline/Stage/Config concerns |

**Gate Result**: ✅ All gates passed - proceed to Phase 0 research

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
src/
├── viz_art/
│   ├── __init__.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── base.py           # Pipeline base class
│   │   ├── stage.py          # PipelineStage base class
│   │   └── connection.py     # Stage connection system
│   ├── config/
│   │   ├── __init__.py
│   │   ├── schema.py         # Pydantic schemas
│   │   └── loader.py         # OmegaConf integration
│   ├── batch/
│   │   ├── __init__.py
│   │   ├── processor.py      # Batch processing logic
│   │   └── reporter.py       # HTML report generation
│   └── utils/
│       ├── __init__.py
│       └── image.py          # Image utilities
│
tests/
├── fixtures/
│   ├── sample_images/        # Test images
│   └── configs/              # Test YAML configs
├── unit/
│   ├── test_pipeline.py
│   ├── test_stage.py
│   ├── test_config.py
│   └── test_batch.py
└── integration/
    └── test_end_to_end.py

examples/
├── simple_pipeline.py        # Basic 2-stage example
├── configs/
│   └── example_config.yaml
└── README.md

scripts/
└── batch_process.py          # CLI script for batch processing
```

**Structure Decision**: Single library project structure. The `src/viz_art/` directory contains all library code organized by functional area (pipeline core, configuration, batch processing). Tests are separated into unit and integration, with fixtures for sample data. Examples demonstrate library usage.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No violations detected. All gates passed.

---

## Phase 1 Re-evaluation (Post-Design)

**Date**: 2025-10-22

After completing Phase 1 design (data-model.md, contracts, quickstart), re-evaluating Constitution Check:

| Gate | Status | Post-Design Notes |
|------|--------|-------------------|
| **Testability** | ✅ PASS | All entities have clear interfaces; Protocol-based design enables mocking; Test examples provided in quickstart.md |
| **Simplicity** | ✅ PASS | Data model remains simple with 10 core entities; No unnecessary abstractions; File-based storage confirmed |
| **Dependencies** | ✅ PASS | All dependencies justified and documented in research.md; Each serves clear purpose |
| **Documentation** | ✅ PASS | Complete API contracts, YAML schema examples, quickstart guide with 3 usage patterns |
| **Maintainability** | ✅ PASS | Clear separation of concerns: Pipeline/Stage/Config/Batch; Immutable results support debugging; Protocol-based extension |

**Additional Observations**:
- Design maintains single-library structure (no complexity creep)
- API contracts are minimal and focused on Phase 1 requirements
- Configuration schema is flexible but validated
- Batch processing follows "continue on error" pattern as specified
- HTML report generation uses simple file references (not complex base64 encoding)

**Gate Result**: ✅ All gates still passed after Phase 1 design - proceed to Phase 2 (task generation via `/speckit.tasks`)
