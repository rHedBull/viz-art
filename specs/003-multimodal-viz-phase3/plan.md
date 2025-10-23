# Implementation Plan: Performance Monitoring & Debugging System

**Branch**: `003-multimodal-viz-phase3` | **Date**: 2025-10-23 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/003-multimodal-viz-phase3/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

This phase adds comprehensive performance monitoring and debugging capabilities to the vision pipeline library. The primary requirement is to provide developers with full observability into pipeline execution through automatic performance profiling (timing and memory tracking), structured audit logging with queryable run history, and ground truth validation with accuracy metrics. The technical approach leverages Loguru for structured logging, PyArrow/Parquet for efficient metrics storage, and existing visualization infrastructure (Plotly) for performance dashboards and error analysis tools.

## Technical Context

**Language/Version**: Python 3.8+ (existing project requirement)
**Primary Dependencies**: Loguru (structured logging), PyArrow (metrics storage), psutil (memory tracking), torchmetrics or scikit-learn (accuracy metrics)
**Storage**: File-based: JSON for audit logs, Parquet for performance metrics, existing OutputSaver for stage artifacts
**Testing**: pytest (existing framework), pytest-benchmark for performance regression tests
**Target Platform**: Linux/macOS/Windows development environments (existing cross-platform support)
**Project Type**: Single library project (viz-art package)
**Performance Goals**: <5% overhead from monitoring instrumentation, sub-2-second audit log queries for 10k runs, 70%+ compression for metrics storage
**Constraints**: File-based storage (no database dependency), backward compatible with existing pipeline API, minimal impact on existing stage execution
**Scale/Scope**: Support for 10k+ pipeline runs in history, 1000+ samples per ground truth dataset, 500+ error cases in browser

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Initial Check (Pre-Phase 0)

**Status**: No project constitution exists yet (`.specify/memory/constitution.md` is template-only).

**Recommendation**: This feature should establish initial constitution principles based on observed patterns:
- Library-first architecture (viz-art as standalone package)
- File-based storage for simplicity (no external database dependencies)
- Test-driven development (pytest framework already established)
- Minimal dependencies (leveraging existing tools where possible)

**Gate Result**: PASS (proceeding with best practices from existing codebase patterns)

### Post-Design Check (After Phase 1)

**Design Review**: All Phase 1 artifacts (research.md, data-model.md, contracts/, quickstart.md) have been generated and reviewed.

**Architecture Alignment**:
- ✓ **Library-First**: All new modules (performance/, audit/, validation/) follow existing viz_art package structure
- ✓ **File-Based Storage**: Parquet for metrics, JSON Lines for logs, no database dependencies
- ✓ **Test-Driven**: Contract tests defined, pytest integration maintained
- ✓ **Minimal Dependencies**: Only 4 core new deps (Loguru, PyArrow, psutil, scikit-learn), 1 optional (torchmetrics)
- ✓ **Backward Compatible**: Decorator-based instrumentation, no breaking changes to Pipeline/Stage APIs
- ✓ **Cross-Platform**: psutil works on Linux/macOS/Windows, graceful GPU degradation

**Design Quality**:
- Research decisions documented with alternatives considered
- Data models use Pydantic validation (existing pattern)
- API contracts follow Protocol-based design (type-safe, testable)
- Quickstart provides clear integration examples

**Gate Result**: PASS

**Recommendation for Future**: Consider formalizing these principles into `.specify/memory/constitution.md` as the project matures.

## Project Structure

### Documentation (this feature)

```text
specs/003-multimodal-viz-phase3/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
src/viz_art/
├── pipeline/            # Existing: Pipeline and Stage base classes
├── types/               # Existing: Data type definitions
├── config/              # Existing: Configuration management
├── batch/               # Existing: Batch processing with OutputSaver
├── visualization/       # Existing: Plotly-based visualizations
├── calibration/         # Existing: Coordinate transforms
├── utils/               # Existing: Utilities
├── performance/         # NEW: Performance profiling (Phase 3.1)
│   ├── __init__.py
│   ├── profiler.py      # Stage timing and memory tracking
│   ├── metrics_storage.py  # PyArrow/Parquet storage
│   └── dashboard.py     # Performance visualization
├── audit/               # NEW: Logging and auditing (Phase 3.2)
│   ├── __init__.py
│   ├── logger.py        # Loguru integration
│   ├── run_tracker.py   # Run ID and audit trail
│   └── query.py         # Audit log query interface
└── validation/          # NEW: Ground truth integration (Phase 3.3)
    ├── __init__.py
    ├── dataset.py       # Golden dataset management
    ├── loaders.py       # Multi-format annotation loading
    ├── metrics.py       # Accuracy calculation utilities
    └── error_analysis.py # Error browser and comparison

tests/
├── unit/
│   ├── test_performance/
│   ├── test_audit/
│   └── test_validation/
├── integration/
│   ├── test_end_to_end_monitoring.py
│   └── test_ground_truth_pipeline.py
└── fixtures/
    ├── sample_audit_logs.json
    ├── sample_metrics.parquet
    └── golden_datasets/
        ├── bboxes/      # COCO format examples
        ├── segmentation/  # Mask format examples
        └── pointclouds/   # 3D label examples
```

**Structure Decision**: Single library project structure (Option 1) as this is a library enhancement. New modules (performance/, audit/, validation/) follow existing package organization pattern under src/viz_art/. File-based storage aligns with current OutputSaver approach (no backend/frontend split needed).

## Complexity Tracking

> **No violations detected** - all design choices align with existing patterns

This feature maintains the established simplicity principles:
- No new external storage dependencies (file-based like existing OutputSaver)
- Reuses existing visualization infrastructure (Plotly from Phase 2)
- Extends existing pipeline hooks (no API breaking changes)
- Follows existing testing patterns (pytest + fixtures)
