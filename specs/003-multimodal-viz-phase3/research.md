# Research: Performance Monitoring & Debugging System

**Feature**: Performance Monitoring & Debugging System
**Branch**: `003-multimodal-viz-phase3`
**Date**: 2025-10-23

## Overview

This document captures research findings and technical decisions for Phase 3 implementation. All NEEDS CLARIFICATION items from the Technical Context have been resolved through research into best practices, existing codebase patterns, and library capabilities.

## Research Topics

### 1. Structured Logging Framework Selection

**Decision**: Loguru

**Rationale**:
- **Simplicity**: Single-import, zero-configuration structured logging with automatic rotation
- **Performance**: Minimal overhead (<2% in benchmarks) acceptable for <5% target
- **Features**: Built-in JSON serialization, context binding, log rotation, compression
- **Python 3.8+ compatibility**: Fully supports target platform
- **Integration**: Works seamlessly with existing Python logging (can intercept stdlib logs)

**Alternatives Considered**:
- **structlog**: More flexible but requires more configuration boilerplate
  - Rejected: Adds complexity without clear benefit for file-based logging
- **stdlib logging + JSON formatter**: Manual setup for rotation and compression
  - Rejected: More code to maintain, no clear advantage over Loguru
- **slog (Rust port)**: Excellent performance but requires FFI bindings
  - Rejected: Adds dependency complexity, overkill for file-based logging

**Implementation Notes**:
- Use `logger.bind(run_id=...)` for automatic run ID injection in all logs
- Configure rotation with `rotation="100 MB"` and `compression="gz"`
- Store logs in `output/logs/{date}.jsonl` for easy querying by date

**References**:
- Loguru docs: https://loguru.readthedocs.io/
- Benchmarks show <2% overhead: https://github.com/Delgan/loguru/issues/280

---

### 2. Metrics Storage Format

**Decision**: PyArrow/Parquet with local file storage

**Rationale**:
- **Compression**: Parquet achieves 70-90% compression for time-series metrics (exceeds 70% target)
- **Query Performance**: Columnar format enables sub-second filtering on 10k+ rows
- **Schema Evolution**: Supports adding new metric columns without breaking existing data
- **No Database**: Aligns with file-based architecture (no PostgreSQL/SQLite dependency)
- **Python Integration**: PyArrow has excellent pandas/numpy interop for analysis

**Alternatives Considered**:
- **SQLite**: Simpler queries but requires schema migrations, larger file sizes
  - Rejected: File size ~3x larger than Parquet, adds SQL dependency
- **JSON Lines**: Human-readable but poor compression and slow filtering
  - Rejected: Fails 70% compression target, slow queries on large datasets
- **HDF5**: Good for numerical data but limited schema evolution
  - Rejected: Overkill for simple time-series, tooling less mature than Parquet
- **CSV**: Minimal dependencies but no compression or type safety
  - Rejected: Unacceptable for production metrics storage

**Implementation Notes**:
- Store metrics in `output/metrics/{stage_name}.parquet` partitioned by stage
- Schema: `run_id (str), timestamp (datetime64), stage_name (str), execution_time_ms (float), cpu_memory_mb (float), gpu_memory_mb (nullable float)`
- Use PyArrow's `write_table()` with `compression='snappy'` for good balance of speed/size
- Query with `pq.read_table(filters=[('timestamp', '>=', start_date)])`

**References**:
- PyArrow Parquet docs: https://arrow.apache.org/docs/python/parquet.html
- Compression benchmarks: https://tech.marksblogg.com/billion-nyc-taxi-rides-parquet.html

---

### 3. Memory Tracking Approach

**Decision**: psutil for cross-platform memory monitoring

**Rationale**:
- **Cross-Platform**: Works on Linux/macOS/Windows (matches target platform)
- **Process-Level**: Tracks per-process memory (RSS, VMS) without kernel modules
- **GPU Support**: Can interface with `pynvml` (NVIDIA Management Library) for GPU memory
- **Minimal Overhead**: C-based library, negligible performance impact
- **Established**: 10+ years of production use, stable API

**Alternatives Considered**:
- **memory_profiler**: Line-by-line profiling, too fine-grained and slow
  - Rejected: Adds 30-50% overhead, overkill for stage-level tracking
- **tracemalloc**: Python stdlib, tracks allocations not total memory
  - Rejected: Misses C-extension allocations (NumPy, OpenCV, Open3D)
- **resource.getrusage()**: POSIX-only, no Windows support
  - Rejected: Fails cross-platform requirement
- **Custom /proc parsing**: Linux-only, brittle
  - Rejected: Not portable, adds maintenance burden

**Implementation Notes**:
- Measure before/after each stage: `process.memory_info().rss / 1024 / 1024` (MB)
- GPU tracking (if available): `pynvml.nvmlDeviceGetMemoryInfo(handle).used`
- Graceful degradation: GPU metrics nullable, falls back to CPU-only on non-NVIDIA systems
- Sample every 100ms during long-running stages for peak memory tracking

**References**:
- psutil docs: https://psutil.readthedocs.io/
- pynvml integration: https://pypi.org/project/nvidia-ml-py/

---

### 4. Accuracy Metrics Library Selection

**Decision**: scikit-learn metrics with optional torchmetrics for advanced use cases

**Rationale**:
- **scikit-learn**: Covers 90% of use cases (precision, recall, F1, confusion matrix)
  - Already a common dependency in ML pipelines
  - Zero-configuration, works with NumPy arrays
  - Well-tested, stable API
- **torchmetrics**: Optional for advanced metrics (mAP, IoU for detection/segmentation)
  - More comprehensive but requires PyTorch dependency
  - Make it optional: only imported if user needs advanced metrics

**Alternatives Considered**:
- **torchmetrics only**: Best metrics coverage but heavy dependency (PyTorch)
  - Rejected: Adds 500MB+ dependency for basic use cases
- **Custom implementations**: Full control but reinventing the wheel
  - Rejected: Testing burden, likely to have edge-case bugs
- **TensorFlow Metrics**: Comparable to torchmetrics but heavier
  - Rejected: Less popular in vision community, larger dependency

**Implementation Notes**:
- Core metrics (always available): `sklearn.metrics.{precision_recall_fscore_support, confusion_matrix}`
- Advanced metrics (optional): `torchmetrics.detection.MeanAveragePrecision`, `torchmetrics.JaccardIndex`
- Lazy import pattern: Only load torchmetrics if user specifies advanced metric type
- Provide metric plugin system: Users can register custom metric functions

**References**:
- scikit-learn metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
- torchmetrics docs: https://torchmetrics.readthedocs.io/

---

### 5. Ground Truth Annotation Format Support

**Decision**: Multi-format loader with schema validation via Pydantic

**Rationale**:
- **Format Coverage**: Support COCO (bboxes), PNG/NPY (segmentation), PLY/PCD with attributes (3D labels)
- **Validation**: Pydantic models catch format errors early with clear error messages
- **Extensibility**: Easy to add new formats via plugin pattern
- **Type Safety**: Static typing helps prevent runtime errors

**Supported Formats** (Priority Order):
1. **COCO JSON** (bboxes, keypoints): Industry standard for object detection
2. **PNG/NPY Masks** (segmentation): Common for semantic/instance segmentation
3. **PLY/PCD with labels** (3D points): Point cloud classification/segmentation
4. **YOLO TXT** (bboxes): Popular in detection community
5. **Custom JSON** (extensible): User-defined schema with Pydantic validation

**Alternatives Considered**:
- **Single format only** (e.g., COCO): Simpler but limits adoption
  - Rejected: Users have existing datasets in various formats
- **Auto-detection without validation**: Flexible but error-prone
  - Rejected: Silent failures worse than upfront validation errors
- **Database storage**: More queryable but adds complexity
  - Rejected: File-based aligns with architecture

**Implementation Notes**:
- Format detection: File extension + schema sniffing (e.g., check for `"annotations"` key in JSON)
- Loader registry: `@register_format("coco")` decorator for plugins
- Validation: Pydantic models for each format (e.g., `COCOAnnotation`, `YOLOAnnotation`)
- Unified internal representation: Convert all formats to common `Annotation` dataclass

**References**:
- COCO format spec: https://cocodataset.org/#format-data
- Pydantic validation: https://docs.pydantic.dev/latest/

---

### 6. Performance Dashboard Visualization

**Decision**: Extend existing Plotly integration from Phase 2

**Rationale**:
- **Consistency**: Reuse visualization infrastructure already in place
- **Interactive**: Plotly provides hover, zoom, filtering out-of-box
- **Time Series**: Built-in support for time-based charts (performance trends)
- **No Additional Dependency**: Already required by Phase 2

**Visualizations Needed**:
1. **Per-Stage Timing**: Horizontal bar chart comparing stage execution times
2. **Memory Usage**: Stacked area chart showing CPU/GPU memory over time
3. **Historical Trends**: Line chart with run times over dates
4. **Aggregate Stats**: Box plot showing p50/p95/p99 across runs
5. **Error Distribution**: Pie chart showing failure counts by stage/error type

**Alternatives Considered**:
- **Matplotlib**: More control but requires more code for interactivity
  - Rejected: Plotly already integrated, better UX for dashboards
- **Seaborn**: Nice statistical plots but static images
  - Rejected: Need interactive charts for filtering
- **Separate dashboard framework** (Dash, Streamlit): Powerful but heavy
  - Rejected: Overkill for embedded library dashboards

**Implementation Notes**:
- Render to HTML: `plotly.io.write_html()` for standalone reports
- Embed in UI: Pass Plotly JSON to frontend (if web UI exists)
- Offline mode: No external CDN dependencies for air-gapped environments

**References**:
- Plotly time series: https://plotly.com/python/time-series/
- Existing viz_art/visualization module for patterns

---

### 7. Audit Log Query Interface

**Decision**: Simple Python API + CLI, no query language

**Rationale**:
- **Simplicity**: Method chaining API (`.filter(stage="detect").after(date).limit(10)`)
- **Type Safety**: IDE autocomplete, compile-time checks
- **No Learning Curve**: No SQL/custom DSL to learn
- **File-Based**: Scan JSON Lines files with date-based partitioning

**API Design**:
```python
from viz_art.audit import AuditQuery

# Query by run ID
logs = AuditQuery().run_id("abc-123").fetch()

# Query by date range and stage
logs = (
    AuditQuery()
    .after("2025-10-01")
    .before("2025-10-23")
    .stage("detection")
    .failed()
    .limit(100)
    .fetch()
)

# Export filtered logs
AuditQuery().after("2025-10-20").export_json("filtered.json")
```

**Alternatives Considered**:
- **SQL interface** (with SQLite): More powerful but requires schema maintenance
  - Rejected: Overkill for simple filtering, breaks file-based architecture
- **grep/jq on command line**: Flexible but hard to use for non-experts
  - Rejected: Provide programmatic API, users can still grep if needed
- **Full-text search** (Whoosh, Elasticsearch): Great for complex queries but heavy
  - Rejected: Adds dependencies, unnecessary for structured logs

**Implementation Notes**:
- Date-based partitioning: `output/logs/2025-10-23.jsonl` for fast date filtering
- Lazy loading: Only read files matching query criteria
- Caching: LRU cache for repeated queries on same date range
- CLI wrapper: `viz-art logs --after 2025-10-20 --stage detection --failed`

**References**:
- Python builder pattern: https://python-patterns.guide/gang-of-four/builder/

---

### 8. Integration with Existing Pipeline

**Decision**: Decorator-based instrumentation with context managers

**Rationale**:
- **Non-Invasive**: Existing stages work without modification
- **Opt-In**: Users can disable monitoring with config flag
- **Clean API**: Decorator syntax familiar to Python developers
- **Backward Compatible**: No breaking changes to Pipeline/Stage base classes

**Integration Points**:
1. **Pipeline.run()**: Wrap with `RunTracker` context manager to generate run ID
2. **Stage.execute()**: Decorate with `@profile` for automatic timing/memory tracking
3. **Pipeline.validate()**: Hook into existing validation for ground truth comparison

**Example**:
```python
from viz_art.audit import RunTracker
from viz_art.performance import profile

class MyStage(Stage):
    @profile  # Auto-instruments timing and memory
    def execute(self, inputs):
        # Existing stage logic unchanged
        return outputs

# In Pipeline
with RunTracker() as run_id:
    results = pipeline.run(inputs)
    # All logs/metrics automatically tagged with run_id
```

**Alternatives Considered**:
- **Explicit calls in each stage**: More control but requires stage modifications
  - Rejected: Breaks backward compatibility, high adoption friction
- **Monkey-patching**: Zero code changes but fragile
  - Rejected: Debugging nightmare, breaks with inheritance
- **Agent-based profiling** (py-spy, pyinstrument): No code changes but external process
  - Rejected: Hard to correlate with run IDs, less accurate for per-stage metrics

**Implementation Notes**:
- Use `functools.wraps` to preserve stage metadata
- Store metrics in thread-local storage for multi-threaded pipelines
- Provide `Pipeline(enable_monitoring=True/False)` config for opt-in/out
- Hook into existing config system (omegaconf) for monitoring settings

**References**:
- Python decorators: https://realpython.com/primer-on-python-decorators/
- Context managers: https://docs.python.org/3/library/contextlib.html

---

## Summary of Decisions

| Topic | Decision | Key Rationale |
|-------|----------|---------------|
| Logging | Loguru | Zero-config structured logging, <2% overhead |
| Metrics Storage | PyArrow/Parquet | 70-90% compression, sub-second queries |
| Memory Tracking | psutil + pynvml | Cross-platform, process-level, GPU support |
| Accuracy Metrics | scikit-learn + optional torchmetrics | Covers common cases, extensible for advanced |
| Annotation Formats | Multi-format with Pydantic validation | Industry standards (COCO, YOLO, PLY) + extensible |
| Dashboard Viz | Plotly (existing Phase 2) | Interactive charts, already integrated |
| Query Interface | Python API (builder pattern) | Type-safe, no query language needed |
| Pipeline Integration | Decorators + context managers | Non-invasive, backward compatible |

## Implementation Risks & Mitigations

### Risk 1: Parquet Query Performance on Large Datasets
- **Mitigation**: Partition by stage and date, use PyArrow filters
- **Fallback**: If >100k runs, document option to use DuckDB for SQL queries

### Risk 2: GPU Memory Tracking Unavailable (Non-NVIDIA)
- **Mitigation**: Graceful degradation to CPU-only, nullable GPU columns
- **Future**: Add support for ROCm (AMD) if users request

### Risk 3: Ground Truth Format Variations (Custom COCO Extensions)
- **Mitigation**: Provide plugin system, users can register custom Pydantic models
- **Documentation**: Show examples of extending base loaders

### Risk 4: Monitoring Overhead Exceeds 5% Target
- **Mitigation**: Make profiling opt-in per stage with `@profile(enabled=config.enable_profiling)`
- **Benchmarking**: Add pytest-benchmark tests to catch regressions

## Next Steps

Proceed to Phase 1: Design artifacts (data-model.md, contracts/, quickstart.md)
