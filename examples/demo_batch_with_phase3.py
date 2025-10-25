#!/usr/bin/env python3
"""
Batch Processing with Phase 3 Integration

Demonstrates the complete integration:
- Phase 2: Batch processing with stage-by-stage views
- Phase 3: Performance, validation, and logging

This uses the EXTENDED batch reporter that includes ALL features.
"""
import sys
import json
import re
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from viz_art.pipeline import Pipeline, PipelineStage
from viz_art.batch import BatchProcessor, HTMLReporter
from viz_art.performance import create_profiler, create_metrics_storage, create_dashboard
from viz_art.audit import create_logger, create_run_tracker
from viz_art.config.schema import BatchConfigItem, OutputSaveConfig
from typing import Dict, Any, List
import numpy as np
from PIL import Image


# Simple stages for the demo
class ImageLoader(PipelineStage):
    @property
    def name(self) -> str:
        return "loader"

    @property
    def input_keys(self) -> List[str]:
        return ["image_path"]

    @property
    def output_keys(self) -> List[str]:
        return ["image"]

    def pre_process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs

    def predict(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        img = Image.open(preprocessed["image_path"])
        return {"image": np.array(img)}

    def post_process(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        return predictions


class GrayscaleStage(PipelineStage):
    @property
    def name(self) -> str:
        return "grayscale"

    @property
    def input_keys(self) -> List[str]:
        return ["image"]

    @property
    def output_keys(self) -> List[str]:
        return ["image"]

    def pre_process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs

    def predict(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        img = preprocessed["image"]
        if len(img.shape) == 3:
            gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
            return {"image": gray.astype(np.uint8)}
        return preprocessed

    def post_process(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        return predictions


class ImageResizer(PipelineStage):
    def __init__(self, target_size=(320, 240)):
        self.target_size = target_size

    @property
    def name(self) -> str:
        return "resizer"

    @property
    def input_keys(self) -> List[str]:
        return ["image"]

    @property
    def output_keys(self) -> List[str]:
        return ["image"]

    def pre_process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs

    def predict(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        img = Image.fromarray(preprocessed["image"])
        resized = img.resize(self.target_size, Image.Resampling.LANCZOS)
        return {"image": np.array(resized)}

    def post_process(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        return predictions


def simulate_ground_truth_validation():
    """Simulate ground truth validation metrics."""
    return {
        "precision": 0.85,
        "recall": 0.82,
        "f1_score": 0.835,
        "total_objects": 15,
        "errors": [
            {"image_id": "sample_001", "type": "false_negative", "predicted": 4, "expected": 5},
            {"image_id": "sample_002", "type": "false_positive", "predicted": 6, "expected": 5},
        ],
    }


def main():
    print("=" * 80)
    print("Batch Processing with Phase 3 Integration Demo")
    print("=" * 80)

    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    output_dir = base_dir / "output" / "batch_phase3_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find sample images
    image_files = sorted(data_dir.glob("sample_*.jpg"))
    if not image_files:
        print("❌ No sample images found. Run: python examples/data/create_sample_data.py")
        return

    print(f"\n📁 Found {len(image_files)} images to process\n")

    # ==============================================================================
    # Phase 3 Setup
    # ==============================================================================

    print("📋 Setting up Phase 3 observability...\n")

    # Performance profiling
    metrics_storage = create_metrics_storage(
        output_dir=output_dir / "metrics",
        retention_days=90,
    )
    profiler = create_profiler(storage=metrics_storage, enabled=True)
    print("✓ Performance profiling enabled")

    # Run tracking
    tracker = create_run_tracker(output_dir=output_dir / "runs")
    print("✓ Run tracking enabled\n")

    # ==============================================================================
    # Build Pipeline
    # ==============================================================================

    pipeline = Pipeline(name="ImageProcessingPipeline")
    pipeline.add_stage(ImageLoader())
    pipeline.add_stage(GrayscaleStage())
    pipeline.add_stage(ImageResizer(target_size=(320, 240)))

    # Batch configuration
    batch_config = BatchConfigItem(
        input_dir=str(data_dir),
        output_dir=str(output_dir),
        file_patterns=["sample_*.jpg"],
        output_mode="validation",
        save_outputs=OutputSaveConfig(
            enabled=True,
            stages=["all"],
            format="png",
        ),
    )

    # ==============================================================================
    # Batch Processing with Phase 3 Tracking
    # ==============================================================================

    config_snapshot = {
        "pipeline": "image_processing",
        "stages": ["loader", "grayscale", "resizer"],
        "target_size": (320, 240),
    }

    with tracker.track(
        config_snapshot=config_snapshot,
        input_files=image_files,
        output_dir=output_dir,
    ) as run_id:
        print(f"🔖 Started run: {run_id}\n")

        # Bind profiler to this run
        profiler._current_run_id = run_id

        # Create logger
        logger = create_logger(
            run_id=run_id,
            output_dir=output_dir / "logs",
            retention="30 days",
        )

        logger.info("Batch processing started", num_images=len(image_files))

        # Run batch processing
        print("🔄 Processing images through pipeline...\n")

        batch_processor = BatchProcessor(
            pipeline=pipeline,
            config=batch_config,
            profiler=profiler,  # Pass profiler for per-stage metrics
        )

        # Run batch processing (stages are profiled automatically)
        batch_result = batch_processor.run()

        logger.info(
            "Batch processing completed",
            total=batch_result.total_files,
            successful=batch_result.successful,
            failed=batch_result.failed,
        )

        print(f"\n✓ Processed {batch_result.successful}/{batch_result.total_files} images\n")

    # ==============================================================================
    # Generate Phase 3 Data
    # ==============================================================================

    print("📊 Preparing Phase 3 metrics...\n")

    # Generate performance charts
    dashboard = create_dashboard(storage=metrics_storage)

    timing_html = dashboard.render_timing_chart(run_id=run_id)
    memory_html = dashboard.render_memory_chart(run_id=run_id)

    # Extract chart body content
    timing_body = re.search(r'<body>(.*?)</body>', timing_html, re.DOTALL)
    memory_body = re.search(r'<body>(.*?)</body>', memory_html, re.DOTALL)

    phase3_metrics = {
        "timing_chart": timing_body.group(1) if timing_body else "",
        "memory_chart": memory_body.group(1) if memory_body else "",
    }

    # Simulate ground truth validation
    phase3_validation = simulate_ground_truth_validation()

    # Load audit logs
    phase3_logs = []
    try:
        log_file = list((output_dir / "logs").glob("*.jsonl"))[0]
        with open(log_file) as f:
            for line in f:
                phase3_logs.append(json.loads(line))
    except:
        pass

    print(f"✓ Phase 3 metrics prepared")
    print(f"  ├─ Performance charts: {len(phase3_metrics)} charts")
    print(f"  ├─ Validation metrics: Precision={phase3_validation['precision']:.1%}")
    print(f"  └─ Audit logs: {len(phase3_logs)} entries\n")

    # ==============================================================================
    # Generate Integrated Report (Phase 2 + Phase 3)
    # ==============================================================================

    print("📄 Generating integrated report...\n")

    reporter = HTMLReporter()
    report_path = output_dir / "batch_report.html"

    reporter.generate(
        batch_result=batch_result,
        output_path=report_path,
        pipeline_name="Image Processing Pipeline",
        phase3_metrics=phase3_metrics,
        phase3_validation=phase3_validation,
        phase3_logs=phase3_logs,
    )

    print(f"✓ Integrated report generated: {report_path}")
    print(f"  Report includes:")
    print(f"  ✓ Phase 2: Stage-by-stage image views")
    print(f"  ✓ Phase 2: Per-image pipeline results")
    print(f"  ✓ Phase 2: Error summary")
    print(f"  ✓ Phase 3: Performance charts (timing + memory)")
    print(f"  ✓ Phase 3: Ground truth validation metrics")
    print(f"  ✓ Phase 3: Audit log entries")

    # ==============================================================================
    # Summary
    # ==============================================================================

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"✓ Run ID: {run_id}")
    print(f"✓ Processed: {batch_result.successful}/{batch_result.total_files} images")
    print(f"✓ Validation: Precision={phase3_validation['precision']:.1%}, Recall={phase3_validation['recall']:.1%}")
    print(f"✓ All outputs: {output_dir}")
    print()
    print("📁 View integrated report (Phase 2 + Phase 3):")
    print(f"   open {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
