#!/usr/bin/env python3
"""
Complete Pipeline Demo - Phase 2 + Phase 3 Integration

Demonstrates a complete pipeline with BOTH:
- Phase 2: Batch processing with stage-by-stage reporting
- Phase 3: Performance monitoring, validation, error analysis

This shows the full integration of all features.
"""
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from viz_art.pipeline import Pipeline, PipelineStage
from viz_art.batch import BatchProcessor, HTMLReporter
from viz_art.performance import create_profiler, create_metrics_storage, create_dashboard
from viz_art.audit import create_logger, create_run_tracker
from typing import Dict, Any, List
import numpy as np
from PIL import Image


# ==============================================================================
# Pipeline Stages (with Phase 3 profiling)
# ==============================================================================

class ImageLoaderStage(PipelineStage):
    """Load images with profiling."""

    @property
    def name(self) -> str:
        return "loader"

    @property
    def input_keys(self) -> List[str]:
        return ["image_path"]

    @property
    def output_keys(self) -> List[str]:
        return ["image", "metadata"]

    def pre_process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs

    def predict(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        img = Image.open(preprocessed["image_path"])
        return {
            "image": np.array(img),
            "metadata": {"shape": np.array(img).shape},
        }

    def post_process(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        return predictions


class GrayscaleStage(PipelineStage):
    """Convert to grayscale with profiling."""

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


class ResizerStage(PipelineStage):
    """Resize images with profiling."""

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


# ==============================================================================
# Main Pipeline with Full Integration
# ==============================================================================

def main():
    print("=" * 80)
    print("Complete Pipeline Demo - Phase 2 + Phase 3 Integration")
    print("=" * 80)

    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    output_dir = base_dir / "output" / "complete_pipeline"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find sample images
    image_files = sorted(data_dir.glob("sample_*.jpg"))
    if not image_files:
        print("❌ No sample images found. Run: python examples/data/create_sample_data.py")
        return

    print(f"\n📁 Found {len(image_files)} images to process\n")

    # ==============================================================================
    # Phase 3: Setup Observability
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

    pipeline = Pipeline(
        stages=[
            ImageLoaderStage(),
            GrayscaleStage(),
            ResizerStage(target_size=(320, 240)),
        ]
    )

    # ==============================================================================
    # Phase 2: Batch Processing with Phase 3 Tracking
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
            output_saver=None,  # We'll save manually for this demo
        )

        results = []
        for img_path in image_files:
            logger.info(f"Processing {img_path.name}")

            with profiler.measure("complete_pipeline"):
                try:
                    result = pipeline.run({"image_path": str(img_path)})

                    # Save output
                    output_path = output_dir / f"processed_{img_path.stem}.png"
                    Image.fromarray(result["image"]).save(output_path)

                    results.append({
                        "input": img_path,
                        "output": output_path,
                        "success": True,
                    })

                    logger.info(f"Successfully processed {img_path.name}", output=str(output_path))
                except Exception as e:
                    results.append({
                        "input": img_path,
                        "error": str(e),
                        "success": False,
                    })
                    logger.error(f"Failed to process {img_path.name}", error=str(e))

        logger.info("Batch processing completed", total=len(results), successful=sum(1 for r in results if r["success"]))

    # ==============================================================================
    # Generate Comprehensive Report (Phase 2 style + Phase 3 metrics)
    # ==============================================================================

    print(f"\n📊 Generating comprehensive report...\n")

    try:
        import re
        from datetime import datetime

        dashboard = create_dashboard(storage=metrics_storage)

        # Generate Phase 3 charts
        timing_html = dashboard.render_timing_chart(run_id=run_id)
        memory_html = dashboard.render_memory_chart(run_id=run_id)

        # Extract chart content
        timing_body = re.search(r'<body>(.*?)</body>', timing_html, re.DOTALL)
        memory_body = re.search(r'<body>(.*?)</body>', memory_html, re.DOTALL)

        timing_content = timing_body.group(1) if timing_body else ""
        memory_content = memory_body.group(1) if memory_body else ""

        # Build stage-by-stage view (Phase 2 style)
        stages_html = ""
        for stage_name in ["loader", "grayscale", "resizer"]:
            stages_html += f"""
            <div class="stage-section">
                <h3>📷 Stage: {stage_name.title()}</h3>
                <div class="image-grid">
            """

            for result in results:
                if result["success"]:
                    img_name = result["input"].name
                    stages_html += f"""
                    <div class="image-card">
                        <img src="processed_{result['input'].stem}.png" alt="{img_name}" style="width: 100%; height: auto;">
                        <div class="image-caption">{img_name}</div>
                    </div>
                    """

            stages_html += """
                </div>
            </div>
            """


    # Summary  
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"✓ Run ID: {run_id}")
    print(f"✓ Metrics stored: {output_dir}/metrics")
    print(f"✓ Logs stored: {output_dir}/logs")
    print()
    print("ℹ️  For HTML reports with visualizations, use the batch processing workflow:")
    print("   python examples/demo_batch_with_phase3.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
