#!/usr/bin/env python3
"""
Simple Demo - Phase 3 Features

Demonstrates Phase 3 observability features without complex pipelines:
- Performance profiling with metrics storage
- Audit logging with run tracking
- Structured logging
- HTML dashboard generation

This is the simplest example to understand Phase 3 capabilities.
"""
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from viz_art.performance import create_profiler, create_metrics_storage, create_dashboard
from viz_art.audit import create_logger, create_run_tracker
import numpy as np


def simulate_image_processing():
    """Simulate some image processing work."""
    time.sleep(0.1)  # Simulate work
    return np.random.rand(480, 640, 3)


def simulate_pointcloud_processing():
    """Simulate some point cloud processing work."""
    time.sleep(0.15)  # Simulate work
    return np.random.rand(1000, 3)


def main():
    print("=" * 80)
    print("Simple Phase 3 Features Demo")
    print("=" * 80)

    output_dir = Path(__file__).parent / "output" / "simple_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ==============================================================================
    # Phase 3 Setup
    # ==============================================================================

    print("\n📋 Setting up Phase 3 features...\n")

    # 1. Performance Profiling
    metrics_storage = create_metrics_storage(
        output_dir=output_dir / "metrics",
        retention_days=90,
    )
    profiler = create_profiler(storage=metrics_storage, enabled=True)
    print("✓ Performance profiling enabled (metrics stored in Parquet)")

    # 2. Run Tracking
    tracker = create_run_tracker(output_dir=output_dir / "runs")
    print("✓ Run tracking enabled (JSON metadata)")

    print()

    # ==============================================================================
    # Execute Pipeline with Full Observability
    # ==============================================================================

    config_snapshot = {
        "demo": "simple_phase3",
        "version": "1.0",
        "features": ["profiling", "logging", "dashboard"],
    }

    with tracker.track(
        config_snapshot=config_snapshot,
        input_files=[Path("sample_data.txt")],  # Hypothetical input
        output_dir=output_dir,
    ) as run_id:
        print(f"🔖 Started run: {run_id}\n")

        # Bind profiler to this run
        profiler._current_run_id = run_id

        # 3. Structured Logging
        logger = create_logger(
            run_id=run_id,
            output_dir=output_dir / "logs",
            retention="30 days",
        )

        logger.info("Demo started", mode="simple")

        # Process some data with profiling
        print("🔄 Processing data with performance monitoring...\n")

        # Stage 1: Image Processing
        logger.info("Processing images", count=3)
        for i in range(3):
            with profiler.measure(f"image_processing"):
                result = simulate_image_processing()
                logger.info(f"Processed image {i+1}", shape=result.shape)

        # Stage 2: Point Cloud Processing
        logger.info("Processing point clouds", count=2)
        for i in range(2):
            with profiler.measure(f"pointcloud_processing"):
                result = simulate_pointcloud_processing()
                logger.info(f"Processed cloud {i+1}", num_points=len(result))

        logger.info("Demo completed successfully")
        print("✓ All processing completed\n")

    # ==============================================================================
    # Generate Reports
    # ==============================================================================

    print("📊 Generating comprehensive report...\n")

    # Generate comprehensive report with embedded charts
    try:
        from datetime import datetime
        import re

        dashboard = create_dashboard(storage=metrics_storage)

        # Generate individual charts
        timing_html = dashboard.render_timing_chart(run_id=run_id)
        memory_html = dashboard.render_memory_chart(run_id=run_id)

        # Extract chart divs (everything between <body> tags)
        timing_body = re.search(r'<body>(.*?)</body>', timing_html, re.DOTALL)
        memory_body = re.search(r'<body>(.*?)</body>', memory_html, re.DOTALL)

        timing_content = timing_body.group(1) if timing_body else "<p>Chart not available</p>"
        memory_content = memory_body.group(1) if memory_body else "<p>Chart not available</p>"


    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"✓ Run ID: {run_id}")
    print(f"✓ Metrics stored: {output_dir}/metrics")
    print()
    print("ℹ️  For HTML reports with visualizations, use the batch processing workflow:")
    print("   python examples/demo_batch_with_phase3.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
