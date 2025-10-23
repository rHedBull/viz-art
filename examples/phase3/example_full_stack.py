"""Example: Full Observability Stack

This example demonstrates combining ALL Phase 3 features:
1. Performance profiling (@profiler decorators)
2. Audit logging (RunTracker context manager)
3. Ground truth validation (accuracy metrics)
4. Comprehensive reporting (performance + accuracy + logs)

This is the complete end-to-end workflow for maximum observability.
"""

from pathlib import Path
from datetime import datetime

from viz_art.performance import create_profiler, create_metrics_storage, create_dashboard
from viz_art.audit import create_run_tracker, create_logger, create_query
from viz_art.validation import (
    create_dataset,
    validate_pipeline,
    create_error_analyzer,
    AnnotationFormat,
)


def simulate_pipeline_stage(stage_name: str, inputs: dict) -> dict:
    """Simulate a pipeline stage execution.

    Args:
        stage_name: Name of the stage
        inputs: Input data

    Returns:
        Output data
    """
    import time
    import random

    # Simulate processing time
    time.sleep(random.uniform(0.1, 0.3))

    # Simulate stage-specific processing
    if stage_name == "preprocessing":
        return {"preprocessed": f"processed_{inputs.get('image', 'unknown')}"}
    elif stage_name == "detection":
        # Return sample detection results
        return {
            "detections": [
                {"bbox": [100, 100, 200, 200], "class": "car", "score": 0.95},
                {"bbox": [300, 150, 150, 180], "class": "person", "score": 0.87},
            ]
        }
    elif stage_name == "postprocessing":
        return {"final": "completed"}
    else:
        return {"output": "unknown"}


def main():
    """Run full observability stack example."""
    print("=" * 80)
    print("Full Observability Stack Example")
    print("=" * 80)

    # Setup output directory
    output_dir = Path("output/full_stack_example")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Setup profiling
    print("\n1. Setting up performance profiling...")

    metrics_storage = create_metrics_storage(
        output_dir / "metrics",
        retention_days=90,
    )
    profiler = create_profiler(metrics_storage, enabled=True)
    dashboard = create_dashboard(metrics_storage)

    print("   ✓ Profiling enabled with 90-day retention")

    # Step 2: Setup run tracking and logging
    print("\n2. Setting up audit logging...")

    tracker = create_run_tracker(output_dir / "runs")

    print("   ✓ Run tracking enabled")

    # Step 3: Setup ground truth validation
    print("\n3. Setting up ground truth validation...")

    # Use existing test fixtures
    dataset_root = Path("tests/fixtures/golden_datasets/bboxes")

    try:
        dataset = create_dataset(
            dataset_id="coco_sample",
            root_path=dataset_root,
            annotation_format=AnnotationFormat.COCO,
            annotation_files=[Path("coco_sample.json")],
            name="COCO Sample Dataset",
        )
        print(f"   ✓ Loaded dataset: {dataset.name} ({dataset.sample_count} samples)")
        validation_enabled = True
    except FileNotFoundError:
        print("   ⚠ Ground truth dataset not found, validation disabled")
        print("     Run example_validation.py first to create fixtures")
        validation_enabled = False

    # Step 4: Execute pipeline with full monitoring
    print("\n4. Executing pipeline with full monitoring...")

    config_snapshot = {
        "pipeline_name": "full_observability_demo",
        "profiling": True,
        "logging": True,
        "validation": validation_enabled,
    }

    input_files = [Path("examples/data/sample_image.jpg")]  # Hypothetical input

    with tracker.track(
        config=config_snapshot,
        input_files=input_files,
        output_dir=output_dir / "runs",
    ) as run_id:
        logger = create_logger(
            run_id,
            output_dir / "logs",
            retention="30 days",
        )

        logger.info("Pipeline started", stages=["preprocessing", "detection", "postprocessing"])

        # Execute stages with profiling
        stages = ["preprocessing", "detection", "postprocessing"]
        stage_outputs = {}

        for stage_name in stages:
            logger.info(f"Executing stage: {stage_name}")

            # Wrap stage execution with profiler
            @profiler
            def execute_stage(inputs):
                return simulate_pipeline_stage(stage_name, inputs)

            try:
                # Execute
                outputs = execute_stage({"image": "sample.jpg"})
                stage_outputs[stage_name] = outputs

                logger.info(
                    f"Stage completed: {stage_name}",
                    output_keys=list(outputs.keys()),
                )

            except Exception as e:
                logger.error(
                    f"Stage failed: {stage_name}",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise

        logger.info("Pipeline completed successfully")

        print(f"   ✓ Pipeline executed with run_id: {run_id}")

        # Step 5: Generate performance dashboard
        print("\n5. Generating performance dashboard...")

        dashboard_path = output_dir / f"performance_dashboard_{run_id[:8]}.html"

        try:
            dashboard.render_timing_chart(
                run_id=run_id,
                output_path=dashboard_path,
            )
            print(f"   ✓ Dashboard saved: {dashboard_path}")
        except Exception as e:
            print(f"   ⚠ Dashboard generation failed: {e}")

        # Step 6: Run validation
        if validation_enabled:
            print("\n6. Running ground truth validation...")

            # Prepare stage outputs for validation
            validation_outputs = {
                "detection": {
                    "image_001.jpg": stage_outputs.get("detection", {}).get("detections", []),
                }
            }

            try:
                validation_results = validate_pipeline(
                    run_id=run_id,
                    dataset=dataset,
                    stage_outputs=validation_outputs,
                    output_dir=output_dir / "runs" / run_id / "validation",
                )

                print("   ✓ Validation complete")

                for stage_name, metrics in validation_results.items():
                    print(f"\n   Stage: {stage_name}")
                    print(f"   - Precision: {metrics.get('precision', 0):.2%}")
                    print(f"   - Recall: {metrics.get('recall', 0):.2%}")
                    print(f"   - F1 Score: {metrics.get('f1_score', 0):.2%}")

            except Exception as e:
                print(f"   ⚠ Validation failed: {e}")
        else:
            print("\n6. Ground truth validation skipped (dataset not available)")

        # Step 7: Error analysis
        if validation_enabled:
            print("\n7. Performing error analysis...")

            analyzer = create_error_analyzer(output_dir / "error_analysis")

            # Get ground truth
            ground_truth_data = {}
            for sample_id, annotation in dataset.iter_samples():
                ground_truth_data[sample_id] = annotation

            # Find failures
            failures = analyzer.find_failures(
                run_id=run_id,
                predictions=validation_outputs.get("detection", {}),
                ground_truth=ground_truth_data,
                iou_threshold=0.5,
            )

            print(f"   ✓ Found {len(failures)} potential errors")

            if failures:
                # Generate error report
                error_report_path = output_dir / "error_analysis_report.html"
                analyzer.generate_error_report(failures, error_report_path)
                print(f"   ✓ Error report saved: {error_report_path}")
        else:
            print("\n7. Error analysis skipped (validation not available)")

        # Step 8: Query logs
        print("\n8. Querying audit logs...")

        query = create_query(output_dir / "logs")
        logs = query.run_id(run_id).fetch()

        print(f"   ✓ Retrieved {len(logs)} log entries for this run")

        # Show sample logs
        if logs:
            print("\n   Sample log entries:")
            for log in logs[:3]:
                level = log.get("level", "INFO")
                message = log.get("message", "")
                print(f"   - [{level}] {message}")

    # Step 9: Generate comprehensive report
    print("\n9. Generating comprehensive report...")

    report_path = output_dir / "comprehensive_report.html"

    report_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Comprehensive Pipeline Report - Run {run_id[:8]}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .section {{ margin: 20px 0; padding: 15px; background: #fafafa; border-left: 4px solid #4CAF50; }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
        .metric-label {{ font-weight: bold; color: #666; }}
        .metric-value {{ font-size: 1.2em; color: #333; }}
        .success {{ color: #4CAF50; }}
        .warning {{ color: #ff9800; }}
        .error {{ color: #f44336; }}
        a {{ color: #4CAF50; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Comprehensive Pipeline Report</h1>
        <p><strong>Run ID:</strong> {run_id}</p>
        <p><strong>Timestamp:</strong> {datetime.utcnow().isoformat()}</p>
        <p><strong>Status:</strong> <span class="success">✓ COMPLETED</span></p>

        <h2>Performance Metrics</h2>
        <div class="section">
            <p>Pipeline executed {len(stages)} stages successfully.</p>
            <div class="metric">
                <span class="metric-label">Total Stages:</span>
                <span class="metric-value">{len(stages)}</span>
            </div>
            <p><a href="{dashboard_path.name}">View detailed performance dashboard →</a></p>
        </div>

        <h2>Validation Results</h2>
        <div class="section">
            {'<p class="success">✓ Validation completed</p>' if validation_enabled else '<p class="warning">⚠ Validation skipped (dataset not available)</p>'}
            {f'<p>Found {len(failures)} error cases</p>' if validation_enabled and 'failures' in locals() else ''}
            {f'<p><a href="error_analysis_report.html">View error analysis report →</a></p>' if validation_enabled and failures else ''}
        </div>

        <h2>Audit Logs</h2>
        <div class="section">
            <p>Captured {len(logs)} log entries for this run.</p>
            <p>Query logs with: <code>viz-art-logs --run-id {run_id}</code></p>
        </div>

        <h2>Quick Links</h2>
        <div class="section">
            <ul>
                <li><a href="{dashboard_path.name}">Performance Dashboard</a></li>
                {'<li><a href="error_analysis_report.html">Error Analysis</a></li>' if validation_enabled and failures else ''}
                <li>Metrics: <code>viz-art-metrics --stage detection</code></li>
                <li>Logs: <code>viz-art-logs --run-id {run_id}</code></li>
            </ul>
        </div>
    </div>
</body>
</html>
"""

    with open(report_path, "w") as f:
        f.write(report_html)

    print(f"   ✓ Report saved: {report_path}")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"✓ Run ID: {run_id}")
    print(f"✓ Stages executed: {len(stages)}")
    print(f"✓ Profiling: Enabled")
    print(f"✓ Audit logging: Enabled ({len(logs)} entries)")
    print(f"✓ Validation: {'Enabled' if validation_enabled else 'Disabled'}")
    print(f"✓ Outputs saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  - Performance dashboard: {dashboard_path}")
    print(f"  - Comprehensive report: {report_path}")
    if validation_enabled and 'failures' in locals() and failures:
        print(f"  - Error analysis: {output_dir}/error_analysis_report.html")
    print("\nCLI commands to explore:")
    print(f"  viz-art-logs --run-id {run_id}")
    print(f"  viz-art-metrics --stage detection")
    print(f"  viz-art-validate --run-id {run_id} --dataset coco_sample")


if __name__ == "__main__":
    main()
