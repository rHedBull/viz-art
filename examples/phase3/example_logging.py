"""
Example demonstrating audit logging and run tracking.

T093-T096: Logging example with queries and export
"""

from pathlib import Path
from datetime import datetime, timedelta
import time

from viz_art.audit import create_run_tracker, create_logger, create_query
from viz_art.types import LogLevel


def simulate_pipeline_run(run_id: str, should_fail: bool = False):
    """
    Simulate a pipeline run with logging.

    T094: Structured logging examples with different log levels
    """
    logger = create_logger(run_id, Path("output/logs"))

    logger.info("Pipeline started", input_count=10, config_version="1.0")

    # Simulate stages
    stages = ["preprocessing", "detection", "postprocessing"]

    for i, stage_name in enumerate(stages):
        # Bind stage context
        stage_logger = logger.bind(stage_name=stage_name)

        stage_logger.info(f"Stage started", stage_index=i)

        # Simulate work
        time.sleep(0.1)

        if should_fail and stage_name == "detection":
            # Simulate failure
            stage_logger.error(
                "Stage failed: Invalid input shape",
                error_type="ValueError",
                input_shape=[3, 224, 224],
                expected_shape=[3, 512, 512],
            )
            raise ValueError("Invalid input shape")

        stage_logger.info(
            f"Stage completed",
            stage_index=i,
            execution_time_ms=100 + i * 50,
        )

    logger.info("Pipeline completed successfully", total_stages=len(stages))


def main():
    """T093: Demonstrate RunTracker context manager usage"""

    print("=== Audit Logging and Run Tracking Example ===\n")

    # Setup run tracker
    tracker = create_run_tracker(Path("output/runs"))

    # Run 1: Successful run
    print("Running successful pipeline...")
    try:
        with tracker.track(
            config_snapshot={"version": "1.0", "mode": "production"},
            input_files=[Path("data/sample1.jpg")],
            output_dir=Path("output/run1"),
        ) as run_id:
            print(f"  Run ID: {run_id}")
            simulate_pipeline_run(run_id, should_fail=False)
            print("  Status: COMPLETED\n")
    except Exception as e:
        print(f"  Status: FAILED - {e}\n")

    # Run 2: Failed run
    print("Running pipeline with simulated failure...")
    try:
        with tracker.track(
            config_snapshot={"version": "1.0", "mode": "test"},
            input_files=[Path("data/sample2.jpg")],
            output_dir=Path("output/run2"),
        ) as run_id:
            print(f"  Run ID: {run_id}")
            simulate_pipeline_run(run_id, should_fail=True)
            print("  Status: COMPLETED\n")
    except Exception as e:
        print(f"  Status: FAILED - {e}\n")

    # Run 3: Another successful run
    print("Running another successful pipeline...")
    try:
        with tracker.track(
            config_snapshot={"version": "1.1", "mode": "production"},
            input_files=[Path("data/sample3.jpg")],
            output_dir=Path("output/run3"),
        ) as run_id:
            print(f"  Run ID: {run_id}")
            simulate_pipeline_run(run_id, should_fail=False)
            print("  Status: COMPLETED\n")
    except Exception as e:
        print(f"  Status: FAILED - {e}\n")

    # T095: Query examples showing filter combinations
    print("=== Querying Audit Logs ===\n")

    query = create_query(Path("output/logs"))

    # Query 1: Recent logs
    print("1. Logs from last hour:")
    recent_logs = query.after(datetime.utcnow() - timedelta(hours=1)).fetch()
    print(f"   Found {len(recent_logs)} log entries\n")

    # Query 2: Failed runs only
    print("2. Failed runs (ERROR and CRITICAL levels):")
    failed_logs = query.failed().fetch()
    print(f"   Found {len(failed_logs)} error entries")
    if failed_logs:
        for log in failed_logs[:3]:
            msg = log.get('record', {}).get('message', 'N/A')
            print(f"   - {msg}")
    print()

    # Query 3: Specific stage
    print("3. Logs for 'detection' stage:")
    detection_logs = query.stage("detection").fetch()
    print(f"   Found {len(detection_logs)} entries for detection stage\n")

    # Query 4: Limit results
    print("4. Last 5 log entries:")
    limited_logs = query.limit(5).fetch()
    print(f"   Found {len(limited_logs)} entries (limited to 5)\n")

    # T096: JSON export example
    print("=== Exporting Logs ===\n")

    # Export failed logs to JSON
    export_path = Path("output/failed_logs.json")
    count = create_query(Path("output/logs")).failed().export_json(export_path)

    print(f"Exported {count} failed log entries to {export_path}")

    # Export recent logs with date filter
    export_path2 = Path("output/recent_logs.json")
    count2 = (
        create_query(Path("output/logs"))
        .after(datetime.utcnow() - timedelta(hours=1))
        .export_json(export_path2)
    )

    print(f"Exported {count2} recent log entries to {export_path2}")

    print("\n=== Example Complete ===")
    print("\nNext steps:")
    print("1. Check output/logs/ for JSON Lines log files")
    print("2. Check output/runs/ for run metadata")
    print("3. Review exported JSON files for filtered logs")
    print("4. Try querying logs with different filter combinations")


if __name__ == "__main__":
    main()
