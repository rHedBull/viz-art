"""Example: Historical Metrics and Long-Term Storage

This example demonstrates:
1. Multi-day metric accumulation
2. Retention policy configuration
3. Aggregate statistics calculation (median, p95, p99)
4. Trend chart generation with date range filtering
5. Automatic cleanup of expired data
"""

from pathlib import Path
from datetime import datetime, timedelta
import time

from viz_art.performance import create_metrics_storage, create_dashboard


def simulate_pipeline_runs(storage, num_runs: int = 30, days: int = 7):
    """
    Simulate multiple pipeline runs across several days.

    Args:
        storage: MetricsStorage instance
        num_runs: Number of runs to simulate
        days: Number of days to spread runs across
    """
    print(f"Simulating {num_runs} pipeline runs over {days} days...")

    stages = ["preprocessing", "detection", "postprocessing"]
    base_time = datetime.utcnow() - timedelta(days=days)

    for i in range(num_runs):
        # Spread runs across days
        run_time = base_time + timedelta(
            days=i * days / num_runs,
            hours=(i % 24),
        )

        run_id = f"run_{i:03d}"

        # Simulate increasing execution times (performance degradation)
        degradation_factor = 1.0 + (i / num_runs) * 0.3  # Up to 30% slower

        for stage_idx, stage in enumerate(stages):
            # Base execution times
            base_exec_times = {
                "preprocessing": 150.0,
                "detection": 500.0,
                "postprocessing": 100.0,
            }

            exec_time = base_exec_times[stage] * degradation_factor

            # Add some noise
            import random
            exec_time *= random.uniform(0.9, 1.1)

            # Write metrics
            storage.write_metrics(
                run_id=run_id,
                stage_name=stage,
                execution_time_ms=exec_time,
                cpu_memory_mb=256.0 * random.uniform(0.8, 1.2),
                gpu_memory_mb=512.0 * random.uniform(0.8, 1.2) if stage == "detection" else None,
                timestamp=run_time + timedelta(seconds=stage_idx * 2),
                success=True,
            )

        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{num_runs} runs")

    print(f"✓ Simulated {num_runs} runs")


def main():
    """Run historical metrics example."""
    print("=" * 80)
    print("Historical Metrics & Long-Term Storage Example")
    print("=" * 80)

    # Setup output directory
    output_dir = Path("output/historical_metrics_example")
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_dir = output_dir / "metrics"

    # Step 1: Create storage with retention policy
    print("\n1. Creating metrics storage with retention policy...")

    # 7-day retention policy for this example
    storage = create_metrics_storage(metrics_dir, retention_days=7)

    print(f"   ✓ Storage created at: {metrics_dir}")
    print(f"   ✓ Retention policy: {storage.retention_days} days")

    # Step 2: Simulate multi-day metric accumulation
    print("\n2. Simulating multi-day pipeline runs...")

    simulate_pipeline_runs(storage, num_runs=50, days=10)

    # Step 3: Query and display aggregate statistics
    print("\n3. Calculating aggregate statistics...")

    for stage_name in ["preprocessing", "detection", "postprocessing"]:
        stats = storage.get_aggregate_stats(
            stage_name=stage_name,
            start_date=datetime.utcnow() - timedelta(days=10),
            end_date=datetime.utcnow(),
        )

        print(f"\n   Stage: {stage_name}")
        print(f"   - Count: {stats['count']} runs")
        print(f"   - Median: {stats['median']:.2f} ms")
        print(f"   - Mean: {stats['mean']:.2f} ms")
        print(f"   - Std Dev: {stats['std']:.2f} ms")
        print(f"   - Min: {stats['min']:.2f} ms")
        print(f"   - Max: {stats['max']:.2f} ms")
        print(f"   - P50: {stats['p50']:.2f} ms")
        print(f"   - P95: {stats['p95']:.2f} ms")
        print(f"   - P99: {stats['p99']:.2f} ms")

    # Step 4: Generate trend charts
    print("\n4. Generating trend charts...")

    dashboard = create_dashboard(storage)

    for stage_name in ["preprocessing", "detection", "postprocessing"]:
        trend_path = output_dir / f"trend_{stage_name}.html"

        try:
            html = dashboard.render_trend_chart(
                stage_name=stage_name,
                start_date=datetime.utcnow() - timedelta(days=10),
                end_date=datetime.utcnow(),
                output_path=trend_path,
                aggregate_by="daily",  # Aggregate by day
            )

            print(f"   ✓ Generated trend chart: {trend_path}")

        except Exception as e:
            print(f"   ✗ Error generating trend for {stage_name}: {e}")

    # Step 5: Demonstrate retention cleanup
    print("\n5. Testing retention policy and cleanup...")

    # Simulate old metrics (before retention window)
    old_date = datetime.utcnow() - timedelta(days=15)

    for i in range(5):
        storage.write_metrics(
            run_id=f"old_run_{i}",
            stage_name="preprocessing",
            execution_time_ms=100.0,
            cpu_memory_mb=128.0,
            timestamp=old_date + timedelta(hours=i),
            success=True,
        )

    print(f"   ✓ Added 5 old metrics (15 days ago)")

    # Force cleanup
    deleted = storage._cleanup_old_metrics()
    print(f"   ✓ Cleanup removed {deleted} expired files/records")

    # Verify old metrics are gone
    old_metrics = storage.query_metrics(
        stage_name="preprocessing",
        start_date=old_date - timedelta(days=1),
        end_date=old_date + timedelta(days=1),
    )

    print(f"   ✓ Remaining old metrics: {len(old_metrics)}")

    # Step 6: Disk usage monitoring
    print("\n6. Checking disk usage...")

    disk_usage = storage._get_disk_usage()

    print(f"   ✓ Total disk: {disk_usage['total_mb']:.0f} MB")
    print(f"   ✓ Used: {disk_usage['used_mb']:.0f} MB ({disk_usage['usage_percent']:.1f}%)")
    print(f"   ✓ Free: {disk_usage['free_mb']:.0f} MB")

    if disk_usage['usage_percent'] > 90:
        print("   ⚠️  Warning: Disk usage is high!")

    # Step 7: Query specific date ranges
    print("\n7. Querying specific date ranges...")

    # Last 3 days
    recent_metrics = storage.query_metrics(
        stage_name="detection",
        start_date=datetime.utcnow() - timedelta(days=3),
        end_date=datetime.utcnow(),
    )

    print(f"   ✓ Last 3 days: {len(recent_metrics)} metrics for 'detection' stage")

    # Specific run IDs
    run_ids = ["run_010", "run_020", "run_030"]
    specific_metrics = storage.query_metrics(
        stage_name="detection",
        run_ids=run_ids,
    )

    print(f"   ✓ Specific runs: {len(specific_metrics)} metrics for runs {run_ids}")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"✓ Storage location: {metrics_dir}")
    print(f"✓ Total metrics stored: {sum(stats['count'] for stage_name in ['preprocessing', 'detection', 'postprocessing'] for stats in [storage.get_aggregate_stats(stage_name)])}")
    print(f"✓ Retention policy: {storage.retention_days} days")
    print(f"✓ Trend charts generated: {len(['preprocessing', 'detection', 'postprocessing'])}")
    print(f"✓ Disk usage: {disk_usage['usage_percent']:.1f}%")
    print("\nNext steps:")
    print("  - Open trend charts in browser to view historical performance")
    print("  - Monitor P95/P99 metrics for performance regressions")
    print("  - Adjust retention policy based on storage constraints")
    print("  - Set up automated alerts for performance degradations")


if __name__ == "__main__":
    main()
