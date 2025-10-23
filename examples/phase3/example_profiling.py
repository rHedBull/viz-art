"""
Example demonstrating performance profiling with @profiler decorator.

T055-T056: Profiling example with dashboard generation
"""

from pathlib import Path
import time
import numpy as np

from viz_art.performance import create_profiler, create_metrics_storage, create_dashboard


# Sample stage to profile
class SampleStage:
    """Example stage for profiling demonstration."""

    def execute(self, inputs):
        """Simulate some processing work."""
        # Simulate computation
        time.sleep(0.1)
        result = np.random.rand(1000, 1000)
        time.sleep(0.05)
        return result


def main():
    """T055: Demonstrate @profiler decorator usage on sample stage"""

    print("=== Performance Profiling Example ===\n")

    # Setup storage and profiler
    output_dir = Path("output/metrics")
    storage = create_metrics_storage(output_dir)
    profiler = create_profiler(storage, enabled=True)

    # Create and profile a sample stage
    stage = SampleStage()

    # Wrap the execute method with profiler
    stage.execute = profiler(stage.execute)

    # Run the stage multiple times to collect metrics
    print("Running sample stage 5 times...")
    for i in range(5):
        result = stage.execute({"data": f"sample_{i}"})
        print(f"  Run {i+1} completed")

    print("\nMetrics collected and stored in Parquet format.")

    # Query metrics to verify
    metrics = storage.query_metrics(stage_name="SampleStage")
    print(f"\nCollected {len(metrics)} metric entries")

    if metrics:
        print("\nSample metrics:")
        for i, metric in enumerate(metrics[:3], 1):
            print(f"  {i}. Execution time: {metric['execution_time_ms']:.2f}ms, "
                  f"CPU memory: {metric['cpu_memory_mb']:.2f}MB")

        # Get aggregate statistics
        stats = storage.get_aggregate_stats("SampleStage")
        print(f"\nAggregate Statistics:")
        print(f"  Mean: {stats['mean']:.2f}ms")
        print(f"  Median: {stats['median']:.2f}ms")
        print(f"  P95: {stats['p95']:.2f}ms")
        print(f"  P99: {stats['p99']:.2f}ms")

    # T056: Generate performance dashboard with HTML output
    print("\n=== Generating Performance Dashboard ===\n")

    dashboard = create_dashboard(storage)

    # Get run_id from first metric
    if metrics:
        run_id = metrics[0]['run_id']
        print(f"Generating dashboard for run: {run_id}")

        # Generate timing chart
        output_path = Path("output/performance_report.html")
        html = dashboard.render_timing_chart(run_id, output_path=output_path)

        print(f"\nDashboard saved to: {output_path}")
        print("Open in browser to view interactive charts!")

        # Also generate memory chart
        memory_path = Path("output/memory_report.html")
        dashboard.render_memory_chart(run_id, output_path=memory_path)
        print(f"Memory chart saved to: {memory_path}")

    print("\n=== Example Complete ===")
    print("\nNext steps:")
    print("1. Open output/performance_report.html in a browser")
    print("2. Try profiling your own pipeline stages")
    print("3. Use the dashboard to identify bottlenecks")


if __name__ == "__main__":
    main()
