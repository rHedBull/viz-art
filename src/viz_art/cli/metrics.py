"""CLI tool for viewing performance metrics.

Usage:
    viz-art-metrics --stage detection --start 2025-10-01
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path


def view_metrics_cli():
    """View performance metrics from command line."""
    parser = argparse.ArgumentParser(
        description="View viz-art performance metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View metrics for detection stage
  viz-art-metrics --stage detection

  # View metrics for date range
  viz-art-metrics --stage detection --start 2025-10-01 --end 2025-10-23

  # Generate dashboard HTML
  viz-art-metrics --stage detection --output dashboard.html
        """,
    )

    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("output/metrics"),
        help="Directory containing metrics files (default: output/metrics)",
    )

    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        help="Stage name to query",
    )

    parser.add_argument(
        "--start",
        type=str,
        help="Start date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--end",
        type=str,
        help="End date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Generate dashboard HTML to this file",
    )

    args = parser.parse_args()

    # Import here to avoid slow startup
    from viz_art.performance import create_metrics_storage, create_dashboard

    # Create storage
    try:
        storage = create_metrics_storage(args.metrics_dir)
    except Exception as e:
        print(f"Error: Could not access metrics storage: {e}", file=sys.stderr)
        sys.exit(1)

    # Parse dates
    start_date = None
    end_date = None

    if args.start:
        try:
            start_date = datetime.fromisoformat(args.start)
        except ValueError:
            print(f"Error: Invalid date format for --start: {args.start}", file=sys.stderr)
            print("Expected format: YYYY-MM-DD", file=sys.stderr)
            sys.exit(1)

    if args.end:
        try:
            end_date = datetime.fromisoformat(args.end)
        except ValueError:
            print(f"Error: Invalid date format for --end: {args.end}", file=sys.stderr)
            print("Expected format: YYYY-MM-DD", file=sys.stderr)
            sys.exit(1)

    # Get aggregate statistics
    try:
        stats = storage.get_aggregate_stats(
            stage_name=args.stage,
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as e:
        print(f"Error querying metrics: {e}", file=sys.stderr)
        sys.exit(1)

    if stats["count"] == 0:
        print(f"No metrics found for stage '{args.stage}'")
        sys.exit(0)

    # Display statistics
    print(f"Performance Metrics for Stage: {args.stage}")
    print("=" * 60)
    print(f"Total Runs:     {stats['count']}")
    print(f"Median Time:    {stats['median']:.2f} ms")
    print(f"Mean Time:      {stats['mean']:.2f} ms")
    print(f"Std Dev:        {stats['std']:.2f} ms")
    print(f"Min Time:       {stats['min']:.2f} ms")
    print(f"Max Time:       {stats['max']:.2f} ms")
    print()
    print("Percentiles:")
    print(f"  P50 (median): {stats['p50']:.2f} ms")
    print(f"  P95:          {stats['p95']:.2f} ms")
    print(f"  P99:          {stats['p99']:.2f} ms")

    # Generate dashboard if requested
    if args.output:
        dashboard = create_dashboard(storage)

        # Need at least one run_id for timing chart
        # Query recent metrics to get a run_id
        recent_metrics = storage.query_metrics(
            stage_name=args.stage,
            start_date=start_date,
            end_date=end_date,
        )

        if recent_metrics:
            # Use most recent run
            recent_metrics.sort(key=lambda m: m["timestamp"], reverse=True)
            run_id = recent_metrics[0]["run_id"]

            try:
                dashboard.render_timing_chart(
                    run_id=run_id,
                    output_path=args.output,
                )
                print(f"\nDashboard saved to: {args.output}")
            except Exception as e:
                print(f"\nWarning: Could not generate dashboard: {e}", file=sys.stderr)
        else:
            print("\nWarning: No recent runs found, cannot generate dashboard", file=sys.stderr)


if __name__ == "__main__":
    view_metrics_cli()
