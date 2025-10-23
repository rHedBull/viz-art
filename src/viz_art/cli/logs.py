"""CLI tool for querying audit logs.

Usage:
    viz-art-logs --after 2025-10-20 --stage detection --failed
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def query_logs_cli():
    """Query audit logs from command line."""
    parser = argparse.ArgumentParser(
        description="Query viz-art audit logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Query all logs from yesterday
  viz-art-logs --after 2025-10-22

  # Query failed runs in detection stage
  viz-art-logs --stage detection --failed --limit 10

  # Export logs to file
  viz-art-logs --after 2025-10-20 --output filtered.json

  # Query specific run
  viz-art-logs --run-id abc-123-def-456
        """,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/logs"),
        help="Directory containing log files (default: output/logs)",
    )

    parser.add_argument(
        "--after",
        type=str,
        help="Filter logs after this date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--before",
        type=str,
        help="Filter logs before this date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--run-id",
        type=str,
        help="Filter by specific run ID",
    )

    parser.add_argument(
        "--stage",
        type=str,
        help="Filter by stage name",
    )

    parser.add_argument(
        "--level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Filter by log level",
    )

    parser.add_argument(
        "--failed",
        action="store_true",
        help="Show only failed runs (ERROR and CRITICAL)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of results (default: 100)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Export results to JSON file",
    )

    args = parser.parse_args()

    # Import here to avoid slow startup
    from viz_art.audit import create_query

    # Build query
    query = create_query(args.output_dir)

    if args.after:
        try:
            after_date = datetime.fromisoformat(args.after)
            query = query.after(after_date)
        except ValueError:
            print(f"Error: Invalid date format for --after: {args.after}", file=sys.stderr)
            print("Expected format: YYYY-MM-DD", file=sys.stderr)
            sys.exit(1)

    if args.before:
        try:
            before_date = datetime.fromisoformat(args.before)
            query = query.before(before_date)
        except ValueError:
            print(f"Error: Invalid date format for --before: {args.before}", file=sys.stderr)
            print("Expected format: YYYY-MM-DD", file=sys.stderr)
            sys.exit(1)

    if args.run_id:
        query = query.run_id(args.run_id)

    if args.stage:
        query = query.stage(args.stage)

    if args.level:
        from viz_art.types.monitoring import LogLevel

        query = query.level(LogLevel[args.level])

    if args.failed:
        query = query.failed()

    query = query.limit(args.limit)

    # Execute query
    try:
        results = query.fetch()
    except FileNotFoundError:
        print(f"Error: Log directory not found: {args.output_dir}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error querying logs: {e}", file=sys.stderr)
        sys.exit(1)

    # Output results
    if args.output:
        # Export to file
        count = query.export_json(args.output)
        print(f"Exported {count} log entries to {args.output}")
    else:
        # Print to stdout
        if not results:
            print("No logs found matching criteria")
        else:
            print(f"Found {len(results)} log entries:\n")
            for log in results:
                timestamp = log.get("timestamp", "N/A")
                level = log.get("level", "INFO")
                stage = log.get("stage_name", "pipeline")
                message = log.get("message", "")
                run_id = log.get("run_id", "")[:8]

                print(f"[{timestamp}] {level:8} {stage:15} (run:{run_id}) {message}")

                # Show metadata if present
                if log.get("metadata"):
                    metadata_str = json.dumps(log["metadata"], indent=2)
                    print(f"  Metadata: {metadata_str}")

                print()


if __name__ == "__main__":
    query_logs_cli()
