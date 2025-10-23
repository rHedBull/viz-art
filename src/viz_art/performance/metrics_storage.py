"""
Metrics storage with Parquet persistence.

T036-T042: MetricsStorage implementation with PyArrow
"""

from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import json
import os
import shutil

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pandas as pd
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False


class MetricsStorage:
    """
    Storage interface for performance metrics using Parquet format.

    T036-T042: Complete metrics storage implementation
    """

    def __init__(self, output_dir: Path, retention_days: int = 365):
        """
        Initialize metrics storage.

        T036: Constructor

        Args:
            output_dir: Directory for Parquet files (e.g., output/metrics/)
            retention_days: Auto-delete metrics older than this (0 = never delete)
        """
        self.output_dir = Path(output_dir)
        self.retention_days = retention_days
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not PYARROW_AVAILABLE:
            raise ImportError(
                "PyArrow is required for metrics storage. "
                "Install with: pip install pyarrow>=13.0"
            )

    def write_metrics(
        self,
        run_id: str,
        stage_name: str,
        execution_time_ms: float,
        cpu_memory_mb: float,
        gpu_memory_mb: Optional[float] = None,
        timestamp: Optional[datetime] = None,
        input_count: int = 1,
        output_count: int = 1,
        success: bool = True,
    ) -> None:
        """
        Write performance metrics for a stage execution.

        T037-T038: Write metrics with Parquet/Snappy compression

        Args:
            run_id: Unique run identifier (UUID v4)
            stage_name: Stage identifier
            execution_time_ms: Stage duration in milliseconds
            cpu_memory_mb: Peak CPU memory in MB
            gpu_memory_mb: Peak GPU memory in MB (None if unavailable)
            timestamp: Metric timestamp (defaults to now if None)
            input_count: Number of inputs processed
            output_count: Number of outputs generated
            success: Whether stage succeeded
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        # Create record
        record = {
            'run_id': run_id,
            'timestamp': timestamp,
            'stage_name': stage_name,
            'stage_index': 0,  # Will be set by pipeline if available
            'execution_time_ms': execution_time_ms,
            'cpu_memory_mb': cpu_memory_mb,
            'gpu_memory_mb': gpu_memory_mb,
            'input_count': input_count,
            'output_count': output_count,
            'success': success,
        }

        # T037: Append metrics to Parquet files partitioned by stage_name
        stage_file = self.output_dir / f"{stage_name}.parquet"

        # Create PyArrow table for this record
        table = pa.table({
            'run_id': [record['run_id']],
            'timestamp': [record['timestamp']],
            'stage_name': [record['stage_name']],
            'stage_index': [record['stage_index']],
            'execution_time_ms': [record['execution_time_ms']],
            'cpu_memory_mb': [record['cpu_memory_mb']],
            'gpu_memory_mb': [record['gpu_memory_mb']],
            'input_count': [record['input_count']],
            'output_count': [record['output_count']],
            'success': [record['success']],
        })

        # T038: Use Snappy compression
        if stage_file.exists():
            # Append to existing file
            existing = pq.read_table(stage_file)
            combined = pa.concat_tables([existing, table])
            pq.write_table(combined, stage_file, compression='snappy')
        else:
            # Create new file
            pq.write_table(table, stage_file, compression='snappy')

        # T156: Automatic cleanup trigger (every 100 writes, check for cleanup)
        # Use a simple counter file to track writes
        self._trigger_cleanup_periodically()

    def query_metrics(
        self,
        stage_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        run_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query performance metrics with filters.

        T039-T040: Query with PyArrow filter pushdown

        Args:
            stage_name: Filter by stage (None = all stages)
            start_date: Filter >= this date (None = no lower bound)
            end_date: Filter <= this date (None = no upper bound)
            run_ids: Filter by specific run IDs (None = all runs)

        Returns:
            List of metric dictionaries
        """
        results = []

        # Determine which files to scan
        if stage_name:
            files = [self.output_dir / f"{stage_name}.parquet"]
        else:
            files = list(self.output_dir.glob("*.parquet"))

        for file in files:
            if not file.exists():
                continue

            # T040: PyArrow filter pushdown for date range
            filters = []
            if start_date:
                filters.append(('timestamp', '>=', start_date))
            if end_date:
                filters.append(('timestamp', '<=', end_date))

            # Read with filters
            try:
                if filters:
                    table = pq.read_table(file, filters=filters)
                else:
                    table = pq.read_table(file)

                # Convert to dict records
                df = table.to_pandas()

                # Additional filtering (run_ids)
                if run_ids:
                    df = df[df['run_id'].isin(run_ids)]

                results.extend(df.to_dict('records'))
            except Exception as e:
                print(f"Warning: Failed to read {file}: {e}")
                continue

        return results

    def get_aggregate_stats(
        self,
        stage_name: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """
        Calculate aggregate statistics for a stage.

        T041-T042: Aggregate stats with Pandas

        Args:
            stage_name: Stage to analyze
            start_date: Filter >= this date (None = all time)
            end_date: Filter <= this date (None = all time)

        Returns:
            Dictionary with statistical metrics
        """
        metrics = self.query_metrics(
            stage_name=stage_name,
            start_date=start_date,
            end_date=end_date,
        )

        if not metrics:
            return {
                'count': 0,
                'median': 0.0,
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'p50': 0.0,
                'p95': 0.0,
                'p99': 0.0,
            }

        # T042: Use Pandas for aggregate statistics
        df = pd.DataFrame(metrics)
        exec_times = df['execution_time_ms']

        return {
            'count': len(exec_times),
            'median': float(exec_times.median()),
            'mean': float(exec_times.mean()),
            'std': float(exec_times.std()),
            'min': float(exec_times.min()),
            'max': float(exec_times.max()),
            'p50': float(exec_times.quantile(0.50)),
            'p95': float(exec_times.quantile(0.95)),
            'p99': float(exec_times.quantile(0.99)),
        }

    def _cleanup_old_metrics(self) -> int:
        """
        Clean up metrics older than retention_days.

        T154-T155: Retention and cleanup implementation

        Returns:
            Number of files deleted
        """
        if self.retention_days <= 0:
            # Retention disabled
            return 0

        deleted_count = 0
        cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)

        # Scan all parquet files
        for parquet_file in self.output_dir.glob("*.parquet"):
            try:
                # Read file metadata to check oldest timestamp
                metadata = pq.read_metadata(parquet_file)
                table = pq.read_table(parquet_file, columns=['timestamp'])
                df = table.to_pandas()

                # Check if all records are older than cutoff
                if df['timestamp'].max() < cutoff_date:
                    # All records are expired, delete the entire file
                    parquet_file.unlink()
                    deleted_count += 1
                elif df['timestamp'].min() < cutoff_date:
                    # Some records are expired, filter and rewrite
                    full_table = pq.read_table(parquet_file)
                    full_df = full_table.to_pandas()
                    filtered_df = full_df[full_df['timestamp'] >= cutoff_date]

                    if len(filtered_df) > 0:
                        # Rewrite with only recent records
                        filtered_table = pa.Table.from_pandas(filtered_df)
                        pq.write_table(filtered_table, parquet_file, compression='snappy')
                    else:
                        # No records left after filtering
                        parquet_file.unlink()
                        deleted_count += 1

            except Exception as e:
                print(f"Warning: Failed to process {parquet_file} for cleanup: {e}")
                continue

        return deleted_count

    def _get_disk_usage(self) -> Dict[str, float]:
        """
        Get disk usage for metrics directory.

        T157: Disk space monitoring

        Returns:
            Dictionary with usage statistics:
            {
                'total_mb': float,
                'used_mb': float,
                'free_mb': float,
                'usage_percent': float,
            }
        """
        try:
            stat = shutil.disk_usage(self.output_dir)
            return {
                'total_mb': stat.total / (1024 * 1024),
                'used_mb': stat.used / (1024 * 1024),
                'free_mb': stat.free / (1024 * 1024),
                'usage_percent': (stat.used / stat.total) * 100,
            }
        except Exception as e:
            print(f"Warning: Failed to get disk usage: {e}")
            return {
                'total_mb': 0.0,
                'used_mb': 0.0,
                'free_mb': 0.0,
                'usage_percent': 0.0,
            }

    def _trigger_cleanup_periodically(self) -> None:
        """
        Trigger cleanup every N writes.

        T156: Automatic cleanup trigger

        Checks a counter file and runs cleanup every 100 writes.
        """
        counter_file = self.output_dir / ".write_counter"

        # Read current count
        if counter_file.exists():
            with open(counter_file, 'r') as f:
                count = int(f.read().strip())
        else:
            count = 0

        count += 1

        # Write updated count
        with open(counter_file, 'w') as f:
            f.write(str(count))

        # Trigger cleanup every 100 writes
        if count % 100 == 0:
            # T158: Check disk usage and warn
            disk_usage = self._get_disk_usage()
            if disk_usage['usage_percent'] > 90:
                print(
                    f"Warning: Disk usage is {disk_usage['usage_percent']:.1f}% "
                    f"({disk_usage['used_mb']:.0f} MB used of {disk_usage['total_mb']:.0f} MB)"
                )

            # Run cleanup
            deleted = self._cleanup_old_metrics()
            if deleted > 0:
                print(f"Cleaned up {deleted} expired metrics files")
