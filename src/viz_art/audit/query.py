"""
Audit log query interface with fluent builder pattern.

T076-T088: AuditQuery implementation
"""

from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import json

from ..types import LogLevel


class AuditQuery:
    """
    Fluent interface for querying audit logs.

    Uses builder pattern for composable filters.
    T076-T088: Complete query interface
    """

    def __init__(self, log_dir: Path = Path("output/logs")):
        """
        Initialize audit query builder.

        T076: Constructor with fluent builder pattern
        """
        self.log_dir = Path(log_dir)
        self._run_id: Optional[str] = None
        self._stage_name: Optional[str] = None
        self._level: Optional[LogLevel] = None
        self._after: Optional[datetime] = None
        self._before: Optional[datetime] = None
        self._failed_only: bool = False
        self._limit: Optional[int] = None

    def run_id(self, run_id: str) -> "AuditQuery":
        """
        Filter by run ID.

        T077: Run ID filter
        """
        self._run_id = run_id
        return self

    def stage(self, stage_name: str) -> "AuditQuery":
        """
        Filter by stage name.

        T078: Stage name filter
        """
        self._stage_name = stage_name
        return self

    def level(self, level: LogLevel) -> "AuditQuery":
        """
        Filter by log level.

        T079: Log level filter
        """
        self._level = level
        return self

    def after(self, timestamp: datetime) -> "AuditQuery":
        """
        Filter logs >= timestamp.

        T080: After timestamp filter
        """
        self._after = timestamp
        return self

    def before(self, timestamp: datetime) -> "AuditQuery":
        """
        Filter logs <= timestamp.

        T081: Before timestamp filter
        """
        self._before = timestamp
        return self

    def failed(self) -> "AuditQuery":
        """
        Filter to ERROR and CRITICAL levels only.

        T082: Failed filter (ERROR + CRITICAL)
        """
        self._failed_only = True
        return self

    def limit(self, count: int) -> "AuditQuery":
        """
        Limit results to first N entries.

        T083: Limit filter
        """
        self._limit = count
        return self

    def fetch(self) -> List[Dict[str, Any]]:
        """
        Execute query and return results.

        T084-T086: Execute query with file scanning and filtering

        Returns:
            List of log entry dictionaries

        Example:
            logs = (
                AuditQuery()
                .after(datetime(2025, 10, 20))
                .stage("detection")
                .failed()
                .limit(10)
                .fetch()
            )
        """
        results = []

        # T085: Date-based file scanning (only load relevant .jsonl files)
        log_files = self._get_relevant_log_files()

        for log_file in log_files:
            if not log_file.exists():
                continue

            try:
                # T086: JSON Lines parsing with filter application
                with open(log_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            entry = json.loads(line)

                            # Apply filters
                            if not self._matches_filters(entry):
                                continue

                            results.append(entry)

                            # Apply limit
                            if self._limit and len(results) >= self._limit:
                                return results

                        except json.JSONDecodeError:
                            # Skip malformed lines
                            continue

            except Exception as e:
                print(f"Warning: Failed to read {log_file}: {e}")
                continue

        return results

    def export_json(self, output_path: Path) -> int:
        """
        Export filtered logs to JSON file.

        T087-T088: Export with result count

        Args:
            output_path: Where to write JSON

        Returns:
            Number of log entries exported

        Example:
            count = (
                AuditQuery()
                .after(datetime(2025, 10, 20))
                .export_json(Path("filtered_logs.json"))
            )
            print(f"Exported {count} log entries")
        """
        # T087: Write filtered results to file
        logs = self.fetch()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(logs, f, indent=2, default=str)

        # T088: Return result count
        return len(logs)

    def _get_relevant_log_files(self) -> List[Path]:
        """
        Get list of log files to scan based on date filters.

        T085: Date-based file scanning
        """
        if not self.log_dir.exists():
            return []

        # If we have date filters, only scan relevant files
        if self._after or self._before:
            # For now, scan all .jsonl files
            # TODO: Optimize by only scanning files within date range
            return sorted(self.log_dir.glob("*.jsonl"))
        else:
            # Scan all .jsonl files
            return sorted(self.log_dir.glob("*.jsonl"))

    def _matches_filters(self, entry: Dict[str, Any]) -> bool:
        """
        Check if log entry matches all filters.

        T086: Filter application logic
        """
        # Filter by run_id
        if self._run_id and entry.get('record', {}).get('extra', {}).get('run_id') != self._run_id:
            return False

        # Filter by stage_name
        if self._stage_name:
            stage = entry.get('record', {}).get('extra', {}).get('stage_name')
            if stage != self._stage_name:
                return False

        # Filter by level
        if self._level:
            level = entry.get('record', {}).get('level', {}).get('name')
            if level != self._level.value:
                return False

        # Filter by failed (ERROR or CRITICAL)
        if self._failed_only:
            level = entry.get('record', {}).get('level', {}).get('name')
            if level not in ['ERROR', 'CRITICAL']:
                return False

        # Filter by timestamp
        if self._after or self._before:
            timestamp_str = entry.get('record', {}).get('time')
            if timestamp_str:
                try:
                    # Parse ISO format timestamp
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

                    if self._after and timestamp < self._after:
                        return False
                    if self._before and timestamp > self._before:
                        return False
                except (ValueError, AttributeError):
                    # Skip entries with invalid timestamps
                    return False

        return True
