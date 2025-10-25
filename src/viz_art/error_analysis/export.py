"""Export functionality for error cases.

This module provides functions for exporting error cases to various formats
(JSON, CSV, Parquet) for external analysis.
"""

from pathlib import Path
from typing import List
import json

from .patterns import ErrorCase


class ErrorExporter:
    """Export error cases to various formats."""

    def export_errors(
        self,
        errors: List[ErrorCase],
        output_path: Path,
        format: str = "json"
    ) -> Path:
        """Export errors to specified format (T054).

        Args:
            errors: List of error cases to export
            output_path: Output file path
            format: Export format ("json", "csv", "parquet")

        Returns:
            Path to exported file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            return self._export_json(errors, output_path)
        elif format == "csv":
            return self._export_csv(errors, output_path)
        elif format == "parquet":
            return self._export_parquet(errors, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_json(self, errors: List[ErrorCase], output_path: Path) -> Path:
        """Export to JSON format.

        Args:
            errors: Error cases
            output_path: Output path

        Returns:
            Path to exported file
        """
        # Convert errors to serializable format
        error_data = []
        for error in errors:
            data = {
                'error_id': error.error_id,
                'run_id': error.run_id,
                'stage_name': error.stage_name,
                'sample_id': error.sample_id,
                'error_type': error.error_type.value,
                'severity': error.severity.value,
                'prediction': str(error.prediction),  # Convert to string for JSON
                'ground_truth': str(error.ground_truth),
                'iou': error.iou,
                'confidence': error.confidence,
                'saved_artifacts': {k: str(v) for k, v in error.saved_artifacts.items()},
                'timestamp': error.timestamp.isoformat(),
                'metadata': error.metadata
            }
            error_data.append(data)

        with open(output_path, 'w') as f:
            json.dump(error_data, f, indent=2)

        return output_path

    def _export_csv(self, errors: List[ErrorCase], output_path: Path) -> Path:
        """Export to CSV format.

        Args:
            errors: Error cases
            output_path: Output path

        Returns:
            Path to exported file
        """
        import csv

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'error_id', 'run_id', 'stage_name', 'sample_id',
                'error_type', 'severity', 'iou', 'confidence', 'timestamp'
            ])

            # Rows
            for error in errors:
                writer.writerow([
                    error.error_id,
                    error.run_id,
                    error.stage_name,
                    error.sample_id,
                    error.error_type.value,
                    error.severity.value,
                    error.iou if error.iou is not None else '',
                    error.confidence if error.confidence is not None else '',
                    error.timestamp.isoformat()
                ])

        return output_path

    def _export_parquet(self, errors: List[ErrorCase], output_path: Path) -> Path:
        """Export to Parquet format.

        Args:
            errors: Error cases
            output_path: Output path

        Returns:
            Path to exported file
        """
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            # Fallback to JSON if PyArrow not available
            return self._export_json(errors, output_path.with_suffix('.json'))

        # Build data dict
        data = {
            'error_id': [e.error_id for e in errors],
            'run_id': [e.run_id for e in errors],
            'stage_name': [e.stage_name for e in errors],
            'sample_id': [e.sample_id for e in errors],
            'error_type': [e.error_type.value for e in errors],
            'severity': [e.severity.value for e in errors],
            'iou': [e.iou if e.iou is not None else -1.0 for e in errors],
            'confidence': [e.confidence if e.confidence is not None else -1.0 for e in errors],
            'timestamp': [e.timestamp for e in errors]
        }

        # Create PyArrow table
        table = pa.table(data)

        # Write to Parquet
        pq.write_table(table, str(output_path))

        return output_path
