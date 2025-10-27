"""Error case browser for interactive exploration.

This module provides functionality for browsing, filtering, and exporting
error cases from validation runs.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import json

from .patterns import ErrorCase, ErrorPattern, ErrorType, ErrorSeverity
from .visualizer import ErrorVisualizer


class ErrorBrowser:
    """Browse and filter error cases."""

    def __init__(self, run_id: str, error_dir: Optional[Path] = None):
        """Initialize error browser.

        Args:
            run_id: Pipeline run identifier
            error_dir: Directory containing error artifacts
        """
        self.run_id = run_id
        self.error_dir = error_dir or Path("output/errors")
        self.visualizer = ErrorVisualizer()
        self._errors_cache: Optional[List[ErrorCase]] = None
        self._patterns_cache: Optional[List[ErrorPattern]] = None

    def load_errors(
        self,
        filters: Optional[Dict[str, Any]] = None,
        use_index: bool = True
    ) -> List[ErrorCase]:
        """Load errors with indexed filtering (T052).

        Args:
            filters: Filter criteria (stage_name, error_type, severity, min_confidence, etc.)
            use_index: Use indexed filtering for performance

        Returns:
            List of ErrorCase instances matching filters
        """
        # Load from JSON file
        error_file = self.error_dir / f"{self.run_id}_errors.json"

        if not error_file.exists():
            return []

        # Load all errors
        if self._errors_cache is None:
            with open(error_file, 'r') as f:
                error_data = json.load(f)

            # Reconstruct ErrorCase objects
            errors = []
            for data in error_data:
                # Convert string paths back to Path objects
                saved_artifacts = {
                    k: Path(v) for k, v in data.get('saved_artifacts', {}).items()
                }

                error = ErrorCase(
                    error_id=data['error_id'],
                    run_id=data['run_id'],
                    stage_name=data['stage_name'],
                    sample_id=data['sample_id'],
                    error_type=ErrorType(data['error_type']),
                    severity=ErrorSeverity(data['severity']),
                    prediction=data['prediction'],
                    ground_truth=data['ground_truth'],
                    iou=data.get('iou'),
                    confidence=data.get('confidence'),
                    saved_artifacts=saved_artifacts,
                    timestamp=data['timestamp'],
                    metadata=data.get('metadata', {})
                )
                errors.append(error)

            self._errors_cache = errors
        else:
            errors = self._errors_cache

        # Apply filters
        if filters:
            filtered = errors

            if 'stage_name' in filters:
                filtered = [e for e in filtered if e.stage_name == filters['stage_name']]

            if 'error_type' in filters:
                error_type = filters['error_type']
                if isinstance(error_type, str):
                    error_type = ErrorType(error_type)
                filtered = [e for e in filtered if e.error_type == error_type]

            if 'severity' in filters:
                severity = filters['severity']
                if isinstance(severity, str):
                    severity = ErrorSeverity(severity)
                filtered = [e for e in filtered if e.severity == severity]

            if 'min_confidence' in filters:
                min_conf = filters['min_confidence']
                filtered = [e for e in filtered if e.confidence and e.confidence >= min_conf]

            if 'max_iou' in filters:
                max_iou = filters['max_iou']
                filtered = [e for e in filtered if e.iou is not None and e.iou <= max_iou]

            return filtered

        return errors

    def get_error_by_id(self, error_id: str) -> Optional[ErrorCase]:
        """Get specific error by ID (T053).

        Args:
            error_id: Error identifier

        Returns:
            ErrorCase if found, None otherwise
        """
        errors = self.load_errors()

        for error in errors:
            if error.error_id == error_id:
                return error

        return None

    def get_patterns(self) -> List[ErrorPattern]:
        """Load error patterns for this run.

        Returns:
            List of ErrorPattern instances
        """
        if self._patterns_cache is not None:
            return self._patterns_cache

        pattern_file = self.error_dir / f"{self.run_id}_patterns.json"

        if not pattern_file.exists():
            return []

        with open(pattern_file, 'r') as f:
            pattern_data = json.load(f)

        # Reconstruct ErrorPattern objects
        patterns = []
        for data in pattern_data:
            from .patterns import PatternStatistics

            stats = PatternStatistics(
                avg_iou=data['statistics']['avg_iou'],
                avg_confidence=data['statistics']['avg_confidence'],
                min_iou=data['statistics']['min_iou'],
                max_iou=data['statistics']['max_iou'],
                sample_preview=data['statistics']['sample_preview']
            )

            pattern = ErrorPattern(
                pattern_id=data['pattern_id'],
                run_id=data['run_id'],
                stage_name=data['stage_name'],
                error_type=ErrorType(data['error_type']),
                severity=ErrorSeverity(data['severity']),
                error_count=data['error_count'],
                affected_samples=data['affected_samples'],
                statistics=stats,
                suggested_cause=data.get('suggested_cause', ''),
                timestamp=data['timestamp']
            )
            patterns.append(pattern)

        self._patterns_cache = patterns
        return patterns

    def visualize_error(
        self,
        error: ErrorCase,
        show_diff: bool = True,
        output_path: Optional[Path] = None
    ) -> Path:
        """Visualize specific error case.

        Args:
            error: Error case to visualize
            show_diff: Show diff visualization
            output_path: Output path for visualization

        Returns:
            Path to saved visualization
        """
        if show_diff:
            return self.visualizer.create_diff_visualization(error, output_path=output_path)
        else:
            return self.visualizer.create_side_by_side_visualization(error, output_path)

    def create_streamlit_ui(self) -> None:
        """Create Streamlit UI component for error browser (T055).

        This creates an interactive UI for browsing errors with:
        - Filters (stage, error type, severity)
        - Pagination
        - Thumbnails with lazy loading
        - Detail view with visualizations
        """
        try:
            import streamlit as st
        except ImportError:
            raise ImportError("Streamlit required for UI. Install with: pip install streamlit")

        st.title(f"Error Browser - Run {self.run_id}")

        # Sidebar filters
        st.sidebar.header("Filters")

        # Load errors to get unique values
        all_errors = self.load_errors()

        if not all_errors:
            st.warning("No errors found for this run.")
            return

        # Filter controls
        stages = sorted(set(e.stage_name for e in all_errors))
        selected_stage = st.sidebar.selectbox("Stage", ["All"] + stages)

        error_types = [et.value for et in ErrorType]
        selected_error_type = st.sidebar.selectbox("Error Type", ["All"] + error_types)

        severities = [s.value for s in ErrorSeverity]
        selected_severity = st.sidebar.selectbox("Severity", ["All"] + severities)

        # Build filters
        filters = {}
        if selected_stage != "All":
            filters['stage_name'] = selected_stage
        if selected_error_type != "All":
            filters['error_type'] = selected_error_type
        if selected_severity != "All":
            filters['severity'] = selected_severity

        # Load filtered errors
        filtered_errors = self.load_errors(filters=filters)

        st.write(f"**{len(filtered_errors)}** errors found")

        # Pagination (T056)
        page_size = 20
        total_pages = (len(filtered_errors) + page_size - 1) // page_size

        page = st.sidebar.number_input("Page", min_value=1, max_value=max(1, total_pages), value=1)
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, len(filtered_errors))

        page_errors = filtered_errors[start_idx:end_idx]

        # Display error patterns
        st.header("Error Patterns")
        patterns = self.get_patterns()

        if patterns:
            for pattern in patterns[:5]:  # Show top 5
                with st.expander(f"{pattern.pattern_id} ({pattern.error_count} errors)"):
                    st.write(f"**Severity**: {pattern.severity.value}")
                    st.write(f"**Suggested Cause**: {pattern.suggested_cause}")
                    st.write(f"**Avg IoU**: {pattern.statistics.avg_iou:.3f}")
                    st.write(f"**Avg Confidence**: {pattern.statistics.avg_confidence:.3f}")

        # Display errors
        st.header(f"Error Cases (Page {page}/{total_pages})")

        for error in page_errors:
            with st.expander(f"{error.sample_id} - {error.error_type.value}"):
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**Error ID**: {error.error_id[:8]}")
                    st.write(f"**Stage**: {error.stage_name}")
                    st.write(f"**Severity**: {error.severity.value}")

                    if error.iou is not None:
                        st.write(f"**IoU**: {error.iou:.3f}")
                    if error.confidence is not None:
                        st.write(f"**Confidence**: {error.confidence:.3f}")

                with col2:
                    # Visualize error (lazy loading)
                    if st.button(f"Show Visualization", key=error.error_id):
                        vis_path = self.visualize_error(error, show_diff=True)
                        st.image(str(vis_path))

        # Export functionality
        if st.sidebar.button("Export Filtered Errors"):
            from .export import ErrorExporter
            exporter = ErrorExporter()
            export_path = exporter.export_errors(
                filtered_errors,
                output_path=self.error_dir / f"{self.run_id}_export.json",
                format="json"
            )
            st.sidebar.success(f"Exported to {export_path}")
