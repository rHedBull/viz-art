"""
Run tracking and lifecycle management.

T067-T075: RunTracker implementation
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from contextlib import contextmanager
import json
import time

from ..types import Run, RunStatus
from ..utils.identifiers import generate_run_id


class RunTracker:
    """
    Run ID generator and context manager.

    Manages run lifecycle: creates ID, tracks start/end, updates status.
    T067-T075: Complete run tracker implementation
    """

    def __init__(self, output_dir: Path):
        """
        Initialize run tracker.

        T067: Constructor

        Args:
            output_dir: Directory for run metadata (e.g., output/runs/)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def track(
        self,
        config_snapshot: Dict[str, Any],
        input_files: List[Path],
        output_dir: Path,
    ):
        """
        Context manager for tracking a pipeline run.

        T068-T073: Track context manager implementation

        Args:
            config_snapshot: Pipeline configuration at run time
            input_files: Input images/point clouds being processed
            output_dir: Where outputs will be saved

        Yields:
            run_id: str (UUID v4)

        Raises:
            Exception: Propagates any exception from pipeline, marks run as FAILED

        Example:
            tracker = RunTracker(Path("output/runs"))
            with tracker.track(config, inputs, output_dir) as run_id:
                logger = create_logger(run_id)
                logger.info("Starting pipeline")
                results = pipeline.run(inputs)
        """
        # T068: Generate run ID and create Run model
        run_id = generate_run_id()
        start_time = time.time()

        # T069: Create Run model instance with status=RUNNING
        run = Run(
            run_id=run_id,
            timestamp=datetime.utcnow(),
            config_snapshot=config_snapshot,
            input_files=input_files,
            status=RunStatus.RUNNING,
            output_dir=output_dir,
        )

        # T070: Save run metadata on context entry
        run_dir = self.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        metadata_file = run_dir / "run_metadata.json"

        with open(metadata_file, 'w') as f:
            f.write(run.model_dump_json(indent=2))

        try:
            yield run_id

            # T071: Update status to COMPLETED on success
            end_time = time.time()
            run.status = RunStatus.COMPLETED
            # T073: Calculate total_duration_ms
            run.total_duration_ms = (end_time - start_time) * 1000.0

            # Save updated metadata
            with open(metadata_file, 'w') as f:
                f.write(run.model_dump_json(indent=2))

        except Exception as e:
            # T072: Update status to FAILED on exception with error message
            end_time = time.time()
            run.status = RunStatus.FAILED
            run.error = str(e)
            run.total_duration_ms = (end_time - start_time) * 1000.0

            # Save updated metadata
            with open(metadata_file, 'w') as f:
                f.write(run.model_dump_json(indent=2))

            # Re-raise exception
            raise

    def get_run_metadata(self, run_id: str) -> Dict[str, Any]:
        """
        Retrieve metadata for a completed run.

        T074-T075: Get run metadata with error handling

        Args:
            run_id: Run to query

        Returns:
            Dictionary with run metadata

        Raises:
            KeyError: If run_id not found
            ValueError: If metadata file is corrupted
        """
        # T074: Load run_metadata.json by run_id
        metadata_file = self.output_dir / run_id / "run_metadata.json"

        if not metadata_file.exists():
            raise KeyError(f"Run not found: {run_id}")

        try:
            # T075: Error handling for missing/corrupted files
            with open(metadata_file, 'r') as f:
                data = json.load(f)
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Corrupted metadata file for run {run_id}: {e}")
        except Exception as e:
            raise ValueError(f"Failed to read metadata for run {run_id}: {e}")
