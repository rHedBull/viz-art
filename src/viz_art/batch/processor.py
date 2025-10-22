"""Batch processing of images through a pipeline.

This module provides the BatchProcessor class for processing multiple images
from a directory through a configured pipeline, with error handling and
result aggregation.
"""

from pathlib import Path
from typing import Generator, List, Dict, Any
from datetime import datetime
from dataclasses import replace
import logging

from viz_art.pipeline.base import Pipeline
from viz_art.pipeline.results import PipelineRun, BatchResult, RunStatus
from viz_art.pipeline.output_saver import OutputSaver
from viz_art.config.schema import BatchConfigItem

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Process multiple images through a pipeline with error handling.

    The BatchProcessor discovers images in a directory, processes each through
    a pipeline, and aggregates results. It supports continue-on-error behavior
    to ensure one bad image doesn't stop the entire batch.

    Example:
        >>> from viz_art.config.loader import load_config
        >>> from viz_art.pipeline.base import Pipeline
        >>> config = load_config("config.yaml")
        >>> pipeline = Pipeline.from_config(config)
        >>> processor = BatchProcessor(pipeline, config.batch_config)
        >>> result = processor.run()
        >>> print(f"Processed {result.successful}/{result.total_files} images")

    Attributes:
        pipeline: Configured Pipeline instance to process images
        config: Batch processing configuration
    """

    # Common image file extensions
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    def __init__(self, pipeline: Pipeline, config: BatchConfigItem):
        """Initialize batch processor.

        Args:
            pipeline: Configured Pipeline instance
            config: Batch processing configuration

        Raises:
            ValueError: If pipeline or config is None
        """
        if pipeline is None:
            raise ValueError("pipeline cannot be None")
        if config is None:
            raise ValueError("config cannot be None")

        self.pipeline = pipeline
        self.config = config
        self._successful_runs: List[PipelineRun] = []
        self._failed_runs: List[PipelineRun] = []

        # Initialize OutputSaver
        self.output_saver = OutputSaver(
            output_mode=config.output_mode,
            save_config=config.save_outputs.model_dump(),
            output_dir=config.output_dir,
        )
        logger.debug(f"OutputSaver initialized in {config.output_mode} mode")

    def discover_images(self) -> Generator[Path, None, None]:
        """Discover image files in input directory.

        Uses the configured file patterns and recursive setting to find images.
        Silently skips non-image files (no logging for skipped files).

        Yields:
            Path objects for discovered image files

        Raises:
            ValueError: If input directory doesn't exist
        """
        input_path = Path(self.config.input_dir)

        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_path}")

        if not input_path.is_dir():
            raise ValueError(f"Input path is not a directory: {input_path}")

        logger.info(f"Discovering images in {input_path}")

        # Track unique files to avoid duplicates from multiple patterns
        discovered_files = set()

        for pattern in self.config.file_patterns:
            if self.config.recursive:
                # Use rglob for recursive search
                matches = input_path.rglob(pattern)
            else:
                # Use glob for single directory
                matches = input_path.glob(pattern)

            for file_path in matches:
                # Skip if not a file or already discovered
                if not file_path.is_file():
                    continue

                if file_path in discovered_files:
                    continue

                # Silently skip non-image files by extension check
                if file_path.suffix.lower() not in self.IMAGE_EXTENSIONS:
                    continue

                discovered_files.add(file_path)
                yield file_path

        logger.info(f"Discovered {len(discovered_files)} image files")

    def run(self) -> BatchResult:
        """Execute batch processing on all discovered images.

        Process flow:
            1. Discover images using configured patterns
            2. Process each image through pipeline
            3. Track successes and failures
            4. Continue on error if configured
            5. Generate HTML report (handled by caller)
            6. Return aggregated results

        Returns:
            BatchResult with statistics and per-image results

        Raises:
            ValueError: If input directory doesn't exist
            RuntimeError: If continue_on_error=False and processing fails
        """
        started_at = datetime.utcnow()
        batch_id = f"batch_{started_at.strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Starting batch processing: {batch_id}")

        # Discover images
        image_paths = list(self.discover_images())
        total_files = len(image_paths)

        if total_files == 0:
            logger.warning("No images found to process")
            # Return empty batch result
            return BatchResult(
                batch_id=batch_id,
                total_files=0,
                successful=0,
                failed=0,
                run_results=[],
                started_at=started_at,
                completed_at=datetime.utcnow(),
                report_path="",  # Will be set by reporter
            )

        logger.info(f"Processing {total_files} images...")

        # Process each image
        for idx, image_path in enumerate(image_paths, start=1):
            try:
                # Log progress
                logger.info(f"[{idx}/{total_files}] Processing {image_path.name}")

                # Run pipeline with image_path as input
                result = self.pipeline.run(image_path=str(image_path))

                # Save stage outputs
                saved_paths = {}
                for stage_name, stage_output in result.items():
                    if not stage_name.startswith('_'):  # Skip metadata keys
                        paths = self.output_saver.save_stage_output(
                            run_id=batch_id,
                            stage_name=stage_name,
                            filename=image_path.name,
                            outputs=stage_output,
                            image_index=idx - 1,
                            is_error=False,
                        )
                        if paths:
                            saved_paths[stage_name] = paths

                # Extract PipelineRun from result if available
                # (Pipeline.run() returns dict, need to construct PipelineRun)
                pipeline_run = self._create_pipeline_run(
                    image_path=image_path,
                    result=result,
                    status=RunStatus.COMPLETED,
                    error=None,
                )

                # Add saved paths to pipeline_run (store as dict in outputs metadata)
                if saved_paths:
                    pipeline_run = replace(
                        pipeline_run,
                        outputs={**pipeline_run.outputs, '_saved_paths': saved_paths}
                    )

                self._successful_runs.append(pipeline_run)

            except Exception as e:
                # Log error
                logger.error(f"[{idx}/{total_files}] Failed to process {image_path.name}: {e}")

                # Create failed run record
                pipeline_run = self._create_pipeline_run(
                    image_path=image_path,
                    result={},
                    status=RunStatus.FAILED,
                    error=str(e),
                )

                self._failed_runs.append(pipeline_run)

                # Re-raise if not continuing on error
                if not self.config.continue_on_error:
                    logger.error("Stopping batch processing due to error")
                    raise RuntimeError(
                        f"Batch processing stopped at {image_path.name}: {e}"
                    ) from e

        # Create output directory if it doesn't exist
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        completed_at = datetime.utcnow()
        duration = (completed_at - started_at).total_seconds()

        # Log summary
        logger.info(f"Batch processing complete in {duration:.2f}s")
        logger.info(
            f"Results: {len(self._successful_runs)} successful, "
            f"{len(self._failed_runs)} failed"
        )

        # Aggregate all runs
        all_runs = self._successful_runs + self._failed_runs

        # Create batch result
        return BatchResult(
            batch_id=batch_id,
            total_files=total_files,
            successful=len(self._successful_runs),
            failed=len(self._failed_runs),
            run_results=all_runs,
            started_at=started_at,
            completed_at=completed_at,
            report_path="",  # Will be set by reporter after HTML generation
        )

    def _create_pipeline_run(
        self,
        image_path: Path,
        result: dict,
        status: RunStatus,
        error: str | None,
    ) -> PipelineRun:
        """Create a PipelineRun object from pipeline execution result.

        Args:
            image_path: Path to processed image
            result: Result dictionary from Pipeline.run()
            status: Run status (COMPLETED or FAILED)
            error: Error message if failed

        Returns:
            PipelineRun instance
        """
        from datetime import datetime as dt

        # Extract metadata from result
        run_id = result.get("_run_id", "")

        # Parse ISO format timestamps if present
        started_at_str = result.get("_started_at")
        completed_at_str = result.get("_completed_at")

        started_at = (
            dt.fromisoformat(started_at_str) if started_at_str else datetime.utcnow()
        )
        completed_at = (
            dt.fromisoformat(completed_at_str) if completed_at_str else datetime.utcnow()
        )

        # Stage results are not included in dict return, would need separate tracking
        stage_results = result.get("_stage_results", [])

        # Extract stage outputs (filter out metadata keys)
        outputs = {
            key: value
            for key, value in result.items()
            if not key.startswith("_")
        }

        return PipelineRun(
            run_id=run_id,
            pipeline_name=self.pipeline.name,
            started_at=started_at,
            completed_at=completed_at,
            status=status,
            inputs={"image_path": str(image_path)},
            outputs=outputs,
            stage_results=stage_results,
            error=error,
        )
