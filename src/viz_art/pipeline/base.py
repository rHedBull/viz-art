"""Core pipeline implementation.

This module provides the Pipeline class that orchestrates stage execution.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import uuid4

from viz_art.pipeline.stage import PipelineStage
from viz_art.pipeline.connection import StageConnection
from viz_art.pipeline.results import PipelineRun, StageResult, StageStatus, RunStatus


class Pipeline:
    """Pipeline orchestrator for multi-stage processing.

    The Pipeline class manages the execution of connected stages in sequence.
    It handles data flow between stages, validates connections, and tracks
    execution results.

    Example:
        >>> pipeline = Pipeline(name="my-pipeline")
        >>> pipeline.add_stage(ImageLoader(name="loader"))
        >>> pipeline.add_stage(SimpleFilter(name="filter"))
        >>> pipeline.connect("loader", "filter", "image", "image")
        >>> results = pipeline.run(image_path="test.jpg")
        >>> print(results["filter"]["filtered_image"])

    Attributes:
        name: Unique identifier for this pipeline
        stages: Ordered list of pipeline stages
        connections: List of stage connections defining data flow
    """

    def __init__(self, name: str, config: Optional[Any] = None):
        """Initialize pipeline.

        Args:
            name: Unique pipeline identifier
            config: Optional configuration object (for future use)

        Raises:
            ValueError: If name is empty or invalid
        """
        if not name or not name.strip():
            raise ValueError("Pipeline name cannot be empty")

        self.name = name.strip()
        self.config = config
        self._stages: List[PipelineStage] = []
        self._stage_map: Dict[str, PipelineStage] = {}
        self._connections: List[StageConnection] = []

    @property
    def stages(self) -> List[PipelineStage]:
        """Return list of registered stages."""
        return self._stages.copy()

    @property
    def connections(self) -> List[StageConnection]:
        """Return list of stage connections."""
        return self._connections.copy()

    def add_stage(self, stage: PipelineStage) -> None:
        """Add a stage to the pipeline.

        Stages are executed in the order they are added.

        Args:
            stage: PipelineStage instance to add

        Raises:
            ValueError: If stage name conflicts with existing stage
            TypeError: If stage doesn't implement PipelineStage protocol
        """
        # Validate stage has required protocol methods
        if not hasattr(stage, "name"):
            raise TypeError("Stage must have 'name' property")
        if not hasattr(stage, "input_keys"):
            raise TypeError("Stage must have 'input_keys' property")
        if not hasattr(stage, "output_keys"):
            raise TypeError("Stage must have 'output_keys' property")
        if not hasattr(stage, "pre_process"):
            raise TypeError("Stage must have 'pre_process' method")
        if not hasattr(stage, "predict"):
            raise TypeError("Stage must have 'predict' method")
        if not hasattr(stage, "post_process"):
            raise TypeError("Stage must have 'post_process' method")

        stage_name = stage.name
        if not stage_name:
            raise ValueError("Stage name cannot be empty")

        if stage_name in self._stage_map:
            raise ValueError(
                f"Stage with name '{stage_name}' already exists in pipeline"
            )

        self._stages.append(stage)
        self._stage_map[stage_name] = stage

    def connect(
        self,
        source_stage: str,
        target_stage: str,
        output_key: str,
        input_key: str,
        required: bool = True,
    ) -> None:
        """Connect output from source stage to input of target stage.

        Args:
            source_stage: Name of stage producing output
            target_stage: Name of stage consuming input
            output_key: Key in source stage outputs
            input_key: Key in target stage inputs
            required: Whether connection must succeed (default True)

        Raises:
            ValueError: If stages don't exist or keys invalid
            ValueError: If connection creates circular dependency
        """
        # Validate stages exist
        if source_stage not in self._stage_map:
            raise ValueError(f"Source stage '{source_stage}' not found in pipeline")
        if target_stage not in self._stage_map:
            raise ValueError(f"Target stage '{target_stage}' not found in pipeline")

        source = self._stage_map[source_stage]
        target = self._stage_map[target_stage]

        # Validate output key exists in source
        if output_key not in source.output_keys:
            raise ValueError(
                f"Output key '{output_key}' not in stage '{source_stage}' output_keys: "
                f"{source.output_keys}"
            )

        # Validate input key exists in target
        if input_key not in target.input_keys:
            raise ValueError(
                f"Input key '{input_key}' not in stage '{target_stage}' input_keys: "
                f"{target.input_keys}"
            )

        # Create connection
        connection = StageConnection(
            source_stage=source_stage,
            target_stage=target_stage,
            output_key=output_key,
            input_key=input_key,
            required=required,
        )

        # Check for circular dependencies after adding this connection
        self._connections.append(connection)
        if self._has_circular_dependency():
            self._connections.pop()  # Remove the problematic connection
            raise ValueError(
                f"Connection from '{source_stage}' to '{target_stage}' creates circular dependency"
            )

    def _has_circular_dependency(self) -> bool:
        """Check if connections create a circular dependency.

        Uses depth-first search to detect cycles in the connection graph.

        Returns:
            bool: True if circular dependency detected, False otherwise
        """
        # Build adjacency list from connections
        graph: Dict[str, List[str]] = {name: [] for name in self._stage_map.keys()}
        for conn in self._connections:
            if conn.source_stage not in graph:
                graph[conn.source_stage] = []
            graph[conn.source_stage].append(conn.target_stage)

        # DFS to detect cycles
        visited = set()
        rec_stack = set()

        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node in graph:
            if node not in visited:
                if has_cycle(node):
                    return True

        return False

    def run(self, **inputs) -> Dict[str, Any]:
        """Execute the pipeline with provided inputs.

        This method executes all stages in sequence, passing outputs from
        one stage as inputs to the next based on configured connections.

        Args:
            **inputs: Keyword arguments providing initial data for first stage

        Returns:
            Dictionary containing:
                - All stage outputs: {stage_name: {output_key: value}}
                - "_run_id": Unique run identifier
                - "_status": Overall run status ("COMPLETED" or "FAILED")
                - "_error": Error message if failed (optional)

        Raises:
            ValueError: If required inputs missing
            RuntimeError: If pipeline execution fails
        """
        run_id = str(uuid4())
        started_at = datetime.utcnow()
        stage_results: List[StageResult] = []
        all_outputs: Dict[str, Dict[str, Any]] = {}

        # Track available data (starts with pipeline inputs)
        available_data = inputs.copy()

        try:
            # Execute stages in order
            for stage in self._stages:
                stage_start = datetime.utcnow()

                try:
                    # Gather inputs for this stage from connections
                    stage_inputs = self._gather_stage_inputs(
                        stage, available_data, all_outputs
                    )

                    # Execute stage
                    preprocessed = stage.pre_process(stage_inputs)
                    predictions = stage.predict(preprocessed)
                    outputs = stage.post_process(predictions)

                    # Calculate duration
                    stage_end = datetime.utcnow()
                    duration_ms = (stage_end - stage_start).total_seconds() * 1000

                    # Store results
                    stage_result = StageResult(
                        stage_name=stage.name,
                        status=StageStatus.SUCCESS,
                        started_at=stage_start,
                        duration_ms=duration_ms,
                        outputs=outputs,
                    )
                    stage_results.append(stage_result)
                    all_outputs[stage.name] = outputs

                    # Make outputs available for next stages
                    available_data.update(outputs)

                except Exception as e:
                    # Stage failed
                    stage_end = datetime.utcnow()
                    duration_ms = (stage_end - stage_start).total_seconds() * 1000

                    stage_result = StageResult(
                        stage_name=stage.name,
                        status=StageStatus.FAILED,
                        started_at=stage_start,
                        duration_ms=duration_ms,
                        outputs={},
                        error=str(e),
                    )
                    stage_results.append(stage_result)

                    # Pipeline fails if any stage fails
                    completed_at = datetime.utcnow()
                    return {
                        "_run_id": run_id,
                        "_status": RunStatus.FAILED.value,
                        "_error": f"Stage '{stage.name}' failed: {str(e)}",
                        "_pipeline_name": self.name,
                        "_started_at": started_at.isoformat(),
                        "_completed_at": completed_at.isoformat(),
                        **all_outputs,
                    }

            # Pipeline completed successfully
            completed_at = datetime.utcnow()
            return {
                "_run_id": run_id,
                "_status": RunStatus.COMPLETED.value,
                "_pipeline_name": self.name,
                "_started_at": started_at.isoformat(),
                "_completed_at": completed_at.isoformat(),
                **all_outputs,
            }

        except Exception as e:
            # Unexpected error during pipeline execution
            completed_at = datetime.utcnow()
            return {
                "_run_id": run_id,
                "_status": RunStatus.FAILED.value,
                "_error": f"Pipeline execution failed: {str(e)}",
                "_pipeline_name": self.name,
                "_started_at": started_at.isoformat(),
                "_completed_at": completed_at.isoformat(),
                **all_outputs,
            }

    def _gather_stage_inputs(
        self,
        stage: PipelineStage,
        available_data: Dict[str, Any],
        all_outputs: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Gather inputs for a stage from connections and available data.

        Args:
            stage: Stage to gather inputs for
            available_data: Data available from pipeline inputs
            all_outputs: Outputs from previously executed stages

        Returns:
            Dictionary of inputs for the stage

        Raises:
            ValueError: If required inputs are missing
        """
        stage_inputs: Dict[str, Any] = {}

        # First, apply connections
        for conn in self._connections:
            if conn.target_stage == stage.name:
                # Find output from source stage
                if conn.source_stage in all_outputs:
                    source_outputs = all_outputs[conn.source_stage]
                    if conn.output_key in source_outputs:
                        stage_inputs[conn.input_key] = source_outputs[conn.output_key]
                    elif conn.required:
                        raise ValueError(
                            f"Required output '{conn.output_key}' not found in "
                            f"stage '{conn.source_stage}' outputs"
                        )

        # Then, fill missing inputs from available data
        for input_key in stage.input_keys:
            if input_key not in stage_inputs:
                if input_key in available_data:
                    stage_inputs[input_key] = available_data[input_key]
                else:
                    raise ValueError(
                        f"Required input '{input_key}' not found for stage '{stage.name}'"
                    )

        return stage_inputs

    @classmethod
    def from_config(cls, config: Any, stage_registry: Optional[Dict[str, type]] = None) -> "Pipeline":
        """Construct pipeline from configuration object.

        Args:
            config: PipelineConfig loaded from YAML
            stage_registry: Optional dict mapping stage_type names to classes
                          If None, attempts to import from examples.stages

        Returns:
            Configured Pipeline instance with stages and connections

        Raises:
            ValueError: If config invalid or stages can't be instantiated
            ImportError: If stage class cannot be imported

        Example:
            >>> from viz_art.config.loader import load_config
            >>> config = load_config("config.yaml")
            >>> pipeline = Pipeline.from_config(config)
        """
        from viz_art.config.schema import PipelineConfig

        # Ensure config is PipelineConfig
        if not isinstance(config, PipelineConfig):
            raise ValueError("config must be a PipelineConfig instance")

        # Create pipeline
        pipeline = cls(name=config.pipeline_name, config=config)

        # Build stage registry if not provided
        if stage_registry is None:
            stage_registry = _build_default_stage_registry()

        # Instantiate and add stages
        for stage_config in config.stages:
            if not stage_config.enabled:
                continue  # Skip disabled stages

            stage_type = stage_config.stage_type
            if stage_type not in stage_registry:
                raise ValueError(
                    f"Stage type '{stage_type}' not found in registry. "
                    f"Available: {list(stage_registry.keys())}"
                )

            # Get stage class
            stage_class = stage_registry[stage_type]

            # Instantiate stage with config
            try:
                # Pass name and **config as kwargs
                stage_instance = stage_class(
                    name=stage_config.name, **stage_config.config
                )
                pipeline.add_stage(stage_instance)
            except Exception as e:
                raise ValueError(
                    f"Failed to instantiate stage '{stage_config.name}' "
                    f"of type '{stage_type}': {str(e)}"
                )

        # Add connections
        for conn_config in config.connections:
            pipeline.connect(
                source_stage=conn_config.source,
                target_stage=conn_config.target,
                output_key=conn_config.output_key,
                input_key=conn_config.input_key,
                required=conn_config.required,
            )

        return pipeline


def _build_default_stage_registry() -> Dict[str, type]:
    """Build default stage registry from examples.stages.

    Returns:
        Dictionary mapping stage type names to stage classes

    Raises:
        ImportError: If example stages cannot be imported
    """
    registry = {}

    try:
        # Import example stages
        from examples.stages.image_loader import ImageLoader
        from examples.stages.simple_filter import SimpleFilter
        from examples.stages.image_resizer import ImageResizer

        registry["ImageLoader"] = ImageLoader
        registry["SimpleFilter"] = SimpleFilter
        registry["ImageResizer"] = ImageResizer

    except ImportError as e:
        # If examples not available, return empty registry
        # Users must provide their own stage_registry
        pass

    return registry
