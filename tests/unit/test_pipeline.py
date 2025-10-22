"""Unit tests for Pipeline class."""

import pytest
from viz_art.pipeline.base import Pipeline
from viz_art.pipeline.results import RunStatus


class TestPipelineCreation:
    """Tests for pipeline initialization."""

    def test_pipeline_creation(self):
        """Test basic pipeline creation."""
        pipeline = Pipeline(name="test-pipeline")
        assert pipeline.name == "test-pipeline"
        assert len(pipeline.stages) == 0
        assert len(pipeline.connections) == 0

    def test_pipeline_empty_name_raises_error(self):
        """Test that empty pipeline name raises ValueError."""
        with pytest.raises(ValueError, match="Pipeline name cannot be empty"):
            Pipeline(name="")

    def test_pipeline_whitespace_name_raises_error(self):
        """Test that whitespace-only name raises ValueError."""
        with pytest.raises(ValueError, match="Pipeline name cannot be empty"):
            Pipeline(name="   ")


class TestPipelineAddStage:
    """Tests for adding stages to pipeline."""

    def test_add_stage_success(self, mock_stage):
        """Test successfully adding a stage."""
        pipeline = Pipeline(name="test")
        pipeline.add_stage(mock_stage)

        assert len(pipeline.stages) == 1
        assert pipeline.stages[0].name == "test_stage"

    def test_add_multiple_stages(self, mock_stage_a, mock_stage_b):
        """Test adding multiple stages in order."""
        pipeline = Pipeline(name="test")
        pipeline.add_stage(mock_stage_a)
        pipeline.add_stage(mock_stage_b)

        assert len(pipeline.stages) == 2
        assert pipeline.stages[0].name == "stage_a"
        assert pipeline.stages[1].name == "stage_b"

    def test_add_duplicate_stage_name_raises_error(self, mock_stage):
        """Test that duplicate stage names raise ValueError."""
        pipeline = Pipeline(name="test")
        pipeline.add_stage(mock_stage)

        # Try to add another stage with same name
        duplicate_stage = type(mock_stage)(
            name="test_stage", input_keys=["x"], output_keys=["y"]
        )

        with pytest.raises(ValueError, match="already exists in pipeline"):
            pipeline.add_stage(duplicate_stage)

    def test_add_invalid_stage_missing_name(self):
        """Test that stage without name property raises TypeError."""
        pipeline = Pipeline(name="test")

        class InvalidStage:
            pass

        with pytest.raises(TypeError, match="must have 'name' property"):
            pipeline.add_stage(InvalidStage())

    def test_add_invalid_stage_missing_methods(self):
        """Test that stage without required methods raises TypeError."""
        pipeline = Pipeline(name="test")

        class InvalidStage:
            name = "invalid"
            input_keys = []
            output_keys = []

        with pytest.raises(TypeError, match="must have 'pre_process' method"):
            pipeline.add_stage(InvalidStage())


class TestPipelineConnect:
    """Tests for connecting stages."""

    def test_connect_stages_success(self, mock_stage_a, mock_stage_b):
        """Test successfully connecting two stages."""
        pipeline = Pipeline(name="test")
        pipeline.add_stage(mock_stage_a)
        pipeline.add_stage(mock_stage_b)

        pipeline.connect("stage_a", "stage_b", "result_a", "result_a")

        assert len(pipeline.connections) == 1
        conn = pipeline.connections[0]
        assert conn.source_stage == "stage_a"
        assert conn.target_stage == "stage_b"

    def test_connect_nonexistent_source_raises_error(self, mock_stage_b):
        """Test connecting from non-existent stage raises ValueError."""
        pipeline = Pipeline(name="test")
        pipeline.add_stage(mock_stage_b)

        with pytest.raises(ValueError, match="Source stage 'unknown' not found"):
            pipeline.connect("unknown", "stage_b", "output", "input")

    def test_connect_nonexistent_target_raises_error(self, mock_stage_a):
        """Test connecting to non-existent stage raises ValueError."""
        pipeline = Pipeline(name="test")
        pipeline.add_stage(mock_stage_a)

        with pytest.raises(ValueError, match="Target stage 'unknown' not found"):
            pipeline.connect("stage_a", "unknown", "output", "input")

    def test_connect_invalid_output_key_raises_error(self, mock_stage_a, mock_stage_b):
        """Test connecting with invalid output key raises ValueError."""
        pipeline = Pipeline(name="test")
        pipeline.add_stage(mock_stage_a)
        pipeline.add_stage(mock_stage_b)

        with pytest.raises(ValueError, match="Output key 'invalid' not in stage"):
            pipeline.connect("stage_a", "stage_b", "invalid", "result_a")

    def test_connect_invalid_input_key_raises_error(self, mock_stage_a, mock_stage_b):
        """Test connecting with invalid input key raises ValueError."""
        pipeline = Pipeline(name="test")
        pipeline.add_stage(mock_stage_a)
        pipeline.add_stage(mock_stage_b)

        with pytest.raises(ValueError, match="Input key 'invalid' not in stage"):
            pipeline.connect("stage_a", "stage_b", "result_a", "invalid")


class TestPipelineCircularDependency:
    """Tests for circular dependency detection."""

    def test_circular_dependency_direct(self, mock_stage_a, mock_stage_b):
        """Test detection of direct circular dependency (A→B→A)."""
        # Create stages that can connect in circle
        stage_a = type(mock_stage_a)(
            name="stage_a", input_keys=["data", "from_b"], output_keys=["to_b"]
        )
        stage_b = type(mock_stage_b)(
            name="stage_b", input_keys=["to_b"], output_keys=["from_b"]
        )

        pipeline = Pipeline(name="test")
        pipeline.add_stage(stage_a)
        pipeline.add_stage(stage_b)

        # First connection is fine
        pipeline.connect("stage_a", "stage_b", "to_b", "to_b")

        # Second connection creates cycle
        with pytest.raises(ValueError, match="creates circular dependency"):
            pipeline.connect("stage_b", "stage_a", "from_b", "from_b")

    def test_circular_dependency_indirect(self, mock_stage_a, mock_stage_b, mock_stage_c):
        """Test detection of indirect circular dependency (A→B→C→A)."""
        # Create 3 stages with potential cycle
        stage_a = type(mock_stage_a)(
            name="stage_a", input_keys=["data", "from_c"], output_keys=["to_b"]
        )
        stage_b = type(mock_stage_b)(
            name="stage_b", input_keys=["to_b"], output_keys=["to_c"]
        )
        stage_c = type(mock_stage_c)(
            name="stage_c", input_keys=["to_c"], output_keys=["from_c"]
        )

        pipeline = Pipeline(name="test")
        pipeline.add_stage(stage_a)
        pipeline.add_stage(stage_b)
        pipeline.add_stage(stage_c)

        # Connect A→B→C
        pipeline.connect("stage_a", "stage_b", "to_b", "to_b")
        pipeline.connect("stage_b", "stage_c", "to_c", "to_c")

        # Connecting C→A creates cycle
        with pytest.raises(ValueError, match="creates circular dependency"):
            pipeline.connect("stage_c", "stage_a", "from_c", "from_c")


class TestPipelineExecution:
    """Tests for pipeline execution."""

    def test_pipeline_single_stage_execution(self, mock_stage):
        """Test executing pipeline with single stage."""
        pipeline = Pipeline(name="test")
        pipeline.add_stage(mock_stage)

        results = pipeline.run(input="test_data")

        assert results["_status"] == RunStatus.COMPLETED.value
        assert "test_stage" in results
        assert results["test_stage"]["output"] == "test_data"

    def test_pipeline_execution_order(self, mock_stage_a, mock_stage_b):
        """Test that stages execute in correct order."""
        pipeline = Pipeline(name="test")
        pipeline.add_stage(mock_stage_a)
        pipeline.add_stage(mock_stage_b)
        pipeline.connect("stage_a", "stage_b", "result_a", "result_a")

        results = pipeline.run(data="test_input")

        assert results["_status"] == RunStatus.COMPLETED.value
        assert "stage_a" in results
        assert "stage_b" in results
        # Verify data flowed through
        assert results["stage_a"]["result_a"] == "test_input"
        assert results["stage_b"]["result_b"] == "test_input"

    def test_pipeline_data_flow(self, mock_stage_a, mock_stage_b, mock_stage_c):
        """Test data flows correctly through 3 stages."""
        pipeline = Pipeline(name="test")
        pipeline.add_stage(mock_stage_a)
        pipeline.add_stage(mock_stage_b)
        pipeline.add_stage(mock_stage_c)

        pipeline.connect("stage_a", "stage_b", "result_a", "result_a")
        pipeline.connect("stage_b", "stage_c", "result_b", "result_b")

        results = pipeline.run(data="initial")

        assert results["_status"] == RunStatus.COMPLETED.value
        # Verify data flowed through all 3 stages
        assert results["stage_a"]["result_a"] == "initial"
        assert results["stage_b"]["result_b"] == "initial"
        assert results["stage_c"]["final"] == "initial"

    def test_pipeline_missing_input_raises_error(self, mock_stage):
        """Test that missing required input raises ValueError."""
        pipeline = Pipeline(name="test")
        pipeline.add_stage(mock_stage)

        results = pipeline.run()  # No input provided

        assert results["_status"] == RunStatus.FAILED.value
        assert "_error" in results
        assert "Required input 'input' not found" in results["_error"]

    def test_pipeline_stage_failure_stops_execution(self, mock_stage_a):
        """Test that stage failure stops execution."""
        from conftest import FailingStage

        # Create failing stage that accepts result from stage_a
        failing_stage = FailingStage(
            name="failing_stage",
            fail_at="predict",
            input_keys=["result_a"],
            output_keys=["output"]
        )

        pipeline = Pipeline(name="test")
        pipeline.add_stage(mock_stage_a)
        pipeline.add_stage(failing_stage)

        # Connect stages
        pipeline.connect("stage_a", "failing_stage", "result_a", "result_a")

        results = pipeline.run(data="test")

        assert results["_status"] == RunStatus.FAILED.value
        assert "_error" in results
        assert "failing_stage" in results["_error"]
        # First stage should have completed
        assert "stage_a" in results

    def test_pipeline_run_metadata(self, mock_stage):
        """Test that run metadata is included in results."""
        pipeline = Pipeline(name="metadata-test")
        pipeline.add_stage(mock_stage)

        results = pipeline.run(input="data")

        assert "_run_id" in results
        assert "_status" in results
        assert "_pipeline_name" in results
        assert results["_pipeline_name"] == "metadata-test"
        assert "_started_at" in results
        assert "_completed_at" in results
