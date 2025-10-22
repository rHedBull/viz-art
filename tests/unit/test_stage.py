"""Unit tests for PipelineStage abstract base class."""

import pytest
from typing import Dict, Any, List


class TestStageProtocolCompliance:
    """Tests for PipelineStage ABC compliance and behavior."""

    def test_stage_has_required_properties(self, mock_stage):
        """Test that stage implements required properties."""
        assert hasattr(mock_stage, "name")
        assert hasattr(mock_stage, "input_keys")
        assert hasattr(mock_stage, "output_keys")

    def test_stage_has_required_methods(self, mock_stage):
        """Test that stage implements required methods."""
        assert hasattr(mock_stage, "pre_process")
        assert hasattr(mock_stage, "predict")
        assert hasattr(mock_stage, "post_process")
        assert callable(mock_stage.pre_process)
        assert callable(mock_stage.predict)
        assert callable(mock_stage.post_process)

    def test_stage_name_property(self, mock_stage):
        """Test that name property returns string."""
        name = mock_stage.name
        assert isinstance(name, str)
        assert len(name) > 0

    def test_stage_input_keys_property(self, mock_stage):
        """Test that input_keys property returns list."""
        input_keys = mock_stage.input_keys
        assert isinstance(input_keys, list)
        assert all(isinstance(key, str) for key in input_keys)

    def test_stage_output_keys_property(self, mock_stage):
        """Test that output_keys property returns list."""
        output_keys = mock_stage.output_keys
        assert isinstance(output_keys, list)
        assert all(isinstance(key, str) for key in output_keys)

    def test_stage_pre_process_signature(self, mock_stage):
        """Test pre_process accepts dict and returns dict."""
        inputs = {"input": "test"}
        result = mock_stage.pre_process(inputs)
        assert isinstance(result, dict)

    def test_stage_predict_signature(self, mock_stage):
        """Test predict accepts dict and returns dict."""
        preprocessed = {"input": "test"}
        result = mock_stage.predict(preprocessed)
        assert isinstance(result, dict)

    def test_stage_post_process_signature(self, mock_stage):
        """Test post_process accepts dict and returns dict."""
        predictions = {"processed_input": "test"}
        result = mock_stage.post_process(predictions)
        assert isinstance(result, dict)

    def test_stage_full_execution_flow(self, mock_stage):
        """Test complete stage execution flow."""
        # Execute full flow
        inputs = {"input": "test_data"}
        preprocessed = mock_stage.pre_process(inputs)
        predictions = mock_stage.predict(preprocessed)
        outputs = mock_stage.post_process(predictions)

        # Verify outputs match declared output_keys
        assert "output" in outputs
        assert outputs["output"] == "test_data"


class TestStageErrorHandling:
    """Tests for stage error handling."""

    def test_failing_stage_pre_process(self, failing_stage):
        """Test stage failing in pre_process."""
        failing_stage.fail_at = "pre_process"

        with pytest.raises(ValueError, match="Intentional pre_process failure"):
            failing_stage.pre_process({"input": "data"})

    def test_failing_stage_predict(self, failing_stage):
        """Test stage failing in predict."""
        preprocessed = failing_stage.pre_process({"input": "data"})
        with pytest.raises(RuntimeError, match="Intentional predict failure"):
            failing_stage.predict(preprocessed)

    def test_failing_stage_post_process(self, failing_stage):
        """Test stage failing in post_process."""
        # Get through pre_process and predict
        failing_stage.fail_at = "never"  # Temporarily disable failure
        preprocessed = failing_stage.pre_process({"input": "data"})
        predictions = failing_stage.predict(preprocessed)

        # Now fail in post_process
        failing_stage.fail_at = "post_process"
        with pytest.raises(ValueError, match="Intentional post_process failure"):
            failing_stage.post_process(predictions)
