"""Pytest configuration and fixtures for viz-art tests."""

import pytest
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from viz_art.pipeline import PipelineStage


# Mock Stage implementations for testing
class MockStage(PipelineStage):
    """Simple mock stage for testing pipeline behavior."""

    def __init__(self, name: str, input_keys: List[str], output_keys: List[str]):
        self._name = name
        self._input_keys = input_keys
        self._output_keys = output_keys

    @property
    def name(self) -> str:
        return self._name

    @property
    def input_keys(self) -> List[str]:
        return self._input_keys

    @property
    def output_keys(self) -> List[str]:
        return self._output_keys

    def pre_process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs

    def predict(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        # Simple passthrough with prefix
        result = {}
        for key in self._input_keys:
            if key in preprocessed:
                result[f"processed_{key}"] = preprocessed[key]
        return result

    def post_process(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        # Map to output keys
        outputs = {}
        for i, output_key in enumerate(self._output_keys):
            pred_key = f"processed_{self._input_keys[i]}"
            if pred_key in predictions:
                outputs[output_key] = predictions[pred_key]
        return outputs


class FailingStage(PipelineStage):
    """Mock stage that always fails - for error testing."""

    def __init__(self, name: str, fail_at: str = "predict", input_keys: List[str] = None, output_keys: List[str] = None):
        self._name = name
        self.fail_at = fail_at
        self._input_keys = input_keys or ["input"]
        self._output_keys = output_keys or ["output"]

    @property
    def name(self) -> str:
        return self._name

    @property
    def input_keys(self) -> List[str]:
        return self._input_keys

    @property
    def output_keys(self) -> List[str]:
        return self._output_keys

    def pre_process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self.fail_at == "pre_process":
            raise ValueError("Intentional pre_process failure")
        return inputs

    def predict(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        if self.fail_at == "predict":
            raise RuntimeError("Intentional predict failure")
        return {"result": "data"}

    def post_process(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        if self.fail_at == "post_process":
            raise ValueError("Intentional post_process failure")
        return {"output": predictions["result"]}


# Pytest fixtures
@pytest.fixture
def mock_stage():
    """Create a simple mock stage."""
    return MockStage(name="test_stage", input_keys=["input"], output_keys=["output"])


@pytest.fixture
def mock_stage_a():
    """Create mock stage A for connection tests."""
    return MockStage(name="stage_a", input_keys=["data"], output_keys=["result_a"])


@pytest.fixture
def mock_stage_b():
    """Create mock stage B for connection tests."""
    return MockStage(name="stage_b", input_keys=["result_a"], output_keys=["result_b"])


@pytest.fixture
def mock_stage_c():
    """Create mock stage C for 3-stage tests."""
    return MockStage(name="stage_c", input_keys=["result_b"], output_keys=["final"])


@pytest.fixture
def failing_stage():
    """Create a stage that fails during predict."""
    return FailingStage(name="failing_stage", fail_at="predict")


@pytest.fixture
def sample_image_path():
    """Return path to a test image."""
    return Path(__file__).parent / "fixtures" / "sample_images" / "test_640x480.jpg"


@pytest.fixture
def sample_image_array():
    """Create a simple test image array."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def fixtures_dir():
    """Return path to fixtures directory."""
    return Path(__file__).parent / "fixtures"
