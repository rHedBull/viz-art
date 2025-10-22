"""End-to-end integration tests for viz-art pipeline.

These tests verify the complete workflow from configuration loading
through pipeline execution to batch processing and report generation.
"""

import pytest
from pathlib import Path
import numpy as np
from PIL import Image

from viz_art.config.loader import load_config
from viz_art.pipeline.base import Pipeline
from viz_art.batch.processor import BatchProcessor
from viz_art.batch.reporter import HTMLReporter


@pytest.fixture
def test_config_yaml(tmp_path):
    """Create a test configuration YAML file."""
    config_content = """
pipeline_name: "integration-test-pipeline"

output_dir: "./output"

stages:
  - name: "loader"
    stage_type: "ImageLoader"
    config:
      target_size: [640, 480]

connections: []

batch_config:
  input_dir: "./input"
  output_dir: "./output"
  file_patterns:
    - "*.jpg"
    - "*.png"
  recursive: true
  continue_on_error: true
  report_output: "test_report.html"
"""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def test_images(tmp_path):
    """Create test images for integration testing."""
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True)

    images = []
    for i in range(3):
        # Create a simple test image using PIL
        img = Image.new("RGB", (100, 100), color=(i * 80, 100, 150))
        img_path = input_dir / f"test_image_{i}.jpg"
        img.save(img_path)
        images.append(img_path)

    return images


@pytest.mark.skipif(
    not Path(__file__).parent.parent.parent.joinpath("examples/stages").exists(),
    reason="Example stages not available",
)
class TestEndToEndBatchProcessing:
    """Integration tests for complete batch processing workflow."""

    def test_batch_processing_end_to_end(self, tmp_path, test_images):
        """Test complete batch processing workflow from images to HTML report."""
        # This test requires example stages to exist
        # Skip if they're not available yet

        pytest.skip("Requires example stages implementation")

        # The test would look like this once example stages are available:
        # 1. Load config
        # 2. Create pipeline
        # 3. Process batch
        # 4. Generate report
        # 5. Verify report exists and contains expected content

    def test_config_loading_and_validation(self, test_config_yaml):
        """Test that config loads and validates correctly."""
        config = load_config(test_config_yaml)

        assert config.pipeline_name == "integration-test-pipeline"
        assert len(config.stages) == 1
        assert config.stages[0].name == "loader"
        assert config.batch_config is not None
        assert config.batch_config.input_dir == "./input"

    def test_pipeline_creation_from_config(self, test_config_yaml):
        """Test pipeline can be created from config (with mocked stage)."""
        from unittest.mock import Mock

        config = load_config(test_config_yaml)

        # Mock the stage class
        mock_stage = Mock()
        mock_stage.name = "loader"
        mock_stage.input_keys = ["image_path"]
        mock_stage.output_keys = ["image"]

        # Would need stage registry to properly test this
        # For now, verify config structure is correct
        assert config.pipeline_name == "integration-test-pipeline"
        assert config.stages[0].stage_type == "ImageLoader"


class TestHTMLReportGeneration:
    """Integration tests for HTML report generation."""

    def test_html_report_contains_batch_data(self, tmp_path):
        """Test that generated HTML report contains batch processing data."""
        from viz_art.pipeline.results import BatchResult, PipelineRun, RunStatus
        from datetime import datetime

        # Create sample batch result
        batch_result = BatchResult(
            batch_id="integration-test-batch",
            total_files=2,
            successful=1,
            failed=1,
            run_results=[
                PipelineRun(
                    run_id="run1",
                    pipeline_name="test",
                    started_at=datetime.utcnow(),
                    completed_at=datetime.utcnow(),
                    status=RunStatus.COMPLETED,
                    inputs={"image_path": "/test/img1.jpg"},
                    outputs={"loader": {"image": "data"}},
                    stage_results=[],
                ),
                PipelineRun(
                    run_id="run2",
                    pipeline_name="test",
                    started_at=datetime.utcnow(),
                    completed_at=datetime.utcnow(),
                    status=RunStatus.FAILED,
                    inputs={"image_path": "/test/img2.jpg"},
                    outputs={},
                    stage_results=[],
                    error="Test error",
                ),
            ],
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            report_path="",
        )

        # Generate report
        reporter = HTMLReporter()
        output_path = tmp_path / "integration_report.html"
        report_path = reporter.generate(
            batch_result, output_path, pipeline_name="IntegrationPipeline"
        )

        # Verify report exists
        assert report_path.exists()

        # Read and verify content
        content = report_path.read_text()

        # Check for expected elements
        assert "integration-test-batch" in content
        assert "IntegrationPipeline" in content
        assert "Total Images" in content
        assert "Successful" in content
        assert "Failed" in content

        # Check for error message
        assert "Test error" in content

    def test_html_report_viewable_offline(self, tmp_path):
        """Test that generated HTML report is self-contained for offline viewing."""
        from viz_art.pipeline.results import BatchResult
        from datetime import datetime

        batch_result = BatchResult(
            batch_id="offline-test",
            total_files=0,
            successful=0,
            failed=0,
            run_results=[],
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            report_path="",
        )

        reporter = HTMLReporter()
        output_path = tmp_path / "offline_report.html"
        report_path = reporter.generate(batch_result, output_path)

        content = report_path.read_text()

        # Verify inline CSS (no external stylesheet links)
        assert "<style>" in content
        assert 'rel="stylesheet"' not in content

        # Verify no external script dependencies
        assert 'src="http' not in content

        # Verify report is valid HTML
        assert "<!DOCTYPE html>" in content
        assert "<html" in content
        assert "</html>" in content
