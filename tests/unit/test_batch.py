"""Unit tests for batch processing functionality."""

import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch

from viz_art.batch.processor import BatchProcessor
from viz_art.batch.reporter import HTMLReporter
from viz_art.config.schema import BatchConfigItem
from viz_art.pipeline.base import Pipeline
from viz_art.pipeline.results import BatchResult, PipelineRun, RunStatus


@pytest.fixture
def mock_pipeline():
    """Create a mock pipeline for testing."""
    pipeline = Mock(spec=Pipeline)
    pipeline.name = "test_pipeline"
    pipeline._stages = []
    return pipeline


@pytest.fixture
def batch_config(tmp_path):
    """Create a batch config for testing."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    return BatchConfigItem(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        file_patterns=["*.jpg", "*.png"],
        recursive=True,
        continue_on_error=True,
        report_output="report.html",
    )


@pytest.fixture
def sample_images(tmp_path):
    """Create sample image files for testing."""
    input_dir = tmp_path / "input"
    input_dir.mkdir(exist_ok=True)

    # Create some dummy image files
    images = []
    for i in range(3):
        img_path = input_dir / f"image{i}.jpg"
        img_path.write_text(f"dummy image {i}")
        images.append(img_path)

    # Create a non-image file
    txt_file = input_dir / "readme.txt"
    txt_file.write_text("not an image")

    return images


class TestBatchProcessor:
    """Tests for BatchProcessor class."""

    def test_init_with_valid_inputs(self, mock_pipeline, batch_config):
        """Test BatchProcessor initialization with valid inputs."""
        processor = BatchProcessor(mock_pipeline, batch_config)
        assert processor.pipeline == mock_pipeline
        assert processor.config == batch_config

    def test_init_with_none_pipeline(self, batch_config):
        """Test BatchProcessor initialization fails with None pipeline."""
        with pytest.raises(ValueError, match="pipeline cannot be None"):
            BatchProcessor(None, batch_config)

    def test_init_with_none_config(self, mock_pipeline):
        """Test BatchProcessor initialization fails with None config."""
        with pytest.raises(ValueError, match="config cannot be None"):
            BatchProcessor(mock_pipeline, None)

    def test_discover_images_finds_images(self, mock_pipeline, batch_config, sample_images):
        """Test that discover_images finds image files correctly."""
        # Update batch config to use correct input dir
        batch_config = BatchConfigItem(
            input_dir=str(sample_images[0].parent),
            output_dir=batch_config.output_dir,
            file_patterns=["*.jpg"],
            recursive=False,
        )

        processor = BatchProcessor(mock_pipeline, batch_config)
        discovered = list(processor.discover_images())

        assert len(discovered) == 3
        assert all(img.suffix == ".jpg" for img in discovered)

    def test_discover_images_skips_non_images(self, mock_pipeline, batch_config, sample_images):
        """Test that discover_images silently skips non-image files."""
        batch_config = BatchConfigItem(
            input_dir=str(sample_images[0].parent),
            output_dir=batch_config.output_dir,
            file_patterns=["*.txt"],
            recursive=False,
        )

        processor = BatchProcessor(mock_pipeline, batch_config)
        discovered = list(processor.discover_images())

        # .txt files should be skipped (not in IMAGE_EXTENSIONS)
        assert len(discovered) == 0

    def test_discover_images_recursive(self, mock_pipeline, tmp_path):
        """Test recursive image discovery."""
        # Create nested directory structure
        input_dir = tmp_path / "input"
        sub_dir = input_dir / "subdir"
        sub_dir.mkdir(parents=True)

        # Create images in different directories
        (input_dir / "img1.jpg").write_text("img1")
        (sub_dir / "img2.png").write_text("img2")

        batch_config = BatchConfigItem(
            input_dir=str(input_dir),
            output_dir=str(tmp_path / "output"),
            file_patterns=["*.jpg", "*.png"],
            recursive=True,
        )

        processor = BatchProcessor(mock_pipeline, batch_config)
        discovered = list(processor.discover_images())

        assert len(discovered) == 2
        filenames = {img.name for img in discovered}
        assert filenames == {"img1.jpg", "img2.png"}

    def test_discover_images_non_recursive(self, mock_pipeline, tmp_path):
        """Test non-recursive image discovery."""
        input_dir = tmp_path / "input"
        sub_dir = input_dir / "subdir"
        sub_dir.mkdir(parents=True)

        # Create images in different directories
        (input_dir / "img1.jpg").write_text("img1")
        (sub_dir / "img2.jpg").write_text("img2")

        batch_config = BatchConfigItem(
            input_dir=str(input_dir),
            output_dir=str(tmp_path / "output"),
            file_patterns=["*.jpg"],
            recursive=False,
        )

        processor = BatchProcessor(mock_pipeline, batch_config)
        discovered = list(processor.discover_images())

        # Should only find img1.jpg (not in subdir)
        assert len(discovered) == 1
        assert discovered[0].name == "img1.jpg"

    def test_discover_images_invalid_directory(self, mock_pipeline):
        """Test discover_images raises error for non-existent directory."""
        batch_config = BatchConfigItem(
            input_dir="/nonexistent/directory",
            output_dir="/tmp/output",
        )

        processor = BatchProcessor(mock_pipeline, batch_config)

        with pytest.raises(ValueError, match="Input directory does not exist"):
            list(processor.discover_images())

    def test_run_empty_directory(self, mock_pipeline, batch_config):
        """Test batch processing with no images returns empty result."""
        processor = BatchProcessor(mock_pipeline, batch_config)
        result = processor.run()

        assert isinstance(result, BatchResult)
        assert result.total_files == 0
        assert result.successful == 0
        assert result.failed == 0
        assert len(result.run_results) == 0

    def test_run_successful_processing(self, mock_pipeline, tmp_path, sample_images):
        """Test successful batch processing of multiple images."""
        # Setup mock pipeline to return success
        mock_pipeline.run.return_value = {
            "_run_id": "test-run-id",
            "_status": "completed",
            "_pipeline_name": "test_pipeline",
            "_started_at": datetime.utcnow().isoformat(),
            "_completed_at": datetime.utcnow().isoformat(),
            "loader": {"image": "dummy"},
        }

        batch_config = BatchConfigItem(
            input_dir=str(sample_images[0].parent),
            output_dir=str(tmp_path / "output"),
            file_patterns=["*.jpg"],
            recursive=False,
        )

        processor = BatchProcessor(mock_pipeline, batch_config)
        result = processor.run()

        assert result.total_files == 3
        assert result.successful == 3
        assert result.failed == 0
        assert len(result.run_results) == 3

    def test_run_continue_on_error(self, mock_pipeline, tmp_path, sample_images):
        """Test that processing continues when an image fails."""
        # Make pipeline fail on second image
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("Simulated processing error")
            return {
                "_run_id": f"run-{call_count[0]}",
                "_status": "completed",
                "_pipeline_name": "test_pipeline",
                "_started_at": datetime.utcnow().isoformat(),
                "_completed_at": datetime.utcnow().isoformat(),
                "stage1": {"output": "data"},
            }

        mock_pipeline.run.side_effect = side_effect

        batch_config = BatchConfigItem(
            input_dir=str(sample_images[0].parent),
            output_dir=str(tmp_path / "output"),
            file_patterns=["*.jpg"],
            recursive=False,
            continue_on_error=True,
        )

        processor = BatchProcessor(mock_pipeline, batch_config)
        result = processor.run()

        # Should process all 3, with 1 failure
        assert result.total_files == 3
        assert result.successful == 2
        assert result.failed == 1

    def test_run_fails_without_continue_on_error(self, mock_pipeline, tmp_path, sample_images):
        """Test that processing stops when continue_on_error=False."""
        # Make pipeline fail on first image
        mock_pipeline.run.side_effect = RuntimeError("Simulated error")

        batch_config = BatchConfigItem(
            input_dir=str(sample_images[0].parent),
            output_dir=str(tmp_path / "output"),
            file_patterns=["*.jpg"],
            recursive=False,
            continue_on_error=False,
        )

        processor = BatchProcessor(mock_pipeline, batch_config)

        with pytest.raises(RuntimeError, match="Batch processing stopped"):
            processor.run()


class TestHTMLReporter:
    """Tests for HTMLReporter class."""

    def test_init_with_default_templates(self):
        """Test HTMLReporter initialization with default templates."""
        reporter = HTMLReporter()
        assert reporter.template_dir.exists()
        assert (reporter.template_dir / "batch_report.html").exists()

    def test_init_with_custom_template_dir(self, tmp_path):
        """Test HTMLReporter initialization with custom template directory."""
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        (template_dir / "batch_report.html").write_text("<html></html>")

        reporter = HTMLReporter(template_dir=template_dir)
        assert reporter.template_dir == template_dir

    def test_init_with_nonexistent_directory(self):
        """Test HTMLReporter initialization fails with non-existent directory."""
        with pytest.raises(ValueError, match="Template directory does not exist"):
            HTMLReporter(template_dir="/nonexistent/templates")

    def test_generate_report_success(self, tmp_path):
        """Test successful HTML report generation."""
        # Create sample batch result
        batch_result = BatchResult(
            batch_id="test-batch",
            total_files=2,
            successful=2,
            failed=0,
            run_results=[
                PipelineRun(
                    run_id="run1",
                    pipeline_name="test",
                    started_at=datetime.utcnow(),
                    completed_at=datetime.utcnow(),
                    status=RunStatus.COMPLETED,
                    inputs={"image_path": "/test/img1.jpg"},
                    outputs={"stage1": {"output": "data"}},
                    stage_results=[],
                ),
                PipelineRun(
                    run_id="run2",
                    pipeline_name="test",
                    started_at=datetime.utcnow(),
                    completed_at=datetime.utcnow(),
                    status=RunStatus.COMPLETED,
                    inputs={"image_path": "/test/img2.jpg"},
                    outputs={"stage1": {"output": "data"}},
                    stage_results=[],
                ),
            ],
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            report_path="",
        )

        reporter = HTMLReporter()
        output_path = tmp_path / "report.html"

        result_path = reporter.generate(batch_result, output_path, pipeline_name="TestPipeline")

        assert result_path.exists()
        assert result_path.is_file()
        content = result_path.read_text()
        assert "TestPipeline" in content
        assert "test-batch" in content

    def test_generate_report_with_none_batch_result(self, tmp_path):
        """Test that generate raises error with None batch_result."""
        reporter = HTMLReporter()
        output_path = tmp_path / "report.html"

        with pytest.raises(ValueError, match="batch_result cannot be None"):
            reporter.generate(None, output_path)

    def test_organize_by_stage(self, tmp_path):
        """Test _organize_by_stage groups results correctly."""
        batch_result = BatchResult(
            batch_id="test",
            total_files=1,
            successful=1,
            failed=0,
            run_results=[
                PipelineRun(
                    run_id="run1",
                    pipeline_name="test",
                    started_at=datetime.utcnow(),
                    completed_at=datetime.utcnow(),
                    status=RunStatus.COMPLETED,
                    inputs={"image_path": "/test/img1.jpg"},
                    outputs={
                        "loader": {"image": "data1"},
                        "filter": {"filtered_image": "data2"},
                    },
                    stage_results=[],
                ),
            ],
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            report_path="",
        )

        reporter = HTMLReporter()
        report_path = tmp_path / "test_report.html"
        stages_data = reporter._organize_by_stage(batch_result, report_path)

        assert "loader" in stages_data
        assert "filter" in stages_data
        assert len(stages_data["loader"]) == 1
        assert len(stages_data["filter"]) == 1
