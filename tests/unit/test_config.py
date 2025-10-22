"""Unit tests for configuration system."""

import pytest
from pathlib import Path
from pydantic import ValidationError

from viz_art.config.schema import (
    StageConfigItem,
    ConnectionItem,
    BatchConfigItem,
    PipelineConfig,
)
from viz_art.config.loader import load_config, load_config_from_dict
from viz_art.pipeline.base import Pipeline


class TestStageConfigItem:
    """Tests for StageConfigItem model."""

    def test_valid_stage_config(self):
        """Test valid stage configuration."""
        stage = StageConfigItem(
            name="test_stage", stage_type="TestStage", config={"param": "value"}
        )

        assert stage.name == "test_stage"
        assert stage.stage_type == "TestStage"
        assert stage.config == {"param": "value"}
        assert stage.enabled is True

    def test_stage_config_with_disabled(self):
        """Test stage can be disabled."""
        stage = StageConfigItem(
            name="disabled", stage_type="TestStage", enabled=False
        )

        assert stage.enabled is False

    def test_stage_empty_name_raises_error(self):
        """Test empty stage name raises ValidationError."""
        with pytest.raises(ValidationError, match="Stage name cannot be empty"):
            StageConfigItem(name="", stage_type="TestStage")

    def test_stage_invalid_name_raises_error(self):
        """Test invalid characters in name raise ValidationError."""
        with pytest.raises(ValidationError, match="alphanumeric"):
            StageConfigItem(name="stage@#$", stage_type="TestStage")

    def test_stage_name_with_hyphens_underscores(self):
        """Test stage name allows hyphens and underscores."""
        stage = StageConfigItem(name="test-stage_1", stage_type="TestStage")
        assert stage.name == "test-stage_1"


class TestConnectionItem:
    """Tests for ConnectionItem model."""

    def test_valid_connection(self):
        """Test valid connection configuration."""
        conn = ConnectionItem(
            source="stage1", target="stage2", output_key="out", input_key="in"
        )

        assert conn.source == "stage1"
        assert conn.target == "stage2"
        assert conn.required is True

    def test_connection_not_required(self):
        """Test connection can be optional."""
        conn = ConnectionItem(
            source="s1", target="s2", output_key="out", input_key="in", required=False
        )

        assert conn.required is False

    def test_connection_empty_field_raises_error(self):
        """Test empty connection field raises ValidationError."""
        with pytest.raises(ValidationError):
            ConnectionItem(
                source="", target="stage2", output_key="out", input_key="in"
            )


class TestBatchConfigItem:
    """Tests for BatchConfigItem model."""

    def test_valid_batch_config(self):
        """Test valid batch configuration."""
        batch = BatchConfigItem(input_dir="./input")

        assert batch.input_dir == "./input"
        assert batch.output_dir == "./output"
        assert "*.png" in batch.file_patterns
        assert batch.recursive is True
        assert batch.continue_on_error is True

    def test_batch_config_custom_values(self):
        """Test batch config with custom values."""
        batch = BatchConfigItem(
            input_dir="./data",
            output_dir="./results",
            file_patterns=["*.tiff"],
            recursive=False,
            continue_on_error=False,
            report_output="custom_report.html",
        )

        assert batch.file_patterns == ["*.tiff"]
        assert batch.recursive is False
        assert batch.report_output == "custom_report.html"

    def test_batch_empty_patterns_raises_error(self):
        """Test empty file patterns raise ValidationError."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            BatchConfigItem(input_dir="./input", file_patterns=[])


class TestPipelineConfig:
    """Tests for PipelineConfig model."""

    def test_valid_pipeline_config(self):
        """Test valid pipeline configuration."""
        config = PipelineConfig(
            pipeline_name="test",
            stages=[
                StageConfigItem(name="stage1", stage_type="Type1"),
                StageConfigItem(name="stage2", stage_type="Type2"),
            ],
            connections=[
                ConnectionItem(
                    source="stage1", target="stage2", output_key="out", input_key="in"
                )
            ],
        )

        assert config.pipeline_name == "test"
        assert len(config.stages) == 2
        assert len(config.connections) == 1

    def test_pipeline_config_empty_stages_raises_error(self):
        """Test empty stages list raises ValidationError."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            PipelineConfig(pipeline_name="test", stages=[])

    def test_pipeline_config_duplicate_stage_names_raises_error(self):
        """Test duplicate stage names raise ValidationError."""
        with pytest.raises(ValidationError, match="must be unique"):
            PipelineConfig(
                pipeline_name="test",
                stages=[
                    StageConfigItem(name="stage1", stage_type="Type1"),
                    StageConfigItem(name="stage1", stage_type="Type2"),  # Duplicate
                ],
            )

    def test_pipeline_config_connection_invalid_source(self):
        """Test connection referencing non-existent source raises error."""
        with pytest.raises(ValidationError, match="not found in stages"):
            PipelineConfig(
                pipeline_name="test",
                stages=[StageConfigItem(name="stage1", stage_type="Type1")],
                connections=[
                    ConnectionItem(
                        source="unknown",  # Not in stages
                        target="stage1",
                        output_key="out",
                        input_key="in",
                    )
                ],
            )


class TestConfigLoader:
    """Tests for configuration loading."""

    def test_load_config_from_dict(self):
        """Test loading config from dictionary."""
        config_dict = {
            "pipeline_name": "dict_test",
            "stages": [{"name": "s1", "stage_type": "TestStage"}],
        }

        config = load_config_from_dict(config_dict)

        assert config.pipeline_name == "dict_test"
        assert len(config.stages) == 1

    def test_load_config_from_file(self, fixtures_dir):
        """Test loading config from YAML file."""
        config_path = fixtures_dir.parent.parent / "examples" / "configs" / "simple_pipeline.yaml"

        if not config_path.exists():
            pytest.skip("Example config not found")

        config = load_config(config_path)

        assert config.pipeline_name == "simple-config-pipeline"
        assert len(config.stages) >= 2

    def test_load_config_nonexistent_file_raises_error(self):
        """Test loading non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_load_config_invalid_yaml_raises_error(self, tmp_path):
        """Test invalid YAML raises ValueError."""
        invalid_yaml = tmp_path / "invalid.yaml"
        invalid_yaml.write_text("invalid: yaml: content:")

        with pytest.raises(ValueError, match="Failed to load config"):
            load_config(invalid_yaml)


class TestPipelineFromConfig:
    """Tests for Pipeline.from_config()."""

    def test_pipeline_from_config_basic(self):
        """Test creating pipeline from config."""
        from conftest import MockStage

        config_dict = {
            "pipeline_name": "config_test",
            "stages": [
                {
                    "name": "stage_a",
                    "stage_type": "MockStage",
                    "config": {"input_keys": ["data"], "output_keys": ["result_a"]},
                },
                {
                    "name": "stage_b",
                    "stage_type": "MockStage",
                    "config": {"input_keys": ["result_a"], "output_keys": ["result_b"]},
                },
            ],
            "connections": [
                {
                    "source": "stage_a",
                    "target": "stage_b",
                    "output_key": "result_a",
                    "input_key": "result_a",
                }
            ],
        }

        config = load_config_from_dict(config_dict)

        # Custom registry
        registry = {"MockStage": MockStage}

        pipeline = Pipeline.from_config(config, stage_registry=registry)

        assert pipeline.name == "config_test"
        assert len(pipeline.stages) == 2
        assert len(pipeline.connections) == 1

    def test_pipeline_from_config_skips_disabled_stages(self):
        """Test disabled stages are skipped."""
        from conftest import MockStage

        config_dict = {
            "pipeline_name": "test",
            "stages": [
                {
                    "name": "enabled",
                    "stage_type": "MockStage",
                    "enabled": True,
                    "config": {"input_keys": ["data"], "output_keys": ["result"]},
                },
                {
                    "name": "disabled",
                    "stage_type": "MockStage",
                    "enabled": False,
                    "config": {"input_keys": ["data"], "output_keys": ["result"]},
                },
            ],
        }

        config = load_config_from_dict(config_dict)
        registry = {"MockStage": MockStage}

        pipeline = Pipeline.from_config(config, stage_registry=registry)

        assert len(pipeline.stages) == 1  # Only enabled stage
        assert pipeline.stages[0].name == "enabled"

    def test_pipeline_from_config_unknown_stage_type_raises_error(self):
        """Test unknown stage type raises ValueError."""
        config_dict = {
            "pipeline_name": "test",
            "stages": [{"name": "s1", "stage_type": "UnknownStage"}],
        }

        config = load_config_from_dict(config_dict)

        with pytest.raises(ValueError, match="not found in registry"):
            Pipeline.from_config(config, stage_registry={})

    def test_pipeline_from_config_with_stage_params(self):
        """Test stage receives config parameters."""
        from conftest import MockStage

        config_dict = {
            "pipeline_name": "test",
            "stages": [
                {
                    "name": "s1",
                    "stage_type": "MockStage",
                    "config": {
                        "input_keys": ["data"],
                        "output_keys": ["result"],
                    },
                }
            ],
        }

        config = load_config_from_dict(config_dict)
        registry = {"MockStage": MockStage}

        pipeline = Pipeline.from_config(config, stage_registry=registry)

        stage = pipeline.stages[0]
        assert stage.input_keys == ["data"]
        assert stage.output_keys == ["result"]
