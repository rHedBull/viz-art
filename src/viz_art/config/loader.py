"""Configuration loader using OmegaConf and Pydantic.

This module provides functions to load YAML configuration files,
resolve variables, and validate against Pydantic schemas.
"""

from pathlib import Path
from typing import Union
from omegaconf import OmegaConf, DictConfig
from pydantic import ValidationError

from viz_art.config.schema import PipelineConfig


def load_config(config_path: Union[str, Path]) -> PipelineConfig:
    """Load and validate pipeline configuration from YAML file.

    This function:
    1. Loads YAML file using OmegaConf
    2. Resolves variable interpolations
    3. Converts to native Python dict
    4. Validates with Pydantic schema
    5. Returns validated PipelineConfig object

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Validated PipelineConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If YAML is invalid or validation fails
        ValidationError: If Pydantic validation fails

    Example:
        >>> config = load_config("configs/my_pipeline.yaml")
        >>> print(config.pipeline_name)
        >>> print(config.stages[0].name)
    """
    path = Path(config_path)

    # Check file exists
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {config_path}")

    try:
        # Load YAML with OmegaConf
        omega_cfg: DictConfig = OmegaConf.load(path)

        # Convert to native Python dict with variable resolution
        # resolve=True ensures ${...} interpolations are resolved
        cfg_dict = OmegaConf.to_container(omega_cfg, resolve=True)

        # Validate with Pydantic
        validated_config = PipelineConfig.model_validate(cfg_dict)

        return validated_config

    except ValidationError as e:
        # Re-raise with more context
        raise ValidationError.from_exception_data(
            title=f"Configuration validation failed for {path}",
            line_errors=e.errors()
        )
    except Exception as e:
        # Handle YAML parsing errors
        raise ValueError(f"Failed to load config from {path}: {str(e)}")


def load_config_from_dict(config_dict: dict) -> PipelineConfig:
    """Load and validate configuration from dictionary.

    Useful for programmatic config creation or testing.

    Args:
        config_dict: Dictionary containing configuration

    Returns:
        Validated PipelineConfig instance

    Raises:
        ValidationError: If validation fails

    Example:
        >>> config_dict = {
        ...     "pipeline_name": "test",
        ...     "stages": [{"name": "s1", "stage_type": "StageA"}],
        ... }
        >>> config = load_config_from_dict(config_dict)
    """
    return PipelineConfig.model_validate(config_dict)
