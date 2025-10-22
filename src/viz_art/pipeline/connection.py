"""Stage connection management.

This module defines the StageConnection class that represents data flow between stages.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class StageConnection:
    """Defines data flow between two pipeline stages.

    A connection specifies how output from one stage becomes input to another.
    Connections are validated during pipeline construction to ensure:
    - Both stages exist in the pipeline
    - Output key exists in source stage's output_keys
    - Input key exists in target stage's input_keys

    Attributes:
        source_stage: Name of the stage producing the output
        target_stage: Name of the stage consuming the input
        output_key: Key in the source stage's output dictionary
        input_key: Key in the target stage's input dictionary
        required: Whether this connection must succeed (default True)

    Example:
        >>> conn = StageConnection(
        ...     source_stage="loader",
        ...     target_stage="filter",
        ...     output_key="image",
        ...     input_key="image",
        ...     required=True
        ... )
    """

    source_stage: str
    target_stage: str
    output_key: str
    input_key: str
    required: bool = True

    def __post_init__(self):
        """Validate connection fields."""
        if not self.source_stage:
            raise ValueError("source_stage cannot be empty")
        if not self.target_stage:
            raise ValueError("target_stage cannot be empty")
        if not self.output_key:
            raise ValueError("output_key cannot be empty")
        if not self.input_key:
            raise ValueError("input_key cannot be empty")
        if self.source_stage == self.target_stage:
            raise ValueError("source_stage and target_stage cannot be the same")
