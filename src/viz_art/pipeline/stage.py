"""Pipeline stage base class definition.

This module defines the abstract base class that all pipeline stages must inherit from.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class PipelineStage(ABC):
    """Abstract base class for all pipeline stages.

    All stages must inherit from this class and implement all abstract methods.
    Stages process data through three phases: pre_process, predict, post_process.
    Each stage declares its input and output keys for data flow validation.

    Example:
        >>> class MyStage(PipelineStage):
        ...     def __init__(self, name: str = "my_stage"):
        ...         self._name = name
        ...
        ...     @property
        ...     def name(self) -> str:
        ...         return self._name
        ...
        ...     @property
        ...     def input_keys(self) -> List[str]:
        ...         return ["image"]
        ...
        ...     @property
        ...     def output_keys(self) -> List[str]:
        ...         return ["processed_image"]
        ...
        ...     def pre_process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        ...         return inputs
        ...
        ...     def predict(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        ...         # Main processing logic
        ...         return {"result": preprocessed["image"]}
        ...
        ...     def post_process(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        ...         return {"processed_image": predictions["result"]}
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name of this stage within the pipeline.

        Returns:
            str: Stage identifier used for reporting and connections
        """
        ...

    @property
    @abstractmethod
    def input_keys(self) -> List[str]:
        """Return list of required input data keys.

        Returns:
            List[str]: Keys that must be present in the input dictionary
        """
        ...

    @property
    @abstractmethod
    def output_keys(self) -> List[str]:
        """Return list of output data keys this stage produces.

        Returns:
            List[str]: Keys that will be present in the output dictionary
        """
        ...

    @abstractmethod
    def pre_process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-processing step before main prediction.

        This method validates inputs and performs any necessary transformations
        before the main processing step.

        Args:
            inputs: Dictionary of input data (keys match self.input_keys)

        Returns:
            Dict[str, Any]: Processed data for predict() method

        Raises:
            ValueError: If required inputs are missing or invalid
        """
        ...

    @abstractmethod
    def predict(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        """Main processing step (prediction, transformation, etc.).

        This is where the core stage logic executes.

        Args:
            preprocessed: Output from pre_process()

        Returns:
            Dict[str, Any]: Prediction/processing results

        Raises:
            RuntimeError: If processing fails
        """
        ...

    @abstractmethod
    def post_process(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Post-processing step to finalize outputs.

        This method formats the prediction results into the final output format.

        Args:
            predictions: Output from predict()

        Returns:
            Dict[str, Any]: Final outputs with keys matching self.output_keys

        Raises:
            ValueError: If output formatting fails
        """
        ...

    def __repr__(self) -> str:
        """Return string representation of stage."""
        return f"{self.__class__.__name__}(name='{self.name}')"
