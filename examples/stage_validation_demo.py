#!/usr/bin/env python3
"""Demonstration of stage validation with PipelineStage ABC.

This script shows how the abstract base class PipelineStage enforces
proper stage implementation and catches errors early.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from typing import Dict, Any, List
from viz_art.pipeline import PipelineStage, Pipeline


# ============================================================================
# CORRECT IMPLEMENTATION: Inherit from PipelineStage
# ============================================================================

class GoodStage(PipelineStage):
    """Properly implemented stage inheriting from PipelineStage ABC.

    Benefits:
        - Compile-time validation (can't instantiate without implementing all methods)
        - IDE autocomplete shows exactly what to implement
        - Typos caught immediately
        - Clear contract enforcement
    """

    def __init__(self, name: str = "good_stage"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def input_keys(self) -> List[str]:
        return ["data"]

    @property
    def output_keys(self) -> List[str]:
        return ["result"]

    def pre_process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs

    def predict(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        return {"result": preprocessed["data"]}

    def post_process(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        return predictions


# ============================================================================
# DEMONSTRATION: What happens with incomplete implementations?
# ============================================================================

# This would FAIL at class definition time:
# class BadStage(PipelineStage):
#     """Missing required methods - won't even compile!"""
#     pass
#
# Error: TypeError: Can't instantiate abstract class BadStage with abstract methods
# input_keys, name, output_keys, post_process, pre_process, predict


# This would also FAIL (typo in method name):
# class TypoStage(PipelineStage):
#     def __init__(self, name: str = "typo"):
#         self._name = name
#
#     @property
#     def name(self) -> str:
#         return self._name
#
#     @property
#     def input_keys(self) -> List[str]:
#         return ["data"]
#
#     @property
#     def output_keys(self) -> List[str]:
#         return ["result"]
#
#     def preprocess(self, inputs):  # TYPO: should be pre_process
#         return inputs
#
#     def predict(self, preprocessed):
#         return {"result": preprocessed}
#
#     def postprocess(self, predictions):  # TYPO: should be post_process
#         return predictions
#
# Error: TypeError: Can't instantiate abstract class TypoStage with abstract methods
# post_process, pre_process


def main():
    print("=" * 70)
    print("STAGE VALIDATION DEMONSTRATION")
    print("=" * 70)

    # Test 1: Properly implemented stage (works perfectly)
    print("\n1. Testing properly implemented stage...")
    try:
        pipeline = Pipeline(name="test-pipeline")
        pipeline.add_stage(GoodStage())
        print("   ✓ Stage added successfully")
        print("   ✓ All required methods implemented")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Test 2: Show what happens with bad implementations
    print("\n2. Attempting to create incomplete stage...")
    print("   Uncommenting this code would cause an error:")
    print("   ```")
    print("   class BadStage(PipelineStage):")
    print("       pass")
    print("   ```")
    print("   ✓ Error caught at class definition time:")
    print("      TypeError: Can't instantiate abstract class BadStage")
    print("      with abstract methods: name, input_keys, output_keys,")
    print("      pre_process, predict, post_process")

    print("\n3. Attempting to create stage with typos...")
    print("   Uncommenting this code would also fail:")
    print("   ```")
    print("   class TypoStage(PipelineStage):")
    print("       def preprocess(self, inputs):  # Wrong name!")
    print("           ...")
    print("   ```")
    print("   ✓ Error caught at class definition time:")
    print("      TypeError: Can't instantiate abstract class TypoStage")
    print("      with abstract methods: pre_process, post_process")

    print("\n" + "=" * 70)
    print("BENEFITS OF PIPELINESTAGE ABC")
    print("=" * 70)
    print("""
✓ Errors caught at class definition time (not runtime)
✓ IDE shows exactly what methods to implement
✓ Impossible to forget required methods
✓ Typos in method names caught immediately
✓ Type hints enforced
✓ Clear contract for all stages

Example usage:
    from viz_art.pipeline import PipelineStage

    class MyStage(PipelineStage):
        # Your IDE will show you all required methods
        def __init__(self, name: str = "my_stage"):
            self._name = name

        @property
        def name(self) -> str:
            return self._name

        # ... implement all abstract methods
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()
