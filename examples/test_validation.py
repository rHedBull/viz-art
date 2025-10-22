"""Example: Point cloud validation with various corrupted test cases.

This script demonstrates the validation framework by testing various
corrupted or invalid point cloud data scenarios.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from viz_art.types.pointcloud import PointCloudValidationRules
from stages.pointcloud_validation import PointCloudValidationStage


def create_test_cases():
    """Create various test point clouds with different quality issues.

    Returns:
        Dictionary of test cases with point arrays and descriptions
    """
    test_cases = {}

    # Valid point cloud
    test_cases["valid"] = {
        "points": np.random.randn(1000, 3).astype(np.float32) * 10,
        "description": "Valid point cloud with 1000 points",
        "expected": "PASS",
    }

    # Too few points
    test_cases["too_few_points"] = {
        "points": np.random.randn(5, 3).astype(np.float32),
        "description": "Too few points (5 points, minimum is 10)",
        "expected": "FAIL",
    }

    # Contains NaN values
    nan_points = np.random.randn(100, 3).astype(np.float32)
    nan_points[10:15, :] = np.nan
    test_cases["contains_nan"] = {
        "points": nan_points,
        "description": "Contains NaN values (5 points with NaN)",
        "expected": "FAIL",
    }

    # Contains Inf values
    inf_points = np.random.randn(100, 3).astype(np.float32)
    inf_points[20:25, 0] = np.inf
    test_cases["contains_inf"] = {
        "points": inf_points,
        "description": "Contains Inf values (5 points with Inf)",
        "expected": "FAIL",
    }

    # Out of coordinate range
    test_cases["out_of_range"] = {
        "points": np.random.randn(100, 3).astype(np.float32) * 100,  # -300 to +300 range
        "description": "Coordinates outside allowed range [-50, 50]",
        "expected": "FAIL",
    }

    # Empty point cloud
    test_cases["empty"] = {
        "points": np.zeros((0, 3), dtype=np.float32),
        "description": "Empty point cloud (0 points)",
        "expected": "FAIL",
    }

    return test_cases


def run_validation_tests():
    """Run validation tests on various corrupted point clouds."""
    print("=" * 70)
    print("Point Cloud Validation Test Suite")
    print("=" * 70)

    # Create validation rules
    rules = PointCloudValidationRules(
        min_points=10,
        max_points=10000,
        check_nan=True,
        check_inf=True,
        coord_range_min=(-50.0, -50.0, -50.0),
        coord_range_max=(50.0, 50.0, 50.0),
        fail_fast=False,  # Check all rules
        log_level="error",
    )

    # Create validator (don't raise on invalid for testing)
    validator = PointCloudValidationStage(
        name="test_validator",
        rules=rules,
        raise_on_invalid=False,  # Don't raise exceptions, return results
    )

    print(f"\nValidation Rules:")
    print(f"  Min points: {rules.min_points}")
    print(f"  Max points: {rules.max_points}")
    print(f"  Check NaN: {rules.check_nan}")
    print(f"  Check Inf: {rules.check_inf}")
    print(f"  Coord range: {rules.coord_range_min} to {rules.coord_range_max}")
    print(f"  Fail fast: {rules.fail_fast}")
    print()

    # Get test cases
    test_cases = create_test_cases()

    # Run tests
    results = []
    print("Running validation tests...")
    print("-" * 70)

    for test_name, test_data in test_cases.items():
        points = test_data["points"]
        description = test_data["description"]
        expected = test_data["expected"]

        print(f"\nTest: {test_name}")
        print(f"  Description: {description}")
        print(f"  Expected: {expected}")
        print(f"  Points shape: {points.shape}")

        # Run validation
        try:
            result = validator.run({"points": points})

            is_valid = result["is_valid"]
            errors = result["validation_errors"]
            warnings = result["validation_warnings"]
            metrics = result["metrics"]

            status = "PASS" if is_valid else "FAIL"
            print(f"  Result: {status}")

            if errors:
                print(f"  Errors:")
                for error in errors:
                    print(f"    - {error}")

            if warnings:
                print(f"  Warnings:")
                for warning in warnings:
                    print(f"    - {warning}")

            print(f"  Metrics:")
            print(f"    - Num points: {metrics['num_points']}")
            print(f"    - Has NaN: {metrics['has_nan']}")
            print(f"    - Has Inf: {metrics['has_inf']}")
            if metrics['coord_ranges']:
                print(f"    - Coord ranges: {metrics['coord_ranges']}")

            # Check if result matches expectation
            correct = (status == expected)
            results.append({
                "test": test_name,
                "expected": expected,
                "actual": status,
                "correct": correct,
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "test": test_name,
                "expected": expected,
                "actual": "ERROR",
                "correct": False,
            })

    # Print summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    total_tests = len(results)
    passed_tests = sum(1 for r in results if r["correct"])
    failed_tests = total_tests - passed_tests

    print(f"\nTotal tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success rate: {passed_tests / total_tests * 100:.1f}%")

    print("\nDetailed Results:")
    print("-" * 70)
    for result in results:
        status_symbol = "✓" if result["correct"] else "✗"
        print(
            f"{status_symbol} {result['test']:<20} "
            f"Expected: {result['expected']:<6} "
            f"Actual: {result['actual']:<6}"
        )

    print("\n" + "=" * 70)
    if failed_tests == 0:
        print("✓ All validation tests passed!")
    else:
        print(f"✗ {failed_tests} test(s) failed")
    print("=" * 70)

    return failed_tests == 0


if __name__ == "__main__":
    success = run_validation_tests()
    sys.exit(0 if success else 1)
