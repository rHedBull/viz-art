# Sample Point Cloud Test Fixtures

This directory contains point cloud test data for unit and integration testing.

## Files

### Valid Test Data

- `small_100pts.{pcd,ply,xyz}` - Small 100-point random cube
  - Format: All supported formats
  - Features: Points + colors
  - Use: Fast unit tests

- `medium_10k.{pcd,ply}` - Medium 10K-point sphere
  - Format: PCD and PLY
  - Features: Points + colors + normals
  - Use: Performance tests, normal estimation

- `large_100k.pcd` - Large 100K-point random cloud
  - Format: PCD only (file size)
  - Features: Points + colors
  - Use: Performance benchmarks, downsampling tests

### Invalid Test Data (for validation testing)

- `empty.pcd` - Empty point cloud (0 points)
  - Use: Error handling tests

- `corrupted_nan.pcd` - Contains NaN values
  - Use: Validation framework tests

- `corrupted_inf.pcd` - Contains Inf values
  - Use: Validation framework tests

## Regenerating Test Data

If test data is corrupted or needs updating:

```bash
cd tests/fixtures/sample_pointclouds/
python generate_test_data.py
```

Requires: `open3d>=0.18`

## Usage in Tests

```python
import pytest
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "sample_pointclouds"

def test_load_pointcloud():
    pcd_path = FIXTURES_DIR / "small_100pts.pcd"
    # ... test code
```

Generated automatically - do not edit manually.
