# Batch Processing Test Example

This directory contains a complete example of batch processing with viz-art.

## Quick Start

### 1. Generate Test Images (Already Done)

```bash
python create_test_images.py
```

This creates 5 sample test images in the `input/` directory.

### 2. Run Batch Processing

**Option A: Using the test script (Recommended)**

```bash
python run_batch_test.py
```

**Option B: Using the CLI script**

```bash
python ../../scripts/batch_process.py --config batch_config.yaml
```

### 3. View the Report

After processing, open the HTML report:

```bash
# Linux
xdg-open output/batch_report.html

# macOS
open output/batch_report.html

# Windows
start output/batch_report.html
```

Or just open `examples/test_batch/output/batch_report.html` in your browser.

## What to Expect

The batch processing will:
1. Discover all `.jpg` images in the `input/` directory
2. Process each image through a 2-stage pipeline:
   - **Loader**: Load and resize images to 640x480
   - **Filter**: Apply Gaussian blur filter
3. Generate an HTML report with:
   - Summary statistics
   - Stage-grouped view of all processed images
   - Per-image detailed results
   - Error section (if any failures)

## Directory Structure

```
test_batch/
├── README.md              # This file
├── batch_config.yaml      # Pipeline configuration
├── create_test_images.py  # Script to generate test images
├── run_batch_test.py      # Quick test script
├── input/                 # Test images directory
│   ├── test_landscape_1.jpg
│   ├── test_landscape_2.jpg
│   ├── test_portrait_1.jpg
│   ├── test_square_1.jpg
│   └── test_square_2.jpg
└── output/                # Generated after processing
    └── batch_report.html  # HTML report
```

## Configuration

Edit `batch_config.yaml` to:
- Change pipeline stages
- Adjust stage parameters
- Modify input/output directories
- Change file patterns
- Enable/disable error handling

## Troubleshooting

**No images found**: Make sure you ran `create_test_images.py` first.

**Import errors**: Make sure you're running from the project root or the test script handles the path correctly.

**Report not generated**: Check the console output for errors during processing.
