"""viz-art: Vision Processing Pipeline Library

A lightweight library for building multi-stage vision processing pipelines.
"""

__version__ = "0.1.0"
__author__ = "viz-art contributors"

# Expose main classes at package level
from viz_art.pipeline.base import Pipeline

__all__ = ["Pipeline", "__version__"]
