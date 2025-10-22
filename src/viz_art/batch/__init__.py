"""Batch processing module for viz-art.

This module provides functionality for processing multiple images through
a pipeline and generating HTML reports.
"""

from viz_art.batch.processor import BatchProcessor
from viz_art.batch.reporter import HTMLReporter

__all__ = ["BatchProcessor", "HTMLReporter"]
