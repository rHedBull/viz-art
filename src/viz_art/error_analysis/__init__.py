"""Error analysis and pattern detection module.

This module provides functionality for detecting, categorizing, and visualizing
error cases, clustering errors into patterns, and browsing failures.
"""

from .patterns import ErrorType, ErrorSeverity, ErrorCase, ErrorPattern

__all__ = [
    "ErrorType",
    "ErrorSeverity",
    "ErrorCase",
    "ErrorPattern",
]
