"""
Identifier generation utilities.

T030: UUID generation for run IDs
"""

import uuid


def generate_run_id() -> str:
    """
    Generate a unique run identifier using UUID v4.

    T030: Run ID generation

    Returns:
        str: UUID v4 string (e.g., "a3b5c7d9-1234-5678-90ab-cdef12345678")
    """
    return str(uuid.uuid4())
