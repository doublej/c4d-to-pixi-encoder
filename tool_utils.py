"""
External tool validation utilities for SS Image Processor.

This module handles validation of external dependencies and tools
required for the image processing pipeline.
"""

from __future__ import annotations

from shutil import which
from typing import List, Tuple


def check_tools() -> Tuple[bool, List[str]]:
    """Check availability of required external tools.

    Args:
        None

    Returns:
        Tuple[bool, List[str]]: (all_ok, problems). If `all_ok` is False, problems lists the issues.
    """
    problems: List[str] = []
    if which("ffmpeg") is None:
        problems.append("ffmpeg not found in PATH")
    return (len(problems) == 0, problems)