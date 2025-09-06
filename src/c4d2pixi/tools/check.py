"""
External tool validation utilities for SS Image Processor.

This module handles validation of external dependencies and tools
required for the image processing pipeline.
"""

from __future__ import annotations

from shutil import which


def check_tools() -> tuple[bool, list[str]]:
    """Check availability of required external tools.

    Args:
        None

    Returns:
        Tuple[bool, List[str]]: (all_ok, problems). If `all_ok` is False, problems lists the issues.
    """
    problems: list[str] = []
    if which("ffmpeg") is None:
        problems.append("ffmpeg not found in PATH")
    return (len(problems) == 0, problems)
