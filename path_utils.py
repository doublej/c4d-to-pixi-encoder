"""
Path and file system utilities for SS Image Processor.

This module handles all path-related functionality including:
- Path containment checking
- Directory filtering logic
- Filename parsing for frame sequences
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple


def is_path_inside(child: Path, parent: Path) -> bool:
    """Return True if `child` is the same as or nested under `parent`.

    Args:
        child (Path): Path to test for containment.
        parent (Path): Parent directory to test against.

    Returns:
        bool: True when `child` is inside `parent`, else False.
    """
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def should_skip_dir(dirname: str) -> bool:
    """Heuristic for directories to skip during scans.

    Skips typical output or hidden/temp folders.

    Args:
        dirname (str): Directory name (not full path).

    Returns:
        bool: True if the directory should be skipped.
    """
    lowered = dirname.lower()
    if lowered in {"outputs", "webp_renders", "h264_renders"}:
        return True
    if dirname.startswith("_") or dirname.startswith("."):
        return True
    return False


def parse_stem(stem: str) -> Optional[Tuple[str, int]]:
    """Parse a filename stem into a (prefix, trailing_number).

    Finds a number at the very end of the stem.
    Examples:
        "render_001" -> ("render_", 1)
        "001" -> ("", 1)
        "no_number" -> None

    Args:
        stem (str): Filename stem to parse.

    Returns:
        Optional[Tuple[str, int]]: (prefix, number) if matched, else None.
    """
    match = re.search(r"(\d+)$", stem)
    if not match:
        return None
    number_str = match.group(1)
    prefix = stem[: -len(number_str)]
    return prefix, int(number_str)