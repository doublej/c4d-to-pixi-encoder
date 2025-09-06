"""
Path and file system utilities for SS Image Processor.

This module handles all path-related functionality including:
- Path containment checking
- Directory filtering logic
- Filename parsing for frame sequences
- Sequence discovery and grouping
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from ..config import MIN_SEQUENCE_FRAMES, SUPPORTED_IMAGE_EXTS, SUPPORTED_OUTPUT_EXTS, SequenceInfo


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
    return dirname.startswith("_") or dirname.startswith(".")


def parse_stem(stem: str) -> tuple[str, int] | None:
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


def find_sequences(base_path: Path, for_viewer: bool = False) -> list[SequenceInfo]:
    """
    Walk base_path and detect frame sequences.

    A sequence is a series of files in the same directory, with the same extension,
    and filenames that end in a number (e.g., "render_001.png", "render_002.png").
    Sequences must have at least MIN_SEQUENCE_FRAMES frames (unless for_viewer mode).

    Args:
        base_path: Root directory to scan for sequences
        for_viewer: If True, uses viewer file extensions and no minimum frame count

    Returns:
        List of SequenceInfo objects describing found sequences
    """
    sequences: list[SequenceInfo] = []
    base = base_path.resolve()

    # Choose extensions based on mode
    valid_exts = SUPPORTED_OUTPUT_EXTS if for_viewer else SUPPORTED_IMAGE_EXTS
    min_frames = 1 if for_viewer else MIN_SEQUENCE_FRAMES

    for dirpath, dirnames, filenames in os.walk(base):
        # Prune directories in-place
        if not for_viewer:
            dirnames[:] = [d for d in dirnames if not should_skip_dir(d)]
        else:
            # For viewer, skip hidden dirs
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != "__pycache__"]

        dpath = Path(dirpath)

        if for_viewer:
            # Simple viewer mode - just collect all valid files
            frames = [dpath / f for f in filenames if Path(f).suffix.lower() in valid_exts]
            if frames:
                frames.sort(key=lambda p: natural_key(p.name))
                rel_dir = dpath.relative_to(base)
                sequences.append(
                    SequenceInfo(
                        dir_path=dpath,
                        rel_dir=rel_dir,
                        prefix="",  # Not used in viewer mode
                        ext="",  # Not used in viewer mode
                        frames=frames,
                    )
                )
        else:
            # Standard mode - group by prefix and extension
            potential_sequences: dict[tuple[str, str], list[tuple[int, Path]]] = {}

            for f_name in filenames:
                f_path = dpath / f_name
                ext = f_path.suffix.lower()
                if ext not in valid_exts:
                    continue

                parsed = parse_stem(f_path.stem)
                if parsed:
                    prefix, frame_num = parsed
                    potential_sequences.setdefault((prefix, ext), []).append((frame_num, f_path))

            # Filter for actual sequences
            for (prefix, ext), frames_with_nums in potential_sequences.items():
                if len(frames_with_nums) < min_frames:
                    continue

                # Sort by frame number and extract just the paths
                frames_with_nums.sort(key=lambda item: item[0])
                sorted_frames = [p for _, p in frames_with_nums]

                rel_dir = dpath.relative_to(base)
                sequences.append(
                    SequenceInfo(dir_path=dpath, rel_dir=rel_dir, prefix=prefix, ext=ext, frames=sorted_frames)
                )

    return sequences


def natural_key(s: str) -> list[object]:
    """Convert string to list of mixed integers and strings for natural sorting."""
    return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", s)]
