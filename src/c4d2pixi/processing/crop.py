"""
Cropping utilities for SS Image Processor.

This module handles all cropping-related functionality including:
- Alpha-aware edge cropping with alignment
- Single image and sequence-wide crop calculations
- FFmpeg crop filter generation
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

# Cropping constants
CROP_ALIGNMENT_PIXELS = 1


def find_transparent_aligned_crop(alpha: np.ndarray, alignment: int = CROP_ALIGNMENT_PIXELS) -> tuple[int, int, int, int]:
    """Compute crop rectangle by trimming fully-transparent blocks from edges.

    Args:
        alpha (np.ndarray): 2D uint8 alpha channel.
        alignment (int): Pixel alignment for crop boundaries.

    Returns:
        Tuple[int, int, int, int]: (x0, y0, x1, y1) crop bounds, half-open.
    """
    h, w = alpha.shape[:2]
    x0, y0, x1, y1 = 0, 0, w, h

    # top
    while (y1 - y0) >= alignment:
        band = alpha[y0 : y0 + alignment, x0:x1]
        if np.any(band):
            break
        y0 += alignment

    # bottom
    while (y1 - y0) >= alignment:
        band = alpha[y1 - alignment : y1, x0:x1]
        if np.any(band):
            break
        y1 -= alignment

    # left
    while (x1 - x0) >= alignment:
        band = alpha[y0:y1, x0 : x0 + alignment]
        if np.any(band):
            break
        x0 += alignment

    # right
    while (x1 - x0) >= alignment:
        band = alpha[y0:y1, x1 - alignment : x1]
        if np.any(band):
            break
        x1 -= alignment

    # Ensure valid rectangle
    x0 = max(0, min(x0, w))
    y0 = max(0, min(y0, h))
    x1 = max(x0, min(x1, w))
    y1 = max(y0, min(y1, h))
    return x0, y0, x1, y1


def compute_aligned_crop(path: Path, alignment: int = CROP_ALIGNMENT_PIXELS) -> tuple[int, int, int, int, int, int]:
    """Determine crop rectangle for an image file.

    Args:
        path (Path): Image file to analyze.
        alignment (int): Pixel alignment for crop boundaries.

    Returns:
        Tuple[int, int, int, int, int, int]: (crop_x, crop_y, crop_w, crop_h, orig_w, orig_h)
    """
    from .image import read_alpha_channel  # Avoid circular import

    alpha, orig_w, orig_h = read_alpha_channel(path)
    if alpha is None:
        return 0, 0, orig_w, orig_h, orig_w, orig_h

    x0, y0, x1, y1 = find_transparent_aligned_crop(alpha, alignment)
    return x0, y0, x1 - x0, y1 - y0, orig_w, orig_h


def crop_filter_from_rect(x: int, y: int, w: int, h: int, orig_w: int, orig_h: int) -> str | None:
    """Generate FFmpeg crop filter string from rectangle, or None if no cropping needed.

    Args:
        x, y, w, h: Crop rectangle (offset and size).
        orig_w, orig_h: Original image dimensions.

    Returns:
        Optional[str]: FFmpeg crop filter string, or None if no cropping.
    """
    if x == 0 and y == 0 and w == orig_w and h == orig_h:
        return None
    return f"crop={w}:{h}:{x}:{y}"


def _presence_blocks(alpha: np.ndarray, block: int) -> tuple[list[bool], list[bool]]:
    """Return per-block presence booleans for columns (x) and rows (y).

    Args:
        alpha (np.ndarray): Alpha channel array.
        block (int): Block size for analysis.

    Returns:
        Tuple[List[bool], List[bool]]: (x_presence, y_presence)
    """
    h, w = alpha.shape[:2]
    n_x = (w + block - 1) // block
    n_y = (h + block - 1) // block

    x_pres = []
    for i in range(n_x):
        x_start = i * block
        x_end = min((i + 1) * block, w)
        col_slice = alpha[:, x_start:x_end]
        x_pres.append(np.any(col_slice))

    y_pres = []
    for j in range(n_y):
        y_start = j * block
        y_end = min((j + 1) * block, h)
        row_slice = alpha[y_start:y_end, :]
        y_pres.append(np.any(row_slice))

    return x_pres, y_pres


def _combine_presence(acc: list[bool], new: list[bool]) -> list[bool]:
    """Combine two presence lists using logical OR.

    Args:
        acc (List[bool]): Accumulator list.
        new (List[bool]): New list to combine.

    Returns:
        List[bool]: Combined presence list.
    """
    return [a or b for a, b in zip(acc, new, strict=False)]


def _bounds_from_presence(pres: list[bool], block: int, limit: int) -> tuple[int, int]:
    """Compute tight bounds from a presence list.

    Args:
        pres (List[bool]): Presence boolean list.
        block (int): Block size.
        limit (int): Maximum coordinate (width or height).

    Returns:
        Tuple[int, int]: (start, end) bounds.
    """
    # Find first and last present blocks
    first = next((i for i, p in enumerate(pres) if p), 0)
    last = len(pres) - 1 - next((i for i, p in enumerate(reversed(pres)) if p), 0)

    start = first * block
    end = min((last + 1) * block, limit)
    return start, end


def compute_sequence_aligned_crop(
    paths: Sequence[Path], alignment: int = CROP_ALIGNMENT_PIXELS
) -> tuple[int, int, int, int, int, int]:
    """Compute crop rectangle for a sequence of images with aligned cropping.

    Args:
        paths (Sequence[Path]): Image paths to analyze.
        alignment (int): Pixel alignment for crop boundaries.

    Returns:
        Tuple[int, int, int, int, int, int]: (crop_x, crop_y, crop_w, crop_h, orig_w, orig_h)
    """
    from .image import read_alpha_channel  # Avoid circular import

    if not paths:
        return 0, 0, 0, 0, 0, 0

    # Read first image to get dimensions
    alpha, orig_w, orig_h = read_alpha_channel(paths[0])
    if alpha is None:
        return 0, 0, orig_w, orig_h, orig_w, orig_h

    # Initialize presence accumulators
    x_acc, y_acc = _presence_blocks(alpha, alignment)

    # Process remaining frames
    for path in paths[1:]:
        alpha, w, h = read_alpha_channel(path)
        if alpha is None or w != orig_w or h != orig_h:
            continue
        x_pres, y_pres = _presence_blocks(alpha, alignment)
        x_acc = _combine_presence(x_acc, x_pres)
        y_acc = _combine_presence(y_acc, y_pres)

    # Compute final bounds
    x0, x1 = _bounds_from_presence(x_acc, alignment, orig_w)
    y0, y1 = _bounds_from_presence(y_acc, alignment, orig_h)

    return x0, y0, x1 - x0, y1 - y0, orig_w, orig_h
