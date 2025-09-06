"""
Image processing utilities for SS Image Processor.

This module handles all image-related functionality including:
- Alpha channel detection and reading
- Image validation and processing
- Image dimension retrieval
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# Image processing constants
MIN_ALPHA_CHANNELS = 4


def read_alpha_channel(path: Path) -> Tuple[Optional[np.ndarray], int, int]:
    """Load image and return (alpha_channel, width, height).

    Args:
        path (Path): Path to image.

    Returns:
        Tuple[Optional[np.ndarray], int, int]: alpha (uint8) or None when missing, width, height.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, 0, 0
    h, w = img.shape[:2]
    if img.ndim == 2:
        return None, w, h
    channels = img.shape[2]
    if channels >= MIN_ALPHA_CHANNELS:
        alpha = img[:, :, 3]
        return alpha, w, h
    if channels == 2:
        # grayscale + alpha
        alpha = img[:, :, 1]
        return alpha, w, h
    return None, w, h


def check_alpha_tiff(path: Path) -> bool:
    """Return True if a TIFF has an alpha channel based on channel count.

    Treats 2-channel (gray+alpha) and MIN_ALPHA_CHANNELS+ channel images as having alpha.
    Does not inspect alpha pixel values; only presence by channels.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return False
    if img.ndim == 2:
        return False
    ch = img.shape[2]
    return ch == 2 or ch >= MIN_ALPHA_CHANNELS


def check_alpha_png(path: Path) -> bool:
    """Return True if a PNG has an alpha channel based on channel count.

    Treats 2-channel (gray+alpha) and MIN_ALPHA_CHANNELS-channel (RGBA) PNGs as having alpha.
    Does not inspect alpha pixel values; only presence by channels.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return False
    if img.ndim == 2:
        return False
    ch = img.shape[2]
    return ch == 2 or ch == MIN_ALPHA_CHANNELS


def check_alpha_exists(path: Path) -> bool:
    """Return True if the image file contains an alpha channel by channel count.

    Dispatches by extension for TIFF and PNG. Other formats return False.
    """
    ext = path.suffix.lower()
    if ext in {".tif", ".tiff"}:
        return check_alpha_tiff(path)
    if ext == ".png":
        return check_alpha_png(path)
    return False


def image_dimensions(path: Path) -> Tuple[int, int]:
    """Return (width, height) of an image or (0, 0) on failure.

    Args:
        path (Path): Image file path.

    Returns:
        Tuple[int, int]: Width and height in pixels, or (0, 0) if unreadable.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return 0, 0
    h, w = img.shape[:2]
    return w, h


def validate_tiff_file(file_path: Path) -> Tuple[bool, str]:
    """Check if a TIFF can be processed directly by encoders.

    Treat TIFFs with alpha channels (MIN_ALPHA_CHANNELS+ channels) as invalid for direct processing
    for WebP fallback pipelines.

    Args:
        file_path (Path): Path to the TIFF file.

    Returns:
        Tuple[bool, str]: (is_valid_for_direct_processing, reason)
    """
    try:
        img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return False, "OpenCV could not read the file."

        if len(img.shape) < 3 or img.shape[2] < MIN_ALPHA_CHANNELS:
            return True, f"Image has {img.shape[2] if len(img.shape) > 2 else 1} channel(s), OK for direct processing."

        return False, f"Image has {img.shape[2]} channels and requires splitting."
    except Exception as e:  # pragma: no cover - defensive
        return False, f"Exception during TIFF validation: {e}"


def split_tiff_channels(file_path: Path, output_dir: Path) -> Tuple[bool, Optional[Path], str]:
    """Split multi-channel TIFF into an RGBA PNG for safer encoding.

    Args:
        file_path (Path): Source TIFF path.
        output_dir (Path): Directory for output PNG.

    Returns:
        Tuple[bool, Optional[Path], str]: (success, output_path, message)
    """
    try:
        img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return False, None, "Could not read TIFF file."

        # Check for multi-channel
        if len(img.shape) < 3 or img.shape[2] < MIN_ALPHA_CHANNELS:
            return False, None, f"TIFF has only {img.shape[2] if len(img.shape) > 2 else 1} channel(s), no split needed."

        # Ensure 4-channel RGBA
        if img.shape[2] > MIN_ALPHA_CHANNELS:
            img = img[:, :, :MIN_ALPHA_CHANNELS]

        # Generate output path
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = file_path.stem
        output_path = output_dir / f"{stem}_rgba.png"

        # Write PNG
        success = cv2.imwrite(str(output_path), img)
        if not success:
            return False, None, "Failed to write PNG output."

        return True, output_path, f"Split to RGBA PNG: {output_path.name}"

    except Exception as e:  # pragma: no cover - defensive
        return False, None, f"Exception during TIFF splitting: {e}"