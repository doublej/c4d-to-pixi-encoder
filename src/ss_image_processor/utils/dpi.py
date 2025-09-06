"""
Utilities for reading and reporting image DPI per sequence.

These helpers are small, single-purpose, and avoid altering pixel data.
They read DPI once per sequence (from the first readable frame) and
provide a normalized value (DEFAULT_DPI, DEFAULT_DPI) for downstream reporting.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

# DPI constants
DEFAULT_DPI = 72.0


@dataclass(frozen=True)
class DpiInfo:
    """Container for DPI information.

    Attributes:
        src_x (float): Source horizontal DPI (pixels per inch).
        src_y (float): Source vertical DPI (pixels per inch).
        normalized_x (float): Target normalized DPI (always DEFAULT_DPI).
        normalized_y (float): Target normalized DPI (always DEFAULT_DPI).
        source_path (Optional[Path]): Path of the frame that provided the DPI, if any.
    """

    src_x: float
    src_y: float
    normalized_x: float
    normalized_y: float
    source_path: Path | None


def _read_image_dpi(path: Path) -> tuple[float, float] | None:
    """Return (x_dpi, y_dpi) for an image, or None when unavailable.

    Args:
        path (Path): Image file path.

    Returns:
        Optional[Tuple[float, float]]: Extracted DPI if present.

    Notes:
        - Pillow exposes DPI via Image.info.get("dpi") for JPEG/PNG/TIFF.
        - For PNG with pHYs (pixels per meter), Pillow converts to DPI.
        - Formats like EXR/DPX generally lack conventional DPI; returns None.
    """
    try:
        with Image.open(path) as im:
            dpi = im.info.get("dpi")
            if isinstance(dpi, tuple) and len(dpi) == 2:
                x, y = dpi
                try:
                    return float(x), float(y)
                except Exception:
                    return None
            # Some plugins use "resolution" or pHYs converted values; best-effort
            res = im.info.get("resolution")
            if isinstance(res, tuple) and len(res) == 2:
                try:
                    return float(res[0]), float(res[1])
                except Exception:
                    return None
    except Exception:
        # Unsupported file or plugin not available
        return None
    return None


def read_sequence_dpi(frames: Sequence[Path], normalized: float = DEFAULT_DPI) -> DpiInfo:
    """Read DPI from the first frame that provides it; default to DEFAULT_DPI if missing.

    Args:
        frames (Sequence[Path]): Ordered sequence of frame paths.
        normalized (float): Target DPI to normalize/report.

    Returns:
        DpiInfo: Collected DPI info with defaults if none were found.
    """
    for p in frames:
        dpi = _read_image_dpi(p)
        if dpi is not None:
            return DpiInfo(
                src_x=dpi[0],
                src_y=dpi[1],
                normalized_x=float(normalized),
                normalized_y=float(normalized),
                source_path=p,
            )
    # Fallback when no frame provides DPI metadata
    return DpiInfo(
        src_x=float(normalized),
        src_y=float(normalized),
        normalized_x=float(normalized),
        normalized_y=float(normalized),
        source_path=None,
    )


def dpi_dict(info: DpiInfo) -> dict:
    """Convert DpiInfo to a JSON-serializable dictionary.

    Args:
        info (DpiInfo): DPI information container.

    Returns:
        dict: Serializable values for writing to JSON.
    """
    return {
        "source_dpi_x": round(float(info.src_x), 4),
        "source_dpi_y": round(float(info.src_y), 4),
        "normalized_dpi_x": round(float(info.normalized_x), 4),
        "normalized_dpi_y": round(float(info.normalized_y), 4),
        "source_path": str(info.source_path) if info.source_path else None,
    }
