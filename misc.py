"""
Miscellaneous utility functions extracted from the main script to keep
the core orchestration focused and readable.

These helpers are intentionally small, single-purpose, and side-effect free
unless performing explicit I/O (e.g., filesystem or subprocess).
"""

from __future__ import annotations

import os
import re
import shlex
import tempfile
from pathlib import Path
from shutil import which
from subprocess import CalledProcessError, run
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import json
import numpy as np
from typing import Dict


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


def validate_tiff_file(file_path: Path) -> Tuple[bool, str]:
    """Check if a TIFF can be processed directly by encoders.

    Treat TIFFs with alpha channels (4+ channels) as invalid for direct processing
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

        if len(img.shape) < 3 or img.shape[2] < 4:
            return True, f"Image has {img.shape[2] if len(img.shape) > 2 else 1} channel(s), OK for direct processing."

        return False, f"Image has {img.shape[2]} channels and requires splitting."
    except Exception as e:  # pragma: no cover - defensive
        return False, f"Exception during TIFF validation: {e}"


def split_tiff_channels(file_path: Path, output_dir: Path) -> Tuple[bool, Optional[Path], str]:
    """Split multi-channel TIFF into an RGBA PNG for safer encoding.

    Args:
        file_path (Path): Source TIFF path.
        output_dir (Path): Directory to place the temporary PNG.

    Returns:
        Tuple[bool, Optional[Path], str]: (success, path_to_rgba_png|None, message)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_png_path = output_dir / f"{file_path.stem}_rgba_temp.png"

    try:
        img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return False, None, "OpenCV could not read file."

        if len(img.shape) < 3 or img.shape[2] < 4:
            return False, None, "Image does not have enough channels to process."

        if img.shape[2] > 4:
            img = img[:, :, :4]

        success = cv2.imwrite(str(temp_png_path), img)
        if not success:
            return False, None, "Failed to write temporary RGBA PNG file."

        return True, temp_png_path, "Successfully created temporary RGBA PNG."
    except Exception as e:  # pragma: no cover - defensive
        return False, None, f"Exception during channel splitting: {e}"


def write_ffconcat_file(frame_paths: List[Path], target_dir: Path) -> Path:
    """Write an ffconcat list file enumerating frames in order.

    Args:
        frame_paths (List[Path]): Sequence of frame file paths.
        target_dir (Path): Directory where the list file will be written.

    Returns:
        Path: Path to the created ffconcat file.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    list_path = target_dir / "frames.ffconcat"
    with list_path.open("w", encoding="utf-8") as f:
        f.write("ffconcat version 1.0\n")
        for p in frame_paths:
            posix = p.resolve().as_posix().replace("'", r"'\''")
            f.write(f"file '{posix}'\n")
    return list_path


def run_subprocess(cmd: List[str]) -> Tuple[int, str]:
    """Run a subprocess command, returning (exit_code, pretty_cmd).

    Args:
        cmd (List[str]): Command and arguments, split.

    Returns:
        Tuple[int, str]: Return code and a shell-quoted pretty string of the command.
    """
    pretty = " ".join(shlex.quote(c) for c in cmd)
    try:
        res = run(cmd, check=True)
        return res.returncode, pretty
    except CalledProcessError as e:
        return e.returncode, pretty


# --------------------------------------------
# Alpha-aware 256px edge-cropping helpers
# --------------------------------------------

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
    if channels >= 4:
        alpha = img[:, :, 3]
        return alpha, w, h
    if channels == 2:
        # grayscale + alpha
        alpha = img[:, :, 1]
        return alpha, w, h
    return None, w, h


def find_transparent_256_crop(alpha: np.ndarray) -> Tuple[int, int, int, int]:
    """Compute crop rectangle by trimming fully-transparent 256px blocks from edges.

    Args:
        alpha (np.ndarray): 2D uint8 alpha channel.

    Returns:
        Tuple[int, int, int, int]: (x0, y0, x1, y1) crop bounds, half-open.
    """
    h, w = alpha.shape[:2]
    x0, y0, x1, y1 = 0, 0, w, h

    # top
    while (y1 - y0) >= 256:
        band = alpha[y0 : y0 + 256, x0:x1]
        if np.any(band):
            break
        y0 += 256

    # bottom
    while (y1 - y0) >= 256:
        band = alpha[y1 - 256 : y1, x0:x1]
        if np.any(band):
            break
        y1 -= 256

    # left
    while (x1 - x0) >= 256:
        band = alpha[y0:y1, x0 : x0 + 256]
        if np.any(band):
            break
        x0 += 256

    # right
    while (x1 - x0) >= 256:
        band = alpha[y0:y1, x1 - 256 : x1]
        if np.any(band):
            break
        x1 -= 256

    # Ensure valid rectangle
    x0 = max(0, min(x0, w))
    y0 = max(0, min(y0, h))
    x1 = max(x0, min(x1, w))
    y1 = max(y0, min(y1, h))
    return x0, y0, x1, y1


def compute_256_crop(path: Path) -> Tuple[int, int, int, int, int, int]:
    """Determine 256px-edge crop rectangle for an image file.

    Args:
        path (Path): Image path.

    Returns:
        Tuple[int,int,int,int,int,int]: (x, y, w, h, orig_w, orig_h)
    """
    alpha, orig_w, orig_h = read_alpha_channel(path)
    if alpha is None or orig_w == 0 or orig_h == 0:
        return 0, 0, orig_w, orig_h, orig_w, orig_h
    x0, y0, x1, y1 = find_transparent_256_crop(alpha)
    return x0, y0, (x1 - x0), (y1 - y0), orig_w, orig_h


def crop_filter_from_rect(x: int, y: int, w: int, h: int, orig_w: int, orig_h: int) -> Optional[str]:
    """Build ffmpeg crop filter expression for a given rectangle.

    Returns None if the rectangle covers the full image.
    """
    if x == 0 and y == 0 and w == orig_w and h == orig_h:
        return None
    return f"crop={w}:{h}:{x}:{y}"


def write_offset_json(json_path: Path, offset_x: int, offset_y: int, crop_w: int, crop_h: int, orig_w: int, orig_h: int) -> None:
    """Write an offset JSON file next to the output.

    Fields:
        - offset_x, offset_y: number of pixels cropped from left and top
        - cropped_width, cropped_height: output dimensions after crop
        - original_width, original_height: source dimensions
    """
    data = {
        "offset_x": int(offset_x),
        "offset_y": int(offset_y),
        "cropped_width": int(crop_w),
        "cropped_height": int(crop_h),
        "original_width": int(orig_w),
        "original_height": int(orig_h),
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_dpi_json(json_path: Path, dpi_payload: Dict) -> None:
    """Write a DPI sidecar JSON next to outputs.

    Args:
        json_path (Path): Target file path for JSON.
        dpi_payload (Dict): Serializable DPI info (e.g., from dpi_utils.dpi_dict).

    Returns:
        None
    """
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(dpi_payload, f, indent=2)


# --------------------------------------------
# Sequence-wide 256px crop (left→right, top→bottom)
# --------------------------------------------

def _presence_blocks(alpha: np.ndarray, block: int) -> Tuple[List[bool], List[bool]]:
    """Return per-256px presence booleans for columns (x) and rows (y).

    Scans left→right for x blocks and top→bottom for y blocks.
    """
    h, w = alpha.shape[:2]
    bx = (w + block - 1) // block
    by = (h + block - 1) // block
    x_pres = []
    for i in range(bx):
        xs, xe = i * block, min((i + 1) * block, w)
        band = alpha[:, xs:xe]
        x_pres.append(bool(np.any(band)))
    y_pres = []
    for j in range(by):
        ys, ye = j * block, min((j + 1) * block, h)
        band = alpha[ys:ye, :]
        y_pres.append(bool(np.any(band)))
    return x_pres, y_pres


def _combine_presence(acc: List[bool], new: List[bool]) -> List[bool]:
    """OR-combine two presence lists, extending to the longer length.

    True always wins.
    """
    m = max(len(acc), len(new))
    out = [False] * m
    for i in range(m):
        a = acc[i] if i < len(acc) else False
        b = new[i] if i < len(new) else False
        out[i] = a or b
    return out


def _bounds_from_presence(pres: List[bool], block: int, limit: int) -> Tuple[int, int]:
    """Find first and last True indices scanning in one direction.

    Returns pixel-space [start, end) rounded to block boundaries and clipped to limit.
    """
    first = None
    last = None
    for i, v in enumerate(pres):
        if v:
            if first is None:
                first = i
            last = i
    if first is None:
        # no data → keep full dimension
        return 0, limit
    start = max(0, min(first * block, limit))
    end = max(start, min((last + 1) * block, limit))
    return start, end


def compute_sequence_256_crop(paths: Sequence[Path], block: int = 256) -> Tuple[int, int, int, int, int, int]:
    """Compute a sequence-wide crop by OR-combining alpha presence across frames.

    Scans in one direction (left→right for columns, top→bottom for rows) per image
    to build presence blocks, combines them with OR (true wins), and returns a crop
    rectangle that contains all data across frames.

    Returns:
        (x, y, w, h, orig_w, orig_h), where orig_* are the max width/height across frames.
    """
    combined_x: List[bool] = []
    combined_y: List[bool] = []
    max_w = 0
    max_h = 0

    for p in paths:
        alpha, w, h = read_alpha_channel(p)
        max_w = max(max_w, w)
        max_h = max(max_h, h)
        if alpha is None or w == 0 or h == 0:
            # No alpha or failed read → contributes nothing (all False)
            continue
        x_pres, y_pres = _presence_blocks(alpha, block)
        combined_x = _combine_presence(combined_x, x_pres)
        combined_y = _combine_presence(combined_y, y_pres)

    x0, x1 = _bounds_from_presence(combined_x, block, max_w)
    y0, y1 = _bounds_from_presence(combined_y, block, max_h)
    return x0, y0, (x1 - x0), (y1 - y0), max_w, max_h
