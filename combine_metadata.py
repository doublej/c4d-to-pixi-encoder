#!/usr/bin/env python3
"""
Combine a pair of sidecar JSON files ("[srcname].dpi.json" and "[srcname].json")
into a single "metadata.json" with computed frame count and source folder.

Designed to be run in a directory that contains the output sidecars created by
the existing pipeline, where `[srcname]` often includes the output extension
(e.g., "myseq.webp" → "myseq.webp.json" and "myseq.webp.dpi.json").
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


# -----------------------------
# Data structures
# -----------------------------

@dataclass(frozen=True)
class PairPaths:
    """Resolved pair of input JSON files for a source name.

    Attributes:
        dpi_json (Path): Path to the `[srcname].dpi.json` file.
        meta_json (Path): Path to the `[srcname].json` file.
    """

    dpi_json: Path
    meta_json: Path


# -----------------------------
# Core helpers (pure)
# -----------------------------

def build_pair_paths(dir_path: Path, srcname: str) -> PairPaths:
    """Construct the two expected file paths for a given `srcname` and directory.

    Args:
        dir_path (Path): Directory containing the JSON files.
        srcname (str): Source name without the trailing `.json`/`.dpi.json`.

    Returns:
        PairPaths: Paths for the dpi sidecar and the main metadata JSON.
    """
    dpi = dir_path / f"{srcname}.dpi.json"
    meta = dir_path / f"{srcname}.json"
    return PairPaths(dpi_json=dpi, meta_json=meta)


def validate_pair_exists(pair: PairPaths) -> None:
    """Validate that both JSON files exist; raise FileNotFoundError if missing.

    Args:
        pair (PairPaths): Resolved pair of JSON paths.
    """
    missing = [p for p in (pair.dpi_json, pair.meta_json) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required file(s): " + ", ".join(str(m) for m in missing)
        )


def load_json(path: Path) -> Dict:
    """Load a JSON file into a dict with basic validation.

    Args:
        path (Path): JSON file path.

    Returns:
        Dict: Parsed JSON object.

    Raises:
        ValueError: When the file is not a JSON object.
    """
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object in {path.name}, got {type(data).__name__}")
    return data


def shallow_merge(left: Dict, right: Dict) -> Dict:
    """Return a new dict with keys from both, with `right` taking precedence.

    Args:
        left (Dict): Base mapping.
        right (Dict): Overrides mapping.

    Returns:
        Dict: Merged mapping.
    """
    merged = dict(left)
    merged.update(right)
    return merged


def derive_source_folder_and_frame_count(dpi_payload: Dict) -> Tuple[Optional[Path], Optional[int]]:
    """Compute source folder and frame count using the DPI payload's `source_path`.

    Counts files in the same folder with the same extension and the same numeric
    sequence prefix (trailing digits ignored when matching the prefix).

    Args:
        dpi_payload (Dict): Data loaded from `[srcname].dpi.json`.

    Returns:
        Tuple[Optional[Path], Optional[int]]: (folder_path, frame_count) or (None, None)
            if `source_path` is absent.
    """
    source_path = dpi_payload.get("source_path")
    if not source_path:
        return None, None

    sp = Path(source_path)
    folder = sp.parent
    ext = sp.suffix.lower()
    stem = sp.stem

    # Extract trailing number to get the prefix; if none, count files with same extension.
    prefix, numeric_suffix = _parse_trailing_number(stem)

    if prefix is None:
        # No numeric suffix detected → count same-ext files in folder.
        count = sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ext)
        return folder, count

    # Count files whose stem shares the prefix and ends with digits, with same extension.
    count = 0
    for p in folder.iterdir():
        if not p.is_file() or p.suffix.lower() != ext:
            continue
        p_prefix, p_num = _parse_trailing_number(p.stem)
        if p_prefix is not None and p_prefix == prefix and p_num is not None:
            count += 1
    return folder, count


def _parse_trailing_number(stem: str) -> Tuple[Optional[str], Optional[int]]:
    """Return (prefix_without_trailing_digits, number) or (None, None) if absent.

    Args:
        stem (str): Filename stem to parse.

    Returns:
        Tuple[Optional[str], Optional[int]]: (prefix, int) when a trailing number is present.
    """
    import re

    m = re.search(r"(\d+)$", stem)
    if not m:
        return None, None
    num_s = m.group(1)
    prefix = stem[: -len(num_s)]
    try:
        return prefix, int(num_s)
    except ValueError:
        return prefix, None


def build_output_payload(
    merged: Dict, source_folder: Optional[Path], frame_count: Optional[int]
) -> Dict:
    """Assemble the final `metadata.json` payload.

    Args:
        merged (Dict): Combined dictionary from the two inputs.
        source_folder (Optional[Path]): Folder containing the frames.
        frame_count (Optional[int]): Number of frames detected.

    Returns:
        Dict: Serializable payload for `metadata.json`.
    """
    payload: Dict = {}
    payload.update(merged)
    payload["source_folder"] = str(source_folder) if source_folder else None
    payload["frame_count"] = int(frame_count) if frame_count is not None else None
    return payload


# -----------------------------
# Orchestration (I/O at the edge)
# -----------------------------

def combine_to_metadata(dir_path: Path, srcname: str, output_name: str = "metadata.json") -> Path:
    """Combine `[srcname].dpi.json` and `[srcname].json` into `metadata.json`.

    Args:
        dir_path (Path): Directory containing the sidecar JSON files.
        srcname (str): Base name used by the sidecars (without `.json` suffixes).
        output_name (str): Output filename; default `metadata.json`.

    Returns:
        Path: Path to the written `metadata.json`.

    Raises:
        FileNotFoundError: When expected input files are missing.
        ValueError: When input JSONs are not objects.
    """
    pair = build_pair_paths(dir_path, srcname)
    validate_pair_exists(pair)

    dpi_data = load_json(pair.dpi_json)
    meta_data = load_json(pair.meta_json)

    # Merge with meta_data taking precedence on key conflicts.
    merged = shallow_merge(dpi_data, meta_data)

    # Derive source folder and frame count from the DPI payload.
    source_folder, frame_count = derive_source_folder_and_frame_count(dpi_data)
    
    payload = build_output_payload(merged, source_folder, frame_count)

    out_path = dir_path / output_name
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    # Cleanup: remove all JSON files in dir_path except the output_name
    for p in dir_path.glob("*.json"):
        if p.name != output_name:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
    return out_path


def autodetect_single_pair(dir_path: Path) -> Optional[str]:
    """Find a single `[srcname]` that has both `.dpi.json` and `.json` files.

    Args:
        dir_path (Path): Directory to scan.

    Returns:
        Optional[str]: The detected `srcname` if exactly one pair exists, else None.
    """
    dpi_map = {}
    json_map = {}
    for p in dir_path.iterdir():
        if not p.is_file() or not p.name.endswith(".json"):
            continue
        if p.name.endswith(".dpi.json"):
            key = p.name[: -len(".dpi.json")]
            dpi_map[key] = p
        else:
            key = p.name[: -len(".json")]
            json_map[key] = p

    candidates = sorted(set(dpi_map.keys()) & set(json_map.keys()))
    if len(candidates) == 1:
        return candidates[0]
    return None


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    p = argparse.ArgumentParser(
        description=(
            "Combine `[srcname].dpi.json` and `[srcname].json` into `metadata.json` "
            "and include frame count and source folder."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "directory",
        type=Path,
        nargs="?",
        default=Path("."),
        help="Directory containing the sidecar JSON files",
    )
    p.add_argument(
        "--srcname",
        type=str,
        help=(
            "Source base name (e.g., 'myseq.webp'). If omitted, the tool tries to autodetect "
            "when exactly one matching pair exists in the directory."
        ),
    )
    p.add_argument(
        "--output-name",
        type=str,
        default="metadata.json",
        help="Name of the combined output file to write",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    dir_path: Path = args.directory.resolve()
    if not dir_path.exists() or not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    srcname: Optional[str] = args.srcname
    if not srcname:
        srcname = autodetect_single_pair(dir_path)
        if not srcname:
            raise ValueError(
                "Cannot autodetect a unique pair. Provide --srcname (e.g., 'myseq.webp')."
            )

    out_path = combine_to_metadata(dir_path, srcname=srcname, output_name=args.output_name)
    print(out_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
