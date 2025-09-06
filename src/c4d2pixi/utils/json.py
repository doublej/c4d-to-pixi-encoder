"""Centralized JSON handling utilities."""

import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON file with error handling.

    Args:
        path: Path to JSON file

    Returns:
        Parsed JSON data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If JSON is invalid
    """
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any], indent: int | None = 2) -> None:
    """Write data to JSON file.

    Args:
        path: Path to write JSON file
        data: Data to serialize
        indent: JSON indentation (None for compact)
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def shallow_merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Shallow merge two dictionaries, right values override left.

    Args:
        left: Base dictionary
        right: Override dictionary

    Returns:
        Merged dictionary
    """
    return {**left, **right}
