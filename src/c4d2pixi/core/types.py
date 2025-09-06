"""
Core data types for c4d-to-pixi-encoder.

This module contains the remaining data classes that are specific to processing
logic and haven't been moved to the consolidated config system.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SequenceInfo:
    """A detected numeric frame sequence."""

    dir_path: Path
    rel_dir: Path
    prefix: str
    ext: str
    frames: list[Path]  # sorted numerically

    def __len__(self) -> int:
        return len(self.frames)
