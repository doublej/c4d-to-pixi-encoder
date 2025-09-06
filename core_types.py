"""
Core data types for SS Image Processor.

This module contains the fundamental data classes and enums used
throughout the application, extracted from main.py for better modularity.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional


class OutputFormat(Enum):
    """Supported output formats."""

    WEBP = "webp"
    AVIF = "avif"

    @property
    def extension(self) -> str:
        """File extension for outputs."""
        return self.value

    def animated_codec_args(self, threads: int, has_alpha: bool = True) -> List[str]:
        """ffmpeg codec args for animated outputs with optional alpha optimization."""
        if self is OutputFormat.WEBP:
            return ["-c:v", "libwebp", "-loop", "0", "-vf", "format=rgba", "-pix_fmt", "yuva420p", "-threads", str(max(1, threads)), ]
        if self is OutputFormat.AVIF:
            if has_alpha:
                return ["-c:v", "libaom-av1", "-vf", "format=rgba", "-pix_fmt", "yuva444p", "-threads", str(max(1, threads)), ]
            else:
                # Optimized non-alpha settings: aomav1_crf30_s6_420p_aqmode1
                return [
                    "-c:v", "libaom-av1", 
                    "-pix_fmt", "yuv420p",
                    "-cpu-used", "6",
                    "-crf", "30",
                    "-aq-mode", "1",
                    "-threads", str(max(1, threads))
                ]
        raise ValueError(f"Unsupported OutputFormat: {self}")

    def still_codec_args(self, has_alpha: bool = True) -> List[str]:
        """ffmpeg codec args for still-frame outputs with optional alpha optimization."""
        if self is OutputFormat.WEBP:
            return ["-vf", "format=rgba", "-c:v", "libwebp", "-pix_fmt", "yuva420p"]
        if self is OutputFormat.AVIF:
            if has_alpha:
                return ["-vf", "format=rgba", "-c:v", "libaom-av1", "-pix_fmt", "yuva444p", "-still-picture", "1", ]
            else:
                # Optimized non-alpha settings: aomav1_crf30_s6_420p_aqmode1
                return [
                    "-c:v", "libaom-av1", 
                    "-pix_fmt", "yuv420p",
                    "-cpu-used", "6", 
                    "-crf", "30",
                    "-aq-mode", "1",
                    "-still-picture", "1"
                ]
        raise ValueError(f"Unsupported OutputFormat: {self}")


class RunMode(Enum):
    """Processing mode: animated or individual frames."""

    ANIMATED = "animated"
    INDIVIDUAL = "individual"


@dataclass(frozen=True)
class Quality:
    """Encoding parameters for WebP or AVIF."""

    mode: str  # "high" | "medium" | "low" | "lossless"
    ffmpeg_args: List[str]

    @staticmethod
    def from_name(name: str, fmt: OutputFormat) -> "Quality":
        """Return a quality preset for a given output format."""
        n = name.lower().strip()
        if fmt is OutputFormat.WEBP:
            if n == "lossless":
                return Quality(mode="lossless", ffmpeg_args=["-lossless", "1", "-compression_level", "3"])
            mapping = {"high": "90", "medium": "80", "low": "70"}
            if n not in mapping:
                raise ValueError(f"Unknown quality preset for webp: {name}")
            q = mapping[n]
            return Quality(mode=n, ffmpeg_args=["-quality", q, "-compression_level", "3"])
        if fmt is OutputFormat.AVIF:
            # Use libaom-av1 settings; include -b:v 0 for CQ mode and -cpu-used for speed.
            if n == "lossless":
                return Quality(mode="lossless", ffmpeg_args=["-crf", "0", "-b:v", "0", "-cpu-used", "6"])
            mapping = {"high": "23", "medium": "30", "low": "40"}
            if n not in mapping:
                raise ValueError(f"Unknown quality preset for avif: {name}")
            crf = mapping[n]
            return Quality(mode=n, ffmpeg_args=["-crf", crf, "-b:v", "0", "-cpu-used", "6"])
        raise ValueError(f"Unknown format for quality settings: {fmt}")


@dataclass
class SequenceInfo:
    """A detected numeric frame sequence."""
    dir_path: Path
    rel_dir: Path
    prefix: str
    ext: str
    frames: List[Path]  # sorted numerically

    def __len__(self) -> int:
        return len(self.frames)


@dataclass(frozen=True)
class Config:
    """Immutable run configuration."""

    base_path: Path
    output_dir: Path
    format: OutputFormat
    quality: Quality
    run_mode: RunMode
    workers: int
    timeout_sec: int
    pad_digits: Optional[int] = None


@dataclass(frozen=True)
class AnimatedEncodeConfig:
    """Configuration for animated sequence encoding."""
    
    frame_paths: List[str]
    out_path: str
    quality_mode: str
    threads: int
    format_value: str
    crop_rect: Optional[Tuple[int, int, int, int]] = None
    offsets_json: Optional[str] = None
    seq_orig_w: Optional[int] = None
    seq_orig_h: Optional[int] = None
    has_alpha: bool = True