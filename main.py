#!/usr/bin/env python3
"""
webpseq: Convert numeric frame sequences to WebP/AVIF (animated or per-frame).

Simplified version without Rich UI dependencies.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import itertools
import os
import signal
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from simple_logger import SimpleLogger

# Misc utilities extracted to keep main focused
from misc import (
    check_tools,
    is_path_inside,
    should_skip_dir,
    parse_stem,
    validate_tiff_file,
    split_tiff_channels,
    write_ffconcat_file,
    run_subprocess,
    compute_sequence_256_crop,
    write_offset_json,
    check_alpha_exists,
    check_alpha_tiff,
    image_dimensions,
)
from dpi_utils import read_sequence_dpi, dpi_dict
from combine_metadata import combine_to_metadata

SUPPORTED_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".exr", ".dpx"}
INDIVIDUAL_SUBDIR = Path("individual_frames")
MAX_WORKER_CAP = 8
DEFAULT_TIMEOUT_SEC = 300


class OutputFormat(Enum):
    """Supported output formats."""

    WEBP = "webp"
    AVIF = "avif"

    @property
    def extension(self) -> str:
        """File extension for outputs."""
        return self.value

    def animated_codec_args(self, threads: int) -> List[str]:
        """ffmpeg codec args for animated outputs preserving alpha."""
        if self is OutputFormat.WEBP:
            return ["-c:v", "libwebp", "-loop", "0", "-vf", "format=rgba", "-pix_fmt", "yuva420p", "-threads", str(max(1, threads)), ]
        if self is OutputFormat.AVIF:
            return ["-c:v", "libaom-av1", "-vf", "format=rgba", "-pix_fmt", "yuva444p", "-threads", str(max(1, threads)), ]
        raise ValueError(f"Unsupported OutputFormat: {self}")

    def still_codec_args(self) -> List[str]:
        """ffmpeg codec args for still-frame outputs preserving alpha."""
        if self is OutputFormat.WEBP:
            return ["-vf", "format=rgba", "-c:v", "libwebp", "-pix_fmt", "yuva420p"]
        if self is OutputFormat.AVIF:
            return ["-vf", "format=rgba", "-c:v", "libaom-av1", "-pix_fmt", "yuva444p", "-still-picture", "1", ]
        raise ValueError(f"Unsupported OutputFormat: {self}")


class RunMode(Enum):
    """Processing mode: animated or individual frames."""

    ANIMATED = "animated"
    INDIVIDUAL = "individual"


class Quality(Enum):
    """Quality presets map to ffmpeg codec args."""

    LOW = "low"
    MID = "mid"
    HIGH = "high"
    MAX = "max"
    WEBP_LOSSLESS = "webp_lossless"

    @staticmethod
    def from_name(name: str, fmt: OutputFormat) -> Quality:
        """Get quality from name, with webp_lossless support."""
        if name == "webp_lossless" and fmt is OutputFormat.WEBP:
            return Quality.WEBP_LOSSLESS
        elif name == "webp_lossless":
            return Quality.MAX
        return Quality(name) if name != "webp_lossless" else Quality.MAX

    @property
    def ffmpeg_args(self) -> List[str]:
        """Return ffmpeg quality arguments for this preset."""
        if self is Quality.LOW:
            return ["-quality", "60", "-compression_level", "6"]
        elif self is Quality.MID:
            return ["-quality", "80", "-compression_level", "4"]
        elif self is Quality.HIGH:
            return ["-quality", "90", "-compression_level", "3"]
        elif self is Quality.MAX:
            return ["-quality", "100", "-compression_level", "0"]
        elif self is Quality.WEBP_LOSSLESS:
            return ["-lossless", "1", "-compression_level", "5"]
        else:
            return []


@dataclass(frozen=True)
class Config:
    """Configuration for a conversion run."""

    input_dirs: List[Path]
    output_dir: Path
    format: OutputFormat
    quality: Quality
    run_mode: RunMode
    workers: int
    timeout_sec: int
    pad_digits: Optional[int] = None
    exclude_dirs: Optional[List[Path]] = None
    auto_crop: bool = False


@dataclass(frozen=True)
class SequenceInfo:
    """Info about one numeric frame sequence."""

    dir_path: Path
    rel_dir: Path
    prefix: str
    ext: str
    frames: List[Path]

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def first(self) -> Path:
        return self.frames[0] if self.frames else Path()

    @property
    def last(self) -> Path:
        return self.frames[-1] if self.frames else Path()


def find_sequences(input_dirs: List[Path], exclude: Optional[List[Path]] = None) -> List[SequenceInfo]:
    """Find numeric frame sequences in the input directories.

    Args:
        input_dirs: Directories to search for frame sequences.
        exclude: Optional list of directories to exclude.

    Returns:
        List[SequenceInfo]: Discovered sequences sorted by directory and prefix.
    """
    sequences: List[SequenceInfo] = []
    exclude_set = set(exclude or [])

    for base in input_dirs:
        for root, dirs, files in os.walk(base, topdown=True):
            root_path = Path(root)

            # Prune unwanted directories
            if root_path in exclude_set or should_skip_dir(root_path.name):
                dirs[:] = []
                continue

            # Group files by prefix and extension
            by_prefix: Dict[Tuple[str, str], List[Path]] = {}
            for f in files:
                fp = Path(f)
                if fp.suffix.lower() not in SUPPORTED_EXTS:
                    continue
                if fp.name.startswith("."):
                    continue

                # Try parsing with and without a common separator
                parsed = parse_stem(fp.stem, separator="_")
                if parsed is None:
                    parsed = parse_stem(fp.stem, separator=".")
                if parsed is None:
                    parsed = parse_stem(fp.stem, separator=None)

                if parsed:
                    prefix, _ = parsed
                    key = (prefix, fp.suffix.lower())
                    by_prefix.setdefault(key, []).append(root_path / f)

            # Build sequences from groups
            for (prefix, ext), frames in by_prefix.items():
                if len(frames) >= 4:
                    frames.sort()
                    rel = root_path.relative_to(base) if is_path_inside(root_path, base) else Path()
                    sequences.append(
                        SequenceInfo(dir_path=root_path, rel_dir=rel, prefix=prefix, ext=ext, frames=frames, ))

    sequences.sort(key=lambda s: (s.rel_dir, s.prefix))
    return sequences


def build_ffmpeg_cmd(list_file: Path, out: Path, quality: Quality, threads: int, fmt: OutputFormat, crop_rect: Optional[Tuple[int, int, int, int]] = None, ) -> List[str]:
    """Build ffmpeg command for animated output with quality preset."""
    base = ["ffmpeg", "-f", "concat", "-safe", "0", "-i", str(list_file), "-hide_banner", "-loglevel", "error", "-nostdin", "-an", ]
    base.extend(fmt.animated_codec_args(threads))
    base.extend(quality.ffmpeg_args)

    if crop_rect is not None:
        x, y, w, h = crop_rect
        base.extend(["-vf", f"crop={w}:{h}:{x}:{y}"])

    base.append(str(out))
    return base


def encode_sequence_animated_task(frame_paths: List[str], out_path: str, quality_mode: str, threads: int, format_value: str, crop_rect: Optional[Tuple[int, int, int, int]] = None, offsets_json: Optional[str] = None, seq_orig_w: Optional[int] = None, seq_orig_h: Optional[int] = None, ) -> Tuple[bool, str, Optional[int]]:
    """
    Child-process-safe function to encode one sequence to an animated image.
    Returns (success, message, output_size_bytes|None).
    """
    try:
        fmt = OutputFormat(format_value)
        q = Quality.from_name(quality_mode, fmt)
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # Reuse if exists
        if out.exists():
            size = out.stat().st_size
            return True, f"SKIP exists: {out.name} ({size} bytes)", size

        with tempfile.TemporaryDirectory(prefix="webpseq_") as td:
            list_file = write_ffconcat_file([Path(p) for p in frame_paths], Path(td))
            cmd = build_ffmpeg_cmd(list_file, out, q, threads, fmt, crop_rect)
            # Avoid printing from subprocess; parent logs the command via returned message
            code, pretty = run_subprocess(cmd, log=False)
            if code != 0:
                return False, f"ffmpeg failed: {pretty}", None

        size = out.stat().st_size if out.exists() else None
        # Optionally write offsets JSON next to the animated output
        if offsets_json and seq_orig_w is not None and seq_orig_h is not None:
            from misc import write_offset_json as _woj
            if crop_rect is not None:
                cx, cy, cw, ch = crop_rect
            else:
                cx, cy, cw, ch = 0, 0, seq_orig_w, seq_orig_h
            _woj(Path(offsets_json), cx, cy, cw, ch, seq_orig_w, seq_orig_h)
        # Include the command string in success message for visibility in parent logs
        return True, f"DONE: {out.name} ({size} bytes) | cmd: {pretty}", size
    except Exception as ex:
        return False, f"EXC {type(ex).__name__}: {ex}", None


def build_ffmpeg_cmd_frame(src: Path, dst: Path, quality: Quality, fmt: OutputFormat, crop_rect: Optional[Tuple[int, int, int, int]] = None, ) -> List[str]:
    """Build ffmpeg command for a single-frame image with quality preset.

    Args:
        src: Input image file.
        dst: Output file path.
        quality: Quality preset to use.
        fmt: Output format.
        crop_rect: Optional crop (x, y, width, height) in pixels.

    Returns:
        List[str]: Command and arguments for subprocess.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    if fmt is OutputFormat.AVIF and src.suffix.lower() in {".tif", ".tiff"}:
        def extract_crf(args: List[str]) -> Optional[str]:
            for i, a in enumerate(args):
                if a == "-crf" and i + 1 < len(args):
                    return args[i + 1]
            return None

        color_crf = extract_crf(quality.ffmpeg_args) or "26"
        # Detect whether TIFF has an alpha channel; if not, avoid alphaextract
        try:
            has_alpha = check_alpha_tiff(src)
        except Exception:
            has_alpha = False

        if not has_alpha:
            # No alpha → encode color only, optional crop, no alphaextract
            base = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-nostdin",
                "-i", str(src),
                "-an",
                "-frames:v", "1",
                "-c:v", "libaom-av1",
                "-still-picture", "1",
                "-pix_fmt", "yuv444p12le",
                "-crf", color_crf,
            ]
            vf = None
            if vf is not None:
                base.extend(["-vf", vf])
            base.append(str(dst))
            return base
        else:
            # Has alpha → split and encode color+alpha (alpha lossless)
            crop_expr = None
            if crop_rect is not None:
                x, y, w, h = crop_rect
                crop_expr = f"[0:v]crop={w}:{h}:{x}:{y}"
            # Ensure we always have an alpha plane for extraction
            head = (crop_expr or "[0:v]") + ",format=rgba"
            filter_chain = f"{head},split=2[c][asrc];[asrc]alphaextract[a]"

            return [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-nostdin",
                "-i", str(src),
                "-filter_complex", filter_chain,
                "-map", "[c]",
                "-map", "[a]",
                "-frames:v", "1",
                "-c:v", "libaom-av1",
                "-still-picture", "1",
                "-pix_fmt:0", "yuv444p12le",
                "-crf:0", color_crf,
                "-pix_fmt:1", "gray12le",
                "-crf:1", "0",
                str(dst),
            ]

    # AVIF + PNG: preserve transparency using explicit alpha extraction
    if fmt is OutputFormat.AVIF and src.suffix.lower() in {".png"}:
        def extract_crf(args: List[str]) -> Optional[str]:
            for i, a in enumerate(args):
                if a == "-crf" and i + 1 < len(args):
                    return args[i + 1]
            return None

        color_crf = extract_crf(quality.ffmpeg_args) or "26"
        # Build filter chain: always ensure an alpha plane exists via format=rgba
        # then optionally crop, split color/alpha and extract alpha.
        chain_base = "[0:v]format=rgba"
        if crop_rect is not None:
            x, y, w, h = crop_rect
            chain_base = f"{chain_base},crop={w}:{h}:{x}:{y}"
        filter_chain = f"{chain_base},split=2[c][asrc];[asrc]alphaextract[a]"

        return [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-nostdin",
            "-i", str(src),
            "-filter_complex", filter_chain,
            "-map", "[c]",
            "-map", "[a]",
            "-frames:v", "1",
            "-c:v", "libaom-av1",
            "-still-picture", "1",
            # Use 8-bit for PNG inputs to avoid unnecessary bit-depth inflation
            "-pix_fmt:0", "yuv444p",
            "-crf:0", color_crf,
            "-pix_fmt:1", "gray",
            "-crf:1", "0",
            str(dst),
        ]

    # Default case for WebP/other
    base = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(src), "-an", ]
    base.extend(fmt.still_codec_args())
    base.extend(quality.ffmpeg_args)

    if crop_rect is not None:
        x, y, w, h = crop_rect
        base.extend(["-vf", f"crop={w}:{h}:{x}:{y}"])

    base.append(str(dst))
    return base


def encode_one_frame(src: Path, dst: Path, quality: Quality, fmt: OutputFormat, crop_rect: Optional[Tuple[int, int, int, int]] = None, orig_width: Optional[int] = None, orig_height: Optional[int] = None, ) -> Tuple[bool, str]:
    """
    Encode one frame to output format, returning (success, message).
    """
    try:
        # Reuse if exists
        if dst.exists():
            size = dst.stat().st_size
            return True, f"SKIP {dst.name} exists ({size} bytes)"

        cmd = build_ffmpeg_cmd_frame(src, dst, quality, fmt, crop_rect)
        code, pretty = run_subprocess(cmd, log=True)
        if code != 0:
            return False, f"FAIL {dst.name} | cmd: {pretty}"
        
        size = dst.stat().st_size if dst.exists() else 0
        return True, f"ok {dst.name}"
    except Exception as ex:
        return False, f"EXC {type(ex).__name__}: {ex}"


def encode_sequence_individual(seq: SequenceInfo, out_root: Path, quality: Quality, threads: int, timeout_sec: int, fmt: OutputFormat, pad_digits: Optional[int] = None, logger: Optional[SimpleLogger] = None, ) -> Tuple[bool, str, int]:
    """
    Encode a sequence to individual frames, optionally with cropping.
    Returns (success, message, files_produced).
    """
    rel_dir = out_root / INDIVIDUAL_SUBDIR / seq.rel_dir
    # Use padding if specified; otherwise dynamic
    tasks: List[Tuple[Path, Path]] = []
    existing = 0
    missing = 0

    if seq.frames and fmt is OutputFormat.AVIF:
        cx, cy, cw, ch, ow, oh = compute_sequence_256_crop(seq.frames)
        if cx is not None:
            crop_tuple: Optional[Tuple[int, int, int, int]] = (cx, cy, cw, ch)
        else:
            crop_tuple = None
            ow = oh = None
    else:
        crop_tuple = None
        ow = oh = None

    for i, src in enumerate(seq.frames):
        parsed = parse_stem(src.stem, separator="_") or parse_stem(src.stem, separator=".") or parse_stem(src.stem, separator=None)
        if parsed:
            _, frame_number = parsed
            if pad_digits:
                out_name = f"{frame_number:0{pad_digits}d}.{fmt.extension}"
            else:
                out_name = f"{frame_number}.{fmt.extension}"
        else:
            out_name = f"{src.stem}.{fmt.extension}"

        dst = rel_dir / out_name
        if dst.exists():
            existing += 1
        else:
            missing += 1
            tasks.append((src, dst))

    # Early return if all exist
    if not tasks:
        return True, f"ALL {existing} frames already exist", 0

    ok = True
    start = time.time()
    produced = 0
    msgs: List[str] = []

    def work(pair: Tuple[Path, Path]) -> Tuple[bool, str]:
        src, dst = pair
        return encode_one_frame(src, dst, quality, fmt, crop_tuple, ow, oh)

    with futures.ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
        futs = [pool.submit(work, t) for t in tasks]
        done, not_done = futures.wait(futs, timeout=timeout_sec)
        if not_done:
            for f in not_done:
                f.cancel()
            ok = False
            msgs.append(f"TIMEOUT after {timeout_sec}s: {len(not_done)} tasks not finished")

        # collect results from completed
        for f in done:
            try:
                success, m = f.result()
                ok = ok and success
                if success:
                    produced += 1
                msgs.append(m)
            except Exception as ex:
                ok = False
                msgs.append(f"EXC {type(ex).__name__}: {ex}")

    elapsed = time.time() - start
    # Write one offsets JSON and one DPI JSON for the entire sequence
    rel_dir = out_root / INDIVIDUAL_SUBDIR / seq.rel_dir
    base_name_str = seq.prefix.strip() or seq.dir_path.name
    base_name = base_name_str.rstrip("._-")
    json_path = rel_dir / f"{base_name}_{seq.ext.lstrip('.')}.json"
    try:
        write_offset_json(json_path, cx, cy, cw, ch, ow, oh)
    except Exception:
        pass
    # DPI sidecar written once per sequence
    try:
        dpi_info = read_sequence_dpi(seq.frames)
        from misc import write_dpi_json as _wdj
        dpi_sidecar = rel_dir / f"{base_name}_{seq.ext.lstrip('.')}.dpi.json"
        _wdj(dpi_sidecar, dpi_dict(dpi_info))
    except Exception:
        pass
    # Combine sidecars into metadata.json per sequence
    try:
        combine_to_metadata(rel_dir, f"{base_name}_{seq.ext.lstrip('.')}", output_name="metadata.json")
    except Exception:
        pass
    return ok, f"{produced}/{missing} created in {elapsed:.1f}s; " + " | ".join(itertools.islice(msgs, 0, 8)), produced


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="webpseq",
        description="Convert numeric frame sequences to WebP/AVIF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Required positional OR -p for base path (for backward compatibility)
    parser.add_argument(
        "input_dirs", 
        nargs="*",  # Made optional to support -p
        type=Path, 
        help="Input directory/directories to scan for sequences"
    )
    parser.add_argument(
        "-p", "--base-path",
        type=Path,
        default=Path("."),
        help="Base path to scan (alternative to positional argument)"
    )
    
    # Output directory
    parser.add_argument(
        "-o", "--output", "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output directory"
    )
    
    # Format
    parser.add_argument(
        "-f", "--format",
        choices=["webp", "avif"],
        default="webp",
        help="Output format"
    )
    
    # Quality
    parser.add_argument(
        "-q", "--quality",
        choices=["low", "medium", "mid", "high", "max", "lossless", "webp_lossless"],
        default="high",
        help="Quality preset"
    )
    
    # Mode - support both -m and -i flags
    parser.add_argument(
        "-m", "--mode",
        choices=["animated", "individual"],
        default="animated",
        help="animated = one animated image per sequence, individual = separate images"
    )
    parser.add_argument(
        "-i", "--individual-frames",
        action="store_true",
        help="Export individual files per frame instead of animated (same as -m individual)"
    )
    
    # Workers
    parser.add_argument(
        "-j", "--jobs", "-w", "--max-workers",
        type=int,
        default=4,
        dest="jobs",
        help=f"Worker threads (max: {MAX_WORKER_CAP})"
    )
    
    # Other options
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SEC,
        help=f"Timeout per sequence in seconds"
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        type=Path,
        help="Directories to exclude from scan"
    )
    parser.add_argument(
        "--pad-digits",
        type=int,
        help="Pad output frame numbers to N digits (e.g., 3 -> 001)"
    )
    parser.add_argument(
        "--auto-crop",
        action="store_true",
        help="Auto-crop AVIF frames to 256x256 based on alpha edges"
    )
    parser.add_argument(
        "-s", "--sequential",
        action="store_true",
        help="Force sequential processing (workers=1)"
    )

    args = parser.parse_args()
    
    # Handle backward compatibility
    if not args.input_dirs:
        args.input_dirs = [args.base_path]
    
    # Handle quality name mapping
    if args.quality == "medium":
        args.quality = "mid"
    elif args.quality == "lossless":
        args.quality = "webp_lossless"
    
    # Handle individual frames flag
    if args.individual_frames:
        args.mode = "individual"
    
    # Handle sequential flag
    if args.sequential:
        args.jobs = 1

    # Setup logger
    output_dir = args.output if hasattr(args, 'output') else args.output_dir
    log_file = output_dir / "processing.log"
    logger = SimpleLogger(log_file)
    
    logger.section("WEBPSEQ CONVERTER")
    logger.info(f"Starting conversion run at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Validate inputs
    ok, probs = check_tools()
    if not ok:
        for p in probs:
            logger.error(f"Missing: {p}")
        return 1

    fmt = OutputFormat(args.format)
    quality = Quality.from_name(args.quality, fmt)
    mode = RunMode(args.mode)
    workers = min(args.jobs, MAX_WORKER_CAP)
    
    config = Config(
        input_dirs=args.input_dirs,
        output_dir=output_dir,
        format=fmt,
        quality=quality,
        run_mode=mode,
        workers=workers,
        timeout_sec=args.timeout,
        pad_digits=args.pad_digits,
        exclude_dirs=args.exclude,
        auto_crop=args.auto_crop,
    )
    
    # Check output safety
    safe = True
    reason = ""
    for input_dir in config.input_dirs:
        if is_path_inside(config.output_dir.resolve(), input_dir.resolve()):
            safe = False
            reason = f"Output '{config.output_dir}' is inside input '{input_dir}'"
            break
    
    if not safe:
        logger.error(f"Unsafe output location: {reason}")
        return 1
    
    # Find sequences
    logger.info("Scanning for frame sequences...")
    sequences = find_sequences(config.input_dirs, config.exclude_dirs)
    
    if not sequences:
        logger.warning("No numeric sequences (>=4 frames) found. Nothing to do.")
        return 0
    
    # Display found sequences
    logger.section(f"Found {len(sequences)} sequences")
    headers = ["#", "Path", "Frames", "Extension"]
    rows = []
    for i, seq in enumerate(sequences, 1):
        display_path = (seq.rel_dir.as_posix() or ".") + f"/{seq.prefix}*"
        rows.append([str(i).zfill(2), display_path, str(len(seq)), seq.ext])
    logger.table(headers, rows)
    
    # Process sequences
    logger.section("Processing")
    successes = 0
    failures = 0
    start_time = time.time()
    
    if config.run_mode is RunMode.INDIVIDUAL:
        # Individual frames mode
        for i, seq in enumerate(sequences, 1):
            display_path = (seq.rel_dir.as_posix() or ".") + f"/{seq.prefix}*"
            logger.progress(i, len(sequences), f"INDIV {display_path}{seq.ext} ({len(seq)} frames)")
            
            ok, msg, produced = encode_sequence_individual(
                seq=seq,
                out_root=config.output_dir,
                quality=config.quality,
                threads=config.workers,
                timeout_sec=config.timeout_sec,
                fmt=config.format,
                pad_digits=config.pad_digits,
                logger=logger,
            )
            
            if ok:
                successes += 1
                logger.success(f"    -> {msg}")
            else:
                failures += 1
                logger.error(f"    -> {msg}")
    
    else:
        # Animated mode
        with futures.ProcessPoolExecutor(max_workers=config.workers) as executor:
            future_to_seq: Dict[futures.Future, Tuple[int, SequenceInfo, Path]] = {}
            
            for i, seq in enumerate(sequences, 1):
                out_path = config.output_dir / seq.rel_dir / f"{seq.prefix or seq.dir_path.name}.{config.format.extension}"
                display_path = (seq.rel_dir.as_posix() or ".") + f"/{seq.prefix}*"
                
                logger.info(f"[{i}/{len(sequences)}] Submitting ANIM {display_path}{seq.ext} ({len(seq)} frames)")
                
                frame_paths = [str(f) for f in seq.frames]
                offsets_json_path = out_path.with_suffix(".json")
                orig_w, orig_h = (None, None)
                crop_rect = None
                
                if config.auto_crop and config.format is OutputFormat.AVIF and seq.frames:
                    cx, cy, cw, ch, orig_w, orig_h = compute_sequence_256_crop(seq.frames)
                    if cx is not None:
                        crop_rect = (cx, cy, cw, ch)
                
                fut = executor.submit(
                    encode_sequence_animated_task,
                    frame_paths,
                    str(out_path),
                    config.quality.value,
                    config.workers,
                    config.format.value,
                    crop_rect,
                    str(offsets_json_path) if crop_rect else None,
                    orig_w,
                    orig_h,
                )
                future_to_seq[fut] = (i, seq, out_path)
            
            # Wait for results
            for fut in futures.as_completed(future_to_seq, timeout=None):
                i, seq, out_path = future_to_seq[fut]
                display_path = (seq.rel_dir.as_posix() or ".") + f"/{seq.prefix}*"
                
                try:
                    fut_result = fut.result(timeout=config.timeout_sec)
                    if fut_result is None:
                        failures += 1
                        logger.error(f"[{i}/{len(sequences)}] ANIM {display_path}{seq.ext} -> None result")
                        continue
                    
                    ok, msg, size = fut_result
                    if ok:
                        successes += 1
                        logger.success(f"[{i}/{len(sequences)}] ANIM {display_path}{seq.ext} -> {out_path.name}")
                        logger.info(f"    -> {msg}")
                        if size is not None:
                            logger.info(f"    size: {size:,} bytes")
                        
                        # Combine metadata
                        try:
                            combine_to_metadata(out_path.parent, out_path.name, output_name="metadata.json")
                            logger.success("    -> wrote metadata.json")
                        except Exception as ex:
                            logger.warning(f"    -> metadata combine failed: {type(ex).__name__}: {ex}")
                    else:
                        failures += 1
                        logger.error(f"[{i}/{len(sequences)}] ANIM {display_path}{seq.ext} -> {out_path.name}")
                        logger.error(f"    -> {msg}")
                        
                except futures.TimeoutError:
                    failures += 1
                    fut.cancel()
                    logger.error(f"[{i}/{len(sequences)}] TIMEOUT after {config.timeout_sec}s for {display_path}{seq.ext}")
                except Exception as ex:
                    failures += 1
                    logger.error(f"[{i}/{len(sequences)}] ERROR on {display_path}{seq.ext}: {type(ex).__name__}: {ex}")
    
    # Summary
    total_time = time.time() - start_time
    total_sequences = len(sequences)
    avg = (total_time / max(1, total_sequences)) if total_sequences else 0.0
    
    logger.section("Summary")
    logger.info(f"Total sequences: {total_sequences}")
    logger.info(f"Successful: {successes}")
    logger.info(f"Failed: {failures}")
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info(f"Average per sequence: {avg:.1f}s")
    logger.info(f"Output directory: {config.output_dir}")
    
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())