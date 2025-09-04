#!/usr/bin/env python3
"""
webpseq: Convert numeric frame sequences to WebP/AVIF (animated or per-frame).

This refactor improves readability and separations of concerns by:
- Introducing enums for output format and run mode
- Using a Config dataclass to pass settings
- Splitting logic into small, single-purpose functions
- Keeping I/O at the edges and core logic pure
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
import sys as _sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from persistent_output import install_persistent_console
from log_viewer import ScrollableLogViewer

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

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

# Ensure progress-style carriage-return writes are persisted as lines
install_persistent_console()

console = Console()
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


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
            prog="webpseq", description="Convert numeric frame sequences to WebP/AVIF (animated or per-frame).", formatter_class=argparse.ArgumentDefaultsHelpFormatter, )
    p.add_argument("-p", "--base-path", type=Path, default=Path("."), help="Base path to scan")
    p.add_argument("-o", "--output-dir", type=Path, help="Output directory. Defaults to 'outputs/{format}_renders'")
    p.add_argument("-q", "--quality", choices=["high", "medium", "low", "lossless"], default="high", help="Quality preset")
    p.add_argument("--format", choices=[f.value for f in OutputFormat], default=OutputFormat.WEBP.value, help="Output format")
    p.add_argument("-i", "--individual-frames", action="store_true", help="Export individual files per frame instead of animated")
    p.add_argument(
            "--pad-digits", type=int, default=None, help=("Zero-padding width for per-frame filenames. "
                                                          "When set, per-sequence numbering starts at 1 and outputs are named 00001, 00002, ... in each folder. "
                                                          "When unset, the input filename is used."), )
    p.add_argument("-w", "--max-workers", type=int, help="Max parallel workers (capped at 8)")
    p.add_argument("-s", "--sequential", action="store_true", help="Force sequential processing")
    p.add_argument("--check-tools", action="store_true", help="Verify external tools and exit")
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> Config:
    """Create a Config object from parsed args."""
    fmt = OutputFormat(args.format)
    quality = Quality.from_name(args.quality, fmt)
    workers = pick_worker_count(args.max_workers, args.sequential)
    base_path = args.base_path
    output_dir = args.output_dir or Path("outputs") / f"{fmt.value}_renders"
    run_mode = RunMode.INDIVIDUAL if args.individual_frames else RunMode.ANIMATED
    return Config(
            base_path=base_path, output_dir=output_dir, format=fmt, quality=quality, run_mode=run_mode, workers=workers, timeout_sec=DEFAULT_TIMEOUT_SEC, pad_digits=args.pad_digits, )


# ------------------------------
# Environment and safety checks
# ------------------------------

def pick_worker_count(requested: Optional[int], sequential: bool) -> int:
    """Determine worker count within the cap."""
    if sequential:
        return 1
    import multiprocessing

    cores = max(1, multiprocessing.cpu_count())
    limit = min(cores, MAX_WORKER_CAP)
    if requested is None:
        return limit
    return max(1, min(requested, MAX_WORKER_CAP))


def is_safe_output_location(base_path: Path, output_dir: Path) -> Tuple[bool, str]:
    """
    Allow output inside the source tree ONLY if under a folder literally named 'outputs'.
    Also reject cases where base lives inside output (overlap).
    """
    base = base_path.resolve()
    out = output_dir.resolve()

    # base inside out → unsafe (scans would traverse outputs)
    if is_path_inside(base, out):
        return False, f"Output directory {out} contains base path {base} (unsafe overlap)."

    # out inside base → must be under 'outputs'
    if is_path_inside(out, base):
        rel = out.relative_to(base)
        if "outputs" not in rel.parts:
            return False, f"Output directory {out} is inside base path {base} but not under an 'outputs' folder."
    return True, ""




def find_sequences(base_path: Path) -> List[SequenceInfo]:
    """
    Walk base_path and detect frame sequences.
    A sequence is a series of files in the same directory, with the same extension,
    and filenames that end in a number (e.g., "render_001.png", "render_002.png").
    Sequences must have at least 4 frames.
    """
    sequences: List[SequenceInfo] = []
    base = base_path.resolve()

    for dirpath, dirnames, filenames in os.walk(base):
        # prune directories in-place
        dirnames[:] = [d for d in dirnames if not should_skip_dir(d)]
        dpath = Path(dirpath)

        # Group files by (prefix, extension) in this directory
        potential_sequences: Dict[Tuple[str, str], List[Tuple[int, Path]]] = {}
        for f_name in filenames:
            f_path = dpath / f_name
            ext = f_path.suffix.lower()
            if ext not in SUPPORTED_EXTS:
                continue

            parsed = parse_stem(f_path.stem)
            if parsed:
                prefix, frame_num = parsed
                potential_sequences.setdefault((prefix, ext), []).append((frame_num, f_path))

        # Filter for actual sequences (>= 4 frames) and create SequenceInfo
        for (prefix, ext), frames_with_nums in potential_sequences.items():
            if len(frames_with_nums) < 4:
                continue

            # Sort by frame number and extract just the paths
            frames_with_nums.sort(key=lambda item: item[0])
            sorted_frames = [p for _, p in frames_with_nums]

            rel_dir = dpath.relative_to(base)
            sequences.append(
                    SequenceInfo(
                            dir_path=dpath, rel_dir=rel_dir, prefix=prefix, ext=ext, frames=sorted_frames, )
            )

    return sequences


# ------------------------------
# Output path resolution
# ------------------------------

def animated_output_path(output_root: Path, seq: SequenceInfo, fmt: OutputFormat) -> Path:
    """
    Place one output file per sequence under output_root mirroring the relative directory.
    """
    # e.g., outputs/webp_renders/<rel_dir>/<prefix_or_dirname>_<ext>.webp
    rel_target_dir = output_root / seq.rel_dir
    base_name_str = seq.prefix.strip() or seq.dir_path.name
    base_name = base_name_str.rstrip("._-")
    basename = f"{base_name}_{seq.ext.lstrip('.')}.{fmt.extension}"
    return rel_target_dir / basename


def individual_output_path(output_root: Path, seq: SequenceInfo, frame: Path, fmt: OutputFormat, frame_index: Optional[int] = None, pad_digits: Optional[int] = None, ) -> Path:
    """
    Place per-frame outputs under:
      output_root/individual_frames/<rel_dir>/<frame_stem>.<format>
    """
    rel_dir = output_root / INDIVIDUAL_SUBDIR / seq.rel_dir
    if pad_digits and frame_index is not None:
        name = f"{frame_index:0{pad_digits}d}.{fmt.extension}"
    else:
        name = f"{frame.stem}.{fmt.extension}"
    return rel_dir / name


def build_ffmpeg_cmd(
    list_file: Path,
    out_path: Path,
    quality: Quality,
    threads: int,
    fmt: OutputFormat,
    crop_rect: Optional[Tuple[int, int, int, int]] = None,
) -> List[str]:
    """Create ffmpeg command for an animated image sequence, with optional crop."""
    base = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-an",
        "-vsync", "0",
        "-r", "30",
    ]
    codec_args = fmt.animated_codec_args(threads)
    if crop_rect is not None:
        args = codec_args.copy()
        if "-vf" in args:
            i = args.index("-vf")
            if i + 1 < len(args):
                existing = args[i + 1]
                x, y, w, h = crop_rect
                crop = f"crop={w}:{h}:{x}:{y}"
                args[i + 1] = f"{crop},{existing}"
                codec_args = args
    return base + codec_args + quality.ffmpeg_args + [str(out_path)]


# ------------------------------
# Animated pipeline (process-level parallelism)
# ------------------------------

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
                cx = cy = 0
                cw = seq_orig_w
                ch = seq_orig_h
            _woj(Path(offsets_json), cx, cy, cw, ch, seq_orig_w, seq_orig_h)
        # Include the command string in success message for visibility in parent logs
        return True, f"DONE: {out.name} ({size} bytes) | cmd: {pretty}", size
    except Exception as ex:
        return False, f"Exception: {type(ex).__name__}: {ex}", None


# ------------------------------
# Individual-frames pipeline (thread-level parallelism)
# ------------------------------

def build_ffmpeg_individual_cmd(
    src: Path,
    dst: Path,
    quality: Quality,
    fmt: OutputFormat,
    crop_rect: Optional[Tuple[int, int, int, int]] = None,
) -> List[str]:
    """Create ffmpeg command for a single frame.

    For AVIF+TIFF, use a tested filter_complex that separates color and alpha,
    encoding color with the configured CRF and alpha losslessly.
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
            # if crop_rect is not None:
            #     x, y, w, h = crop_rect
            #     vf = f"crop={w}:{h}:{x}:{y}"
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
            # Keep alpha lossless to preserve edges cleanly
            "-crf:1", "0",
            str(dst),
        ]

    base = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-i", str(src),
        "-an",
    ]
    codec_args = fmt.still_codec_args()
    # Merge crop filter with existing -vf if needed
    if crop_rect is not None:
        # codec_args like ['-vf','format=rgba', ...]; merge into format chain
        args = codec_args.copy()
        if "-vf" in args:
            i = args.index("-vf")
            if i + 1 < len(args):
                existing = args[i + 1]
                x, y, w, h = crop_rect
                crop = f"crop={w}:{h}:{x}:{y}"
                args[i + 1] = f"{crop},{existing}"
                codec_args = args
    return base + codec_args + quality.ffmpeg_args + [str(dst)]


def encode_one_frame(
    src: Path,
    dst: Path,
    quality: Quality,
    fmt: OutputFormat,
    crop_rect: Optional[Tuple[int, int, int, int]],
    seq_orig_w: int,
    seq_orig_h: int,
) -> Tuple[bool, str]:
    """Encode one frame with an optional sequence-wide crop and write offsets JSON."""
    if dst.exists():
        return True, f"skip {dst.name}"

    temp_src_file: Optional[Path] = None
    source_to_encode = src
    message_suffix = ""

    try:
        # TIFF handling:
        # - For AVIF we can feed TIFF directly to ffmpeg; alpha is preserved via pix_fmt yuva444p.
        # - For WEBP, some TIFFs with >=4 channels can be problematic; convert to PNG as a safe fallback.
        if src.suffix.lower() in {".tif", ".tiff"} and fmt is OutputFormat.WEBP:
            is_valid, reason = validate_tiff_file(src)
            if not is_valid:
                temp_dir = dst.parent / f"temp_{dst.stem}"
                success, temp_png_path, split_msg = split_tiff_channels(src, temp_dir)

                if success and temp_png_path:
                    source_to_encode = temp_png_path
                    temp_src_file = temp_png_path
                    message_suffix = " (re-encoded via PNG)"
                else:
                    return False, f"FAIL: TIFF processing failed: {split_msg}"

        # Encode with ffmpeg
        cmd = build_ffmpeg_individual_cmd(source_to_encode, dst, quality, fmt, crop_rect)
        code, pretty = run_subprocess(cmd)

        if code != 0:
            return False, f"FAIL({code}) {pretty}"

        return True, f"ok {dst.name}{message_suffix} | cmd: {pretty}"
    finally:
        # Clean up temporary file if it was created
        if temp_src_file:
            try:
                parent_dir = temp_src_file.parent
                temp_src_file.unlink()
                parent_dir.rmdir()
            except OSError:
                pass


def encode_sequence_individual(seq: SequenceInfo, out_root: Path, quality: Quality, threads: int, timeout_sec: int, fmt: OutputFormat, pad_digits: Optional[int] = None, ) -> Tuple[bool, str, int]:
    """
    Convert one sequence to individual frames using thread pool.
    Returns (success, message, produced_count).
    """
    # Decide crop based on alpha channel presence in the first frame only
    first = seq.frames[0]
    if not check_alpha_exists(first):
        # No alpha channel → no need to crop by transparency
        ow, oh = image_dimensions(first)
        cx = cy = 0
        cw, ch = ow, oh
        crop_tuple = None
    else:
        # Compute sequence-wide crop once (left→right, top→bottom; true wins)
        cx, cy, cw, ch, ow, oh = compute_sequence_256_crop(seq.frames)
        crop_tuple = None
        if not (cx == 0 and cy == 0 and cw == ow and ch == oh):
            crop_tuple = (cx, cy, cw, ch)

    # Build tasks for missing outputs
    tasks: List[Tuple[Path, Path]] = []
    for idx, f in enumerate(seq.frames, start=1):
        dst = individual_output_path(out_root, seq, f, fmt, frame_index=idx, pad_digits=pad_digits)
        if not dst.exists():
            tasks.append((f, dst))

    total = len(seq.frames)
    missing = len(tasks)
    if missing == 0:
        return True, f"SKIP all {total} frames exist", 0

    produced = 0
    start = time.time()
    ok = True
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

class GracefulExit(Exception):
    """Signal-initiated shutdown."""


def install_signal_handlers(stop_event: threading.Event) -> None:
    """Handle Ctrl+C gracefully by setting an event and raising in main thread."""

    def handler(signum, frame):
        stop_event.set()
        console.print("\n[yellow]Ctrl+C received. Attempting graceful shutdown...[/]", file=sys.stderr)
        raise GracefulExit()

    signal.signal(signal.SIGINT, handler)


def start_controls_listener(
    pause_event: threading.Event, 
    stop_event: threading.Event,
    log_viewer: Optional[ScrollableLogViewer] = None
) -> threading.Thread:
    """Start a background thread to listen for simple controls from stdin.

    Controls:
      - 'p' + Enter: toggle pause/resume
      - 'q' + Enter: request quit (graceful)
      - 'u' + Enter: scroll up in log history
      - 'd' + Enter: scroll down in log history
      - 't' + Enter: scroll to top of log history
      - 'b' + Enter: scroll to bottom of log history

    Returns:
        Thread: The daemon thread handling input.
    """
    import os as _os

    def _reader() -> None:
        if not _sys.stdin:
            return
        try:
            is_tty = _sys.stdin.isatty()
        except Exception:
            is_tty = False
        # Use line-buffered input to avoid raw mode complexity
        while not stop_event.is_set():
            try:
                line = _sys.stdin.readline()
                if not line:
                    # EOF or closed stdin
                    break
                cmd = line.strip().lower()
                if cmd == "p":
                    if pause_event.is_set():
                        pause_event.clear()
                    else:
                        pause_event.set()
                elif cmd == "q":
                    stop_event.set()
                    try:
                        # Trigger the SIGINT handler for consistent shutdown path
                        _os.kill(_os.getpid(), signal.SIGINT)
                    except Exception:
                        pass
                elif log_viewer:
                    # Scrolling controls
                    if cmd == "u":
                        log_viewer.scroll_up(5)
                    elif cmd == "d":
                        log_viewer.scroll_down(5)
                    elif cmd == "t":
                        log_viewer.scroll_to_top()
                    elif cmd == "b":
                        log_viewer.scroll_to_bottom()
            except Exception:
                break

    t = threading.Thread(target=_reader, name="controls-listener", daemon=True)
    t.start()
    return t


def create_run_header(config: Config, safe_msg: str) -> Panel:
    """Build the header panel describing the run configuration."""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="dim")
    table.add_column()
    table.add_row("Source:", str(config.base_path.resolve()))
    table.add_row("Output:", str(config.output_dir.resolve()))
    table.add_row("Safety:", safe_msg or "ok")
    table.add_row("Format:", config.format.value.upper())
    table.add_row("Mode:", "individual-frames" if config.run_mode is RunMode.INDIVIDUAL else "animated")
    table.add_row("Quality:", config.quality.mode)
    table.add_row("Workers:", f"{config.workers} (cap {MAX_WORKER_CAP})")
    return Panel(table, title="[bold cyan]Run Configuration[/bold cyan]", border_style="cyan", title_align="left")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)

    if args.check_tools:
        ok, probs = check_tools()
        if ok:
            console.print("[bold green]Tools OK:[/] ffmpeg")
            return 0
        for p in probs:
            console.print(f"[bold red]Missing:[/] {p}", file=sys.stderr)
        return 1

    config = build_config(args)

    safe, reason = is_safe_output_location(config.base_path, config.output_dir)
    if not safe:
        console.print(f"[bold red]Unsafe output location:[/] {reason}", file=sys.stderr)
        console.print("No files were written.", file=sys.stderr)
        return 1

    tools_ok, probs = check_tools()
    if not tools_ok:
        for p in probs:
            console.print(f"[bold red]Missing:[/] {p}", file=sys.stderr)
        return 1

    header = create_run_header(config, "")

    t0 = time.time()
    sequences = find_sequences(config.base_path)
    if not sequences:
        console.print("[yellow]No numeric sequences (>=4 frames) found. Nothing to do.[/]")
        return 0

    sequences_table = Table(title=f"Discovered {len(sequences)} sequences", show_header=True, header_style="bold magenta")
    sequences_table.add_column("#", style="dim", width=4, justify="right")
    sequences_table.add_column("Path")
    sequences_table.add_column("Frames", justify="right")
    sequences_table.add_column("Extension")
    for idx, seq in enumerate(sequences, 1):
        display_path = (seq.rel_dir.as_posix() or ".") + f"/{seq.prefix}*"
        sequences_table.add_row(f"{idx:02d}", display_path, str(len(seq)), seq.ext)

    stop_ev = threading.Event()
    pause_ev = threading.Event()
    install_signal_handlers(stop_ev)

    successes = 0
    failures = 0
    produced_total = 0
    perseq_times: List[float] = []

    layout = Layout()
    layout.split_column(
            Layout(header, name="header", size=8), Layout(name="main", ratio=1), Layout(name="footer", size=3), )

    # Create scrollable log viewer with file persistence
    log_file_path = config.output_dir / "processing.log"
    log_viewer = ScrollableLogViewer(
        max_visible_lines=25,  # Adjust based on typical terminal height
        max_history=5000,
        log_file=log_file_path
    )
    
    # Add initial sequences table to log
    from rich.console import Console
    temp_console = Console(file=None, force_terminal=False)
    with temp_console.capture() as capture:
        temp_console.print(sequences_table)
    table_output = capture.get()
    for line in table_output.split('\n'):
        if line.strip():
            log_viewer.add_log(line)
    
    # Create a renderable that always shows the current log panel
    class LogPanel:
        def __rich__(self):
            return log_viewer.get_panel(title="Log History")
    
    layout["main"].update(LogPanel())

    def get_footer() -> Panel:
        total = successes + failures
        elapsed = time.time() - t0
        state = "[yellow]PAUSED[/]" if pause_ev.is_set() else "running"
        controls = "[dim]Controls: 'p' pause/resume | 'q' quit | 'u' scroll up | 'd' scroll down | 't' top | 'b' bottom[/dim]"
        return Panel(
                f"Processed: {total}/{len(sequences)} | "
                f"Success: [green]{successes}[/] | "
                f"Failed: [red]{failures}[/] | "
                f"Elapsed: {elapsed:.1f}s | State: {state}\n{controls}",
                title="[cyan]Status[/]",
                border_style="cyan",
        )

    layout["footer"].update(get_footer())

    try:
        # Use main screen buffer so all logs remain visible in the terminal
        with Live(layout, screen=False, redirect_stderr=False, refresh_per_second=2) as live:
            # Start background listener for pause/quit controls
            _ = start_controls_listener(pause_ev, stop_ev, log_viewer)
            if config.run_mode is RunMode.INDIVIDUAL:
                # Per-sequence sequential vs threaded per-frame inside
                for i, seq in enumerate(sequences, 1):

                    if stop_ev.is_set():
                        break

                    # Pause between sequences if requested
                    while pause_ev.is_set() and not stop_ev.is_set():
                        layout["footer"].update(get_footer())
                        time.sleep(0.2)

                    display_path = (seq.rel_dir.as_posix() or ".") + f"/{seq.prefix}*"
                    log_viewer.add_log(f"[{i:02d}/{len(sequences)}] [bold]INDIV[/] {display_path}{seq.ext} ({len(seq)} frames)")

                    ok, msg, produced = encode_sequence_individual(
                            seq=seq, out_root=config.output_dir, quality=config.quality, threads=config.workers, timeout_sec=config.timeout_sec, fmt=config.format, pad_digits=config.pad_digits, )
                    produced_total += produced
                    log_viewer.add_log(f"    -> {msg}", style="green" if ok else "red")

                    if ok:
                        successes += 1
                    else:
                        failures += 1
                    layout["footer"].update(get_footer())
            else:
                # Animated: process pool across sequences
                with futures.ProcessPoolExecutor(max_workers=config.workers) as pool:
                    fut_map: Dict[futures.Future, Tuple[int, SequenceInfo, Path]] = {}
                    for i, seq in enumerate(sequences, 1):
                        if stop_ev.is_set():
                            break
                        while pause_ev.is_set() and not stop_ev.is_set():
                            layout["footer"].update(get_footer())
                            time.sleep(0.2)
                        out_path = animated_output_path(config.output_dir, seq, config.format)
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        # Read DPI once per sequence and write sidecar JSON next to the animated output target
                        try:
                            dpi_info = read_sequence_dpi(seq.frames)
                            from misc import write_dpi_json as _wdj
                            _wdj(out_path.with_suffix(out_path.suffix + ".dpi.json"), dpi_dict(dpi_info))
                        except Exception:
                            pass
                        frame_paths = [p.as_posix() for p in seq.frames]
                        # Skip crop when the first frame has no alpha channel
                        if not check_alpha_exists(seq.frames[0]):
                            ow, oh = image_dimensions(seq.frames[0])
                            crop_tuple = None
                            cx = cy = 0
                            cw, ch = ow, oh
                        else:
                            # Compute sequence-wide crop
                            cx, cy, cw, ch, ow, oh = compute_sequence_256_crop(seq.frames)
                            crop_tuple: Optional[Tuple[int, int, int, int]] = None
                            if not (cx == 0 and cy == 0 and cw == ow and ch == oh):
                                crop_tuple = (cx, cy, cw, ch)
                        offsets_path = out_path.with_suffix(out_path.suffix + ".json")
                        fut = pool.submit(
                            encode_sequence_animated_task,
                            frame_paths,
                            out_path.as_posix(),
                            config.quality.mode,
                            max(1, config.workers),
                            config.format.value,
                            crop_tuple,
                            offsets_path.as_posix(),
                            ow,
                            oh,
                        )
                        fut_map[fut] = (i, seq, out_path)

                    for fut, (i, seq, out_path) in fut_map.items():
                        if stop_ev.is_set():
                            try:
                                pool.shutdown(cancel_futures=True)
                            except Exception:
                                pass
                            break
                        while pause_ev.is_set() and not stop_ev.is_set():
                            layout["footer"].update(get_footer())
                            time.sleep(0.2)
                        display_path = (seq.rel_dir.as_posix() or ".") + f"/{seq.prefix}*"
                        try:
                            ok, msg, size = fut.result(timeout=config.timeout_sec)
                            if ok:
                                successes += 1
                            else:
                                failures += 1
                            log_viewer.add_log(f"[{i:02d}/{len(sequences)}] [bold]ANIM [/] {display_path}{seq.ext} -> {out_path.name}")
                            log_viewer.add_log(f"    -> {msg}", style="green" if ok else "red")
                            if ok and size is not None:
                                log_viewer.add_log(f"    size: [green]{size:,}[/] bytes")
                            # After successful encode, combine sidecars into metadata.json
                            if ok:
                                try:
                                    combine_to_metadata(out_path.parent, out_path.name, output_name="metadata.json")
                                    log_viewer.add_log("    -> wrote metadata.json", style="green")
                                except Exception as ex:
                                    log_viewer.add_log(f"    -> metadata combine failed: {type(ex).__name__}: {ex}", style="yellow")
                        except futures.TimeoutError:
                            failures += 1
                            fut.cancel()
                            log_viewer.add_log(f"[{i:02d}/{len(sequences)}] [bold red]TIMEOUT[/] after {config.timeout_sec}s for {display_path}{seq.ext}", style="red")
                        except Exception as ex:
                            failures += 1
                            log_viewer.add_log(f"[{i:02d}/{len(sequences)}] [bold red]ERROR[/] on {display_path}{seq.ext}: {type(ex).__name__}: {ex}", style="red")
                        finally:
                            layout["footer"].update(get_footer())

    except GracefulExit:
        # Message already printed by signal handler
        pass
    except KeyboardInterrupt:
        console.print("Interrupted.", file=sys.stderr)
    finally:
        pass

    total_time = time.time() - t0
    total_sequences = len(sequences)
    avg = (total_time / max(1, total_sequences)) if total_sequences else 0.0

    console.print()
    summary_table = Table(show_header=False, box=None, padding=(0, 1))
    summary_table.add_column(style="dim")
    summary_table.add_column()
    summary_table.add_row("Sequences OK:", f"[green]{successes}[/]")
    summary_table.add_row("Sequences Failed:", f"[red]{failures}[/]" if failures else "0")
    summary_table.add_row("Total Time:", f"{total_time:.1f}s")
    summary_table.add_row("Avg Time/Seq:", f"{avg:.1f}s")
    if config.run_mode is RunMode.INDIVIDUAL:
        summary_table.add_row("Frames Created:", f"{produced_total:,}")

    console.print(Panel(summary_table, title="[bold cyan]Summary[/bold cyan]", border_style="cyan", title_align="left"))

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
