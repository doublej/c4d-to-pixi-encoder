#!/usr/bin/env python3
"""
webpseq: Convert numeric frame sequences to WebP (animated or per-frame).

Usage:
  See "Usage" section below or run with -h.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import itertools
import os
import re
import shlex
import signal
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from subprocess import CalledProcessError, run
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import cv2
import numpy as np


from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

SUPPORTED_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".exr", ".dpx"}

INDIVIDUAL_SUBDIR = Path("individual_frames")

MAX_WORKER_CAP = 8
DEFAULT_TIMEOUT_SEC = 300

# ------------------------------
# Data structures
# ------------------------------

@dataclass(frozen=True)
class Quality:
    """Encoding parameters for WebP or AVIF."""
    mode: str  # "high" | "medium" | "low" | "lossless"
    ffmpeg_args: List[str]

    @staticmethod
    def from_name(name: str, format: str) -> "Quality":
        n = name.lower().strip()
        if format == "webp":
            if n == "lossless":
                return Quality(
                    mode="lossless",
                    ffmpeg_args=["-lossless", "1", "-compression_level", "3"],
                )
            mapping = {"high": "90", "medium": "80", "low": "70"}
            if n not in mapping:
                raise ValueError(f"Unknown quality preset for webp: {name}")
            q = mapping[n]
            return Quality(mode=n, ffmpeg_args=["-quality", q, "-compression_level", "3"])
        elif format == "avif":
            # Use libaom-av1 settings; include -b:v 0 for CQ mode and -cpu-used for speed.
            # Lower CRF = better quality.
            if n == "lossless":
                return Quality(mode="lossless", ffmpeg_args=["-crf", "0", "-b:v", "0", "-cpu-used", "6"])
            mapping = {"high": "23", "medium": "30", "low": "40"}
            if n not in mapping:
                raise ValueError(f"Unknown quality preset for avif: {name}")
            crf = mapping[n]
            return Quality(mode=n, ffmpeg_args=["-crf", crf, "-b:v", "0", "-cpu-used", "6"])
        else:
            raise ValueError(f"Unknown format for quality settings: {format}")


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


# ------------------------------
# CLI parsing and entry point
# ------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        prog="webpseq",
        description="Convert numeric frame sequences to WebP (animated or per-frame).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-p", "--base-path", type=Path, default=Path("."), help="Base path to scan")
    p.add_argument("-o", "--output-dir", type=Path, help="Output directory. Defaults to 'outputs/{format}_renders'")
    p.add_argument("-q", "--quality", choices=["high", "medium", "low", "lossless"], default="high", help="Quality preset")
    p.add_argument("--format", choices=["webp", "avif"], default="webp", help="Output format")
    p.add_argument("-i", "--individual-frames", action="store_true", help="Export individual WebP per frame instead of animated")
    p.add_argument(
        "--pad-digits",
        type=int,
        default=None,
        help=(
            "Zero-padding width for per-frame filenames. "
            "When set, per-sequence numbering starts at 1 and outputs are named 00001, 00002, ... in each folder. "
            "When unset, the input filename is used."
        ),
    )
    p.add_argument("-w", "--max-workers", type=int, help="Max parallel workers (capped at 8)")
    p.add_argument("-s", "--sequential", action="store_true", help="Force sequential processing")
    p.add_argument("--check-tools", action="store_true", help="Verify external tools and exit")
    return p.parse_args(argv)


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


def check_tools() -> Tuple[bool, List[str]]:
    """Verify required external tools are available."""
    problems: List[str] = []
    if which("ffmpeg") is None:
        problems.append("ffmpeg not found in PATH")
    return (len(problems) == 0, problems)


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


def is_path_inside(child: Path, parent: Path) -> bool:
    """True if child is the same as or nested under parent."""
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


# ------------------------------
# Scanning for sequences
# ------------------------------

def should_skip_dir(dirname: str) -> bool:
    """Skip output-like and hidden/temp folders."""
    lowered = dirname.lower()
    if lowered in {"outputs", "webp_renders", "h264_renders"}:
        return True
    if dirname.startswith("_") or dirname.startswith("."):
        return True
    return False

def parse_stem(stem: str) -> Optional[Tuple[str, int]]:
    """
    Parses a stem into a sequence prefix and a frame number.
    Finds a number at the very end of the stem.
    e.g., "render_001" -> ("render_", 1)
          "001" -> ("", 1)
          "no_number" -> None
    """
    match = re.search(r"(\d+)$", stem)
    if not match:
        return None
    number_str = match.group(1)
    prefix = stem[: -len(number_str)]
    return prefix, int(number_str)


# ------------------------------
# TIFF validation and processing
# ------------------------------

def validate_tiff_file(file_path: Path) -> Tuple[bool, str]:
    """
    Check if a TIFF file can be processed directly by cwebp.
    TIFFs with alpha channels (4+ channels) are considered invalid for direct processing.
    Returns (is_valid_for_direct_processing, reason).
    """
    try:
        img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return False, "OpenCV could not read the file."

        if len(img.shape) < 3 or img.shape[2] < 4:
            return True, f"Image has {img.shape[2] if len(img.shape) > 2 else 1} channel(s), OK for direct processing."

        # Has 4 or more channels, needs special handling
        return False, f"Image has {img.shape[2]} channels and requires splitting."
    except Exception as e:
        return False, f"Exception during TIFF validation: {e}"


def split_tiff_channels(file_path: Path, output_dir: Path) -> Tuple[bool, Optional[Path], str]:
    """
    Reads a multi-channel TIFF and saves it as a temporary RGBA PNG file
    that cwebp can handle reliably.
    Returns (success, path_to_rgba_png|None, message).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_png_path = output_dir / f"{file_path.stem}_rgba_temp.png"

    try:
        # Read image with all channels
        img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return False, None, "OpenCV could not read file."

        if len(img.shape) < 3 or img.shape[2] < 4:
            return False, None, "Image does not have enough channels to process."

        # If more than 4 channels, just take the first 4.
        # cwebp will handle RGBA, so we hope these are the right ones.
        if img.shape[2] > 4:
            img = img[:, :, :4]

        # Save as PNG. cv2.imwrite expects BGRA for 4-channel PNGs.
        success = cv2.imwrite(str(temp_png_path), img)
        if not success:
            return False, None, "Failed to write temporary RGBA PNG file."

        return True, temp_png_path, "Successfully created temporary RGBA PNG."

    except Exception as e:
        return False, None, f"Exception during channel splitting: {e}"


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
                    dir_path=dpath,
                    rel_dir=rel_dir,
                    prefix=prefix,
                    ext=ext,
                    frames=sorted_frames,
                )
            )

    return sequences


# ------------------------------
# Output path resolution
# ------------------------------

def animated_output_path(output_root: Path, seq: SequenceInfo, format: str) -> Path:
    """
    Place one output file per sequence under output_root mirroring the relative directory.
    """
    # e.g., outputs/webp_renders/<rel_dir>/<prefix_or_dirname>_<ext>.webp
    rel_target_dir = output_root / seq.rel_dir
    base_name_str = seq.prefix.strip() or seq.dir_path.name
    base_name = base_name_str.rstrip("._-")
    basename = f"{base_name}_{seq.ext.lstrip('.')}.{format}"
    return rel_target_dir / basename


def individual_output_path(
    output_root: Path,
    seq: SequenceInfo,
    frame: Path,
    format: str,
    frame_index: Optional[int] = None,
    pad_digits: Optional[int] = None,
) -> Path:
    """
    Place per-frame outputs under:
      output_root/individual_frames/<rel_dir>/<frame_stem>.<format>
    """
    rel_dir = output_root / INDIVIDUAL_SUBDIR / seq.rel_dir
    if pad_digits and frame_index is not None:
        name = f"{frame_index:0{pad_digits}d}.{format}"
    else:
        name = f"{frame.stem}.{format}"
    return rel_dir / name


# ------------------------------
# Encoding helpers
# ------------------------------

def build_ffmpeg_cmd(
    list_file: Path,
    out_path: Path,
    quality: Quality,
    threads: int,
    format: str,
) -> List[str]:
    """Create ffmpeg command for an animated image sequence."""
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
    if format == "webp":
        codec_args = [
            "-c:v", "libwebp",
            "-loop", "0",
            # Force RGBA upstream so alpha is preserved through filters
            "-vf", "format=rgba",
            "-pix_fmt", "yuva420p",
            "-threads", str(max(1, threads)),
        ]
    elif format == "avif":
        codec_args = [
            "-c:v", "libaom-av1",
            # Ensure alpha channel flows; convert to RGBA before encode
            "-vf", "format=rgba",
            # Use 4:4:4 with alpha to avoid subsampling-related alpha drops
            "-pix_fmt", "yuva444p",
            "-threads", str(max(1, threads)),
        ]
    else:
        raise ValueError(f"Unsupported format for animated encoding: {format}")

    return base + codec_args + quality.ffmpeg_args + [str(out_path)]


def write_ffconcat_file(frame_paths: List[Path], target_dir: Path) -> Path:
    """
    Write an ffconcat list file enumerating frames in order.
    Returns the path to the created file.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    list_path = target_dir / "frames.ffconcat"
    with list_path.open("w", encoding="utf-8") as f:
        f.write("ffconcat version 1.0\n")
        for p in frame_paths:
            # escape single quotes in POSIX path for ffconcat format
            posix = p.resolve().as_posix().replace("'", r"'\''")
            f.write(f"file '{posix}'\n")
    return list_path


def run_subprocess(cmd: List[str]) -> Tuple[int, str]:
    """Run a command and return (exit_code, pretty_cmd)."""
    pretty = " ".join(shlex.quote(c) for c in cmd)
    try:
        res = run(cmd, check=True)
        return res.returncode, pretty
    except CalledProcessError as e:
        return e.returncode, pretty


# ------------------------------
# Animated pipeline (process-level parallelism)
# ------------------------------

def encode_sequence_animated_task(
    frame_paths: List[str],
    out_path: str,
    quality_mode: str,
    threads: int,
    format: str,
) -> Tuple[bool, str, Optional[int]]:
    """
    Child-process-safe function to encode one sequence to an animated image.
    Returns (success, message, output_size_bytes|None).
    """
    try:
        q = Quality.from_name(quality_mode, format)
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # Reuse if exists
        if out.exists():
            size = out.stat().st_size
            return True, f"SKIP exists: {out.name} ({size} bytes)", size

        with tempfile.TemporaryDirectory(prefix="webpseq_") as td:
            list_file = write_ffconcat_file([Path(p) for p in frame_paths], Path(td))
            cmd = build_ffmpeg_cmd(list_file, out, q, threads, format)
            code, pretty = run_subprocess(cmd)
            if code != 0:
                return False, f"ffmpeg failed: {pretty}", None

        size = out.stat().st_size if out.exists() else None
        return True, f"DONE: {out.name} ({size} bytes)", size
    except Exception as ex:
        return False, f"Exception: {type(ex).__name__}: {ex}", None


# ------------------------------
# Individual-frames pipeline (thread-level parallelism)
# ------------------------------

def build_ffmpeg_individual_cmd(src: Path, dst: Path, quality: Quality, format: str) -> List[str]:
    """Create ffmpeg command for a single frame."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    base = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-i", str(src),
        "-an",
    ]
    if format == "webp":
        codec_args = ["-vf", "format=rgba", "-c:v", "libwebp", "-pix_fmt", "yuva420p"]
    elif format == "avif":
        # Force RGBA input and encode with alpha-capable pixel format
        codec_args = ["-vf", "format=rgba", "-c:v", "libaom-av1", "-pix_fmt", "yuva444p", "-still-picture", "1"]
    else:
        raise ValueError(f"Unsupported format for individual encoding: {format}")

    return base + codec_args + quality.ffmpeg_args + [str(dst)]


def encode_one_frame(src: Path, dst: Path, quality: Quality, format: str) -> Tuple[bool, str]:
    """Encode one frame, skipping if output exists."""
    if dst.exists():
        return True, f"skip {dst.name}"

    temp_src_file: Optional[Path] = None
    source_to_encode = src
    message_suffix = ""

    try:
        # Handle TIFF files with potential channel issues by converting to a temp PNG
        if src.suffix.lower() in {".tif", ".tiff"}:
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
        cmd = build_ffmpeg_individual_cmd(source_to_encode, dst, quality, format)
        code, pretty = run_subprocess(cmd)

        if code != 0:
            return False, f"FAIL({code}) {pretty}"

        return True, f"ok {dst.name}{message_suffix}"
    finally:
        # Clean up temporary file if it was created
        if temp_src_file:
            try:
                parent_dir = temp_src_file.parent
                temp_src_file.unlink()
                parent_dir.rmdir()
            except OSError:
                pass


def encode_sequence_individual(
    seq: SequenceInfo,
    out_root: Path,
    quality: Quality,
    threads: int,
    timeout_sec: int,
    format: str,
    pad_digits: Optional[int] = None,
) -> Tuple[bool, str, int]:
    """
    Convert one sequence to individual frames using thread pool.
    Returns (success, message, produced_count).
    """
    # Build tasks for missing outputs
    tasks: List[Tuple[Path, Path]] = []
    for idx, f in enumerate(seq.frames, start=1):
        dst = individual_output_path(out_root, seq, f, format, frame_index=idx, pad_digits=pad_digits)
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
        return encode_one_frame(src, dst, quality, format)

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
    return ok, f"{produced}/{missing} created in {elapsed:.1f}s; " + " | ".join(itertools.islice(msgs, 0, 8)), produced


# ------------------------------
# Orchestration
# ------------------------------

class GracefulExit(Exception):
    """Signal-initiated shutdown."""


def install_signal_handlers(stop_event: threading.Event) -> None:
    """Handle Ctrl+C gracefully by setting an event and raising in main thread."""
    def handler(signum, frame):
        stop_event.set()
        console.print("\n[yellow]Ctrl+C received. Attempting graceful shutdown...[/]", file=sys.stderr)
        raise GracefulExit()
    signal.signal(signal.SIGINT, handler)

def create_run_header(base_path: Path, output_dir: Path, safe_msg: str, workers: int, mode: str, quality: Quality, format: str) -> Panel:
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="dim")
    table.add_column()
    table.add_row("Source:", str(base_path.resolve()))
    table.add_row("Output:", str(output_dir.resolve()))
    table.add_row("Safety:", safe_msg or "ok")
    table.add_row("Format:", format.upper())
    table.add_row("Mode:", 'individual-frames' if mode == 'individual' else 'animated')
    table.add_row("Quality:", quality.mode)
    table.add_row("Workers:", f"{workers} (cap {MAX_WORKER_CAP})")
    return Panel(table, title="[bold cyan]Run Configuration[/bold cyan]", border_style="cyan", title_align="left")

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.check_tools:
        ok, probs = check_tools()
        if ok:
            console.print("[bold green]Tools OK:[/] ffmpeg")
            return 0
        for p in probs:
            console.print(f"[bold red]Missing:[/] {p}", file=sys.stderr)
        return 1

    quality = Quality.from_name(args.quality, args.format)
    workers = pick_worker_count(args.max_workers, args.sequential)

    base_path = args.base_path
    output_dir = args.output_dir or Path("outputs") / f"{args.format}_renders"

    safe, reason = is_safe_output_location(base_path, output_dir)
    if not safe:
        console.print(f"[bold red]Unsafe output location:[/] {reason}", file=sys.stderr)
        console.print("No files were written.", file=sys.stderr)
        return 1

    tools_ok, probs = check_tools()
    if not tools_ok:
        for p in probs:
            console.print(f"[bold red]Missing:[/] {p}", file=sys.stderr)
        return 1

    header = create_run_header(base_path, output_dir, "", workers, "individual" if args.individual_frames else "animated", quality, args.format)

    t0 = time.time()
    sequences = find_sequences(base_path)
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
    install_signal_handlers(stop_ev)

    successes = 0
    failures = 0
    produced_total = 0
    perseq_times: List[float] = []

    layout = Layout()
    layout.split_column(
        Layout(header, name="header", size=8),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=3),
    )

    log_items: List = [sequences_table]
    log_panel = Panel(Group(*log_items), title="[cyan]Log[/]", border_style="cyan")
    layout["main"].update(log_panel)

    def get_footer() -> Panel:
        total = successes + failures
        elapsed = time.time() - t0
        return Panel(
            f"Processed: {total}/{len(sequences)} | "
            f"Success: [green]{successes}[/] | "
            f"Failed: [red]{failures}[/] | "
            f"Elapsed: {elapsed:.1f}s",
            title="[cyan]Status[/]",
            border_style="cyan",
        )

    layout["footer"].update(get_footer())

    try:
        with Live(layout, screen=True, redirect_stderr=False, refresh_per_second=4) as live:
            if args.individual_frames:
                # Per-sequence sequential vs threaded per-frame inside
                for i, seq in enumerate(sequences, 1):
                    if stop_ev.is_set():
                        break
                    display_path = (seq.rel_dir.as_posix() or ".") + f"/{seq.prefix}*"
                    log_items.append(Text(f"[{i:02d}/{len(sequences)}] [bold]INDIV[/] {display_path}{seq.ext} ({len(seq)} frames)"))
                    layout["main"].update(Panel(Group(*log_items), title="[cyan]Log[/]", border_style="cyan"))

                    ok, msg, produced = encode_sequence_individual(
                        seq=seq,
                        out_root=output_dir,
                        quality=quality,
                        threads=workers,
                        timeout_sec=DEFAULT_TIMEOUT_SEC,
                        format=args.format,
                        pad_digits=args.pad_digits,
                    )
                    produced_total += produced
                    log_items.append(Text(f"    -> {msg}", style="green" if ok else "red"))
                    layout["main"].update(Panel(Group(*log_items), title="[cyan]Log[/]", border_style="cyan"))

                    if ok:
                        successes += 1
                    else:
                        failures += 1
                    layout["footer"].update(get_footer())
            else:
                # Animated: process pool across sequences
                with futures.ProcessPoolExecutor(max_workers=workers) as pool:
                    fut_map: Dict[futures.Future, Tuple[int, SequenceInfo, Path]] = {}
                    for i, seq in enumerate(sequences, 1):
                        out_path = animated_output_path(output_dir, seq, args.format)
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        frame_paths = [p.as_posix() for p in seq.frames]
                        fut = pool.submit(
                            encode_sequence_animated_task,
                            frame_paths,
                            out_path.as_posix(),
                            quality.mode,
                            max(1, workers),
                            args.format,
                        )
                        fut_map[fut] = (i, seq, out_path)

                    for fut, (i, seq, out_path) in fut_map.items():
                        display_path = (seq.rel_dir.as_posix() or ".") + f"/{seq.prefix}*"
                        try:
                            ok, msg, size = fut.result(timeout=DEFAULT_TIMEOUT_SEC)
                            if ok:
                                successes += 1
                            else:
                                failures += 1
                            log_items.append(Text(f"[{i:02d}/{len(sequences)}] [bold]ANIM [/] {display_path}{seq.ext} -> {out_path.name}"))
                            log_items.append(Text(f"    -> {msg}", style="green" if ok else "red"))
                            if ok and size is not None:
                                log_items.append(Text(f"    size: [green]{size:,}[/] bytes"))
                        except futures.TimeoutError:
                            failures += 1
                            fut.cancel()
                            log_items.append(Text(f"[{i:02d}/{len(sequences)}] [bold red]TIMEOUT[/] after {DEFAULT_TIMEOUT_SEC}s for {display_path}{seq.ext}", style="red"))
                        except Exception as ex:
                            failures += 1
                            log_items.append(Text(f"[{i:02d}/{len(sequences)}] [bold red]ERROR[/] on {display_path}{seq.ext}: {type(ex).__name__}: {ex}", style="red"))
                        finally:
                            layout["main"].update(Panel(Group(*log_items), title="[cyan]Log[/]", border_style="cyan"))
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
    if args.individual_frames:
        summary_table.add_row("Frames Created:", f"{produced_total:,}")

    console.print(Panel(summary_table, title="[bold cyan]Summary[/bold cyan]", border_style="cyan", title_align="left"))

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
