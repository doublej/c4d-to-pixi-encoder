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


from rich.console import Console
from rich.panel import Panel
from rich.table import Table



console = Console()

SUPPORTED_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".exr", ".dpx"}

DEFAULT_OUTPUT = Path("outputs") / "webp_renders"
INDIVIDUAL_SUBDIR = Path("individual_frames")

MAX_WORKER_CAP = 8
DEFAULT_TIMEOUT_SEC = 300

# ------------------------------
# Data structures
# ------------------------------

@dataclass(frozen=True)
class Quality:
    """Encoding parameters for WebP."""
    mode: str  # "high" | "medium" | "low" | "lossless"
    ffmpeg_args: List[str]
    cwebp_args: List[str]

    @staticmethod
    def from_name(name: str) -> "Quality":
        n = name.lower().strip()
        if n == "lossless":
            return Quality(
                mode="lossless",
                ffmpeg_args=["-lossless", "1", "-compression_level", "3"],
                cwebp_args=["-lossless", "-z", "3"],  # fast lossless
            )
        mapping = {
            "high": "90",
            "medium": "80",
            "low": "70",
        }
        if n not in mapping:
            raise ValueError(f"Unknown quality preset: {name}")
        q = mapping[n]
        return Quality(
            mode=n,
            ffmpeg_args=["-quality", q, "-compression_level", "3"],
            cwebp_args=["-q", q],
        )


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
    p.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Output directory")
    p.add_argument("-q", "--quality", choices=["high", "medium", "low", "lossless"], default="high", help="Quality preset")
    p.add_argument("-i", "--individual-frames", action="store_true", help="Export individual WebP per frame instead of animated")
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


def check_tools(require_cwebp: bool) -> Tuple[bool, List[str]]:
    """Verify required external tools are available."""
    problems: List[str] = []
    if which("ffmpeg") is None:
        problems.append("ffmpeg not found in PATH")
    if require_cwebp and which("cwebp") is None:
        problems.append("cwebp not found in PATH (required for --individual-frames)")
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




def split_tiff_channels(file_path: Path, output_dir: Path) -> Tuple[bool, List[Path], str]:
    """
    Split a 4-channel TIFF file into RGB and alpha components using OpenCV.
    Returns (success, [rgb_path, alpha_path], error_message).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = file_path.stem

    rgb_path = output_dir / f"{stem}_rgb.png"
    alpha_path = output_dir / f"{stem}_alpha.png"

    try:
        # Read the 4-channel TIFF image
        img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return False, [], "OpenCV could not read the image."

        if img.shape[2] != 4:
            return False, [], f"Image does not have 4 channels, but {img.shape[2]}."

        # Split the channels
        b, g, r, a = cv2.split(img)
        rgb = cv2.merge((r, g, b))
        
        # Save the RGB and alpha images
        cv2.imwrite(str(rgb_path), rgb)
        cv2.imwrite(str(alpha_path), a)

        return True, [rgb_path, alpha_path], ""
    except Exception as e:
        return False, [], f"Exception during channel splitting: {e}"





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

def animated_output_path(output_root: Path, seq: SequenceInfo) -> Path:
    """
    Place one .webp per sequence under output_root mirroring the relative directory.
    """
    # e.g., outputs/webp_renders/<rel_dir>/<prefix_or_dirname>_<ext>.webp
    rel_target_dir = output_root / seq.rel_dir
    base_name_str = seq.prefix.strip() or seq.dir_path.name
    base_name = base_name_str.rstrip("._-")
    basename = f"{base_name}_{seq.ext.lstrip('.')}.webp"
    return rel_target_dir / basename


def individual_output_path(output_root: Path, seq: SequenceInfo, frame: Path) -> Path:
    """
    Place per-frame outputs under:
      output_root/individual_frames/<rel_dir>/<frame_stem>.webp
    """
    rel_dir = output_root / INDIVIDUAL_SUBDIR / seq.rel_dir
    return rel_dir / f"{frame.stem}.webp"


# ------------------------------
# Encoding helpers
# ------------------------------

def build_ffmpeg_cmd(
    list_file: Path,
    out_path: Path,
    quality: Quality,
    threads: int,
) -> List[str]:
    """Create ffmpeg command for animated WebP."""
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
        "-c:v", "libwebp",
        "-loop", "0",
        "-pix_fmt", "yuva420p",
        "-threads", str(max(1, threads)),
    ]
    return base + quality.ffmpeg_args + [str(out_path)]


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
) -> Tuple[bool, str, Optional[int]]:
    """
    Child-process-safe function to encode one sequence to animated WebP.
    Returns (success, message, output_size_bytes|None).
    """
    try:
        q = Quality.from_name(quality_mode)
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # Reuse if exists
        if out.exists():
            size = out.stat().st_size
            return True, f"SKIP exists: {out.name} ({size} bytes)", size

        with tempfile.TemporaryDirectory(prefix="webpseq_") as td:
            list_file = write_ffconcat_file([Path(p) for p in frame_paths], Path(td))
            cmd = build_ffmpeg_cmd(list_file, out, q, threads)
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

def build_cwebp_cmd(src: Path, dst: Path, quality: Quality) -> List[str]:
    """Create cwebp command for a single frame."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    base = ["cwebp", "-mt"]
    # progress is default; avoid -quiet to allow progress reporting
    if quality.mode == "lossless":
        args = quality.cwebp_args
    else:
        args = quality.cwebp_args
    return base + args + [src.as_posix(), "-o", dst.as_posix()]


def encode_one_frame(src: Path, dst: Path, quality: Quality) -> Tuple[bool, str]:
    """Encode one frame, skipping if output exists."""
    if dst.exists():
        return True, f"skip {dst.name}"

    # Handle TIFF files with potential channel issues
    if src.suffix.lower() in {".tif", ".tiff"}:
        try:
            img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
            if img is not None and img.shape[2] == 4:
                console.print(f"[blue]Splitting 4-channel TIFF: {src.name}[/]")
                
                # Try to split the TIFF into RGB and alpha
                temp_dir = dst.parent / "temp_split"
                success, split_files, split_error = split_tiff_channels(src, temp_dir)

                if success and len(split_files) >= 2:
                    rgb_path, alpha_path = split_files[:2]

                    # Convert RGB to WebP first
                    rgb_dst = dst.parent / f"{dst.stem}_rgb{quality.mode}.webp"
                    rgb_cmd = build_cwebp_cmd(rgb_path, rgb_dst, quality)
                    rgb_code, rgb_pretty = run_subprocess(rgb_cmd)

                    if rgb_code != 0:
                        return False, f"FAIL RGB({rgb_code}) {rgb_pretty}"

                    # Convert alpha to WebP
                    alpha_dst = dst.parent / f"{dst.stem}_alpha{quality.mode}.webp"
                    alpha_cmd = build_cwebp_cmd(alpha_path, alpha_dst, quality)
                    alpha_code, alpha_pretty = run_subprocess(alpha_cmd)

                    if alpha_code != 0:
                        return False, f"FAIL Alpha({alpha_code}) {alpha_pretty}"

                    # For now, just use the RGB version as the main output
                    # TODO: Implement alpha recombination for WebP
                    import shutil
                    shutil.copy2(rgb_dst, dst)

                    # Clean up temp files
                    try:
                        rgb_path.unlink(missing_ok=True)
                        alpha_path.unlink(missing_ok=True)
                        rgb_dst.unlink(missing_ok=True)
                        alpha_dst.unlink(missing_ok=True)
                        temp_dir.rmdir()
                    except:
                        pass

                    return True, f"ok {dst.name} (split channels)"
                else:
                    return False, f"Channel splitting failed: {split_error}"
        except Exception as e:
            console.print(f"[yellow]Could not process TIFF {src.name} with OpenCV: {e}[/]")

    # Normal processing for other formats
    cmd = build_cwebp_cmd(src, dst, quality)
    code, pretty = run_subprocess(cmd)
    if code != 0:
        return False, f"FAIL({code}) {pretty}"
    return True, f"ok {dst.name}"


def encode_sequence_individual(
    seq: SequenceInfo,
    out_root: Path,
    quality: Quality,
    threads: int,
    timeout_sec: int,
) -> Tuple[bool, str, int]:
    """
    Convert one sequence to individual frames using thread pool.
    Returns (success, message, produced_count).
    """
    # Build tasks for missing outputs
    tasks: List[Tuple[Path, Path]] = []
    for f in seq.frames:
        dst = individual_output_path(out_root, seq, f)
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
        return encode_one_frame(src, dst, quality)

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
        console.print("\n[yellow]Ctrl+C received. Attempting graceful shutdown...[/]")
        raise GracefulExit()
    signal.signal(signal.SIGINT, handler)

def print_run_header(base_path: Path, output_dir: Path, safe_msg: str, workers: int, mode: str, quality: Quality) -> None:
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="dim")
    table.add_column()
    table.add_row("Source:", str(base_path.resolve()))
    table.add_row("Output:", str(output_dir.resolve()))
    table.add_row("Safety:", safe_msg or "ok")
    table.add_row("Mode:", 'individual-frames' if mode == 'individual' else 'animated')
    table.add_row("Quality:", quality.mode)
    table.add_row("Workers:", f"{workers} (cap {MAX_WORKER_CAP})")
    console.print(Panel(table, title="[bold cyan]Run Configuration[/bold cyan]", border_style="cyan", title_align="left"))

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    require_cwebp = bool(args.individual_frames)
    if args.check_tools:
        ok, probs = check_tools(require_cwebp=require_cwebp)
        if ok:
            console.print("[bold green]Tools OK:[/] ffmpeg" + (" + cwebp" if require_cwebp else ""))
            return 0
        for p in probs:
            console.print(f"[bold red]Missing:[/] {p}")
        return 1

    quality = Quality.from_name(args.quality)
    workers = pick_worker_count(args.max_workers, args.sequential)

    base_path = args.base_path
    output_dir = args.output_dir

    safe, reason = is_safe_output_location(base_path, output_dir)
    if not safe:
        console.print(f"[bold red]Unsafe output location:[/] {reason}")
        console.print("No files were written.")
        return 1

    tools_ok, probs = check_tools(require_cwebp=require_cwebp)
    if not tools_ok:
        for p in probs:
            console.print(f"[bold red]Missing:[/] {p}")
        return 1

    print_run_header(base_path, output_dir, "", workers, "individual" if args.individual_frames else "animated", quality)

    t0 = time.time()
    sequences = find_sequences(base_path)
    if not sequences:
        console.print("[yellow]No numeric sequences (>=4 frames) found. Nothing to do.[/]")
        return 0

    table = Table(title=f"Discovered {len(sequences)} sequences", show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Path")
    table.add_column("Frames", justify="right")
    table.add_column("Extension")
    for idx, seq in enumerate(sequences, 1):
        display_path = (seq.rel_dir.as_posix() or ".") + f"/{seq.prefix}*"
        table.add_row(f"{idx:02d}", display_path, str(len(seq)), seq.ext)
    console.print(table)

    stop_ev = threading.Event()
    install_signal_handlers(stop_ev)

    successes = 0
    failures = 0
    produced_total = 0
    perseq_times: List[float] = []

    try:
        if args.individual_frames:
            # Per-sequence sequential vs threaded per-frame inside
            for i, seq in enumerate(sequences, 1):
                if stop_ev.is_set():
                    break
                display_path = (seq.rel_dir.as_posix() or ".") + f"/{seq.prefix}*"
                console.print(f"[{i:02d}/{len(sequences)}] [bold]INDIV[/] {display_path}{seq.ext} ({len(seq)} frames)")
                ok, msg, produced = encode_sequence_individual(
                    seq=seq,
                    out_root=output_dir,
                    quality=quality,
                    threads=workers,
                    timeout_sec=DEFAULT_TIMEOUT_SEC,
                )
                produced_total += produced
                console.print("    " + msg)
                if ok:
                    successes += 1
                else:
                    failures += 1
        else:
            # Animated: process pool across sequences
            with futures.ProcessPoolExecutor(max_workers=workers) as pool:
                fut_map: Dict[futures.Future, Tuple[int, SequenceInfo, Path]] = {}
                for i, seq in enumerate(sequences, 1):
                    out_path = animated_output_path(output_dir, seq)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    frame_paths = [p.as_posix() for p in seq.frames]
                    fut = pool.submit(
                        encode_sequence_animated_task,
                        frame_paths,
                        out_path.as_posix(),
                        quality.mode,
                        max(1, workers),
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
                        console.print(f"[{i:02d}/{len(sequences)}] [bold]ANIM [/] {display_path}{seq.ext} -> {out_path.name}")
                        console.print(f"    {msg}")
                        if ok and size is not None:
                            console.print(f"    size: [green]{size:,}[/] bytes")
                    except futures.TimeoutError:
                        failures += 1
                        fut.cancel()
                        console.print(f"[{i:02d}/{len(sequences)}] [bold red]TIMEOUT[/] after {DEFAULT_TIMEOUT_SEC}s for {display_path}{seq.ext}")
                    except Exception as ex:
                        failures += 1
                        console.print(f"[{i:02d}/{len(sequences)}] [bold red]ERROR[/] on {display_path}{seq.ext}: {type(ex).__name__}: {ex}")

    except GracefulExit:
        # Message already printed by signal handler
        pass
    except KeyboardInterrupt:
        console.print("Interrupted.")
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
