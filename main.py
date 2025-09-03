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
    """A detected numeric frame sequence within a single directory and extension."""
    dir_path: Path
    rel_dir: Path
    ext: str
    frames: List[Path]  # sorted numerically by stem

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

def is_numeric_stem(path: Path) -> bool:
    """Return True if basename without extension is purely numeric (no sign, no spaces)."""
    return path.stem.isdigit()


def find_sequences(base_path: Path) -> List[SequenceInfo]:
    """
    Walk base_path and detect sequences where filenames are purely numeric
    and count >= 4, per extension within a directory.
    """
    sequences: List[SequenceInfo] = []
    base = base_path.resolve()

    for dirpath, dirnames, filenames in os.walk(base):
        # prune directories in-place
        dirnames[:] = [d for d in dirnames if not should_skip_dir(d)]
        dpath = Path(dirpath)

        files = [Path(dirpath) / f for f in filenames]
        by_ext: Dict[str, List[Path]] = {}
        for f in files:
            ext = f.suffix.lower()
            if ext in SUPPORTED_EXTS and is_numeric_stem(f):
                by_ext.setdefault(ext, []).append(f)

        for ext, items in by_ext.items():
            if len(items) < 4:
                continue
            # sort numerically by stem
            items_sorted = sorted(items, key=lambda p: int(p.stem))
            rel_dir = dpath.relative_to(base)
            sequences.append(SequenceInfo(dir_path=dpath, rel_dir=rel_dir, ext=ext, frames=items_sorted))

    return sequences


# ------------------------------
# Output path resolution
# ------------------------------

def animated_output_path(output_root: Path, seq: SequenceInfo) -> Path:
    """
    Place one .webp per sequence under output_root mirroring the relative directory.
    To avoid collisions between multiple ext-based sequences in the same folder,
    suffix the directory name with the extension.
    """
    # e.g., outputs/webp_renders/<rel_dir>/<dirname>_<ext>.webp
    rel_target_dir = output_root / seq.rel_dir
    basename = f"{seq.dir_path.name}_{seq.ext.lstrip('.')}.webp"
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
        print("\nCtrl+C received. Attempting graceful shutdown...", file=sys.stderr)
        raise GracefulExit()
    signal.signal(signal.SIGINT, handler)

def print_run_header(base_path: Path, output_dir: Path, safe_msg: str, workers: int, mode: str, quality: Quality) -> None:
    print(f"Source:  {base_path.resolve()}")
    print(f"Output:  {output_dir.resolve()}")
    print(f"Safety:  {safe_msg or 'ok'}")
    print(f"Mode:    {'individual-frames' if mode == 'individual' else 'animated'}")
    print(f"Quality: {quality.mode}")
    print(f"Workers: {workers} (cap {MAX_WORKER_CAP})")
    print("-" * 60)

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    require_cwebp = bool(args.individual_frames)
    if args.check_tools:
        ok, probs = check_tools(require_cwebp=require_cwebp)
        if ok:
            print("Tools OK: ffmpeg" + (" + cwebp" if require_cwebp else ""))
            return 0
        for p in probs:
            print(f"Missing: {p}", file=sys.stderr)
        return 1

    quality = Quality.from_name(args.quality)
    workers = pick_worker_count(args.max_workers, args.sequential)

    base_path = args.base_path
    output_dir = args.output_dir

    safe, reason = is_safe_output_location(base_path, output_dir)
    if not safe:
        print(f"Unsafe output location: {reason}", file=sys.stderr)
        print("No files were written.", file=sys.stderr)
        return 1

    tools_ok, probs = check_tools(require_cwebp=require_cwebp)
    if not tools_ok:
        for p in probs:
            print(f"Missing: {p}", file=sys.stderr)
        return 1

    print_run_header(base_path, output_dir, "", workers, "individual" if args.individual_frames else "animated", quality)

    t0 = time.time()
    sequences = find_sequences(base_path)
    if not sequences:
        print("No numeric sequences (>=4 frames) found. Nothing to do.")
        return 0

    print(f"Discovered {len(sequences)} sequences:")
    for idx, seq in enumerate(sequences, 1):
        print(f"  [{idx:02d}] {seq.rel_dir.as_posix()}  ({len(seq)} frames, {seq.ext})")
    print("-" * 60)

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
                print(f"[{i:02d}/{len(sequences)}] INDIV {seq.rel_dir.as_posix()} {seq.ext} ({len(seq)} frames)")
                ok, msg, produced = encode_sequence_individual(
                    seq=seq,
                    out_root=output_dir,
                    quality=quality,
                    threads=workers,
                    timeout_sec=DEFAULT_TIMEOUT_SEC,
                )
                produced_total += produced
                print("    " + msg)
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
                    try:
                        ok, msg, size = fut.result(timeout=DEFAULT_TIMEOUT_SEC)
                        if ok:
                            successes += 1
                        else:
                            failures += 1
                        print(f"[{i:02d}/{len(sequences)}] ANIM  {seq.rel_dir.as_posix()} {seq.ext} -> {out_path.name}")
                        print(f"    {msg}")
                        if ok and size is not None:
                            print(f"    size: {size} bytes")
                    except futures.TimeoutError:
                        failures += 1
                        fut.cancel()
                        print(f"[{i:02d}/{len(sequences)}] TIMEOUT after {DEFAULT_TIMEOUT_SEC}s: {seq.rel_dir.as_posix()} {seq.ext}", file=sys.stderr)
                    except Exception as ex:
                        failures += 1
                        print(f"[{i:02d}/{len(sequences)}] ERROR: {type(ex).__name__}: {ex}", file=sys.stderr)

    except GracefulExit:
        print("Interrupted. Cleaning up...", file=sys.stderr)
        # rely on executor contexts to clean up
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
    finally:
        pass

    total_time = time.time() - t0
    total_sequences = len(sequences)
    avg = (total_time / max(1, total_sequences)) if total_sequences else 0.0
    print("-" * 60)
    print(f"Totals: sequences ok={successes}, failed={failures}, time={total_time:.1f}s, avg/seq={avg:.1f}s")
    if args.individual_frames:
        print(f"Frames created (this run): {produced_total}")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())