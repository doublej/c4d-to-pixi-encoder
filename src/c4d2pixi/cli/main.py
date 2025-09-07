#!/usr/bin/env python3
"""
c4d2pixi: Convert Cinema 4D rendered sequences to WebP/AVIF (animated or per-frame).

This refactor improves readability and separations of concerns by:
- Introducing enums for output format and run mode
- Using a Config dataclass to pass settings
- Splitting logic into small, single-purpose functions
- Keeping I/O at the edges and core logic pure
"""

from __future__ import annotations

# Standard library imports
import argparse
import concurrent.futures as futures
import contextlib
import os
import signal
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path

# Local application imports
from ..cli.metadata import combine_to_metadata, write_dpi_json, write_offset_json
from ..config import (
    MIN_SEQUENCE_FRAMES,
    PAUSE_CHECK_INTERVAL,
    STATUS_PRINT_INTERVAL,
    AnimatedEncodeConfig,
    Config,
    OutputFormat,
    Quality,
    RunMode,
    SequenceInfo,
)
from ..core.mapping import DirectoryMapper, OutputCategory
from ..output.logger import SimpleLogger
from ..output.persistent import install_persistent_console
from ..processing.crop import compute_sequence_aligned_crop
from ..processing.image import check_alpha_exists, image_dimensions
from ..processing.sequence import SequenceProcessor
from ..tools.check import check_tools
from ..utils.dpi import dpi_dict, read_sequence_dpi
from ..utils.path import is_path_inside, parse_stem, should_skip_dir

# Constants now imported from config.py

# Ensure progress-style carriage-return writes are persisted as lines
install_persistent_console()

SUPPORTED_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".exr", ".dpx"}
MAX_WORKER_CAP = 8
DEFAULT_TIMEOUT_SEC = 300


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        prog="c4d2pixi",
        description="Convert numeric frame sequences to WebP/AVIF (animated or per-frame).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-p", "--base-path", type=Path, default=Path("."), help="Base path to scan")
    p.add_argument("-o", "--output-dir", type=Path, help="Output directory. Defaults to 'outputs/{format}_renders'")
    p.add_argument(
        "-q", "--quality", choices=["high", "medium", "low", "lossless"], default="high", help="Quality preset"
    )
    p.add_argument(
        "--format", choices=[f.value for f in OutputFormat], default=OutputFormat.WEBP.value, help="Output format"
    )
    p.add_argument(
        "-i", "--individual-frames", action="store_true", help="Export individual files per frame instead of animated"
    )
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
    p.add_argument("--first-frame-only", action="store_true", help="Process only the first frame of each sequence")
    p.add_argument("--extract-scenes", action="store_true", help="Extract first frame as scene for transitions")
    p.add_argument("--no-extract-room-stills", dest="extract_room_stills", action="store_false", default=True, help="Disable extraction of room stills (enabled by default)")
    p.add_argument("--crop-alignment", type=int, default=256, help="Pixel alignment for crop boundaries (default: 256)")
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
        base_path=base_path,
        output_dir=output_dir,
        format=fmt,
        quality=quality,
        run_mode=run_mode,
        workers=workers,
        timeout_sec=DEFAULT_TIMEOUT_SEC,
        pad_digits=args.pad_digits,
        first_frame_only=args.first_frame_only,
        extract_scenes=args.extract_scenes,
        extract_room_stills=args.extract_room_stills,
        crop_alignment=args.crop_alignment,
    )


# ------------------------------
# Environment and safety checks
# ------------------------------


def pick_worker_count(requested: int | None, sequential: bool) -> int:
    """Determine worker count within the cap."""
    if sequential:
        return 1
    import multiprocessing

    cores = max(1, multiprocessing.cpu_count())
    limit = min(cores, MAX_WORKER_CAP)
    if requested is None:
        return limit
    return max(1, min(requested, MAX_WORKER_CAP))


def is_safe_output_location(base_path: Path, output_dir: Path) -> tuple[bool, str]:
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


def find_sequences(base_path: Path, first_frame_only: bool = False) -> list[SequenceInfo]:
    """
    Walk base_path and detect frame sequences.
    A sequence is a series of files in the same directory, with the same extension,
    and filenames that end in a number (e.g., "render_001.png", "render_002.png").
    Sequences must have at least MIN_SEQUENCE_FRAMES frames.

    Args:
        base_path: Root directory to scan
        first_frame_only: If True, only include the first frame of each sequence
    """
    sequences: list[SequenceInfo] = []
    base = base_path.resolve()

    for dirpath, dirnames, filenames in os.walk(base):
        # prune directories in-place
        dirnames[:] = [d for d in dirnames if not should_skip_dir(d)]
        dpath = Path(dirpath)

        # Group files by (prefix, extension) in this directory
        potential_sequences: dict[tuple[str, str], list[tuple[int, Path]]] = {}
        for f_name in filenames:
            f_path = dpath / f_name
            ext = f_path.suffix.lower()
            if ext not in SUPPORTED_EXTS:
                continue

            parsed = parse_stem(f_path.stem)
            if parsed:
                prefix, frame_num = parsed
                potential_sequences.setdefault((prefix, ext), []).append((frame_num, f_path))

        # Filter for actual sequences (>= MIN_SEQUENCE_FRAMES frames) and create SequenceInfo
        for (prefix, ext), frames_with_nums in potential_sequences.items():
            if len(frames_with_nums) < MIN_SEQUENCE_FRAMES:
                continue

            # Sort by frame number and extract just the paths
            frames_with_nums.sort(key=lambda item: item[0])
            sorted_frames = [p for _, p in frames_with_nums]

            # If first_frame_only is True, only keep the first frame
            if first_frame_only:
                sorted_frames = sorted_frames[:1]

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


def extract_scene_from_sequence(
    seq: SequenceInfo,
    config: Config,
    logger: SimpleLogger
) -> tuple[bool, str]:
    """
    Extract the first frame of a sequence as a scene image.
    Only applies to transition sequences.
    """
    # Check if we have a mapping for this directory
    naming = DirectoryMapper.get_output_naming(seq.rel_dir, seq.prefix)

    if not naming or naming.category != OutputCategory.TRANSITION:
        # Only extract scenes for transitions
        return True, "skip non-transition"

    # Get the first frame
    if not seq.frames:
        return False, "no frames in sequence"

    first_frame = seq.frames[0]

    # Generate scene output path
    scene_dir = config.output_dir / "scenes"
    scene_dir.mkdir(parents=True, exist_ok=True)

    # Extract scene info to create proper scene name
    scene_info = DirectoryMapper.extract_scene_info(naming)
    if scene_info:
        scene_type, time_of_day = scene_info
        scene_name = f"scene_{scene_type}_{time_of_day}.{config.format.extension}"
    else:
        scene_name = naming.format_scene_name(config.format.extension)

    scene_path = scene_dir / scene_name

    # Skip if already exists
    if scene_path.exists():
        return True, f"scene exists: {scene_name}"

    # Process the first frame
    from ..processing.sequence import SequenceProcessor

    # Check for alpha and compute crop if needed
    has_alpha = check_alpha_exists(first_frame)
    if not has_alpha:
        ow, oh = image_dimensions(first_frame)
        crop_tuple = None
    else:
        # For scenes, compute crop just for the first frame
        cx, cy, cw, ch, ow, oh = compute_sequence_aligned_crop([first_frame], config.crop_alignment)
        crop_tuple = None if (cx == 0 and cy == 0 and cw == ow and ch == oh) else (cx, cy, cw, ch)

    # Encode the frame
    success, msg = SequenceProcessor.encode_one_frame(
        first_frame,
        scene_path,
        config.quality,
        config.format,
        crop_tuple,
        ow,
        oh
    )

    if success:
        return True, f"extracted scene: {scene_name}"
    else:
        return False, f"failed to extract scene: {msg}"


def extract_room_still_from_sequence(
    seq: SequenceInfo,
    config: Config,
    logger: SimpleLogger,
    extract_both_ends: bool = True
) -> tuple[bool, str]:
    """
    Extract room stills from a place-to-place transition.
    The first frame represents the starting room.
    The last frame represents the destination room.
    """
    # Check if we have a mapping for this directory
    naming = DirectoryMapper.get_output_naming(seq.rel_dir, seq.prefix)

    if not naming:
        return True, "skip unmapped"

    # Process both first and last frames for transitions
    results = []
    frames_to_extract = [
        ("first", seq.frames[0] if seq.frames else None),
    ]
    
    # Add last frame if we want both ends and have multiple frames
    if extract_both_ends and len(seq.frames) > 1:
        frames_to_extract.append(("last", seq.frames[-1]))
    
    for position, frame in frames_to_extract:
        if not frame:
            continue
            
        # Get the room still name for this position
        room_still_name = naming.get_room_still_name(config.format.extension, position)
        if not room_still_name:
            # Not a transition or couldn't extract room name
            continue

        # Generate room still output path
        room_dir = config.output_dir / "room_stills"
        room_dir.mkdir(parents=True, exist_ok=True)

        room_path = room_dir / room_still_name

        # Skip if already exists
        if room_path.exists():
            results.append(f"exists: {room_still_name}")
            continue

        # Process the frame
        from ..processing.sequence import SequenceProcessor

        # Check for alpha and compute crop if needed
        has_alpha = check_alpha_exists(frame)
        if not has_alpha:
            ow, oh = image_dimensions(frame)
            crop_tuple = None
        else:
            # For room stills, compute crop just for this frame
            cx, cy, cw, ch, ow, oh = compute_sequence_aligned_crop([frame], config.crop_alignment)
            crop_tuple = None if (cx == 0 and cy == 0 and cw == ow and ch == oh) else (cx, cy, cw, ch)

        # Encode the frame
        success, msg = SequenceProcessor.encode_one_frame(
            frame,
            room_path,
            config.quality,
            config.format,
            crop_tuple,
            ow,
            oh
        )

        if success:
            results.append(f"extracted: {room_still_name}")
        else:
            return False, f"failed to extract room still: {msg}"
    
    if results:
        return True, "; ".join(results)
    else:
        return True, "skip non-transition"


def animated_output_path(output_root: Path, seq: SequenceInfo, fmt: OutputFormat) -> Path:
    """
    Place one output file per sequence under output_root with new naming convention.
    """
    # Check if we have a mapping for this directory
    naming = DirectoryMapper.get_output_naming(seq.rel_dir, seq.prefix)

    if naming:
        # Use mapped naming convention
        rel_target_dir = output_root / naming.category_dir
        basename = naming.format_animated_name(fmt.extension)
    else:
        # Fall back to original naming
        rel_target_dir = output_root / seq.rel_dir
        base_name_str = seq.prefix.strip() or seq.dir_path.name
        base_name = base_name_str.rstrip("._-")
        basename = f"{base_name}_{seq.ext.lstrip('.')}.{fmt.extension}"

    return rel_target_dir / basename


# ------------------------------
# Animated pipeline (process-level parallelism)
# ------------------------------


class GracefulExitError(Exception):
    """Signal-initiated shutdown."""


def install_signal_handlers(stop_event: threading.Event) -> None:
    """Handle Ctrl+C gracefully by setting an event and raising in main thread."""

    def handler(signum, frame):
        stop_event.set()
        print("\nCtrl+C received. Attempting graceful shutdown...", file=sys.stderr)
        raise GracefulExitError()

    signal.signal(signal.SIGINT, handler)


def start_controls_listener(pause_event: threading.Event, stop_event: threading.Event) -> threading.Thread:
    """Start a background thread to listen for simple controls from stdin.

    Controls:
      - 'p' + Enter: toggle pause/resume
      - 'q' + Enter: request quit (graceful)

    Returns:
        Thread: The daemon thread handling input.
    """
    import os as _os

    def _reader() -> None:
        if not sys.stdin:
            return
        # Use line-buffered input to avoid raw mode complexity
        while not stop_event.is_set():
            try:
                line = sys.stdin.readline()
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
                    with contextlib.suppress(Exception):
                        # Trigger the SIGINT handler for consistent shutdown path
                        _os.kill(_os.getpid(), signal.SIGINT)
            except Exception:
                break

    t = threading.Thread(target=_reader, name="controls-listener", daemon=True)
    t.start()
    return t


def print_run_header(logger: SimpleLogger, config: Config, safe_msg: str) -> None:
    """Print the run configuration."""
    logger.section("Run Configuration")
    rows = [
        ["Source:", str(config.base_path.resolve())],
        ["Output:", str(config.output_dir.resolve())],
        ["Safety:", safe_msg or "ok"],
        ["Format:", config.format.value.upper()],
        ["Mode:", "individual-frames" if config.run_mode is RunMode.INDIVIDUAL else "animated"],
        ["Quality:", config.quality.mode],
        ["Workers:", f"{config.workers} (cap {MAX_WORKER_CAP})"],
    ]
    for label, value in rows:
        logger.log(f"{label:<12} {value}")


def process_individual_sequences(
    sequences: list[SequenceInfo],
    config: Config,
    logger: SimpleLogger,
    stop_ev: threading.Event,
    pause_ev: threading.Event,
    print_status_fn,
) -> tuple[int, int, int]:
    """Process sequences in individual frame mode."""
    successes = failures = produced_total = 0

    for i, seq in enumerate(sequences, 1):
        if stop_ev.is_set():
            break

        # Pause between sequences if requested
        while pause_ev.is_set() and not stop_ev.is_set():
            time.sleep(PAUSE_CHECK_INTERVAL)

        display_path = (seq.rel_dir.as_posix() or ".") + f"/{seq.prefix}*"
        logger.info(f"[{i:02d}/{len(sequences)}] INDIV {display_path}{seq.ext} ({len(seq)} frames)")

        ok, msg, produced = SequenceProcessor.encode_sequence_individual(
            seq=seq,
            out_root=config.output_dir,
            quality=config.quality,
            threads=config.workers,
            timeout_sec=config.timeout_sec,
            fmt=config.format,
            pad_digits=config.pad_digits,
            crop_alignment=config.crop_alignment,
        )
        produced_total += produced

        if ok:
            logger.success(f"    -> {msg}")
            successes += 1
            
            # Extract scene if requested (for transitions only)
            if config.extract_scenes:
                scene_ok, scene_msg = extract_scene_from_sequence(seq, config, logger)
                if scene_ok and "skip" not in scene_msg:
                    logger.success(f"    -> {scene_msg}")
                elif not scene_ok:
                    logger.warning(f"    -> {scene_msg}")
            
            # Extract room still for transitions (works with first_frame_only too)
            if config.extract_room_stills:
                room_ok, room_msg = extract_room_still_from_sequence(seq, config, logger)
                if room_ok and "skip" not in room_msg:
                    logger.success(f"    -> {room_msg}")
                elif not room_ok:
                    logger.warning(f"    -> {room_msg}")
        else:
            logger.error(f"    -> {msg}")
            failures += 1

        # Print status periodically
        if i % STATUS_PRINT_INTERVAL == 0 or i == len(sequences):
            print_status_fn()

    return successes, failures, produced_total


def process_animated_sequences(
    sequences: list[SequenceInfo],
    config: Config,
    logger: SimpleLogger,
    stop_ev: threading.Event,
    pause_ev: threading.Event,
    print_status_fn,
) -> tuple[int, int]:
    """Process sequences in animated mode using process pool."""
    successes = failures = 0

    with futures.ProcessPoolExecutor(max_workers=config.workers) as pool:
        fut_map: dict[futures.Future, tuple[int, SequenceInfo, Path]] = {}

        for i, seq in enumerate(sequences, 1):
            if stop_ev.is_set():
                break
            while pause_ev.is_set() and not stop_ev.is_set():
                time.sleep(PAUSE_CHECK_INTERVAL)

            out_path = animated_output_path(config.output_dir, seq, config.format)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # Read DPI once per sequence and write sidecar JSON next to the animated output target
            try:
                dpi_info = read_sequence_dpi(seq.frames)
                write_dpi_json(out_path.with_suffix(out_path.suffix + ".dpi.json"), dpi_dict(dpi_info))
            except Exception:
                pass

            frame_paths = [p.as_posix() for p in seq.frames]

            # Detect alpha channel for optimization
            has_alpha = check_alpha_exists(seq.frames[0])

            # Skip crop when the first frame has no alpha channel
            if not has_alpha:
                ow, oh = image_dimensions(seq.frames[0])
                crop_tuple = None
                cx = cy = 0
                cw, ch = ow, oh
            else:
                # Compute sequence-wide crop
                cx, cy, cw, ch, ow, oh = compute_sequence_aligned_crop(seq.frames, config.crop_alignment)
                crop_tuple: tuple[int, int, int, int] | None = None
                if not (cx == 0 and cy == 0 and cw == ow and ch == oh):
                    crop_tuple = (cx, cy, cw, ch)

            offsets_path = out_path.with_suffix(out_path.suffix + ".json")
            encode_config = AnimatedEncodeConfig(
                frame_paths=frame_paths,
                out_path=out_path.as_posix(),
                quality_mode=config.quality.mode,
                threads=max(1, config.workers),
                format_value=config.format.value,
                crop_rect=crop_tuple,
                offsets_json=offsets_path.as_posix(),
                seq_orig_w=ow,
                seq_orig_h=oh,
                has_alpha=has_alpha,
            )

            display_path = (seq.rel_dir.as_posix() or ".") + f"/{seq.prefix}*"
            logger.info(f"[{i:02d}/{len(sequences)}] ANIM {display_path}{seq.ext} ({len(seq)} frames)")

            fut = pool.submit(SequenceProcessor.encode_animated_task, encode_config)
            fut_map[fut] = (i, seq, out_path)

        # Collect results
        for fut in futures.as_completed(fut_map.keys(), timeout=config.timeout_sec):
            i, seq, out_path = fut_map[fut]
            display_path = (seq.rel_dir.as_posix() or ".") + f"/{seq.prefix}*"

            try:
                ok, msg, produced = fut.result(timeout=config.timeout_sec)
                if ok:
                    logger.success(f"[{i:02d}/{len(sequences)}] ANIM {display_path}{seq.ext} -> {msg}")
                    successes += 1

                    # Try to combine metadata
                    try:
                        metadata_entries = combine_to_metadata(seq.frames, (cx, cy))
                        json_path = out_path.with_suffix(out_path.suffix + ".json")
                        write_offset_json(json_path, metadata_entries)
                        logger.success("    -> wrote metadata.json")
                    except Exception as ex:
                        logger.warning(f"    -> metadata combine failed: {type(ex).__name__}: {ex}")

                    # Extract scene if requested (for transitions only)
                    if config.extract_scenes:
                        scene_ok, scene_msg = extract_scene_from_sequence(seq, config, logger)
                        if scene_ok and "skip" not in scene_msg:
                            logger.success(f"    -> {scene_msg}")
                        elif not scene_ok:
                            logger.warning(f"    -> {scene_msg}")
                    
                    # Extract room still for transitions
                    if config.extract_room_stills:
                        room_ok, room_msg = extract_room_still_from_sequence(seq, config, logger)
                        if room_ok and "skip" not in room_msg:
                            logger.success(f"    -> {room_msg}")
                        elif not room_ok:
                            logger.warning(f"    -> {room_msg}")
                else:
                    logger.error(f"[{i:02d}/{len(sequences)}] ANIM {display_path}{seq.ext} -> {msg}")
                    failures += 1
            except futures.TimeoutError:
                failures += 1
                fut.cancel()
                logger.error(
                    f"[{i:02d}/{len(sequences)}] TIMEOUT after {config.timeout_sec}s for {display_path}{seq.ext}"
                )
            except Exception as ex:
                failures += 1
                logger.error(f"[{i:02d}/{len(sequences)}] ERROR on {display_path}{seq.ext}: {type(ex).__name__}: {ex}")
            finally:
                # Print status periodically
                if i % STATUS_PRINT_INTERVAL == 0 or i == len(sequences):
                    print_status_fn()

    return successes, failures


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)

    if args.check_tools:
        ok, probs = check_tools()
        if ok:
            print("Tools OK: ffmpeg")
            return 0
        for p in probs:
            print(f"Missing: {p}", file=sys.stderr)
        return 1

    config = build_config(args)
    log_file = config.output_dir / "processing.log"
    config.output_dir.mkdir(parents=True, exist_ok=True)
    logger = SimpleLogger(log_file)

    safe, reason = is_safe_output_location(config.base_path, config.output_dir)
    if not safe:
        logger.error(f"Unsafe output location: {reason}")
        logger.error("No files were written.")
        return 1

    tools_ok, probs = check_tools()
    if not tools_ok:
        for p in probs:
            logger.error(f"Missing: {p}")
        return 1

    print_run_header(logger, config, "")

    t0 = time.time()
    sequences = find_sequences(config.base_path, config.first_frame_only)
    if not sequences:
        logger.warning(f"No numeric sequences (>={MIN_SEQUENCE_FRAMES} frames) found. Nothing to do.")
        return 0

    if config.first_frame_only:
        logger.info("First-frame-only mode enabled - processing only the first frame of each sequence")

    # Print discovered sequences table
    logger.section(f"Discovered {len(sequences)} sequences")
    headers = ["#", "Path", "Frames", "Extension"]
    rows = []
    for idx, seq in enumerate(sequences, 1):
        display_path = (seq.rel_dir.as_posix() or ".") + f"/{seq.prefix}*"
        rows.append([f"{idx:02d}", display_path, str(len(seq)), seq.ext])
    logger.table(headers, rows)

    stop_ev = threading.Event()
    pause_ev = threading.Event()
    install_signal_handlers(stop_ev)

    successes = 0
    failures = 0
    produced_total = 0

    def print_status():
        total = successes + failures
        elapsed = time.time() - t0
        state = "PAUSED" if pause_ev.is_set() else "running"
        logger.log(
            f"Processed: {total}/{len(sequences)} | Success: {successes} | Failed: {failures} | Elapsed: {elapsed:.1f}s | State: {state}"
        )
        logger.log("Controls: 'p' pause/resume | 'q' quit")

    try:
        # Start background listener for pause/quit controls
        _ = start_controls_listener(pause_ev, stop_ev)
        logger.info("Processing started. Controls: 'p' pause/resume | 'q' quit")

        if config.run_mode is RunMode.INDIVIDUAL:
            successes, failures, produced_total = process_individual_sequences(
                sequences, config, logger, stop_ev, pause_ev, print_status
            )
        else:
            successes, failures = process_animated_sequences(sequences, config, logger, stop_ev, pause_ev, print_status)

    except GracefulExitError:
        # Message already printed by signal handler
        pass
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)

    total_time = time.time() - t0
    total_sequences = len(sequences)
    avg = (total_time / max(1, total_sequences)) if total_sequences else 0.0

    # Print summary
    logger.section("Summary")
    rows = [
        ["Sequences OK:", str(successes)],
        ["Sequences Failed:", str(failures)],
        ["Total Time:", f"{total_time:.1f}s"],
        ["Avg Time/Seq:", f"{avg:.1f}s"],
    ]
    if config.run_mode is RunMode.INDIVIDUAL:
        rows.append(["Frames Created:", f"{produced_total:,}"])

    for label, value in rows:
        logger.log(f"{label:<20} {value}")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
