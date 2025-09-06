"""
Sequence processing module for SS Image Processor.

This module handles the encoding of image sequences into WebP/AVIF formats,
both for individual frames and animated sequences.
"""

from __future__ import annotations

import concurrent.futures as futures
import contextlib
import itertools
import tempfile
import time
from pathlib import Path

from ..cli.metadata import combine_to_metadata, write_dpi_json, write_offset_json
from ..core.mapping import DirectoryMapper, OutputCategory
from ..core.types import AnimatedEncodeConfig, OutputFormat, Quality, SequenceInfo
from ..utils.dpi import dpi_dict, read_sequence_dpi
from ..utils.subprocess import run_subprocess, write_ffconcat_file
from .crop import compute_sequence_aligned_crop
from .ffmpeg import FFmpegCommandBuilder
from .image import check_alpha_exists, image_dimensions, split_tiff_channels, validate_tiff_file

INDIVIDUAL_SUBDIR = Path("individual_frames")


class SequenceProcessor:
    """Processor for encoding image sequences."""

    @staticmethod
    def encode_animated_task(config: AnimatedEncodeConfig) -> tuple[bool, str, int | None]:
        """
        Child-process-safe function to encode one sequence to an animated image.
        Returns (success, message, output_size_bytes|None).
        """
        try:
            fmt = OutputFormat(config.format_value)
            q = Quality.from_name(config.quality_mode, fmt)
            out = Path(config.out_path)
            out.parent.mkdir(parents=True, exist_ok=True)

            # Reuse if exists
            if out.exists():
                size = out.stat().st_size
                return True, f"SKIP exists: {out.name} ({size} bytes)", size

            with tempfile.TemporaryDirectory(prefix="c4d2pixi_") as td:
                list_file = write_ffconcat_file([Path(p) for p in config.frame_paths], Path(td))
                cmd = FFmpegCommandBuilder.build_animated_cmd(
                    list_file, out, q, config.threads, fmt, config.crop_rect, config.has_alpha
                )
                # Avoid printing from subprocess; parent logs the command via returned message
                code, pretty = run_subprocess(cmd, log=False)
                if code != 0:
                    return False, f"ffmpeg failed: {pretty}", None

            size = out.stat().st_size if out.exists() else None
            # Optionally write offsets JSON next to the animated output
            if config.offsets_json and config.seq_orig_w is not None and config.seq_orig_h is not None:
                if config.crop_rect is not None:
                    cx, cy, cw, ch = config.crop_rect
                else:
                    cx = cy = 0
                    cw = config.seq_orig_w
                    ch = config.seq_orig_h
                write_offset_json(Path(config.offsets_json), cx, cy, cw, ch, config.seq_orig_w, config.seq_orig_h)
            # Include the command string in success message for visibility in parent logs
            return True, f"DONE: {out.name} ({size} bytes) | cmd: {pretty}", size
        except Exception as ex:
            return False, f"Exception: {type(ex).__name__}: {ex}", None

    @staticmethod
    def encode_one_frame(
        src: Path,
        dst: Path,
        quality: Quality,
        fmt: OutputFormat,
        crop_rect: tuple[int, int, int, int] | None,
        seq_orig_w: int,
        seq_orig_h: int,
    ) -> tuple[bool, str]:
        """Encode one frame with an optional sequence-wide crop and write offsets JSON."""
        if dst.exists():
            return True, f"skip {dst.name}"

        temp_src_file: Path | None = None
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
            cmd = FFmpegCommandBuilder.build_individual_cmd(source_to_encode, dst, quality, fmt, crop_rect)
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

    @staticmethod
    def encode_sequence_individual(
        seq: SequenceInfo,
        out_root: Path,
        quality: Quality,
        threads: int,
        timeout_sec: int,
        fmt: OutputFormat,
        pad_digits: int | None = None,
    ) -> tuple[bool, str, int]:
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
        tasks: list[tuple[Path, Path]] = []
        for idx, f in enumerate(seq.frames, start=1):
            dst = SequenceProcessor._individual_output_path(
                out_root, seq, f, fmt, frame_index=idx, pad_digits=pad_digits
            )
            if not dst.exists():
                tasks.append((f, dst))

        total = len(seq.frames)
        missing = len(tasks)
        if missing == 0:
            return True, f"SKIP all {total} frames exist", 0

        produced = 0
        start = time.time()
        ok = True
        msgs: list[str] = []

        def work(pair: tuple[Path, Path]) -> tuple[bool, str]:
            src, dst = pair
            return SequenceProcessor.encode_one_frame(src, dst, quality, fmt, crop_tuple, ow, oh)

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
        
        # Get the output naming convention
        naming = DirectoryMapper.get_output_naming(seq.rel_dir, seq.prefix)
        
        if naming:
            # Use mapped naming convention for metadata
            rel_dir = out_root / naming.category_dir
            # Generate a consolidated metadata.json for the category
            metadata_path = rel_dir / "metadata.json"
            
            # Write metadata for this sequence
            with contextlib.suppress(Exception):
                # Read existing metadata if it exists
                metadata = {}
                if metadata_path.exists():
                    import json
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                
                # Add this sequence's metadata
                sequence_key = naming.prefix
                metadata[sequence_key] = {
                    "offset": {"x": cx, "y": cy},
                    "dimensions": {"width": cw, "height": ch},
                    "original": {"width": ow, "height": oh},
                    "frames": len(seq.frames),
                    "dpi": dpi_dict(read_sequence_dpi(seq.frames)) if seq.frames else {}
                }
                
                # Write updated metadata
                import json
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
        else:
            # Fall back to original metadata handling
            rel_dir = out_root / INDIVIDUAL_SUBDIR / seq.rel_dir
            base_name_str = seq.prefix.strip() or seq.dir_path.name
            base_name = base_name_str.rstrip("._-")
            json_path = rel_dir / f"{base_name}_{seq.ext.lstrip('.')}.json"
            with contextlib.suppress(Exception):
                write_offset_json(json_path, cx, cy, cw, ch, ow, oh)
            # DPI sidecar written once per sequence
            with contextlib.suppress(Exception):
                dpi_info = read_sequence_dpi(seq.frames)
                dpi_sidecar = rel_dir / f"{base_name}_{seq.ext.lstrip('.')}.dpi.json"
                write_dpi_json(dpi_sidecar, dpi_dict(dpi_info))
            # Combine sidecars into metadata.json per sequence
            with contextlib.suppress(Exception):
                combine_to_metadata(rel_dir, f"{base_name}_{seq.ext.lstrip('.')}", output_name="metadata.json")
        return (
            ok,
            f"{produced}/{missing} created in {elapsed:.1f}s; " + " | ".join(itertools.islice(msgs, 0, 8)),
            produced,
        )

    @staticmethod
    def _individual_output_path(
        output_root: Path,
        seq: SequenceInfo,
        frame: Path,
        fmt: OutputFormat,
        frame_index: int | None = None,
        pad_digits: int | None = None,
    ) -> Path:
        """
        Place per-frame outputs using new naming convention.
        """
        # Check if we have a mapping for this directory
        naming = DirectoryMapper.get_output_naming(seq.rel_dir, seq.prefix)
        
        if naming:
            # Use mapped naming convention
            rel_dir = output_root / naming.category_dir
            # For individual frames, use frame index starting from 0
            if frame_index is not None:
                # Convert 1-based index to 0-based for output
                zero_based_index = frame_index - 1
                name = naming.format_frame_name(zero_based_index, fmt.extension)
            else:
                # Extract frame number from filename
                import re
                match = re.search(r'(\d+)', frame.stem)
                if match:
                    frame_num = int(match.group(1))
                    name = naming.format_frame_name(frame_num, fmt.extension)
                else:
                    name = f"{frame.stem}.{fmt.extension}"
        else:
            # Fall back to original naming
            rel_dir = output_root / INDIVIDUAL_SUBDIR / seq.rel_dir
            if pad_digits and frame_index is not None:
                name = f"{frame_index:0{pad_digits}d}.{fmt.extension}"
            else:
                name = f"{frame.stem}.{fmt.extension}"
        
        return rel_dir / name
