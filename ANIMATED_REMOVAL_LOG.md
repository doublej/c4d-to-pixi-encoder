# ANIMATED_REMOVAL_LOG.md

This document logs all code removed during the animated mode removal process.

## 1. Configuration (src/c4d2pixi/config.py)

### Lines 214-222: RunMode enum removed
```python
# =============================================================================
# RUN MODE ENUM
# =============================================================================

class RunMode(str, Enum):
    """Processing mode: animated or individual frames."""

    ANIMATED = "animated"
    INDIVIDUAL = "individual"
```

### Lines 154-184: animated_codec_args method removed
```python
    def animated_codec_args(self, threads: int, has_alpha: bool = True) -> list[str]:
        """FFmpeg codec arguments for animated outputs with optional alpha optimization."""
        if self == OutputFormat.WEBP:
            return [
                "-c:v", "libwebp",
                "-loop", "0",
                "-vf", "format=rgba",
                "-pix_fmt", "yuva420p",
                "-threads", str(max(1, threads)),
            ]

        if self == OutputFormat.AVIF:
            if has_alpha:
                return [
                    "-c:v", "libaom-av1",
                    "-vf", "format=rgba",
                    "-pix_fmt", "yuva444p",
                    "-threads", str(max(1, threads)),
                ]
            else:
                # Optimized non-alpha settings: aomav1_s6_420p_aqmode1
                return [
                    "-c:v", "libaom-av1",
                    "-pix_fmt", "yuv444p",
                    "-cpu-used", "6",
                    "-aq-mode", "1",
                    "-threads", str(max(1, threads)),
                ]

        raise ValueError(f"Unsupported OutputFormat: {self}")
```

### Line 365: run_mode field removed from Config class
```python
    run_mode: RunMode
```

## 2. FFmpeg Command Building (src/c4d2pixi/processing/ffmpeg.py)

### Line 12: DEFAULT_FRAME_RATE import removed
```python
from ..config import DEFAULT_FRAME_RATE, OutputFormat, Quality
```
Changed to:
```python
from ..config import OutputFormat, Quality
```

### Lines 19-59: build_animated_cmd method removed
```python
    @staticmethod
    def build_animated_cmd(
        list_file: Path,
        out_path: Path,
        quality: Quality,
        threads: int,
        fmt: OutputFormat,
        crop_rect: tuple[int, int, int, int] | None = None,
        has_alpha: bool = True,
    ) -> list[str]:
        """Create ffmpeg command for an animated image sequence, with optional crop and alpha optimization."""
        base = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-an",
            "-vsync",
            "0",
            "-r",
            str(DEFAULT_FRAME_RATE),
        ]
        codec_args = fmt.animated_codec_args(threads, has_alpha)
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
```

## 3. Utils - Subprocess (src/c4d2pixi/utils/subprocess.py)

### Lines 151-179: write_ffconcat_file function removed
```python
def write_ffconcat_file(frame_files: list[Path], temp_dir: Path) -> Path:
    """
    Writes an ffconcat file for ffmpeg to use when concatenating frames.
    Returns the path to the created concat file.
    """
    list_file = temp_dir / "ffconcat.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        f.write("ffconcat version 1.0\n")
        for frame_file in frame_files:
            # Use absolute paths to avoid issues with relative paths
            abs_path = frame_file.resolve()
            # Escape special characters for ffmpeg
            escaped_path = str(abs_path).replace("\\", "/")
            f.write(f"file '{escaped_path}'\n")
    return list_file
```

### Lines 181-194: DEFAULT_FRAME_RATE constant removed
```python
# =============================================================================
# FRAME RATE CONSTANTS FOR ANIMATED ENCODING
# =============================================================================

# Frame rate for animated WebP/AVIF files.
# Some formats don't support very low frame rates; 15fps is a good balance between
# smoothness and file size for most uses.
DEFAULT_FRAME_RATE = 15
```

## 4. Sequence Processing (src/c4d2pixi/processing/sequence.py)

### Line 5: Module docstring updated
```python
# Before:
both for individual frames and animated sequences.
# After:
for individual frames.
```

### Line 13: tempfile import removed
```python
import tempfile
```

### Line 18: AnimatedEncodeConfig import removed
```python
# Before:
from ..config import AnimatedEncodeConfig, OutputFormat, Quality, SequenceInfo
# After:
from ..config import OutputFormat, Quality, SequenceInfo
```

### Line 21: write_ffconcat_file import removed
```python
# Before:
from ..utils.subprocess import run_subprocess, write_ffconcat_file
# After:
from ..utils.subprocess import run_subprocess
```

### Lines 33-72: encode_animated_task method removed
```python
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
```

## Quality Settings Fixed

### AVIF hardcoded CRF values removed:
- Lines 179 & 205: Removed `"-crf", "30"` from both animated_codec_args and still_codec_args non-alpha cases
- Updated comments from "aomav1_crf30_s6_420p_aqmode1" to "aomav1_s6_420p_aqmode1"

## 3. Configuration Cleanup (src/c4d2pixi/config.py) - Additional Removals

### Lines 259-277: CodecSettings class removed
```python
# =============================================================================
# CODEC SETTINGS
# =============================================================================

class CodecSettings(BaseModel):
    """Centralized codec configuration for different output formats."""

    webp_animated_base: list[str] = ["-c:v", "libwebp", "-loop", "0"]
    webp_still_base: list[str] = ["-c:v", "libwebp"]
    webp_pixel_format: str = "yuva420p"

    avif_animated_base: list[str] = ["-c:v", "libaom-av1"]
    avif_still_base: list[str] = ["-c:v", "libaom-av1", "-still-picture", "1"]
    avif_alpha_pixel_format: str = "yuva444p"
    avif_no_alpha_pixel_format: str = "yuv444p"

    # AVIF optimization settings for non-alpha content
    avif_no_alpha_optimized: dict[str, str] = {
        "cpu_used": "6",
        "crf": "30",
        "aq_mode": "1"
    }
```

### Line 412: codec reference removed from AppConfig
```python
    codec: CodecSettings = CodecSettings()
```

### Line 447: DEFAULT_FRAME_RATE constant removed
```python
DEFAULT_FRAME_RATE = app_config.sequence.default_frame_rate
```

### Lines 470-472: get_default_frame_rate function removed
```python
def get_default_frame_rate() -> int:
    """Get default frame rate setting."""
    return app_config.sequence.default_frame_rate
```

## 4. Main CLI (src/c4d2pixi/cli/main.py)

### Line 3: Updated module docstring
```python
c4d2pixi: Convert Cinema 4D rendered sequences to WebP/AVIF (animated or per-frame).
```
Changed to:
```python
c4d2pixi: Convert Cinema 4D rendered sequences to WebP/AVIF per-frame.
```

### Line 16: concurrent.futures import removed
```python
import concurrent.futures as futures
```

### Line 36: RunMode import removed
```python
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
```
Changed to:
```python
from ..config import (
    MIN_SEQUENCE_FRAMES,
    PAUSE_CHECK_INTERVAL,
    STATUS_PRINT_INTERVAL,
    Config,
    OutputFormat,
    Quality,
    SequenceInfo,
)
```

### Line 63: Updated argparse description
```python
        description="Convert numeric frame sequences to WebP/AVIF (animated or per-frame).",
```
Changed to:
```python
        description="Convert numeric frame sequences to WebP/AVIF per-frame.",
```

### Lines 74-76: --individual-frames CLI argument removed
```python
    p.add_argument(
        "-i", "--individual-frames", action="store_true", help="Export individual files per frame instead of animated"
    )
```

### Line 104: run_mode assignment removed from build_config
```python
    run_mode = RunMode.INDIVIDUAL if args.individual_frames else RunMode.ANIMATED
```

### Line 110: run_mode field removed from Config constructor
```python
        run_mode=run_mode,
```

### Lines 376-394: animated_output_path function removed
```python
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
```

### Line 466: Mode display logic simplified
```python
        ["Mode:", "individual-frames" if config.run_mode is RunMode.INDIVIDUAL else "animated"],
```
Changed to:
```python
        ["Mode:", "individual-frames"],
```

### Lines 538-658: process_animated_sequences function removed
```python
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
```

### Lines 733-738: RunMode condition removed from main processing logic
```python
        if config.run_mode is RunMode.INDIVIDUAL:
            successes, failures, produced_total = process_individual_sequences(
                sequences, config, logger, stop_ev, pause_ev, print_status
            )
        else:
            successes, failures = process_animated_sequences(sequences, config, logger, stop_ev, pause_ev, print_status)
```
Changed to:
```python
        successes, failures, produced_total = process_individual_sequences(
            sequences, config, logger, stop_ev, pause_ev, print_status
        )
```

### Lines 758-759: RunMode condition removed from summary display
```python
    if config.run_mode is RunMode.INDIVIDUAL:
        rows.append(["Frames Created:", f"{produced_total:,}"])
```
Changed to:
```python
    rows.append(["Frames Created:", f"{produced_total:,}"])
```

## 5. Utilities and Tests

### Lines 25-47: write_ffconcat_file function removed from src/c4d2pixi/utils/subprocess.py
```python
def write_ffconcat_file(frame_paths: list[Path], target_dir: Path) -> Path:
    """Write FFmpeg concat demuxer file for frame sequence.

    Args:
        frame_paths: List of frame paths in order
        target_dir: Directory to write concat file

    Returns:
        Path to created concat file
    """
    concat_path = target_dir / "input.ffconcat"

    with open(concat_path, "w") as f:
        f.write("ffconcat version 1.0\n")
        for frame_path in frame_paths:
            escaped = str(frame_path).replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")
            f.write("duration 0.033\n")  # 30fps default

        # Repeat last frame for duration
        if frame_paths:
            escaped = str(frame_paths[-1]).replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")

    return concat_path
```

### Lines 354-392: AnimatedEncodeConfig class removed from src/c4d2pixi/config.py
```python
class AnimatedEncodeConfig(BaseModel):
    """Configuration for animated sequence encoding operations."""

    frame_paths: list[str]
    out_path: str
    quality_mode: str
    threads: Annotated[int, Field(ge=1, le=32)]
    format_value: str

    # Optional parameters
    crop_rect: tuple[int, int, int, int] | None = None
    offsets_json: str | None = None
    seq_orig_w: int | None = None
    seq_orig_h: int | None = None
    has_alpha: bool = True

    class Config:
        frozen = True

    @field_validator('frame_paths')
    @classmethod
    def validate_frame_paths_not_empty(cls, v):
        """Ensure frame paths list is not empty."""
        if not v:
            raise ValueError("frame_paths cannot be empty")
        return v

    @field_validator('crop_rect')
    @classmethod
    def validate_crop_rect(cls, v):
        """Validate crop rectangle dimensions."""
        if v is not None:
            x, y, w, h = v
            if w <= 0 or h <= 0:
                raise ValueError(f"Crop dimensions must be positive: width={w}, height={h}")
            if x < 0 or y < 0:
                raise ValueError(f"Crop coordinates must be non-negative: x={x}, y={y}")
        return v
```

### Tests Directory
No animated-related tests were found in the tests/ directory. All existing test files (test_alpha_exists.py, test_combine_metadata.py, test_persistent_output.py) contain only utility function tests unrelated to animated processing.