"""
Consolidated configuration system for c4d-to-pixi-encoder.

This module provides a centralized Pydantic-based configuration system that consolidates
all constants, settings, and data types from across the codebase into a single,
well-organized structure with environment variable support and comprehensive validation.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings

# =============================================================================
# TIME SETTINGS
# =============================================================================

class TimeSettings(BaseModel):
    """Time interval configurations for application operations."""

    status_print_interval: Annotated[int, Field(
        default=5,
        gt=0,
        description="Interval in seconds between status updates"
    )] = 5

    pause_check_interval: Annotated[float, Field(
        default=0.2,
        gt=0.0,
        le=1.0,
        description="Interval in seconds for pause/interrupt checking"
    )] = 0.2


# =============================================================================
# SEQUENCE SETTINGS
# =============================================================================

class SequenceSettings(BaseModel):
    """Configuration for frame sequence processing."""

    min_sequence_frames: Annotated[int, Field(
        default=4,
        ge=1,
        description="Minimum number of frames required to constitute a sequence"
    )] = 4

    default_frame_rate: Annotated[int, Field(
        default=30,
        ge=1,
        le=120,
        description="Default frame rate for animated output (fps)"
    )] = 30


# =============================================================================
# IMAGE PROCESSING SETTINGS
# =============================================================================

class ImageSettings(BaseModel):
    """Image processing and validation settings."""

    default_dpi: Annotated[float, Field(
        default=72.0,
        gt=0.0,
        description="Default DPI value for image metadata"
    )] = 72.0

    crop_alignment_pixels: Annotated[int, Field(
        default=256,
        gt=0,
        description="Pixel alignment for crop operations (power of 2 recommended)"
    )] = 256

    min_alpha_channels: Annotated[int, Field(
        default=4,
        ge=3,
        le=4,
        description="Minimum color channels required for alpha detection (RGBA)"
    )] = 4

    @field_validator('crop_alignment_pixels')
    @classmethod
    def validate_crop_alignment(cls, v):
        """Ensure crop alignment is a power of 2 for optimal performance."""
        if v & (v - 1) != 0:
            raise ValueError(f"crop_alignment_pixels should be a power of 2, got {v}")
        return v


# =============================================================================
# WORKER SETTINGS
# =============================================================================

class WorkerSettings(BaseModel):
    """Worker thread and timeout configurations."""

    max_worker_cap: Annotated[int, Field(
        default=8,
        ge=1,
        le=32,
        description="Maximum number of worker threads for parallel processing"
    )] = 8

    default_timeout_sec: Annotated[int, Field(
        default=300,
        gt=0,
        description="Default timeout in seconds for processing operations"
    )] = 300


# =============================================================================
# FILE EXTENSIONS AND PATHS
# =============================================================================

class FileExtensions(BaseModel):
    """Supported file extensions and directory configurations."""

    supported_image_exts: Annotated[set[str], Field(
        default={".tif", ".tiff", ".png", ".jpg", ".jpeg", ".exr", ".dpx"},
        description="Supported input image file extensions"
    )] = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".exr", ".dpx"}

    supported_output_exts: Annotated[set[str], Field(
        default={".webp", ".avif", ".png", ".jpg", ".jpeg"},
        description="Supported output image file extensions"
    )] = {".webp", ".avif", ".png", ".jpg", ".jpeg"}

    individual_subdir: Annotated[Path, Field(
        default=Path("individual_frames"),
        description="Subdirectory name for individual frame outputs"
    )] = Path("individual_frames")


# =============================================================================
# OUTPUT FORMAT ENUM WITH CODEC METHODS
# =============================================================================

class OutputFormat(str, Enum):
    """Supported output formats with integrated codec configurations."""

    WEBP = "webp"
    AVIF = "avif"

    @property
    def extension(self) -> str:
        """File extension for outputs."""
        return self.value

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
                # Optimized non-alpha settings: aomav1_crf30_s6_420p_aqmode1
                return [
                    "-c:v", "libaom-av1",
                    "-pix_fmt", "yuv444p",
                    "-cpu-used", "6",
                    "-crf", "30",
                    "-aq-mode", "1",
                    "-threads", str(max(1, threads)),
                ]

        raise ValueError(f"Unsupported OutputFormat: {self}")

    def still_codec_args(self, has_alpha: bool = True) -> list[str]:
        """FFmpeg codec arguments for still-frame outputs with optional alpha optimization."""
        if self == OutputFormat.WEBP:
            return ["-vf", "format=rgba", "-c:v", "libwebp", "-pix_fmt", "yuva420p"]

        if self == OutputFormat.AVIF:
            if has_alpha:
                return [
                    "-vf", "format=rgba",
                    "-c:v", "libaom-av1",
                    "-pix_fmt", "yuva444p",
                    "-still-picture", "1",
                ]
            else:
                # Optimized non-alpha settings: aomav1_crf30_s6_420p_aqmode1
                return [
                    "-c:v", "libaom-av1",
                    "-pix_fmt", "yuv420p",
                    "-cpu-used", "6",
                    "-crf", "30",
                    "-aq-mode", "1",
                    "-still-picture", "1",
                ]

        raise ValueError(f"Unsupported OutputFormat: {self}")


# =============================================================================
# RUN MODE ENUM
# =============================================================================

class RunMode(str, Enum):
    """Processing mode: animated or individual frames."""

    ANIMATED = "animated"
    INDIVIDUAL = "individual"


# =============================================================================
# QUALITY CONFIGURATION
# =============================================================================

class Quality(BaseModel):
    """Encoding quality parameters with validation and factory methods."""

    mode: Annotated[Literal["high", "medium", "low", "lossless"], Field(
        description="Quality preset mode"
    )]

    ffmpeg_args: Annotated[list[str], Field(
        description="FFmpeg arguments for this quality level"
    )]

    class Config:
        frozen = True

    @classmethod
    def from_name(cls, name: str, fmt: OutputFormat) -> Quality:
        """Create a Quality instance from preset name and output format.

        Args:
            name: Quality preset name ("high", "medium", "low", "lossless")
            fmt: Target output format

        Returns:
            Quality instance with appropriate FFmpeg arguments

        Raises:
            ValueError: For unknown quality presets or formats
        """
        n = name.lower().strip()

        if fmt == OutputFormat.WEBP:
            if n == "lossless":
                return cls(
                    mode="lossless",
                    ffmpeg_args=["-lossless", "1", "-compression_level", "3"]
                )

            mapping = {"high": "90", "medium": "80", "low": "70"}
            if n not in mapping:
                raise ValueError(f"Unknown quality preset for WebP: {name}")

            quality_value = mapping[n]
            return cls(
                mode=n,
                ffmpeg_args=["-quality", quality_value, "-compression_level", "3"]
            )

        if fmt == OutputFormat.AVIF:
            # Use libaom-av1 settings; include -b:v 0 for CQ mode and -cpu-used for speed
            if n == "lossless":
                return cls(
                    mode="lossless",
                    ffmpeg_args=["-crf", "0", "-b:v", "0", "-cpu-used", "8"]
                )

            mapping = {"high": "23", "medium": "30", "low": "40"}
            if n not in mapping:
                raise ValueError(f"Unknown quality preset for AVIF: {name}")

            crf_value = mapping[n]
            return cls(
                mode=n,
                ffmpeg_args=["-crf", crf_value, "-b:v", "0", "-cpu-used", "8"]
            )

        raise ValueError(f"Unknown format for quality settings: {fmt}")


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


# =============================================================================
# SEQUENCE INFORMATION
# =============================================================================

class SequenceInfo(BaseModel):
    """A detected numeric frame sequence with validation."""

    dir_path: Path
    rel_dir: Path
    prefix: str
    ext: str
    frames: list[Path]

    def __len__(self) -> int:
        """Return number of frames in sequence."""
        return len(self.frames)

    @field_validator('frames')
    @classmethod
    def validate_frames_not_empty(cls, v):
        """Ensure frames list is not empty."""
        if not v:
            raise ValueError("Sequence must contain at least one frame")
        return v

    @field_validator('ext')
    @classmethod
    def validate_extension_format(cls, v):
        """Ensure extension starts with dot."""
        if not v.startswith('.'):
            raise ValueError(f"Extension must start with dot, got: {v}")
        return v


# =============================================================================
# PROCESSING CONFIGURATIONS
# =============================================================================

class Config(BaseModel):
    """Main processing configuration with comprehensive validation."""

    base_path: Path
    output_dir: Path
    format: OutputFormat
    quality: Quality
    run_mode: RunMode
    workers: Annotated[int, Field(ge=1, le=32)]
    timeout_sec: Annotated[int, Field(gt=0)]

    # Optional parameters with defaults
    pad_digits: Annotated[int | None, Field(default=None, ge=1, le=10)] = None
    first_frame_only: bool = False
    extract_scenes: bool = False
    extract_room_stills: bool = True
    crop_alignment: Annotated[int, Field(default=256, gt=0)] = 256

    class Config:
        frozen = True

    @field_validator('crop_alignment')
    @classmethod
    def validate_crop_alignment(cls, v):
        """Ensure crop alignment is a power of 2."""
        if v & (v - 1) != 0:
            raise ValueError(f"crop_alignment must be a power of 2, got {v}")
        return v

    @field_validator('base_path', 'output_dir')
    @classmethod
    def validate_paths(cls, v):
        """Convert relative paths to absolute."""
        if not v.is_absolute():
            v = v.resolve()
        return v


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


# =============================================================================
# MAIN APPLICATION CONFIGURATION
# =============================================================================

class AppConfig(BaseSettings):
    """
    Main application configuration with environment variable support.

    All settings can be overridden via environment variables with C4D2PIXI_ prefix.
    Example: C4D2PIXI_TIME__STATUS_PRINT_INTERVAL=10
    """

    # Component configurations
    time: TimeSettings = TimeSettings()
    sequence: SequenceSettings = SequenceSettings()
    image: ImageSettings = ImageSettings()
    worker: WorkerSettings = WorkerSettings()
    file_extensions: FileExtensions = FileExtensions()
    codec: CodecSettings = CodecSettings()

    class Config:
        env_prefix = "C4D2PIXI_"
        env_nested_delimiter = "__"
        case_sensitive = False

    def get_supported_input_extensions(self) -> set[str]:
        """Get supported input file extensions."""
        return self.file_extensions.supported_image_exts

    def get_supported_output_extensions(self) -> set[str]:
        """Get supported output file extensions."""
        return self.file_extensions.supported_output_exts

    def is_supported_input_file(self, path: Path) -> bool:
        """Check if file has supported input extension."""
        return path.suffix.lower() in self.file_extensions.supported_image_exts

    def is_supported_output_file(self, path: Path) -> bool:
        """Check if file has supported output extension."""
        return path.suffix.lower() in self.file_extensions.supported_output_exts


# =============================================================================
# DEFAULT INSTANCE
# =============================================================================

# Create default configuration instance for easy importing
app_config = AppConfig()

# Export commonly used constants for backward compatibility
STATUS_PRINT_INTERVAL = app_config.time.status_print_interval
PAUSE_CHECK_INTERVAL = app_config.time.pause_check_interval
MIN_SEQUENCE_FRAMES = app_config.sequence.min_sequence_frames
DEFAULT_FRAME_RATE = app_config.sequence.default_frame_rate
DEFAULT_DPI = app_config.image.default_dpi
CROP_ALIGNMENT_PIXELS = app_config.image.crop_alignment_pixels
MIN_ALPHA_CHANNELS = app_config.image.min_alpha_channels
MAX_WORKER_CAP = app_config.worker.max_worker_cap
DEFAULT_TIMEOUT_SEC = app_config.worker.default_timeout_sec
SUPPORTED_IMAGE_EXTS = app_config.file_extensions.supported_image_exts
SUPPORTED_OUTPUT_EXTS = app_config.file_extensions.supported_output_exts
INDIVIDUAL_SUBDIR = app_config.file_extensions.individual_subdir


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_config_from_env() -> AppConfig:
    """Create a new configuration instance from environment variables."""
    return AppConfig()

def get_quality_preset(name: str, format_type: OutputFormat) -> Quality:
    """Get quality preset for given name and format."""
    return Quality.from_name(name, format_type)

def get_default_frame_rate() -> int:
    """Get default frame rate setting."""
    return app_config.sequence.default_frame_rate
