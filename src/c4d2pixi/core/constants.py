"""Centralized constants for the application."""

from pathlib import Path

# Time intervals
STATUS_PRINT_INTERVAL = 5
PAUSE_CHECK_INTERVAL = 0.2

# Sequence configuration
MIN_SEQUENCE_FRAMES = 4
DEFAULT_FRAME_RATE = 30

# Image processing
DEFAULT_DPI = 72.0
CROP_ALIGNMENT_PIXELS = 256

# Worker configuration
MAX_WORKER_CAP = 8
DEFAULT_TIMEOUT_SEC = 300

# File extensions
SUPPORTED_IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".exr", ".dpx"}
SUPPORTED_OUTPUT_EXTS = {".webp", ".avif", ".png", ".jpg", ".jpeg"}

# Directory names
INDIVIDUAL_SUBDIR = Path("individual_frames")

# Alpha channel configuration
MIN_ALPHA_CHANNELS = 4
