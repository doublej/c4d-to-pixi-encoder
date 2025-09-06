# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SS Image Processor (webpseq) - A Python tool for converting numeric frame sequences to WebP/AVIF format, supporting both animated sequences and per-frame output with advanced features.

## Development Commands

### Running the Application
```bash
# Main script (simplified version without Rich UI)
uv run python main.py [options]

# Main script with Rich UI
uv run python main_with_rich.py [options]

# Example: Convert frames to WebP
uv run python main.py -f webp -q mid /path/to/frames

# Example: Convert to individual frames with cropping
uv run python main.py -m individual -f avif -q high --crop /path/to/frames
```

### Testing
```bash
# Run all tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_alpha_exists.py

# Run with coverage
uv run pytest --cov=. tests/
```

### Dependencies
```bash
# Install/sync dependencies
uv sync

# Add new dependency
uv add <package-name>
```

## Core Architecture

### Entry Points

- **main.py** - Simplified CLI without Rich UI dependencies
- **main_with_rich.py** - Full-featured version with Rich progress UI
- **main_simple.py** - Backup simplified version

### Key Components

1. **Core Processing Pipeline**
   - `OutputFormat` enum: webp/avif selection
   - `RunMode` enum: animated/individual processing
   - `Quality` enum: low/mid/high/max/webp_lossless presets
   - `Config` dataclass: centralized configuration
   - `SequenceInfo` dataclass: frame sequence metadata

2. **Utility Modules**
   - **misc.py**: FFmpeg interface, alpha detection, crop calculations, subprocess management
   - **dpi_utils.py**: DPI normalization and metadata (`DpiInfo` dataclass)
   - **combine_metadata.py**: JSON metadata aggregation (`PairPaths` for offset/DPI pairs)
   - **persistent_output.py**: Console output persistence (`PersistentStream`)
   - **simple_logger.py**: Lightweight logging alternative to Rich

3. **Supporting Tools**
   - **build_viewer.py**: HTML viewer generation for frame sequences
   - **log_viewer.py**: Scrollable log visualization with Rich

### Processing Flow

1. **Discovery**: Find frame sequences via pattern matching (e.g., `frame_0001.png`)
2. **Validation**: Check alpha channels, validate TIFF files, compute dimensions
3. **Optimization**: Optional 256-aligned cropping for transparency
4. **Encoding**: FFmpeg pipelines with format-specific codec arguments
5. **Metadata**: Generate offset/DPI JSON files for downstream tools
6. **Monitoring**: Live progress tracking (Rich) or simple logging

### FFmpeg Integration

- WebP animated: `libwebp` codec with `yuva420p` pixel format
- AVIF animated: `libaom-av1` codec with `yuva444p` pixel format
- Individual frames: Same codecs with still-picture flags
- Crop filters applied via FFmpeg's `crop` video filter
- Concatenation via `ffconcat` demuxer for frame sequences

### Testing Strategy

Tests focus on critical path validation:
- Alpha channel detection (PNG/TIFF with various channel configurations)
- Metadata combination and JSON structure
- Console output persistence and line handling
- Image dimension and crop calculations