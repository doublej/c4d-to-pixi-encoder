# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SS Image Processor (webpseq) - A Python tool for converting numeric frame sequences to WebP/AVIF format, supporting both animated sequences and per-frame output with advanced features like alpha channel handling, cropping, and metadata tracking.

## Development Commands

### Running the Application
```bash
# Main script
uv run python main.py [options]

# Example usage
uv run python main.py -f webp -q mid /path/to/frames
```

### Testing
```bash
# Run all tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_alpha_exists.py

# Run with verbose output
uv run pytest -v tests/
```

### Dependencies
```bash
# Install dependencies
uv pip install -r pyproject.toml

# Add new dependency
uv add <package-name>

# Sync dependencies
uv sync
```

## Core Architecture

### Main Components

1. **main.py** - Core orchestration logic with:
   - Enums for `OutputFormat` (webp/avif) and `RunMode` (single/batch/triage)
   - `Config` dataclass for configuration management
   - Frame sequence detection and processing
   - Rich UI with live progress monitoring
   - Thread pool for concurrent processing

2. **misc.py** - Utility functions including:
   - Tool availability checking (ffmpeg)
   - Path utilities and stem parsing
   - TIFF/PNG alpha channel detection
   - Subprocess runners for external tools
   - Image cropping and offset calculations
   - Transparency edge detection

3. **persistent_output.py** - Console output persistence utility

4. **dpi_utils.py** - DPI/resolution handling utilities

5. **combine_metadata.py** - Metadata aggregation and processing

6. **build_viewer.py** - Viewer application builder

### Key Design Patterns

- **Atomic Functions**: Small, single-purpose functions (5-25 lines) with clear contracts
- **Side Effects at Edges**: I/O operations kept at boundaries, core logic is pure
- **Early Validation**: Input validation at function entry points
- **Dataclass Configuration**: Settings passed via `Config` dataclass
- **Enum-based State**: Using enums for output formats and run modes

### External Dependencies

- **FFmpeg**: Required for WebP/AVIF conversion
- **Rich**: Terminal UI and progress monitoring
- **Pillow**: Image manipulation
- **OpenCV**: Advanced image processing
- **NumPy**: Array operations

### File Processing Flow

1. Scan directories for frame sequences (PNG/TIFF files)
2. Parse filenames to extract frame numbers
3. Detect alpha channels for proper encoding
4. Apply optional cropping based on transparency detection
5. Convert using FFmpeg with format-specific pipelines
6. Track progress with Rich UI
7. Generate metadata and offset JSON files

### Testing Approach

- Unit tests in `tests/` directory covering:
  - Alpha channel detection (PNG/TIFF)
  - Metadata combination
  - Persistent output functionality
- Tests use pytest with tmp_path fixtures
- Mock image generation for test cases

## Important Notes from AGENTS.md

- Write atomic, production-quality code
- Use UV for running the project
- Follow function granularity rules (5-25 lines per function)
- Prefer clarity over cleverness
- Minimize external dependencies
- Keep I/O and side effects at the edges