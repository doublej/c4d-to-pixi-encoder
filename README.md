# c4d-to-pixi-encoder

A high-performance Python tool for converting Cinema 4D rendered image sequences to WebP/AVIF format, optimized for PixiJS applications with transparency support and metadata preservation.

## Features

- 🎬 **Animated & Individual Output**: Create animated WebP/AVIF or export individual frames
- 🖼️ **Smart Alpha Handling**: Automatic alpha channel detection and optimization
- ✂️ **256-Aligned Cropping**: Optimized transparent edge cropping for smaller files
- 🔧 **Parallel Processing**: Multi-threaded/process execution for maximum performance
- 📊 **Metadata Preservation**: DPI and offset information tracking
- 🎯 **First-Frame Mode**: Quick preview mode for testing
- 🎨 **Quality Presets**: Low, Medium, High, and Lossless encoding options

## Installation

### Prerequisites

- Python 3.10+
- FFmpeg with libwebp and libaom-av1 support
- UV package manager (recommended)

### Setup

```bash
# Clone the repository
git clone git@github.com:doublej/c4d-to-pixi-encoder.git
cd c4d-to-pixi-encoder

# Install dependencies with UV (recommended)
uv sync

# Or install in development mode
uv pip install -e .
```

## Usage

### Basic Command

```bash
# Convert all sequences to animated WebP (default)
uv run c4d2pixi

# Convert to AVIF format with high quality
uv run c4d2pixi --format avif -q high

# Export individual frames instead of animated
uv run c4d2pixi -i --format webp
```

### Command-Line Options

```bash
uv run c4d2pixi [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `-p, --base-path PATH` | Base directory to scan for sequences | Current directory |
| `-o, --output-dir PATH` | Output directory | `outputs/{format}_renders` |
| `-q, --quality` | Quality preset: `low`, `medium`, `high`, `lossless` | `high` |
| `--format` | Output format: `webp`, `avif` | `webp` |
| `-i, --individual-frames` | Export individual frames instead of animated | False |
| `--first-frame-only` | Process only the first frame of each sequence | False |
| `--pad-digits N` | Zero-padding width for frame numbers | None (use original) |
| `-w, --max-workers N` | Maximum parallel workers (capped at 8) | CPU count |
| `-s, --sequential` | Force sequential processing | False |
| `--check-tools` | Verify external tools and exit | False |

### Examples

#### Convert to Animated WebP
```bash
# High quality animated WebP (default)
uv run c4d2pixi /path/to/frames

# Medium quality for smaller files
uv run c4d2pixi -q medium /path/to/frames

# Lossless WebP (large files)
uv run c4d2pixi -q lossless /path/to/frames
```

#### Convert to AVIF
```bash
# Animated AVIF with automatic alpha optimization
uv run c4d2pixi --format avif /path/to/frames

# Individual AVIF frames with cropping
uv run c4d2pixi -i --format avif --crop /path/to/frames
```

#### Individual Frame Export
```bash
# Export each frame as individual WebP
uv run c4d2pixi -i /path/to/frames

# With custom padding (e.g., frame_00001.webp)
uv run c4d2pixi -i --pad-digits 5 /path/to/frames
```

#### Quick Testing
```bash
# Process only first frame of each sequence (fast preview)
uv run c4d2pixi --first-frame-only /path/to/frames

# Check if FFmpeg is properly installed
uv run c4d2pixi --check-tools
```

## Project Structure

```
src/ss_image_processor/
├── cli/            # Command-line interfaces
│   ├── main.py     # Main entry point
│   ├── viewer.py   # HTML viewer generator
│   └── metadata.py # Metadata combination tool
├── core/           # Core types and configuration
│   ├── types.py    # Data classes and enums
│   └── constants.py # Centralized constants
├── processing/     # Image and video processing
│   ├── sequence.py # Sequence encoding logic
│   ├── ffmpeg.py   # FFmpeg command builder
│   ├── image.py    # Image utilities
│   └── crop.py     # Cropping algorithms
├── utils/          # Utility functions
│   ├── path.py     # Path and sequence discovery
│   ├── dpi.py      # DPI normalization
│   ├── json.py     # JSON utilities
│   └── subprocess.py # Process management
└── output/         # Output and logging
    ├── logger.py   # Simple logging
    └── persistent.py # Console output persistence
```

## Development

### Setup Development Environment

```bash
# Install with development dependencies
uv add --dev black ruff mypy pytest pytest-cov pre-commit

# Install pre-commit hooks
pre-commit install

# Run tests
uv run pytest tests/

# Format code
black src/ tests/
ruff check --fix src/ tests/

# Type checking
uv run mypy src/
```

### Code Quality Tools

The project uses:
- **Black**: Code formatting (120 char line length)
- **Ruff**: Fast Python linter
- **MyPy**: Static type checking
- **Pre-commit**: Git hooks for automatic checks
- **Pytest**: Testing framework

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=ss_image_processor

# Run specific test
uv run pytest tests/test_alpha_exists.py
```

## Features in Detail

### Sequence Detection
- Automatically finds numbered image sequences (e.g., `frame_001.png`, `frame_002.png`)
- Supports: TIFF, PNG, JPG, EXR, DPX formats
- Minimum 4 frames required for sequence detection

### Alpha Channel Optimization
- Automatic detection of alpha channels in PNG/TIFF files
- Optimized encoding for images without alpha
- 256-pixel aligned cropping for transparent edges

### Metadata Handling
- Preserves DPI information from source images
- Generates offset JSON for cropped images
- Combines metadata for downstream tools

### Performance
- Multi-process encoding for animated sequences
- Multi-threaded individual frame processing
- Intelligent caching and skip logic for existing files

## Troubleshooting

### FFmpeg Not Found
```bash
# Check if FFmpeg is installed
uv run c4d2pixi --check-tools

# Install FFmpeg (macOS)
brew install ffmpeg

# Install FFmpeg (Ubuntu/Debian)
sudo apt install ffmpeg
```

### Out of Memory
- Reduce worker count: `-w 2`
- Use sequential mode: `-s`
- Process smaller batches

### Quality Issues
- Use higher quality preset: `-q high` or `-q lossless`
- AVIF generally provides better quality than WebP at same file size
- For transparency, ensure source images have proper alpha channels

## License

MIT