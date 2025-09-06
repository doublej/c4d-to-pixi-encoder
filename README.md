# c4d-to-pixi-encoder

Converts Cinema 4D rendered image sequences to WebP/AVIF format optimized for PixiJS applications.

## Installation

```bash
uv sync
```

## Usage

```bash
# Convert sequences to animated WebP
uv run c4d2pixi

# Convert to AVIF with high quality
uv run c4d2pixi --format avif -q high

# Export individual frames
uv run c4d2pixi -i
```[

## Directory Mapping

The tool automatically maps input directories to standardized output names:

### Transitions
| Input Directory | Output Name |
|----------------|-------------|
| `transitions/seatingarea_x_collection/seatingarea_x_collection_night_nobook` | `transition_seat-coll_night-nb` |
| `transitions/seatingarea_x_collection/seatingarea_x_collection_day` | `transition_seat-coll_day` |
| `transitions/seatingarea_x_collection/seatingarea_x_collection_night` | `transition_seat-coll_night` |
| `transitions/seatingarea_x_turntable/seatingarea_x_turntable_day` | `transition_seat-turn_day` |
| `transitions/seatingarea_x_turntable/seatingarea_x_turntable_night` | `transition_seat-turn_night` |
| `transitions/collection_x_turntable/collection_x_turntable_day` | `transition_coll-turn_day` |
| `transitions/collection_x_turntable/collection_x_turntable_night` | `transition_coll-turn_night` |

### Microanimations
| Input Directory | Output Name |
|----------------|-------------|
| `microanimations/book/book_idle_day` | `micro_book-idle_day` |
| `microanimations/book/book_idle_night` | `micro_book-idle_night` |
| `book_transitions_day/Main to shelves` | `micro_book-shelf_day` |
| `book_transitions_day/Main to turntable` | `micro_book-turn_day` |
| `book_transitions_night/Main to shelves` | `micro_book-shelf_night` |
| `book_transitions_night/Main to turntable` | `micro_book-turn_night` |
| `microanimations/turntable/turntable_close_red_record_spinning_day` | `micro_vinyl-spin_day` |
| `microanimations/turntable/turntable_close_red_record_spinning_night` | `micro_vinyl-spin_night` |
| `microanimations/turntable/turntable_close_tonearm_day` | `micro_tonearm_day` |
| `microanimations/turntable/turntable_close_tonearm_night` | `micro_tonearm_night` |
| `microanimations/turntable/turntable_far_black_record_spinning_day` | `micro_vinyl-spin-far_day` |
| `microanimations/turntable/turntable_far_black_record_spinning_night` | `micro_vinyl-spin-far_night` |

## Development

```bash
# Run tests
uv run pytest

# Format code
black src/ tests/
ruff check --fix src/ tests/
```