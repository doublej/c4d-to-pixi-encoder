"""
Directory-to-output mapping configuration for SS Image Processor.

This module defines the mapping between input directory names and output file naming conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import NamedTuple


class OutputCategory(Enum):
    """Output category for processed files."""

    TRANSITION = "transition"
    MICROANIMATION = "micro"
    SCENE = "scene"


class DirectoryMapping(NamedTuple):
    """Maps a directory pattern to output naming."""

    input_pattern: str  # Directory name or path pattern
    category: OutputCategory
    base_name: str  # Base name for output files
    time_of_day: str | None  # 'day' or 'night' or None
    variant: str | None  # Additional variant identifier


# Define the mapping configuration
DIRECTORY_MAPPINGS = [
    # TRANSITIONS - with nested structure (order matters - more specific first)
    DirectoryMapping("transitions/seatingarea_x_collection/seatingarea_x_collection_night_nobook", OutputCategory.TRANSITION, "seat-coll", "night-nb", None),
    DirectoryMapping("transitions/seatingarea_x_collection/seatingarea_x_collection_day", OutputCategory.TRANSITION, "seat-coll", "day", None),
    DirectoryMapping("transitions/seatingarea_x_collection/seatingarea_x_collection_night", OutputCategory.TRANSITION, "seat-coll", "night", None),
    DirectoryMapping("transitions/seatingarea_x_turntable/seatingarea_x_turntable_day", OutputCategory.TRANSITION, "seat-turn", "day", None),
    DirectoryMapping("transitions/seatingarea_x_turntable/seatingarea_x_turntable_night", OutputCategory.TRANSITION, "seat-turn", "night", None),
    DirectoryMapping("transitions/collection_x_turntable/collection_x_turntable_day", OutputCategory.TRANSITION, "coll-turn", "day", None),
    DirectoryMapping("transitions/collection_x_turntable/collection_x_turntable_night", OutputCategory.TRANSITION, "coll-turn", "night", None),

    # MICROANIMATIONS - Book with nested structure
    DirectoryMapping("microanimations/book/book_idle_day", OutputCategory.MICROANIMATION, "book-idle", "day", None),
    DirectoryMapping("microanimations/book/book_idle_night", OutputCategory.MICROANIMATION, "book-idle", "night", None),
    DirectoryMapping("book_transitions_day/Main to shelves", OutputCategory.MICROANIMATION, "book-shelf", "day", None),
    DirectoryMapping("book_transitions_day/Main to turntable", OutputCategory.MICROANIMATION, "book-turn", "day", None),
    DirectoryMapping("book_transitions_night/Main to shelves", OutputCategory.MICROANIMATION, "book-shelf", "night", None),
    DirectoryMapping("book_transitions_night/Main to turntable", OutputCategory.MICROANIMATION, "book-turn", "night", None),

    # MICROANIMATIONS - Turntable with nested structure
    DirectoryMapping("microanimations/turntable/turntable_close_red_record_spinning_day", OutputCategory.MICROANIMATION, "vinyl-spin", "day", None),
    DirectoryMapping("microanimations/turntable/turntable_close_red_record_spinning_night", OutputCategory.MICROANIMATION, "vinyl-spin", "night", None),
    DirectoryMapping("microanimations/turntable/turntable_close_tonearm_day", OutputCategory.MICROANIMATION, "tonearm", "day", None),
    DirectoryMapping("microanimations/turntable/turntable_close_tonearm_night", OutputCategory.MICROANIMATION, "tonearm", "night", None),
    DirectoryMapping("microanimations/turntable/turntable_far_black_record_spinning_day", OutputCategory.MICROANIMATION, "vinyl-spin-far", "day", None),
    DirectoryMapping("microanimations/turntable/turntable_far_black_record_spinning_night", OutputCategory.MICROANIMATION, "vinyl-spin-far", "night", None),
]


@dataclass
class OutputNaming:
    """Resolved output naming for a sequence."""

    category: OutputCategory
    prefix: str  # Full prefix like "transition_seat-coll_day"
    category_dir: str  # Output subdirectory like "transitions"
    frame_pattern: str  # Pattern for frame numbering

    def format_frame_name(self, frame_index: int, extension: str) -> str:
        """Format the output filename for a specific frame."""
        return f"{self.prefix}_{frame_index:03d}.{extension}"

    def format_animated_name(self, extension: str) -> str:
        """Format the output filename for an animated sequence.
        
        For animated sequences, we don't add '_animated' suffix,
        just use the prefix directly.
        """
        return f"{self.prefix}.{extension}"

    def format_scene_name(self, extension: str) -> str:
        """Format the output filename for a scene (first frame)."""
        # Extract base scene name from prefix
        parts = self.prefix.split('_')
        if len(parts) >= 3:
            # For transition_seat-coll_day -> scene_seat-coll_day
            # For micro_book-idle_day -> scene_book-idle_day
            scene_name = f"scene_{parts[1]}_{parts[2]}"
        else:
            scene_name = f"scene_{parts[1]}" if len(parts) > 1 else "scene_unknown"
        return f"{scene_name}.{extension}"
    
    def get_room_still_name(self, extension: str) -> str | None:
        """
        For place-to-place transitions, extract the room name (first location).
        Returns the room still filename or None if not a transition.
        """
        if self.category != OutputCategory.TRANSITION:
            return None
        
        # Parse the transition name to extract the first room
        parts = self.prefix.split('_')
        if len(parts) >= 3:
            # Format is transition_place1-place2_timeofday
            transition_name = parts[1]  # e.g., "seat-coll"
            time_of_day = parts[2]  # e.g., "day"
            
            # Extract first place from transition name
            if '-' in transition_name:
                first_place = transition_name.split('-')[0]  # e.g., "seat"
                return f"room_{first_place}_{time_of_day}.{extension}"
        
        return None


class DirectoryMapper:
    """Maps input directories to output naming conventions."""

    @staticmethod
    def get_output_naming(rel_dir: Path, prefix: str = "") -> OutputNaming | None:
        """
        Determine output naming based on relative directory path.
        
        Args:
            rel_dir: Relative directory path from base
            prefix: Optional prefix from the sequence
            
        Returns:
            OutputNaming object or None if no mapping found
        """
        # Convert Path to string for matching
        dir_str = str(rel_dir).replace('\\', '/')

        # Try to match against known patterns
        for mapping in DIRECTORY_MAPPINGS:
            # Check if the input pattern matches the directory
            if mapping.input_pattern in dir_str or dir_str.endswith(mapping.input_pattern):
                # Build the output prefix
                category_prefix = mapping.category.value

                # Construct full prefix based on category
                if mapping.variant:
                    output_prefix = f"{category_prefix}_{mapping.base_name}_{mapping.variant}"
                elif mapping.time_of_day:
                    output_prefix = f"{category_prefix}_{mapping.base_name}_{mapping.time_of_day}"
                else:
                    output_prefix = f"{category_prefix}_{mapping.base_name}"

                # Determine category directory
                if mapping.category == OutputCategory.TRANSITION:
                    category_dir = "transitions"
                elif mapping.category == OutputCategory.MICROANIMATION:
                    category_dir = "microanimations"
                else:
                    category_dir = "scenes"

                return OutputNaming(
                    category=mapping.category,
                    prefix=output_prefix,
                    category_dir=category_dir,
                    frame_pattern="{:03d}"
                )

        # No mapping found - return None to use default naming
        return None

    @staticmethod
    def extract_scene_info(output_naming: OutputNaming) -> tuple[str, str] | None:
        """
        Extract scene information from output naming for scene generation.
        
        Returns:
            Tuple of (scene_type, time_of_day) or None
        """
        if not output_naming:
            return None

        # Parse the prefix to extract scene components
        parts = output_naming.prefix.split('_')
        if len(parts) >= 3:
            # Format is category_name_timeofday
            scene_type = parts[1]  # e.g., "seat-coll"
            time_of_day = parts[2]  # e.g., "day"

            # Map to simplified scene names
            scene_map = {
                "seat-coll": "seat",
                "seat-turn": "seat",
                "coll-turn": "coll",
                "book-idle": "book",
                "book-shelf": "book",
                "book-turn": "book",
                "vinyl-spin": "turn",
                "tonearm": "turn",
            }

            simplified_scene = scene_map.get(scene_type, scene_type)
            return simplified_scene, time_of_day

        return None
