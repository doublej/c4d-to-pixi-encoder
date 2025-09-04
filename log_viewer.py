"""
Scrollable log viewer for Rich console with persistent history.

This module provides a scrollable log viewer that preserves all log entries
and allows scrolling through history even when new entries are added.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, List, Optional, Union

from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.text import Text


class ScrollableLogViewer:
    """A scrollable log viewer that maintains history and supports navigation.
    
    Args:
        max_visible_lines: Number of lines to show in the viewer at once
        max_history: Maximum number of log entries to keep in history
        log_file: Optional path to write logs to disk for persistence
    """
    
    def __init__(
        self,
        max_visible_lines: int = 20,
        max_history: int = 10000,
        log_file: Optional[Path] = None
    ) -> None:
        self.max_visible_lines = max_visible_lines
        self.max_history = max_history
        self.log_file = log_file
        self.logs: Deque[Text] = deque(maxlen=max_history)
        self.scroll_offset = 0
        self._log_file_handle = None
        
        if self.log_file:
            self._init_log_file()
    
    def _init_log_file(self) -> None:
        """Initialize the log file with timestamp header."""
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"Session started: {datetime.now().isoformat()}\n")
                f.write(f"{'='*60}\n")
    
    def add_log(self, message: Union[str, Text], style: Optional[str] = None) -> None:
        """Add a log entry to the viewer.
        
        Args:
            message: The log message (string or Rich Text object)
            style: Optional Rich style to apply to the message
        """
        if isinstance(message, str):
            log_entry = Text(message, style=style)
        else:
            log_entry = message
        
        self.logs.append(log_entry)
        
        # Write to file if configured
        if self.log_file:
            timestamp = datetime.now().strftime("%H:%M:%S")
            plain_text = log_entry.plain if hasattr(log_entry, 'plain') else str(log_entry)
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] {plain_text}\n")
        
        # Auto-scroll to bottom when new log is added
        self.scroll_to_bottom()
    
    def scroll_up(self, lines: int = 1) -> None:
        """Scroll up by specified number of lines."""
        self.scroll_offset = max(0, self.scroll_offset - lines)
    
    def scroll_down(self, lines: int = 1) -> None:
        """Scroll down by specified number of lines."""
        max_offset = max(0, len(self.logs) - self.max_visible_lines)
        self.scroll_offset = min(max_offset, self.scroll_offset + lines)
    
    def scroll_to_top(self) -> None:
        """Scroll to the top of the log history."""
        self.scroll_offset = 0
    
    def scroll_to_bottom(self) -> None:
        """Scroll to the bottom of the log history."""
        self.scroll_offset = max(0, len(self.logs) - self.max_visible_lines)
    
    def get_visible_logs(self) -> List[Text]:
        """Get the currently visible log entries based on scroll position.
        
        Returns:
            List of Text objects that should be displayed
        """
        if not self.logs:
            # Return empty lines to maintain stable height
            return [Text("") for _ in range(self.max_visible_lines)]
        
        start_idx = self.scroll_offset
        end_idx = min(start_idx + self.max_visible_lines, len(self.logs))
        
        visible = list(self.logs)[start_idx:end_idx]
        
        # Pad with empty lines to maintain consistent height
        while len(visible) < self.max_visible_lines:
            visible.append(Text(""))
        
        return visible
    
    def get_panel(self, title: str = "Log History") -> Panel:
        """Get a Rich Panel containing the visible logs.
        
        Args:
            title: Title for the panel
            
        Returns:
            Panel object ready for rendering
        """
        visible_logs = self.get_visible_logs()
        
        # Combine logs into a single Text object with newlines
        combined = Text()
        for i, log in enumerate(visible_logs):
            if i > 0:
                combined.append("\n")
            combined.append(log)
        
        # Add scroll indicators
        scroll_info = self._get_scroll_info()
        if scroll_info:
            title = f"{title} {scroll_info}"
        
        return Panel(
            combined,
            title=f"[cyan]{title}[/]",
            border_style="cyan",
            title_align="left"
        )
    
    def _get_scroll_info(self) -> str:
        """Get scroll position indicator string."""
        if not self.logs:
            return ""
        
        total = len(self.logs)
        visible_start = self.scroll_offset + 1
        visible_end = min(self.scroll_offset + self.max_visible_lines, total)
        
        if total <= self.max_visible_lines:
            return f"[showing all {total}]"
        
        position = ""
        if self.scroll_offset == 0:
            position = "TOP"
        elif self.scroll_offset >= total - self.max_visible_lines:
            position = "BOTTOM"
        else:
            position = f"{visible_start}-{visible_end}"
        
        return f"[{position} of {total}]"
    
    def clear(self) -> None:
        """Clear all log entries."""
        self.logs.clear()
        self.scroll_offset = 0