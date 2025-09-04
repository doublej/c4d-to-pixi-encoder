"""
Simple logging system without Rich dependencies.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, TextIO


class SimpleLogger:
    """Simple logger that writes to console and file."""
    
    def __init__(self, log_file: Optional[Path] = None):
        self.log_file = log_file
        self.start_time = time.time()
        
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"Session started: {datetime.now().isoformat()}\n")
                f.write(f"{'='*60}\n")
    
    def log(self, message: str, prefix: str = "", error: bool = False) -> None:
        """Log a message to console and file.
        
        Args:
            message: The message to log
            prefix: Optional prefix like [INFO], [ERROR], etc.
            error: Whether to write to stderr instead of stdout
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Format the message
        if prefix:
            formatted = f"[{timestamp}] {prefix} {message}"
        else:
            formatted = f"[{timestamp}] {message}"
        
        # Write to console
        output = sys.stderr if error else sys.stdout
        print(formatted, file=output, flush=True)
        
        # Write to file
        if self.log_file:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(formatted + '\n')
            except Exception:
                pass  # Don't fail on logging errors
    
    def progress(self, current: int, total: int, description: str = "") -> None:
        """Show simple progress indicator.
        
        Args:
            current: Current item number
            total: Total items
            description: Optional description
        """
        percent = (current / total * 100) if total > 0 else 0
        elapsed = time.time() - self.start_time
        
        if description:
            self.log(f"[{current}/{total}] ({percent:.1f}%) {description} - {elapsed:.1f}s elapsed")
        else:
            self.log(f"[{current}/{total}] ({percent:.1f}%) - {elapsed:.1f}s elapsed")
    
    def table(self, headers: List[str], rows: List[List[str]]) -> None:
        """Print a simple table.
        
        Args:
            headers: Column headers
            rows: Table rows
        """
        if not headers or not rows:
            return
        
        # Calculate column widths
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(widths):
                    widths[i] = max(widths[i], len(str(cell)))
        
        # Print separator
        separator = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
        self.log(separator)
        
        # Print headers
        header_row = "|" + "|".join(f" {h:<{w}} " for h, w in zip(headers, widths)) + "|"
        self.log(header_row)
        self.log(separator)
        
        # Print rows
        for row in rows:
            row_str = "|" + "|".join(f" {str(cell):<{w}} " for cell, w in zip(row, widths)) + "|"
            self.log(row_str)
        
        self.log(separator)
    
    def section(self, title: str) -> None:
        """Print a section header.
        
        Args:
            title: Section title
        """
        self.log("")
        self.log("=" * 60)
        self.log(title.center(60))
        self.log("=" * 60)
    
    def success(self, message: str) -> None:
        """Log a success message."""
        self.log(message, prefix="[SUCCESS]")
    
    def error(self, message: str) -> None:
        """Log an error message."""
        self.log(message, prefix="[ERROR]", error=True)
    
    def warning(self, message: str) -> None:
        """Log a warning message."""
        self.log(message, prefix="[WARNING]")
    
    def info(self, message: str) -> None:
        """Log an info message."""
        self.log(message, prefix="[INFO]")