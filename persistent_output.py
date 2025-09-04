"""
Persistent console stream wrappers to ensure all output is preserved on screen.

This module converts in-place updates that rely on carriage returns ("\r")
into newline-terminated lines so that progress-style output does not overwrite
previous content in the terminal.

Usage:
    from persistent_output import install_persistent_console
    install_persistent_console()  # Wraps sys.stdout and sys.stderr
"""

from __future__ import annotations

import io
import sys
from typing import Optional, TextIO


class PersistentStream(io.TextIOBase):
    """A text stream wrapper that turns carriage returns into newlines.

    The wrapper ensures that any write containing "\r" (used by progress bars
    to redraw a single terminal line) is emitted as a completed line, preventing
    content from being overwritten.

    Args:
        stream (TextIO): The underlying text stream to write to (e.g., sys.stdout).

    Notes:
        - "\r\n" is normalized to "\n" to avoid double spacing.
        - Buffers partial lines until a newline or carriage return completes them.
        - Flush does not force an extra newline; partial lines are written on close.
    """

    def __init__(self, stream: TextIO) -> None:
        self._stream: TextIO = stream
        self._buffer: str = ""

    # ---------------
    # TextIOBase API
    # ---------------
    def writable(self) -> bool:  # type: ignore[override]
        return True

    @property
    def encoding(self) -> Optional[str]:  # type: ignore[override]
        return getattr(self._stream, "encoding", None)

    @property
    def errors(self) -> Optional[str]:  # type: ignore[override]
        return getattr(self._stream, "errors", None)

    def isatty(self) -> bool:  # type: ignore[override]
        try:
            return bool(self._stream.isatty())
        except Exception:
            return False

    def fileno(self) -> int:  # type: ignore[override]
        try:
            return int(self._stream.fileno())
        except Exception as ex:  # pragma: no cover - mirrors TextIOBase behavior
            raise io.UnsupportedOperation("fileno") from ex

    def write(self, s: str) -> int:  # type: ignore[override]
        """Write text, converting CR updates into newline-terminated lines.

        Args:
            s (str): Text to write.

        Returns:
            int: Number of characters accepted from input.
        """
        if not isinstance(s, str):
            s = str(s)
        if not s:
            return 0

        # Normalize CRLF to LF first to avoid duplicate newlines.
        s_norm = s.replace("\r\n", "\n")

        # Split on carriage returns; each CR completes the current line.
        parts = s_norm.split("\r")
        for i, part in enumerate(parts):
            self._buffer += part
            if i < len(parts) - 1:
                self._emit_line(self._buffer)
                self._buffer = ""

        # Now handle any full lines present in the buffer via LF.
        lines = self._buffer.split("\n")
        for line in lines[:-1]:
            self._emit_line(line)
        self._buffer = lines[-1]

        return len(s)

    def flush(self) -> None:  # type: ignore[override]
        try:
            self._stream.flush()
        except Exception:
            pass

    def close(self) -> None:  # type: ignore[override]
        try:
            # On close, emit any dangling partial line as a full line.
            if self._buffer:
                self._emit_line(self._buffer)
                self._buffer = ""
        finally:
            try:
                self._stream.flush()
            except Exception:
                pass

    # ---------------
    # Helpers
    # ---------------
    def _emit_line(self, line: str) -> None:
        # Avoid trailing CR artifacts and ensure a single newline terminator.
        try:
            self._stream.write(line + "\n")
            self._stream.flush()
        except Exception:
            # Never raise from logging writes; best-effort only.
            pass


def install_persistent_console(*, wrap_stdout: bool = True, wrap_stderr: bool = True) -> None:
    """Install persistent wrappers on sys.stdout and/or sys.stderr.

    Args:
        wrap_stdout (bool): If True, wrap sys.stdout.
        wrap_stderr (bool): If True, wrap sys.stderr.

    Returns:
        None

    Errors:
        Never raises; if wrapping fails, leaves streams as-is.
    """
    try:
        if wrap_stdout and not isinstance(sys.stdout, PersistentStream):
            sys.stdout = PersistentStream(sys.stdout)  # type: ignore[assignment]
        if wrap_stderr and not isinstance(sys.stderr, PersistentStream):
            sys.stderr = PersistentStream(sys.stderr)  # type: ignore[assignment]
    except Exception:
        # Best-effort: if we can't wrap, don't interrupt program execution.
        pass

